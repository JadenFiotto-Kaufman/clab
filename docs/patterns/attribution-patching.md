---
title: Attribution Patching
one_liner: "How do I rank every component with one backward? grad times the counterfactual's difference, the gradient check, and where it misleads."
tags: [patterns, attribution, gradient, grad, patching, heads]
related: [docs/patterns/activation-patching.md, docs/patterns/head-and-neuron-patching.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/sweeps.md]
sources: [clab/ops/transforms.py, clab/steps/batches.py, clab/compile/schedule.py, clab/engines/nnterp.py, clab/engines/hooks.py, documents/attribution_patching.json, documents/real/attribution_patching.json]
---

# Attribution Patching

Attribution patching estimates what [activation patching](activation-patching.md) would measure, to first order: the metric's gradient at a site on the base run, times the counterfactual's value there less the base's, summed over the site's features. One backward gives the gradient at every site of a forward, so a whole layer-by-head map costs two or three forwards where patching costs a forward per component. In clab the gradient is a read: an entry whose first op is `grad` ([interventions.md](../usage/interventions.md#a-gradient-at-a-site); the op is in [FORMAT.md#transforms](../../FORMAT.md#transforms)).

## The document

`documents/attribution_patching.json` reads layer 0's residual stream at the last token on both prompts, and in a third forward over the base prompt takes the gradient there:

```json
"difference": {"kind": "op", "op": "sub", "of": "counterfactual.v_cf", "with": "base.v"},
"patched": {
  "kind": "forward",
  "input": "pairs.input",
  "interventions": {
    "h":      {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"},
                                                   {"op": "scale", "factor": "difference"},
                                                   {"op": "sum", "over": ["hidden_size"]}]},
    "metric": {"site": {"component": "lm_head_output"}, "pos": -1,
               "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}
  }
},
"estimate": {"kind": "op", "op": "mean", "of": "patched.h"}
```

`patched.h` is one number per row: that row's estimate of what swapping `counterfactual.v_cf` in would do to its logit difference. `estimate` is the mean over rows. The entry writes nothing, so `patched.metric` equals the base run's logit difference. How `grad` reads, per row and at one site, is [interventions](../usage/interventions.md#a-gradient-at-a-site).

To keep to two forwards, read the base values in the forward that takes the gradient, and combine them in op steps after it. `documents/real/attribution_patching.json` does this for Llama-3.2-1B's 512 heads:

- one base forward reads every head's result and its gradient;
- steps take `sub`, `scale`, `sum` over `pos` and `head_dim`, and `mean` over rows;
- the result is a `[layers, num_heads]` map from two forwards and one backward.

Its layer 12 head 28 comes first, at about +2.20, where [head patching](head-and-neuron-patching.md) measures +2.46 for the same head (`tests/test_real.py` `test_attribution_patching_ranks_layer_12_head_28_first`).

## Check the gradient

A gradient check tests the quantity the estimate uses: the gradient's dot with the counterfactual difference, the counterfactual's value at the read less the base's. It compares that with a central difference of the metric along the same difference, `(metric(+e) - metric(-e)) / 2e`, from two forwards that add `e` times the difference at the site and `-e` times it. `central` scales their difference by `factor` 1/(2e), 2.5 at the `e` of 0.2 used here. This complete document runs both on GPT-2 at layer 6's last token over `weekdays/train` and puts the gap in `gap`:

```json
{
  "description": "The gradient check along the counterfactual difference: the gradient at layer 6's last token dotted with the counterfactual's value there less the base's, against a central difference of two forwards that add 0.2 and -0.2 times that difference there.",
  "model": {
    "key": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {
    "target": {"component": "layer_output", "layers": 6},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "scored": {"metric": {"site": "head", "pos": -1,
                          "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}
  },
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                       "interventions": {"v_cf": {"site": "target", "pos": -1}}},
    "taken": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"v": {"site": "target", "pos": -1},
       "g": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}]}},
      "scored"]},
    "difference": {"kind": "op", "op": "sub", "of": "counterfactual.v_cf", "with": "taken.v"},
    "term": {"kind": "op", "op": "scale", "of": "difference", "factor": "taken.g"},
    "along": {"kind": "op", "op": "sum", "of": "term", "over": ["hidden_size"]},
    "up": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "difference", "scale": 0.2}]}}, "scored"]},
    "down": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "difference", "scale": -0.2}]}}, "scored"]},
    "rise": {"kind": "op", "op": "sub", "of": "up.metric", "with": "down.metric"},
    "central": {"kind": "op", "op": "scale", "of": "rise", "factor": 2.5},
    "gap": {"kind": "op", "op": "sub", "of": "along", "with": "central"}
  },
  "saves": ["along", "central", "gap"]
}
```

On GPT-2 (`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU, float32, the default `--batch-size` 8), row 0's `along` is 0.31854 and its `central` 0.31860. Over the four rows `gap` is at most 1.6e-4, where the largest `along` is 0.41. The table below reruns the document with each `e`, `factor` 1/(2e).

| `e` | 1.6 | 0.8 | 0.4 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| largest `gap` | 1.44e-2 | 3.61e-3 | 9.31e-4 | 1.58e-4 | 2.41e-4 | 6.62e-4 |

- **A gap that falls about 4 times each time the step halves says the gradient is right.** A central difference's own error falls as the step squared: from 1.6 to 0.8 to 0.4 the gap fell 4.0 and 3.9 times.
- **Below some step, rounding takes over.** The two forwards' rounding, divided by twice the step, grows as the step falls; here it took over below 0.2, where GPT-2's large logits round.
- **Pass the check at the step whose gap is least**, 0.2 here, where that gap is a small fraction of the largest `along`; set the fraction before the run.
- **Adding one number to every element checks nothing on a model with LayerNorm.** LayerNorm removes each vector's mean, so the derivative along that direction is about zero however the gradient is wired. On GPT-2 the gradient summed over `hidden_size` at layer 0's last token was below 3e-7 on every row. The counterfactual difference is the direction the estimate uses, so check along it.

A `grad` of the wrong entry runs without a refusal and maps that entry, so run this check once on each new document. A gap far above the tolerance at every step is that case: check that `of` names the entry you meant. On the corpus model the corpus document's `patched.h` is the same central difference along `difference`, to 2e-5 (`tests/test_grad.py` `test_the_gradient_is_the_central_difference_of_the_metric`).

Three nulls, each exact on both engines:

- **A constant's gradient is zero.** A `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended to its ops must be exactly zero. In the gradient's forward, beside `metric`:

  ```json
  "zero": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
                                              {"op": "mul", "with": 0}]},
  "g0": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "zero"}]}
  ```

  Use `mul`, not `scale`. A `scale` 0 entry alone at the head, listed before the metric, is a write: it zeroes the logits the metric reads, so the real `grad` is zero too ([interventions](../usage/interventions.md#a-gradient-at-a-site)).

  It tests the wiring: that the `grad` differentiates the entry its `of` names, through that entry's ops, and that nothing else reaches the backward. It says nothing about the gradient's size, which the check above tests.
- **A counterfactual that is the base prompt gives zero.** Read `v_cf` in a forward over `pairs.input` instead: `difference` is exactly zero, and so is every row's attribution. It checks that the two reads are one selection, which a constant's null does not.
- **A `grad` sees the order at its site.** A `grad` listed before a write of `{"op": "scale", "factor": 2}` at its site is exactly twice the `grad` listed after it.

## Variations

- **Every layer and position.** A site at `"layers": ":"` read at a window of static width (`{"index": "-3:"}`) gives a `[layers, rows, pos, hidden_size]` gradient from the same one backward. Each layer's slice equals a forward that takes it at that layer alone. The `difference` it is scaled by is read at the same site and positions on both prompts. With `resid` that site, and `counterfactual` reading `z` as `base` does:

  ```json
  "base": {"kind": "forward", "input": "pairs.input", "interventions": {
    "z": {"site": "resid", "pos": {"index": "-3:"}},
    "dz": {"site": "resid", "pos": {"index": "-3:"}, "ops": [{"op": "grad", "of": "metric"}]},
    "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}},
  "difference": {"kind": "op", "op": "sub", "of": "counterfactual.z", "with": "base.z"},
  "term": {"kind": "op", "op": "scale", "of": "difference", "factor": "base.dz"},
  "per_row": {"kind": "op", "op": "sum", "of": "term", "over": ["hidden_size"]},
  "map": {"kind": "op", "op": "mean", "of": "per_row", "over": ["rows"]}
  ```

  `map` is `[layers, pos]`. At each layer and position, `per_row` is the check's `along` there, from the map's one backward. So the check's two nudged forwards, at one layer and the last position, test the map's own gradient. The constant's null (`zero` and `g0`) may sit in this forward too.
- **Heads.** On `attention_premix` the gradient keeps the `num_heads` and `head_dim` axes; sum over `head_dim` (and `pos`) for one number per head. A `select` on `num_heads` keeps the gradient of the heads it picks.
- **Several sites, one backward.** Every `grad` entry naming the same `of` is served by one backward, so the residual stream and the heads of a model come from one forward together.

## Pitfalls

- **The estimate is linear, and patching is not.** At the early layers of the token the counterfactual changes, one patch restores the whole answer, and the linear estimate falls far short of it. Rank with attribution, then confirm the components you report with real patching ([activation patching](activation-patching.md)).
- **`of` is summed over every axis it has.** A metric read at several positions, or kept per layer, is differentiated as one sum; reduce or select it to the one number per row you mean before `grad` takes it.
- **A bfloat16 model's gradient is bfloat16.** Recompute the components you report in float32 ([ops.md](../usage/ops.md#the-float32-rule)).
