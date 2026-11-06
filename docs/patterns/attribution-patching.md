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

A gradient check compares `grad` with a central difference of the metric: two forwards that move the site by plus and minus a small step. Adding a number moves every element alike, so its derivative is the gradient summed over `hidden_size`. This complete document runs both and puts the gap in `gap`:

```json
{
  "description": "The gradient check: layer 0's gradient at the last token, summed over hidden_size, against a central difference of two forwards that add 0.001 and -0.001 to every element there.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {
    "target": {"component": "layer_output", "layers": 0},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "scored": {"metric": {"site": "head", "pos": -1,
                          "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}
  },
  "steps": {
    "taken": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"g": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "sum", "over": ["hidden_size"]}]}},
      "scored"]},
    "up": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": 0.001}]}}, "scored"]},
    "down": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": -0.001}]}}, "scored"]},
    "rise": {"kind": "op", "op": "sub", "of": "up.metric", "with": "down.metric"},
    "central": {"kind": "op", "op": "scale", "of": "rise", "factor": 500},
    "gap": {"kind": "op", "op": "sub", "of": "taken.g", "with": "central"}
  },
  "saves": ["taken.g", "central", "gap"]
}
```

A central difference's own error falls as the step squared. So above float32's rounding, a gap that falls about 100 times when the step falls 10 times says the gradient is right. At a small step the two forwards' rounding, divided by twice the step, takes over, and the ratio degrades. So set a tolerance before the run, and pass the check where the gap is below a small fraction of the largest gradient.

A `grad` of the wrong entry runs without a refusal and maps that entry, so run this check once on each new document. A gap far above the tolerance that does not shrink with the step is that case: check that `of` names the entry you meant. On the corpus model in float32 (`tests/test_grad.py` `test_the_gradient_is_the_central_difference_of_the_metric`):

- the gradients are about ±6.8, and `gap` is at most 0.0068, a thousandth of the largest;
- at a step of 0.01 the gap is 95 times larger;
- the corpus document's `patched.h` is the same central difference taken along `difference`, to 2e-5.

Three nulls, each exact on both engines:

- **A constant's gradient is zero.** A `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended to its ops must be exactly zero. In the gradient's forward, beside `metric`:

  ```json
  "zero": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
                                              {"op": "mul", "with": 0}]},
  "g0": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "zero"}]}
  ```

  Use `mul`, not `scale`. A `scale` 0 entry alone at the head, listed before the metric, is a write: it zeroes the logits the metric reads, so the real `grad` is zero too ([interventions](../usage/interventions.md#a-gradient-at-a-site)).
- **A counterfactual that is the base prompt gives zero.** Read `v_cf` in a forward over `pairs.input` instead: `difference` is exactly zero, and so is every row's attribution. It checks that the two reads are one selection, which a constant's null does not.
- **A `grad` sees the order at its site.** A `grad` listed before a write of `{"op": "scale", "factor": 2}` at its site is exactly twice the `grad` listed after it.

## Variations

- **Every layer and position.** A site at `"layers": ":"` read at a window of static width (`{"index": "-3:"}`) gives a `[layers, rows, pos, hidden_size]` gradient from the same one backward. Each layer's slice equals a forward that takes it at that layer alone. The `difference` it is scaled by is read at the same site and positions on both prompts. With `resid` that site, `target` layer 0, and `counterfactual` reading `z` as `base` does:

  ```json
  "base": {"kind": "forward", "input": "pairs.input", "interventions": {
    "z": {"site": "resid", "pos": {"index": "-3:"}},
    "dz": {"site": "resid", "pos": {"index": "-3:"}, "ops": [{"op": "grad", "of": "metric"}]},
    "g": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "sum", "over": ["hidden_size"]}]},
    "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}},
  "difference": {"kind": "op", "op": "sub", "of": "counterfactual.z", "with": "base.z"},
  "term": {"kind": "op", "op": "scale", "of": "difference", "factor": "base.dz"},
  "per_row": {"kind": "op", "op": "sum", "of": "term", "over": ["hidden_size"]},
  "map": {"kind": "op", "op": "mean", "of": "per_row", "over": ["rows"]}
  ```

  `map` is `[layers, pos]`. `g` is the gradient check's point entry, in the map's own forward: it shares the one backward and equals the map's slice at layer 0, the last position, summed over `hidden_size`. So the check above tests the map's own gradient. The constant's null (`zero` and `g0`) may sit in this forward too.
- **Heads.** On `attention_premix` the gradient keeps the `num_heads` and `head_dim` axes; sum over `head_dim` (and `pos`) for one number per head. A `select` on `num_heads` keeps the gradient of the heads it picks.
- **Several sites, one backward.** Every `grad` entry naming the same `of` is served by one backward, so the residual stream and the heads of a model come from one forward together.

## Pitfalls

- **The estimate is linear, and patching is not.** At the early layers of the token the counterfactual changes, one patch restores the whole answer, and the linear estimate falls far short of it. Rank with attribution, then confirm the components you report with real patching ([activation patching](activation-patching.md)).
- **`of` is summed over every axis it has.** A metric read at several positions, or kept per layer, is differentiated as one sum; reduce or select it to the one number per row you mean before `grad` takes it.
- **A bfloat16 model's gradient is bfloat16.** Recompute the components you report in float32 ([ops.md](../usage/ops.md#the-float32-rule)).
