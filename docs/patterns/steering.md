---
title: Steering
one_liner: "Steering with a direction: harvest a mean difference, add it at a dose, sweep the dose, renormalize, clamp, and keep the write on through every generated token."
tags: [patterns, steering]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/generate.md, docs/patterns/generation-experiments.md, docs/patterns/ablation.md]
sources: [clab/ops/edits.py, clab/ops/reductions.py, clab/steps/batches.py]
---

# Steering

Steering adds a direction to the residual stream to push the model toward a behavior, and measures what moves. In clab the direction is a value earlier steps compute (a mean, a difference of means), the write is `add` with a `scale` (the dose), and the evidence is a sweep over that dose with the unsteered run beside it.

## The canonical document

`documents/steer_renormalize.json` takes its direction as the mean residual stream over the counterfactual prompts, then adds four times it at the last position, alone and followed by a renormalize:

```json
"interventions": {
  "steer": {"steer": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "direction", "scale": 4.0}]}},
  "restore": {"restore": {"site": "target", "pos": -1, "ops": [{"op": "renormalize"}]}}
},
"steps": {
  "harvest": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]", "interventions": {"resid": {"site": "target", "pos": -1}}},
  "direction": {"kind": "op", "op": "mean", "of": "harvest.resid"},
  "steered": {"kind": "forward", "input": "pairs.input", "interventions": ["steer", {"logits": {"site": "head", "pos": -1}}]},
  "steered_renormed": {"kind": "forward", "input": "pairs.input",
                       "interventions": ["steer", "restore", {"logits": {"site": "head", "pos": -1}}]}
}
```

Adding a vector also lengthens the activation. Here the base residual stream at the last token has norm about 0.075 and `steered.steer` about 0.375; `steered_renormed.restore` is back to the base norm to float rounding and still points where the steered vector does (cosine 1). Comparing `steered_ld` with `renormed_ld` separates the effect of the new direction from that of the longer vector. `renormalize` rescales to the value the site had before the step's writes, so `restore` must come after `steer` in the step's list; listed before it, it changes nothing ([interventions](../usage/interventions.md#edit-recipes)).

## A mean difference at a dose

A direction for a behavior is usually the difference between mean activations on prompts that show it and prompts that do not. The difference is an edit run as a step, `sub` over two means, and `normalize` makes it unit length so that the dose is a length:

```json
{
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"days": {"path": "weekdays/train"}, "counts": {"path": "counting/train"}},
  "sites": {
    "target": {"component": "layer_output", "layers": 0},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "base": {"kind": "forward", "input": "days.input", "interventions": {
      "resid": {"site": "target", "pos": -1}, "logits": {"site": "head", "pos": -1}}},
    "other": {"kind": "forward", "input": "counts.input", "interventions": {"resid": {"site": "target", "pos": -1}}},
    "mean_days": {"kind": "op", "op": "mean", "of": "base.resid"},
    "mean_counts": {"kind": "op", "op": "mean", "of": "other.resid"},
    "difference": {"kind": "op", "op": "sub", "of": "mean_counts", "with": "mean_days"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "steered": {"kind": "forward", "input": "days.input", "interventions": {
      "steer": {"site": "target", "pos": -1, "ops": [
        {"op": "add", "with": "direction", "scale": {"sweep": [0.0, 0.02, 0.08, 0.32], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "kl": {"kind": "op", "op": "kl", "of": "steered.logits", "against": "base.logits"},
    "drift": {"kind": "op", "op": "mean", "of": "kl"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "steered.logits", "a": "days.cf_answer", "b": "days.base_answer"},
    "effect": {"kind": "op", "op": "mean", "of": "ld"}
  },
  "saves": ["difference", "base.resid", "drift", "effect"]
}
```

The means come from two datasets, which is allowed because a mean has no rows. `base` runs once; `steered` and the steps after it run once per dose. `clab table <out> drift effect`:

```text
at             drift                   effect
{"dose":0.0}   0.0                     0.0009160377085208893
{"dose":0.02}  0.00020074972417205572  0.0005409605801105499
{"dose":0.08}  0.001653727376833558    -0.00017407722771167755
{"dose":0.32}  0.004028182942420244    -0.00015350989997386932
```

How to read and choose the doses:

- **Dose 0 is the null** ([interventions](../usage/interventions.md#the-null-control)). `add` at `"scale": 0.0` leaves every logit bit-identical to the unsteered run, so its `drift` is exactly 0 and its `effect` is the baseline.
- **Doses are lengths against the activation's norm.** `base.resid` is saved so the norm can be read back: about 0.075 here, so the doses run from about a quarter of the activation's length to four times it. A dose far below the norm does nothing; one far above it replaces the activation.
- **`effect` is the target, `drift` the cost.** `kl` against the unsteered logits measures how far the whole distribution moved. The useful band is where the target metric moves while the drift stays small; past it the model is being broken, not steered. On a real model, generate at each dose and read the text as well.
- **Check the difference before using it.** `difference` is saved for its norm. Two prompt sets that are the same prompts in another order (the base and counterfactual columns of `data/weekdays/train.json` are) give a difference of rounding noise, about 4e-9, which `normalize` turns into a unit vector pointing nowhere in particular; a difference of exactly zero makes `normalize` NaN, and every dose of it, 0 included, NaN.

## Clamping

`documents/clamp_write.json` bounds every element of layer 1's residual stream at every token, BOS included, and reads the result back after the write:

```json
"clamp": {"site": "target", "pos": {"index": ":"}, "ops": [{"op": "clamp", "min": -0.01, "max": 0.01}]},
"resid": {"site": "target", "pos": -1}
```

`clamped.resid` lies in [-0.01, 0.01] where `clean.resid` reaches -0.04. The bounds are set against this checkpoint's residual stream, whose entries there are of order 0.01; read the site first and set bounds from what it holds. Clamping a featurizer's coordinates instead (one feature held at a value) goes between an `encode` and its `decode` ([featurizers](../usage/featurizers.md)).

## Through every generated token

In a `generate` step a write's `pos` names forward calls by its `frame` ([generate](../usage/generate.md)). `pos: -1` is the last prompt token in the prefill, and the generated tokens see that write only through the cache. To steer every token, write at every call:

```json
"steered": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "min_new_tokens": 6, "do_sample": false,
  "interventions": {"steer": {"site": "target", "pos": {"frame": ":"},
                              "ops": [{"op": "add", "with": "direction", "scale": 8.0}]}}}
```

`{"frame": ":"}` is every prompt token in the prefill and each generated token at its decode call; `{"frame": "1:"}` is the generated tokens alone. To steer the prompt's last token and every generated token, and no other prompt token, write two entries, `{"frame": 0, "index": -1}` and `{"frame": "1:"}` ([generate](../usage/generate.md#writes-during-generation)). On the tiny model, the same dose at `pos: -1` leaves all six generated tokens as they were, while at `{"frame": ":"}` it changes five of them; at `"scale": 0.0` the ids equal the unsteered generation's. Score the text with `detokenize` and `contains` ([generation experiments](generation-experiments.md)), and keep `do_sample: false` so the steered and unsteered runs differ by the write alone.

## Pitfalls

- A steering write at `pos: -1` in a `generate` step is in the prefill only.
- `renormalize` listed before the `add` it should correct does nothing.
- A difference of means over the same prompts is noise; check its norm before `normalize`.
