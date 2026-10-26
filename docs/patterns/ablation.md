---
title: Ablation
one_liner: "Zero, mean, resample, noise and direction ablation: a mean as an operand across datasets, a ragged mean over an entity's tokens, a direction ablated at every layer, and the hydra effect."
tags: [patterns, ablation]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/patterns/activation-patching.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/attention.md, docs/usage/positions.md]
sources: [clab/ops/edits.py, clab/ops/reductions.py, clab/compile/arguments.py]
---

# Ablation

Ablation removes what a component contributes and measures what the model loses: a test of necessity, with no counterfactual prompt. In clab it is a write in one forward, scored against a clean forward over the same rows. The kinds differ only in what the write puts in the component's place, and that choice decides what the result means.

## The shape of an ablation document

`documents/zero_ablation.json`:

```json
"clean": {"kind": "forward", "input": "prompts.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
"clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "prompts.cf_answer", "b": "prompts.base_answer"},
"zeroed": {"kind": "forward", "input": "prompts.input", "interventions": {
  "zero": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]},
  "logits": {"site": "head", "pos": -1}}},
"zeroed_ld": {"kind": "op", "op": "logit_diff", "of": "zeroed.logits", "a": "prompts.cf_answer", "b": "prompts.base_answer"}
```

The clean forward is the baseline and runs on the same rows at the same batch size, so `clean_ld` and `zeroed_ld` differ by the ablation alone; `clab table <out> clean_ld zeroed_ld` puts them side by side. `swap` with a number writes it into every element of the selection, here the whole residual stream at the last token.

## What to put in its place

| kind | the write | corpus | what it removes |
|---|---|---|---|
| zero | `{"op": "swap", "with": 0.0}` | `documents/zero_ablation.json` | everything, and adds a value the model never produces |
| mean | `{"op": "swap", "with": "mean"}`, `mean` a step over a harvest | `documents/mean_ablation.json` | what varies across inputs; keeps the typical value |
| resample | `{"op": "swap", "with": "noise.v_noise"}`, a read on another prompt per row | `documents/hydra_effect.json` | this input's value, replaced by another real input's |
| noise | `{"op": "noise", "scale": 0.5, "seed": 7}` | `documents/noise_write.json` | some of the signal, at a dose |
| direction | `{"op": "project_out", "direction": "direction"}` | `documents/direction_ablation.json` | one direction, the rest kept |

A zero vector is a point no forward reaches: the downstream norms and attention see an input off their distribution, so a zero ablation measures that shock as well as the loss and overstates importance. Mean and resample ablation keep the value on the distribution and are the usual choice. The noise op draws from its `seed`, the row and the token position alone, so every forward and every batch size adds the same noise; sweep `seed` (`{"sweep": [1, 2, 3]}`) to see how much one draw's result depends on the draw.

The harvest-then-ablate chain of `documents/mean_ablation.json`:

```json
"harvest": {"kind": "forward", "input": "prompts.input", "interventions": {"acts": {"site": "target", "pos": -1}}},
"mean": {"kind": "op", "op": "mean", "of": "harvest.acts"},
"ablated": {"kind": "forward", "input": "prompts.input", "interventions": {
  "ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]},
  "logits": {"site": "head", "pos": -1}}}
```

`mean` removes the rows and the position, leaving one `hidden_size` vector, which broadcasts over every row the write selects.

## A mean from another dataset

A value with a rows axis belongs to its dataset's rows, but a mean has none, so it can come from a reference set and be written into another. Declare both datasets, harvest on one, ablate on the other:

```json
"data": {"pairs": {"path": "weekdays/train"}, "other": {"path": "weekdays_aligned/data"}},
"steps": {
  "harvest": {"kind": "forward", "input": "pairs.input", "interventions": {"acts": {"site": "target", "pos": -1}}},
  "mean": {"kind": "op", "op": "mean", "of": "harvest.acts"},
  "ablated": {"kind": "forward", "input": "other.input", "interventions": {
    "ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]},
    "logits": {"site": "head", "pos": -1}}}
}
```

Swapping `harvest.acts` itself into `other`'s rows is refused: `'other.input' has a row per example of 'other' and 'harvest.acts' of 'pairs'; an op takes one dataset's rows`.

## A mean over an entity's tokens

`documents/entity_mean_ablation.json` harvests every token of each row's entity, `{"scope": {"column": "prompts.entity"}, "index": ":"}`: three positions for " Thursday", one for " Friday". The read is as wide as its widest row, NaN past a shorter row's own, and `mean` averages the positions present, so the result is one vector that lands at any position. That mean weighs every token equally, so a row whose entity has three tokens counts three times. To weigh rows equally, average over the positions first:

```json
"per_row": {"kind": "op", "op": "mean", "of": "harvest.acts", "over": ["pos"]},
"mean": {"kind": "op", "op": "mean", "of": "per_row"}
```

On this corpus the two means differ by up to 0.007 in a coordinate, against a residual stream whose entries are around 0.01. A row whose entity is not in its prompt is refused by name before any forward.

## A direction at every layer

`documents/direction_ablation.json` (tiny GPT-2, five layers) takes each layer's mean residual stream at the last token as that layer's direction and removes it from every token at every layer, in one entry:

```json
"direction": {"kind": "op", "op": "mean", "of": "harvest.acts"},
"ablate": {"site": "every", "pos": {"index": ":"}, "ops": [
  {"op": "project_out", "direction": "direction"},
  {"op": "cosine", "with": "direction"}
]}
```

`harvest.acts` is read at `layers: ":"`, so the default `mean` keeps its `layers` axis: one direction per layer, labelled 0 to 4, and at layer L the write takes the slice labelled L. The `cosine` after the edit scores what each layer's stream keeps along its own direction; it is zero to within about 4e-6, which checks the write. For one direction removed at every layer, reduce the layers too, `"over": ["layers", "rows", "pos"]`. A direction read at fewer layers than the write is made at is refused before any forward. Writes across layers in general: [interventions](../usage/interventions.md#writes-across-layers).

## Heads and patterns

Ablating attention heads one at a time is a `select` on `attention_premix` with a sweep over `num_heads`, as [head and neuron patching](head-and-neuron-patching.md) does, with a zero or a mean in place of the counterfactual read. `documents/attention_knockout.json` zeroes one head's attention pattern instead, which leaves the head's values in place and removes what it moves ([attention](attention.md)).

## The hydra effect

When one component is ablated, later ones can change their output to make up for it, so the drop in the total effect understates what the component did. `documents/hydra_effect.json` measures that on the tiny Llama. It resample-ablates layer 0's attention at the last token, and measures a layer's direct effect by writing its attention output in as the final residual stream and reading the answer's logit:

```json
"ablated": {"kind": "forward", "input": "sample.input", "interventions": {
  "resample": {"site": "abl", "pos": -1, "ops": [{"op": "swap", "with": "noise.v_noise"}]},
  "logits": {"site": "head", "pos": -1},
  "a1": {"site": "probe1", "pos": -1}}},
"inj1_abl": {"kind": "forward", "input": "sample.input", "interventions": {
  "inject": {"site": "resid_final", "pos": -1, "ops": [{"op": "swap", "with": "ablated.a1"}]},
  "logits": {"site": "head", "pos": -1}}}
```

`ablated.a1` is layer 1's attention output read under the ablation, and a later forward writes it in: a read inside an intervened forward is an operand like any other. The saves read as: `te_clean` and `te_abl` are the answer's logit without and with the ablation (the total effect is their difference); `de0_c` is layer 0's direct effect in the clean run, what the ablation should remove; `de1_c` and `de1_a` are layer 1's direct effect clean and ablated. Compensation is `de1_a` moving away from `de1_c` in the direction that cancels the loss of `de0_c`.

## Pitfalls

- Zero ablation overstates importance; prefer a mean or a resample, and keep the zero as the extreme.
- A mean over a ragged read weighs tokens, not rows.
- A `mean` keeps `layers`: per-layer directions unless `over` names it.
- Score the ablated run against the clean run of the same document, not one from another run at another batch size: batch shape moves the last bits ([running](../usage/running.md)).
