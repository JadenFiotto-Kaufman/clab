---
title: Ablation
one_liner: "What does a component contribute? Zero, mean, resample (permute) and optimal-constant ablation, a direction at every layer, and the hydra effect."
tags: [patterns, ablation]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/usage/featurizers.md, docs/usage/fit.md, docs/patterns/activation-patching.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/attention.md, docs/usage/positions.md]
sources: [clab/ops/edits.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/featurizers/constant.py, clab/compile/arguments.py]
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
| resample | `{"op": "swap", "with": "donor"}`, `donor` a `permute` of a read at the site | `documents/ablation_pair.json` | this input's value, replaced by another real input's |
| noise | `{"op": "noise", "scale": 0.5, "seed": 7}` | `documents/noise_write.json` | some of the signal, at a dose |
| direction | `{"op": "project_out", "direction": "direction"}` | `documents/direction_ablation.json` | one direction, the rest kept |
| optimal | `{"op": "apply", "featurizer": "fit.c"}`, `c` a `constant` drawn as the mean and trained | `documents/ablation_pair.json` | this input's value, replaced by the one value that best keeps a chosen score |

A zero vector is a point no forward reaches: the downstream norms and attention see an input off their distribution. So a zero ablation measures that shock as well as the loss, and overstates importance. Mean and resample ablation keep the value on the distribution and are the usual choice; keep the zero as the extreme. `noise` adds the same noise in every forward ([FORMAT.md#edits](../../FORMAT.md#edits)). For `noise` and for `permute`, sweep `seed` (`{"sweep": [1, 2, 3]}`) to see how much one draw moves the result.

The harvest-then-ablate chain of `documents/mean_ablation.json`:

```json
"harvest": {"kind": "forward", "input": "prompts.input", "interventions": {"acts": {"site": "target", "pos": -1}}},
"mean": {"kind": "op", "op": "mean", "of": "harvest.acts"},
"ablated": {"kind": "forward", "input": "prompts.input", "interventions": {
  "ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]},
  "logits": {"site": "head", "pos": -1}}}
```

`mean` removes the rows and the position, leaving one `hidden_size` vector, which broadcasts over every row the write selects.

## Resample by permuting the rows

`permute` reorders the rows of a read the document already has, row i holding row π(i)'s, so the donor is an op step and costs no forward ([FORMAT.md#transforms](../../FORMAT.md#transforms) has what its record holds). From `documents/ablation_pair.json`, one attention head's premix at every position (`ld` is the logit difference at the last):

```json
"clean": {"kind": "forward", "input": "pairs.input", "interventions": {
  "z": {"site": "z", "pos": {"index": "0:12"}}, "ld": {"site": "head", "pos": -1, "ops": [...]}}},
"donor": {"kind": "op", "op": "permute", "of": "clean.z", "seed": 0, "derangement": true},
"donor_identity": {"kind": "op", "op": "permute", "of": "clean.z", "seed": null},
"resampled": {"kind": "forward", "input": "pairs.input", "interventions": {
  "abl": {"site": "z", "pos": {"index": "0:12"}, "ops": [{"op": "swap", "with": "donor"}]},
  "ld": {"site": "head", "pos": -1, "ops": [...]}}}
```

- **`seed: null` is the null control.** The same swap with `donor_identity` (the document's `null`) equals the clean run.
- **`derangement: true` leaves no row its own value**, which would be a null inside the average. It needs two rows, in the dataset and, in a fit's body, in every minibatch; fewer is refused by name.
- **In a fit's body, write `derangement: true`.** Each training minibatch draws π over its own rows from the seed alone. Over two rows a seed may draw the identity (seed 0 does), and then no training step resamples anything. The step's record holds the π of the scoring pass over every training row, which no training step used, so nothing on the run shows it.
- **A fit's minibatches of one size pair their rows the same way.** With two rows and `derangement`, each row's donor is always its minibatch partner. To pair rows across the whole dataset, permute a read made before the fit: the body takes its rows by id, row i's being row π(i)'s.
- **Positions pair by index.** Row i holds row π(i)'s positions, so the swap meets them as it would a read of another input. Both are at one static width (`{"index": "0:12"}`, or `"-3:"`), and every prompt is one token length where the write covers whole prompts. A ragged read permuted is refused as the operand of another forward; a reduction of it is not.
- **π pairs any two rows.** A constraint on which rows pair (a donor with another name, or another template) is data: a column holding each row's donor prompt, read by a forward of its own, as `documents/hydra_effect.json` reads its `resample_input`.

## The optimal constant

A `constant` featurizer holds one vector of the site's width: drawn from `init`, trained by a fit, and written in place of the value by `apply`. What the score loses under the trained constant is what no constant can stand in for. From `documents/ablation_pair.json`:

```json
"featurizers": {"c": {"kind": "constant", "init": "mean"}},
"mean": {"kind": "op", "op": "mean", "of": "clean.z", "over": ["rows", "pos", "num_heads"]},
"fit": {"kind": "fit", "train": ["c"], "objective": {"sq": 1.0}, "epochs": 2, "batch_size": 2, "seed": 0,
        "optimizer": {"name": "adam", "lr": 0.01}, "steps": {
  "abl": {"kind": "forward", "input": "pairs.input", "interventions": {
    "abl": {"site": "z", "pos": {"index": "0:12"}, "ops": [{"op": "apply", "featurizer": "fit.c"}]},
    "ld": {"site": "head", "pos": -1, "ops": [...]}}},
  "change": {"kind": "op", "op": "sub", "of": "abl.ld", "with": "clean.ld"},
  "sq": {"kind": "op", "op": "square", "of": "change"}}},
"optimal": {"kind": "forward", "input": "pairs.input", "interventions": {
  "abl": {"site": "z", "pos": {"index": "0:12"}, "ops": [{"op": "apply", "featurizer": "fit.c"}]}, "ld": ...}}
```

- **Drawn as the mean, it is mean ablation, the fit's null control.** The document's `mean_ablated` applies the bare `c`, which gives `{"op": "swap", "with": "mean"}`'s numbers, and so does `fit.c` at `epochs: 0` ([fit](../usage/fit.md#epochs-0-is-a-null-control)). `"init": 0` is zero ablation the same way.
- **Training starts from there.** In `documents/ablation_pair.json`'s two epochs the squared change of each row's logit difference falls from mean ablation's 8.2e-5 to 8.9e-6.
- **`init` is one vector laid out on the site's last axis.** The mean is over every other axis, `num_heads` included, which the site's `select` keeps one of. Refused by name when the document compiles:
  - an `init` with rows, a `layers` axis or another axis;
  - one in another featurizer's coordinates, or on a sweep axis;
  - one a step makes after the constant's first use.

  One that keeps other indices of the last axis than the use does is refused before any forward, once the model's sizes resolve them.
- **A head set needs a site and a constant per head.** The vector is shared across every axis but the last, so at a site that keeps several heads every head takes the same vector, and `init: "mean"` is the heads' common mean. For each head's own mean and optimum, declare a site per head (`"select": {"num_heads": [h]}`) with a constant of its own, as `documents/ablation_pair.json` does for its one head.
- **It fits the score it is trained on, and may move the rest to do it.** A constant trained to hold one logit difference can hold it by distorting the rest of the next-token distribution. Score it on held-out rows and against a second measure (a `kl` against the clean logits) before reading the remaining loss as what the head's input-specific part carries. `fit.c` after the fit is an ordinary featurizer: saved, it is a bundle holding `vector`, which another document loads with `file_path` ([bundles.md](../usage/bundles.md)).

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

`harvest.acts` itself, with its rows, is refused on `other`'s rows ([interventions](../usage/interventions.md#operands)).

## A mean over an entity's tokens

`documents/entity_mean_ablation.json` harvests every token of each row's entity, `{"scope": {"column": "prompts.entity"}, "index": ":"}`: three positions for " Thursday", one for " Friday". The read is as wide as its widest row, NaN past a shorter row's own. `mean` averages the positions present, so the result is one vector that lands at any position. That mean weighs every token equally, so a row whose entity has three tokens counts three times ([ops](../usage/ops.md#reductions-and-over)). To weigh rows equally, average over the positions first:

```json
"per_row": {"kind": "op", "op": "mean", "of": "harvest.acts", "over": ["pos"]},
"mean": {"kind": "op", "op": "mean", "of": "per_row"}
```

The two means differ wherever entities differ in length. A row whose entity is not in its prompt is refused by name before any forward.

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

`ablated.a1` is layer 1's attention output read under the ablation, and a later forward writes it in: a read inside an intervened forward is an operand like any other. The saves:

- `te_clean` and `te_abl` are the answer's logit without and with the ablation; the total effect is their difference.
- `de0_c` is layer 0's direct effect in the clean run, what the ablation should remove.
- `de1_c` and `de1_a` are layer 1's direct effect, clean and ablated.
- Compensation is `de1_a` moving away from `de1_c` in the direction that cancels the loss of `de0_c`.
