---
title: Ops
one_liner: "Scoring and reducing values: metrics on logits or probe scores, reductions and over, transforms like readout, argmax, top_k and detokenize, op steps versus ops inside an entry, and the float32 rule."
tags: [usage, ops]
related: [docs/usage/interventions.md, docs/usage/data.md, docs/concepts/values-and-layout.md, docs/errors/alignment-missing.md, docs/patterns/logit-lens.md, docs/patterns/probing.md]
sources: [clab/ops/__init__.py, clab/ops/metrics.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/arguments.py]
---

# Ops

An op is one function that runs in two places: as an `op` step over a value an earlier step produced, or inside an entry of `interventions` on the entry's selection. Ops come in four groups: metrics score each example, reductions remove axes, edits change contents and write back ([interventions](interventions.md)), transforms change the layout. Every op's arguments and axes are in [FORMAT.md#ops](../../FORMAT.md#ops); this page is how to use them.

## Canonical example

The two scores of `documents/patching.json`, and the accuracy over rows that usually follows them:

```json
"iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
"logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
"accuracy": {"kind": "op", "op": "mean", "of": "iia"}
```

`of` names the value: an entry (`patched.logits`), an op step (`iia`), or a `generate` step's new token ids (`answer`). `iia` is 1 or 0 per row and position, `logit_diff` one number per row and position, `accuracy` one number. A step holds one op; a chain is one step per op.

## A step or an entry

Inside an entry, the ops after its last edit run on its value as each batch comes off the model; the entry binds their result. It is the same function as the step: `token_prob` inside the head's entry and as a step over the saved logits give the same numbers to the bit. Scoring inside the entry keeps memory flat, since a row's logits never outlive its batch; `documents/real/head_patching.json` scores its 96 points that way:

```json
"hit": {"site": "head", "pos": -1, "ops": [{"op": "match", "expected": "days.cf_answer"}]}
```

An entry sees one batch, so an op that aggregates over rows is refused there (`'mean' aggregates over rows, and inside an entry it sees one batch; make it a step after the forward`), and so is any reduction `over` `layers`: inside an entry an op acts at each layer in turn, and the `layers` axis exists only on what the entry binds. Make those op steps after the forward.

If the `mean` was meant as the value a write puts in, a mean ablation, it is an op step *before* the forward, over a read of the site, and the write swaps it in ([interventions](interventions.md#edit-recipes)). Moved after the forward, the entry has no edit left and only reads: the run gives the unpatched numbers, to the bit, without a refusal.

## Metrics

A metric returns one number per row and position, in float32. Most take logits (or a probe's scores) and an answer column. How a column becomes answer tokens (exact spelling, leading spaces, `strip`, `lower`, sets) is on [data](data.md#answer-columns); here is the choice of metric.

- `match`: 1 where the most likely token is the answer, else 0, an exact tie going to the lower id. The interchange accuracy of patching.
- `logit_diff`: one answer's logit less another's. Use it for a contrast between two answers: a single logit carries an offset shared by every token, which the softmax ignores and a difference cancels.
- `token_prob`: the answer's probability, the soft version of `match`. `cross_entropy` is minus its log.
- `kl` and `js`: how far a whole next-token distribution moved, against another read of the same layout (`documents/faithfulness.json` scores a patched run against the clean one), or of the same layout without its `layers` axis, which then compares every layer with that one distribution: a lens against the model's `logits` is the wiring check in [logit lens](../patterns/logit-lens.md#check-the-wiring-then-trust-the-lens). `against` must be a named value.
- `cosine`: a residual-stream value against another of the same layout or one vector, over `hidden_size` (`documents/cosine_metric.json`).
- `contains`: whether a row's decoded generation contains its answer ([generate](generate.md)).

An answer column whose row is a set of strings (`[" Saturday", " Sunday"]`) is any of them: `match` hits any, `token_prob` sums them. `logit_diff` and `token_logit` need one id, and preflight refuses a set there by row. An answer of several tokens scores the read's position i against token i, so the read must carry that many positions, and preflight refuses a row where it does not; a scope with `before` reads the positions that predict an answer ([positions](positions.md)).

**Vocabulary or classes.** After `encode` through a linear probe the last axis is `classes`, the distinct values of the probe's column, and the same metrics score it against a column of those labels ([probing](../patterns/probing.md)). `argmax` and `top_k` rank `vocab_size` by default; on probe scores give `"over": "classes"`, or they are refused.

## Reductions and `over`

`mean`, `sum` and `min` remove the rows and positions by default and keep a `layers` axis, which says which layer each value is. `over` names the axes to remove instead:

| `of` laid out | op | result |
|---|---|---|
| `layers, rows, pos` | `{"op": "mean"}` | `layers`: one number per layer |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers"]}` | `rows, pos` |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers", "rows", "pos"]}` | one number |
| `rows, pos, hidden_size` | `{"op": "mean", "over": ["pos"]}` | `rows, hidden_size`: one vector per row |

A reduction that would remove no axis is refused (`a reduction of a value laid out [] removes no axis of it`). A reduction takes the values that are there: over a read whose rows hold different numbers of positions, the `mean` is over every position present, so a row with three tokens weighs three times a row with one ([ablation](../patterns/ablation.md#a-mean-over-an-entitys-tokens)). `pca` is a reduction whose value is a bundle a `pca` featurizer loads ([trained subspaces](../patterns/trained-subspaces.md)).

## Transforms

- `readout` projects a residual-stream value through the model's final norm and head: the logit lens ([logit lens](../patterns/logit-lens.md)). It takes the components in the stream's basis (`token_embeddings`, `layer_input`, `attention_output`, `layer_mid`, `mlp_output`, `layer_output`) and refuses any other by name, and a site that selects part of `hidden_size`.
- `argmax` and `top_k` rank along `over`, by default the vocabulary: `argmax` gives the top id, `top_k` the `k` best `ids` with their `probs` (a float32 softmax). Over another axis (`key` on an attention pattern, `features` after an `encode`), `top_k` gives `values` instead of `probs`. As a step over logits, `top_k` also holds the decoded `tokens`; `clab table --csv` prints them whole. Each id is decoded alone, which drops a SentencePiece tokenizer's word-start `▁`: the corpus Llama's `▁Fed` (17972) prints as `Fed`, like `Handler` (4598), while `pos_token` keeps the `▁`. Read `ids` to tell ` Fed` from `Fed`. A later step cannot name a `top_k` step's `ids`, `probs` or `tokens` (`'top3.probs' names no value produced before this step`): save it and read the fields from `plan.results` ([output](output.md#in-python)).
- `detokenize` turns a `generate` step's ids into text, stopped before each row's first EOS ([generate](generate.md)).
- `encode` turns a value into a featurizer's coordinates ([featurizers](featurizers.md)).

## The float32 rule

Metrics and the reductions `mean`, `sum` and `min` compute in float32 and return float32, whatever the model's dtype. An edit computes in float32, its value and every operand upcast, and rounds once to the value's dtype, because it writes back. A read keeps the model's dtype. On a model loaded `"dtype": "bfloat16"`, the manifest's `results.<name>.dtype` reads:

| value | dtype |
|---|---|
| a read: `base.resid`, `base.logits` | bfloat16 |
| a metric, as a step or inside an entry | float32 |
| `mean`, `sum`, `min` | float32 |
| a write's value (`steered.steer`) and an edit as a step | bfloat16 |
| `top_k` over a vocabulary: `probs` | float32 |
| `top_k` over another axis: `values` | bfloat16 |
| `pca`'s `basis` and `mean` | bfloat16 |
| `argmax`, `top_k`'s `ids` | int64 |

`clab show` prints the dtype the model was served in (`served: bfloat16 on cpu`); [values and layout](../concepts/values-and-layout.md) explains the mechanism. What it means for a bfloat16 model's numbers:

- An op adds no rounding of its own. A probability or a cosine near 1 keeps float32's resolution, and a mean of bfloat16 reads is the float32 mean of them, not the bfloat16 one (they differ by about 6e-5 on the tiny model's residual stream).
- An edit rounds once: `add` of a float32 mean at `"scale": 4.0` writes `(value + 4 * mean)` computed in float32 and then rounded, which differs from adding in bfloat16.
- It cannot undo the model's own rounding. Logits a bfloat16 model returns are bfloat16: one near 20 is a multiple of 0.125, and a `logit_diff` of two of them moves in those steps. A difference finer than that needs the model in float32.
- Ranks are the same either way: `match`, `argmax` and `top_k` pick the same ids from a bfloat16 value and its float32 copy.

## Missing values

A row or position that a read did not reach (a generation that stopped early, a position past a shorter row's own) is NaN, and so is every metric and edit of it; `argmax` and `top_k` give it id -1. Reductions skip it: a `mean` is over the values present, NaN only where none is. Where missingness comes from and how to make it a refusal: [alignment missing](../errors/alignment-missing.md).

## Pitfalls

- `top_k` and `argmax` rank the vocabulary unless told otherwise; on probe scores, attention keys or features give `over`.
- A `mean` keeps `layers` by default; one number over layers needs `"over": ["layers", "rows", "pos"]`.
- `logit_diff` on a set answer is refused; score a set with `token_prob` or `match`.
