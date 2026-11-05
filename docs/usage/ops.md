---
title: Ops
one_liner: "Scoring and reducing values: metrics on logits or probe scores, reductions and over (norm, dot, count, std, entropy among them), transforms like readout, argmax, top_k, detokenize and grad, elementwise arithmetic and comparisons, op steps versus ops inside an entry, and the float32 rule."
tags: [usage, ops]
related: [docs/usage/interventions.md, docs/usage/data.md, docs/usage/output.md, docs/concepts/values-and-layout.md, docs/errors/alignment-missing.md, docs/patterns/logit-lens.md, docs/patterns/probing.md, docs/patterns/steering.md]
sources: [clab/ops/__init__.py, clab/ops/metrics.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/arguments.py, clab/steps/__init__.py, clab/steps/batches.py]
---

# Ops

An op is one function that runs in two places: as an `op` step over a value an earlier step produced, or inside an entry of `interventions` on the entry's selection. Ops come in four groups: metrics score each example, reductions remove axes, edits change contents and write back ([interventions](interventions.md)), transforms change the layout or compute on each entry without writing back. Every op's arguments and axes are in [FORMAT.md#ops](../../FORMAT.md#ops); this page is how to use them.

## Canonical example

The two scores of `documents/patching.json`, and the accuracy over rows that usually follows them:

```json
"iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
"logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
"accuracy": {"kind": "op", "op": "mean", "of": "iia"}
```

`of` names the value: an entry (`patched.logits`), an op step (`iia`), or a `generate` step's new token ids (`answer`). `iia` is 1 or 0 per row and position, `logit_diff` one number per row and position, `accuracy` one number. A step holds one op; a chain is one step per op.

## A step or an entry

Inside an entry, the ops after its last edit run on its value as each batch comes off the model; the entry binds their result. It is the same function as the step: `token_prob` or `kl` inside the head's entry and as a step over the saved logits give the same numbers to the bit. `documents/real/head_patching.json` scores its 96 points that way:

```json
"hit": {"site": "head", "pos": -1, "ops": [{"op": "match", "expected": "days.cf_answer"}]}
```

Scoring inside the entry keeps memory flat, since a row's logits never outlive its batch. A value a step binds stays on the model's device until the run ends, when the saved ones move to the CPU (`clab/steps/__init__.py` `detached_cpu_clone`); a read's batches are joined on that device, briefly twice the value's size (`clab/steps/batches.py` `join_batches`); and an op step computes over the whole value at once. So an op step over two saved `[rows, pos, vocab_size]` float32 reads builds temporaries their size: a `kl` between two `[64, 127, 256000]` reads makes several of 8 GB each, and a run that does not fit stops at that step, naming it and every value the run holds, with the steps before it saved ([running.md](running.md#when-a-step-runs-out-of-memory)). Score in the later forward's head entry instead, `{"op": "kl", "against": "clean.logits"}`, which keeps only the reference.

An entry sees one batch, so an op that aggregates over rows is refused there (`'mean' aggregates over rows, and inside an entry it sees one batch; make it a step after the forward`), and so is any reduction `over` `layers`: inside an entry an op acts at each layer in turn, and the `layers` axis exists only on what the entry binds. Make those op steps after the forward. A reduction over `pos` alone keeps the rows and may stay in the entry: `{"op": "mean", "over": ["pos"]}` on a read at `layers: ":"` and `{"index": ":"}` gives the step form's numbers to the bit, and only the mean leaves the model.

If the `mean` was meant as the value a write puts in, a mean ablation, it is an op step *before* the forward, over a read of the site, and the write swaps it in ([interventions](interventions.md#edit-recipes)). Moved after the forward, the entry has no edit left and only reads: the run gives the unpatched numbers, to the bit, without a refusal.

## Metrics

A metric returns one number per row and position, in float32. Most take logits (or a probe's scores) and an answer column. How a column becomes answer tokens (exact spelling, leading spaces, `strip`, `lower`, sets) is on [data](data.md#answer-columns); here is the choice of metric.

- `match`: 1 where the most likely token is the answer, else 0, an exact tie going to the lower id. The interchange accuracy of patching.
- `logit_diff`: one answer's logit less another's. Use it for a contrast between two answers: a single logit carries an offset shared by every token, which the softmax ignores and a difference cancels.
- `token_prob`: the answer's probability, the soft version of `match`. `cross_entropy` is minus its log.
- `kl` and `js`: how far a whole next-token distribution moved, against another read of the same layout (`documents/faithfulness.json` scores a patched run against the clean one), or of the same layout without its `layers` axis, which then compares every layer with that one distribution: a lens against the model's `logits` is the wiring check in [logit lens](../patterns/logit-lens.md#check-the-wiring-then-trust-the-lens). `against` must be a named value. `kl` is KL(`of` ‖ `against`), the value scored the first distribution and the named one the second, and `"reverse": true` makes it KL(`against` ‖ `of`); the two directions give different numbers. In a step over two reads of one layout either may be `of` (`documents/faithfulness.json` scores `patched.logits` against `clean.logits`). The side is fixed in two places: inside an entry `of` is the entry's value, and an operand may leave out the `layers` axis but not carry one `of` lacks, so a lens is `of` against the logits. `reverse` names the direction in either place: SAEBench reports KL(clean ‖ patched), which the patched forward's head entry scores as `{"op": "kl", "against": "clean.logits", "reverse": true}`, and a tuned lens is trained on KL(final ‖ lens), `kl` of the lens with `reverse` ([logit lens](../patterns/logit-lens.md#a-tuned-lens)). `js` is symmetric and takes none.
- `cosine`: a residual-stream value against another of the same layout or one vector, over `hidden_size` (`documents/cosine_metric.json`).
- `contains`: whether a row's decoded generation contains its answer ([generate](generate.md)).

An answer the same on every row is a literal in the column's place: `{"op": "token_prob", "token": {"text": "</think>"}}` scores `</think>` on every row, and `{"op": "contains", "of": "text", "text": {"text": "\\boxed{", "strip": true}}` looks for one fixed text. A literal is read as a column holding its text on every row is, flags and sets included ([data](data.md#answer-columns)), and gives the same bits.

An answer column whose row is a set of strings (`[" Saturday", " Sunday"]`) is any of them: `match` hits any, `token_prob` sums them. `logit_diff` and `token_logit` need one id, and preflight refuses a set there by row. An answer of several tokens scores the read's position i against token i, so the read must carry that many positions, and preflight refuses a row where it does not; a scope with `before` reads the positions that predict an answer ([positions](positions.md)).

**A metric against an answer column needs the whole vocabulary.** It looks each answer up by token id along `vocab_size`, so on a head narrowed with `select: {"vocab_size": [...]}` the ids do not index the axis, and compile refuses it: `steps.iia.expected: the value 'match' scores keeps 3 of its vocab_size (select), and it scores 'pairs.cf_answer' by its id over the whole vocabulary, where index i of the selection is not id i: score the whole vocabulary (read it without the select, and narrow what the metric gives), or rank within the selection by index (argmax or top_k over it)`. Score the full head. To look at a few candidates' logits as well, read the head a second time with the `select` and no metric: those values equal the full head's at the selected ids, and an `argmax` over them is an index into the `select` list.

**Vocabulary or classes.** After `encode` through a linear probe the last axis is `classes`, the distinct values of the probe's column, and the same metrics score it against a column of those labels ([probing](../patterns/probing.md)). `argmax` and `top_k` rank `vocab_size` by default; on probe scores give `"over": "classes"`, or they are refused.

## Reductions and `over`

`mean`, `sum`, `min`, `max` and `std` remove the rows and positions by default and keep a `layers` axis, which says which layer each value is. `over` names the axes to remove instead:

| `of` laid out | op | result |
|---|---|---|
| `layers, rows, pos` | `{"op": "mean"}` | `layers`: one number per layer |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers"]}` | `rows, pos` |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers", "rows", "pos"]}` | one number |
| `rows, pos, hidden_size` | `{"op": "mean", "over": ["pos"]}` | `rows, hidden_size`: one vector per row |

`std` is the sample standard deviation, `sqrt(sum((x - mean)^2) / (n - 1))` over the n values there, NaN where fewer than two are. `norm` and `dot` reduce a vector's features: by default over `hidden_size`, which a value without one names in `over`. `{"kind": "op", "op": "norm", "of": "base.resid"}` is each row and position's length, the `sqrt` of the `sum` of the `square` to the bit, and `"p"` gives another Lp norm (`1`, the sum of absolute values); `{"kind": "op", "op": "dot", "of": "base.resid", "with": "direction"}` is each vector's component along a unit `direction`, `mul` by it then `sum`, to the bit (`cosine` divides that by both lengths). `count` counts the positions along `over`, by default `pos`, that hold the value, a vector counted once, so it removes the features with them: on `weekdays/train` a read at the whole-entity scope counts 3, 1, 1 and 1, each row's own tokens, a generation's read at `{"frame": "1:"}` counts the decode calls each row made, and a row a read never reached counts 0, so `count` over `rows` of a scored read is how many rows it scored. In ids a hole is -1, and every reduction leaves it out, `count` too: over a ragged `expert_indices` or `argmax` read, a `min` is the least of a row's own ids and a `mean` their mean, never pulled down by the -1 past a shorter row's positions. A `count` is a number on every row, 0 on a row with no position there (a row that is all NaN), so a `mean` of it over `rows` takes those rows in too.

`entropy` is the entropy in nats of a softmax over the axes `over` names, by default `vocab_size` where the value has it, else `num_experts`, so `{"op": "entropy"}` at the head is the next-token distribution's and at `router_logits` the routing's ([expert ablation](../patterns/expert-ablation.md#routing-entropy-and-usage)); an attention score's is `"over": ["key"]`. A missing entry has no share of the softmax (a row's keys past its own), a vector with none is NaN, and a uniform softmax over n gives ln n.

A reduction that would remove no axis is refused (`a reduction of a value laid out [] removes no axis of it`). A reduction takes the values that are there: over a read whose rows hold different numbers of positions, the `mean` is over every position present, so a row with three tokens weighs three times a row with one ([ablation](../patterns/ablation.md#a-mean-over-an-entitys-tokens)). `pca` is a reduction whose value is a bundle a `pca` featurizer loads ([trained subspaces](../patterns/trained-subspaces.md)).

## What is not an op

The ops are the table in [FORMAT.md#ops](../../FORMAT.md#ops) and nothing else. No op reduces across sweep points, since an op sees one point at a time, so a ranking of heads across points is not one; there is no grouped mean, no statistic with an interval and no `where` with two branches (a mask is `mul` by a comparison, and a second branch is `mul` by the opposite comparison, then `add`, as steps). As steps, a relative effect is `sub` then `div`, a recovery the same, and a rate such as "logit difference above zero" is `gt` then `mean`; what is left (a confidence interval, a ranking across points, a join across runs) is computed in Python from `Plan.load(out).results` ([output](output.md#in-python)).

## Transforms

- `readout` projects a residual-stream value through the model's final norm and head: the logit lens ([logit lens](../patterns/logit-lens.md)). It takes the components in the stream's basis (`token_embeddings`, `layer_input`, `attention_output`, `layer_mid`, `mlp_output`, `layer_output`) and refuses any other by name, and a site that selects part of `hidden_size`.
- `argmax` and `top_k` rank along `over`, by default the vocabulary: `argmax` gives the top id, `top_k` the `k` best `ids` with their `probs` (a float32 softmax). Over another axis (`key` on an attention pattern, `features` after an `encode`), `top_k` gives `values` instead of `probs`. As a step over logits, `top_k` also holds the decoded `tokens`; `clab table --csv` prints them whole. Each id is spelled as the tokenizer's piece, as `pos_token` spells it, a SentencePiece word-start `▁` kept: the corpus Llama's 17972 is `▁Fed`, and 4598 is `Handler`. A later step cannot name a `top_k` step's `ids`, `probs` or `tokens` (`'top3.probs' names no value produced before this step`): save it and read the fields from `plan.results` ([output](output.md#in-python)).
- `detokenize` turns a `generate` step's ids into text, stopped before each row's first EOS ([generate](generate.md)); with `after` (a column or a literal), each row's text after the first occurrence of it, the whole text where it does not occur ([generate](generate.md#score-what-follows-a-text-after)).
- `encode` turns a value into a featurizer's coordinates ([featurizers](featurizers.md)).
- `grad` puts the gradient of a later entry of the same forward in the value's place, in its layout; it is the first op of an entry, and the ops after it act on the gradient ([interventions](interventions.md#a-gradient-at-a-site)).

`abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div` and `pow` compute on each entry in float32 and keep the layout; each one's formula is its row in [FORMAT.md#transforms](../../FORMAT.md#transforms). They are transforms, not edits, so in a read they change what the entry binds and nothing the model sees: `{"op": "neg"}` in a read is the negated value, where `{"op": "scale", "factor": -1}` would write it back, and `{"op": "mul", "with": "direction"}` the product, where `scale` by `direction` would. An edit stays an edit in a read: `add`, `sub`, `scale` and `normalize` in an entry write back what they compute, whatever ops follow them, so a difference or a relative effect of a read is `sub` then `div` as steps after the forward, and a product inside the read is `mul`. `div` takes a number or a named value, laid out by the operand rule as `scale`'s `factor` is, and is NaN wherever that is 0, so a relative effect, as steps `sub` of the clean score then `div` by it, is missing rather than infinite on a row whose clean score is zero, and a `mean` leaves that row out. `log` is `-inf` at 0 and NaN below it, `sqrt` NaN below 0. In ids a hole, -1, is NaN to each of them, a named operand's too, as it is to a comparison.

`eq`, `ne`, `gt`, `ge`, `lt` and `le` compare each entry with a number or a named value: 1 where the comparison holds, 0 where not, in the value's dtype (float32 on ids), and NaN where either side is missing, a -1 id included, so `eq` and `ne` of the same arguments sum to exactly 1 wherever both are there. A rate is the `mean` of one, over the rows that are there: `{"kind": "op", "op": "gt", "of": "logit_diff", "with": 0}`, then a `mean` of it, is the fraction of rows whose patched logit difference is above zero. A product with one is a mask: in a read or as a step `{"op": "mul", "with": "<the comparison>"}` keeps each entry where it holds and zeroes it where not, and in a write `{"op": "scale", "factor": "<the comparison>"}` writes that masked value back. Ids compare as they are, and a comparison of a hole is a hole: on `weekdays/train` read at every token (11, 9, 9 and 9 positions), `{"kind": "op", "op": "eq", "of": "top", "with": "other_top"}`, of the logits' `argmax` and the `argmax` of their `square`, is NaN at the six positions past the shorter rows' own, and a `mean` of it, 0.737, is the agreement over the 38 positions there.

## The float32 rule

Metrics, every reduction but `pca` and the elementwise arithmetic (`abs` to `pow`) compute in float32 and return float32, whatever the model's dtype. An edit computes in float32, its value and every operand upcast, and rounds once to the value's dtype, because it writes back. A comparison compares in float32 and gives its 0 or 1 in the value's dtype, float32 on ids, which hold no NaN for a hole. A read keeps the model's dtype. On a model loaded `"dtype": "bfloat16"`, the manifest's `results.<name>.dtype` reads:

| value | dtype |
|---|---|
| a read: `base.resid`, `base.logits` | bfloat16 |
| a metric, as a step or inside an entry | float32 |
| `mean`, `sum`, `min`, `max`, `std`, `count`, `norm`, `dot`, `entropy` | float32 |
| `abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div`, `pow` | float32 |
| a comparison (`eq`, `ne`, `gt`, `ge`, `lt`, `le`) of a read | bfloat16 |
| a comparison of ids (`argmax`) | float32 |
| a write's value (`steered.steer`) and an edit as a step | bfloat16 |
| `top_k` over a vocabulary: `probs` | float32 |
| `top_k` over another axis: `values` | bfloat16 |
| `pca`'s `basis` and `mean`, a bundle | float32 |
| `argmax`, `top_k`'s `ids` | int64 |

`clab show` prints the dtype the model was served in (`served: bfloat16 on cpu`); [values and layout](../concepts/values-and-layout.md) explains the mechanism. What it means for a bfloat16 model's numbers:

- An op adds no rounding of its own. A probability or a cosine near 1 keeps float32's resolution, and a mean of bfloat16 reads is the float32 mean of them, not the bfloat16 one (they differ by about 6e-5 on the tiny model's residual stream).
- An edit rounds once: `add` of a float32 mean at `"scale": 4.0` writes `(value + 4 * mean)` computed in float32 and then rounded, which differs from adding in bfloat16.
- It cannot undo the model's own rounding. Logits a bfloat16 model returns are bfloat16: one near 20 is a multiple of 0.125, and a `logit_diff` of two of them moves in those steps. A difference finer than that needs the model in float32.
- Ranks are the same either way: `match`, `argmax` and `top_k` pick the same ids from a bfloat16 value and its float32 copy.
- A bundle is float32 whatever the model's dtype, a `pca`'s as a fit's: its basis is orthonormal to float32's precision in the file, and a featurizer rounds it to the value's dtype where it meets the value, as every featurizer's math runs in that dtype.
- A null that holds to the bit in bfloat16 proves the read and the write are at one place; it does not make an effect below the dtype's rounding readable. On Qwen3-4B, a head whose effect is exactly zero in float32 shows about ±0.02 in a mean logit difference over 120 rows in bfloat16, early layers included. Recompute a few points in float32 before reading effects near that size.

## Missing values

A row or position that a read did not reach (a generation that stopped early, a position past a shorter row's own) is NaN, and so is every metric, edit and elementwise transform of it, a comparison's included; `argmax` and `top_k` give it id -1, and a comparison, an elementwise transform or a reduction takes that id as missing again. Reductions skip it: a `mean` is over the values present, NaN only where none is, and `count` counts only the positions present. Where missingness comes from and how to make it a refusal: [alignment missing](../errors/alignment-missing.md).

## Pitfalls

- `top_k` and `argmax` rank the vocabulary unless told otherwise; on probe scores, attention keys or features give `over`.
- A `mean` keeps `layers` by default; one number over layers needs `"over": ["layers", "rows", "pos"]`.
- `logit_diff` on a set answer is refused; score a set with `token_prob` or `match`.
