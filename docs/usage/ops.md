---
title: Ops
one_liner: "Which op scores, reduces or transforms a value? Metrics, KL in either direction, reductions (median, count), top_k over rows and positions, permute, arithmetic, and step versus entry."
tags: [usage, ops]
related: [docs/usage/interventions.md, docs/usage/data.md, docs/usage/output.md, docs/concepts/values-and-layout.md, docs/errors/alignment-missing.md, docs/patterns/logit-lens.md, docs/patterns/probing.md, docs/patterns/steering.md]
sources: [clab/ops/__init__.py, clab/ops/metrics.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/arguments.py, clab/steps/__init__.py, clab/steps/batches.py]
---

# Ops

An op is one function that runs in two places: as an `op` step over a value an earlier step produced, or inside an entry of `interventions` on the entry's selection. Metrics score each example; reductions remove axes; edits change contents and write back ([interventions](interventions.md)); transforms change the layout or compute on each entry without writing back. Every op's arguments and axes are in [FORMAT.md#ops](../../FORMAT.md#ops); this page is how to use them.

## Canonical example

The two scores of `documents/patching.json`, and the accuracy over rows that usually follows them:

```json
"iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
"logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
"accuracy": {"kind": "op", "op": "mean", "of": "iia"}
```

`of` names the value: an entry (`patched.logits`), an op step (`iia`), or a `generate` step's new token ids (`answer`). `iia` is 1 or 0 per row and position, `logit_diff` one number per row and position, `accuracy` one number. A step holds one op; a chain is one step per op.

## A step or an entry

Inside an entry, the ops after its last edit run on its value as each batch comes off the model; the entry binds their result. It is the same function as the step: `token_prob` or `kl` inside the head's entry and as a step over the saved logits give the same numbers ([running](running.md#when-two-runs-give-the-same-numbers)). `documents/real/head_patching.json` scores its 96 points that way:

```json
"hit": {"site": "head", "pos": -1, "ops": [{"op": "match", "expected": "days.cf_answer"}]}
```

Scoring inside the entry keeps memory flat, since a row's logits never outlive its batch. An op step holds more:

- A value a step binds stays on the model's device until the run ends, when the saved ones move to the CPU (`clab/steps/__init__.py` `detached_cpu_clone`).
- A read's batches are joined on that device, briefly twice the value's size (`clab/steps/batches.py` `join_batches`).
- An op step computes over the whole value at once. Over two saved `[rows, pos, vocab_size]` reads, `kl` builds several float32 temporaries of `rows × pos × vocab_size × 4` bytes each.
- A run that does not fit stops at that step, naming it and every value the run holds, with the steps before it saved ([running](running.md#when-a-step-runs-out-of-memory)).

Score in the later forward's head entry instead, `{"op": "kl", "against": "clean.logits"}`, which keeps only the reference. What an entry cannot hold:

- **An entry sees one batch.** An op that aggregates over rows is refused there (`'mean' aggregates over rows, and inside an entry it sees one batch`); make it an op step after the forward.
- **An entry acts at each layer in turn.** A reduction `over` `layers` is refused there, since the `layers` axis exists only on what the entry binds.
- **A reduction over `pos` alone may stay.** `{"op": "mean", "over": ["pos"]}` on a read at `layers: ":"` and `{"index": ":"}` gives the step form's numbers, and only the mean leaves the model.
- **A mean ablation's mean is a step before the forward.** It is an op step over a read of the site, and the write swaps it in ([interventions](interventions.md#edit-recipes)). Moved after the forward, as the refusal says, the entry has no edit left and only reads: the run gives the unpatched numbers, without a refusal.

## Metrics

A metric returns one number per row and position, in float32. Most take logits (or a probe's scores) and an answer column. How a column becomes answer tokens (spelling, leading spaces, `strip`, `lower`, sets, literals) is [data](data.md#answer-columns); here is the choice of metric.

- `match`: 1 where the most likely token is the answer, else 0, an exact tie going to the lower id. The interchange accuracy of patching.
- `logit_diff`: one answer's logit less another's. Use it for a contrast between two answers: a single logit carries an offset shared by every token, which the softmax ignores and a difference cancels.
- `token_logit`: one answer's raw logit. It carries the offset every token shares, so compare it only with another logit of the same read; a `logit_diff` is that difference.
- `token_prob`: the answer's probability, the soft version of `match`. `cross_entropy` is minus its log.
- `kl` and `js`: how far a whole next-token distribution moved, against another read (below).
- `cosine`: a residual-stream value against another of the same layout or one vector, over `hidden_size` (`documents/cosine_metric.json`).
- `contains`: whether a row's decoded generation contains its answer ([generate](generate.md)).

`kl` and `js` compare two distributions:

| field | what it is |
|---|---|
| `of` | the first distribution: the value scored, inside an entry the entry's value |
| `against` | the second, a named value: the same layout as `of`, or that layout without its `layers` axis, which compares every layer with one distribution |
| `"reverse": true` | makes `kl` KL(`against` ‖ `of`) instead of KL(`of` ‖ `against`); the two directions give different numbers |
| `js` | symmetric, so it takes no `reverse` |

- **`reverse` names the direction in either place.** In a step over two reads of one layout either may be `of`. Inside an entry `of` is the entry's value, and an operand may leave out `layers` but not carry one `of` lacks, so a lens is `of` against the logits ([logit lens](../patterns/logit-lens.md#check-the-wiring-then-trust-the-lens)).
- **KL(clean ‖ patched)**, as SAEBench reports it, is the patched forward's head entry `{"op": "kl", "against": "clean.logits", "reverse": true}`. `documents/faithfulness.json` scores KL(patched ‖ clean).
- **A tuned lens** is trained on KL(final ‖ lens): `kl` of the lens with `reverse` ([logit lens](../patterns/logit-lens.md#a-tuned-lens)).
- **A `kl` near zero can be a little below it.** Float32 rounding does that to two near-equal distributions (`documents/expert_knockout.json`'s `knockout.kl` at expert 2, row 2); read it as zero.

An answer the same on every row is a literal in the column's place: `{"op": "token_prob", "token": {"text": "</think>"}}` scores `</think>` on every row ([data](data.md#answer-columns)).

An answer column whose row is a set of strings (`[" Saturday", " Sunday"]`) is any of them: `match` hits any, `token_prob` sums them. `logit_diff` and `token_logit` need one id, and preflight refuses a set there by row; score a set with `token_prob` or `match`. An answer of several tokens scores the read's position i against token i. The read must carry that many positions, and preflight refuses a row where it does not; a scope with `before` reads the positions that predict an answer ([positions](positions.md)).

**A metric against an answer column needs the whole vocabulary.** It looks each answer up by token id along `vocab_size`, so on a head narrowed with `select: {"vocab_size": [...]}` the ids do not index the axis. Compile refuses it: `the value 'match' scores keeps 3 of its vocab_size (select), and it scores 'pairs.cf_answer' by its id over the whole vocabulary`. Score the full head. To look at a few candidates' logits as well, read the head a second time with the `select` and no metric. Those values equal the full head's at the selected ids, and an `argmax` over them is an index into the `select` list.

**Vocabulary or classes.** After `encode` through a linear probe the last axis is `classes`, the distinct values of the probe's column. The same metrics score it against a column of those labels ([probing](../patterns/probing.md)). `argmax` and `top_k` rank `vocab_size` by default; on probe scores give `"over": "classes"`, or they are refused.

## Reductions and `over`

`mean`, `sum`, `min`, `max`, `std` and `median` remove the rows and positions by default and keep a `layers` axis, which says which layer each value is. `over` names the axes to remove instead:

| `of` laid out | op | result |
|---|---|---|
| `layers, rows, pos` | `{"op": "mean"}` | `layers`: one number per layer |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers"]}` | `rows, pos` |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers", "rows", "pos"]}` | one number |
| `rows, pos, hidden_size` | `{"op": "mean", "over": ["pos"]}` | `rows, hidden_size`: one vector per row |

- `std` is the sample standard deviation, n − 1 in the denominator, NaN where fewer than two values are there. `std` of a metric over `rows` is the spread a mean effect is read against.
- `median` of an even count is the mean of the two middle values, as numpy's is (torch's is the lower of the two), and NaN where none is there. A steering dose set from the activations is the `median` of a `norm` ([steering](../patterns/steering.md#a-dose-set-from-the-activations)).
- `norm` and `dot` reduce over `hidden_size` by default; a value without one names its axis in `over`.
- `norm` takes `p` for another Lp norm (`1` is the sum of absolute values). At `p` 2 it equals the `sqrt` of the `sum` of the `square`.
- `dot` with a unit direction is each vector's component along it, `mul` then `sum`; `cosine` divides that by both lengths.
- `count` counts the positions along `over` (by default `pos`) that hold a value, a vector counted once, and removes the feature axes with them.
- `count` over `pos` is each row's own number of positions: of a read at `{"frame": "1:"}`, the decode calls the row made; of a row the read never reached, 0. `count` over `rows` of a scored read is how many rows it scored.
- A `count` is a number on every row, 0 where nothing is there, so a `mean` of it over `rows` takes those rows in.
- `entropy` is the entropy in nats of a float32 softmax over `over`: by default `vocab_size`, else `num_experts` ([expert ablation](../patterns/expert-ablation.md#routing-entropy-and-usage)). An attention score's is `"over": ["key"]`.
- A missing entry takes no share of `entropy`'s softmax, a vector with none is NaN, and a uniform softmax over n gives ln n.
- In ids a hole is -1, and every reduction leaves it out, `count` included: a `min` over a ragged `argmax` read is the least of a row's own ids.
- Over a ragged read a reduction takes the positions present, so a `mean` weighs positions, not rows: a row with three tokens weighs three times a row with one ([ablation](../patterns/ablation.md#a-mean-over-an-entitys-tokens)).
- A reduction that would remove no axis is refused: `a reduction of a value laid out [] removes no axis of it`.
- `pca` is a reduction whose value is the bundle a `pca` featurizer loads ([trained subspaces](../patterns/trained-subspaces.md)).

## What is not an op

The ops are the table in [FORMAT.md#ops](../../FORMAT.md#ops) and nothing else. No op reduces across sweep points, since an op sees one point at a time, so a ranking of heads across points is not one. There is no grouped mean, no statistic with an interval and no `where` with two branches. As steps:

- a mask is `mul` by a comparison, and a second branch is `mul` by the opposite comparison, then `add`;
- a relative effect is `sub` then `div`, and a recovery the same;
- a rate such as "logit difference above zero" is `gt` then `mean`.

What is left (a confidence interval, a ranking across points, a join across runs) is computed in Python from `Plan.load(out).results` ([output](output.md#in-python)).

## Transforms

- `readout` projects a residual-stream value through the model's final norm and head: the logit lens ([logit lens](../patterns/logit-lens.md)).
- `readout` takes `token_embeddings`, `layer_input`, `attention_output`, `layer_mid`, `mlp_output` and `layer_output`. It refuses any other component by name, and a site that selects part of `hidden_size`.
- `argmax` and `top_k` rank along `over`, by default `vocab_size`; on probe scores, attention keys or features give `over`. `argmax` gives the top id; `top_k` gives the `k` best `ids` with their `probs`, a float32 softmax.
- Over another axis (`key` on an attention pattern, `features` after an `encode`), `top_k` gives `values` instead of `probs`.
- As a step over logits, `top_k` also holds the decoded `tokens`; `clab table --csv` prints them whole.
- Each id is spelled as the tokenizer's piece, as `pos_token` spells it, a SentencePiece word-start `▁` kept: the corpus Llama's 17972 is `▁Fed`.
- A later step cannot name a `top_k` step's `ids`, `probs` or `tokens` (`'top3.probs' names no value produced before this step`). Save it and read the fields from `plan.results` ([output](output.md#in-python)).
- `top_k` over a list of axes ranks all their entries together, removes every listed axis and adds `k`.
- A feature's max-activating examples are one such step: `{"op": "top_k", "of": "harvest.feats", "k": 10, "over": ["rows", "pos"]}` is each feature's ten largest latents over every (row, position) pair, laid out `[features, k]`. A `top_k` over `rows` of a `max` over `pos` instead ranks each row by its best position and loses which position that was.
- Over a list, `ids` is one tensor per listed axis, keyed by axis name (`plan.results["top"]["ids"]["rows"]`, and `["pos"]`), beside `values`. A list of one axis (`"over": ["pos"]`) gives this form; a bare axis name gives one tensor.
- An exact tie goes to the lower index along the first axis listed, then the next.
- `clab table` of a `top_k` over `rows` prints a row per pick, with the picked row's columns and the picked position's token ([output](output.md#clab-table)).
- A list of axes is refused inside an entry, since an entry's batches do not join an index per axis; one axis named alone is not.
- `detokenize` turns a `generate` step's ids into text, stopped before each row's first EOS ([generate](generate.md)).
- `detokenize` with `after` (a column or a literal) keeps each row's text after the first occurrence of it, and the whole text where it does not occur.
- `encode` gives a featurizer's coordinates ([featurizers](featurizers.md)).
- `grad` is the first op of an entry. It puts the gradient of a later entry of the same forward in the value's place, in its layout, and the ops after it act on the gradient ([interventions](interventions.md#a-gradient-at-a-site)).
- `permute` reorders a value's rows, row i holding row π(i)'s, π drawn from `seed` alone. It is an op step only, and its record holds what it drew (`permutation`). A resample ablation swaps a permuted read into a later forward ([ablation](../patterns/ablation.md#resample-by-permuting-the-rows)).

### Arithmetic

`abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div` and `pow` compute on each entry in float32 and keep the layout. Each one's formula is its row in [FORMAT.md#transforms](../../FORMAT.md#transforms).

| to compute | ops |
|---|---|
| an effect's size whatever its sign | `abs` |
| a log-probability from a `token_prob` | `log` |
| a probability back from a log-probability | `exp` |
| whether two effects agree in direction | `sign` of each, then `eq` |
| a root-mean-square | `square`, `mean`, then `sqrt` |
| a power other than 2 | `pow` with its `exponent` |
| a read negated, or multiplied by a value, without writing back | `neg`, `mul` |
| a relative effect | `sub` of the clean score, then `div` by it, as steps |

- **They are transforms, not edits.** In a read they change only what the entry binds: `{"op": "neg"}` is the negated value, where `{"op": "scale", "factor": -1}` would write it back. An edit in a read still writes ([interventions](interventions.md#the-order-ops-run-in)).
- `div` takes a number or a named value, laid out by the operand rule as `scale`'s `factor` is. It is NaN wherever the divisor is 0, so a relative effect on a zero clean score is missing, and a `mean` leaves that row out.
- `log` is `-inf` at 0 and NaN below it; `sqrt` is NaN below 0.
- In ids a hole (-1) is NaN to every elementwise transform, a named operand's too.

### Comparisons

- `eq`, `ne`, `gt`, `ge`, `lt` and `le` compare each entry with a number or a named value. They give 1 where the comparison holds and 0 where not, in the value's dtype (float32 on ids).
- A comparison is NaN where either side is missing, a -1 id included, so `eq` and `ne` of the same arguments sum to exactly 1 wherever both are there.
- A comparison of a hole is a hole, so an agreement over a ragged read is over the positions present.
- A rate is the `mean` of a comparison: `{"kind": "op", "op": "gt", "of": "logit_diff", "with": 0}`, then a `mean` of it, is the fraction of rows whose patched logit difference is above zero.
- A comparison is a mask. `{"op": "mul", "with": "<the comparison>"}` in a read or a step keeps each entry where it holds and zeroes the rest. `{"op": "scale", "factor": "<the comparison>"}` in a write writes that masked value back.

## The float32 rule

Metrics, every reduction but `pca` and the elementwise arithmetic (`abs` to `pow`) compute in float32 and return float32, whatever the model's dtype. An edit computes in float32, its value and every operand upcast, and rounds once to the value's dtype, because it writes back. A comparison compares in float32 and gives its 0 or 1 in the value's dtype, float32 on ids, which hold no NaN for a hole. A read keeps the model's dtype. On a model loaded `"dtype": "bfloat16"`, the manifest's `results.<name>.dtype` reads:

| value | dtype |
|---|---|
| a read: `base.resid`, `base.logits` | bfloat16 |
| a metric, as a step or inside an entry | float32 |
| `mean`, `sum`, `min`, `max`, `std`, `median`, `count`, `norm`, `dot`, `entropy` | float32 |
| `abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div`, `pow` | float32 |
| a comparison (`eq`, `ne`, `gt`, `ge`, `lt`, `le`) of a read | bfloat16 |
| a comparison of ids (`argmax`) | float32 |
| a write's value (`steered.steer`) and an edit as a step | bfloat16 |
| `top_k` over a vocabulary: `probs` | float32 |
| `top_k` over another axis: `values` | bfloat16 |
| `pca`'s `basis` and `mean`, a bundle | float32 |
| `argmax`, `top_k`'s `ids` | int64 |

`clab show` prints the dtype the model was served in (`served: bfloat16 on cpu`); [values and layout](../concepts/values-and-layout.md) explains the mechanism. What it means for a bfloat16 model's numbers:

- An op adds no rounding of its own. A probability or a cosine near 1 keeps float32's resolution.
- A mean of bfloat16 reads is the float32 mean of them; the bfloat16 mean would be that mean rounded once more.
- An edit rounds once: `add` of a float32 mean at `"scale": 4.0` writes `(value + 4 * mean)` computed in float32 and then rounded, which differs from adding in bfloat16.
- It cannot undo the model's own rounding. Logits a bfloat16 model returns are bfloat16: one near 20 is a multiple of 0.125, and a `logit_diff` of two of them moves in those steps. A difference finer than that needs the model in float32.
- Ranks are the same either way: `match`, `argmax` and `top_k` pick the same ids from a bfloat16 value and its float32 copy.
- A bundle is float32 whatever the model's dtype, a `pca`'s as a fit's. Its basis is orthonormal to float32's precision in the file.
- A featurizer rounds its bundle to the value's dtype where it meets the value, since every featurizer's math runs in that dtype.
- A null that equals the clean run in bfloat16 proves the read and the write are at one place. It does not make an effect below the dtype's rounding readable.
- In bfloat16 every write that changes a value changes the rounding of every operation after it. A component with no effect in float32 still moves a bfloat16 run's logits by about their rounding step, at every layer. Recompute a few points in float32 before reading effects near that size.

## Missing values

A row or position that a read did not reach (a generation that stopped early, a position past a shorter row's own) is NaN. So is every metric, edit and elementwise transform of it, a comparison's included. `argmax` and `top_k` give it id -1, and a comparison, an elementwise transform or a reduction takes that id as missing again. Reductions skip it: a `mean` is over the values present, NaN only where none is, and `count` counts only the positions present. Where missingness comes from and how to make it a refusal: [alignment missing](../errors/alignment-missing.md).
