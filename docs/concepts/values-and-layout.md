---
title: Values and Layout
one_liner: "How a value is laid out and compared: rows and positions first, the layers axis and alignment by label, select keeping an axis, missing entries as NaN, dtype rules, and why batch shape moves the last bits."
tags: [concepts, layout, layers, select, dtype, nan]
related: [docs/usage/sites.md, docs/concepts/positions.md, docs/usage/ops.md, docs/errors/alignment-missing.md, docs/usage/running.md, docs/patterns/logit-lens.md]
sources: [clab/format/address.py, clab/format/selectors.py, clab/steps/batches.py, clab/ops/transforms.py, clab/plan.py]
---

# Values and Layout

## The idea

Every value a document names has a layout: a list of axis names, recorded as `dims` on each entry in the manifest and printed by `clab show`. An op checks its operands against those names before any forward, acts along them, and says what layout it returns ([FORMAT.md#ops](../../FORMAT.md#ops)). So the layout is how clab knows which numbers to compare with which, and reading it is how you know what a saved tensor's indices mean.

## Rows and positions first

A selection of a component is laid out `(rows, pos, *the component's other axes, in its order)`. The component table's `batch` becomes `rows`, one per example of the step's dataset, and its token axis (`seq`, or `query` on an attention pattern) becomes `pos`, the positions the entry's `pos` picked ([FORMAT.md#components](../../FORMAT.md#components)). From the corpus, read at `pos: -1`:

| component | layout of the selection |
|---|---|
| `layer_output` | `rows, pos, hidden_size` |
| `attention_queries` | `rows, pos, num_heads, qk_head_dim` |
| `attention_premix` | `rows, pos, num_heads, head_dim` |
| `attention_probabilities` | `rows, pos, num_heads, key` |
| `lm_head_output` | `rows, pos, vocab_size` |

Ops then reshape it: a metric removes the vocabulary axis (`rows, pos`), `readout` turns `hidden_size` into `vocab_size`, `top_k` replaces the axis it ranks with `k`, and a reduction removes rows and positions unless `over` says otherwise.

## The layers axis

A site's `layers` written as an int gives that layer's value with no layers axis. A slice or a list gives one value with a `layers` axis first, labelled with the model's layers it picked, in the order picked: `"layers": [1, 0]` is index 0 for layer 1, and `clab table` prints the model's layer in its `layer` column, not the index.

Ops act per layer and align by label, not by index. An op over a value with a `layers` axis acts at each of its labels, and an operand that also has one is sliced at the same label: `kl` between a lens read at `":"` and one read at `[1, 0]` pairs layer 0 with layer 0 and is exactly 0 at both. An operand missing a label the op acts at is refused by name:

```text
clab: kl_ab: operand 'lens.lb' holds layers [1], and this op acts at layers [0, 1]; at each layer it takes the operand's slice of that layer, and it holds none of [0]: read it at those layers too
```

An operand without a `layers` axis is the same at every layer: `kl` of every layer's lens against the model's `logits` compares each layer with the one final distribution. A reduction keeps the `layers` axis unless `over` names it, so a `mean` of a per-layer metric is one number per layer; `"over": ["rows", "pos", "layers"]` averages it away. Inside an entry no op sees a `layers` axis, since the entry runs at one layer at a time ([usage/interventions.md](../usage/interventions.md)).

## `select` keeps the axis

A site's `select` narrows a feature axis and keeps it in the layout, so a site has one layout whatever it selects: `{"num_heads": 2}` gives `num_heads` of size 1, the same value as `[2]`. The kept indices are in the order written, and that order is the axis: with `{"num_heads": [3, 1]}`, index 0 along `num_heads` is head 3. Three consequences:

- A saved value's nested lists, in `clab table` or the tensor, follow the select order; the manifest's entry records the `select` that says which head each index is.
- `argmax` and `top_k` over a selected axis return indices into the selection. With `"select": {"key": [5, 0]}`, an `argmax` over `key` of 0 means key 5.
- An operand must hold the same picks in the same order as the write it meets; anything else is refused before it can swap head 3 into head 1:

```text
clab: patched.patch: operand 'counterfactual.v_cf' holds num_heads [3, 1], and this write's selection holds [1, 3]; a select list is positional, so they would trade places: select num_heads [1, 3] at the read
```

## The key axis

An attention pattern's `key` axis is each row's own keys, numbered as positions: key 0 is BOS where there is one, key j the row's token j. Unselected, it is as wide as the input's longest row, and past a shorter row's own keys a read is NaN and a write leaves the padding's keys as they were. A selector on `key` picks a fixed number of keys counted from the start (`0`, `[0, 2]`, `"0:3"`), the same keys on every row long enough to have them; one counted from the end is refused, since rows differ in length. The axis grows with every decode call, so a pattern is read in the prefill only.

## Missing entries

A value has a hole where a row holds no data: positions past a row's own on a ragged read, keys past a row's own, and every entry of a row that is `alignment_missing` ([concepts/positions.md](positions.md#decided-by-the-data-or-a-result)). A hole is NaN in a float value and -1 in ids (`argmax`, `top_k`'s `ids`); `top_k` gives no token text there. Metrics and edits of a hole are holes; `mean`, `sum` and `min` reduce over the entries that are there. How holes travel through a chain of ops and show up in tables is [errors/alignment-missing.md](../errors/alignment-missing.md).

## Broadcasting

An operand with no rows and no positions, such as a `mean` over every row and position, or a number, is the same for every row and position it meets. One that has positions does not broadcast along them: a metric that compares two distributions position by position refuses a pair read at different widths:

```text
clab: steps.kl_win_vs_last.against: 'clean.logits' holds 1 position(s) per row and the value 'kl' scores holds 3; position i scores against position i, so both are read at as many (a pos axis does not broadcast)
```

A write's operand meets the write's own picks, rows and positions as [usage/interventions.md](../usage/interventions.md) describes.

## Dtypes

A value's dtype is recorded beside its layout (`results.<name>.dtype` in the manifest). A read is the model's own tensor, so it has the model's dtype. An op that scores or averages upcasts what it is given before computing, and since every bfloat16 or float16 number is exact in float32, the upcast loses nothing: the only rounding left in a score or a mean is float32's. An edit computes the same way and then rounds its result once, to the value's dtype, because inside an entry that result is written back into the model. Which ops do which, the dtype each kind of value comes back in, and what that means for a bfloat16 model's numbers are the float32 rule in [usage/ops.md](../usage/ops.md#the-float32-rule).

## Why batch shape moves the last bits

Rows run in consecutive batches of the run's batch size, each padded on the left to the input's longest row, with an attention mask over the padding and position ids that count each row's real tokens. A row's numbers therefore do not depend on which rows share its batch, except that at some CPU thread counts a row padded well past its own length moves in the last bits (about 1e-8) with its companions. Every row's numbers depend on the batch's shape: a different number of rows is a different matrix multiply, whose sums may be ordered differently, and a generation's decode calls are `[batch, 1, ...]` multiplies. On the corpus Llama in float32 that moves logits by about 1e-7 and residual-stream values by about 2e-9 between batch sizes 1 and 6; two runs at one batch size are equal to the bit. Choosing a batch size and comparing runs is [usage/running.md](../usage/running.md#batch-size-and-reproducibility).
