---
title: Values and Layout
one_liner: "How is a value laid out and aligned: its axes, the layers axis, select, missing entries, dtypes, and batch shape?"
tags: [concepts, layout, layers, select, dtype, nan]
related: [docs/usage/sites.md, docs/concepts/positions.md, docs/usage/ops.md, docs/errors/alignment-missing.md, docs/usage/running.md, docs/patterns/logit-lens.md]
sources: [clab/format/address.py, clab/format/selectors.py, clab/steps/batches.py, clab/ops/transforms.py, clab/plan.py]
---

# Values and Layout

## The idea

Every value a document names has a layout: a list of axis names, recorded as `dims` on its record in the manifest and printed by `clab show`. Beside `dims` the record carries the labels that say what each axis holds:

- `dataset`: whose rows;
- `layers`: the model's layers;
- `read`: the entry whose positions it holds;
- `width`: how many positions every row holds, where that is one number;
- `kept`: the indices a `select` keeps;
- `space`: the featurizer coordinates it is in.

Each read and write also carries its resolved `positions` ([FORMAT.md#derived-fields](../../FORMAT.md#derived-fields)). An op checks its operands against those names and labels before any forward, acts along them, and says what layout it returns ([FORMAT.md#ops](../../FORMAT.md#ops)). So the layout is how clab knows which numbers to compare with which, and reading it is how you know what a saved tensor's indices mean. A saved result carries the same layout and labels in the manifest's `results`.

A value's record also says which sweep axes it depends on (`axes`): its own entry's sweeps, those of what its ops name, and, for an entry of a forward, the forward's, which are its writes'. A value is made once per distinct point on those axes, and a saved one is keyed by them ([names-and-sweeps.md](names-and-sweeps.md)).

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

A site's `layers` written as an int gives that layer's value with no layers axis. A slice or a list gives one value with a `layers` axis first, labelled with the model's layers it picked, in the order picked: `"layers": [1, 0]` is index 0 for layer 1. `clab table` prints the model's layer in its `layer` column, not the index.

Ops act per layer and align by label, not by index:

- An op over a value with a `layers` axis acts at each of its labels.
- An operand that also has one is sliced at the same label, whatever order each lists its layers in ([logit-lens](../patterns/logit-lens.md#comparing-two-lenses-kl-by-layer-label) compares two lenses this way).
- An operand missing a label the op acts at is refused by name: `operand 'lens.lb' holds layers [1], and this op acts at layers [0, 1]; …`.
- An operand without a `layers` axis is the same at every layer.
- A reduction keeps the `layers` axis unless `over` names it, so a `mean` of a per-layer metric is one number per layer; `"over": ["rows", "pos", "layers"]` averages it away.

Inside an entry no op sees a `layers` axis, since the entry runs at one layer at a time ([usage/interventions.md](../usage/interventions.md)).

## `select` keeps the axis

A site's `select` narrows a feature axis and keeps it in the layout, so a site has one layout whatever it selects: `{"num_heads": 2}` gives `num_heads` of size 1, the same value as `[2]`. The kept indices are in the order written, and that order is the axis: with `{"num_heads": [3, 1]}`, index 0 along `num_heads` is head 3. The record's `kept` says which head each index is, and a saved value's nested lists, in `clab table` or the tensor, follow that order. What follows for operands and head sets is [usage/sites.md](../usage/sites.md#select).

## An index counts along the value

An op that returns an index, `argmax` or `top_k`'s `ids`, counts along the axis as the value holds it, not along the model's:

- Over a selected axis it is an index into the selection. With `"select": {"key": [5, 0]}`, an `argmax` over `key` of 0 means key 5.
- Over `pos` it is an index along the read's positions on that row, not a position. Under `{"index": "1:"}`, an `argmax` over `pos` of j is position j + 1; the record's `positions` map it back.

## The key axis

An attention pattern's `key` axis is each row's own keys, numbered as positions: key 0 is BOS where there is one, key j the row's token j. Unselected, it is as wide as the input's longest row. Past a shorter row's own keys a read is NaN, and a write leaves the padding's keys as they were. A selector on `key` picks a fixed number of keys counted from the start (`0`, `[0, 2]`, `"0:3"`), the same keys on every row long enough to have them; one counted from the end is refused, since rows differ in length. The axis grows with every decode call, so a pattern is read in the prefill only.

## Missing entries

A value has a hole where a row holds no data:

- positions past a row's own on a ragged read;
- keys past a row's own on an attention pattern;
- every entry of a row that is `alignment_missing` ([concepts/positions.md](positions.md#decided-by-the-data-or-a-result)).

A hole is NaN in a float value and -1 in ids (`argmax`, `top_k`'s `ids`); `top_k` gives no token text there. How a hole passes through metrics, comparisons, reductions and tables is [errors/alignment-missing.md](../errors/alignment-missing.md#how-it-propagates).

## Broadcasting

One rule says what an operand may be ([FORMAT.md#ops](../../FORMAT.md#ops)): it may leave out any axis of the value it meets, and broadcasts over it. Every axis it has is that value's, in its order, matched by label. So a number, or a `mean` over every row and position, is the same everywhere; one vector broadcasts over rows and positions; a per-row number broadcasts over the positions and features. An axis the operand does have must agree:

- `layers` by model layer, as above;
- `rows` by dataset;
- a selected axis by its kept indices;
- `pos` by the read whose positions both hold, else by width.

So a metric that compares two distributions position by position refuses a pair read at different widths: `'clean.logits' holds 1 position(s) per row and the value 'kl' scores holds 3; …`. A write's operand meets the write's own picks, rows and positions as [usage/interventions.md](../usage/interventions.md) describes.

## Dtypes

A value's dtype is recorded beside its layout (`results.<name>.dtype` in the manifest). A read is the model's own tensor, so it keeps the model's dtype. What each op returns, and what that means for a bfloat16 model's numbers, is the float32 rule in [usage/ops.md](../usage/ops.md#the-float32-rule).

## Why batch shape moves the last bits

A row's numbers depend on the shape of the batch it runs in, so compare values only at one batch size, padding and thread count ([usage/running.md](../usage/running.md#when-two-runs-give-the-same-numbers)).
