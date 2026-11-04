---
title: An Operand Does Not Fit
one_liner: "Why a swap, add or metric refuses the value it is given: an operand's axes are among the value's, matched by label position for position, head for head and layer for layer, and each message says which of those failed and how to fix it."
tags: [errors, operands, interventions, layout, refusals]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/concepts/values-and-layout.md, docs/usage/positions.md, docs/errors/index.md, docs/errors/symptom-index.md]
sources: [clab/compile/arguments.py, clab/compile/interventions.py, clab/compile/schedule.py]
---

# An Operand Does Not Fit

An operand is a named value an op takes beside the value it acts on: `swap`'s and `add`'s `with`, `scale`'s `factor`, `kl`'s `against`, `cosine`'s `with`. clab refuses an operand that cannot meet that value element for element: at compile where the document decides it (`clab/compile/arguments.py` `mismatch`, and for an operand from the write's own forward `clab/compile/interventions.py` `compile_entry`), and at preflight where the model's order and the tokenized rows do (`unreached_site_refusal` in `clab/compile/schedule.py`; `check_operands` and `check_window` in `clab/compile/interventions.py`; and `mismatch` again once the rows have said each scope's width, from `clab/compile/preflight.py` `prepare`).

## What an operand may be

One rule, stated in [FORMAT.md](../../FORMAT.md#ops): an operand may leave out any axis of the value it meets, and broadcasts over the axes it leaves out; every axis it has is one of the value's, in the value's order, with the same label. So a number fits anywhere, a `mean` step's one vector fits every row and position, and a value of the same layout fits element for element. Each message below names the label that differs.

A `layers` axis aligns by label: at layer L the op takes the operand's slice labelled L, in whatever order either lists its layers, and an operand without a `layers` axis is the same at every layer. [interventions.md](../usage/interventions.md) covers operands in use and [values-and-layout.md](../concepts/values-and-layout.md) covers layouts.

`documents/patching.json` reads one position and writes one position on the same dataset's rows; `documents/mean_ablation.json` swaps in one vector, a `mean` step over a read:

```json
{"ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]}}
```

## `… an operand's axes are among the value's …`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' is laid out ['rows', 'pos', 'vocab_size']; 'swap' acts on a value laid out ['rows', 'pos', 'hidden_size'], and an operand's axes are among the value's, in its order (those it leaves out it broadcasts over)
```

The operand has an axis the value does not: here it was read at the head and is written into the residual stream. Read it at a site of the value's layout, or remove the extra axis with an op first.

## `… holds a different number of positions on each row`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds a different number of positions on each row, which pair with no other value's; read both at a static index without a scope ('index': -1, or '-3:'), at one occurrence of a scope in the prefill, not until, whose rows give its width, or over decode calls at a frame of a static number of them with an index of static width ({'frame': '1:4', 'index': ':3'})
steps.patched.interventions.patch.ops[0].with: the value 'swap' acts on holds a different number of positions on each row, …
```

In the first the operand is ragged, in the second the write is: a per-position value from another forward has nothing to pair with on a row whose count differs. Compile refuses the pair wherever a side's `pos` leaves its width open, whatever the prompts hold: here the read was `{"index": ":"}`, and so are an open slice (`"2:"`) without a scope, several occurrences of a scope, an `until` scope, a scope beyond the prefill, a frame of every decode call and a decode call read with an open index (`{"frame": 3}`). An index does not help a scope of those kinds, which holds a different number of positions per row whatever its index. One occurrence of a scope in the prefill, not `until`, is not refused here, since its rows say how many positions it holds ([the next section](#-a-scoped-read-pairs-with-another-forwards-value-when-every-rows-scope-holds-the-same-number-of-tokens)). Read both at a static index without a scope (`"index": -1`, `"index": "-3:"`), at one occurrence of a scope in the prefill, or reduce the operand to one vector first: a `mean` step's vector, which has no `pos` axis, broadcasts over a ragged write. [positions.md](../usage/positions.md#static-width-or-ragged) says which `pos` forms are static.

A read at a decode call holds 0 or 1 positions per row, since a row may stop before that call, so it has no static width: `kl` or `js` with it on either side, against another decode read or a prefill read, is refused with the first message. Score a decode read on its own instead, with `argmax` or a metric against a one-token answer column.

## `… a scoped read pairs with another forward's value when every row's scope holds the same number of tokens`

```text
patched.patch: operand 'counterfactual.v_cf' holds 1 position on rows 0-1, 3 and 3 positions on row 2; a scoped read pairs with another forward's value when every row's scope holds the same number of tokens: pick one with 'index' (-1, or '-2:'), or reduce it
patched.patch: operand 'counterfactual.v_cf' meets the value 'swap' acts on, which holds 3 positions on row 0 and 1 position on rows 1-3; …
steps.patched.interventions.patch.ops[0].with: 'counterfactual' holds 1 position on rows 0-1, 3 and 3 positions on row 2; …
```

A scope of one occurrence in the prefill, not `until`, with no `index` or one of no fixed number, holds on each row as many positions as its index picks among the row's tokens of the text, which only the tokenizer knows, so preflight counts them before any forward. Where every row holds as many, that is the read's width and it pairs as a static one does: `documents/entity_patch_scope.json`, whose entities are each one token, gives what the same reads with `"index": -1` give, to the bit. In the first message, `documents/entity_patch.json` without its `index` on `weekdays/train`, ` Thursday` is three tokens on row 2 of the counterfactual and every other entity one. In the second the write's own scope is the one whose rows differ, its operand read at `-1`. The third is a loaded read (a `load` step's): its rows' positions came with it from its run, so compile refuses it already, with the document path and no `operand`. Pick the tokens with `index` (`-1`, the last; `"-2:"`, the last two, which a row with fewer refuses), or reduce the scope over `pos` first. Where the rows agree on a width the other side does not have, the refusal is the next section's.

## `… has a pos axis, and this write lands beyond the prefill …`

```text
steps.swapped.interventions.w.ops[0].with: 'clean.r' has a pos axis, and this write lands beyond the prefill: at each decode call it names, as the call runs, on that call's one position, where an operand's positions are not split among the calls; write at one decode call, {'frame': 3, 'index': 0}, with the operand read at that call the same way, an entry per call
```

A generation's write over decode calls lands at each call as it runs, on that call's one token, and an operand's positions are not handed out one per call: an earlier generation's read over `{"frame": "1:4", "index": ":3"}` cannot be swapped in over the same frame. Write at one call and one position, `{"frame": 3, "index": 0}`, beside a read at that call written the same way, one entry for each call to be written. An operand with no `pos` axis (a `mean` step's vector) broadcasts over every call.

## `… holds N position(s) per row and the value … holds M`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds 10 position(s) per row and the value 'swap' acts on holds 1; position i meets position i, so both are read at as many (a pos axis does not broadcast)
steps.k.against: 'counterfactual.logits' holds 2 position(s) per row and the value 'kl' scores holds 1; position i scores against position i, so both are read at as many (a pos axis does not broadcast)
```

Both have a width, and the widths differ: here the read was `{"index": "0:10"}` (or `"-2:"`) and the write or the scored read `pos: -1`. Read both at as many positions. Where both widths are static the plan's compile refuses it; where one is a whole scope's, preflight does once the rows have confirmed it, prefixed with the entry (`patched.patch: operand 'counterfactual.v_cf' holds 1 position(s) per row and the value 'swap' acts on holds 3; …`).

## `a select list is positional`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds num_heads [1, 3], and this write's selection holds [3, 1]; a select list is positional, so they would trade places: select num_heads [3, 1] at the read
```

A `select` list keeps the indices in the order written, so the operand's first head meets the write's first head. Write one list at both sites; a named site that both entries use guarantees it. The same message names the other mismatches of a selected axis:

- other indices: `… holds num_heads [0, 2], and this write's selection holds [1, 3]; an operand holds the same ones: select num_heads [1, 3] at the read`;
- one head against several: `… holds num_heads [1], and this write's selection holds [1, 3]; an operand broadcasts over rows and positions only: …`, since one head does not stand for several;
- every head against some: `… holds every num_heads, and this write's selection holds [3, 1]; an operand holds the same ones: …`. Read the operand at the same named site as the write, whose `num_heads` sweep then zips with the write's, one head per point, as `documents/head_patching.json` does.

For an op step over two reads the message ends `select one num_heads list at both reads`. [sites.md](../usage/sites.md) covers `select`.

## Layers

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds layers [1], and this op acts at layers [0, 1]; at each layer it takes the operand's slice of that layer, and it holds none of [0]: read it at those layers too
steps.k.against: 'lens.logits' has a layers axis, and this op acts at no layer (on a model's own value, or one reduced over its layers); read the operand at one layer, or reduce it over layers
```

The operand needs a slice for every layer the op acts at. A value at no layer (`lm_head_output`, `logits`, or a reduction over `layers`) takes no operand with a `layers` axis, inside an entry as in an op step.

## Coordinates

```text
steps.patched.interventions.patch.ops[1].with: 'counterfactual.v_cf' is in the site's own layout; this bracket is in the coordinates of 'rot', so an operand is encoded as it is
```

Between an `encode` and its `decode`, an operand is in the same coordinates: encode it at its read with the same featurizer and `select`, as `documents/das_apply.json` does with `v_cf`. [featurizers.md](../usage/featurizers.md) covers brackets.

## Rows of another dataset

```text
steps.patched.interventions.patch.ops[0].with: 'pairs.input' has a row per example of 'pairs' and 'counterfactual.v_cf' of 'other'; an op takes one dataset's rows
```

An op takes one dataset's rows, its operands' and columns' alike. Two datasets that read the same file are two; read both inputs from one dataset.

## An operand from the same forward

```text
steps.patched.interventions.patch.ops[0].with: 'patched.logits' is this entry or one this forward makes after it; an operand names an earlier entry
steps.patched.interventions.patch.ops[0].with: 'patched.read' runs 'readout' once it is off the model, after this forward; an operand from the same forward is an entry with no op after its edits
patched.patch: operand 'patched.read' is read at layer_output layer 1, which the nnterp engine's forward reaches no earlier than this write's layer_output layer 0; an operand from the write's own forward is an earlier entry at this site, or at a site the forward reaches first
```

An entry may use an earlier entry of its own forward, live, as the forward reaches it. So that entry comes earlier in the step, does nothing after its last edit, and sits at this site or at one the model computes earlier. The third message comes from preflight, which writes the forward's schedule in the engine's order. Inside a `generate` step an entry of the same step is not an operand at all (`… is an entry of this generation; an operand from an entry of its own step is served in a forward, not inside a generate`); read it in a `forward`.

## Two inputs of different lengths

```text
patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11, so position 0 pairs ',' with 'urs'; use a scope to name the token in each row
patched.look_there: operand 'counterfactual.their_pattern' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there, so their key axes do not line up
```

A read and a write at the same unscoped `pos` on two inputs pair position i with position i (the first message is `pos: 5` on both). On a row whose two prompts tokenize to different lengths, every pair must lie within the tokens both prompts begin with alike, counted from the start, or end with alike, counted from the end, the first token past either counted in (`clab/compile/interventions.py` `check_window`). The fix is a scope on each side that names the token in its own row, as `documents/entity_patch.json` does:

```json
{"v_cf": {"site": "target", "pos": {"scope": {"column": "pairs.counterfactual_inputs_variables[0].entity"}, "index": -1}}}
```

A write to an attention pattern or its scores needs the two inputs to tokenize to the same length on every row, since its `key` axis counts the row's tokens.

Rows of equal length are not checked: a window there pairs whatever tokens share an index. That case runs, and [symptom-index.md](symptom-index.md#a-window-between-rows-of-equal-length) shows it.
