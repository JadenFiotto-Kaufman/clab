---
title: An Operand Does Not Fit
one_liner: "Why a swap, add, kl or cosine refuses its operand: layout, width, positions, selected heads, layers, coordinates, dataset, a same-forward entry, or two inputs of different lengths, with the fix for each message."
tags: [errors, operands, interventions, layout, refusals]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/concepts/values-and-layout.md, docs/usage/positions.md, docs/errors/index.md, docs/errors/symptom-index.md]
sources: [nnspec/compile/arguments.py, nnspec/compile/interventions.py, nnspec/compile/schedule.py, nnspec/compile/preflight.py]
---

# An Operand Does Not Fit

An operand is a named value an op takes beside the value it acts on: `swap`'s and `add`'s `with`, `scale`'s `factor`, `kl`'s `against`, `cosine`'s `with`. An operand that cannot meet that value element for element is refused (`nnspec/compile/arguments.py` `mismatch`). Compile refuses it where the document decides it, and preflight where the model's order or the tokenized rows do.

## What an operand may be

One rule, stated in [FORMAT.md](../../FORMAT.md#ops): an operand may leave out any axis of the value it meets, and broadcasts over the axes it leaves out. Every axis it has is one of the value's, in the value's order. So a number fits anywhere, and a `mean` step's vector fits every row and position. A `pos` axis never broadcasts: position i meets position i. Each message below names what differs.

`documents/mean_ablation.json` swaps in one vector, a `mean` step over a read:

```json
{"ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]}}
```

## `… an operand's axes are among the value's …`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' is laid out ['rows', 'pos', 'vocab_size']; 'swap' acts on a value laid out ['rows', 'pos', 'hidden_size'], and an operand's axes are among the value's, in its order (those it leaves out it broadcasts over)
```

The operand has an axis the value does not: here it was read at the head and is written into the residual stream. Read it at a site of the value's layout, or remove the extra axis with an op first.

## `… has no width compile can fix`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' has no width compile can fix (an open index beyond the prefill, an `until` scope, several occurrences), and two values of different reads pair only at a fixed width, whether one forward or two made them; read both at a static index without a scope ('index': -1, or '-3:'), in the prefill at an open index or one occurrence of a scope, not until, whose rows give its width, or over decode calls at a frame of a static number of them with an index of static width ({'frame': '1:4', 'index': ':3'})
steps.patched.interventions.patch.ops[0].with: the value 'swap' acts on has no width compile can fix (an open index beyond the prefill, an `until` scope, several occurrences), …
```

In the first the operand's width is open, in the second the write's: `documents/patching.json` with `{"scope": {"column": "pairs.entity", "until": true}}` on each side. A side whose width no row can settle pairs with no value of another read, whatever the prompts: an `until` scope, several occurrences, an open index or frame beyond the prefill. [positions.md](../usage/positions.md#static-width-or-ragged) lists which `pos` forms pair. In the prefill an open index and one occurrence of a scope are the exceptions, since their rows say their width ([below](#-which-an-open-index-has-where-every-row-holds-as-many-positions)). A permuted read is held to the read it permutes. The fixes:

- read both at a static index without a scope (`"index": -1`, `"index": "-3:"`);
- read in the prefill at an open index, or at one occurrence of a scope there;
- reduce the operand to one vector first: a `mean` step's vector has no `pos` axis, so it broadcasts over a write of any width.

**Two entries of one forward are two reads.** An op step over them, `{"kind": "op", "op": "sub", "of": "f.a", "with": "f.b"}` with both read at an `until` scope, or any other `pos` this section lists, is refused with the first message, though both hold the same positions on every row. Both at `{"index": ":"}` pair where every row is one length and are refused in preflight on rows of different lengths ([below](#-which-an-open-index-has-where-every-row-holds-as-many-positions)). Read both at a static index, as above. A write's operand from its own forward is the exception: preflight compares it with the write row by row ([below](#an-operand-from-the-same-forward)).

A read at a decode call holds 0 or 1 positions per row, since a row may stop before that call. So it pairs with nothing: `kl` or `js` with it on either side is refused with the first message. Score a decode read on its own, with `argmax` or a metric against a one-token answer column.

## `… which a scoped read has where every row's scope holds the same number of tokens`

```text
patched.patch: operand 'counterfactual.v_cf' holds 1 position on rows 0-1, 3 and 3 positions on row 2; two values of different reads pair only at a fixed width, whether one forward or two made them, which a scoped read has where every row's scope holds the same number of tokens: pick one with 'index' (-1, or '-2:'), or reduce it
patched.patch: operand 'counterfactual.v_cf' meets the value 'swap' acts on, which holds 3 positions on row 0 and 1 position on rows 1-3; …
steps.patched.interventions.patch.ops[0].with: 'counterfactual' holds 1 position on rows 0-1, 3 and 3 positions on row 2; …
```

- **Preflight counts a scope's width.** A scope of one occurrence in the prefill, not `until`, holds on each row as many positions as its index picks among the scope's tokens. Only the tokenizer knows that number, so preflight counts it before any forward.
- **Equal counts pair.** Where every row holds as many, that is the read's width, and it pairs as a static width does. `documents/entity_patch_scope.json`, whose entities are one token each, gives the same numbers as the same reads at `"index": -1` ([running.md](../usage/running.md#when-two-runs-give-the-same-numbers)).
- **Which side.** The first message is the operand's scope (`documents/entity_patch.json` without its `index`, where ` Thursday` is three tokens on one row); the second is the write's own.
- **A loaded read is refused at compile.** Its rows' positions came with it from its run, so compile refuses it, with the document path and no `operand` (the third message).
- **The fix.** Pick the tokens with `index` (`-1`, the last; `"-2:"`, the last two, which a row with fewer refuses), or reduce the scope over `pos` first.

Where the rows agree on a width the other side does not have, the refusal is [`… holds N position(s) per row …`](#-holds-n-positions-per-row-and-the-value--holds-m).

## `… which an open index has where every row holds as many positions`

```text
patched.patch: operand 'counterfactual.v_cf' holds 9 positions on rows 0-1, 3 and 11 positions on row 2; two values of different reads pair only at a fixed width, whether one forward or two made them, which an open index has where every row holds as many positions: read a static span every row holds ('index': '0:9', or '-9:'), or reduce it
resampled.abl: operand 'donor' reorders the rows of 'harvest.acts', which holds 11 positions on row 0 and 9 positions on rows 1-3; …
steps.resampled.interventions.abl.ops[0].with: 'donor' reorders the rows of 'cached' (a load of 'harvest.acts'), which holds 11 positions on row 0 and 9 positions on rows 1-3; …
```

- **Preflight counts an open index's width.** In the prefill, `{"index": ":"}` (or `"2:"`, or no `pos`) holds on each row as many positions as the row's prompt gives it. Compile leaves the width to the rows, and preflight counts them before any forward.
- **Rows of one length pair.** There the count is the read's width, and it pairs as a static width does. `documents/ablation_pair.json` with `":"` in place of each `"0:12"` gives its own numbers, equal on its 12-token prompts ([running.md](../usage/running.md#when-two-runs-give-the-same-numbers)).
- **Rows of different lengths are refused**, as in the first message (`documents/patching.json` with `v_cf` at `":"`). Each count is named with its rows.
- **Two reads of one forward pair the same way.** A `kl` of two `":"` reads of one forward on such rows is refused alike: they are different reads.
- **A permuted read is its read's rows reordered**, so the refusal names the read it permutes (the second message), and a permuted load names the load step and the value it loads (the third, at compile, since a load carries its rows' positions).
- **The fix.** Read a static span every row holds, as long as the shortest row's count, that keeps the read's own start or end: `"0:9"` or `"-9:"` for `":"`, `"2:9"` or `"-7:"` for `"2:"`. Or reduce the read over `pos` first.

## `… has a pos axis, and this write lands beyond the prefill …`

```text
steps.swapped.interventions.w.ops[0].with: 'clean.r' has a pos axis, and this write lands beyond the prefill: at each decode call it names, as the call runs, on that call's one position, where an operand's positions are not split among the calls; …
```

A generation's write over decode calls lands at each call as it runs, on that call's one token. An operand's positions are not handed out one per call, so an earlier generation's read over `{"frame": "1:4", "index": ":3"}` cannot be swapped in over the same frame.

- **An operand without a `pos` axis** (a `mean` step's vector) broadcasts over every call; reduce a per-position read over `pos` first.
- **A per-position operand** takes one write per call, each entry at one call (`{"frame": 3, "index": 0}`) with an operand of one position.
- **Read that operand where every row has it:** in the prefill at a static index, or at a decode call that `min_new_tokens` makes every row reach. An operand read at a decode call is missing on every row whose generation stopped before that call, and a write that lands on such a row is refused ([alignment-missing.md](alignment-missing.md#how-it-propagates)).

## `… holds N position(s) per row and the value … holds M`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds 10 position(s) per row and the value 'swap' acts on holds 1; position i meets position i, so both are read at as many (a pos axis does not broadcast)
steps.k.against: 'counterfactual.logits' holds 2 position(s) per row and the value 'kl' scores holds 1; position i scores against position i, so both are read at as many (a pos axis does not broadcast)
```

Both sides have a width, and the widths differ: a read at `{"index": "0:10"}` against a write at `-1`. Read both at as many positions. Compile refuses two static widths. Where one is a whole scope's, preflight refuses once the rows have confirmed it, prefixed with the entry (`patched.patch: operand 'counterfactual.v_cf' holds 1 position(s) per row and the value 'swap' acts on holds 3; …`).

## `a select list is positional`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds num_heads [1, 3], and this write's selection holds [3, 1]; a select list is positional, so they would trade places: select num_heads [3, 1] at the read
```

A `select` list keeps the indices in the order written, so the operand's first head meets the write's first head. Write one list at both sites; a named site that both entries use guarantees it. The same message names the other mismatches of a selected axis:

- other indices: `… holds num_heads [0, 2], and this write's selection holds [1, 3]; an operand holds the same ones: select num_heads [1, 3] at the read`;
- one head against several: `… holds num_heads [1], and this write's selection holds [1, 3]; an operand broadcasts over rows and positions only: …`, since one head does not stand for several;
- every head against some: `… holds every num_heads, and this write's selection holds [3, 1]; an operand holds the same ones: …`. Read the operand at the same named site as the write, whose `num_heads` sweep then zips with the write's, one head per point, as `documents/head_patching.json` does.

For an op step over two reads the message ends `select one num_heads list at both reads`. [sites.md](../usage/sites.md#select) covers `select`.

## Layers

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds layers [1], and this op acts at layers [0, 1]; at each layer it takes the operand's slice of that layer, and it holds none of [0]: read it at those layers too
steps.k.against: 'lens.logits' has a layers axis, and this op acts at no layer (on a model's own value, or one reduced over its layers); read the operand at one layer, or reduce it over layers
```

The operand needs a slice for every layer the op acts at; [interventions.md](../usage/interventions.md#writes-across-layers) says how layers pair. A value at no layer (`lm_head_output`, `logits`, or a reduction over `layers`) takes no operand with a `layers` axis, inside an entry as in an op step.

## Coordinates

```text
steps.patched.interventions.patch.ops[1].with: 'counterfactual.v_cf' is in the site's own layout; this bracket is in the coordinates of 'rot', so an operand is encoded as it is
```

Inside an `encode` bracket the operand must be encoded too: encode it at its read with the same featurizer and `select`, as `documents/das_apply.json` does with `v_cf` ([featurizers.md](../usage/featurizers.md#brackets)).

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
patched.w: operand 'patched.r' holds no position on row 1 (alignment_missing: its pos picks none of the row's tokens) and this write lands at [8] there; a write takes an operand read on every row it lands on
patched.w: operand 'patched.r' holds 2 position(s) on row 0 and this write 1; position i meets position i, so an entry of the write's own forward without one width on every row is read at as many on each
```

An entry may use an earlier entry of its own forward, live, as the forward reaches it. That operand:

- comes earlier in the step;
- does nothing after its last edit;
- sits at the write's site, or at one the model computes earlier, which preflight checks in the engine's order (the third message);
- where either side's width is open, holds a value on every row the write lands on (the fourth message: a read at `{"index": "10:"}` that a short row does not reach);
- where either side's width is open, holds as many positions as the write on each row (the fifth: a read at `"9:"` against a write at `-1`).

Inside a `generate` step an entry of the same step is no operand (`… is an entry of this generation; an operand from an entry of its own step is served in a forward, not inside a generate`). Read it in a `forward`.

## Two inputs of different lengths

```text
patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11, so pairs.counterfactual_inputs[0]'s position 5 (',') pairs with pairs.input's position 5 ('urs'), and a position pairs across prompts of different lengths only where every token between it and the end it is counted from is the same on both; use a scope to name the token in each row
patched.look_there at num_heads=0: operand 'counterfactual.their_pattern' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there, so their key axes do not line up
```

A read and a write at the same unscoped `pos` on two inputs pair position i with position i. The first message is `pos: 5` on both; it names each input's position, counted along its row, and the token there. On a row whose two prompts tokenize to different lengths, a position pairs across prompts of different lengths only where every token between it and the end it is counted from is the same on both, whatever the two tokens it pairs (`nnspec/compile/interventions.py` `check_window`). On `weekdays/train`'s first row, `-5` pairs `day` with `▁Saturday` and runs, and `5` and `-6` are refused ([positions.md](../usage/positions.md#static-width-or-ragged)). The fix is a scope on each side that names the token in its own row, as `documents/entity_patch.json` does:

```json
{"v_cf": {"site": "target", "pos": {"scope": {"column": "pairs.counterfactual_inputs_variables[0].entity"}, "index": -1}}}
```

A write to an attention pattern or its scores needs the two inputs to tokenize to the same length on every row, since its `key` axis counts the row's tokens.

Rows of equal length are not checked: a window there pairs whatever tokens share an index, and runs ([symptom-index.md](symptom-index.md#a-window-between-rows-of-equal-length)).
