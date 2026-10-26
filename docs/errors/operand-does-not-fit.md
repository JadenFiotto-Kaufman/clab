---
title: An Operand Does Not Fit
one_liner: "Why a swap, add or metric refuses the value it is given: an operand meets the value position for position, head for head and layer for layer, and each message says which of those failed and how to fix it."
tags: [errors, operands, interventions, layout, refusals]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/concepts/values-and-layout.md, docs/usage/positions.md, docs/errors/index.md, docs/errors/symptom-index.md]
sources: [clab/compile/interventions.py, clab/compile/arguments.py]
---

# An Operand Does Not Fit

An operand is a named value an op takes beside the value it acts on: `swap`'s and `add`'s `with`, `kl`'s `against`, `cosine`'s `with`. clab refuses an operand that cannot meet that value element for element: at compile where the document decides it (`operands` in `clab/compile/interventions.py` and `paired` in `clab/compile/arguments.py`), and at preflight where the model's sizes and the tokenized rows do (`check_operands`, `check_labels` and `check_layers` in `clab/compile/interventions.py`).

## What an operand may be

Inside a write (the ops up to the entry's last edit), and for a metric, an operand is one of:

- a number, or a named value of one number, which broadcasts everywhere;
- one vector of the value's features (`[hidden_size]`, `[num_heads, head_dim]`), the same on every row and position, as a `mean` step over a read gives;
- the value's own layout: rows of the same dataset, the same static number of positions, the same `select` indices in the same order, in the same coordinates (inside the same `encode`s).

A `layers` axis aligns by label: at layer L the op takes the operand's slice labelled L, in whatever order either lists its layers, and an operand without a `layers` axis is the same at every layer. [interventions.md](../usage/interventions.md) covers operands in use and [values-and-layout.md](../concepts/values-and-layout.md) covers layouts.

`documents/patching.json` reads one position and writes one position on the same dataset's rows; `documents/mean_ablation.json` swaps in one vector, a `mean` step over a read:

```json
{"ablate": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]}}
```

## `… so an operand is …`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' is laid out ['rows', 'pos', 'hidden_size']; this write's value is ['rows', 'pos', 'hidden_size'] here, so an operand is ['hidden_size']
```

The list after "so an operand is" is what this write accepts. When it is the features alone, as here, although the two layouts match, the write is ragged: it holds a different number of positions on each row, so a per-position value from another forward has nothing to pair with. A scope with the default index `":"` is ragged (`{"scope": {"text": " is"}}` covers however many tokens the text has on each row), and so are an open slice (`"2:"`) and a frame of every decode call. Give the write and the read a static width (`"index": -1`, or `"index": "-3:"` beside the scope), or reduce the operand to one vector first. The same message names a metric's operand of another layout (`'iia' is laid out ['rows', 'pos']; 'kl' scores a value laid out ['rows', 'pos', 'vocab_size'] …`).

## `… holds a different number of positions on each row`

```text
steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' holds a different number of positions on each row, which pair with no static width's; read it at a static index (pos -1, {'index': '-3:'}), or reduce it to one vector
```

The operand is the ragged one. The fix is the same; for a metric both sides need a static width. [positions.md](../usage/positions.md) says which `pos` forms are static.

## `… token(s) wide, narrowed to …`

```text
patched.patch: operand 'counterfactual.v_cf' is 3 token(s) wide, narrowed to {}; this write's selection is 1 wide, narrowed to {}
```

Preflight compares the number of positions and the size of every `select`. Here the read was `{"index": "-3:"}` and the write `pos: -1`. A one-head read written into a two-head selection fails the same way (`narrowed to {'num_heads': 1}` against `{'num_heads': 2}`): a vector broadcasts over rows and positions, never over heads. Between two reads a metric step says it at compile:

```text
steps.k.against: 'counterfactual.logits' holds 2 position(s) per row and the value 'kl' scores holds 1; position i scores against position i, so both are read at as many (a pos axis does not broadcast)
```

## `a select list is positional`

```text
patched.patch: operand 'counterfactual.v_cf' holds num_heads [1, 3], and this write's selection holds [3, 1]; a select list is positional, so they would trade places: select num_heads [3, 1] at the read
```

A `select` list keeps the indices in the order written, so the operand's first head meets the write's first head. Write one list at both sites; a named site that both entries use guarantees it. With other indices the message ends `an operand holds the same ones`, and for an op step over two reads `select one num_heads list at both reads`. [sites.md](../usage/sites.md) covers `select`.

## Layers

```text
patched.patch: operand 'counterfactual.v_cf' holds layers [1], and this op acts at layers [0, 1]; at each layer it takes the operand's slice of that layer, and it holds none of [0]: read it at those layers too
k: operand 'lens.l' has a layers axis, and this op acts at no layer (on a model's own value, or one reduced over its layers); read the operand at one layer, or reduce it over layers
```

The operand needs a slice for every layer the op acts at. A value at no layer (`lm_head_output`, `logits`, or a reduction over `layers`) takes no operand with a `layers` axis; inside an entry compile says so: `… an op takes an operand's slice of each layer it acts at, and lm_head_output is the model's own value, at no layer …`.

## Coordinates

```text
steps.patched.interventions.patch.ops[1].with: 'counterfactual.v_cf' is in the site's own layout; this bracket is in the coordinates of 'rot', so an operand is encoded as it is
```

Between an `encode` and its `decode`, an operand is in the same coordinates: encode it at its read with the same featurizer and `select`, as `documents/das_apply.json` does with `v_cf`. [featurizers.md](../usage/featurizers.md) covers brackets.

## An operand from the same forward

```text
steps.patched.interventions.patch.ops[0].with: 'patched.logits' is this entry or one this forward makes after it; an operand names an earlier entry
steps.patched.interventions.patch.ops[0].with: 'patched.read' runs 'readout' once it is off the model, after this forward; an operand from the same forward is an entry with no op after its edits
patched.patch: operand 'patched.read' is read at layer_output layer 1, which the nnterp engine's forward reaches no earlier than this write's layer_output layer 0; an operand from the write's own forward is an earlier entry at this site, or at a site the forward reaches first
```

An entry may use an earlier entry of its own forward, live, as the forward reaches it. So that entry comes earlier in the step, does nothing after its last edit, and sits at this site or at one the model computes earlier. Inside a `generate` step an entry of the same step is not an operand at all (`… is an entry of this generation …`); read it in a `forward`.

## Two inputs of different lengths

```text
patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11, so position 0 pairs '▁is' with 'urs'; use a scope to name the token in each row
patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there, so their key axes do not line up
```

A read and a write at the same unscoped `pos` on two inputs pair position i with position i. On a row whose two prompts tokenize to different lengths, every pair must lie within the tokens both prompts begin with alike, counted from the start, or end with alike, counted from the end, the first token past either counted in (`clab/compile/interventions.py` `windowed`). The fix is a scope on each side that names the token in its own row, as `documents/entity_patch.json` does:

```json
{"v_cf": {"site": "target", "pos": {"scope": {"column": "pairs.counterfactual_inputs_variables[0].entity"}, "index": -1}}}
```

A write to an attention pattern or its scores needs the two inputs to tokenize to the same length on every row, since its `key` axis counts the row's tokens.

Rows of equal length are not checked: a window there pairs whatever tokens share an index. That case runs, and [symptom-index.md](symptom-index.md#a-window-between-rows-of-equal-length) shows it.
