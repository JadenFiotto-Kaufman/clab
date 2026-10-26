---
title: Sites
one_liner: "Choosing where to read or write: the component table, layers as an int or a selector, select to narrow heads, neurons or keys, the layout a site's value has, and when attention needs eager."
tags: [usage, sites, components, layers, select, heads, attention]
related: [docs/usage/positions.md, docs/concepts/values-and-layout.md, docs/usage/interventions.md, docs/usage/engines.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/attention.md, docs/patterns/logit-lens.md]
sources: [clab/format/address.py, clab/format/selectors.py, clab/plan.py, clab/compile/interventions.py]
---

# Sites

A site is where an entry reads or writes: a `component` of the model, the `layers` it is taken at, and an optional `select` that keeps part of one of its axes. Which tokens is the entry's `pos`, not the site's ([positions.md](positions.md)). The fields are in [FORMAT.md#sitesname-or-a-site-inline](../../FORMAT.md#sitesname-or-a-site-inline) and every component's layout in [FORMAT.md#components](../../FORMAT.md#components).

## Canonical example

From `documents/head_patching.json`: a site declared once under `sites` and named by two entries, and the head written inline where only one entry uses it.

```json
"sites": {
  "target": {"component": "attention_premix", "layers": 0, "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}},
  "head": {"component": "lm_head_output"}
}
```

`target` is layer 0's per-head attention result, one head at a time across four sweep points; `head` is the model's own head, which takes no `layers`. An entry names a site (`"site": "target"`) or writes one in place (`"site": {"component": "lm_head_output"}`).

## Choosing a component

| to work with | component | notes |
|---|---|---|
| the residual stream between blocks | `layer_output` | layer i's output is layer i+1's `layer_input`; layer 0's `layer_input` is `token_embeddings` |
| the stream between attention and MLP | `layer_mid` | `layer_input` + `attention_output` |
| what a sublayer adds to the stream | `attention_output`, `mlp_output` | `layer_mid` + `mlp_output` is `layer_output` |
| one head's result | `attention_premix` + `select: {"num_heads": ...}` | no eager needed; [head-and-neuron-patching](../patterns/head-and-neuron-patching.md) |
| an attention pattern, or the scores before softmax | `attention_probabilities`, `attention_scores` | eager; [attention](../patterns/attention.md) |
| queries, keys, values | `attention_queries`, `attention_keys`, `attention_values` | eager; keys and values per KV head, before `repeat_kv` |
| MLP neurons | `mlp_activation` (the activation of the gate), `mlp_neurons` (what the down projection sums) | different on a gated MLP |
| the next-token distribution | `logits` (after any softcap), `lm_head_output` (the head alone) | equal on a model without a softcap |
| a logit lens | a stream component with `readout` | [logit-lens](../patterns/logit-lens.md) |

The identities in the notes hold to the bit on the corpus Llama, a pre-norm block; check them on your architecture before relying on one.

## `layers`

A block's component needs `layers`; the model's own values (`token_embeddings`, `final_norm_output`, `lm_head_output`, `logits`) refuse it. `layers` is a selector over the model's layers ([FORMAT.md#pos](../../FORMAT.md#pos) has the grammar):

- **An int** is that layer, negative counting from the last: `-1` is the last block. The value has no layers axis.
- **A slice or a list** reads every layer it picks in one forward and stacks them, a `layers` axis first, labelled with the layers picked, in that order: `":"` is every layer, `"-2:"` the last two, `[1, 0]` layer 1 then layer 0. `clab table` prints the model's layer number in its `layer` column.
- **A sweep** (`{"sweep": [0, 1]}`) is one point, and one forward, per layer. Use it when each layer is a different intervention; to read many layers, a selector gives the same numbers in one forward ([sweeps.md](sweeps.md#widen-the-read-instead)).

`0` and `[0]` read the same numbers, but `[0]` keeps a `layers` axis of size 1: an extra leading axis in the saved tensor and a `layer` column in the table. As an operand either meets a write at layer 0 alike, since operands align by layer label. A layer the model does not have is refused before any forward: `clab: f.x: layer 5; the model has 2 layers`. A write across several layers runs its ops at each, as [interventions.md](interventions.md) describes.

## `select`

`select` narrows one or more of the component's feature axes, `{<axis>: <selector>}`: `num_heads` or `num_kv_heads` for heads, `head_dim`, `intermediate_size` for neurons, `hidden_size`, `vocab_size`, or `key` on an attention pattern. The component table lists which axes each component has; naming another is refused with the list (`attention_premix has no axis 'seq' to select; its axes: num_heads, head_dim`).

- **The axis stays.** `{"num_heads": 2}` is a `num_heads` axis of size 1, the same value as `[2]`, so a site has one layout whatever it selects. On `attention_premix` at `pos: -1` that is `[rows, 1, 1, head_dim]`.
- **A list keeps its order, and the order is positional.** `{"num_heads": [3, 1]}` puts head 3 at index 0. An operand must carry the same picks in the same order as the write it meets, or it is refused; an `argmax` or `top_k` over the selected axis returns indices into the selection, not head numbers. Build every site of a head set from one sorted list. See [values-and-layout.md](../concepts/values-and-layout.md#select-keeps-the-axis).
- **A write selects on one axis.** A read may select heads and dimensions together; a write is refused with `an entry that writes selects on one axis; this one selects num_heads, head_dim`.
- **`key` counts from the start.** A key selector picks the same keys of every row (`0` for BOS, `[0, 2]`, `"0:3"`); `-1` is refused, since rows differ in length. The `key` axis is each row's own tokens, as positions count them.
- **`readout` needs the whole stream.** Selecting `hidden_size` and then `readout` is refused, since the final norm needs every dimension; read the site whole, `readout`, then narrow the logits. Which components `readout` takes is in [ops.md](ops.md#transforms).
- **A metric needs the whole vocabulary.** `match`, `logit_diff` and `token_prob` look an answer up by token id, so a head narrowed on `vocab_size` gives them the wrong axis; score the full head ([ops.md](ops.md#metrics)).

A pick outside the axis is refused: `select num_heads [4] outside the model's 4`. A `select` may be swept, as in the canonical example.

## Attention needs eager

Six components are read inside the attention's forward: `attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities` and `attention_head_outputs`. Fused attention (`sdpa`, which the corpus model loads by default) computes them inside one call, and both engines refuse them before any forward with the same reason:

```text
clab: f.x: the nnterp engine does not serve attention_probabilities: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Load the model with eager attention, in the `model` block, as `documents/attention_knockout.json` does:

```json
"model": {"key": "...", "revision": "...", "dtype": "float32", "attn_implementation": "eager"}
```

`attention_premix`, the output projection's input, needs no eager; on the corpus Llama it equals `attention_head_outputs` to the bit. Under eager the hooks engine still refuses these six, since it reaches module boundaries only; `attention_premix` and the MLP's interior it serves ([engines.md](engines.md#cross-checking-with-hooks)).

## The layout of a site's value

A read's value is laid out `(rows, pos, *the component's other axes)`, with a `layers` axis first when `layers` is a slice or a list: `attention_premix` at layer 0 is `rows, pos, num_heads, head_dim`; `attention_probabilities` is `rows, pos, num_heads, key`, its `pos` being the query positions. `clab show` prints each saved value's layout. Why the axes are in that order and how ops align them is [values-and-layout.md](../concepts/values-and-layout.md).
