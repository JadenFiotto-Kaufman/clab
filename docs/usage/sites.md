---
title: Sites
one_liner: "Where do I read or write? Choosing a component and layers, and select over heads, neurons, experts or keys, from the start or relative to each query."
tags: [usage, sites, components, layers, select, heads, attention, mixture-of-experts]
related: [docs/usage/positions.md, docs/concepts/values-and-layout.md, docs/usage/interventions.md, docs/usage/engines.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/attention.md, docs/patterns/logit-lens.md, docs/patterns/expert-ablation.md]
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
| the normed stream a sublayer is called with | `attention_norm_output`, `mlp_norm_output` | the input of a probe or an SAE trained on normed activations; not a stream component, so `readout` refuses them |
| what the MLP is called with | `mlp_input` | the same tensor as `mlp_norm_output` on the corpus Llama |
| what the head projects | `final_norm_output` | the final norm of the last `layer_output`; `lm_head_output` is the head applied to it |
| one head's result | `attention_premix` + `select: {"num_heads": ...}` | no eager needed; [head-and-neuron-patching](../patterns/head-and-neuron-patching.md) |
| an attention pattern, or the scores before softmax | `attention_probabilities`, `attention_scores` | eager; [attention](../patterns/attention.md) |
| queries, keys, values | `attention_queries`, `attention_keys`, `attention_values` | eager; keys and values per KV head, before `repeat_kv` |
| MLP neurons | `mlp_activation` (the activation of the gate), `mlp_neurons` (what the down projection sums) | different on a gated MLP |
| a mixture of experts' routing | `router_logits`, `expert_weights`, `expert_indices` | [below](#a-mixture-of-experts) |
| what its experts add | `expert_outputs`, `routed_output`, `shared_expert_output` | [below](#a-mixture-of-experts) |
| the next-token distribution | `logits` (after any softcap), `lm_head_output` (the head alone) | equal on a model without a softcap |
| a logit lens | a stream component with `readout` | [logit-lens](../patterns/logit-lens.md) |

The identities in the notes hold on the corpus checkpoints: the stream's and the norms' on the tiny Llama, the mixture's on the tiny Mixtral of `documents/expert_knockout.json`. Check them on your architecture before relying on one.

### A mixture of experts

At a block whose MLP is a mixture of experts, these components are its values, by nnterp's names and layouts:

| component | what it holds | layout |
|---|---|---|
| `router_logits` | the router's logits, one per expert, before its scoring | `rows, pos, num_experts` |
| `expert_weights` | the weight each of a token's `top_k` routing slots scales its expert's output by | `rows, pos, top_k` |
| `expert_indices` | the expert each slot sends the token to, int64 | `rows, pos, top_k` |
| `expert_outputs` | each slot's expert output times its weight | `rows, pos, top_k, hidden_size` |
| `routed_output` | the slots' sum, without the shared expert | `rows, pos, hidden_size` |
| `shared_expert_output` | the shared expert's output, where the mixture has one | `rows, pos, hidden_size` |

- **The slots sum to `routed_output`.** On the tiny Mixtral, which has no shared expert, `routed_output` is `mlp_output`. With a shared expert, `routed_output` + `shared_expert_output` is `mlp_output` on most families.
- **`select` counts experts and slots** along `num_experts` and `top_k`: `{"num_experts": 3}` is expert 3's logit. The model's sizes are read off the first block whose MLP is a mixture.
- **`mlp_output` is still what the block adds.** A dense MLP's `mlp_activation` and `mlp_neurons` are refused at a mixture block, and a mixture's values at a dense block or on a model without one ([errors](../errors/index.md)).
- **One forward reads any of a mixture's values**, in forward order.
- **Which values a family serves** is nnterp's `support()`, per block, and so is what its router logits mean (a softmax router, a sigmoid one, a top-k then softmax). The hooks engine serves fewer ([engines](engines.md#a-mixture-of-experts-on-hooks)).

A write to `expert_indices` takes a `swap` of ids only; knockouts, forced routes, usage and routing entropy are [expert ablation](../patterns/expert-ablation.md), and how the engine instruments a mixture is [developing/engines.md](../developing/engines.md).

## `layers`

A block's component needs `layers`; the model's own values (`token_embeddings`, `final_norm_output`, `lm_head_output`, `logits`) refuse it. `layers` is a selector over the model's layers ([FORMAT.md#pos](../../FORMAT.md#pos) has the grammar):

- **An int** is that layer, negative counting from the last: `-1` is the last block. The value has no layers axis.
- **A slice or a list** reads every layer it picks in one forward and stacks them, a `layers` axis first, labelled with the layers picked, in that order. `":"` is every layer, `"-2:"` the last two, `[1, 0]` layer 1 then layer 0. `clab table` prints the model's layer number in its `layer` column.
- **A sweep** (`{"sweep": [0, 1]}`) is one point per layer. A write's points run one forward each; a read's share one forward, as a selector does, and give the same numbers one point per layer. Sweep when each layer is a different intervention; to reduce or compare across layers, read them with a selector ([sweeps.md](sweeps.md#widen-the-read-instead)).

`0` and `[0]` read the same numbers, but `[0]` keeps a `layers` axis of size 1: an extra leading axis in the saved tensor and a `layer` column in the table. As an operand either meets a write at layer 0 alike, since operands align by layer label. A layer the model does not have is refused before any forward: `clab: f.x: layer 5; the model has 2 layers`. A write across several layers runs its ops at each, as [interventions.md](interventions.md) describes.

## `select`

`select` narrows one or more of the component's feature axes, `{<axis>: <selector>}`. The axes are `num_heads` or `num_kv_heads` for heads, `head_dim` or `qk_head_dim` for a head's dimensions, `intermediate_size` for neurons, `hidden_size`, `vocab_size`, `num_experts` or `top_k` on a mixture's values, or `key` on an attention pattern. `clab describe components` lists which axes each component has; naming another is refused with the list (`attention_premix has no axis 'seq' to select; its axes: num_heads, head_dim`).

- **The axis stays.** `{"num_heads": 2}` is a `num_heads` axis of size 1, the same value as `[2]`. On `attention_premix` at `pos: -1` that is `[rows, 1, 1, head_dim]` ([why](../concepts/values-and-layout.md#select-keeps-the-axis)).
- **A list keeps its order, and the order is positional.** `{"num_heads": [3, 1]}` puts head 3 at index 0. An operand must carry the same picks in the same order as the write it meets, or it is refused: `'c.v' holds num_heads [3, 1], and this write's selection holds [1, 3]; a select list is positional, …`. Build every site of a head set from one sorted list.
- **Map an index along a selected axis back through the select list.** `argmax`, `top_k` and `clab table` count within the selection, not along the model's axis ([values-and-layout.md](../concepts/values-and-layout.md#an-index-counts-along-the-value)).
- **A write selects on one axis.** A read may select heads and dimensions together; a write is refused with `an entry that writes selects on one axis; this one selects num_heads, head_dim`.
- **`key` counts from the start.** A key selector picks the same keys of every row (`0` for BOS, `[0, 2]`, `"0:3"`); `-1` is refused, since rows differ in length ([the key axis](../concepts/values-and-layout.md#the-key-axis)).
- **Or from the query: `{"key": {"offset": k}}`.** Use it for a score at a fixed distance from each query, where a key counted from the start would be a different token at each query. At each query position p the read holds, it is the row's key p + k: `{"offset": -1}` is the previous token, `{"offset": 0}` the query itself. A list, `{"offset": [-1, -2]}`, is one key per offset in the order written, so the `key` axis is as wide as the offsets are many.
- **An induction stripe is one offset read.** On 50 random tokens repeated after BOS, `{"site": {"component": "attention_probabilities", "layers": ":", "select": {"key": {"offset": -49}}}, "pos": {"index": "51:100"}, "ops": [{"op": "mean", "over": ["pos"]}]}` is each head's mean weight on the token after the current token's earlier copy. The read gives the same numbers as the whole pattern sliced at each row's p + k.
- **An offset read is NaN where p + k is no key of the row**, before key 0 or past a shorter row's last. A key ahead of the query holds the causal mask's 0. A table labels each entry with its key p + k and that key's token, or with its `offset` once the queries are reduced away.
- **Offsets are a read's.** A write is refused one (`keys relative to the query are a read's: …`), and an operand at offsets meets only a value at the same offsets.
- **`readout` needs the whole stream.** Selecting `hidden_size` and then `readout` is refused, since the final norm needs every dimension; read the site whole, `readout`, then narrow the logits. Which components `readout` takes is in [ops.md](ops.md#transforms).
- **A metric needs the whole vocabulary.** `match`, `logit_diff` and `token_prob` look an answer up by token id, so a head narrowed on `vocab_size` is refused under one: `the value 'match' scores keeps 3 of its vocab_size (select), …`. Score the full head and narrow what the metric gives ([ops.md](ops.md#metrics)).

A pick outside the axis is refused: `select num_heads [4] outside the model's 4`. A `select` may be swept, as in the canonical example.

## Attention needs eager

These components are read inside the attention's forward: `attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities` and `attention_head_outputs`. Fused attention (`sdpa`, which the corpus model loads by default) computes them inside one call, and the nnterp engine refuses them before any forward with a reason that asks for eager:

```text
clab: clean.pattern: the nnterp engine does not serve attention_probabilities: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Load the model with eager attention, in the `model` block, as `documents/attention_knockout.json` does:

```json
"model": {"key": "...", "revision": "...", "dtype": "float32", "attn_implementation": "eager"}
```

`attention_premix`, the output projection's input, needs no eager; on the corpus Llama it gives the same numbers as `attention_head_outputs`. Under eager the hooks engine still refuses the components above, since it reaches module boundaries only; it serves `attention_premix` and the MLP's interior ([engines.md](engines.md#cross-checking-with-hooks)). The vllm engine needs no eager either, since vLLM's attention backend is its own. There the queries, keys, values and head outputs are what goes into and comes out of its kernel. The scores and the pattern are recomputed from the queries and keys, read-only and in the prefill alone ([engines.md](engines.md#running-on-vllm)).

## The layout of a site's value

A read's value is laid out `(rows, pos, *the component's other axes)`, with a `layers` axis first when `layers` is a slice or a list; on an attention pattern `pos` is the query positions. `clab show` prints each saved value's layout. Why the axes are in that order, and how ops align them, is [values-and-layout.md](../concepts/values-and-layout.md#rows-and-positions-first).
