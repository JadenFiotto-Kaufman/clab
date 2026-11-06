---
title: Sites
one_liner: "Choosing where to read or write: the component table, a mixture of experts' routing and experts, layers as an int or a selector, select to narrow heads, neurons, experts or keys (counted from the start or from each query), the layout a site's value has, and when attention needs eager."
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
| one head's result | `attention_premix` + `select: {"num_heads": ...}` | no eager needed; [head-and-neuron-patching](../patterns/head-and-neuron-patching.md) |
| an attention pattern, or the scores before softmax | `attention_probabilities`, `attention_scores` | eager; [attention](../patterns/attention.md) |
| queries, keys, values | `attention_queries`, `attention_keys`, `attention_values` | eager; keys and values per KV head, before `repeat_kv` |
| MLP neurons | `mlp_activation` (the activation of the gate), `mlp_neurons` (what the down projection sums) | different on a gated MLP |
| a mixture of experts' routing | `router_logits` (`num_experts` wide), `expert_weights`, `expert_indices` (`top_k` slots, the indices int64) | below |
| what its experts add | `expert_outputs` (each slot's, weighted), `routed_output`, `shared_expert_output` | the slots sum to `routed_output`; `routed_output` + `shared_expert_output` is `mlp_output` on most families |
| the next-token distribution | `logits` (after any softcap), `lm_head_output` (the head alone) | equal on a model without a softcap |
| a logit lens | a stream component with `readout` | [logit-lens](../patterns/logit-lens.md) |

The identities in the notes hold to the bit on the corpus Llama, a pre-norm block; check them on your architecture before relying on one.

### A mixture of experts

At a block whose MLP is a mixture of experts, six components are its values, by nnterp's names and layouts: the router's logits before its scoring, the weight and the expert of each of a token's `top_k` routing slots, each slot's output times its weight, their sum, and the shared expert's output where the mixture has one. The model's sizes then include `num_experts` and `top_k`, which `select` counts among (`{"num_experts": 18}` is expert 18's logit), read off the first block whose MLP is a mixture. `mlp_output` is still what the block adds; `mlp_activation` and `mlp_neurons`, a dense MLP's values, are refused there, the refusal naming what the block serves (`block 0's MLP is a mixture of experts, which has no single activation or down projection; at block 0 its MLP is served as mlp_norm_output, mlp_input, router_logits, ...`). On a dense block each mixture value is refused at that layer, and on a model with no mixture at every layer (`no block of this model has a mixture of experts`); on a model nnterp has no family for, whose mixture its default family finds but names no value of, at every layer too (`nnterp has no family for this architecture (it reads the model under its default family), so its mixture of experts is not addressed; its MLPs are served as mlp_input`); `shared_expert_output` on a mixture without one (`this mixture has no shared expert`). `expert_indices` holds ids, which the model indexes its experts with, so a write there is a `swap` of ids: any other edit is refused at compile (`'add' would compute on expert_indices, the ids the model indexes its num_experts with, as numbers, and the write would cast the result back to ids, truncated; an index takes a swap, ...`), a constant that is no id under the model's `num_experts` before any forward (`swap with 4: expert_indices holds ids, and an id is a whole number from 0 to 3 (the model's num_experts, 4)`), and a named value's entry that is none as the write lands, before the model reads it, a hole (-1) included (`writes a hole, -1, into expert_indices at row 0, token 7, where its operand holds no id ...`), where a generation's read on a row that stopped before its call is refused earlier, before the forward (`operand 'g.idx' holds no value on row 0 (alignment_missing) ...`): transformers' grouped expert matmul drops an id outside its experts and shifts the groups after it, which changes the routed output at every position of every row, finite throughout. Which values a family has, and what its logits mean (a softmax router, a sigmoid one, a top-k then softmax), is nnterp's: its `support()` answers per block, and the hooks engine serves fewer of them ([engines](engines.md#a-mixture-of-experts-on-hooks)). One forward reads any of them in forward order, on a fresh engine too: the nnterp engine instruments, before the run's first step, each module of the mixture whose forward nnterp reads a value inside, which nnsight would otherwise do at the first trace that reads it, too late on JetMoE, whose router logits are its router's projection's output, read before the indices inside the router's forward. Knockouts, forced routes, usage and routing entropy are [expert ablation](../patterns/expert-ablation.md).

## `layers`

A block's component needs `layers`; the model's own values (`token_embeddings`, `final_norm_output`, `lm_head_output`, `logits`) refuse it. `layers` is a selector over the model's layers ([FORMAT.md#pos](../../FORMAT.md#pos) has the grammar):

- **An int** is that layer, negative counting from the last: `-1` is the last block. The value has no layers axis.
- **A slice or a list** reads every layer it picks in one forward and stacks them, a `layers` axis first, labelled with the layers picked, in that order: `":"` is every layer, `"-2:"` the last two, `[1, 0]` layer 1 then layer 0. `clab table` prints the model's layer number in its `layer` column.
- **A sweep** (`{"sweep": [0, 1]}`) is one point per layer. A write's points run one forward each; a read's share one forward, as a selector does, and give the same numbers one point per layer. Sweep when each layer is a different intervention; to reduce or compare across layers, read them with a selector ([sweeps.md](sweeps.md#widen-the-read-instead)).

`0` and `[0]` read the same numbers, but `[0]` keeps a `layers` axis of size 1: an extra leading axis in the saved tensor and a `layer` column in the table. As an operand either meets a write at layer 0 alike, since operands align by layer label. A layer the model does not have is refused before any forward: `clab: f.x: layer 5; the model has 2 layers`. A write across several layers runs its ops at each, as [interventions.md](interventions.md) describes.

## `select`

`select` narrows one or more of the component's feature axes, `{<axis>: <selector>}`: `num_heads` or `num_kv_heads` for heads, `head_dim`, `intermediate_size` for neurons, `hidden_size`, `vocab_size`, or `key` on an attention pattern. The component table lists which axes each component has; naming another is refused with the list (`attention_premix has no axis 'seq' to select; its axes: num_heads, head_dim`).

- **The axis stays.** `{"num_heads": 2}` is a `num_heads` axis of size 1, the same value as `[2]`, so a site has one layout whatever it selects. On `attention_premix` at `pos: -1` that is `[rows, 1, 1, head_dim]`.
- **A list keeps its order, and the order is positional.** `{"num_heads": [3, 1]}` puts head 3 at index 0. An operand must carry the same picks in the same order as the write it meets, or it is refused; an `argmax` or `top_k` over the selected axis returns indices into the selection, not head numbers. Build every site of a head set from one sorted list. See [values-and-layout.md](../concepts/values-and-layout.md#select-keeps-the-axis).
- **A write selects on one axis.** A read may select heads and dimensions together; a write is refused with `an entry that writes selects on one axis; this one selects num_heads, head_dim`.
- **`key` counts from the start.** A key selector picks the same keys of every row (`0` for BOS, `[0, 2]`, `"0:3"`); `-1` is refused, since rows differ in length. The `key` axis is each row's own tokens, as positions count them.
- **Or from the query: `{"key": {"offset": k}}`.** At each query position p the read holds, it is the row's key p + k: `{"offset": -1}` is the previous token, `{"offset": 0}` the query itself, and a list, `{"offset": [-1, -2]}`, one key per offset in the order written, so the `key` axis is as wide as the offsets are many. An entry is NaN where p + k is no key of the row (before key 0, or past a shorter row's last), and a key ahead of the query holds the causal mask's 0. An induction stripe on a sequence of 50 random tokens repeated is one read, `{"site": {"component": "attention_probabilities", "layers": ":", "select": {"key": {"offset": -49}}}, "pos": {"index": "51:100"}, "ops": [{"op": "mean", "over": ["pos"]}]}`; the read equals the whole pattern sliced at each row's p + k to the bit. A table labels each entry with its key p + k and that key's token, or with its `offset` once the queries are reduced away. A write is refused one, and an operand at offsets meets only a value at the same offsets.
- **`readout` needs the whole stream.** Selecting `hidden_size` and then `readout` is refused, since the final norm needs every dimension; read the site whole, `readout`, then narrow the logits. Which components `readout` takes is in [ops.md](ops.md#transforms).
- **A metric needs the whole vocabulary.** `match`, `logit_diff` and `token_prob` look an answer up by token id, so a head narrowed on `vocab_size` is refused under one: `the value 'match' scores keeps 3 of its vocab_size (select), and it scores 'pairs.cf_answer' by its id over the whole vocabulary, where index i of the selection is not id i: ...`. Score the full head and narrow what the metric gives ([ops.md](ops.md#metrics)).

A pick outside the axis is refused: `select num_heads [4] outside the model's 4`. A `select` may be swept, as in the canonical example.

## Attention needs eager

Six components are read inside the attention's forward: `attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities` and `attention_head_outputs`. Fused attention (`sdpa`, which the corpus model loads by default) computes them inside one call, and the nnterp engine refuses them before any forward with a reason that asks for eager:

```text
clab: clean.pattern: the nnterp engine does not serve attention_probabilities: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Load the model with eager attention, in the `model` block, as `documents/attention_knockout.json` does:

```json
"model": {"key": "...", "revision": "...", "dtype": "float32", "attn_implementation": "eager"}
```

`attention_premix`, the output projection's input, needs no eager; on the corpus Llama it equals `attention_head_outputs` to the bit. Under eager the hooks engine still refuses these six, since it reaches module boundaries only; `attention_premix` and the MLP's interior it serves ([engines.md](engines.md#cross-checking-with-hooks)). The vllm engine needs no eager either: vLLM's attention backend is its own, the queries, keys, values and head outputs are what goes into and comes out of its kernel, and the scores and the pattern are recomputed from the queries and keys, read-only and in the prefill alone ([engines.md](engines.md#running-on-vllm)).

## The layout of a site's value

A read's value is laid out `(rows, pos, *the component's other axes)`, with a `layers` axis first when `layers` is a slice or a list: `attention_premix` at layer 0 is `rows, pos, num_heads, head_dim`; `attention_probabilities` is `rows, pos, num_heads, key`, its `pos` being the query positions. `clab show` prints each saved value's layout. Why the axes are in that order and how ops align them is [values-and-layout.md](../concepts/values-and-layout.md).
