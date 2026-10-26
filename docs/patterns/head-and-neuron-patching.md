---
title: Head and Neuron Patching
one_liner: "Patching one head or a few neurons at a time: select on attention_premix and mlp_activation, cross a layers sweep with a heads sweep, head sets, and the positional meaning of select lists."
tags: [patterns, heads]
related: [docs/patterns/activation-patching.md, docs/usage/sites.md, docs/usage/sweeps.md, docs/usage/interventions.md, docs/patterns/attention.md, docs/usage/engines.md]
sources: [clab/compile/interventions.py, clab/steps/batches.py, clab/format/address.py]
---

# Head and Neuron Patching

[Activation patching](activation-patching.md) of the residual stream moves everything a layer computed at a position. To ask which attention head or which MLP neurons carry it, patch a narrower site: a `select` keeps some indices of one axis of the component, and the write replaces those and leaves the rest of the layer alone. The documents are the canonical patching document with a different site.

## One head at a time

`documents/head_patching.json`:

```json
"target": {"component": "attention_premix", "layers": 0, "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}}
```

`attention_premix` is every head's result as the attention's output projection takes it, laid out `(rows, pos, num_heads, head_dim)`. A head's result is one index of `num_heads`, and the output projection is linear, so patching that index changes exactly that head's share of what attention adds to the stream. The read on the counterfactual prompt and the write on the base prompt use the same site, and the sweep makes one point per head, each a whole plan with its own read and write:

```
at               row  pos  pos_token  input                              cf_answer  base_answer  value
{"num_heads":0}  0    10   ▁is        If today is Thursday, tomorrow is   Sunday     Friday      -0.027985602617263794
...
{"num_heads":3}  3    8    ▁is        If today is Sunday, tomorrow is     Saturday   Monday      -0.19152364134788513
```

The projection's input sits at a module boundary, so both engines reach it, and `--engine hooks` gives the same numbers as nnterp to the bit. `attention_head_outputs` is the same tensor read inside the attention's forward; nnterp serves it under `attn_implementation: "eager"` only, and there it equals `attention_premix` to the bit. Switching a model from sdpa to eager moves the last bits of everything downstream (about 2e-8 in this document's logit differences), so compare runs under one attention implementation ([engines](../usage/engines.md)).

## Layers crossed with heads

Sweep `layers` on the same site. Each sweep is labelled by its field (`layers`, `num_heads`), different labels cross, and the step gets a point per pair:

```json
"target": {"component": "attention_premix", "layers": {"sweep": [0, 1]},
           "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}}
```

Eight points, keyed `layers=0,num_heads=0` to `layers=1,num_heads=3`; a `mean` step over `logit_diff` gives one number per point, which `clab table` lists with its `at`. `layers` here is a sweep, not a selector: `"layers": ":"` would patch head h at every layer in one forward ([interventions](../usage/interventions.md#writes-across-layers)). `documents/real/head_patching.json` is the same grid on Llama-3.2-1B, three layers by 32 heads, 96 points; it scores the head inside the entry (`match` and `logit_diff` in the head's `ops`) so no row of logits outlives its batch, and averages each point in a step. It needs a GPU and a gated checkpoint: read it, do not expect to run it on CPU.

## Head sets

A list keeps several heads, and one write patches them together:

```json
"select": {"num_heads": {"sweep": [[0, 1], [2, 3], [0, 1, 2, 3]]}}
```

Each point's coordinate is the list (`num_heads=[0,1]`). Heads that back each other up show an effect only together, so a set that moves the answer when no member does alone is a finding, not noise.

## Select lists are positional

A narrowed axis keeps its indices in the order the list writes them: a read with `{"num_heads": [3, 1]}` holds head 3 at index 0 and head 1 at index 1. An operand's indices on that axis must be the write's, in the same order, and preflight refuses anything else before any forward:

```text
patched.patch: operand 'counterfactual.v_cf' holds num_heads [3, 1], and this write's selection holds [1, 3]; a select list is positional, so they would trade places: select num_heads [1, 3] at the read
```

A one-head read written into a two-head selection is refused too: an operand broadcasts over rows and positions, never over heads. Declaring one site and naming it in both the read and the write, as the corpus does, makes the two agree by construction. A write narrows one axis: `{"num_heads": [1], "head_dim": [0, 1]}` on a write is refused. The general rule is in [sites](../usage/sites.md).

## Neurons

`documents/neuron_patching.json` narrows an MLP instead, four neurons at a time:

```json
"target": {"component": "mlp_activation", "layers": 0,
           "select": {"intermediate_size": {"sweep": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]}}}
```

Which MLP component is "the neuron" matters on a gated MLP (Llama's `down(act(gate x) * up x)`). `mlp_activation` is `act(gate x)`: patching it swaps the gate and keeps the base prompt's `up x`, so the neuron's output becomes the counterfactual's gate times the base's up projection. `mlp_neurons` is what the down projection sums, the product: patching it swaps the neuron's whole contribution. The two give different numbers on this document; choose by the question, and say which in the result. On a plain MLP, `mlp_activation` is `act(up x)`.

## Heads under grouped-query attention

`attention_keys` and `attention_values` are narrowed by `num_kv_heads` and read before they are repeated across query heads. On a model whose attention groups query heads, one key or value head serves several query heads, so patching it moves all of them; patch `attention_premix` for one query head's result.

## Pitfalls

- A `layers` selector on a head patch patches that head at every layer at once; sweep `layers` for a layer by head grid.
- The read and the write must select the same heads in the same order; one site used for both guarantees it.
- A head's result (`attention_premix`) and its pattern (`attention_probabilities`) answer different questions: what it writes versus where it looks ([attention](attention.md)).
