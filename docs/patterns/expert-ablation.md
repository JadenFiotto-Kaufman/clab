---
title: Expert ablation
one_liner: "Knocking out one expert of a mixture of experts and reading its routing: the slots' mask read with ne and written with scale in a later forward, the null of an expert no slot holds, the forced route through the router's logits, every expert in one document, expert usage with eq, and routing entropy."
tags: [patterns, ablation, mixture-of-experts, routing, experts]
related: [docs/usage/sites.md, docs/patterns/ablation.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/engines.md]
sources: [documents/expert_knockout.json, clab/format/address.py, clab/engines/standard.py, clab/ops/reductions.py, clab/ops/transforms.py]
---

# Expert ablation

A mixture of experts routes each token to `top_k` of its `num_experts` experts, and what the block adds is the weighted sum of those experts' outputs. Removing one expert asks what the model loses without it; forcing the router away from it asks what it loses when the next expert stands in. Both are writes at one block, scored against a clean forward over the same rows, as any [ablation](ablation.md) is. The six components a mixture serves, and where each engine serves them, are in [sites](../usage/sites.md#a-mixture-of-experts).

## The knockout

`documents/expert_knockout.json` (the tiny Mixtral, 4 experts, 2 per token) zeroes every slot that chose expert e at block 0. The clean forward reads the mask, 1 where a slot chose another expert and 0 where it chose e; the knockout forward scales each slot's weight by it:

```json
"clean": {"kind": "forward", "input": "pairs.input", "interventions": {
  "mask": {"site": "slots", "pos": {"index": ":8"},
           "ops": [{"op": "ne", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}}]},
  "logits": {"site": "head", "pos": -1}}},
"knockout": {"kind": "forward", "input": "pairs.input", "interventions": {
  "kill": {"site": "weights", "pos": {"index": ":8"}, "ops": [{"op": "scale", "factor": "clean.mask"}]},
  "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "clean.logits"}]}}}
```

`slots` is `expert_indices` and `weights` is `expert_weights`, both at block 0. Expert e's slots then add nothing to `routed_output`: it is the clean run's less those slots' `expert_outputs`, to float32 rounding. The other slots keep their weights; nothing renormalizes them, so a token that chose e loses that share of the mixture's output.

The mask comes from the clean forward because the routing at the knocked-out block is the clean run's: nothing before the block changed, so the indices the knockout forward meets there are the ones the clean forward read. Two rules shape the entries:

- **The mask is read at an index of static width**, here the 8 tokens every prompt has (`":64"` on 64-token windows). An operand from another forward meets the write position by position, and `{"index": ":"}` holds a different number on each row, which is refused: `'clean.mask' holds a different number of positions on each row, which pair with no other value's`.
- **The mask cannot come from the knockout's own forward.** An entry's `ne` runs once the value is off the model, so an operand naming it in its own forward is refused: `'patched.idx' runs 'ne' once it is off the model, after this forward; an operand from the same forward is an entry with no op after its edits`.

## The null

The same knockout with an expert id no slot holds must give the clean numbers to the bit. `num_experts` is never an index, so the corpus document's `none` reads `{"op": "ne", "with": 4}`, a mask of ones; its forward's logits equal the clean ones to the bit and its KL is exactly 0. A null that is not exact means the mask and the write are not at the same slots.

## The forced route

Writing a very negative logit for expert e into `router_logits` before the router scores it re-routes every token that chose e to its next choice, with the weight the router gives that expert:

```json
"one_expert": {"component": "router_logits", "layers": 0,
               "select": {"num_experts": {"sweep": [0, 1, 2, 3], "label": "expert"}}},
"away": {"site": "one_expert", "pos": {"index": ":"}, "ops": [{"op": "swap", "with": -1e9}]}
```

No slot chooses e in that forward. The forced route needs no mask and no static width, and on a softmax router it renormalizes over the rest, so it measures a substitute where the knockout measures a gap. On OLMoE-1B-7B's block 8, over 300 windows of 64 tokens in bfloat16, zeroing expert 18 raises the continuation loss by 0.0051 nats and forcing it away by 0.0042, the null exactly 0.

## Every expert

A sweep over the expert id with one `label` makes each expert a point: the clean forward reads every mask in its one forward (points that differ only in what a forward reads share it), and the knockout forward runs once per expert, its operand the matching point's mask. The forced route's `select` sweeps under the same label, so `knockout.kl` and `forced.kl` are four points each, keyed by `expert` ([sweeps](../usage/sweeps.md#labels-zip-or-cross)). A grid of blocks by experts crosses a `layers` sweep under another label.

## Routing entropy and usage

```json
"chosen": {"site": "slots", "pos": {"index": ":"},
           "ops": [{"op": "eq", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}},
                   {"op": "sum", "over": ["pos", "top_k"]}]},
"routing": {"site": "router", "pos": {"index": ":"}, "ops": [{"op": "entropy"}]},
"usage": {"kind": "op", "op": "sum", "of": "clean.chosen", "over": ["rows"]}
```

`chosen` counts each row's slots that chose e and `usage` sums them over rows: the four experts' usages add up to every slot of every token, padding never among them, since a read holds each row's own positions and a comparison of a hole is a hole. `entropy` of `router_logits` is each token's routing entropy in nats, over `num_experts` by default, ln of the count where the router is uniform. It is the routing distribution's on a softmax router; on a sigmoid router (DeepSeek-V3, GLM-4-MoE) or one that takes the top k before its softmax (JetMoE, GPT-OSS) the softmax of the logits is not what routes, and `expert_weights` is the value to read.

## Pitfalls

- A knockout at block L changes the stream after it, so later blocks may route differently, and the effect includes that re-routing; read a later block's `expert_indices` in both forwards to see how much.
- The weights of the slots left are not renormalized after a knockout; the forced route renormalizes on a softmax router. They answer different questions, and both are reported.
- Read `expert_weights` rather than recomputing them from the logits: whether a family renormalizes its top k (OLMoE does not) or scales them is the family's, and the read is what multiplies each slot.
- A bit-exact null in bfloat16 certifies the wiring, not effects of 0.005 nats; recompute a few experts in float32 before ranking them ([ops](../usage/ops.md#the-float32-rule)).
- On the hooks engine the routing pair and, on most families, the router's logits are refused; run expert ablations on nnterp and cross-check `routed_output` on hooks ([engines](../usage/engines.md#a-mixture-of-experts-on-hooks)).
