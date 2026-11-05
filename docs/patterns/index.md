---
title: Patterns Index
one_liner: "How do I run a technique? One page per technique, each built on a corpus document: activation patching, head and neuron patching with path patching, ablation, expert ablation, attribution patching, attention, logit and tuned lenses, steering, generation experiments, DAS and DBM, head masking, SAEs, probes, LoRA, cached harvests."
tags: [patterns, index]
related: [docs/index.md, docs/usage/index.md]
---

# Patterns Index

Up: [docs/index.md](../index.md) — all sections.

One technique per page, each built on a document under `documents/`; the usage pages own the parts it uses.

## Patching and ablation

- [activation-patching](activation-patching.md) — Which position or layer carries the answer? Swap a counterfactual's value in by position, window or entity, map layer by position, corrupt and restore.
- [head-and-neuron-patching](head-and-neuron-patching.md) — Which heads or neurons carry it? Patch one head or a few neurons at a time, head sets, and path patching.
- [ablation](ablation.md) — What does a component contribute? Zero, mean, resample (permute) and optimal-constant ablation, a direction at every layer, and the hydra effect.
- [expert-ablation](expert-ablation.md) — What does one expert of a mixture do? Knock it out, force a route, every expert in one document, routing usage and entropy.
- [attribution-patching](attribution-patching.md) — How do I rank every component with one backward? grad times the counterfactual's difference, the gradient check, and where it misleads.
- [attention](attention.md) — Where does a head look? Patterns, previous-token and induction stripes, knocking out heads or keys, patching a pattern, queries after RoPE.

## Reading, steering and generation

- [logit-lens](logit-lens.md) — What does each layer predict? The logit lens and its wiring check, top_k per layer, kl between lenses, and a tuned lens.
- [steering](steering.md) — How do I push the model with a direction? A mean difference at a dose set from the activations, a matched random control, clamping, and steering every generated token.
- [generation-experiments](generation-experiments.md) — Did the model say the answer, and how sure was it? contains or a whole-word scope, the text after a marker, a reasoning trace's length, teacher forcing, and chat turns.

## Trained featurizers

- [trained-subspaces](trained-subspaces.md) — Where is a variable stored? DAS and DBM, scored held out against PCA and random subspaces of matched rank.
- [head-masking](head-masking.md) — Which heads does a behavior need? A learned gate per head of every layer, its nulls, the L1 weight, and the heads kept.
- [sae](sae.md) — How do I use a sparse autoencoder? Splice a published one in, check its L0, ablate features, find max-activating examples, train one.
- [probing](probing.md) — Is a variable linearly decodable, and at which layer? A linear probe per layer, its untrained floor, many probes in one fit.
- [lora](lora.md) — How do I train and apply a low-rank adapter? Through the frozen model, on in chosen steps, across sites.
- [cached-harvest](cached-harvest.md) — How do I harvest once and reuse it? Save it, load it in another document, fit on it without a forward, harvest in shards.
