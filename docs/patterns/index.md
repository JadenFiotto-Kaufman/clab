---
title: Patterns Index
one_liner: "One interpretability technique per page, each built on a corpus document: patching, attribution patching, ablation, expert ablation, attention, logit lens, steering, generation, trained subspaces, head masking, SAEs, probes, LoRA and cached harvests."
tags: [patterns, index]
related: [docs/index.md, docs/usage/index.md]
---

# Patterns Index

Up: [docs/index.md](../index.md) — all sections.

One technique per page, each built on a document under `documents/`; the usage pages own the parts it uses.

## Patching and ablation

- [activation-patching](activation-patching.md) — Interchange interventions: swap one position, a window or an entity's token from a counterfactual run into a base run, score with match and logit_diff, map layer by position with paired sweeps, and corrupt and restore with noise.
- [head-and-neuron-patching](head-and-neuron-patching.md) — Patching one head or a few neurons at a time: select on attention_premix, mlp_activation versus mlp_neurons, cross a layers sweep with a heads sweep, head sets, path patching, and the positional meaning of select lists.
- [ablation](ablation.md) — Zero, mean, resample, noise, direction and optimal ablation: resample as a permutation of a read's rows, the optimal constant as a trained featurizer, a mean as an operand across datasets, a ragged mean over an entity's tokens, a direction ablated at every layer, and the hydra effect.
- [expert-ablation](expert-ablation.md) — Knocking out one expert of a mixture of experts and reading its routing: the slots' mask read with ne and written with scale in a later forward, the null of an expert no slot holds, the forced route through the router's logits, every expert in one document, expert usage with eq, and routing entropy.
- [attribution-patching](attribution-patching.md) — Attribution patching with grad: the gradient of a metric at a site times the counterfactual's difference there, every component of a model in two or three forwards and one backward, checked against a central difference, and where it misleads.
- [attention](attention.md) — Experiments inside attention: knocking out a head or keys through the scores or the pattern renormalized by sum, patching a head's pattern while keeping its values, and reading queries and keys after RoPE.

## Reading, steering and generation

- [logit-lens](logit-lens.md) — The logit lens in one forward: readout over a layers selector, a kl check against the model's logits, kl between lenses aligned by layer label, top_k to see what each layer would say, and a tuned lens with a trained affine translator per layer.
- [steering](steering.md) — Steering with a direction: harvest a mean difference, add it at a dose beside a matched random direction, sweep the dose, renormalize, clamp, and keep the write on through every generated token.
- [generation-experiments](generation-experiments.md) — Scoring what the model says: generated answers, confidence before the answer, contains over decoded text, forced answers by teacher forcing, answer spellings, and positions inside chat turns.

## Trained featurizers

- [trained-subspaces](trained-subspaces.md) — DAS, DBM and their controls: fit a rotation or a mask, score held-out accuracy, and compare against PCA and random subspaces of matched rank, a fit at epochs 0 among them.
- [head-masking](head-masking.md) — Which attention heads carry a behavior, learned as a mask: desiderata-based masking with one gate unit per head of every layer (over), a dropped head taking the corrupt run's value through apply with, the bit-exact null, an L1 sweep, and reading the heads kept.
- [sae](sae.md) — Sparse autoencoders: splice a published JumpReLU or TopK dictionary in, ablate features through encode and decode, check its L0, and train a dictionary inside a run.
- [probing](probing.md) — Linear probes: train a linear featurizer on a label column, score its classes with the same metrics as logits, sweep layers to see where a variable is decodable against the floor of an untrained probe, and train many probes in one fit.
- [lora](lora.md) — Low-rank adapters: train a LoRA through the frozen model to change an answer, apply it as a named intervention in some steps and not others, and adapt across sites with with.
- [cached-harvest](cached-harvest.md) — Harvest once, use it many times: save a harvest, bring it into another document with a load step, fit a probe or take a pca on it without a forward, make a large harvest in shards with skip and take, and what must match for a load to compile.
