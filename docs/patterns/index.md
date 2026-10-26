---
title: Patterns Index
one_liner: "One interpretability technique per page, each built on a corpus document: patching, ablation, attention, logit lens, steering, generation, trained subspaces, SAEs, probes and LoRA."
tags: [patterns, index]
related: [docs/index.md, docs/usage/index.md]
---

# Patterns Index

Up: [docs/index.md](../index.md) — all sections.

One technique per page, each built on a document under `documents/`; the usage pages own the parts it uses.

## Patching and ablation

- [activation-patching](activation-patching.md) — Interchange interventions: swap one position, a window or an entity's token from a counterfactual run into a base run, score with match and logit_diff, map layer by position with paired sweeps, and corrupt and restore with noise.
- [head-and-neuron-patching](head-and-neuron-patching.md) — Patching one head or a few neurons at a time: select on attention_premix, mlp_activation versus mlp_neurons, cross a layers sweep with a heads sweep, head sets, and the positional meaning of select lists.
- [ablation](ablation.md) — Zero, mean, resample, noise and direction ablation: a mean as an operand across datasets, a ragged mean over an entity's tokens, a direction ablated at every layer, and the hydra effect.
- [attention](attention.md) — Experiments inside attention: knocking out a head or keys through the scores or the pattern renormalized by sum, patching a head's pattern while keeping its values, and reading queries and keys after RoPE.

## Reading, steering and generation

- [logit-lens](logit-lens.md) — The logit lens in one forward: readout over a layers selector, a kl check against the model's logits, kl between lenses aligned by layer label, and top_k to see what each layer would say.
- [steering](steering.md) — Steering with a direction: harvest a mean difference, add it at a dose, sweep the dose, renormalize, clamp, and keep the write on through every generated token.
- [generation-experiments](generation-experiments.md) — Scoring what the model says: generated answers, confidence before the answer, contains over decoded text, forced answers by teacher forcing, answer spellings, and positions inside chat turns.

## Trained featurizers

- [trained-subspaces](trained-subspaces.md) — DAS, DBM and their controls: fit a rotation or a mask, score held-out accuracy, and compare against PCA and random subspaces of matched rank, a fit at epochs 0 among them.
- [sae](sae.md) — Sparse autoencoders: splice a published JumpReLU or TopK dictionary in, ablate features through encode and decode, check its L0, and train a dictionary inside a run.
- [probing](probing.md) — Linear probes: train a linear featurizer on a label column, score its classes with the same metrics as logits, and sweep layers to see where a variable is decodable, against the floor of an untrained probe.
- [lora](lora.md) — Low-rank adapters: train a LoRA through the frozen model to change an answer, apply it as a named intervention in some steps and not others, and adapt across sites with with.
