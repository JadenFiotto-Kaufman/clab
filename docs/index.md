---
title: nnspec Docs
one_liner: The router for docs/ — which section answers which question, and the pages most tasks start from.
tags: [index]
---

# nnspec Docs

Find the section that answers your question, then the one page inside it. Read that page before writing a document.

## Sections

- [usage/](usage/index.md) — How do I write and run a document? One page per part of the format and per stage of a run: a document's anatomy, data, sites, positions, interventions, ops, sweeps, generation, featurizers, fits, bundles, running, engines, output.
- [patterns/](patterns/index.md) — How do I run a technique? One page per technique, each built on a corpus document: activation patching, head and neuron patching with path patching, ablation, expert ablation, attribution patching, attention, logit and tuned lenses, steering, generation experiments, DAS and DBM, head masking, SAEs, probes, LoRA, cached harvests.
- [concepts/](concepts/index.md) — Why does it behave this way? The plan's lifecycle, one session per run, the position model, values and layout, names and sweeps.
- [errors/](errors/index.md) — What does this refusal mean, or why are the numbers wrong when nothing was refused? The index is keyed by the refusal's message, with a page each for the refusals a line cannot explain: an operand that does not fit, a position not in a row, an answer that does not line up. [symptom-index.md](errors/symptom-index.md) covers wrong numbers that nothing refused, and [alignment-missing.md](errors/alignment-missing.md) the rows a generation did not reach.
- [reference/](reference/index.md) — What is the exact name? [FORMAT.md](../FORMAT.md), the spec generated from the code, by section; the CLI; the Python API; the glossary.
- [developing/](developing/index.md) — How is nnspec built, and how do I change it? Architecture, compile and run, extending, the engine protocol, testing, contributing.

## Most-used pages

- [usage/writing-a-document.md](usage/writing-a-document.md) — the anatomy of a document, read top to bottom
- [patterns/index.md](patterns/index.md) — the technique you came to run, one page each
- [usage/positions.md](usage/positions.md) — pointing `pos` at the tokens you mean
- [usage/interventions.md](usage/interventions.md) — reading and writing at a site
- [usage/ops.md](usage/ops.md) — scoring and reducing values
- [usage/sweeps.md](usage/sweeps.md) — one document at many settings
- [usage/running.md](usage/running.md) — `nnspec run` and `nnspec.run`, batch size, where output goes
- [usage/output.md](usage/output.md) — reading a run back: `nnspec show`, `nnspec table`, the manifest
- [errors/index.md](errors/index.md) — a refusal, looked up by its message
- [errors/symptom-index.md](errors/symptom-index.md) — it ran, nothing was refused, and the numbers are wrong
- [FORMAT.md](../FORMAT.md) — every field, step kind, op, component, featurizer kind and engine option
- [reference/cli.md](reference/cli.md) — every command and flag
