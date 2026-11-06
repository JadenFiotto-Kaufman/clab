---
title: Usage Index
one_liner: How to write and run a clab document, one page per part of the format and per stage of a run.
tags: [usage, index]
related: [docs/index.md, docs/patterns/index.md, docs/reference/index.md]
---

# Usage Index

Up: [docs/index.md](../index.md) — all sections.

Find the part of the document you are writing or the stage of the run you are at; each page links [FORMAT.md](../../FORMAT.md) for the exact fields.

## Declaring

- [writing-a-document](writing-a-document.md) — How do I write a document? The root keys, one patching experiment read top to bottom, how steps name each other's values, and saves.
- [data](data.md) — How do I load rows and pick what a step runs over? File, inline and hub datasets, narrowing, renamed and derived columns, chats, teacher forcing and answer columns.

## Locating

- [sites](sites.md) — Where do I read or write? Choosing a component and layers, and select over heads, neurons, experts or keys, from the start or relative to each query.
- [positions](positions.md) — Which tokens does an entry touch? pos for the last token, a word, the token before it, a chat turn, text up to a marker and generated tokens, and checking where it landed.

## Reading and writing

- [interventions](interventions.md) — How do I read or write a value at a site? Entries, edits, the order they run in, operands, gradients, writes across layers, named bundles and the null control.
- [ops](ops.md) — Which op scores, reduces or transforms a value? Metrics, KL in either direction, reductions (median, count), top_k over rows and positions, permute, arithmetic, and step versus entry.

## Repeating

- [sweeps](sweeps.md) — How do I run one document at many settings? What a sweep may stand for, labels that zip or cross, each point's cost, and reading swept results.
- [generate](generate.md) — How do I generate, and read or write as it generates? Options, frames (0, 1:, :, -1) for the prefill and decode calls, a read at every call, writes through every generated token, and rows that stop early.

## Featurizers and training

- [featurizers](featurizers.md) — How do I use a learned or loaded space in an entry? Declaring each kind, encode, decode and apply, selecting features, and a published SAE.
- [fit](fit.md) — How do I train a featurizer inside a run? Objective, optimizer, epochs, early stopping, evals, history, what a fit costs, and the epochs-0 null.
- [bundles](bundles.md) — How do I save a trained featurizer and load it in another document? The files, file_path, the header checks and the two-document workflow.

## Running

- [running](running.md) — How do I run a document, and why would two runs differ? The CLI and Python, batch size, padding, readout, memory, the corpus root and tp.
- [engines](engines.md) — Which engine runs it? nnterp, hooks, ndif and vllm: what each serves, devices, options, quantized and sharded models.
- [output](output.md) — How do I read a run back? clab show, clab table, plan.results, the manifest's positions, the run record, and loading a saved value elsewhere.
