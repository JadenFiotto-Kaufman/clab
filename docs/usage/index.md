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

- [writing-a-document](writing-a-document.md) — The anatomy of a clab document: the root keys, a four-step patching experiment read top to bottom, how steps name each other's values, saves, and what is checked before the model loads.
- [data](data.md) — Declaring datasets and inputs: file, inline and hub rows; split, shuffle, take, columns and template; text and chat inputs, and image parts in a chat on a vision-language checkpoint; list inputs for teacher forcing; answer columns with strip, lower and sets.

## Locating

- [sites](sites.md) — Choosing where to read or write: the component table, layers as an int or a selector, select to narrow heads, neurons or keys (counted, or named by a scope), the layout a site's value has, and when attention needs eager.
- [positions](positions.md) — Pointing pos at the tokens you mean: the last token, an answer's tokens, a word by occurrence, the token before one, a chat turn, an image's tokens or a region of them, every token, the generated tokens, and reading back where each read and write landed.

## Reading and writing

- [interventions](interventions.md) — Reading and writing at a site: entries, reads versus writes, the order ops run in, operands from earlier steps or the same forward, writes across layers, named bundles of entries, and the null control every document should carry.
- [ops](ops.md) — Scoring and reducing values: metrics on logits or probe scores, reductions and over, transforms like readout, argmax, top_k and detokenize, op steps versus ops inside an entry, and the float32 rule.

## Repeating

- [sweeps](sweeps.md) — Running one document at many settings: what a sweep can stand for, labels that zip or cross, what each point costs, when one wider read replaces a sweep, and reading swept results by coordinate.
- [generate](generate.md) — Generation steps: options passed to the model's generate, deterministic decoding, frames for the prefill and decode calls, writes kept on through generation, writes found online at generated text, EOS and rows that stop early.

## Featurizers and training

- [featurizers](featurizers.md) — Declaring a featurizer and using it in an entry: the six kinds, encode, decode and apply, nested brackets, selecting features, and loading a published sparse autoencoder with the fields the file cannot state.
- [fit](fit.md) — Training a featurizer inside a run: train, objective, optimizer, epochs and minibatches, early stopping, eval substitutions, the history and members a fit produces, and epochs 0 as a null control.
- [bundles](bundles.md) — Saving a trained or harvested featurizer and loading it in another document: the files a run writes, file_path, the header checks at compile and before any forward, the two-document workflow, and reading back a run that loads one.

## Running

- [running](running.md) — Running a document from the command line or Python: clab run and its options, choosing batch size, bit reproducibility, where output goes and the corpus root, and the four calls behind clab.run.
- [engines](engines.md) — Choosing an engine: nnterp, hooks and ndif, what each can serve, device and device maps, engine options, running on an NDIF server or its local simulation, and what the server must have installed.
- [output](output.md) — Reading a run back: the files a run writes, clab show and clab table and their columns, plan.table in Python, why tabling needs the data files, the manifest, and the run record with where every read and write landed.
