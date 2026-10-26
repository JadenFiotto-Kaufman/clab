---
title: Reference Index
one_liner: Routes to FORMAT.md's sections (the format, ops, components, kinds, engines, sweeps, output), the CLI, the Python API and the glossary.
tags: [reference, index]
related: [docs/index.md, FORMAT.md]
---

# Reference Index

Up: [docs/index.md](../index.md) — all sections.

Exact names: the generated spec first, then the command line, the Python API and the glossary.

## FORMAT.md

[FORMAT.md](../../FORMAT.md), generated from the code by `clab describe --markdown`, section by section:

- [root keys and records](../../FORMAT.md#root-keys) — the root keys, then the fields of every record a document is made of, the [`pos`](../../FORMAT.md#pos) grammar among them.
- [step kinds](../../FORMAT.md#step-kinds) — `forward`, `generate`, `op` and `fit`, each with its fields.
- [derived fields](../../FORMAT.md#derived-fields) — the fields a document leaves out, which compile and the engine fill in.
- [ops](../../FORMAT.md#ops) — every op's arguments and what it does to the axes, grouped as metrics, reductions, edits and transforms.
- [components](../../FORMAT.md#components) — every component a site can name, its layout, and the axes `select` narrows.
- [featurizer kinds](../../FORMAT.md#featurizer-kinds) — the six kinds, the fields each takes, and what its ops do.
- [engines and run options](../../FORMAT.md#engines-and-run-options) — the engine protocol, `nnterp`, `hooks` and `ndif`, and each engine's options.
- [names and scoping](../../FORMAT.md#names-and-scoping) — how a reference resolves, what a scope sees, and which dataset a value's rows belong to.
- [sweeps](../../FORMAT.md#sweeps) — where a sweep may stand, labels, zip and cross, points and coordinates.
- [saves and the output](../../FORMAT.md#saves-and-the-output) — the files a run writes, the manifest, and the rows and columns of a table.
- [the worked document](../../FORMAT.md#a-worked-document) — `documents/patching.json` in full.

## Commands and Python

- [cli](cli.md) — Every clab command and flag (run, show, table, describe), with defaults, exit statuses and refusals.
- [python-api](python-api.md) — The public Python surface: clab.run, Plan and its methods, engines.load and execute, results and run, describe.

## Words

- [glossary](glossary.md) — clab's words, each in one sentence with the page that explains it.
