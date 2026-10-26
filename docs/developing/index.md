---
title: Developing Index
one_liner: "How clab is built and how to change it: architecture, compile and run, extending, the engine protocol, testing and contributing."
tags: [developing, index]
related: [docs/index.md, docs/concepts/index.md]
---

# Developing Index

Up: [docs/index.md](../index.md) — all sections.

For changing clab itself: read architecture first, then the page for the part you are changing.

## How clab is built

- [architecture](architecture.md) — How clab is built: the chain from format to output, the import direction, one concept per module, and the five constraints that keep a document running the same everywhere.
- [compile-and-run](compile-and-run.md) — The two passes and the run: compile's checks with the document alone, preflight's with the model, how steps.run walks a plan and visits sites, and what belongs to the engine.
- [engines](engines.md) — The engine protocol and how to add an engine: what execute must do, what steps.run asks for, records by site, live operands, and refusing by name.

## Changing it

- [extending](extending.md) — Adding an op, a featurizer kind, a component or a step kind: the decorator, the signature as the document's contract, where the tables come from, and the tests to add.
- [testing](testing.md) — Running and extending the suite: offline CPU runs, snapshot pins and re-pinning, the GPU real-model tests, the FORMAT drift test, the docs test, and pyright.
- [contributing](contributing.md) — How changes land: branch and commit conventions, format changes with before-and-after JSON, one spelling per thing, the corpus runs unchanged, regenerating FORMAT.md, and the docs standard.
