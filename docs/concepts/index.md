---
title: Concepts Index
one_liner: "Why nnspec behaves as it does: the plan's lifecycle, one session per run, the position model, values and layout, names and sweeps."
tags: [concepts, index]
related: [docs/index.md, docs/usage/index.md]
---

# Concepts Index

Up: [docs/index.md](../index.md) — all sections.

Read one of these when a usage page's rule surprises you; each explains a mechanism the usage pages rely on.

## Running a plan

- [plan-lifecycle](plan-lifecycle.md) — What happens between a document and its results, and which stage raised a refusal?
- [one-session](one-session.md) — Why does a document run the same locally and on an NDIF server, and what must the server have?

## Positions, values and names

- [positions](positions.md) — How does nnspec count positions: prompt tokens, generation calls, where a write lands, static and ragged widths?
- [values-and-layout](values-and-layout.md) — How is a value laid out and aligned: its axes, the layers axis, select, missing entries, dtypes, and batch shape?
- [names-and-sweeps](names-and-sweeps.md) — How do names resolve, and how do sweeps multiply into points?
