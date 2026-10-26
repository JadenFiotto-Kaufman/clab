---
title: Concepts Index
one_liner: "Why clab behaves as it does: the plan's lifecycle, one session per run, the position model, values and layout, names and sweeps."
tags: [concepts, index]
related: [docs/index.md, docs/usage/index.md]
---

# Concepts Index

Up: [docs/index.md](../index.md) — all sections.

Read one of these when a usage page's rule surprises you; each explains a mechanism the usage pages rely on.

## Running a plan

- [plan-lifecycle](plan-lifecycle.md) — What happens between a document and its results: Plan, compile with the document alone, preflight with the model, the run inside one session, results and the output directory, and which stage refuses what.
- [one-session](one-session.md) — Why a document runs the same locally and on a server: one session per run, the plan as pure data, nothing decided from a tensor on the client, the engine as the only client object, and what ships by reference.

## Positions, values and names

- [positions](positions.md) — The position model: every input token is a position, a generation is a timeline of calls, where a write lands, scoped writes found online, static widths versus ragged rows, and why some rows are alignment_missing.
- [values-and-layout](values-and-layout.md) — How a value is laid out and compared: rows and positions first, the layers axis and alignment by label, select keeping an axis, missing entries as NaN, dtype rules, and why batch shape moves the last bits.
- [names-and-sweeps](names-and-sweeps.md) — How names resolve and sweeps multiply: one flat namespace per scope, a fit's own scope, bundles as templates, sweep labels as axes, points, coordinates, and how a consumer finds the producer's point.
