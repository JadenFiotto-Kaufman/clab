---
title: Compile and Run
one_liner: "The two passes and the run: compile's checks with the document alone, preflight's with the model, how steps.run walks a plan and visits sites, and what belongs to the engine."
tags: [developing, compile, preflight, run]
related: [docs/developing/architecture.md, docs/developing/engines.md, docs/concepts/plan-lifecycle.md, docs/errors/index.md]
sources: [clab/compile/__init__.py, clab/compile/preflight.py, clab/compile/state.py, clab/compile/interventions.py, clab/steps/__init__.py, clab/steps/batches.py, clab/steps/generate.py, clab/steps/fit.py, clab/plan.py]
---

# Compile and Run

A document is checked twice before anything runs: compile, with the document, its data files and bundle headers alone; and preflight, inside the run, with the model in hand. Then `steps.run` walks the plan step by step. This page says where each piece of that lives, so a new check or a new step lands in the right place. The stages from the user's side, and which refuses what, are [concepts/plan-lifecycle.md](../concepts/plan-lifecycle.md).

## Where a check belongs

Decide everything as early as the information allows.

| what it needs | where it runs | how it refuses |
|---|---|---|
| a record's own shape: fields, types, an op's arguments against its signature, a featurizer's fields | building the plan: the validators in `clab/plan.py` | `PlanError`, starting with the document path |
| the document, the data files, bundle headers | compile, `clab/compile/` | `fail(path, message)` in `clab/plan.py`: a `PlanError` starting with the document path |
| the tokenizer, a size of the model, what the engine serves, forward order | preflight, `clab/compile/preflight.py` | `ValueError` starting with `<step>.<entry>` |

Two refusals of the patching document, one from each pass:

```text
steps.iia.of: 'patched.nothing' names no value produced before this step; known: counterfactual.v_cf, patched.logits, patched.patch
counterfactual.v_cf: layer 5; the model has 2 layers
```

The first is compile's: `of` names a value no step produced, and the message starts with the document path. The second is preflight's: `"layers": 5` compiles, because compile does not know the model has two layers, and the refusal names the entry. A refusal inside a fit's body is prefixed with the fit's name, and under an eval with the eval and the datasets it reads. `PlanError` is a `ValueError`, so a caller catches both as one, and `clab run` prints either as `clab: <message>` and exits 1.

## Compile

`clab/compile/__init__.py` `compile` returns a new plan with `compiled` true. In order:

1. Each dataset's rows are made once (`dataset`: a file's rows in its `split`, inline rows, or a hub split; then `shuffle`, `take` and `columns`), and their digest and row count recorded. A hub dataset's rows are checked and left out of the plan; the run loads them again where it runs.
2. Sweeps are found, and each featurizer, then each step, is compiled at every point of the axes it uses (`expand`). A featurizer's bundle is loaded and checked key by key against the document (`featurizer`).
3. Steps compile in document order (`step` dispatches by kind). Each name a step produces is registered in `Compiling.names` as a `Known` (`clab/compile/state.py` `Known`: its layout, the dataset its rows align to, the coordinates it is in, its width). That table is how a later reference resolves and how every value's layout is known before any forward; a name is referenced only after it exists.
4. Saves are checked, and each entry is marked `gathered` when something uses its value (`gathered`): a save, an op, or a fit objective's scalar. The run takes off the model only what is gathered.
5. Sites and named interventions are inlined; the compiled plan holds neither.

The concepts compile resolves each have a module with their checks at compile and what they resolve to in the run: `clab/compile/answers.py`, `clab/compile/positions.py`, `clab/compile/arguments.py`, `clab/compile/interventions.py` and `clab/compile/uses.py`. A new check joins the module of its concept.

## Preflight

`clab/compile/preflight.py` `preflight` is the first thing `steps.run` does, so under the ndif engine it runs on the server. It:

- refuses every component the plan reads or writes that the engine does not serve, as the engine's `unserved` says; `clab/compile/interventions.py` `refuse` does it;
- per step (`prepare`): encodes each input once and records its tokens in `plan.run.tokens`; checks each entry's `layers` and `select` against `engine.sizes()`; resolves its positions per row; checks token columns against the tokenizer and each op's layers and labels; and checks each write's operands (`check_operands`), including that an operand read earlier in the same forward is at a site the engine reaches first, by `engine.rank`;
- finds each featurizer's width at its uses (`clab/compile/uses.py` `widths`), and prepares each fit's body over its training rows and under each eval.

It returns a `Found` (`clab/compile/state.py` `Found`): encoded inputs and widths, positions per row, resolved `select` indices, answers, featurizer widths and fit bodies. Everything the run needs from the model is in it.

## The run

`clab/steps/__init__.py` `run`:

1. `held`: the plan with every dataset's rows in it, a hub dataset's loaded here and refused unless their digest is the one compile recorded.
2. The versions and the model's layer count go into `plan.run`; then `preflight`.
3. `made`: each declared featurizer is made where the model is.
4. Under `torch.no_grad()`, every step and every point in document order, each through `KINDS[step.kind](engine, plan, name, step, state, batch_size, found)`.
5. Each saved value goes into `plan.results` as a detached CPU clone (`finished`); positions, tokens, dtypes and devices go into `plan.run`.

`state` is a plain dict in `run`'s frame. `put` and `lookup` (`clab/format/sweeps.py`) keep each value under its name by its point's coordinates, so a consumer finds the producer's point that agrees with its own.

The step kinds:

- **forward** (`clab/steps/__init__.py` `forward`): the input in batches, padded on the left to the input's longest row (`clab/steps/batches.py` `chunked`). For each batch, `bind` makes the records: a write per layer an entry is made at, its ops bound to tensors, then a read per layer for a gathered entry, named `<entry>@<layer>`. `engine.forward` does them; `read` makes positions past a row's own NaN and runs the ops after the entry's last edit (`with_ops`); `stacked` puts a `layers` axis first; `joined` joins the batches along rows.
- **op** (`op`): `ops.call` once over the whole value `of` names, its operands bound and aligned to `of`'s layers by label.
- **generate** (`clab/steps/generate.py` `generate`): prefill entries bind as a forward's; an entry past the prefill goes to the engine as records with `calls` (and `online` for a write scoped to text), and its positions resolve after the generation from what each row said.
- **fit** (`clab/steps/fit.py` `fit`): trains its featurizers over minibatches, running the body's steps through the same `KINDS` table (`body_run`) with gradients for the featurizers' tensors alone, then scores the body under each eval.

## What belongs to the engine

The engine loads and places the model, runs one batch through it (a forward, or a generation), and at each site the records name, in its own forward order, hands the value to `clab/steps/batches.py` `visit` and sets back what that returns. It also answers what preflight and the steps ask: its tokenizer, sizes, EOS ids, readout, what it does not serve, and where a component falls in its forward.

Everything else is `clab/steps/` in plain torch, the same code for every engine: padding, positions, binding operands, op order, NaN for missing positions, joins, metrics, featurizers and the fit loop. That is why the nnterp and hooks engines give the same results to the bit on every corpus document they both serve (`tests/test_documents.py` `test_the_engines_agree_to_the_bit`). The protocol in full is [engines.md](engines.md).
