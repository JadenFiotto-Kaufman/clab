---
title: Architecture
one_liner: "How nnspec is built: the chain from document to output directory, the import direction, one concept per module, and where the code holds the five constraints that keep a document running the same everywhere."
tags: [developing, architecture]
related: [docs/developing/compile-and-run.md, docs/developing/engines.md, docs/developing/extending.md, docs/concepts/one-session.md, docs/concepts/plan-lifecycle.md]
sources: [nnspec/__init__.py, nnspec/plan.py, nnspec/format/__init__.py, nnspec/compile/__init__.py, nnspec/compile/state.py, nnspec/steps/__init__.py, nnspec/engines/__init__.py, nnspec/engines/nnterp.py, nnspec/engines/vllm.py, nnspec/output/__init__.py, nnspec/docs.py, nnspec/cli.py]
---

# Architecture

nnspec is a pipeline over one piece of data, the `Plan`: a JSON document becomes a plan, compile resolves it without a model, an engine runs it, and the filled plan is written to a directory. This page maps the stages to modules, states the direction imports run, and shows where the code holds the constraints that let one document run the same on every engine.

## The chain

```text
document.json
  Plan(path)              nnspec/plan.py       the records, validated as written
  plan.compile()          nnspec/compile/      names, sweeps, rows, bundles, axes, who serves whom; no model
  load(name, plan.model)  nnspec/engines/      the checkpoint loaded and placed, or a shell that loads it in its run
  engine.execute(plan)    nnspec/engines/      start_run's checks, then the run, in one session where the engine opens one
    steps.run(...)        nnspec/steps/        rows in view, preflight with the model, then every step at each point that runs
  plan.dump(out)          nnspec/output/       one file per save, manifest.json and tokens.json
  Plan.load(out), table   nnspec/output/       the directory read back as a plan, and as tables
```

`nnspec/__init__.py` `run` is that sequence up to `dump` in one call, and `nnspec/cli.py` is a thin command line over it, `Plan.compile` and `Plan.load`. Every arrow hands on the same pydantic model in a later state: uncompiled, compiled (`compiled` true, the derived fields filled), then filled (`results` and `run` set). Compile and preflight write what they decide into the plan's own records: each value's labelled layout and the sweep axes it depends on (`nnspec/plan.py` `Labelled`), and each forward's `schedule`. What else preflight learns for the run is one `Findings` (`nnspec/compile/state.py` `Findings`), and every value lives in a `ChainMap` local to `steps.run`, under its name and its coordinates on its own axes.

Two registries sit beside the chain: `nnspec/ops/` (every op, a plain function over tensors) and `nnspec/featurizers/` (every featurizer kind, one class each). The plan validates documents against them, compile lays values out with them, the run calls them, and `nnspec/docs.py` reads them to print `nnspec describe` and generate [FORMAT.md](../../FORMAT.md).

## Import direction

From `grep -rn '^from nnspec\|^import nnspec' nnspec/`, each layer imports only the layers above it:

| layer | imports from nnspec |
|---|---|
| `nnspec/format/` | nothing |
| `nnspec/featurizers/` | format |
| `nnspec/ops/` | format, featurizers |
| `nnspec/plan.py` | format, featurizers, ops |
| `nnspec/compile/` | format, featurizers, ops, plan |
| `nnspec/steps/` | format, featurizers, ops, plan, compile |
| `nnspec/engines/` | format, plan, steps |
| `nnspec/output/` | format, featurizers, ops, plan, compile |
| `nnspec/docs.py` | format, featurizers, ops, plan, compile, output |
| `nnspec/cli.py`, `nnspec/__init__.py` | plan, docs (the CLI also format, compile and output) |

The chain runs format, featurizers, ops, plan, compile, steps, engines; output sits beside steps (it reads and writes a plan and never runs one), with docs and the entry points on top. A reference that points back up the chain is deferred, so the direction holds at import time:

- A `Plan` method that belongs to a later stage imports it inside the method: `Plan.compile` imports `nnspec.compile`; `Plan.dump`, `Plan.load` and `Plan.table` import `nnspec.output`. A `load` step reads a run's directory through output's one reader, imported inside the functions that read it: `nnspec/compile/__init__.py` `loaded_run`, its manifest, and `nnspec/plan.py` `loaded_value`, its value.
- The entry points import the engines inside the function that loads one: `run` in `nnspec/__init__.py`; `main` in `nnspec/cli.py`; `describe`, `markdown` and `run_options` in `nnspec/docs.py`.
- A type needed only in annotations is imported under `TYPE_CHECKING`: the `Engine` that `compile` and `steps` functions take, the `Kind` in `nnspec/format/registry.py`.

So importing nnspec, building a plan and compiling it loads neither nnsight, nnterp nor any engine; none of the three is in `sys.modules` after `Plan("documents/patching.json").compile()`. A new import that points up the chain goes inside the function that needs it, or under `TYPE_CHECKING` when only the type is needed.

## One concept per module

Each module owns one concept and opens with a docstring that teaches it. A function goes in the module of the concept it serves; a new concept gets a new module.

| concept | module |
|---|---|
| components, their layouts and what narrows them | `nnspec/format/address.py` |
| selectors (layers, calls, occurrences, keys) | `nnspec/format/selectors.py` |
| a function's signature as the document's contract | `nnspec/format/registry.py` |
| sweeps, points and the run's state by point | `nnspec/format/sweeps.py` |
| datasets, encoding and padding | `nnspec/format/data.py` |
| the records, uncompiled and compiled | `nnspec/plan.py` |
| the compile driver, names and scoping | `nnspec/compile/__init__.py` |
| what compile carries step to step, and what preflight hands the run | `nnspec/compile/state.py` (`Compiling`, `Findings`) |
| everything checked with the model | `nnspec/compile/preflight.py` |
| `pos` resolved per row | `nnspec/compile/positions.py` |
| a column scored against a vocabulary | `nnspec/compile/answers.py` |
| an op's arguments against what compile knows | `nnspec/compile/arguments.py` |
| entries, their brackets and operands | `nnspec/compile/interventions.py` |
| the forward as a compiled schedule: which points one forward serves, the sites it visits and the records at each | `nnspec/compile/schedule.py` |
| a featurizer's uses, widths and bundle header | `nnspec/compile/uses.py` |
| the run's walk, forward and op steps | `nnspec/steps/__init__.py` |
| batches, a schedule bound to one, and a site's visit | `nnspec/steps/batches.py` |
| the generate step | `nnspec/steps/generate.py` |
| the fit step | `nnspec/steps/fit.py` |
| the engine protocol | `nnspec/engines/__init__.py` |
| one engine each | `nnspec/engines/nnterp.py`, `nnspec/engines/hooks.py`, `nnspec/engines/ndif.py`, `nnspec/engines/vllm.py` |
| what the engines share | `nnspec/engines/standard.py` |
| one op group each | `nnspec/ops/metrics.py`, `nnspec/ops/reductions.py`, `nnspec/ops/edits.py`, `nnspec/ops/transforms.py` |
| one featurizer kind each | one module per kind under `nnspec/featurizers/`, `nnspec/featurizers/subspace.py` to `nnspec/featurizers/constant.py` |
| bundle files and headers | `nnspec/featurizers/bundles.py` |
| the output directory | `nnspec/output/__init__.py` |
| saved values as tables | `nnspec/output/table.py` |
| the format's tables and FORMAT.md | `nnspec/docs.py` |

## Where the code holds the five constraints

The constraints themselves, and why a document needs them to run the same locally and on a server, are in [concepts/one-session.md](../concepts/one-session.md). This is where each is enforced, and what a change must keep.

- **[One session per run](../concepts/one-session.md#one-session-per-run-and-remote-is-the-only-difference).** `nnspec/engines/nnterp.py` `NnterpEngine.execute` opens `self.model.session(remote=self.remote)` once and calls `steps.run` inside it. It saves the one dict that comes home ([one-session.md](../concepts/one-session.md#the-idea) says what it holds).
  - The ndif engine (`nnspec/engines/ndif.py` `NdifEngine`) inherits that `execute` and changes only how the model is built (a weightless shell) and `remote`.
  - The hooks engine calls the same `steps.run` in plain Python.
  - The vllm engine (`nnspec/engines/vllm.py` `VllmEngine.execute`) calls it in plain Python too, each batch one trace that is one request to vLLM's worker, outside any session. Inside one, nnsight would mark what a trace saved by object id in a set the session keeps across its traces, and a later batch's objects could be taken for saved.
  - A new step or check runs inside `steps.run`, never as a second trace or session.
- **[The plan is pure data](../concepts/one-session.md#the-compiled-plan-is-pure-data).** `Plan` and every record are pydantic models (`nnspec/plan.py` `Record` refuses unknown fields). None of its methods touches a model, and a plan pickles and comes back equal (`tests/test_plan.py` `test_a_plan_pickles_and_rebuilds_from_its_dump`). A method added to `Plan` stays a view.
- **[Nothing decided from a tensor on the client](../concepts/one-session.md#nothing-is-decided-from-a-tensor-on-the-client).** Compile reads only the document, the data files and bundle headers.
  - Everything that needs the tokenizer or a size of the model is `nnspec/compile/preflight.py` `preflight`. `nnspec/steps/__init__.py` `run` calls it before any step, inside the session: on the server, under ndif.
  - Before any session opens, every engine's `execute` calls `nnspec/steps/__init__.py` `start_run`. It checks what the plan and the engine's declarations decide, `gradless` and `LOADED` among them ([engines.md](engines.md#execute) lists the checks).
  - The vllm engine's `execute` refuses what vLLM cannot do before `start_run`, from the plan and the checkpoint's generation config.
  - A check from the plan and the engine's declarations alone goes in `start_run`. A check that needs the model goes in preflight, never in compile or before the session. [compile-and-run.md](compile-and-run.md#where-a-check-belongs) has every seam.
- **[The engine is the only client object](../concepts/one-session.md#the-engine-and-its-model-are-the-only-client-side-objects-in-the-session).** An engine holds its model and its configuration and nothing else (`nnspec/engines/__init__.py` `Engine`: `name`, `spec`, `model`, `device`, `options`, `remote`, `stops`, `cuda`). The session body names the engine, the plan and the run's settings. The run's cross-step state is a dict that lives in `steps.run`'s frame and never leaves it.
- **[Nothing ships by value](../concepts/one-session.md#nothing-of-nnspec-ships-by-value-it-resolves-by-import-on-the-server).** nnspec is installed where the session runs, so its modules travel by reference. `tests/conftest.py` `stand_in` serializes the request a real ndif run makes and runs it against a loaded model. `by_reference` checks that the request names `nnspec.plan`, `nnspec.engines.ndif` and `nnspec.steps` and carries no class or function by value; tests compare each trip to `TRIP`. Code the session body reaches lives in an installed nnspec module.

How compile, preflight and the run divide the work is [compile-and-run.md](compile-and-run.md); what an engine must do is [engines.md](engines.md).
