---
title: Architecture
one_liner: "How clab is built: the chain from format to output, the import direction, one concept per module, and the five constraints that keep a document running the same everywhere."
tags: [developing, architecture]
related: [docs/developing/compile-and-run.md, docs/developing/engines.md, docs/developing/extending.md, docs/concepts/one-session.md, docs/concepts/plan-lifecycle.md]
sources: [clab/__init__.py, clab/plan.py, clab/format/__init__.py, clab/compile/__init__.py, clab/steps/__init__.py, clab/engines/__init__.py, clab/engines/nnterp.py, clab/output/__init__.py, clab/docs.py, clab/cli.py]
---

# Architecture

clab is a pipeline over one piece of data, the `Plan`: a JSON document becomes a plan, compile resolves it without a model, an engine runs it, and the filled plan is written to a directory. This page maps the stages to modules, states the direction imports run, and shows where the code holds the constraints that let one document run the same on every engine.

## The chain

```text
document.json
  Plan(path)              clab/plan.py       the records, validated as written
  plan.compile()          clab/compile/      names, sweeps, rows, bundles resolved; no model
  load(name, plan.model)  clab/engines/      the checkpoint loaded and placed
  engine.execute(plan)    clab/engines/      the run opened, in one session
    steps.run(...)        clab/steps/        preflight with the model, then every step and point
  plan.dump(out)          clab/output/       one file per save, and manifest.json
  Plan.load(out), table   clab/output/       the directory read back as a plan, and as tables
```

`clab/__init__.py` `run` is that sequence up to `dump` in one call, and `clab/cli.py` is a thin command line over it and `Plan.load`. Every arrow hands on the same pydantic model in a later state: uncompiled, compiled (`compiled` true, the derived fields filled), then filled (`results` and `run` set). Inside the run, what preflight learns is one `Found` (`clab/compile/state.py` `Found`), and every value lives in a dict local to `steps.run`, under its name and its point's coordinates.

Two registries sit beside the chain: `clab/ops/` (every op, a plain function over tensors) and `clab/featurizers/` (every featurizer kind, one class each). The plan validates documents against them, compile lays values out with them, the run calls them, and `clab/docs.py` reads them to print `clab describe` and generate [FORMAT.md](../../FORMAT.md).

## Import direction

From `grep -rn '^from clab\|^import clab' clab/`, each layer imports only the layers above it:

| layer | imports from clab |
|---|---|
| `clab/format/` | nothing |
| `clab/featurizers/` | format |
| `clab/ops/` | format, featurizers |
| `clab/plan.py` | format, featurizers, ops |
| `clab/compile/` | format, featurizers, ops, plan |
| `clab/steps/` | format, featurizers, ops, plan, compile |
| `clab/engines/` | format, plan, steps |
| `clab/output/` | format, featurizers, ops, plan, compile |
| `clab/docs.py` | format, featurizers, ops, plan, compile, output |
| `clab/cli.py`, `clab/__init__.py` | plan, docs (the CLI also format and output) |

The chain runs format, featurizers, ops, plan, compile, steps, engines; output sits beside steps (it reads and writes a plan and never runs one), with docs and the entry points on top. A reference that points back up the chain is deferred, so the direction holds at import time:

- A `Plan` method that belongs to a later stage imports it inside the method: `Plan.compile` imports `clab.compile`; `Plan.dump`, `Plan.load` and `Plan.table` import `clab.output`.
- The entry points import the engines inside the function that loads one: `run` in `clab/__init__.py`; `main` in `clab/cli.py`; `describe` in `clab/docs.py`.
- A type needed only in annotations is imported under `TYPE_CHECKING`: the `Engine` that `compile` and `steps` functions take, the `Kind` in `clab/format/registry.py`.

So importing clab, building a plan and compiling it loads neither nnsight, nnterp nor any engine; none of the three is in `sys.modules` after `Plan("documents/patching.json").compile()`. A new import that points up the chain goes inside the function that needs it, or under `TYPE_CHECKING` when only the type is needed.

## One concept per module

Each module owns one concept and opens with a docstring that teaches it. A function goes in the module of the concept it serves; a new concept gets a new module.

| concept | module |
|---|---|
| components, their layouts and what narrows them | `clab/format/address.py` |
| selectors (layers, calls, occurrences, keys) | `clab/format/selectors.py` |
| a function's signature as the document's contract | `clab/format/registry.py` |
| sweeps, points and the run's state by point | `clab/format/sweeps.py` |
| datasets, encoding and padding | `clab/format/data.py` |
| the records, uncompiled and compiled | `clab/plan.py` |
| the compile driver, names and scoping | `clab/compile/__init__.py` |
| everything checked with the model | `clab/compile/preflight.py` |
| `pos` resolved per row | `clab/compile/positions.py` |
| a column scored against a vocabulary | `clab/compile/answers.py` |
| an op's arguments against what compile knows | `clab/compile/arguments.py` |
| entries, their brackets and operands | `clab/compile/interventions.py` |
| a featurizer's uses, widths and bundle header | `clab/compile/uses.py` |
| the run's walk, forward and op steps | `clab/steps/__init__.py` |
| batches, records and a site's visit | `clab/steps/batches.py` |
| the generate step | `clab/steps/generate.py` |
| the fit step | `clab/steps/fit.py` |
| the engine protocol | `clab/engines/__init__.py` |
| one engine each | `clab/engines/nnterp.py`, `clab/engines/hooks.py`, `clab/engines/ndif.py` |
| what the engines share | `clab/engines/standard.py` |
| one op group each | `clab/ops/metrics.py`, `clab/ops/reductions.py`, `clab/ops/edits.py`, `clab/ops/transforms.py` |
| one featurizer kind each | `clab/featurizers/subspace.py`, ... `clab/featurizers/lora.py` |
| bundle files and headers | `clab/featurizers/bundles.py` |
| the output directory | `clab/output/__init__.py` |
| saved values as tables | `clab/output/table.py` |
| the format's tables and FORMAT.md | `clab/docs.py` |

## Where the code holds the five constraints

The constraints themselves, and why a document needs them to run the same locally and on a server, are in [concepts/one-session.md](../concepts/one-session.md). This is where each is enforced, and what a change must keep.

- **One session per run.** `clab/engines/nnterp.py` `NnterpEngine.execute` opens `self.model.session(remote=self.remote)` once, calls `steps.run` inside it, and saves one dict holding `results` and `run`, which is all that comes home. The ndif engine (`clab/engines/ndif.py` `NdifEngine`) inherits that `execute` and changes only how the model is built (a weightless shell) and `remote`. The hooks engine calls the same `steps.run` in plain Python. A new step or check runs inside `steps.run`, never as a second trace or session.
- **The plan is pure data.** `Plan` and every record are pydantic models (`clab/plan.py` `Record` refuses unknown fields). Its methods are views and serialization (`compile` returns a new plan; `dump`, `load`, `write`, `table`), none touches a model, and a plan pickles and comes back equal (`tests/test_plan.py` `test_a_plan_pickles_and_rebuilds_from_its_dump`). A method added to `Plan` stays a view.
- **Nothing decided from a tensor on the client.** Compile reads only the document, the data files and bundle headers. Everything that needs the tokenizer or a size of the model is `clab/compile/preflight.py` `preflight`, which `clab/steps/__init__.py` `run` calls first, inside the session: on the server, under ndif. Before the session opens, `execute` only checks that the plan is compiled and pins the model block the engine loaded (`clab/steps/__init__.py` `begin`). A check that needs the model goes in preflight, never in compile or before the session.
- **The engine is the only client object.** An engine holds its model and its configuration and nothing else (`clab/engines/__init__.py` `Engine`: `name`, `spec`, `model`, `device`, `options`). The session body names the engine, the plan and the batch size; the run's cross-step state is a dict that lives in `steps.run`'s frame and never leaves it.
- **Nothing ships by value.** clab is installed where the session runs, so its modules travel by reference. `tests/conftest.py` `stand_in` serializes the request a real ndif run makes, runs it against a loaded model, and `by_reference` checks that it names `clab.plan`, `clab.engines.ndif` and `clab.steps` and carries no class or function by value; tests compare each trip to `TRIP`. Code the session body reaches lives in an installed clab module.

How compile, preflight and the run divide the work is [compile-and-run.md](compile-and-run.md); what an engine must do is [engines.md](engines.md).
