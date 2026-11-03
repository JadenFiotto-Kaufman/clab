---
title: Python API
one_liner: "The public Python surface: clab.run, Plan and its methods, engines.load and execute, results and run, describe."
tags: [reference, python, api]
related: [docs/usage/running.md, docs/usage/output.md, docs/reference/cli.md, docs/concepts/plan-lifecycle.md, docs/developing/engines.md, FORMAT.md]
sources: [clab/__init__.py, clab/plan.py, clab/engines/__init__.py, clab/output/__init__.py, clab/output/table.py, clab/docs.py]
---

# Python API

`import clab` gives `clab.run`, `clab.Plan` and `clab.describe`; `clab.engines.load` loads an engine. This page lists their signatures and what each returns and refuses. When to call which is [running.md](../usage/running.md); reading results is [output.md](../usage/output.md).

## `clab.run`

```python
clab.run(plan, engine="nnterp", out=None, batch_size=8, device=None, padding="dataset", readout="full", **options) -> Plan
```

| parameter | meaning |
|---|---|
| `plan` | a document's path, or a `Plan`, compiled or not |
| `engine` | a registered engine's name, or an engine `load` returned |
| `out` | the output directory; default `out/<document name>` under the current directory (`out/plan` for a plan built from a dict) |
| `batch_size` | rows per forward or generation call |
| `device` | where a named engine places the model; default `"cpu"` |
| `padding` | how each batch is padded: `"dataset"` (the exact default), `"batch"` or `"sorted"` ([running.md](../usage/running.md#batch-size-and-reproducibility)); any other is refused before the model loads |
| `readout` | how a forward computes the logits it reads: `"full"` (the exact default) or `"at-reads"` ([running.md](../usage/running.md#the-head-at-the-positions-read)); any other is refused before the model loads |
| `**options` | the named engine's own options, such as `remote="local"` for `ndif` or `tp=2` for `nnterp` |

It compiles the plan, loads the engine when given a name, executes, dumps into `out`, and returns the filled plan. A loaded engine takes no `device` or options: `the loaded nnterp engine runs on cpu with {'tp': 1}; device and options are given when an engine loads`. Under torchrun (`tp`) every rank runs and returns the filled plan, and rank 0 alone writes `out`. A step that runs out of CUDA memory stops the run: `out` is written with the steps that finished and the refusal in `plan.run.stopped`, then the refusal is raised as a `ValueError` ending `the steps before it are saved in <out>` ([running.md](../usage/running.md#when-a-step-runs-out-of-memory)).

## `clab.Plan`

```python
Plan(path)        # read the JSON document at path
Plan(**fields)    # build one from the document's keys as Python values
```

`Plan(path)` records the file's absolute path as `plan.document`; its data paths and bundles resolve against the corpus root above it. A plan built from fields has no document, so its name is `plan` and it resolves against the working directory. Both raise `PlanError` on a malformed document.

| method | does |
|---|---|
| `compile() -> Plan` | a new, compiled plan: rows loaded, sites and bundles of entries inlined, sweeps expanded into points, names resolved. A compiled plan returns itself. Raises `PlanError` |
| `dump(out)` | writes the output directory: one file per save, `manifest.json` and `tokens.json` (the rows' tokens, `run.tokens`). A plan that has not run raises `ValueError` (`the plan has not run, so it has no results; execute it first`) |
| `Plan.load(out)` | classmethod: the filled plan an output directory holds, with `results` and `run` |
| `table(*names, columns=None) -> list[dict]` | the rows `clab table` prints, one dict per row. `columns` lists the dataset columns to include; by default, those the values' steps used. A plan that has not run raises `ValueError`, as `dump` does |
| `write(path)` | writes the plan as a JSON document in its current state, without `results`, `run` and `document`, defaults left out |

Besides the document's own keys (`description`, `model`, `data`, `sites`, `interventions`, `featurizers`, `steps`, `saves`), a plan holds:

| attribute | holds |
|---|---|
| `document` | the document file's path: absolute when read from a file, relative to the corpus root in a manifest, `None` when built from fields |
| `compiled` | whether this is the compiled plan |
| `sweeps` | each sweep axis by label, as `Axis(values)`, in axis order: `plan.sweeps["dose"].values[point.at["dose"]]` is a point's value on that axis |
| `results` | each saved value by name: a tensor, a dict of tensors, or a list or dict where it holds text. A value that depends on sweep axes is a dict keyed by its coordinate on those axes alone (the axis's value, a tuple of values over several); one that depends on none is the value itself, whatever its step's sweeps |
| `run` | a `Run` once the plan has run, else `None`: `engine`, `batch_size`, `device`, `options`, `versions`, `dtype`, `devices`, `sizes`, `started`, `seconds`, `tokens`, `costs` and `stopped`, described in [output.md](../usage/output.md#the-run-record) |

A swept step is a list of points in `plan.steps`, each a whole step with its `at`. The fields compile and preflight fill on each record (a value's `dims`, labels and `axes`, an entry's `positions` per row under its dataset, a forward's or generation's `schedule`, a step's `dataset` and `at`, a dataset's `sha256` and `n_rows`) are listed in [FORMAT.md](../../FORMAT.md#derived-fields); where a read landed is `plan.steps["patched"].interventions["logits"].positions`, `{"pairs": [[10], [8], [8], [8]]}` on `documents/patching.json`.

A plan's `repr` says its stage: `Plan(patching, uncompiled)`, `Plan(patching, compiled, 1 point)`, `Plan(patching, run on nnterp, 2 results)`. `str` and `print` give the same one line; the records are `plan.steps`, `plan.featurizers` and `plan.model_dump()`.

## Engines

```python
from clab.engines import ENGINES, load

engine = load(name, model, device="cpu", **options)
filled = engine.execute(plan, batch_size=8, padding="dataset", readout="full")
```

`ENGINES` maps each registered engine's name to its class: `nnterp`, `hooks`, `ndif`.

`load` builds engine `name` for `model`, a plan's model block (`compiled.model`), on `device`, with the engine's own `options`, and loads the checkpoint. It refuses, before anything loads, an unknown engine (`unknown engine 'bogus'; valid: nnterp, hooks, ndif`), an option the engine does not declare, and one of the wrong type (`remote: 'yes' is not one of True, 'local'`).

| engine attribute | holds |
|---|---|
| `name` | the name it registered under |
| `spec` | the model block it loaded |
| `model` | the loaded model |
| `device` | the device asked for; `None` when an NDIF server places the model |
| `options` | its options after their defaults (`{"tp": 1}`, `{"remote": "local"}`) |
| `stops` | whether it ends a forward right after the schedule's stop site (nnterp and hooks do; ndif runs every forward whole) |
| `tokenizer` | the checkpoint's tokenizer |
| `sizes()` | the model's sizes (`num_layers`, `hidden_size`, `num_heads`, ...), which `plan.run.sizes` records |

`execute(plan, *, batch_size=8, padding="dataset", readout="full")` runs a compiled plan, its batches padded as `padding` says and its forwards' logits computed as `readout` says: preflight, then every step. It fills the plan's `results` and `run` in place and returns the same plan. It refuses a plan that is not compiled (`plan is not compiled; call plan.compile()`) and one whose model block differs from the one the engine loaded, naming each field that differs (`the plan's model is not the one this engine loaded: key 'hf-internal-testing/tiny-random-gpt2' where the engine's is 'hf-internal-testing/tiny-random-LlamaForCausalLM', revision …`). One engine executes any number of plans that pin its model block. A step that runs out of CUDA memory fills `results`, `run.costs` and `run.stopped` for the steps that finished and raises the refusal. Writing an engine is [developing/engines.md](../developing/engines.md).

## `clab.describe`

```python
clab.describe(what=None) -> str
```

The text `clab describe` prints: `what` is one of `records`, `steps`, `ops`, `components`, `featurizers`, `pos`, `engines`, or `None` for all of them ([cli.md](cli.md#clab-describe)).

## Errors

Every refusal is a `ValueError`. Those raised while reading or compiling a document are `PlanError`, its subclass in `clab/plan.py`, and their message starts with the document path of what is wrong: `steps.iia.of: 'patched.logits' names no value produced before this step; known: counterfactual.v_cf`. Refusals after the model loads name the step or entry instead (`counterfactual.v_cf: layer 5; the model has 2 layers`). Under the `ndif` engine with a real server, a refusal raised on the server arrives as nnsight's `RemoteError`. Messages and their fixes: [errors/index.md](../errors/index.md).
