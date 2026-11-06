---
title: Python API
one_liner: "The public Python surface: nnspec.run, Plan and its methods, engines.load and execute, results and run, describe."
tags: [reference, python, api]
related: [docs/usage/running.md, docs/usage/output.md, docs/reference/cli.md, docs/concepts/plan-lifecycle.md, docs/developing/engines.md, FORMAT.md]
sources: [nnspec/__init__.py, nnspec/plan.py, nnspec/engines/__init__.py, nnspec/output/__init__.py, nnspec/output/table.py, nnspec/docs.py, nnspec/cli.py]
---

# Python API

`import nnspec` gives `nnspec.run`, `nnspec.Plan` and `nnspec.describe`; `nnspec.engines.load` loads an engine, and `nnspec.output.manifest` reads a run's manifest. This page lists their signatures and what each returns and refuses. When to call which is [running.md](../usage/running.md); reading results is [output.md](../usage/output.md).

## `nnspec.run`

```python
nnspec.run(plan, engine="nnterp", out=None, batch_size=8, device=None, padding="dataset", readout="full", **options) -> Plan
```

| parameter | meaning |
|---|---|
| `plan` | a document's path, or a `Plan`, compiled or not |
| `engine` | a registered engine's name, or an engine `load` returned |
| `out` | the output directory; default `out/<document name>` under the current directory (`out/plan` for a plan built from fields) |
| `batch_size` | rows per forward or generation call |
| `device` | where a named engine places the model; default `None`, the engine's own: `cpu`, or `cuda:0` under `vllm` |
| `padding` | how each batch is padded: `"dataset"`, the default, the reference the others are compared against; `"batch"`; or `"sorted"` ([running.md](../usage/running.md#batch-size-and-padding)). Any other is refused before the model loads |
| `readout` | how a forward computes the logits it reads: `"full"`, the default, the reference the others are compared against; or `"at-reads"` ([running.md](../usage/running.md#the-head-at-the-positions-read)). Any other is refused before the model loads |
| `**options` | the named engine's own options, such as `remote="local"` for `ndif` or `tp=2` for `nnterp` |

It compiles the plan, loads the engine when given a name, executes, dumps into `out`, and returns the filled plan. A loaded engine takes no `device` or options: `the loaded nnterp engine runs on cpu with {'tp': 1}; device and options are given when an engine loads`. Under torchrun (`tp`) every rank runs and returns the filled plan, and rank 0 alone writes `out`. A step that runs out of CUDA memory stops the run: `out` is written with the steps that finished and the refusal in `plan.run.stopped`, then the refusal is raised as a `ValueError` ending `the steps before it are saved in <out>` ([running.md](../usage/running.md#when-a-step-runs-out-of-memory)).

## `nnspec.Plan`

```python
Plan(path)        # read the JSON document at path
Plan(**fields)    # build one from the document's keys as Python values
```

`Plan(path)` records the file's absolute path as `plan.document`; its data paths and bundles resolve against the corpus root above it. A plan built from fields has no document, so its name is `plan` and it resolves against the working directory. `Plan(fields)`, a dict passed whole, is refused with `TypeError` ([below](#errors)).

| method | does |
|---|---|
| `compile() -> Plan` | a new, compiled plan: rows loaded, sites and bundles of entries inlined, sweeps expanded into points, names resolved. A compiled plan returns itself. Raises `PlanError` |
| `dump(out)` | writes the output directory: one file per save, `manifest.json` and `tokens.json` (the rows' tokens, `run.tokens`). A plan that has not run raises `ValueError` (`the plan has not run, so it has no results; execute it first`) |
| `Plan.load(out)` | classmethod: the filled plan an output directory holds, with `results` and `run` |
| `table(*names, columns=None) -> list[dict]` | the rows `nnspec table` prints, one dict per row. `columns` lists the dataset columns to include; by default, those the values' steps used. A plan that has not run raises `ValueError`, as `dump` does |
| `write(path)` | writes the plan as a JSON document in its current state, without `results`, `run` and `document`, defaults left out |

Besides the document's own keys (`description`, `model`, `data`, `sites`, `interventions`, `featurizers`, `steps`, `saves`), a plan holds:

| attribute | holds |
|---|---|
| `document` | the document file's path: absolute when read from a file, relative to the corpus root in a manifest, `None` when built from fields |
| `compiled` | whether this is the compiled plan |
| `sweeps` | each sweep axis by label, as `Axis(values)`, in axis order: `plan.sweeps["dose"].values[point.at["dose"]]` is a point's value on that axis |
| `results` | each saved value by name: a tensor, a dict of tensors, or a list or dict where it holds text. A value that depends on sweep axes is a dict keyed by its coordinate on those axes alone (the axis's value, a tuple of values over several). One that depends on none is the value itself, whatever its step's sweeps |
| `run` | a `Run` once the plan has run, else `None`: `engine`, `batch_size`, `padding`, `readout`, `device`, `options`, `versions`, `threads`, `dtype`, `devices`, `sizes`, `started`, `seconds`, `tokens`, `costs` and `stopped`, described in [output.md](../usage/output.md#the-run-record) |

A swept step is a list of points in `plan.steps`, each a whole step with its `at`. Compile and preflight fill fields on each record: a value's `dims`, labels and `axes`, an entry's `positions` per row under its dataset, a forward's or generation's `schedule`, a step's `dataset` and `at`, a dataset's `sha256` and `n_rows`. [FORMAT.md](../../FORMAT.md#derived-fields) lists them. Where a read landed is `plan.steps["patched"].interventions["logits"].positions`, `{"pairs": [[10], [8], [8], [8]]}` on `documents/patching.json`.

A plan's `repr` says its stage: `Plan(patching, uncompiled)`, `Plan(patching, compiled, 1 point)`, `Plan(patching, run on nnterp, 2 results)`. `str` and `print` give the same one line; the records are `plan.steps`, `plan.featurizers` and `plan.model_dump()`.

## Engines

```python
from nnspec.engines import ENGINES, load

engine = load(name, model, device=None, **options)
filled = engine.execute(plan, batch_size=8, padding="dataset", readout="full")
```

`ENGINES` maps each registered engine's name to its class: `nnterp`, `hooks`, `ndif`, `vllm`.

`load` builds engine `name` for `model`, a plan's model block (`compiled.model`), on `device`, with the engine's own `options`, and loads the checkpoint. `device=None` is the engine's own default: `cpu`, or `cuda:0` for `vllm`. It refuses, before anything loads, an unknown engine (`unknown engine 'bogus'; valid: nnterp, hooks, ndif, vllm`), an option the engine does not declare, and one of the wrong type (`remote: 'yes' is not one of True, 'local'`).

| engine attribute | holds |
|---|---|
| `name` | the name it registered under |
| `device` | the device it loaded on; `None` when an NDIF server places the model |
| `options` | its options after their defaults (`{"tp": 1}`, `{"remote": "local"}`) |
| `tokenizer` | the checkpoint's tokenizer |
| `sizes()` | the model's sizes (`num_layers`, `hidden_size`, `num_heads`, ...), which `plan.run.sizes` records |

The rest of an engine's surface is the protocol an engine implements ([developing/engines.md](../developing/engines.md)).

`execute(plan, *, batch_size=8, padding="dataset", readout="full")` runs a compiled plan, its batches padded as `padding` says and its forwards' logits computed as `readout` says: preflight, then every step. It fills the plan's `results` and `run` in place and returns the same plan.

It refuses a plan that is not compiled (`plan is not compiled; call plan.compile()`). It refuses one whose model block differs from the one the engine loaded, naming each field that differs (`the plan's model is not the one this engine loaded: key 'hf-internal-testing/tiny-random-gpt2' where the engine's is 'hf-internal-testing/tiny-random-LlamaForCausalLM', revision …`). One engine executes any number of plans that pin its model block. A step that runs out of CUDA memory fills `results`, `run.costs` and `run.stopped` for the steps that finished and raises the refusal.

## `nnspec.output.manifest`

```python
from nnspec.output import manifest

held, results = manifest(out)
```

The manifest of the run at `out`, as JSON, with each record's `positions` put back in place from the manifest's shared list, and apart from it the `results`: each saved value's record by name. `held["steps"]` is what two runs compare to check that every read and write landed alike ([running.md](../usage/running.md#comparing-two-runs)). A directory without `manifest.json` is refused with `FileNotFoundError`.

## `nnspec.describe`

```python
nnspec.describe(what=None) -> str
```

The text `nnspec describe` prints: `what` is one of `records`, `steps`, `ops`, `components`, `featurizers`, `pos`, `engines`, or `None` for all of them ([cli.md](cli.md#nnspec-describe)).

## Errors

| exception | raised for | example |
|---|---|---|
| `nnspec.plan.PlanError`, a `ValueError` | a document that does not read or compile; the message starts with the document path of what is wrong | `steps.iia.of: 'patched.logits' names no value produced before this step; known: counterfactual.v_cf` |
| `ValueError` | a refusal after the model loads, which names the step or entry; a run option; an engine option; a plan not compiled or not run | `counterfactual.v_cf: layer 5; the model has 2 layers` |
| `FileNotFoundError` | a document, a run's `manifest.json` or its `tokens.json` that is not there | `no document at 'documents/nope.json'`; `no run at 'out/nope' (no manifest.json); run the document first` |
| `TypeError` | `Plan(dict)`: a document's fields passed as one dict | `Plan(dict): a document's fields are keywords, Plan(**fields); …` |
| nnsight's `RemoteError` | a refusal raised on an NDIF server under the `ndif` engine | its last line is nnspec's message |

On the command line (`nnspec/cli.py` `command`), `nnspec run` catches `ValueError`, `NotImplementedError`, `OSError` and `RemoteError`, and `nnspec show` and `nnspec table` catch `ValueError` and `OSError`; each prints one `nnspec:` line and exits 1. Messages and their fixes: [errors/index.md](../errors/index.md).
