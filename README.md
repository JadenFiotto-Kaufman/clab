# clab

clab runs mechanistic-interpretability experiments on transformer language models from declarative JSON
documents. A document names a pinned checkpoint, its datasets, the forwards and generations to run with
their writes and reads on the model's internals, the ops that score or reduce what they read, and what to
save. clab reads it
into a `Plan`, compiles the plan (rows loaded, names resolved, sweeps expanded into points), runs it inside one
nnsight session (so the same document runs locally or on a remote nnsight server), and writes each saved value
to a directory with a manifest.

Two engines ship: `nnterp` (primary: nnterp's standardized names on every architecture, remote-capable) and
`hooks` (plain HuggingFace with forward hooks, for module-boundary components).

## Install

```
uv sync
```

## Run a document

```
uv run clab run documents/patching.json                  # writes out/patching/
uv run clab run documents/patching.json --engine hooks --out out/patching-hooks --batch-size 2
uv run clab run documents/patching.json --remote local   # nnsight's in-process serialize/deserialize path
uv run clab run documents/generated_answer.json --batch-size 2
```

## Train a featurizer, then apply it

A `fit` step trains featurizers (a rotation's subspace, a mask over units, a probe) inside the run, through
the frozen model: its body is the experiment, run over minibatches, and what it trains is saved as a bundle
another document loads. `das.json` trains a rank-8 rotation and saves it as `fit.rot`; `das_apply.json`
loads that bundle and scores it on the held-out split, checking its header (model, site, kind, `k`, width)
key by key first:

```
uv run clab run documents/das.json --batch-size 2 --out documents/artifacts    # writes fit.rot.safetensors
uv run clab run documents/das_apply.json --batch-size 2                        # loads it
uv run clab table out/das_apply iia                                            # das's held-out iia, to the bit
```

`pca_control.json` loads what `pca_harvest.json` writes the same way. `documents/artifacts/sae.safetensors`,
the dictionary `sae_feature_ablation.json` loads, is a committed fixture (`uv run python tests/fixtures/sae.py`
writes it again). A fit's members are saved like any value; `fit.history.<op>` is a body op's mean on the
training rows after each epoch:

```python
plan = clab.run("documents/dbm.json", batch_size=2)
plan.results["fit.mask"][0.05]       # the gate trained at L1 weight 0.05: {"theta": tensor [16]}
(plan.results["fit.mask"][0.05]["theta"] > 0).sum()    # the units its hard mask keeps
```

## Look at a run

```
uv run clab show out/generate_probe                      # each result: its step, dims, axes, shape; how it ran
uv run clab table out/generate_probe p_answer patched    # the saved values as one table, joined on row and step
uv run clab table out/generate_probe p_answer --csv      # the same as CSV on stdout, every cell whole
```

`show` prints the run's engine, remote flag, batch size, versions, start time and seconds, then one line per
saved value: the step that made it, its `dims`, its sweep axes and its shape (per coordinate when swept).
`table` prints `plan.table(name, ...)`: a row per example and position (`pos`/`pos_token`, and in a
generation the forward `call` that holds it), the point's coordinates `at` when the step is swept, the
dataset columns the step used, and one column per name; a row a read beyond the prefill did not reach says
`alignment_missing` in place of a value. Aligned output cuts long cells short;
`--csv` keeps them whole. Both are `Plan.load(out)` and one call, run from the corpus root (where the
dataset files are).

## From Python

```python
import clab
from clab import Plan

plan = clab.run("documents/patching.json", engine="nnterp", out="out/patching")
plan.results["iia"]                  # tensor [4, 1]: 1 where the patched model's top token is the answer
plan                                 # Plan(patching, run on nnterp, 2 results)

swept = clab.run("documents/pos_sweep.json")
swept.results["iia"][-2]             # the point where the patch landed at pos -2: tensor [4, 1]

back = Plan.load("out/patching")     # the filled plan again, from its directory
back.table("iia", "logit_diff")      # one row per example: row, pos, pos_token, iia, logit_diff, the columns used
```

`clab.run(plan_or_path, engine=..., out=None, batch_size=8, remote=False)` is the four steps of a run, which
are also there to take one at a time:

```python
from clab.engines import ENGINES

plan = Plan("documents/pos_sweep.json")       # Plan(pos_sweep, uncompiled): the document as a model
compiled = plan.compile()                     # Plan(pos_sweep, compiled, 3 points)
engine = ENGINES["nnterp"](compiled.model)    # loads the checkpoint; one engine serves many plans
filled = engine.execute(compiled, batch_size=8)
filled.dump("out/pos_sweep")
```

A plan's methods are views and serialization, never execution: `compile`, `dump`, `Plan.load`, `write` (the
plan as a JSON document, in its current state), `table`; `plan.run` says how it ran (engine, remote, batch
size, versions, timing, and how each input tokenized). `clab.describe()` prints the format's tables.

## Extending clab

Every extensible thing is a plain function or class registered by a decorator under its name, and its
signature is its contract: what a document may write, checked by `inspect.signature` with the valid choices
named in every error.

- **Ops** (`clab/ops.py`): `@op(group, needs=..., removes=..., adds=..., bundle=None)` registers the function
  and declares its group (`metrics`, `reductions`, `edits`, `transforms`), the axes the input must carry, the
  axes it removes (a tuple, or the name of the argument that lists them) and appends, and, when its value is
  a bundle a featurizer loads, that featurizer's kind. A parameter annotated `Value`, `Tokens` or
  `Featurizer` names a thing the run binds; `engine`, `original` and `dims` are filled by the run, never by
  a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is `"with"`).
- **Step kinds** (`clab/steps.py`, `@kind`), **featurizer kinds** (`clab/featurizers.py`, `@kind`) and
  **engines** (`clab/engines/`, `@engine("<name>")`). An engine loads its checkpoint in `__init__` from a
  plan's `model` block; `execute(plan, batch_size=, remote=)` is its one entry point.

## The output

One file per saved name: `<name>.safetensors` when the value is a tensor or a dict of tensors, `<name>.json`
otherwise. A swept value is one file keyed by coordinate (`pos=-1`, `pos=-2`, ...). `manifest.json` is the
plan itself, without the rows it read from files, with each result replaced by where it is and what it is:

```json
"results": {"iia": {"file": "iia.safetensors", "step": "iia", "dims": ["rows", "pos"], "axes": ["pos"],
                    "shape": {"pos=-1": [4, 1], "pos=-2": [4, 1], "pos=-3": [4, 1]}, "dtype": {...}}}
```

`step` names the step that made the value, and the manifest's `steps` hold that step as compiled, so a result
links back to what computed it. `run.tokens` holds each input's tokens per row and `run.positions` the token
every positioned read and write resolved to, which is how `table` labels `pos_token`.

## The format

`FORMAT.md` is the one-page reference: root keys, every record and its fields, step kinds, every op with its
arguments and axis behaviour (grouped as metrics, reductions and edits), every component, featurizer kinds,
the `pos` grammar (three stages: `frame` picks forward calls, the prefill or decode calls, `scope` text
within their tokens, `index` positions within that), names and scoping, sweeps and the output. It is generated from the code (`clab describe
--markdown`), and `schema.json` is the plan's JSON Schema (`clab schema`); a test fails when either is stale.
`documents/` holds the corpus of 31 documents; `data/` their datasets.

## Tests

```
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run pytest tests/ -q
uvx pyright
```

`CLAB_UPDATE_SNAPSHOTS=1` rewrites the checked-in snapshots under `tests/snapshots/`.

Numbers reproduce to the bit at a fixed `--batch-size`. Every batch of an input is padded to that input's
longest row, so a row's numbers do not depend on which rows share its batch; they do depend on the batch's
shape, and a trailing partial batch (a smaller matrix) can differ in the last bits. The snapshots pin the
batch size for that reason.
