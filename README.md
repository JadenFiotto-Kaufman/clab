# clab

clab runs mechanistic-interpretability experiments on transformer language models from declarative JSON
documents. A document names a pinned checkpoint, its datasets, the forwards to run with their writes and
reads on the model's internals, the ops that score or reduce what they read, and what to save. clab reads it
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
```

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
the `pos` grammar, names and scoping, sweeps and the output. It is generated from the code (`clab describe
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
