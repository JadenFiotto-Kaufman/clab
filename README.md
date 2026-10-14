# clab

clab runs mechanistic-interpretability experiments on transformer language models from declarative JSON
documents. A document names a pinned checkpoint, its datasets, the forwards to run with their writes and
reads on the model's internals, the ops that score or reduce what they read, and what to save. clab
compiles it into a plan of plain data, runs the plan inside one nnsight session (so the same document runs
locally or on a remote nnsight server), and writes each saved value to a directory with a manifest.

Two engines ship: `nnterp` (primary: nnterp's standardized names on every architecture, remote-capable)
and `hooks` (plain HuggingFace with forward hooks, for module-boundary components).

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

From Python:

```python
import clab

plan = clab.run("documents/patching.json", engine="nnterp", out="out/patching")
plan = clab.load("out/patching")            # the filled plan back, its tensors loaded
rows = clab.table(plan, "logit_diff")       # one flat table over every point of a saved value
```

## The format

`FORMAT.md` is the one-page reference: root keys, step kinds, every op with its arguments and axis
behaviour, every component, featurizer kinds, the `pos` grammar, names and scoping, sweeps and saves. It is
generated from the code (`clab describe --markdown`), and `clab describe [ops|steps|components|featurizers|pos]`
prints any of its tables. `documents/` holds the corpus of 31 documents; `data/` their datasets.

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
