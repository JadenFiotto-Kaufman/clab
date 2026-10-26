---
title: Running a Document
one_liner: "Running a document from the command line or Python: clab run and its options, choosing batch size, bit reproducibility, where output goes and the corpus root, and the four calls behind clab.run."
tags: [usage, running, cli, python, batch-size, reproducibility]
related: [docs/usage/engines.md, docs/usage/output.md, docs/usage/writing-a-document.md, docs/usage/bundles.md, docs/reference/cli.md, docs/reference/python-api.md, docs/concepts/plan-lifecycle.md]
sources: [clab/cli.py, clab/__init__.py, clab/plan.py, clab/engines/__init__.py, clab/format/data.py]
---

# Running a Document

`clab run DOC` compiles a document, loads its checkpoint on an engine, runs every step, and writes the saved values to an output directory. `clab.run` does the same from Python and returns the filled plan. This page is the workflow and its choices; every flag and signature is in [cli.md](../reference/cli.md) and [python-api.md](../reference/python-api.md).

## From the command line

```text
$ uv run clab run documents/patching.json
patching on nnterp (batch size 8, device cpu) -> out/patching
  iia.safetensors              iia              [4, 1]
  logit_diff.safetensors       logit_diff       [4, 1]
  out/patching/manifest.json
```

It prints each file it wrote with the step that made it and the value's shape, then the manifest's path. Read the results with `clab show` and `clab table` ([output.md](output.md)). A refusal prints `clab: <where>: <what>`, exits with status 1, and writes nothing.

Lines that transformers and torch print while the model loads are theirs. On the corpus checkpoint one says its config's `pad_token_id` is -1, outside the vocabulary. It can be ignored: clab pads each batch itself, with the tokenizer's pad token, and masks the padding.

`--engine`, `--device` and `-o key=value` choose where and how the model runs: `--engine hooks` to cross-check, `--device cuda:0` or `--device auto` for a GPU, `--engine ndif` for an NDIF server ([engines.md](engines.md)). With `HF_HUB_OFFLINE=1` the checkpoint and any hub dataset load from the local cache.

## Check before you run

Reading and compiling a document needs no model, and catches most mistakes:

```text
uv run python -c "from clab import Plan; Plan('my_experiment.json').compile()"
```

A refusal raises `clab.plan.PlanError`, a `ValueError`, and the traceback's last line is the message `clab run` would print.

What needs the tokenizer or the model's sizes (positions, layers, answer tokens) is checked after the model loads, before any forward ([writing-a-document.md](writing-a-document.md#what-is-checked-before-the-model-loads)). For a first run on a large dataset, add `"take": 8` to the dataset ([data.md](data.md)).

## Where output goes

- **`--out DIR`** names the output directory. Without it, the run writes `out/<document name>` under the current directory: `documents/patching.json` goes to `out/patching`.
- **The directory is not cleared.** A second run into it replaces the files it writes and leaves every other file there. The new manifest names only its own files, so `clab show` is right, but a stale file sits beside them. Give each run its own `--out`.
- **A training document's bundle goes to `--out` too.** The document that loads it names that file in `file_path` ([bundles.md](bundles.md)).

## The corpus root

Data paths and bundle files resolve against the document's corpus root: the nearest directory above the document that holds a `data/` folder. `"path": "weekdays/train"` reads `data/weekdays/train.json` there, and `documents/real/` shares `data/` with `documents/`. The manifest records the document's path relative to the root (`documents/patching.json`).

A document copied to a scratch directory with no `data/` above it is refused:

```text
clab: data.pairs.path: no file data/weekdays/train.json under the corpus root
```

Give the scratch directory a `data/` (a copy or a link to the repo's) and put the document below it, as `scratch/data/` beside `scratch/documents/my_experiment.json`. A document with inline `rows` and no bundles needs no root. A plan built in Python from a dict rather than read from a file resolves against the working directory.

Reading a run back resolves against the working directory too: `clab table` reads a file dataset from `data/` there, and `clab show` and `clab table` read a loaded bundle from its `file_path` there. Run them from the corpus root the document ran under ([output.md](output.md#tabling-needs-the-rows)).

## Batch size and reproducibility

`--batch-size N` (default 8) is how many rows go through the model in one forward or generation call. Each batch is padded on the left to the longest row of the step's whole input, not of the batch, so memory grows with the batch size times that longest row.

**Results reproduce to the bit at a fixed batch size.** Two runs of one document at one batch size write byte-identical value files. A different batch size can change the last bits, because a row's numbers depend on the batch's shape: `documents/patching.json` at `--batch-size 3` (a batch of 3, then 1) differs from `--batch-size 4` (one batch of 4) by 3e-8 in `logit_diff`, and a generation differs at its first call. Compare runs at one batch size. With N rows, every batch size of N or more is the same single batch.

The CPU thread count can move the last bits too: at some thread counts a row padded well past its own length moves (about 1e-8 on prompts of 1 to 12 tokens) with which rows share its batch, so a reordered or reshuffled dataset compares row for row to the bit only at the thread count it was run at. Compare runs at one batch size, one row order and one thread count; [engines.md](engines.md#comparing-two-runs-to-the-bit) has the byte-for-byte check.

A fit's body runs at the fit's own `batch_size`, whatever the run's ([fit.md](fit.md)).

## From Python

```python
import clab

plan = clab.run("documents/patching.json", out="out/patching", batch_size=8)
plan.results["iia"]          # tensor [4, 1]
```

`clab.run` takes the command line's options as arguments: `engine=`, `device=`, `batch_size=`, and the engine's own options as keywords (`remote="local"`). Without `out` it writes `out/<document name>`. Refusals raise `ValueError`.

### One model, many documents

Loading the checkpoint is the slow part. Load an engine once and pass it to each run:

```python
import clab
from clab import Plan
from clab.engines import load

engine = load("nnterp", Plan("documents/patching.json").model, "cpu")
for name in ["patching", "zero_ablation", "pos_sweep"]:
    clab.run(f"documents/{name}.json", engine=engine, out=f"out/{name}")
```

Every document must pin the same `model` block as the one the engine loaded; another is refused (`this engine loaded hf-internal-testing/tiny-random-LlamaForCausalLM@9fb1...; the plan pins hf-internal-testing/tiny-random-gpt2@7103...`). The engine keeps the device and options it loaded with, so `clab.run(..., engine=engine, device="cuda:0")` is refused.

### The four calls behind `clab.run`

```python
from clab import Plan
from clab.engines import load

compiled = Plan("documents/patching.json").compile()   # read and compile: no model yet
engine = load("nnterp", compiled.model, "cpu")         # load the checkpoint
filled = engine.execute(compiled, batch_size=8)        # preflight, then every step
filled.dump("out/patching")                            # the output directory
```

Call them yourself to stop between stages: compile a batch of documents before loading anything, inspect a compiled plan (its sweep points, each entry's `dims`), or keep results in memory without writing a directory. `execute` fills the plan you pass and returns it, and refuses one that is not compiled. What each stage checks is [plan-lifecycle.md](../concepts/plan-lifecycle.md).

## Real models

`documents/real/` holds five experiments on `meta-llama/Llama-3.2-1B`, a gated checkpoint, in float32, over the 42 weekday pairs of `weekdays_full/data`: `behavior.json` (the baseline: the model answers every prompt), `layer_sweep.json` (where the day is, block by block, at its own token and at the last), `head_patching.json` (each head of layers 10 to 12; layer 12's head 28 alone moves the answer), `pattern_patching.json` (that layer's attention patterns, which move nothing) and `das.json` (a trained subspace at layer 8, its held-out accuracy by rank). Each document's `description` states its finding. They read the corpus's `data/` and want a GPU:

```text
uv run clab run documents/real/head_patching.json --device cuda --batch-size 16
uv run clab run documents/real/das.json --engine hooks --device cuda --batch-size 16
```

`pattern_patching.json` reads the attention pattern, inside the attention's forward, so its model block gives `"attn_implementation": "eager"`; the hooks engine refuses it by name, and so does an NDIF deployment that runs sdpa. Through `--engine ndif`, the server must serve the checkpoint at the document's revision, and in float32 for the numbers to be the ones the descriptions state ([engines.md](engines.md#what-the-server-must-have)). `tests/test_real.py` pins them ([testing.md](../developing/testing.md#gpu-real-model-tests)).

## Pitfalls

- **Comparing numbers across batch sizes.** A difference in the last bits between two runs is the batch shape, not the intervention. Rerun at one batch size before reading anything into it.
- **The default output directory is under the current directory.** Running from a repository root writes `out/` into it.
- **Reusing `--out` keeps old files.** A save renamed between runs leaves its old file behind.
