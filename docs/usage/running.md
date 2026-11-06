---
title: Running a Document
one_liner: "How do I run a document, and why would two runs differ? The CLI and Python, batch size, padding, readout, memory, the corpus root and tp."
tags: [usage, running, cli, python, batch-size, padding, reproducibility]
related: [docs/usage/engines.md, docs/usage/output.md, docs/usage/writing-a-document.md, docs/usage/fit.md, docs/usage/bundles.md, docs/reference/cli.md, docs/reference/python-api.md, docs/concepts/plan-lifecycle.md, docs/developing/testing.md]
sources: [clab/cli.py, clab/__init__.py, clab/plan.py, clab/engines/__init__.py, clab/format/data.py, clab/steps/__init__.py, clab/steps/batches.py, clab/compile/schedule.py, clab/output/__init__.py]
---

# Running a Document

`clab run DOC` compiles a document, loads its checkpoint on an engine, runs every step, and writes the saved values to an output directory. `clab.run` does the same from Python and returns the filled plan. This page is the workflow and its choices; every flag and signature is in [cli.md](../reference/cli.md) and [python-api.md](../reference/python-api.md).

## From the command line

```text
$ clab run documents/patching.json
patching on nnterp (batch size 8, device cpu, tp 1) -> out/patching
  iia.safetensors         iia         [4,1]
  logit_diff.safetensors  logit_diff  [4,1]
  out/patching/manifest.json
```

It prints each file it wrote, with the step that made it and the value's shape, then the manifest's path. A swept value's shape is counted over its points (`3 points, each [4,1]`). Read the results with `clab show` and `clab table` ([output.md](output.md)). A refusal prints `clab: <where>: <what>`, exits with status 1, and writes nothing; a step that runs out of CUDA memory is the one exception ([below](#when-a-step-runs-out-of-memory)).

Lines that transformers and torch print while the model loads are theirs. On the corpus checkpoint one says the generation flag `pad_token_id` is not valid and may be ignored. It can be: clab pads each batch itself, with the tokenizer's pad token, else its EOS token, else id 0 (`clab/format/data.py` `pad`), and masks the padding.

The options choose where the model runs and how its rows are fed:

- `--engine hooks` cross-checks on plain transformers, `--engine ndif` runs on an NDIF server, and `--engine vllm` runs on vLLM's worker on a GPU ([engines.md](engines.md)).
- `--device cuda:0` or `--device auto` puts the model on a GPU. Run a `bfloat16` model there: a CPU without bfloat16 matrix instructions runs bfloat16 matmuls much slower than float32.
- `-o key=value` sets one of the engine's own options (`-o remote=local`, `-o tp=2`).
- `--batch-size`, `--padding` and `--readout` set how many rows share a forward, how each batch is padded and where the head runs ([below](#batch-size-and-padding)).
- With `HF_HUB_OFFLINE=1` the checkpoint and any hub dataset load from the local cache.

## Check before you run

`clab compile DOC` reads and compiles a document and does nothing more: no engine, no model, no output directory. It catches most mistakes, each by the document path of what is wrong. On success it prints what `clab run` would run, each run of a step in run order: its kind, a forward's or generation's input, the sweep point it runs at, and the points it serves.

```text
$ clab compile documents/pos_sweep.json
pos_sweep: compiled, not run
  step            kind     input                           at      serves
  counterfactual  forward  pairs.counterfactual_inputs[0]  —       —
  patched         forward  pairs.input                     pos=-1  —
  patched         forward  pairs.input                     pos=-2  —
  patched         forward  pairs.input                     pos=-3  —
  iia             op       —                               pos=-1  —
  …
```

- **A refusal is the line `clab run` would print**, `clab: <where>: <what>` on stderr, with exit status 1; success exits 0.
- **It takes only the document.** `clab run`'s options never reach compile: `--batch-size`, `--padding` and `--readout` are checked before it, `--engine`, `--device` and `-o` as the engine loads after it.
- **From Python** it is `Plan(path).compile()`, which raises `clab.plan.PlanError`, a `ValueError`, with the same message.
- **A forward that serves several sweep points is one row**, its `serves` cell `all 3 points` (`documents/generate_probe.json`); every other point is a row of its own.

What is checked when:

| stage | what it checks | a refusal it gives |
|---|---|---|
| `clab compile`: reading and compiling, no model | each record's fields, every reference, column, save and sweep, the data files, bundle headers, a `generate` option's type | `steps.iia.of: 'patched.logits' names no value produced before this step; …` |
| `clab run`, before the checkpoint loads | the document's path, `--batch-size`, `--padding`, `--readout`, `--device`, the engine and its options ([cli.md](../reference/cli.md#clab-run) lists each refusal) | `device 'gpu': not a torch device …` |
| after the model loads, before any forward | positions, layers, `select` indices and answer tokens, against the tokenizer and the model's sizes; a component the engine does not serve | `counterfactual.v_cf: layer 5; the model has 2 layers` |
| when the step runs | an option the model's `generate` rejects; running out of CUDA memory | `answer: generate rejected 'bogus_opt': …` |

Offline, a checkpoint the cache lacks is refused as the engine loads it. For a first run on a large dataset, add `"take": 8` to the dataset ([data.md](data.md#narrowing-and-renaming-rows)). Which stage gave a refusal is [plan-lifecycle.md](../concepts/plan-lifecycle.md#which-stage-refused).

## When a step runs out of memory

A step that runs out of CUDA memory stops the run. The refusal names the step (and its point, when swept), the batch size, torch's first sentences and every value the run holds, with its shape, dtype and device, then what to do:

```text
clab: steps.logit_diff: out of CUDA memory at batch size 8 (CUDA out of memory. Tried to allocate 20.00 MiB. ...); the run holds counterfactual.v_cf [4, 1, 16] float32 on cpu, patched.logits [4, 1, 32000] float32 on cpu, iia [4, 1] float32 on cpu, each on the model's device until the run ends; run with a smaller --batch-size, or name and save fewer values; the steps before it are saved in out/patching
```

- **The finished steps are written** as if the run had ended there: their saves, their `costs`, and the refusal as `run.stopped`, which `clab show` prints on its `stopped:` line.
- **A save of a step the run did not finish has no file**, and `clab table` refuses it naming the stop.
- **From Python**, `clab.run` writes the same directory and then raises the refusal as a `ValueError`.
- **Every value a step binds stays on the model's device until the run ends.** Saving or naming fewer values frees memory as a smaller batch does; an entry nothing names is never taken off the model ([ops.md](ops.md#a-step-or-an-entry)).
- **A fit's refusal names the fit's own `batch_size`** and says `give steps.<fit> a smaller batch_size`. What bounds a fit's memory is [fit.md](fit.md#what-a-fit-costs).

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

Give the scratch directory a `data/` (a copy or a link to the repo's) and put the document below it, as `scratch/data/` beside `scratch/documents/my_experiment.json`. A document with inline `rows` and no bundles needs no root. A plan built in Python from fields, `Plan(**fields)`, resolves against the working directory ([python-api.md](../reference/python-api.md#clabplan)).

Reading a run back resolves against the working directory too: `clab table` reads a file dataset from `data/` there, and `clab show` and `clab table` read a loaded bundle from its `file_path` there. Run them from the corpus root the document ran under ([output.md](output.md#tabling-needs-the-rows)).

## Batch size and padding

`--batch-size N` (default 8) is how many rows go through the model in one forward or generation call. `--padding` says what each batch is padded to, on the left:

| `--padding` | each batch is padded to | rows run in | a row's last bits move with | use it for |
|---|---|---|---|---|
| `dataset`, the default | the longest row of the step's whole input | dataset order | that longest row, which sets every batch's width | numbers you compare, featurizers you train |
| `batch` | its own longest row | dataset order | the width of the batch the row lands in | fewer tokens fed, in row order |
| `sorted` | its own longest row | longest first, rows of one length in dataset order | the width of the batch the row lands in | the fewest tokens fed: a large harvest whose lengths vary |

- **Under the default, one long row makes every batch as wide as it is.** Memory and compute grow with the batch size times that row. A large gap between a step's `tokens` and `real` in `clab show` is padding ([output.md](output.md#the-run-record)).
- **`sorted` puts every value back in dataset row order before any step reads it.** Results, positions, `alignment_missing` rows and tables are laid out as under the default.
- **A generation's batches and a fit's minibatches are padded the same way.** A fit's body runs at the fit's own `batch_size`, whatever `--batch-size` says ([fit.md](fit.md#epochs-minibatches-and-the-learning-rate)).
- **Train featurizers under the default.** A fit can turn a last-bit move in what it reads into a larger move in what it trains.

### The head at the positions read

`--readout` says where a forward computes the logits it reads:

- **`full`, the default,** computes the head at every position of every row, as the model does, and a read takes its positions from that.
- **`at-reads`** acts on a `forward` step whose every read of `logits` names fewer than every position (`pos: -1`, a scope). The forward stops before the model's own head. Each read takes the last block's output at its positions and reads it out there with the engine's readout: the final norm, the head, and what the model does after the head (`clab/compile/schedule.py` `read_out_at_reads`). The head's compute and memory are then those of the positions read.
- **The full head runs anyway** where a forward reads `logits` at every position, where a `grad` entry takes or differentiates a read of `logits`, or where something writes at `final_norm_output`, `lm_head_output` or `logits`, which the readout would not see. A read of `lm_head_output`, the head's own projection, keeps it too.
- **`at-reads` moves the last bits.** The head is then a matmul of another shape, which sums in another order, so a logit can move in its last bits. In bfloat16 a logit moves by a whole ulp when it moves at all, so a saved probability can move visibly.
- **On the hooks engine the readout is the standard form** (`clab/engines/hooks.py` `readout`): the final norm, the head and the config's `final_logit_softcapping`. On a family that does more after the head (Cohere's `logit_scale`, Granite's `logits_scaling`), a read of `logits` under `at-reads` there leaves that out. Run such a model under `at-reads` on the nnterp engine.

The run records both choices (`run.padding`, `run.readout`), and `clab show` prints them on its first line.

## When two runs give the same numbers

**Two runs give the same bits only at one batch shape.** One document run twice at one batch size, padding, readout, device and CPU thread count writes byte-identical value files. Change any of them and a row's numbers can move in their last bits:

- **Batch size and padding.** A batch of another size or width is a matmul of another shape, which sums in another order. With N rows, every batch size of N or more is one batch.
- **Regrouped rows.** Under `batch` and `sorted` a row's width is its batch's, so a reordered or reshuffled dataset can move a row's numbers. Under the default it is the step's longest row whatever batch the row lands in, though at some thread counts the batch's other rows can still move its last bits.
- **Readout.** `--readout at-reads` computes the head in another shape ([above](#the-head-at-the-positions-read)).
- **Threads and machine.** A CPU's BLAS splits a matmul's reduction by thread, so another thread count can move the last bits, and its kernels differ by CPU. Compare runs made on one machine.

The run records each setting: `run.batch_size`, `run.padding`, `run.readout`, `run.devices` and `run.threads`, torch's intra-op thread count (`torch.get_num_threads()`). `clab show` prints all of them but `run.threads`; read that one from the manifest's `run`. Set the thread count with `OMP_NUM_THREADS` for `clab run`, or `torch.set_num_threads` before `clab.run`, and check `run.threads` to see what the run used. The suite pins its own (`tests/conftest.py` `THREADS`). Compare runs only where every setting agrees.

In float16 the padding can move a row's numbers well past the last bits. Compare with published numbers in float32.

### Comparing two runs

Two runs give the same numbers when their value files do. Every file in the output directory but `manifest.json` holds values or, in `tokens.json`, the tokens, so compare those byte for byte. The manifest always differs, since it records the engine and the time. The same recipe checks a second engine, hooks or ndif under `remote=local`, against nnterp:

```text
clab run documents/patching.json --out out/nnterp
clab run documents/patching.json --engine hooks --out out/hooks
clab run documents/patching.json --engine ndif -o remote=local --out out/ndif
for e in hooks ndif; do for f in out/nnterp/*; do [ "${f##*/}" = manifest.json ] || cmp "$f" "out/$e/${f##*/}"; done; done
```

`cmp` prints nothing where the files match, NaN rows and JSON values included. That every read and write landed alike is in the manifests' `steps`, which hold each entry's `positions` and each forward's schedule. Compare them as `clab.output.manifest`, the directory's one reader, gives them, each entry's positions in place: `manifest("out/nnterp")[0]["steps"] == manifest("out/hooks")[0]["steps"]` is `True` here, and so is the same against `out/ndif`.

## From Python

```python
import clab

plan = clab.run("documents/patching.json", out="out/patching", batch_size=8)
plan.results["iia"]          # tensor [4, 1]
```

`clab.run` takes the command line's options as arguments: `engine=`, `device=`, `batch_size=`, `padding=`, `readout=`, and the engine's own options as keywords (`remote="local"`, `tp=2`). Without `out` it writes `out/<document name>`. A refusal raises `ValueError`, and a missing document `FileNotFoundError` ([python-api.md](../reference/python-api.md#errors)).

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

Every document must pin the same `model` block as the one the engine loaded. Another is refused, naming each field that differs (`the plan's model is not the one this engine loaded: key 'hf-internal-testing/tiny-random-gpt2' where the engine's is 'hf-internal-testing/tiny-random-LlamaForCausalLM', revision …`). The engine keeps the device and options it loaded with, so `clab.run(..., engine=engine, device="cuda:0")` is refused (`the loaded nnterp engine runs on cpu with {'tp': 1}; device and options are given when an engine loads`).

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

## Sharding a model: `-o tp`

`-o tp=N` shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under `torchrun --nproc-per-node=N`. Each rank runs on its own GPU with `--device cuda`, or with `--device cpu` where none is visible:

```text
torchrun --nproc-per-node=2 --no-python clab run documents/patching.json --device cpu -o tp=2
```

Every rank runs the whole plan, and rank 0 alone writes the output directory and prints; `clab show` reads `tp 2`. What `tp` refuses (a value the ranks hold in pieces, a bitsandbytes `dtype`, torchrun around an engine without `tp`) and how close its numbers are to unsharded ones is [engines.md](engines.md#quantized-and-sharded-models). Outside torchrun `-o tp=2` is refused before anything loads:

```text
clab: tp 2 shards the model over 2 processes, one per rank, that torchrun starts: `torchrun --nproc-per-node=2 --no-python clab run <document> -o tp=2`; this process is not one of them (WORLD_SIZE is unset)
```

## Real models

The documents under `documents/real/` want a GPU and a gated checkpoint in the local cache; how to run them and their tests is [testing.md](../developing/testing.md#gpu-real-model-tests).
