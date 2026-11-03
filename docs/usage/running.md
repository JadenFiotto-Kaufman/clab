---
title: Running a Document
one_liner: "Running a document from the command line or Python: clab run and its options, what is refused before the model loads, a step that runs out of memory, choosing batch size and padding, bit reproducibility, where output goes and the corpus root, sharding with tp, and the four calls behind clab.run."
tags: [usage, running, cli, python, batch-size, padding, reproducibility]
related: [docs/usage/engines.md, docs/usage/output.md, docs/usage/writing-a-document.md, docs/usage/bundles.md, docs/reference/cli.md, docs/reference/python-api.md, docs/concepts/plan-lifecycle.md]
sources: [clab/cli.py, clab/__init__.py, clab/plan.py, clab/engines/__init__.py, clab/format/data.py, clab/steps/batches.py]
---

# Running a Document

`clab run DOC` compiles a document, loads its checkpoint on an engine, runs every step, and writes the saved values to an output directory. `clab.run` does the same from Python and returns the filled plan. This page is the workflow and its choices; every flag and signature is in [cli.md](../reference/cli.md) and [python-api.md](../reference/python-api.md).

## From the command line

```text
$ uv run clab run documents/patching.json
patching on nnterp (batch size 8, device cpu, tp 1) -> out/patching
  iia.safetensors              iia              [4,1]
  logit_diff.safetensors       logit_diff       [4,1]
  out/patching/manifest.json
```

It prints each file it wrote with the step that made it and the value's shape (a swept value's counted over its points, `3 points, each [4,1]`), then the manifest's path. Read the results with `clab show` and `clab table` ([output.md](output.md)). A refusal prints `clab: <where>: <what>`, exits with status 1, and writes nothing; a step that runs out of CUDA memory is the one exception ([below](#when-a-step-runs-out-of-memory)).

Lines that transformers and torch print while the model loads are theirs. On the corpus checkpoint one says the generation flag `pad_token_id` is not valid and may be ignored. It can be: clab pads each batch itself, with the tokenizer's pad token, else its EOS token, else id 0 (`clab/format/data.py` `pad`), and masks the padding.

`--engine`, `--device` and `-o key=value` choose where and how the model runs, `--batch-size` and `--padding` how its rows are batched ([below](#batch-size-and-reproducibility)), and `--readout` where a forward computes the logits it reads ([below](#the-head-at-the-positions-read)): `--engine hooks` to cross-check, `--device cuda:0` or `--device auto` for a GPU, `--engine ndif` for an NDIF server, `--engine vllm` for vLLM's worker on a GPU ([engines.md](engines.md)). With `HF_HUB_OFFLINE=1` the checkpoint and any hub dataset load from the local cache.

## Check before you run

Reading and compiling a document needs no model, and catches most mistakes:

```text
uv run python -c "from clab import Plan; Plan('my_experiment.json').compile()"
```

A refusal raises `clab.plan.PlanError`, a `ValueError`, and the traceback's last line is the message `clab run` would print.

`clab run` also refuses what it can tell is wrong before it loads the checkpoint, each by name:

```text
clab: no document at 'documents/nope.json'
clab: batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)
clab: padding 'rows': one of 'dataset', 'batch', 'sorted' (--padding)
clab: readout 'last': one of 'full', 'at-reads' (--readout)
clab: device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes ('auto', 'balanced', 'balanced_low_0', 'sequential')
clab: device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)
clab: steps.answer.max_new_tokens: '6' is not a value generate takes: '<=' not supported between instances of 'str' and 'int'
clab: model.key 'hf-internal-testing/tiny-random-LlamaForCausalLM' at revision '0000000000000000000000000000000000000000': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; fetch it online first (...)
```

What needs the tokenizer or the model's sizes (positions, layers, answer tokens) is checked after the model loads, before any forward ([writing-a-document.md](writing-a-document.md#what-is-checked-before-the-model-loads)). For a first run on a large dataset, add `"take": 8` to the dataset ([data.md](data.md)).

## When a step runs out of memory

A step that runs out of CUDA memory stops the run. The refusal names the step (and its point, when swept), the batch size, torch's own first sentences and every value the run holds, with its shape, dtype and device, then what to do:

```text
clab: steps.logit_diff: out of CUDA memory at batch size 8 (CUDA out of memory. Tried to allocate 20.00 MiB. ...); the run holds counterfactual.v_cf [4, 1, 16] float32 on cpu, patched.logits [4, 1, 32000] float32 on cpu, iia [4, 1] float32 on cpu, each on the model's device until the run ends; run with a smaller --batch-size, or name and save fewer values; the steps before it are saved in out/patching
```

A fit, and each of its eval steps, runs in batches of the fit's own `batch_size`, so its refusal names that batch size and says `give steps.<fit> a smaller batch_size` in place of `--batch-size`.

The steps that finished are written as if the run had ended there: their saves, their `costs`, and the refusal as `run.stopped`, which `clab show` prints on its `stopped:` line. A save of a step the run did not finish has no file, and `clab table` refuses it naming the stop. From Python, `clab.run` writes the same directory and then raises the refusal as a `ValueError`. Every value a step binds stays on the model's device until the run ends, so saving fewer values, or naming fewer (an entry nothing names is never taken off the model), frees memory as a smaller batch does.

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

Give the scratch directory a `data/` (a copy or a link to the repo's) and put the document below it, as `scratch/data/` beside `scratch/documents/my_experiment.json`. A document with inline `rows` and no bundles needs no root. A plan built in Python from a dict, `Plan(**fields)`, resolves against the working directory; `Plan(fields)` is read as a path and raises `TypeError`.

Reading a run back resolves against the working directory too: `clab table` reads a file dataset from `data/` there, and `clab show` and `clab table` read a loaded bundle from its `file_path` there. Run them from the corpus root the document ran under ([output.md](output.md#tabling-needs-the-rows)).

## Batch size and reproducibility

`--batch-size N` (default 8) is how many rows go through the model in one forward or generation call. By default (`--padding dataset`) each batch is padded on the left to the longest row of the step's whole input, not of the batch, so memory and compute grow with the batch size times that longest row: one long row makes every batch as wide as it is (on Pythia-2.8B in float32, two forwards over the 5,153 rows of LAMBADA take about 930 s).

**`--padding batch` and `--padding sorted` feed the padding the default feeds for nothing, and give up the last bits.** `batch` pads each batch to its own longest row, the rows in dataset order; `sorted` batches a step's rows longest first, pads each batch to its own longest row, and puts every value back in dataset row order before any step reads it, so results, positions, `alignment_missing` rows and tables are laid out as under the default. A fit's minibatches and a generation's batches are padded the same way. On a 10,000-row harvest of AG News through Llama-3.2-1B in bfloat16 at batch size 64, whose longest row is five times the mean, the default feeds 2.73M tokens for 0.52M real ones in 33 s; `batch` feeds 1.22M in 15 s and `sorted` 0.53M in 7 s. The price is the default's guarantee: under the default a row's numbers do not depend on which rows share its batch, though they do depend on the dataset's longest row, which sets every batch's width, while under `batch` and `sorted` they move with the width of the batch the row lands in, in a forward's values in float32 by up to 6e-8 on the tiny corpus, 3.3e-5 (3.9e-7 relative) on GPT-2 and 1.9e-5 (3.5e-6 relative) on Llama-3.2-1B over 2,000 AG News rows; a featurizer Adam trains through such a forward can move by about its learning rate (Adam scales a rounding-sized gradient along a flat direction up to the step size; SGD moves it in the last bits), so train and compare featurizers under the default. In bfloat16 that is rounding-sized: the harvest's vectors move by 1.4% of their norm on average, and a probe trained on them scores 0.832 to 0.833 held out under each padding. Each mode is deterministic: one document at one batch size, padding and thread count gives the same bits on every run, and the run records its padding (`run.padding`, the first line of `clab show`). Use the default where numbers are compared to the bit, and `sorted` for a large harvest whose input lengths vary.

**Results reproduce to the bit at a fixed batch size, padding and thread count.** Two runs of one document at one batch size, one padding and one thread count write byte-identical value files. A different batch size can change the last bits, because a row's numbers depend on the batch's shape: at 12 threads, `documents/patching.json` at `--batch-size 3` (a batch of 3, then 1) differs from `--batch-size 4` (one batch of 4) by 3e-8 in `logit_diff`, and a generation differs at its first call. Compare runs at one batch size. With N rows, every batch size of N or more is the same single batch.

The CPU thread count moves the last bits too, since the BLAS splits a matmul's reduction by thread. At 8, 12 or 24 threads a row's numbers do not depend on which rows share its batch on MKL's generic code path, which every AMD CPU takes and the snapshots were made on (on an Intel CPU MKL selects its own AVX2 kernels, the snapshots' last bits differ, and a row's matmul bits can depend on how many rows share its batch); at 4 or fewer they can differ by about 1e-8 with its batch companions, so a reordered or reshuffled dataset compares row for row to the bit only at the thread count it was run at. torch's thread count is `OMP_NUM_THREADS`, or the host's core count without it, and the run records it (`run.threads`); the suite pins 12 (`tests/conftest.py`). Compare runs at one batch size, one row order and one thread count; [engines.md](engines.md#comparing-two-runs-to-the-bit) has the byte-for-byte check.

In float16 the padding moves a row's numbers far past the last bits: on Pythia-2.8B, per-row log-likelihoods differ from an unpadded evaluation by up to 0.5 nats, and 32 of 5,153 exact-match rows change. Compare with published numbers in float32.

On a CPU without bfloat16 matrix instructions, the class of host the snapshots were made on, bfloat16 runs about 80 times slower than float32 (an attribution-patching document took 110 s against 1.3 s), so run bfloat16 on a GPU.

A fit's body runs at the fit's own `batch_size`, whatever the run's ([fit.md](fit.md)).

## The head at the positions read

**`--readout at-reads` runs the head only at the positions a forward reads `logits` at, and gives up the last bits.** By default (`--readout full`) a forward that reads `logits` at one position computes them at every position of every row, as the model does, and the read takes its positions from that: at batch size 256 on Llama-3.1-8B's 128,256-token vocabulary in bfloat16, 64-token prompts make 4.2 GB of logits a batch. Under `at-reads`, a forward whose every read of `logits` names fewer than every position (`pos: -1`, a scope) takes the last block's output at those positions instead and reads it out there with the engine's readout, the final norm, the head and what the model does after the head (`clab/steps/batches.py` `value_in_prefill`), and the forward stops before the model's own head. A read of every position, a write at `logits`, `lm_head_output` or `final_norm_output` (which a readout of the last block's output would not see) and a `grad` entry that takes or differentiates them keep the full head; so do a read of `lm_head_output`, the head's own projection, before any softcap, and a generation, whose prefill computes the head at the last position alone either way. Under `ndif` the forward runs whole on the server, so `at-reads` saves nothing there.

The price is the default's guarantee: the head is then a matmul of another shape, which sums in another order, so a logit moves in its last bits: by up to 7e-8 on the corpus in float32 (`documents/readout.json`, `documents/chat_turn.json`), more in bfloat16 on a GPU. Steering 5,000 64-token prompts through Llama-3.1-8B in bfloat16 and reading the top 5 next tokens, the forward step takes 49.7 s instead of 52.0 s at batch size 64 and peaks at 15.95 GiB instead of 16.52 GiB, the weights 14.97 GiB of it; at batch size 256 it takes 47.4 s instead of 50.0 s and peaks at 18.92 GiB instead of 21.19 GiB, and 380 of its 25,000 saved probabilities move, by up to 0.002 and 6% of the value, since a bfloat16 logit that moves moves by a whole ulp (at 64 none do; the top 5 ids stay). The engines agree to the bit under either readout, and the run records its readout (`run.readout`, the first line of `clab show`). Use the default where numbers are compared to the bit.

On the hooks engine the readout is the standard form (`clab/engines/hooks.py` `readout`: the final norm, the head and the config's `final_logit_softcapping`), not the model's own: on a family that scales after the head (Cohere's `logit_scale`, Granite's `logits_scaling`) a read of `logits` under `at-reads` there leaves that out and moves by more than the last bits. Run such a model under `at-reads` on the nnterp engine.

## From Python

```python
import clab

plan = clab.run("documents/patching.json", out="out/patching", batch_size=8)
plan.results["iia"]          # tensor [4, 1]
```

`clab.run` takes the command line's options as arguments: `engine=`, `device=`, `batch_size=`, `padding=`, `readout=`, and the engine's own options as keywords (`remote="local"`, `tp=2`). Without `out` it writes `out/<document name>`. Refusals raise `ValueError`.

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

Every document must pin the same `model` block as the one the engine loaded; another is refused, naming each field that differs (`the plan's model is not the one this engine loaded: key 'hf-internal-testing/tiny-random-gpt2' where the engine's is 'hf-internal-testing/tiny-random-LlamaForCausalLM', revision '71034c5d...' where the engine's is '9fb19125...'`). The engine keeps the device and options it loaded with, so `clab.run(..., engine=engine, device="cuda:0")` is refused (`the loaded nnterp engine runs on cpu with {'tp': 1}; device and options are given when an engine loads`).

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

`-o tp=N` shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under `torchrun --nproc-per-node=N`, each rank on its own GPU (`--device cuda`; `--device cpu` where none is visible):

```text
uv run torchrun --nproc-per-node=2 --no-python clab run documents/real/behavior.json --device cuda -o tp=2 --batch-size 16
```

Every rank runs the whole plan, and rank 0 alone writes the output directory and prints; `clab show` reads `tp 2`. What `tp` refuses (a value the ranks hold in pieces, a bitsandbytes `dtype`, torchrun around an engine without `tp`) and how close its numbers are to unsharded ones is [engines.md](engines.md#quantized-and-sharded-models). Outside torchrun `-o tp=2` is refused before anything loads:

```text
clab: tp 2 shards the model over 2 processes, one per rank, that torchrun starts: `torchrun --nproc-per-node=2 --no-python clab run <document> -o tp=2`; this process is not one of them (WORLD_SIZE is unset)
```

## Real models

`documents/real/` holds five experiments on `meta-llama/Llama-3.2-1B`, a gated checkpoint, in float32, over the 42 weekday pairs of `weekdays_full/data`: `behavior.json` (the baseline: the model answers every prompt), `layer_sweep.json` (where the day is, block by block, at its own token and at the last), `head_patching.json` (each head of layers 10 to 12; layer 12's head 28 alone moves the answer), `pattern_patching.json` (that layer's attention patterns, which move nothing) and `das.json` (a trained subspace at layer 8, its held-out accuracy by rank). Each document's `description` states its finding. They read the corpus's `data/` and want a GPU:

```text
uv run clab run documents/real/head_patching.json --device cuda --batch-size 16
uv run clab run documents/real/das.json --engine hooks --device cuda --batch-size 16
```

`pattern_patching.json` reads the attention pattern, inside the attention's forward, so its model block gives `"attn_implementation": "eager"`; the hooks engine refuses it by name, and so does an NDIF deployment that runs sdpa. Through `--engine ndif`, the server must serve the checkpoint at the document's revision, and in float32 for the numbers to be the ones the descriptions state ([engines.md](engines.md#what-the-server-must-have)). `tests/test_real.py` pins them ([testing.md](../developing/testing.md#gpu-real-model-tests)).

## Pitfalls

- **Comparing numbers across batch sizes or paddings.** A difference in the last bits between two runs is the batch shape, not the intervention. Rerun at one batch size and padding before reading anything into it.
- **The default output directory is under the current directory.** Running from a repository root writes `out/` into it.
- **Reusing `--out` keeps old files.** A save renamed between runs leaves its old file behind.
