---
title: Command Line
one_liner: "Every clab command and flag (run, show, table, describe), with defaults, exit statuses and refusals."
tags: [reference, cli]
related: [docs/usage/running.md, docs/usage/output.md, docs/usage/engines.md, docs/reference/python-api.md, FORMAT.md]
sources: [clab/cli.py, clab/__init__.py, clab/docs.py]
---

# Command Line

`clab` has four commands: `run`, `show`, `table` and `describe`. Inside the repository's environment, call it as `uv run clab <command>`. This page lists every argument; the workflow around them is [running.md](../usage/running.md), and reading output is [output.md](../usage/output.md).

Every command exits 0 on success. A refusal or error that clab reports is printed to stderr as `clab: <message>`, and the command exits 1. A malformed command line (an unknown flag, a missing argument, a value outside a flag's choices) gets argparse's usage message and exit status 2 ([errors/index.md](../errors/index.md#errors-that-are-not-clabs)).

## `clab run`

```text
clab run [--engine ENGINE] [--out OUT] [--batch-size N] [--padding dataset|batch|sorted] [--readout full|at-reads] [--device D] [-o KEY=VALUE ...] DOCUMENT
```

Compiles the document, loads the engine, runs every step, and writes the output directory.

| argument | default | meaning |
|---|---|---|
| `DOCUMENT` | required | the document's JSON file. Data paths and bundles resolve against the nearest directory above it that holds `data/` |
| `--engine` | `nnterp` | the engine: `nnterp`, `hooks`, `ndif`, `vllm`, or one registered beside them ([engines.md](../usage/engines.md)); another name is refused by name |
| `--out` | `out/<document file name>` in the current directory | the output directory, created when missing and never cleared |
| `--batch-size` | `8` | how many rows go through the model per forward or generation call; a positive integer |
| `--padding` | `dataset` | how each batch is padded: `dataset`, to the input's longest row, the exact run; `batch`, to its own longest row; `sorted`, the rows batched longest first, each batch to its own longest row, every value put back in row order ([running.md](../usage/running.md#batch-size-and-reproducibility)) |
| `--readout` | `full` | how a forward computes the logits it reads: `full`, the model's head over every position, the exact run; `at-reads`, a forward's reads of `logits` that name fewer than every position read out from the last block's output at those positions alone ([running.md](../usage/running.md#the-head-at-the-positions-read)) |
| `--device` | the engine's own: `cpu`, or `cuda:0` under `--engine vllm` | where the engine places the model: one torch device (`cuda:0`) or a device map transformers computes (`auto`, `balanced`, ...); under `-o tp`, `cuda` or `cpu`. Under `--engine ndif` with a real server the server places it, and any value but `cpu` is refused; under `--engine vllm` it is `cuda:0`, the first of the GPUs vLLM runs on, and any other is refused |
| `-o KEY=VALUE` | none | one of the engine's own options, repeatable. `VALUE` is read as JSON (`true`, `2`, `{"a": 1}`) and otherwise as text (`local`). `clab describe engines` lists each engine's options: `nnterp` has `tp`, `ndif` has `remote`, `vllm` has `tp`, `memory`, `max_len` and `invariant`, `hooks` has none |

`-o tp=N` shards the model over N ranks, one process each, which torchrun starts; every rank runs the plan and rank 0 alone writes the output and prints ([engines.md](../usage/engines.md#quantized-and-sharded-models)):

```text
torchrun --nproc-per-node=2 --no-python clab run documents/patching.json --device cpu -o tp=2
```

Refused before the model loads:

```text
clab: no document at 'documents/nope.json'
clab: batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)
clab: padding 'rows': one of 'dataset', 'batch', 'sorted' (--padding)
clab: readout 'last': one of 'full', 'at-reads' (--readout)
clab: device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes ('auto', 'balanced', 'balanced_low_0', 'sequential')
clab: device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)
clab: -o device: it is given as --device
clab: -o batch_size: it is given as --batch-size
clab: -o padding: it is given as --padding
clab: -o readout: it is given as --readout
clab: -o 'foo': an option is KEY=VALUE
clab: unknown engine 'bogus'; valid: nnterp, hooks, ndif, vllm
clab: the nnterp engine takes no 'foo'; its options: tp (where the model runs is `device`, given as --device)
clab: tp 2 shards the model over 2 processes, one per rank, that torchrun starts: `torchrun --nproc-per-node=2 --no-python clab run <document> -o tp=2`; this process is not one of them (WORLD_SIZE is unset)
```

Offline, a checkpoint whose `revision` the cache lacks is refused as the engine loads it (the nnterp engine's weights, and a model key its loader rejects, at the first run that calls the model): `clab: model.key '<key>' at revision '<revision>': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; fetch it online first (…)`.

On success it prints one line naming the run with its engine options, one line per saved file (the file, the step that made it, the value's shape, counted over its points when swept) and the manifest's path:

```text
patching on ndif (batch size 8, device cpu, remote "local") -> out/patching
  iia.safetensors              iia              [4,1]
  logit_diff.safetensors       logit_diff       [4,1]
  out/patching/manifest.json
pos_sweep on nnterp (batch size 8, device cpu, tp 1) -> out/pos_sweep
  iia.safetensors              iia              3 points, each [4,1]
  logit_diff.safetensors       logit_diff       3 points, each [4,1]
  out/pos_sweep/manifest.json
```

Under `--engine ndif`, a refusal raised on the server arrives as nnsight's `RemoteError`, and `clab run` prints its last line. A step that runs out of CUDA memory stops the run: `clab run` prints the refusal, which ends `the steps before it are saved in <out>`, and exits 1 with the finished steps written ([running.md](../usage/running.md#when-a-step-runs-out-of-memory)).

## `clab show`

```text
clab show OUT
```

Prints output directory `OUT`'s run in four parts:

- the run record: engine, device and options, batch size and padding, start time and duration, the served dtype and devices, the versions, and a `stopped:` line with the refusal when a step ran out of CUDA memory;
- what each run of a step cost, one row per run at the point that ran it: `serves` the points it served (`—` itself alone, `all 3 points`, or each one), its seconds, its peak memory on each CUDA device, the model calls it made (`forwards`), the tokens they fed, padding included (`tokens`), and the rows' own of those (`real`);
- where each entry resolved, one row per entry and distinct positions per row (`[10], [8], [8], [8]`; one list and `on all 4 rows` where every row agrees; `—` for a row it did not reach; past twelve rows the first three, `…`, the last and the row count, and a row past twelve positions its first three and last), with the point it is at or how many points share it;
- per saved value, the step that made it, its dims, the sweep axes the value depends on and its shape (`3 points, each [4,1]` when swept).

```text
$ clab show out/pos_sweep
pos_sweep: run on nnterp, device cpu, tp 1, batch size 2, padding dataset, readout full, started 2026-11-02T13:02:15, 0.225 s
  served: float32 on cpu
  versions: clab 0.0.1+…, nnsight 0.8.1, nnterp …, torch …, transformers …, python …
  step            at      serves  seconds  memory  forwards  tokens  real
  counterfactual  —       —       0.005    —       2         44      38
  patched         pos=-1  —       0.006    —       2         44      38
  …
  logit_diff      pos=-3  —       0.000    —       0         0       0
  entry                at            positions
  counterfactual.v_cf  —             [8], [8], [10], [8]
  patched.patch        pos=-1        [10], [8], [8], [8]
  patched.logits       all 3 points  [10], [8], [8], [8]
  patched.patch        pos=-2        [9], [7], [7], [7]
  patched.patch        pos=-3        [8], [6], [6], [6]
  result      step        dims       axes  shape
  iia         iia         rows, pos  pos   3 points, each [4,1]
  logit_diff  logit_diff  rows, pos  pos   3 points, each [4,1]
```

It reads `OUT/manifest.json`, and without one refuses: `clab: no run at 'OUT' (no manifest.json); run the document first`. A run that loaded a bundle has it read again from its `file_path`, relative to the current directory, so run `show` from the corpus root ([bundles.md](../usage/bundles.md#reading-back-a-run-that-loads-a-bundle)).

## `clab table`

```text
clab table [--csv] OUT NAME [NAME ...]
```

| argument | meaning |
|---|---|
| `OUT` | a run's output directory |
| `NAME` | one or more saved names; several are joined on the axes they share |
| `--csv` | CSV with a header row, every cell whole. Without it, aligned columns with cells over 48 characters cut short with `…` |

The rows and columns are described in [output.md](../usage/output.md#clab-table). It reads dataset rows to label the table: a file's from `data/` under the current directory, a hub dataset's from the hub or cache ([output.md](../usage/output.md#tabling-needs-the-rows)). Refused:

```text
clab: 'patched.logits' was not saved; add it to saves
clab: 'held_iia' has a row per example of 'test' and 'fit.train.iia' of 'train'; a table joins the values of one dataset
clab: 'iia' has a pos column counting positions of pairs.input, and 'counterfactual.v_cf' one counting positions of pairs.counterfactual_inputs[0]: the column does not mean one thing for both, so they do not join; table them one at a time
clab: 'fit.rot' is a trained featurizer's bundle, which has no rows to table; read it from plan.results
clab: 'logit_diff' has no result: the run stopped before its step; steps.logit_diff: out of CUDA memory …
```

## `clab describe`

```text
clab describe [WHAT] [--markdown] [--verbose]
```

Prints the format's tables, read off the code. `FORMAT.md` is the same content rendered as one page.

| `WHAT` | prints | in FORMAT.md |
|---|---|---|
| `records` | each record's fields, types and defaults | [root keys](../../FORMAT.md#root-keys) |
| `steps` | each step kind's fields, and the fit's `optimizer` and `early_stop` | [step kinds](../../FORMAT.md#step-kinds) |
| `ops` | every op's arguments and what it does to the axes, by group | [ops](../../FORMAT.md#ops) |
| `components` | every component, its layout and what narrows it | [components](../../FORMAT.md#components) |
| `featurizers` | each featurizer kind's fields and ops | [featurizer kinds](../../FORMAT.md#featurizer-kinds) |
| `pos` | the `pos` grammar and its rules | [pos](../../FORMAT.md#pos) |
| `engines` | each engine and its options | [engines and run options](../../FORMAT.md#engines-and-run-options) |
| omitted | all of the above | |

`--markdown` writes `FORMAT.md` at the repository root, beside the package, from whatever directory it runs in, ignoring `WHAT`; it prints nothing, and with `--verbose` it prints `wrote <path>`.

## From Python

`clab.cli.main(argv)` runs one command and returns its exit status: `main(["table", "out/patching", "iia"])` prints the table and returns 0.
