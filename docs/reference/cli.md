---
title: Command Line
one_liner: "Every nnspec command and flag (run, compile, show, table, describe), with defaults, exit statuses and refusals."
tags: [reference, cli]
related: [docs/usage/running.md, docs/usage/output.md, docs/usage/engines.md, docs/reference/python-api.md, FORMAT.md]
sources: [nnspec/cli.py, nnspec/__init__.py, nnspec/docs.py, nnspec/steps/__init__.py]
---

# Command Line

`nnspec` has five commands: `run`, `compile`, `show`, `table` and `describe`. They are spelled here as inside the repository's venv ([README.md](../../README.md#install) has the spelling from outside it). This page lists every argument and refusal; the workflow around them is [running.md](../usage/running.md), and reading output is [output.md](../usage/output.md).

Every command exits 0 on success. A refusal or error that nnspec reports is printed to stderr as `nnspec: <message>`, and the command exits 1. A malformed command line (an unknown flag, a missing argument, a non-integer `--batch-size`, an unknown `WHAT`) gets argparse's usage message and exit status 2 ([errors/index.md](../errors/index.md#errors-that-are-not-nnspecs)).

## `nnspec run`

```text
nnspec run [--engine ENGINE] [--out OUT] [--batch-size N] [--padding dataset|batch|sorted] [--readout full|at-reads] [--device D] [-o KEY=VALUE ...] DOCUMENT
```

Compiles the document, loads the engine, runs every step, and writes the output directory.

| argument | default | meaning |
|---|---|---|
| `DOCUMENT` | required | the document's JSON file. Data paths, bundles and a `load`'s `from` resolve against its corpus root: the nearest directory holding `data/`, from the document's own directory up, else the directory above the document's own ([running.md](../usage/running.md#the-corpus-root)) |
| `--engine` | `nnterp` | the engine: `nnterp`, `hooks`, `ndif`, `vllm`, or one registered beside them ([engines.md](../usage/engines.md)); another name is refused by name |
| `--out` | `out/<document file name>` in the current directory | the output directory, created when missing and never cleared |
| `--batch-size` | `8` | how many rows go through the model per forward or generation call; a positive integer |
| `--padding` | `dataset` | how each batch is padded: `dataset`, the default, the reference the others are compared against; `batch`; or `sorted` ([running.md](../usage/running.md#batch-size-and-padding)) |
| `--readout` | `full` | how a forward computes the logits it reads: `full`, the default, the reference the others are compared against; or `at-reads` ([running.md](../usage/running.md#the-head-at-the-positions-read)) |
| `--device` | the engine's own: `cpu`, or `cuda:0` under `--engine vllm` | where the engine places the model: one torch device (`cuda:0`) or a device map transformers computes (`auto`, `balanced`, ...). Under `-o tp`, `cuda` or `cpu`. Under `--engine ndif` with a real server the server places it, and any value but `cpu` is refused. Under `--engine vllm` it is `cuda:0`, and any other is refused |
| `-o KEY=VALUE` | none | one of the engine's own options, repeatable; `nnspec describe engines` lists each engine's. `VALUE` is read as JSON (`true`, `2`, `{"a": 1}`) and otherwise as text (`local`) |

`-o tp=N` runs under torchrun, one process per rank ([running.md](../usage/running.md#sharding-a-model--o-tp)).

Refused before the model loads:

```text
nnspec: no document at 'documents/nope.json'
nnspec: batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)
nnspec: padding 'rows': one of 'dataset', 'batch', 'sorted' (--padding)
nnspec: readout 'last': one of 'full', 'at-reads' (--readout)
nnspec: device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes ('auto', 'balanced', 'balanced_low_0', 'sequential')
nnspec: device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)
nnspec: -o device: it is given as --device
nnspec: -o batch_size: it is given as --batch-size
nnspec: -o padding: it is given as --padding
nnspec: -o readout: it is given as --readout
nnspec: -o 'foo': an option is KEY=VALUE
nnspec: unknown engine 'bogus'; valid: nnterp, hooks, ndif, vllm
nnspec: the nnterp engine takes no 'foo'; its options: tp (where the model runs is `device`, given as --device)
nnspec: tp 2 shards the model over 2 processes, one per rank, that torchrun starts: `torchrun --nproc-per-node=2 --no-python nnspec run <document> -o tp=2`; this process is not one of them (WORLD_SIZE is unset)
```

Offline, a checkpoint whose `revision` the cache lacks is refused as the engine loads it (the nnterp engine's weights, and a model key its loader rejects, at the first run that calls the model): `nnspec: model.key '<key>' at revision '<revision>': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; …`. What compile and preflight refuse, and when, is [running.md](../usage/running.md#check-before-you-run).

On success it prints one line naming the run with its engine options, one line per saved file (the file, the step that made it, the value's shape, counted over its points when swept) and the manifest's path:

```text
patching on ndif (batch size 8, device cpu, remote "local") -> out/patching
  iia.safetensors         iia         [4,1]
  logit_diff.safetensors  logit_diff  [4,1]
  out/patching/manifest.json
pos_sweep on nnterp (batch size 8, device cpu, tp 1) -> out/pos_sweep
  iia.safetensors         iia         3 points, each [4,1]
  logit_diff.safetensors  logit_diff  3 points, each [4,1]
  out/pos_sweep/manifest.json
```

Under `--engine ndif`, a refusal raised on the server arrives as nnsight's `RemoteError`, and `nnspec run` prints its last line. A step that runs out of CUDA memory stops the run: `nnspec run` prints the refusal, which ends `the steps before it are saved in <out>`, and exits 1 with the finished steps written ([running.md](../usage/running.md#when-a-step-runs-out-of-memory)).

## `nnspec compile`

```text
nnspec compile DOCUMENT
```

Reads and compiles the document, as `nnspec run` does before any engine loads, and nothing more: no engine, no model, no output directory. It reads the data files, a hub dataset's rows (from the cache under `HF_HUB_OFFLINE=1`, otherwise from the hub first) and bundle files, and imports transformers only where a `generate` step's options need checking. `DOCUMENT` resolves its data and bundles as under `nnspec run`. It takes no other argument, since none of `nnspec run`'s options reaches compile. On success it prints the plan as `nnspec show` prints a compiled plan that has not run, then exits 0. Each row is one run of a step, in run order, with the step's kind, a forward's or generation's input, the point it runs at (`—` unswept) and the points it serves:

```text
patching: compiled, not run
  step            kind     input                           at  serves
  counterfactual  forward  pairs.counterfactual_inputs[0]  —   —
  patched         forward  pairs.input                     —   —
  iia             op       —                               —   —
  logit_diff      op       —                               —   —
```

A refusal is the line `nnspec run` prints for the same document, on stderr, and exit status 1:

```text
nnspec: no document at 'documents/nope.json'
nnspec: steps.iia.of: 'patched.nothing' names no value produced before this step; known: counterfactual.v_cf, patched.logits, patched.patch
nnspec: model: 'torch_dtype': the dtype is spelled `dtype`
```

What it cannot tell is [running.md](../usage/running.md#check-before-you-run).

## `nnspec show`

```text
nnspec show OUT
```

Prints output directory `OUT`'s run record, then three tables: what each step cost, where each entry resolved, and each saved value ([output.md](../usage/output.md#nnspec-show) reads them). Refused:

```text
nnspec: no run at 'OUT' (no manifest.json); run the document first
```

A run that loaded a bundle has it read again from its `file_path`, relative to the current directory, so run `show` from the corpus root ([bundles.md](../usage/bundles.md#reading-back-a-run-that-loads-a-bundle)).

## `nnspec table`

```text
nnspec table [--csv] OUT NAME [NAME ...]
```

| argument | meaning |
|---|---|
| `OUT` | a run's output directory |
| `NAME` | one or more saved names; several are joined on the axes they share |
| `--csv` | CSV with a header row, every cell whole. Without it, aligned columns with cells over 48 characters cut short with `…` |

The rows and columns are [output.md](../usage/output.md#nnspec-table). It reads dataset rows to label the table: a file's from `data/` under the current directory, a hub dataset's from the hub or cache ([output.md](../usage/output.md#tabling-needs-the-rows)). Refused:

```text
nnspec: 'patched.logits' was not saved; add it to saves
nnspec: 'held_iia' has a row per example of 'test' and 'fit.train.iia' of 'train'; a table joins the values of one dataset
nnspec: 'iia' has a pos column counting positions of pairs.input, and 'counterfactual.v_cf' one counting positions of pairs.counterfactual_inputs[0]: the column does not mean one thing for both, so they do not join; table them one at a time
nnspec: 'fit.rot' is a trained featurizer's bundle, which has no rows to table; read it from plan.results
nnspec: 'top' is a top_k over rows, whose rows are its picks, and a top_k over rows tables alone; table it by itself
nnspec: 'f.a' and 'f.b' share no row and pos together: on no row do they hold the same pos, so they join to no row; table them one at a time, or read at a pos both hold
nnspec: 'f.d' and 'f.e' share no layer: 'f.d' holds layer [1] and 'f.e' layer [0], so they join to no row; table them one at a time, or read at a layer both hold
nnspec: 'g.never' reached no row (each is alignment_missing), so it holds no position to meet the other's and they join to no row; table them one at a time
nnspec: 'topm' has no rows to table: it is laid out ['k'], a top_k over a value with no rows axis, its ids and values one list each; read it from plan.results
nnspec: 'logit_diff' has no result: the run stopped before its step; steps.logit_diff: out of CUDA memory …
nnspec: data.pairs.path: no file data/weekdays/train.json under the corpus root
nnspec: data.pairs: data/weekdays/train.json is not what the run read (its rows' digest differs)
```

## `nnspec describe`

```text
nnspec describe [WHAT] [--markdown] [--verbose]
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

`nnspec.cli.main(argv)` runs one command and returns its exit status: `main(["table", "out/patching", "iia"])` prints the table and returns 0.
