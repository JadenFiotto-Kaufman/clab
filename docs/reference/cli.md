---
title: Command Line
one_liner: "Every clab command and flag (run, show, table, describe), with defaults, exit statuses and refusals."
tags: [reference, cli]
related: [docs/usage/running.md, docs/usage/output.md, docs/usage/engines.md, docs/reference/python-api.md, FORMAT.md]
sources: [clab/cli.py, clab/__init__.py, clab/docs.py]
---

# Command Line

`clab` has four commands: `run`, `show`, `table` and `describe`. Inside the repository's environment, call it as `uv run clab <command>`. This page lists every argument; the workflow around them is [running.md](../usage/running.md), and reading output is [output.md](../usage/output.md).

Every command exits 0 on success. A refusal or error that clab reports is printed to stderr as `clab: <message>`, and the command exits 1. A malformed command line (an unknown flag, a missing argument, a value outside a flag's choices) gets argparse's usage message and exit status 2. A few errors come from Python, not clab, as a traceback: `clab run` on a document path that does not exist raises `FileNotFoundError`, and the others are listed in [errors/index.md](../errors/index.md#errors-that-are-not-clabs).

## `clab run`

```text
clab run [--engine {nnterp,hooks,ndif}] [--out OUT] [--batch-size N] [--device D] [-o KEY=VALUE ...] DOCUMENT
```

Compiles the document, loads the engine, runs every step, and writes the output directory.

| argument | default | meaning |
|---|---|---|
| `DOCUMENT` | required | the document's JSON file. Data paths and bundles resolve against the nearest directory above it that holds `data/` |
| `--engine` | `nnterp` | the engine: `nnterp`, `hooks` or `ndif` ([engines.md](../usage/engines.md)) |
| `--out` | `out/<document file name>` in the current directory | the output directory, created when missing and never cleared |
| `--batch-size` | `8` | how many rows go through the model per forward or generation call; a positive integer |
| `--device` | `cpu` | where the engine places the model: one torch device (`cuda:0`) or a device map transformers computes (`auto`, `balanced`, ...). Under `--engine ndif` with a real server the server places it, and any value but `cpu` is refused |
| `-o KEY=VALUE` | none | one of the engine's own options, repeatable. `VALUE` is read as JSON (`true`, `2`, `{"a": 1}`) and otherwise as text (`local`). `clab describe engines` lists each engine's options; `ndif` has `remote` |

Refused before the model loads:

```text
clab: -o device: it is given as --device
clab: -o batch_size: it is given as --batch-size
clab: -o 'foo': an option is KEY=VALUE
clab: the nnterp engine takes no 'foo'; its options: device
```

On success it prints one line naming the run, one line per saved file (the file, the step that made it, the value's shape) and the manifest's path:

```text
patching on ndif (batch size 8, device cpu, remote "local") -> out/patching
  iia.safetensors              iia              [4, 1]
  logit_diff.safetensors       logit_diff       [4, 1]
  out/patching/manifest.json
```

Under `--engine ndif`, a refusal raised on the server arrives as nnsight's `RemoteError`, and `clab run` prints its last line.

## `clab show`

```text
clab show OUT
```

Prints the run record of output directory `OUT` (engine, device and options, batch size, start time and duration, the served dtype and devices, the versions) and, per saved value, the step that made it, its dims, its sweep axes and its shape. It reads `OUT/manifest.json`, and exits 1 when there is none. A run that loaded a bundle has it read again from its `file_path`, relative to the current directory, so run `show` from the corpus root ([bundles.md](../usage/bundles.md#reading-back-a-run-that-loads-a-bundle)).

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
clab: 'fit.held_out.iia' has a row per example of 'test' and 'fit.train.iia' of 'train'; a table joins the values of one dataset
clab: a bundle has no rows to table; read it from plan.results
```

## `clab describe`

```text
clab describe [WHAT] [--markdown]
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

`--markdown` writes `FORMAT.md` into the current directory, ignoring `WHAT`, and prints `wrote FORMAT.md`. Run it from the repository root to regenerate the repository's copy.

## From Python

`clab.cli.main(argv)` runs one command and returns its exit status: `main(["table", "out/patching", "iia"])` prints the table and returns 0.
