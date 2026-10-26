---
title: Output
one_liner: "Reading a run back: the files a run writes, clab show and clab table and their columns, plan.table in Python, why tabling needs the data files, the manifest, and the run record with where every read and write landed."
tags: [usage, output, table, manifest]
related: [docs/usage/running.md, docs/usage/positions.md, docs/usage/sweeps.md, docs/usage/bundles.md, docs/errors/alignment-missing.md, docs/reference/cli.md, docs/reference/python-api.md, docs/concepts/plan-lifecycle.md]
sources: [clab/output/__init__.py, clab/output/table.py, clab/plan.py, clab/cli.py]
---

# Output

A run writes one directory: a file per saved value and `manifest.json`, which records the compiled plan, how the run went, and which token every read and write landed on. `clab show` prints what the directory holds, `clab table` prints saved values as rows labelled with their tokens and data, and `Plan.load` reads it all back in Python. The exact file rules are in [FORMAT.md](../../FORMAT.md#saves-and-the-output).

## The directory

```text
out/patching/
  iia.safetensors
  logit_diff.safetensors
  manifest.json
```

- **One file per name in `saves`**, named after it: `<name>.safetensors` for a tensor or a dict of tensors, `<name>.json` for anything else. A value that holds text is JSON: `top_k` as a step holds its decoded `tokens`, so `documents/readout.json` writes `top3.json`.
- **A swept value is one file** holding every point, keyed by coordinate: `pos=-1`, `pos=-2`, or `layers=0,pos=-1` over two axes ([sweeps.md](sweeps.md)).
- **A saved featurizer is a bundle** whose file carries the header a loading document checks ([bundles.md](bundles.md)).
- **`manifest.json`** is the compiled plan, with sites inlined, sweeps expanded and each entry's layout (`dims`) filled in, plus `run`, plus `results`: for each save its `file`, the `step` that made it, `dims`, sweep `axes`, `shape` and `dtype`. It holds no values, and no rows from a file or the hub. Inline rows stay in it.

Where the directory goes and what happens when you reuse one is [running.md](running.md#where-output-goes).

## `clab show`

```text
$ clab show out/patching
patching: run on nnterp, device cpu, batch size 8, started 2026-10-26T13:48:04, 0.169 s
  served: float32 on cpu
  versions: clab 0.0.1+e2499ea, nnsight 0.8.1, nnterp ..., torch ..., transformers ..., python ...
  result      step        dims       axes  shape
  iia         iia         rows, pos  —     [4,1]
  logit_diff  logit_diff  rows, pos  —     [4,1]
```

The first lines are the run record: the engine, its device and options, the batch size, when it started and how long it took; the dtypes and devices of the model's parameters; and the versions of the packages that ran it. Then each saved value with the step that made it, its axes (`dims`), the sweep labels it varies over (`axes`) and its shape, per coordinate when swept. `served:` is what the model actually ran in, which is the line to check when a device map or a server places the model; after a run on an NDIF server the first line says `device placed by the server`, and `versions:` are the server's.

## `clab table`

`clab table OUT NAME [NAME ...]` prints saved values as rows, one per example and position, aligned; `--csv` prints CSV with every cell whole (aligned cells are cut at 48 characters).

```text
$ clab table out/generated_answer p_answer
example_id  row  pos  pos_token  call  input                              cf_answer  value                   alignment_missing
substr      0    12   ▁substr    2     If today is Thursday, tomorrow is   Sunday    2.9666021873708814e-05
energ       1    12   calc       4     If today is Friday, tomorrow is     Monday    3.15390425384976e-05
Friday      2                          If today is Saturday, tomorrow is   Friday                            true
Saturday    3                          If today is Sunday, tomorrow is     Saturday                          true
```

| column | present when | holds |
|---|---|---|
| `at` | the value's step is swept | the point's coordinates as an object of values, such as `{"pos": -1}` ([sweeps.md](sweeps.md#reading-swept-results)) |
| `epochs` | a fit's `history` | the epoch, from 0 |
| `example_id` | the dataset has that column | the row's `example_id` |
| `row` | the value has a rows axis | the row's index in the dataset, after `split`, `shuffle` and `take` |
| `layer` | the value has a `layers` axis | the model's layer, not the index along the axis |
| `pos`, `pos_token` | the value has a positions axis | the position the read resolved to, as `pos` counts (BOS 0, then the prompt, then in a generation the generated tokens), and the token there |
| `call` | the read was in a generation | the forward call that holds the position: 0 the prefill, k decode call k |
| `key`, `key_token` | an attention pattern | each key position and its token, the row's own keys only |
| dataset columns | always, with rows | the columns the value's chain of steps used: the input, the answers |
| `value` | one name | the element, or the remaining axes as nested lists; with several names, one column per name |
| `alignment_missing` | some row was not reached | `true` on a row a read beyond the prefill never reached, with no value ([alignment-missing.md](../errors/alignment-missing.md)) |

- **Several names join on their shared axes.** A value with fewer axes repeats beside every row it agrees with: a per-layer `mean` beside a per-row metric repeats on every row of its layer. Names with rows of two different datasets are refused. Names that share no value of an axis, such as two reads at different positions, or values swept over different axes, join to no rows: the table prints nothing and exits 0. Table them one at a time.
- **A value without rows** gives one row per element, labelled with its axis indices; a scalar is one row holding `value`.
- **Positions past a row's own are left out.** A ragged read on a short row has fewer table rows, not NaN rows.
- **Only saved names can be tabled**, and a bundle cannot be: read it from `plan.results`.

## In Python

```python
from clab import Plan

plan = Plan.load("out/patching")
plan.results["logit_diff"]                # tensor [4, 1]
plan.table("iia", "logit_diff")           # the rows clab table prints, as dicts
plan.table("iia", columns=["entity"])     # choose the dataset columns; [] for none
plan.run.positions["patched.logits"]      # [[10], [8], [8], [8]]
```

`plan.table` takes the same names and returns a list of dicts; `columns=` replaces the default dataset columns, which the command line cannot do. A swept result is a dict keyed by coordinate; the key's form in each place is in [sweeps.md](sweeps.md#reading-swept-results). A value loaded from JSON is a tensor again where it was one.

## Tabling needs the rows

`clab table` and `plan.table` label rows with dataset columns, so they need each dataset's rows. Inline rows are in the manifest. A file's rows are read again from `data/` under the working directory, and a hub dataset's from the hub or its cache, each checked against the digest compile recorded. Run `clab table` from the corpus root, and leave the data unchanged after the run:

```text
clab: data.pairs.path: no file data/weekdays/train.json under the corpus root
clab: data.pairs: data/weekdays/train.json is not what the run read (its rows' digest differs)
```

`clab show` and `plan.results` need no rows. A run that loads a bundle reads it again from its `file_path` for `show` and `table` alike, so run them from the corpus root with the bundle still there ([bundles.md](bundles.md#reading-back-a-run-that-loads-a-bundle)).

## The run record

`plan.run`, saved as `run` in the manifest, says how the plan ran and where everything landed:

| field | holds |
|---|---|
| `engine`, `batch_size`, `device`, `options` | how it was asked to run; `device` is None when an NDIF server placed the model, and `options` holds the engine's options after their defaults |
| `dtype`, `devices` | the distinct dtypes and devices of the model's parameters where it ran |
| `versions` | clab with its commit, nnsight, nnterp, torch, transformers and python, on the machine that ran the model |
| `started`, `seconds` | when the run began and how long it took |
| `num_layers` | the model's layer count, which a site's `layers` counts against |
| `tokens` | each input's token strings per row, under the input's name (`pairs.input`); each `generate` step's generated tokens per row, under the step's name |
| `positions` | each read and write's resolved positions per row, under `<step>.<entry>` (in a fit's eval, `<fit>.<eval>.<step>.<entry>`); `"alignment_missing"` for a row it did not reach |

A swept step's `tokens` and `positions` are keyed by coordinate. From `documents/patching.json`:

```json
"tokens": {"pairs.input": [["<s>", "▁If", "▁today", "▁is", "▁Th", "urs", "day", ",", "▁tom", "orrow", "▁is"], ...]},
"positions": {"counterfactual.v_cf": [[8], [8], [10], [8]], "patched.patch": [[10], [8], [8], [8]], ...}
```

`positions` is how you check that a `pos` landed on the token you meant before trusting a number; [positions.md](positions.md) has the recipes. Generated rows that stopped early show up here as `alignment_missing`.

## Pitfalls

- **The table shows the columns the value's steps used, not every column.** `patching.json`'s `iia` shows `input` and `cf_answer`; to see `entity` beside it, use `plan.table("iia", columns=["entity"])`.
- **`row` counts rows after `shuffle` and `take`**, not the file's order. Give rows an `example_id` to keep them identifiable ([data.md](data.md#example_id)).
- **A long cell is cut in the aligned view.** Use `--csv` or `plan.table` before reading a list value or long text.
