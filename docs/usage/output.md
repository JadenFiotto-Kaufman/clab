---
title: Output
one_liner: "How do I read a run back? clab show, clab table, plan.results, the manifest's positions, the run record, and loading a saved value elsewhere."
tags: [usage, output, table, manifest]
related: [docs/usage/running.md, docs/usage/positions.md, docs/usage/sweeps.md, docs/usage/bundles.md, docs/patterns/cached-harvest.md, docs/errors/alignment-missing.md, docs/reference/cli.md, docs/reference/python-api.md, docs/concepts/plan-lifecycle.md]
sources: [clab/output/__init__.py, clab/output/table.py, clab/plan.py, clab/cli.py]
---

# Output

A run writes one directory: a file per saved value, `manifest.json`, which records the compiled plan, how the run went, and which token every read and write landed on, and `tokens.json`, every input's tokens. `clab show` prints what the directory holds, `clab table` prints saved values as rows labelled with their tokens and data, and `Plan.load` reads it all back in Python. The exact file rules are in [FORMAT.md](../../FORMAT.md#saves-and-the-output).

## The directory

```text
out/patching/
  iia.safetensors
  logit_diff.safetensors
  manifest.json
  tokens.json
```

- **One file per name in `saves`**, named after it: `<name>.safetensors` for a tensor or a dict of tensors, `<name>.json` for anything else. A value that holds text is JSON: `top_k` as a step holds its decoded `tokens`, so `documents/readout.json` writes `top3.json`.
- **A swept value is one file** holding one entry per point of the sweep axes the value depends on (its `axes`), keyed by coordinate: `pos=-1`, `pos=-2`, or `layers=0,pos=-1` over two axes. A value that depends on none of its step's sweeps is one tensor: `documents/generate_probe.json` sweeps its reads over three frames, and its generated ids, `patched`, are one `[4, 3]` tensor ([sweeps.md](sweeps.md)).
- **A saved featurizer is a bundle** whose file carries the header a loading document checks ([bundles.md](bundles.md)).
- **`manifest.json`** is the compiled plan as the run completed it, with the run record and a record of each save ([below](#the-manifest)). It holds no values.
- **`tokens.json`** is the run record's `tokens`: each input's token strings per row and what each generation's rows said. It holds every token of every row, so it sits beside the manifest rather than in it. `Plan.load` reads it back into `plan.run.tokens`, and `clab table` labels `pos_token` from it. A directory without it is refused: `<out>/tokens.json is missing, and the manifest holds no tokens; …`.

Where the directory goes and what happens when you reuse one is [running.md](running.md#where-output-goes).

### The manifest

| key | holds |
|---|---|
| `description`, `model`, `data`, `steps`, `saves`, `document`, `compiled` | the compiled plan as the run completed it: sites inlined, sweeps expanded, each record's derived fields filled in ([FORMAT.md](../../FORMAT.md#derived-fields)). `data` holds each dataset's `sha256` and `n_rows`, and inline rows, but no rows from a file or the hub |
| `run` | the run record, without its `tokens` ([below](#the-run-record)) |
| `results` | for each save its `file`, the `step` that made it (an eval's member's, its eval step: `fit.held_out.ce`), its layout and labels, the sweep `axes` it depends on, `shape` and `dtype` |
| `positions` | each distinct list of resolved positions once, which a read's or write's record names by index ([below](#where-each-read-and-write-landed)) |

The manifest leaves out every field at its default but `model.dtype`, `run.padding` and `run.readout`, so a script reading `manifest.json` raw takes a missing key as the default. `Plan.load(out)` and `clab.output.manifest(out)`, the directory's one reader, put each record's positions back in place.

## `clab show`

```text
$ clab show out/patching
patching: run on nnterp, device cpu, tp 1, batch size 8, padding dataset, readout full, started 2026-11-05T17:58:19, 0.231 s
  served: float32 on cpu
  versions: clab 0.0.1+…, nnsight …, nnterp …, torch …, transformers …, python …
  step            at  serves  seconds  memory  forwards  tokens  real
  counterfactual  —   —       0.003    —       1         44      38
  patched         —   —       0.004    —       1         44      38
  iia             —   —       0.001    —       0         0       0
  logit_diff      —   —       0.000    —       0         0       0
  entry                at  positions
  counterfactual.v_cf  —   [8], [8], [10], [8]
  patched.patch        —   [10], [8], [8], [8]
  patched.logits       —   [10], [8], [8], [8]
  result      step        dims       axes  shape
  iia         iia         rows, pos  —     [4,1]
  logit_diff  logit_diff  rows, pos  —     [4,1]
```

The first lines are the run record. The first line names the engine, its device and options, the batch size, padding and readout, when it started and how long it took. `served:` gives the dtypes and devices of the model's parameters, and `versions:` the packages that ran it. A `stopped:` line follows when a step ran out of CUDA memory ([running.md](running.md#when-a-step-runs-out-of-memory)). Then three tables:

- **What each step cost**, one row per run of a step, at the point that ran it. `serves` lists the points that run served (`—` for itself alone, `all 3 points` for every point of the step). Then come its seconds, its peak memory on each CUDA device (`—` off CUDA), the model calls it made, the tokens they fed (`tokens`, padding included) and the rows' own of those (`real`). A forward or generation runs once for every set of points that differ only in what they read, so a read-only sweep is one row.
- **Where each entry resolved**, one row per entry (a `load`'s, the read it carries) and distinct positions, with the point it is at or how many points share it. Each row's positions print in row order: `[10] on all 4 rows` where every row agrees, `—` for a row the entry did not reach. Past twelve rows that differ it prints the first three rows', `…`, the last row's and the count (`[8], [10,11,12], [8], …, [8] (42 rows; clab table shows each)`). A row of more than twelve positions prints as its first three and its last (`[0,1,2,…,14]`). `clab table` and the manifest hold every row's.
- **Each saved value**: the step that made it, its layout (`dims`), the sweep axes it depends on (`axes`) and its shape, counted over its points when swept (`3 points, each [4,1]`; the manifest holds each point's).

Seconds vary from run to run. A step that calls the model is timed with the GPU's queue drained at both ends. An op step or a `load` is timed as the host runs it, so its seconds leave out what its kernels still had to do, which the next step that calls the model waits for before its own timer starts.

From `documents/generate_probe.json`, whose reads sweep three frames of one generation:

```text
  step            at       serves        seconds  memory  forwards  tokens  real
  counterfactual  —        —             0.014    —       3         52      46
  patched         frame=0  all 3 points  0.012    —       3         52      46
  ...
  result    step      dims       axes   shape
  p_answer  p_answer  rows, pos  frame  3 points, each [4,1]
  patched   patched   rows, new  —      [4,3]
```

A value that holds text prints its whole layout, laid out like the ids: a `top_k` step's `tokens` read `[4,1,3]` beside `"ids":[4,1,3]`. `served:` is what the model actually ran in, which is the line to check when a device map, a quantized `dtype` or a server places the model (`bfloat16, uint8 on cpu` under `nf4`). After a run on an NDIF server the first line says `device placed by the server`, and `versions:` are the server's.

## `clab table`

`clab table OUT NAME [NAME ...]` prints saved values as rows, one per example and position, aligned; `--csv` prints CSV with every cell whole (aligned cells are cut at 48 characters). Read a list value or long text with `--csv` or `plan.table`.

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
| `row` | the value has a rows axis | the row's index in the dataset after `split`, `shuffle`, `skip` and `take`, not the file's order; give rows an `example_id` to keep them identifiable ([data.md](data.md#example_id)) |
| `layer` | the value has a `layers` axis | the model's layer, not the index along the axis |
| `pos`, `pos_token` | the value has a positions axis | the position the read resolved to, as `pos` counts (BOS 0, then the prompt, then in a generation the generated tokens), and the token there; for a `permute`'s value, whose rows hold other rows' positions, `pos` alone, the plain index, and no columns of the read's input |
| `call` | the read was in a generation | the forward call that holds the position: 0 the prefill, k decode call k |
| `key`, `key_token` | an attention pattern | each key position and its token, the row's own keys only |
| dataset columns | always, with rows | the columns the value's chain of steps used: the input, the answers; `plan.table(..., columns=[...])` chooses others |
| `value` | one name | the element, or the remaining axes as nested lists; with several names, one column per name |
| `alignment_missing` | some row was not reached | `true` on a row a read beyond the prefill never reached, with no value ([alignment-missing.md](../errors/alignment-missing.md)) |
| `k` | a `top_k` over `rows` | the pick's rank, from 0 (below) |

- **Several names join on the keys they share** (the sweep coordinates, `row`, `layer`, `pos`, `key`) and broadcast over the rest, as [FORMAT.md](../../FORMAT.md#saves-and-the-output) states. A value with fewer axes repeats beside every row it agrees with: a per-layer `mean` beside a per-row metric repeats on every row of its layer. Values swept over different axes cross: `iia` over `pos` beside a match over `layers` gives a row per example at every pair of points. Two `fit.history` values pair by epoch.
- **Two reads of one input at different positions share no `pos`**, so they join to no rows: the table prints nothing and exits 0. Table them one at a time.
- **Some joins are refused by name.** Names with rows of two datasets (`'held_iia' has a row per example of 'test' and 'fit.train.iia' of 'train'; a table joins the values of one dataset`), and names whose `pos` counts the positions of different inputs (`'iia' has a pos column counting positions of pairs.input, and 'counterfactual.v_cf' one counting positions of pairs.counterfactual_inputs[0]: …`).
- **A value without rows** gives one row per element, labelled with its axis indices; a scalar is one row holding `value`.
- **Positions past a row's own are left out.** A ragged read on a short row has fewer table rows, not NaN rows.
- **Only saved names can be tabled, and a bundle cannot be**: `'fit.rot' is a trained featurizer's bundle, which has no rows to table; read it from plan.results`. A `top_k` value with no rows (a `top_k` of a `mean`) is refused the same way (`'topm' has no rows to table: it is laid out ['k'], a top_k over a value with no rows axis, …`); read it as `Plan.load(OUT).results[name]`. A save the run stopped before has no result, and `clab table` refuses it naming the stop.
- **A `top_k` over `rows`**, alone or in a list of axes, gives a row per pick. The value's other axes come first as index columns, as a value without rows has them, then `k`. Then comes the row the ranked value's table has at the pick (`example_id`, `row`, `layer`, `pos` and `key` with their tokens, `call` in a generation). Then any other ranked axis's index under its name (`vocab_size`, `features`), the row's dataset columns, and `value`, the ranked value there. Its `pos` is the position the read resolved to, whether `pos` was ranked or kept. A pick past the entries there (id -1) is left out.
- **A `top_k` over `rows` tables alone**: joined with another name it is refused (`'top' is a top_k over rows, whose rows are its picks, and a top_k over rows tables alone; table it by itself`). A `top_k` over `pos` alone keeps `rows` and tables a row per example, its `k` picks in `value`.

## In Python

```python
from clab import Plan

plan = Plan.load("out/patching")
plan.results["logit_diff"]                # tensor [4, 1]
plan.table("iia", "logit_diff")           # the rows clab table prints, as dicts
plan.table("iia", columns=["entity"])     # choose the dataset columns; [] for none
plan.steps["patched"].interventions["logits"].positions   # {'pairs': [[10], [8], [8], [8]]}
plan.run.costs["patched"]                 # {'seconds': 0.006, 'memory': None, 'forwards': 1, 'tokens': 44, 'real': 38}
```

`plan.table` takes the same names and returns a list of dicts; `columns=` replaces the default dataset columns, which the command line cannot do. A swept result is a dict keyed by coordinate; the key's form in each place is in [sweeps.md](sweeps.md#reading-swept-results). A value loaded from JSON is a tensor again where it was one. `Plan.load` needs no data files, but it rebuilds the plan, so a run that loaded a bundle loads only where that bundle's `file_path` resolves ([bundles.md](bundles.md#reading-back-a-run-that-loads-a-bundle)).

## Loading a saved value into another document

A `load` step brings one saved value into another document with its manifest record: `{"kind": "load", "from": "out/patching", "value": "iia"}` is `iia` as the run saved it, with its layout, its dataset and its sweep axes. The recipe and what the load checks are [cached-harvest.md](../patterns/cached-harvest.md). A bundle loads as a featurizer instead ([bundles.md](bundles.md)).

## Tabling needs the rows

`clab table` and `plan.table` label rows with dataset columns, so they need each dataset's rows. Inline rows are in the manifest. A file's rows are read again from `data/` under the working directory, and a hub dataset's from the hub or its cache, each checked against the digest compile recorded. Run `clab table` from the corpus root, and leave the data unchanged after the run:

```text
clab: data.pairs.path: no file data/weekdays/train.json under the corpus root
clab: data.pairs: data/weekdays/train.json is not what the run read (its rows' digest differs)
```

`clab show` and `plan.results` need no rows. A run that loads a bundle reads it again from its `file_path` for `show` and `table` alike, so run them from the corpus root with the bundle still there ([bundles.md](bundles.md#reading-back-a-run-that-loads-a-bundle)).

## The run record

`plan.run`, saved as `run` in the manifest, says how the plan ran and what each step cost:

| field | holds |
|---|---|
| `engine`, `batch_size`, `padding`, `readout`, `device`, `options` | how it was asked to run; `device` is None when an NDIF server placed the model, and `options` holds the engine's options after their defaults |
| `dtype`, `devices` | the distinct dtypes and devices of the model's parameters where it ran |
| `versions` | clab with its commit, nnsight, nnterp, torch, transformers and python, on the machine that ran the model |
| `threads` | torch's intra-op thread count on that machine at the run's start (`torch.get_num_threads()`); `clab show` does not print it |
| `started`, `seconds` | when the run began and how long it took; a shell's weight load (the nnterp engine's, at its first run that calls the model) happens before the clock starts |
| `sizes` | the model's sizes (`num_layers`, `hidden_size`, `num_heads`, ...), which a site's `layers` and `select` count among |
| `tokens` | each input's token strings per row, under the input's name (`pairs.input`); each `generate` step's generated tokens per row, under the step's name (an eval's generation under its eval step's, `fit.<eval>.<gen>`); a swept step's keyed by coordinate; written to `tokens.json`, not the manifest |
| `costs` | each step's runs by name, a fit's eval steps' under `<fit>.<eval>.<step>` after the fit's: `seconds`, the peak `memory` on each CUDA device in bytes (null off CUDA and on a server), the model calls in `forwards`, the input `tokens` they fed, padding included, and of those the rows' own, `real`; a swept step's keyed by the coordinate of the point that ran, with the points it served in `serves` |
| `stopped` | the refusal of the step that ran out of CUDA memory, when one did; absent from the manifest otherwise, and `None` in Python |

From `documents/patching.json`, its `tokens.json` and the manifest's `run.costs`:

```json
{"pairs.counterfactual_inputs[0]": [...], "pairs.input": [["<s>", "▁If", "▁today", "▁is", "▁Th", "urs", "day", ",", "▁tom", "orrow", "▁is"], ...]}
"costs": {"counterfactual": {"seconds": 0.004, "memory": null, "forwards": 1, "tokens": 44, "real": 38}, ...}
```

`tokens` is what the model was fed and `real` the part of it that is the rows' own; the rest is padding. A step's `seconds` and `memory` follow `tokens`, not `real`. A large gap between them is padding, which `--padding sorted` cuts while laying out every value as before ([running.md](running.md#batch-size-and-padding)).

## Where each read and write landed

Each read's and write's record in the manifest's `steps` holds `positions`, under the dataset whose rows they are (an eval step's under its eval's). Its value is the index of its resolved positions per row in the manifest's top-level `positions`, which holds each distinct list once, so records that landed alike share one; `"alignment_missing"` stands for a row it did not reach. A swept step is a list of its points, each with its own records.

From `documents/patching.json`, `steps.patched.interventions.logits.positions` is `{"pairs": 1}` and the manifest's `positions[1]` is `[[10], [8], [8], [8]]`, which `clab show` prints as `patched.logits  —  [10], [8], [8], [8]`; `patched.patch` names the same list. `Plan.load` and `clab.output.manifest` put each record's list back in place: `plan.steps["patched"].interventions["logits"].positions` is `{"pairs": [[10], [8], [8], [8]]}`.

`positions` is how you check that a `pos` landed on the token you meant before trusting a number; [positions.md](positions.md#check-where-it-landed) has the recipes. Generated rows that stopped early show up here as `alignment_missing`.

Each forward's and generation's record holds its `schedule`. At a point another point's forward served, it is that point's coordinates (`serving`). At a point that ran its forward, it is the sites the forward visited in the engine's order, with the reads and writes done at each, and the site it stopped after (`stop`). [FORMAT.md](../../FORMAT.md#derived-fields) lists every field.
