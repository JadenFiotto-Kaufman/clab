---
title: Cached Harvests
one_liner: "How do I harvest once and reuse it? Save it, load it in another document, fit on it without a forward, harvest in shards."
tags: [patterns, load, cache, harvest, shards, skip]
related: [docs/usage/output.md, docs/usage/data.md, docs/usage/fit.md, docs/usage/bundles.md, docs/usage/sweeps.md, docs/usage/engines.md, docs/patterns/probing.md, docs/patterns/sae.md]
sources: [clab/plan.py, clab/compile/__init__.py, clab/compile/preflight.py, clab/steps/__init__.py, clab/output/__init__.py, clab/engines/nnterp.py]
---

# Cached Harvests

A harvest is a read over every row of a dataset, saved; a later document that only fits or reduces it needs the values, not the forwards. A `load` step brings a value an earlier run saved into another document, with the record that run's manifest holds for it, so every step after it treats it as the value it was. Its fields are in [FORMAT.md](../../FORMAT.md#load).

## Split a document at its harvest

`documents/pca_harvest.json` reads layer 0's output at every token after BOS and takes a `pca` of it in one document. Saved, the read becomes a cache:

```json
{
  "model": {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
            "dtype": "float32"},
  "data": {"prompts": {"path": "weekdays/train"}},
  "steps": {
    "harvest": {"kind": "forward", "input": "prompts.input",
                "interventions": {"acts": {"site": {"component": "layer_output", "layers": 0},
                                           "pos": {"index": "1:"}}}}
  },
  "saves": ["harvest.acts"]
}
```

Run it with `--out out/prompts_l0` inside the corpus. A later document that uses the read keeps the same `model` and `data` blocks, and loads the read in place of the forward:

```json
"steps": {
  "harvest": {"kind": "load", "from": "out/prompts_l0", "value": "harvest.acts"},
  "basis": {"kind": "op", "op": "pca", "of": "harvest", "k": 4}
}
```

- **The step's value is named by the step,** `harvest`, as an op step's is.
- **`basis` equals pca_harvest.json's basis** ([running.md](../usage/running.md#when-two-runs-give-the-same-numbers)). Its bundle's header names the site, layer and width of the read the load carries, so `documents/pca_control.json` loads it as it loads the original.
- **A probe or a dictionary fits the same way.** A fit's body takes the loaded value's rows by id, as it takes a harvest made in its own document ([fit.md](../usage/fit.md)). `sae_train.json` with its harvest loaded trains an equal dictionary.
- **A document of loads and op steps loads no weights.**
- **An `encode` of a loaded read uses the featurizer at the read's site.** A run that loads a value and saves it passes on the read it carries. A load of that save encodes as the first load does.
- **An op's value over a read, loaded or not, is no read's own,** and an `encode` of it is refused.

## What the load checks

Compile reads the run's `manifest.json` and the value's file, and refuses by name, before any model loads:

| refused | message, after `steps.<step>.from` or `.value` |
|---|---|
| no run there | `no run at out/x under the corpus root (no manifest.json); …` |
| another model block, a loader key or `dtype` included | `out/x ran another model than this document's: attn_implementation None where this document's is 'eager'` |
| a name the run did not save | `out/x saved no 'harvest.nope'; it saved harvest.acts, basis` |
| a manifest whose value records hold no `axes` | `out/x/manifest.json predates the sweep axes a saved value's record holds ('axes'), …` |
| a manifest whose fit has an `eval` and no `evals` | `out/x/manifest.json predates a fit's evals as steps ('evals'), …` |
| the run's dataset under another name, or not declared | `'harvest.acts' was made over the rows of dataset 'prompts', which this document does not declare; …` |
| other rows, or the same rows in another order | `'harvest.acts' was made over the rows of dataset 'prompts', and this document's 'prompts' is not the rows out/x read, in their order: as many rows, whose digest differs` |
| fewer or more rows | `… is not the rows out/x read, in their order: 3 rows where it read 4` |
| a chat dataset rendered with another `template` | `'read.acts' was made over the rows of dataset 'chats', whose chats out/x rendered with template {'add_generation_prompt': True}, and this document's 'chats' has None` |
| a sweep it depends on, over other values here | `'harvest.acts' is swept over 'layers', [0, 1] in out/x, and this document sweeps 'layers' over [1, 0]; …` |

**What must match:**

- The `model` block is the run's, field for field. `dtype` defaults to `float32`, so a load of a bfloat16 harvest writes `"dtype": "bfloat16"` too.
- The loading document declares the dataset under the run's name.
- Its rows after `split`, `shuffle`, `skip`, `take` and `columns` are the run's: the same `sha256` and `n_rows`.
- A chat dataset renders with the run's `template`, which the digest does not cover.

**A swept value:**

- It loads at every point of the axes it depends on, each point the saved one there.
- The load declares each of the run's sweeps it depends on that the loading document does not, with the run's label and values.
- So a harvest swept over `layers` loads with no other field swept, keyed `layers=0`, `layers=1` as the run's was.
- A sweep the loading document declares under that label must have the same values.

**What does not load:**

- A bundle (a `pca`'s value, a trained featurizer). It is refused with the `file_path` that loads it as a featurizer ([bundles.md](../usage/bundles.md)): `'basis' is a bundle, which a featurizer of its kind loads, with 'file_path': 'out/prompts_l0/basis.safetensors'`.
- A value in a featurizer's coordinates.
- A generation's value, or one holding its positions.
- A load inside a fit's body.

A value an op took out of the coordinates, by changing the last axis they lie on, is in none and loads. A per-token L0, `[{"op": "encode", "featurizer": "sae"}, {"op": "gt", "with": 0}, {"op": "sum", "over": ["features"]}]`, loads laid out `[rows, pos]`.

## Positions and the table

A loaded read keeps its positions per row, and preflight tokenizes its input as a forward would. So `clab table` labels `pos` and `pos_token` as the harvesting run's table does, and `clab show` lists the read under `<step>.<entry>` (`harvest.acts`). The tokens are this run's, from the same checkpoint and the same rows.

## A harvest in shards

A harvest too large for one run is made in shards: one shuffled order, cut with `skip` and `take` ([data.md](../usage/data.md#narrowing-and-renaming-rows)), one run per shard. On the 42 rows of `data/weekdays_full/data.json`:

| shard | dataset block |
|---|---|
| 0 | `{"path": "weekdays_full/data", "shuffle": true, "skip": 0, "take": 20}` |
| 1 | `{"path": "weekdays_full/data", "shuffle": true, "skip": 20, "take": 22}` |

The two shards' values, joined in order, equal the harvest of all 42 shuffled rows. A load of a shard needs a dataset block, under the shard's dataset name, that gives the same rows: `{"path": "weekdays_full/data", "shuffle": true, "skip": 20}` loads shard 1. Nothing in a document joins several shards into one value.

## Under `ndif`

The loaded value travels in the request, inside the plan, as a loaded bundle's tensors do, so the server reads no file of yours. `-o remote=local` runs what a server runs, and its result equals the nnterp engine's. A small cached operand therefore serves a model you cannot load yourself, as a counterfactual read saved from one NDIF run and patched into another.

A request carries at most 512 MiB of loaded tensors over a document's loads (`clab/steps/__init__.py` `LOADED`). A document that loads more is refused before anything is sent, with each load's size. Run it where the file is, on the nnterp or hooks engine:

```text
clab: harvest, held: the values this document loads hold 805.7 MiB of tensors (harvest 402.8 MiB, held 402.8 MiB), and a request carries at most 512 MiB inside its plan; …
```

## Pitfalls

- **The cache is the run that made it.** A row's values depend on that run's batch size and padding ([running.md](../usage/running.md#batch-size-and-padding)). The loading document reproduces numbers computed from the cache, not a fresh harvest at another batch shape.
- **Change the data, rerun the harvest.** An edited row, a new `take` or another `shuffle` moves the digest, and every load of the old cache is refused until the harvest runs again.
- **Save what you will load.** Only names in the harvesting run's `saves` have files; a held-out harvest the downstream document scores needs saving too.
