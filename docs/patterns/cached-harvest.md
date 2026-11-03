---
title: Cached Harvests
one_liner: "Harvest once, use it many times: save a harvest, bring it into another document with a load step, fit a probe or take a pca on it without a forward, make a large harvest in shards with skip and take, and what must match for a load to compile."
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
            "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"},
  "data": {"prompts": {"path": "weekdays/train"}},
  "steps": {
    "harvest": {"kind": "forward", "input": "prompts.input",
                "interventions": {"acts": {"site": {"component": "layer_output", "layers": 0},
                                           "pos": {"index": "1:"}}}}
  },
  "saves": ["harvest.acts"]
}
```

Run it with `--out out/prompts_l0` inside the corpus. Every later document that uses the read starts from it; the same `model` and `data` blocks, and in place of the forward:

```json
"steps": {
  "harvest": {"kind": "load", "from": "out/prompts_l0", "value": "harvest.acts"},
  "basis": {"kind": "op", "op": "pca", "of": "harvest", "k": 4}
}
```

The step's value is named by the step, `harvest`, as an op step's is. `basis` is pca_harvest.json's basis to the bit, and its bundle's header names the site, layer and width of the read the load carries, so `documents/pca_control.json` loads it as it loads the original. A probe or a dictionary fits the same way: a fit's body takes the loaded value's rows by id, as it takes a harvest made in its own document ([fit.md](../usage/fit.md)), and an `encode` of it uses the featurizer at the read's site. A run that loads a value and saves it passes on the read it carries, so a load of that save encodes as the first load does; an op's value over a read, loaded or not, is no read's own, and an `encode` of it is refused. `sae_train.json` with its harvest loaded trains the same dictionary to the bit. On Llama-3.2-1B in bfloat16, a 4-way probe over a 10,000-row cache of one residual vector per row, scored on a 7,600-row held-out cache, gives the held-out accuracy of the document that harvested both (0.8326) to the bit, in 4 s where that document took 64.

## What the load checks

Compile reads the run's `manifest.json` and the value's file, and refuses by name, before any model loads:

| refused | message, after `steps.<step>.from` or `.value` |
|---|---|
| no run there | `no run at out/x under the corpus root (no manifest.json); run the document that saves the value there first, with --out out/x` |
| another model block, a loader key included | `out/x ran another model than this document's: attn_implementation None where this document's is 'eager'` |
| a name the run did not save | `out/x saved no 'harvest.nope'; it saved harvest.acts, basis` |
| a manifest an older clab wrote, whose value records have no `axes` | `out/x/manifest.json predates the sweep axes a saved value's record holds ('axes'), so it does not say which sweeps its values depend on; rerun the document that wrote it` |
| a manifest an older clab wrote, whose fit has an `eval` and no `evals` | `out/x/manifest.json predates a fit's evals as steps ('evals'), so its fit 'fit' does not hold its evals' records; rerun the document that wrote it` |
| other rows, or the same rows in another order | `'harvest.acts' was made over the rows of dataset 'prompts', and this document's 'prompts' is not the rows out/x read, in their order: as many rows, whose digest differs` |
| a chat dataset rendered with another `template` | `'read.acts' was made over the rows of dataset 'chats', whose chats out/x rendered with template {'add_generation_prompt': True}, and this document's 'chats' has None` |
| a sweep it depends on, over other values here | `'harvest.acts' is swept over 'layers', [0, 1] in out/x, and this document sweeps 'layers' over [1, 0]; …` |

So the loading document declares the dataset under the same name, and its rows after `split`, `shuffle`, `skip`, `take` and `columns` must be the run's: the same `sha256` and `n_rows`, and for chats the same `template`, which the digest does not cover. A swept value loads at every point of the axes it depends on, each point the saved one there: the load declares each of the run's sweeps it depends on that the loading document does not, with the run's label and values, so a harvest swept over `layers` loads with no other field swept and comes out keyed `layers=0`, `layers=1` as the run's did; a sweep the document declares under that label must have the same values. A bundle (a `pca`'s value, a trained featurizer) is refused with the `file_path` that loads it as a featurizer ([bundles.md](../usage/bundles.md)); so are a value in a featurizer's coordinates, a generation's value or one holding its positions, and a load inside a fit's body. A value an op took out of the coordinates, by changing the last axis they lie on, is in none and loads: a per-token L0, `[{"op": "encode", "featurizer": "sae"}, {"op": "gt", "with": 0}, {"op": "sum", "over": ["features"]}]`, laid out `[rows, pos]`.

## Positions and the table

A loaded read keeps its positions per row, and preflight tokenizes its input as a forward would, so `clab table` labels `pos` and `pos_token` exactly as the harvesting run's table does, and `clab show` lists the read under `<step>.<entry>` (`harvest.acts`). The tokens are this run's, from the same checkpoint and the same rows.

## A harvest in shards

`skip` drops the first N rows, after `shuffle` and before `take`, so one shuffled order splits into shards, each a run of its own:

```json
"cache": {"hf": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
          "split": "train", "shuffle": true, "skip": 20000, "take": 10000}
```

is rows 20,000 to 29,999 of the order `shuffle` gives. A document loads one shard per `load`, with the same dataset block; loading several shards as one value is not a step clab has.

## Under `ndif`

The loaded value travels in the request, inside the plan, as a loaded bundle's tensors do, so the server reads no file of yours, and `-o remote=local` runs what a server runs: the result equals the nnterp engine's to the bit. A small cached operand therefore serves a model you cannot load yourself, as a counterfactual read saved from one NDIF run and patched into another. A request carries at most 512 MiB of loaded tensors over a document's loads; a document that loads more is refused before anything is sent, with each load's size:

```text
clab: harvest, held: the values this document loads hold 600.0 MiB of tensors (harvest 400.0 MiB, held 200.0 MiB), and a request carries at most 512 MiB inside its plan; run it where the file is, on the nnterp or hooks engine
```

## Pitfalls

- **The cache is the run that made it.** A row's values depend on that run's batch size and padding ([running.md](../usage/running.md#batch-size-and-reproducibility)); the loading document reproduces numbers computed from the cache, not a fresh harvest at another batch shape.
- **Change the data, rerun the harvest.** An edited row, a new `take` or another `shuffle` moves the digest, and every load of the old cache is refused until the harvest runs again.
- **Save what you will load.** Only names in the harvesting run's `saves` have files; a held-out harvest the downstream document scores needs saving too.
