---
title: Bundles
one_liner: "Saving a trained or harvested featurizer and loading it in another document: the files a run writes, file_path, the header checks at compile and before any forward, the two-document workflow, and reading back a run that loads one."
tags: [usage, bundles, featurizers, safetensors, file_path]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/usage/output.md, docs/usage/running.md, docs/patterns/trained-subspaces.md, docs/patterns/sae.md, docs/patterns/lora.md]
sources: [clab/featurizers/bundles.py, clab/compile/uses.py, clab/compile/__init__.py, clab/compile/preflight.py, clab/plan.py]
---

# Bundles

A bundle is a safetensors file holding one featurizer's tensors and a header that says what made them: the kind and its fields, the model, the site and the width. A run writes one for every saved `<fit>.<featurizer>` and every saved `pca` op; a featurizer loads one with `file_path`, and the header is checked against the loading document before anything runs. The header's keys are in [FORMAT.md](../../FORMAT.md#featurizer-kinds).

## The two-document workflow

The training document saves the trained featurizer as a member of its fit. `documents/das.json`:

```json
"saves": ["held_iia", "fit.train.iia", "fit.train.ce", "fit.rot"]
```

Run it with `--out` inside the corpus, so the bundle lands under the corpus root. One file per save, named after it, so `fit.rot` becomes `fit.rot.safetensors`:

```text
clab run documents/das.json --out documents/artifacts
  ...
  fit.rot.safetensors          fit              {"basis":[16,8]}
clab run documents/das_apply.json
```

The loading document declares the featurizer with `file_path` and uses it by its declared name, since nothing trains it there. `documents/das_apply.json`:

```json
"rot": {"kind": "subspace", "k": 8, "parametrization": "cayley", "file_path": "documents/artifacts/fit.rot.safetensors"}
```

`das_apply.json`'s `iia` is bit-identical to `das.json`'s `held_iia`, which `das.json` scores after its fit through `fit.rot`: the same test rows through the same rotation. The corpus has three more pairs written the same way: `documents/pca_harvest.json` and `documents/pca_control.json` (`basis`), `documents/sae_train.json` and `documents/sae_train_ablation.json` (`fit.dictionary`), `documents/lora_train.json` and `documents/lora_apply.json` (`fit.lora`).

`--out` writes the training run's whole output directory there, `manifest.json` included, and a second training run into the same directory replaces the manifest but leaves the other bundles.

## `file_path`

`file_path` is relative to the corpus root: the nearest directory above the document that holds a `data/` folder ([running.md](running.md#the-corpus-root)). An absolute path is read as written. It cannot be a sweep.

A field the bundle's header records may be left out of the loading document, and the header's value is used: `das_apply.json` could drop `k` and `parametrization`, and `lora_apply.json` gives neither `rank` nor `alpha`. A field the document does give must match. The header's `document` key, the name of the document that wrote it, is not checked.

To see a header, from the corpus root:

```python
from safetensors import safe_open
with safe_open("documents/artifacts/fit.rot.safetensors", "pt") as f:
    print(dict(sorted(f.metadata().items())))
```

```text
{'component': 'layer_output', 'document': 'das', 'format': 'clab/1', 'k': '8', 'kind': 'subspace', 'layer': '0', 'model.dtype': 'float32', 'model.key': 'hf-internal-testing/tiny-random-LlamaForCausalLM', 'model.revision': '9fb191250dd56d0ba7ec9785a025ed29c03d5998', 'parametrization': 'cayley', 'width': '16'}
```

## What is checked, and when

Compile reads the bundle into the plan and checks it with the document alone; what needs the model is checked in preflight, after the model loads and before any forward.

| check | stage | refusal |
|---|---|---|
| the file exists and is safetensors | compile | `documents/artifacts/nope.safetensors does not exist under the corpus root; run the document that writes it first` |
| it holds exactly the kind's tensors | compile | `documents/artifacts/sae.safetensors holds W_dec, W_enc, b_dec, b_enc; a subspace bundle holds basis` |
| the kind's fields the document gives | compile | `header 'k' is '8'; this document expects '4'` |
| the model's `key`, `revision` and `dtype` | compile | `header 'model.dtype' is 'float32'; this document expects 'bfloat16'` |
| the component, and a layer written as a non-negative int, at every use | compile | `header 'layer' is '0'; this document uses 'rot' at layer_output layer 1 (counterfactual.v_cf)` |
| a layer counted from the last, or a site that picks several | preflight | `header 'layer' is '0'; this document uses 'rot' at layer_output layer -1, which is layer 1` |
| the indices the site's `select` keeps, written from the start, at every use: the ones `select` records, none where the header has no `select` | compile | `header 'select' is '{"num_heads": [1]}'; this document uses 'c' at attention_premix layer 0, whose selection keeps num_heads [0] (a.w)` |
| a `select` counted from the end, or a slice open at its end | preflight | `header has no 'select': it was made at the whole component, or before headers recorded a selection; this document uses 'c' at attention_premix layer 0, whose selection keeps num_heads [3]` |
| the width of every site it is used at | preflight | `header 'width' is '16'; this document uses 'rot' at layer_output, whose width is 8` |
| a featurizer over axes: its tensors laid out as the header records, a gate's `theta` with as many units on each axis as `layer` and `kept.<axis>` name, an affine's `W` and `b` with a map per layer `layer` names at the header's `width` | compile | `'theta' is [20], and its header lays a gate over ['layers', 'num_heads'] out on layer [0, 1, 2, 3, 4], kept.num_heads [0, 1, 2, 3]: 'theta' is [5, 4], …` |
| a gate over axes: the indices its site selects on each axis but `layers` | preflight | `header 'kept.num_heads' is '[0, 1, 2, 3]'; this document uses 'heads' at attention_premix, whose selection keeps num_heads [1, 2]` |
| a cross-site adapter's `with` site | preflight | [lora.md](../patterns/lora.md) |

A file with no clab header is a published one, and only an `sae` reads it ([featurizers.md](featurizers.md#loading-a-published-sparse-autoencoder)); any other kind refuses it (`has no clab header; a subspace bundle is one clab wrote`). A loaded featurizer is fixed: a fit refuses to train it.

## `pca` bundles

The `pca` op's value is a bundle: the top-`k` directions (`basis`, `[width, k]`) and the `mean` they are centred on, with a header like a fit's. `documents/pca_harvest.json` reads layer 0's output at every token after BOS and saves the op step `basis`:

```json
"basis": {"kind": "op", "op": "pca", "of": "harvest.acts", "k": 4}
```

`documents/pca_control.json` loads it as `{"kind": "pca", "k": 4, "file_path": "documents/artifacts/basis.safetensors"}`. A `pca` needs more vectors than its `k`, so a read of one position per row on four rows is refused before any forward (`pca of 4 vectors has no rank-4 basis; it needs more than 4`); read a range of positions to give it more.

## A hand-made `linear` bundle

A probe trained outside clab loads as a `linear` bundle only with clab's header: a file without one is refused (`... has no clab header; a linear bundle is one clab wrote`). The file holds `W`, `[width, classes]`, and `b`, `[classes]`, and the probe's scores are `h @ W + b`, each class at its index in the class list: the distinct values of the document's `classes` column, sorted. Every header value is a string:

```python
save_file({"W": W, "b": b}, "documents/artifacts/probe.safetensors", metadata={
    "format": "clab/1", "kind": "linear", "classes": json.dumps(["weekday", "weekend"]),
    "model.key": key, "model.revision": revision, "model.dtype": "float32",
    "component": "layer_output", "layer": "1", "width": "16", "document": "hand_made"})
```

- `classes` is the class list as `json.dumps` writes it, its `", "` separator included: `'["weekday","weekend"]'` is refused against `'["weekday", "weekend"]'`. It is checked, with `kind` and the three `model.` keys (the document's `model` block as written), when the document compiles.
- `component` and `layer` (the one layer as an int, or a JSON list of several) are checked at every use when the document compiles, and `width`, the site's last axis, before any forward; a header without one of them is refused.
- `document` says what made the bundle; nothing checks it.

## Swept training runs

A fit or `pca` step under a sweep writes one file holding a bundle per point, each key prefixed by its coordinate (`mask.mask=0.01/theta`). A featurizer loads one bundle, so loading that file is refused:

```text
clab: featurizers.rot.file_path: documents/artifacts/fit.mask.safetensors holds one bundle per point of a sweep (mask.mask=0.0001, mask.mask=0.002, mask.mask=0.01, mask.mask=0.05); a featurizer loads one bundle, so run the document that writes it unswept
```

Run the training document again with the sweep replaced by the value you want to keep.

## Reproducing a fit's numbers in the apply document

A fit's body, its evals included, runs in batches of the fit's `batch_size`; the apply document, and a training document's own steps after its fit, run in batches of `--batch-size` (default 8). A row's numbers are guaranteed to the bit only at one batch shape ([running.md](running.md)). On the CPU, the apply document run at the batch size the held-out numbers were made at reproduces them to the bit: a probe fitted with `batch_size` 10 gives eval scores that its apply document reproduces to the bit at `--batch-size 10` and to about 2e-7 at the default, and `das_apply.json`'s `iia` equals `das.json`'s `held_iia`, scored by steps after the fit, at `--batch-size` 1, 2 and 8. Compare the two at the batch size each was made at.

On a GPU the loaded featurizer can give numbers that differ from the trained one's in the last bits even at the fit's batch size: on Llama-3.2-1B, a rank-1 subspace at batch size 50 gave logit differences up to 2.1e-5 apart on 63 of 200 rows, with identical `match`. Compare at a tolerance there.

## Reading back a run that loads a bundle

`clab show`, `clab table` and `Plan.load` rebuild the plan from the run's manifest, which names the bundle by its `file_path`, and read it again from the working directory. Run them from the corpus root, with the bundle still there; from anywhere else they stop with the compile refusal (in Python, a `PlanError`):

```text
clab: documents/artifacts/fit.rot.safetensors does not exist under the corpus root; run the document that writes it first
```

The directory it looked in is the working directory. A `file_path` written as an absolute path is read from any directory.
