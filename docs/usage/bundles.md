---
title: Bundles
one_liner: "How do I save a trained featurizer and load it in another document? The files, file_path, the header checks and the two-document workflow."
tags: [usage, bundles, featurizers, safetensors, file_path]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/usage/output.md, docs/usage/running.md, docs/patterns/trained-subspaces.md, docs/patterns/sae.md, docs/patterns/lora.md, docs/patterns/probing.md]
sources: [clab/featurizers/bundles.py, clab/featurizers/__init__.py, clab/compile/uses.py, clab/compile/__init__.py, clab/compile/preflight.py, clab/plan.py]
---

# Bundles

A bundle is a safetensors file holding one featurizer's tensors and a header that says what made them: the kind and its fields, the model, the site and the width. A run writes one for every saved `<fit>.<featurizer>` and every saved `pca` op. A featurizer loads one with `file_path`, and the header is checked against the loading document before anything runs. The header's keys are in [FORMAT.md](../../FORMAT.md#featurizer-kinds).

## The two-document workflow

The training document saves the trained featurizer as a member of its fit. `documents/das.json`:

```json
"saves": ["held_iia", "fit.train.iia", "fit.train.ce", "fit.rot"]
```

Run it with `--out` inside the corpus, so the bundle lands under the corpus root. One file per save, named after it, so `fit.rot` becomes `fit.rot.safetensors`:

```text
clab run documents/das.json --out documents/artifacts
das on nnterp (batch size 8, device cpu, tp 1) -> documents/artifacts
  held_iia.safetensors       held_iia       [2,1]
  fit.train.iia.safetensors  fit.train.iia  [2,1]
  fit.train.ce.safetensors   fit.train.ce   [2,1]
  fit.rot.safetensors        fit            {"basis":[16,8]}
  documents/artifacts/manifest.json
clab run documents/das_apply.json
```

The loading document declares the featurizer with `file_path` and uses it by its declared name, since nothing trains it there. `documents/das_apply.json`:

```json
"rot": {"kind": "subspace", "k": 8, "parametrization": "cayley", "file_path": "documents/artifacts/fit.rot.safetensors"}
```

`das_apply.json`'s `iia` equals `das.json`'s `held_iia`, which `das.json` scores after its fit through `fit.rot`: the same test rows through the same rotation (`tests/test_fit.py` `test_das_apply_scores_the_saved_rotation_as_das_scored_it_held_out`). The corpus has three more pairs written the same way: `documents/pca_harvest.json` and `documents/pca_control.json` (`basis`), `documents/sae_train.json` and `documents/sae_train_ablation.json` (`fit.dictionary`), `documents/lora_train.json` and `documents/lora_apply.json` (`fit.lora`).

`--out` writes the training run's whole output directory there, `manifest.json` included. A second training run into the same directory replaces the manifest but leaves the other bundles:

- **The first run can no longer be read back.** After `documents/pca_harvest.json` runs into `das.json`'s `--out`, `clab table <out> held_iia` is refused with `'held_iia' was not saved; add it to saves`, though `held_iia.safetensors` is still there.
- **Give each training run its own `--out`.** Copy each bundle file to where the loading document's `file_path` points, or run each training document into its own directory once and point `file_path` there.

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

Compile reads the bundle into the plan and checks it with the document alone. What needs the model is checked in preflight, after the model loads and before any forward.

| check | stage | refusal |
|---|---|---|
| the file exists and is safetensors | compile | `documents/artifacts/nope.safetensors does not exist under the corpus root; run the document that writes it first` |
| it holds exactly the kind's tensors | compile | `documents/artifacts/sae.safetensors holds W_dec, W_enc, b_dec, b_enc; a subspace bundle holds basis` |
| the kind's fields the document gives | compile | `header 'k' is '8'; this document expects '4'` |
| the model's `key`, `revision` and `dtype` | compile | `header 'model.dtype' is 'float32'; this document expects 'bfloat16'` |
| the component, and a layer written as a non-negative int, at every use | compile | `header 'layer' is '0'; this document uses 'rot' at layer_output layer 1 (counterfactual.v_cf)` |
| a layer counted from the last, or a site that picks several | preflight | `header 'layer' is '0'; this document uses 'rot' at layer_output layer -1, which is layer 1` |
| the indices `select` records, at a site whose `select` is written from the start, or at the whole component | compile | `header 'select' is '{"num_heads": [1]}'; this document uses 'lora' at attention_premix layer 0, whose selection keeps num_heads [0] (adapted.adapt)` |
| the same, at a `select` counted from the end or a slice open at its end | preflight | `header 'select' is '{"num_heads": [1]}'; this document uses 'lora' at attention_premix layer 0, whose selection keeps num_heads [3]` |
| a header with no `select`, used at a site that selects | compile, or preflight from the end | `header has no 'select': it was made at the whole component` |
| the width of every site it is used at | preflight | `header 'width' is '16'; this document uses 'rot' at layer_output, whose width is 8` |
| a gate over axes: `theta` with as many units on each axis as `layer` and `kept.<axis>` name | compile | `'theta' is [20], and its header lays a gate over ['layers', 'num_heads'] out on layer [0, 1, 2, 3, 4], kept.num_heads [0, 1, 2, 3]: 'theta' is [5, 4], …` |
| a gate over axes: the indices its site selects on each axis but `layers` | preflight | `header 'kept.num_heads' is '[0, 1, 2, 3]'; this document uses 'heads' at attention_premix, whose selection keeps num_heads [1, 2]` |
| an affine over layers: a map per layer `layer` names, at the header's `width` | compile | `'W' is [3, 32, 32], and its header lays an affine over ['layers'] out on layer [0, 1, 2, 3], width 32: …` |
| a cross-site adapter's `with` site | preflight | [lora.md](../patterns/lora.md#across-sites-with) |

An index the header records matches however the document spells it: an adapter trained at `{"num_heads": [1]}` loads at `1`, `-3` and `"1:2"` on a 4-head layer. A file with no clab header is a published one, and only an `sae` reads it ([featurizers.md](featurizers.md#loading-a-published-sparse-autoencoder)); any other kind refuses it (`has no clab header; a subspace bundle is one clab wrote`).

## `pca` bundles

The `pca` op's value is a bundle: the top-`k` directions (`basis`, `[width, k]`) and the `mean` they are centred on, with a header like a fit's. `documents/pca_harvest.json` reads layer 0's output at every token after BOS and saves the op step `basis`:

```json
"basis": {"kind": "op", "op": "pca", "of": "harvest.acts", "k": 4}
```

`documents/pca_control.json` loads it as `{"kind": "pca", "k": 4, "file_path": "documents/artifacts/basis.safetensors"}`. A `pca` needs more vectors than its `k`, so a read of one position per row on four rows is refused before any forward (`pca of 4 vectors has no rank-4 basis; it needs more than 4`). Read a range of positions to give it more.

## A hand-made `linear` bundle

A probe trained outside clab loads as a `linear` bundle only with clab's header: a file without one is refused (`... has no clab header; a linear bundle is one clab wrote`). The file holds `W`, `[width, classes]`, and `b`, `[classes]`, and the probe's scores are `h @ W + b`. Each class is at its index in the class list: the distinct values of the document's `classes` column, sorted. Every header value is a string:

```python
save_file({"W": W, "b": b}, "documents/artifacts/probe.safetensors", metadata={
    "format": "clab/1", "kind": "linear", "classes": json.dumps(["weekday", "weekend"]),
    "model.key": key, "model.revision": revision, "model.dtype": "float32",
    "component": "layer_output", "layer": "1", "width": "16", "document": "hand_made"})
```

- `classes` is the class list as `json.dumps` writes it, its `", "` separator included: `'["weekday","weekend"]'` is refused against `'["weekday", "weekend"]'`. It is checked, with `kind` and the three `model.` keys (the document's `model` block as written), when the document compiles.
- `component` and `layer` (the one layer as an int, or a JSON list of several) are checked at every use when the document compiles, and `width`, the site's last axis, before any forward. A header without one of the three is refused (`header 'width' is None; this document uses 'probe' at layer_output, whose width is 16`).
- The tensors' shapes are not checked against `width`. A `W` of another first axis fails at the first forward with torch's own error (`mm: mat1 and mat2 shapes cannot be multiplied (3x16 and 8x2)`).

## Swept training runs

A fit or `pca` step under a sweep writes one file holding a bundle per point, each key prefixed by its coordinate (`mask.mask=0.01/theta`). A featurizer loads one bundle, so loading that file is refused:

```text
clab: featurizers.rot.file_path: documents/artifacts/fit.mask.safetensors holds one bundle per point of a sweep (mask.mask=0.0001, mask.mask=0.002, mask.mask=0.01, mask.mask=0.05); a featurizer loads one bundle, so run the document that writes it unswept
```

Run the training document again with the sweep replaced by the value you want to keep.

## Reproducing a fit's numbers in the apply document

A fit's evals run in batches of the fit's `batch_size`, and the apply document in batches of `--batch-size` ([fit.md](fit.md#epochs-minibatches-and-the-learning-rate)). Run the apply document at the batch size the fit's numbers were made at, and compare where every setting agrees ([running.md](running.md#when-two-runs-give-the-same-numbers)).

## Reading back a run that loads a bundle

`clab show`, `clab table` and `Plan.load` rebuild the plan from the run's manifest, which names the bundle by its `file_path`, and read it again from the working directory. Run them from the corpus root, with the bundle still there. From anywhere else they stop with the compile refusal (in Python, a `PlanError`):

```text
clab: documents/artifacts/fit.rot.safetensors does not exist under the corpus root; run the document that writes it first
```

A `file_path` written as an absolute path is read from any directory.
