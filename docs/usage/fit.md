---
title: Fit
one_liner: "Training a featurizer inside a run: train, objective, optimizer, epochs and minibatches, early stopping, eval substitutions, the history and members a fit produces, and epochs 0 as a null control."
tags: [usage, fit, training, objective, eval, history, optimizer]
related: [docs/usage/featurizers.md, docs/usage/bundles.md, docs/usage/sweeps.md, docs/usage/ops.md, docs/concepts/names-and-sweeps.md, docs/patterns/trained-subspaces.md, docs/patterns/probing.md, docs/patterns/sae.md, docs/patterns/lora.md]
sources: [clab/steps/fit.py, clab/compile/__init__.py, clab/plan.py, clab/featurizers/__init__.py]
---

# Fit

A `fit` step trains the featurizers its `train` names, inside the run, through the frozen model: it runs its own `steps` (the body) over minibatches of the training rows and steps an optimizer on a weighted `objective`. Gradients are taken for the featurizers' tensors alone. The fields are in [FORMAT.md](../../FORMAT.md#fit), the optimizer and early stopping in [`fit` blocks](../../FORMAT.md#fit-blocks).

## The canonical fit

The fit of `documents/das.json`, whose body pulls in the root bundles `cf_read` and `das`:

```json
"fit": {
  "kind": "fit",
  "train": ["rot"],
  "objective": {"ce": 1.0},
  "epochs": 10,
  "batch_size": 16,
  "seed": 0,
  "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
  "early_stop": {"metric": "iia", "mode": "max", "patience": 3},
  "eval": {"held_out": {"train": "test"}, "train": {}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "train.counterfactual_inputs[0]", "interventions": "cf_read"},
    "patched": {"kind": "forward", "input": "train.input", "interventions": "das"},
    "iia": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "train.cf_answer", "b": "train.base_answer"},
    "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"}
  }
}
```

The run draws `rot` from the fit's `seed` and makes the optimizer. Each epoch it shuffles the training rows (by `seed` and the epoch), runs the body on each minibatch of `batch_size` rows with gradients on, and takes one optimizer step per minibatch on `1.0 × ce`. Then it runs the body once more over every training row without gradients: that scored pass is the epoch's [history](#history) entry and what `early_stop` judges. After the last epoch it restores the best epoch's tensors and runs the body once per [`eval`](#evals).

## The body

The body is a scope of its own. It sees every value made before the fit, and inside it a trained featurizer answers to both its declared name and `<fit>.<name>` (`rot` and `fit.rot`). A value from before the fit with a row per example is refused, since the body runs over minibatches of its rows; read it inside the body instead. A value with no rows (a `mean`, a direction) is fine.

```text
clab: steps.fit.steps: the body uses 'pre.v', which has a row per example of the whole dataset; the body runs over minibatches of its rows, so such a value is one it makes
```

The body's datasets are paired row by row, so they hold as many rows as each other. A body holds no fit, and its step names are not dataset or featurizer names.

## The objective

`objective` is `{term: weight}`, and the optimizer minimizes the weighted sum; a negative weight maximizes a term. A term is one of:

- **a body op**, averaged over every axis, missing values left out: `ce` above;
- **a trained featurizer's scalar**, `<featurizer>.<scalar>`: a gate's `mask` (the mean of its soft mask, the sparsity pressure in `documents/dbm.json`), an sae's `reconstruction` and `l1` (`documents/sae_train.json`). Subspaces, probes and adapters have none.

An sae's scalars are over the coordinates the body read of it with an `encode` no `decode` closes. A body that only brackets the sae has nothing to compute them on, and the run stops with `an sae's scalars are over what the body's encodes of it saw; this body has none`. Anything else is refused at compile:

```text
clab: steps.fit.objective.patched.logits: not a body op or a trained featurizer's scalar; body ops: iia, ce
```

A metric term is a float32 loss whatever the model's dtype, as every metric is ([ops.md](ops.md)).

## Epochs, minibatches and the learning rate

- **Count the optimizer steps**: `epochs × ⌈training rows / batch_size⌉`. `das.json` has two training rows and `batch_size` 16, so one step per epoch, ten in all.
- **Adam and AdamW move each parameter by about `lr` per step**, whatever the gradient's size, so `lr × steps` roughly bounds how far a featurizer gets. `das.json`'s ten steps turn its rotation's subspace by about 16 degrees from the draw at `lr` 0.001 and 66 degrees at 0.01. When a history barely moves, suspect `lr × steps` first.
- **An epoch is two passes** over the training rows: one in minibatches with gradients, one scored.
- **Memory follows the fit's `batch_size`.** A minibatch's forwards run with gradients on, so their activations are kept for the backward. The body's forwards always run in batches of the fit's own `batch_size`: `das.json` run at `--batch-size` 1 and 8 gives bit-identical bundles and evals. Lower the fit's `batch_size` to fit in memory, and expect more optimizer steps per epoch.
- **The model's weights never change.** A forward before the fit and one after it give bit-identical logits.

## Early stopping

`early_stop` names a body op (`metric`), `mode` `max` or `min`, and `patience`. The fit ends after `patience` epochs without improvement on the scored pass, and the best epoch's tensors are restored, not the last's: `das.json` with `{"metric": "ce", "mode": "max", "patience": 1}` stops after two epochs, and its `fit.rot` is bit-identical to a one-epoch fit's. Without `early_stop` the last epoch is kept.

## Evals

Each name under `eval` runs the body once more after training, scored, with each dataset on the left read as the one on the right; `{}` is the data it trained on. Every reference to the replaced dataset follows, columns and scopes included: under `held_out`, `train.cf_answer` reads `test.cf_answer`. A replacement may hold a different number of rows, and its values are aligned to its own rows, so `fit.held_out.iia` and `fit.train.iia` cannot share one `clab table`. An eval is not named `history` or after a featurizer the fit trains, and has no dot.

## Members

Nothing leaves the body except through these names:

| member | what it is |
|---|---|
| `<fit>.<featurizer>` | the trained featurizer; saved, it is a [bundle](bundles.md) |
| `<fit>.<eval>.<op>` | a body op's value from the run named `<eval>` (`fit.held_out.iia`) |
| `<fit>.<eval>.<step>.<entry>` | a body entry's value from that run (`fit.held_out.patched.logits`) |
| `<fit>.history.<op>` | each body op's mean after each epoch, `[epochs run]` |
| `<fit>.history.<featurizer>.<scalar>` | each trained featurizer's scalar after each epoch, `[epochs run]` |

**After the fit, the bare name is the declared featurizer as drawn at the run's start, never trained.** In `das.json`, a step after the fit that encodes with `rot` scores the untrained seed-0 subspace; with `fit.rot` it reproduces `fit.held_out.iia` to the bit. Name `<fit>.<featurizer>` everywhere after the fit.

### History

A history holds one entry per epoch run: fewer when `early_stop` ends the fit, none (`[0]`) at `epochs: 0`. Every body op gets one, in the objective or not, so an aggregate you want to watch belongs in the body: a `mean` of a `match` there is both `<fit>.<eval>.acc` and `<fit>.history.acc`. `clab table <out> fit.history.ce` prints a row per epoch, with an `epochs` column. Under a sweep, every member is a dict keyed by coordinate ([sweeps.md](sweeps.md)).

## `epochs: 0` is a null control

`epochs: 0` trains nothing: the featurizers are scored as drawn, every eval runs, and every history is empty. What the draw is makes it the control:

- **an adapter** starts as the identity exactly: `documents/lora_train.json` at `epochs: 0` gives `fit.held_out.ce` bit-identical to a clean forward's;
- **a subspace** is a uniformly random one of the fit's seed: `das.json` at `epochs: 0` gives the numbers of a declared `{"kind": "subspace", "k": 8, "seed": 0}` used without a fit, to the bit;
- **a probe** scores every class 0, and `match` breaks the tie to the lower class id, so its accuracy is the share of rows whose label sorts first.

To run the control beside the trained featurizer, sweep `epochs`; with a sweep on `seed` too, `"epochs": {"sweep": [0, 10]}, "seed": {"sweep": [0, 1, 2]}` gives six fits that differ in nothing else, keyed `epochs=0,seed=1` and so on.

## Sweeping a fit

Any of the fit's values may be a sweep, names such as `optimizer.name` excepted: an objective weight (`dbm.json` sweeps its sparsity weight, whose default label is the term's key, `mask.mask`), `lr`, `epochs`, `seed`, or the `layers` of a site the body uses, which trains one featurizer per layer. Each point is a whole fit with its own members. A swept fit's bundle file holds a bundle per point, which no featurizer can load ([bundles.md](bundles.md)).
