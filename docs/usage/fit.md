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

The fit of `documents/das.json` and the step before it, which harvests the counterfactual's value once. Its two datasets are the splits of `data/weekdays/data.json` ([the corpus data](data.md#the-corpus-data)):

```json
"data": {"train": {"path": "weekdays/data", "split": "train"},
         "test": {"path": "weekdays/data", "split": "test"}}
```

```json
"counterfactual": {"kind": "forward", "input": "train.counterfactual_inputs[0]",
                   "interventions": {"h": {"site": "target", "pos": -1}}},
"fit": {
  "kind": "fit",
  "train": ["rot"],
  "objective": {"ce": 1.0},
  "epochs": 10,
  "batch_size": 16,
  "seed": 0,
  "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
  "early_stop": {"metric": "iia", "mode": "max", "patience": 3},
  "eval": {"train": {}},
  "steps": {
    "v_cf": {"kind": "op", "op": "encode", "featurizer": "fit.rot", "of": "counterfactual.h"},
    "patched": {"kind": "forward", "input": "train.input", "interventions": {
      "patch": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.rot"},
                                                     {"op": "swap", "with": "v_cf"},
                                                     {"op": "decode", "featurizer": "fit.rot"}]},
      "logits": {"site": "head", "pos": -1}}},
    "iia": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "train.cf_answer", "b": "train.base_answer"},
    "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"}
  }
}
```

The counterfactual's value does not depend on the training, so it is read once, before the fit, and the body's `encode` step takes its rows at each minibatch's ids. The held-out split is scored by steps after the fit, through `fit.rot` (`held_iia`). The run draws `rot` from the fit's `seed` and makes the optimizer. Each epoch it shuffles the training rows (by `seed` and the epoch), runs the body on each minibatch of `batch_size` rows with gradients on, and takes one optimizer step per minibatch on `1.0 × ce`. Then it runs the body once more over every training row without gradients: that scored pass is the epoch's [history](#history) entry and what `early_stop` judges. After the last epoch it restores the best epoch's tensors, and each [`eval`](#evals) runs after the fit: the body's steps once more, as ordinary steps over every row.

## The body

The body is a scope of its own. It sees every value made before the fit, and inside it a trained featurizer answers to both its declared name and `<fit>.<name>` (`rot` and `fit.rot`): compile writes the bare name as `fit.rot`, so the two are one featurizer, swept or not. A value from before the fit with a row per example of one of the body's datasets (a harvest) comes into every run of the body at that run's rows, taken by their ids, as made: so what does not depend on the training is a step before the fit, made once, and the model runs while the fit trains only where the body has a forward. `documents/sae_train.json`'s body is one `encode` step over its harvest, and its fit makes no model call. A value with no rows (a `mean`, a direction) is fine too. A harvest of another dataset than the body's is refused, since a minibatch's ids would pair its rows with others:

```text
clab: steps.fit.steps.patched.interventions.patch.ops[1].with: 'train.input' has a row per example of 'train' and 'v_cf' of 'test'; an op takes one dataset's rows
```

The body's datasets are those its forwards read (with none, the one its harvests are of), paired row by row, so they hold as many rows as each other. A reduction over rows inside the body is the minibatch's statistic; a statistic of the dataset is a step before the fit. A body holds no fit, and its step names are not dataset or featurizer names. The rules are in [FORMAT.md](../../FORMAT.md#fit).

## The objective

`objective` is `{term: weight}`, and the optimizer minimizes the weighted sum; a negative weight maximizes a term. A term is one of:

- **a body op**, averaged over every axis, missing values left out: `ce` above;
- **a trained featurizer's scalar**, `<featurizer>.<scalar>`: a gate's `mask` (the mean of its soft mask, the sparsity pressure in `documents/dbm.json`), an sae's `reconstruction` and `l1` (`documents/sae_train.json`). Subspaces, probes, adapters and affine maps have none.

An sae's scalars are over the coordinates the body reads of it with an `encode` no `decode` closes (an entry's or an `encode` step's). A body that only brackets the sae has nothing to compute them on, and compile refuses the term: `steps.fit.objective.dictionary.reconstruction: 'reconstruction' is over the coordinates the body reads of 'dictionary', an encode no decode closes ...`. Compile refuses anything that is neither, and an objective none of whose terms depends on what the fit trains, which has no gradient to step on:

```text
clab: steps.fit.objective.patched.logits: not a body op or a trained featurizer's scalar; body ops: v_cf, iia, ce
clab: steps.fit.objective: none of its terms (pce) depends on what the fit trains (rot): each is a body op over values no trained featurizer reaches, so there is no gradient to step on; make a term of an op over a value the trained featurizer's encode, decode or apply made, or of its scalar
```

A metric term is a float32 loss whatever the model's dtype, as every metric is ([ops.md](ops.md)).

## Epochs, minibatches and the learning rate

- **Count the optimizer steps**: `epochs × ⌈training rows / batch_size⌉`. `das.json` has two training rows and `batch_size` 16, so one step per epoch, ten in all.
- **Adam and AdamW move each parameter by about `lr` per step**, whatever the gradient's size, so `lr × steps` roughly bounds how far a featurizer gets. `das.json`'s ten steps turn its rotation's subspace by about 16 degrees from the draw at `lr` 0.001 and 59 degrees at 0.01 (the largest principal angle). When a history barely moves, suspect `lr × steps` first, both ways: too small turns nothing, and on a wide subspace too large scrambles the rotation and leaves the curve flat at the floor ([trained subspaces](../patterns/trained-subspaces.md#pitfalls)).
- **An epoch is two passes** over the training rows: one in minibatches with gradients, one scored.
- **Memory follows the fit's `batch_size`.** A minibatch's forwards run with gradients on, so their activations are kept for the backward. The body's forwards, and its evals', always run in batches of the fit's own `batch_size`, whatever `--batch-size` says. A step before the fit runs at `--batch-size`, so a harvest made there moves the last bits of what trains on it: `das.json` with its counterfactual read inside the body gives bit-identical bundles and evals at `--batch-size` 1 and 8, and `das.json` as written does not. Lower the fit's `batch_size` to fit in memory, and expect more optimizer steps per epoch.
- **The model's weights never change.** A forward before the fit and one after it give bit-identical logits.

## Early stopping

`early_stop` names a body op (`metric`), `mode` `max` or `min`, and `patience`. The fit ends after `patience` epochs without improvement on the scored pass, and the best epoch's tensors are restored, not the last's: `das.json` with `{"metric": "ce", "mode": "max", "patience": 1}` stops after two epochs, and its `fit.rot` is bit-identical to a one-epoch fit's. Without `early_stop` the last epoch is kept.

## Evals

Each name under `eval` is the body once more after training, scored, with each dataset on the left read as the one on the right; `{}` is the data it trained on. Every reference to the replaced dataset follows, columns and scopes included: under `held_out`, `train.cf_answer` reads `test.cf_answer`. Compile writes each eval out as ordinary steps after the fit, in the fit's record (its `evals`): the body's steps named `<fit>.<eval>.<step>` (`lora_train.json`'s `fit.held_out.adapted` and `fit.held_out.ce`), each over every row of its dataset, through `<fit>.<featurizer>`, on the fit's sweep axes, a forward in batches of the fit's `batch_size`. Each is then a step like any other: `clab show` lists its cost and where each of its entries resolved, and a refusal names it (`fit.held_out.adapted.adapt: row 0: …`). A replacement may hold a different number of rows, in a body that reads two datasets too, and its values are aligned to its own rows, so `documents/lora_train.json`'s `fit.held_out.ce` and `fit.train.ce` cannot share one `clab table`. An eval cannot replace a dataset that a harvest the body reads has a row per example of, since no step made the harvest over the other's rows; evaluate on other data in steps after the fit, as `das.json` does:

```text
clab: steps.fit.eval.held_out: the body uses 'counterfactual.h', made before the fit with a row per example of 'train', and this eval reads 'train' as 'test', over whose rows no step made it; evaluate on 'test' in steps after the fit, which reach what it trained as 'fit.rot'
```
 A chat renders with its own dataset's `template`, so a dataset and its replacement give the same one (`dataset 'train' is read as 'test', and their templates differ`); a body that reads text, not chats, gives neither one, since a `template` on a dataset no step reads as a chat is refused. An eval is not named `history` or after a featurizer the fit trains, and has no dot.

## Members

Nothing leaves the body except through these names:

| member | what it is |
|---|---|
| `<fit>.<featurizer>` | the trained featurizer; saved, it is a [bundle](bundles.md) |
| `<fit>.<eval>.<op>` | a body op's value in the eval named `<eval>`, its step's own (`fit.held_out.ce` in `lora_train.json`) |
| `<fit>.<eval>.<step>.<entry>` | a body entry's value in that eval, its forward's (`fit.held_out.adapted.logits`) |
| `<fit>.history.<op>` | each body op's mean after each epoch, `[epochs run]` |
| `<fit>.history.<featurizer>.<scalar>` | each trained featurizer's scalar after each epoch, `[epochs run]`; an sae's only when the body reads it with an `encode` no `decode` closes |

**After the fit, the bare name is the declared featurizer as drawn at the run's start, never trained.** In `das.json`, the held-out steps after the fit encode with `fit.rot`; written with `rot` they score the untrained seed-0 subspace. Name `<fit>.<featurizer>` everywhere after the fit.

That draw comes from the featurizer's own `seed`, 0 when unset, whatever the fit's: with `"seed": 1` on the fit and none on `rot`, the fit starts from the seed-1 draw and the bare `rot` is the seed-0 one. To make the bare name the fit's starting draw, as a control of matched rank does ([trained subspaces](../patterns/trained-subspaces.md#controls-of-matched-rank)), give the featurizer the fit's seed.

### History

A history holds one entry per epoch run: fewer when `early_stop` ends the fit, none (`[0]`) at `epochs: 0`. Every body op gets one, in the objective or not, so an aggregate you want to watch belongs in the body: a `mean` of a `match` there is both `<fit>.<eval>.acc` and `<fit>.history.acc`. `clab table <out> fit.history.ce` prints a row per epoch, with an `epochs` column. Under a sweep, every member is a dict keyed by coordinate ([sweeps.md](sweeps.md)).

## `epochs: 0` is a null control

`epochs: 0` trains nothing: the featurizers are scored as drawn, every eval runs, and every history is empty. What the draw is makes it the control:

- **an adapter** starts as the identity exactly: `documents/lora_train.json` at `epochs: 0` gives `fit.held_out.ce` bit-identical to a clean forward's;
- **an affine map** starts as the identity exactly too, its `W` and `b` zero, and draws nothing from the seed;
- **a subspace** is a uniformly random one of the fit's seed: `das.json` at `epochs: 0` gives the `held_iia` of a declared `{"kind": "subspace", "k": 8, "seed": 0}` used without a fit, to the bit;
- **a probe** scores every class 0, and `match` breaks the tie to the lower class id, so its accuracy is the share of rows whose label sorts first.

To run the control beside the trained featurizer, sweep `epochs`; with a sweep on `seed` too, `"epochs": {"sweep": [0, 10]}, "seed": {"sweep": [0, 1, 2]}` gives six fits that differ in nothing else, keyed `epochs=0,seed=1` and so on. The trained bundle is then one per point, which no document can load ([sweeping a fit](#sweeping-a-fit)).

## Sweeping a fit

Any of the fit's values may be a sweep, names such as `optimizer.name` excepted: an objective weight (`dbm.json` sweeps its sparsity weight, whose default label is the term's key, `mask.mask`), `lr`, `epochs`, `seed`, or the `layers` of a site the body uses, which trains one featurizer per layer. An [affine over layers](featurizers.md#an-affine-map) trains a map per layer in one fit instead, each minibatch's one forward serving all of them. Each point is a whole fit with its own members. A swept fit's bundle file holds a bundle per point, which no featurizer can load ([bundles.md](bundles.md)).
