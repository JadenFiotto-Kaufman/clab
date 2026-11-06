---
title: Fit
one_liner: "How do I train a featurizer inside a run? Objective, optimizer, epochs, early stopping, evals, history, what a fit costs, and the epochs-0 null."
tags: [usage, fit, training, objective, eval, history, optimizer, memory]
related: [docs/usage/featurizers.md, docs/usage/bundles.md, docs/usage/sweeps.md, docs/usage/ops.md, docs/usage/running.md, docs/concepts/names-and-sweeps.md, docs/patterns/trained-subspaces.md, docs/patterns/probing.md, docs/patterns/sae.md, docs/patterns/lora.md]
sources: [clab/steps/fit.py, clab/steps/batches.py, clab/compile/__init__.py, clab/plan.py, clab/featurizers/__init__.py]
---

# Fit

A `fit` step trains the featurizers its `train` names, inside the run, through the frozen model. It runs its own `steps` (the body) over minibatches of the training rows and steps an optimizer on a weighted `objective`. Gradients are taken for the featurizers' tensors alone. The fields are in [FORMAT.md](../../FORMAT.md#fit), the optimizer and early stopping in [`fit` blocks](../../FORMAT.md#fit-blocks).

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

What the run does with it:

- `counterfactual` runs once, before the fit, since its value does not depend on the training.
- The run draws `rot` from the fit's `seed` and makes the optimizer.
- Each epoch shuffles the training rows, by `seed` and the epoch.
- The body runs on each minibatch of `batch_size` rows with gradients on, and the optimizer takes one step per minibatch on `1.0 × ce`.
- Then the body runs once more over every training row, without gradients. That scored pass is the epoch's [history](#history) entry and what `early_stop` judges.
- After the last epoch the run restores the best epoch's tensors; without `early_stop` it keeps the last epoch's.
- Each [`eval`](#evals) then runs the body over every row of its data, as ordinary steps.
- Steps after the fit score the held-out split through `fit.rot` (`held_iia`).

## The body

The body is a scope of its own, and it sees every value made before the fit. Inside it a trained featurizer answers to both its declared name and `<fit>.<name>` (`rot` and `fit.rot`). Compile writes the bare name as `fit.rot`, so the two are one featurizer, swept or not.

A value from before the fit with a row per example of one of the body's datasets is a harvest. Each run of the body takes the harvest's rows by that run's row ids, as made. So a value that does not depend on the training belongs in a step before the fit, made once, and the model runs during training only where the body has a forward. `documents/sae_train.json`'s body is one `encode` step over its harvest, and its fit makes no model call. A value with no rows (a `mean`, a direction) is fine too.

A harvest of another dataset than the body's is refused, since a minibatch's ids would pair its rows with others:

```text
clab: steps.fit.steps.patched.interventions.patch.ops[1].with: 'train.input' has a row per example of 'train' and 'v_cf' of 'test'; an op takes one dataset's rows
```

The body's datasets are those its forwards read (with none, the one its harvests are of), paired row by row, so they hold as many rows as each other. A reduction over rows inside the body is the minibatch's statistic; a statistic of the dataset is a step before the fit. A body holds no fit, and its step names are not dataset or featurizer names. The rules are in [FORMAT.md](../../FORMAT.md#fit).

## The objective

`objective` is `{term: weight}`, and the optimizer minimizes the weighted sum; a negative weight maximizes a term. A term is one of:

- **a body op**, averaged over every axis, missing values left out: `ce` above;
- **a trained featurizer's scalar**, `<featurizer>.<scalar>`. Only a gate and an sae have scalars: a gate's `mask` (the mean of its soft mask, the sparsity pressure in `documents/dbm.json`), an sae's `reconstruction` and `l1` (`documents/sae_train.json`).

An sae's scalars are over the coordinates the body reads of it with an `encode` no `decode` closes: an entry's, or an `encode` step's. A body that only brackets the sae has nothing to compute them on, and compile refuses the term: `steps.fit.objective.dictionary.reconstruction: 'reconstruction' is over the coordinates the body reads of 'dictionary', an encode no decode closes ...`.

Compile also refuses a term that is neither, and an objective none of whose terms depends on what the fit trains, which has no gradient to step on:

```text
clab: steps.fit.objective.patched.logits: not a body op or a trained featurizer's scalar; body ops: v_cf, iia, ce
clab: steps.fit.objective: none of its terms (pce) depends on what the fit trains (rot): each is a body op over values no trained featurizer reaches, so there is no gradient to step on; make a term of an op over a value the trained featurizer's encode, decode or apply made, or of its scalar
```

A metric term is a float32 loss whatever the model's dtype, as every metric is ([ops.md](ops.md#the-float32-rule)).

## Epochs, minibatches and the learning rate

- **Count the optimizer steps**: `epochs × ⌈training rows / batch_size⌉`. `das.json` has two training rows and `batch_size` 16, so one step per epoch, ten in all.
- **Adam and AdamW move each parameter by about `lr` per step**, whatever the gradient's size, so `lr × steps` roughly bounds how far a featurizer gets. When a history barely moves, suspect `lr × steps` first, both ways. Too small turns nothing; on a wide subspace, too large scrambles the rotation and leaves the curve flat at the floor ([trained subspaces](../patterns/trained-subspaces.md#pitfalls)).
- **The body runs at the fit's `batch_size`.** The body's forwards, and its evals', run in batches of the fit's own `batch_size`, whatever `--batch-size` says. A step before the fit runs at `--batch-size`. Another `--batch-size` can move a harvest's last bits, and the optimizer carries that change into what the fit trains. To keep `--batch-size` out of a fit's numbers, read the harvest inside the body, at the cost of a forward per minibatch ([running.md](running.md#when-two-runs-give-the-same-numbers)).
- **The model's weights never change.** Gradients reach the trained featurizers' tensors alone, so a forward after the fit gives the logits a forward before it gave.

## What a fit costs

A fit holds three things on the model's device, and its `batch_size` bounds only the first:

- **The minibatch's backward.** A minibatch's forwards run with gradients on and keep their activations for the backward. A smaller `batch_size` shrinks them and takes more optimizer steps per epoch.
- **The scored pass.** After every epoch the body runs once more over every training row, in batches of `batch_size`, and each value it reads is joined over all of those rows on the model's device. Nothing turns this pass off, so it adds a forward per batch of training rows to every epoch: `tuned_lens.json`'s fit makes 2 × (4 + 4) forwards for 2 epochs of 4 minibatches (`tests/test_fit.py`).
- **The evals.** Every eval is ordinary steps after the fit, and each value they make stays on the model's device until the run ends, one set per sweep point.

To fit the scored pass and the evals in memory, read fewer positions, train on fewer rows, and score logits inside the head entry rather than reading them whole ([ops.md](ops.md#a-step-or-an-entry)). A step that runs out of memory is refused with every value the run holds ([running.md](running.md#when-a-step-runs-out-of-memory)).

## Early stopping

`early_stop` names a body op (`metric`), `mode` `max` or `min`, and `patience`. The fit ends after `patience` epochs without improvement on the scored pass, and the best epoch's tensors are restored, not the last's. `das.json` with `{"metric": "ce", "mode": "max", "patience": 1}` stops after two epochs, and its `fit.rot` equals a one-epoch fit's. Without `early_stop` the last epoch is kept.

## Evals

Each name under `eval` runs the body once more after training, scored, with each dataset on the left read as the one on the right:

- **`{}` is the data it trained on**; `{"train": "test"}` reads `test` wherever the body says `train`.
- **Every reference to the replaced dataset follows**, columns and scopes included: under `held_out`, `train.cf_answer` reads `test.cf_answer`.
- **An eval is ordinary steps after the fit.** Compile writes the body's steps out as `<fit>.<eval>.<step>` (`lora_train.json`'s `fit.held_out.adapted` and `fit.held_out.ce`), listed in the fit's record (its `evals`). Each runs over every row of its dataset, through `<fit>.<featurizer>`, on the fit's sweep axes, in batches of the fit's `batch_size`.
- **It reports as a step.** `clab show` lists its cost and where each of its entries resolved, and a refusal names it (`fit.held_out.adapted.adapt: row 0: …`).
- **A replacement may hold another number of rows**, and its values are aligned to its own rows. So `lora_train.json`'s `fit.held_out.ce` and `fit.train.ce` cannot share one `clab table` (`'fit.held_out.ce' has a row per example of 'test' and 'fit.train.ce' of 'train'; a table joins the values of one dataset`).
- **A chat dataset's replacement renders with the same `template`**; one with another is refused (`dataset 'train' is read as 'test', and their templates differ`).
- **An eval's name** is not `history` or a featurizer the fit trains, and has no dot.

An eval cannot replace the dataset of a harvest the body reads, since no step made the harvest over the other's rows. Evaluate on other data in steps after the fit, as `das.json` does:

```text
clab: steps.fit.eval.held_out: the body uses 'counterfactual.h', made before the fit with a row per example of 'train', and this eval reads 'train' as 'test', over whose rows no step made it; evaluate on 'test' in steps after the fit, which reach what it trained as 'fit.rot'
```

## Members

Nothing leaves the body except through these names:

| member | what it is |
|---|---|
| `<fit>.<featurizer>` | the trained featurizer; saved, it is a [bundle](bundles.md) |
| `<fit>.<featurizer>.<tensor>` | one of its tensors as the fit left it, the best epoch's: a value (`fit.probe.W`, `[hidden_size, classes]`), laid out as [featurizers.md](featurizers.md#its-tensors-as-values) lists each kind's |
| `<fit>.<eval>.<op>` | a body op's value in the eval named `<eval>`, its step's own (`fit.held_out.ce` in `lora_train.json`) |
| `<fit>.<eval>.<step>.<entry>` | a body entry's value in that eval, its forward's (`fit.held_out.adapted.logits`) |
| `<fit>.history.<op>` | each body op's mean after each epoch, `[epochs run]` |
| `<fit>.history.<featurizer>.<scalar>` | each trained featurizer's scalar after each epoch, `[epochs run]`; an sae's only when the body reads it with an `encode` no `decode` closes |

**After the fit, the bare name is the declared featurizer as drawn, never trained.** In `das.json`, the held-out steps after the fit encode with `fit.rot`; written with `rot` they score the untrained seed-0 subspace. Name `<fit>.<featurizer>` everywhere after the fit. Its tensors follow the same rule: `fit.rot.basis` is the trained basis, `rot.basis` the draw's.

That draw comes from the featurizer's own `seed`, 0 when unset, whatever the fit's. With `"seed": 1` on the fit and none on `rot`, the fit starts from the seed-1 draw and the bare `rot` is the seed-0 one. To make the bare name the fit's starting draw, as a control of matched rank does ([trained subspaces](../patterns/trained-subspaces.md#controls-of-matched-rank)), give the featurizer the fit's seed.

### History

A history holds one entry per epoch run: fewer when `early_stop` ends the fit, none (`[0]`) at `epochs: 0`. Every body op gets one, in the objective or not, so an aggregate you want to watch belongs in the body: a `mean` of a `match` there is both `<fit>.<eval>.acc` and `<fit>.history.acc`. `clab table <out> fit.history.ce` prints a row per epoch, with an `epochs` column. Under a sweep, every member is a dict keyed by coordinate ([sweeps.md](sweeps.md#reading-swept-results)).

## `epochs: 0` is a null control

`epochs: 0` trains nothing: the featurizers are scored as drawn, every eval runs, and every history is empty. What each kind's draw is makes it the control:

| kind | what `epochs: 0` scores | takes a `seed` |
|---|---|---|
| `subspace` | a uniformly random subspace: `das.json` at `epochs: 0` gives the `held_iia` of a declared `{"kind": "subspace", "k": 8, "seed": 0}` used without a fit ([trained subspaces](../patterns/trained-subspaces.md#controls-of-matched-rank)) | yes |
| `gate` | every unit on, so `apply` is the value, and `apply` with `with` gives the clean run ([head masking](../patterns/head-masking.md#the-null-and-its-mirror)) | no |
| `sae` | the dictionary drawn from `latents` and the seed | yes |
| `linear` | every class scored 0; `match` breaks the tie to the lower class id, so accuracy is the share of rows whose label sorts first ([probing](../patterns/probing.md#the-floor)) | no |
| `lora` | the identity, since `B` is drawn as zeros: `lora_train.json` gives a clean forward's `fit.held_out.ce` ([LoRA](../patterns/lora.md#train-an-adapter)) | yes, for `A` |
| `affine` | the identity, `W` and `b` zero ([tuned lens](../patterns/logit-lens.md#a-tuned-lens)) | no |
| `constant` | its `init`: `ablation_pair.json` gives mean ablation's logit difference ([ablation](../patterns/ablation.md#the-optimal-constant)) | no |

A `pca` is never trained. The seed is the featurizer's own `seed`, else the fit's; a kind that takes none refuses one (`'constant' takes no 'seed'; its arguments: init, file_path`).

To run the control beside the trained featurizer, sweep `epochs`. With a sweep on `seed` too, `"epochs": {"sweep": [0, 10]}, "seed": {"sweep": [0, 1, 2]}` gives six fits that differ in nothing else, keyed `epochs=0,seed=1` and so on. The trained bundle is then one per point, which no document can load ([sweeping a fit](#sweeping-a-fit)).

## Sweeping a fit

Any of the fit's values may be a sweep, names such as `optimizer.name` and `train` excepted: an objective weight, `lr`, `epochs`, `seed`, the `layers` of a site the body uses, which trains one featurizer per layer, or a body op's `featurizer` over featurizers the fit does not train ([featurizers.md](featurizers.md#a-sweep-over-featurizers)). A sweep over the featurizers the fit trains is refused, since each point would train every one and use one. `dbm.json` sweeps its sparsity weight, whose default label is the term's key, `mask.mask`. Each point is a whole fit with its own members and its own forwards; an affine over layers trains a map per layer in one fit instead ([tuned lens](../patterns/logit-lens.md#a-tuned-lens)). A swept fit's bundle file holds a bundle per point, which no featurizer can load ([bundles.md](bundles.md#swept-training-runs)).
