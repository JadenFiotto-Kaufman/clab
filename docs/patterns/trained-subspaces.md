---
title: Trained Subspaces
one_liner: "DAS, DBM and their controls: fit a rotation or a mask, score held-out accuracy, and compare against PCA and random subspaces of matched rank, a fit at epochs 0 among them."
tags: [patterns, subspaces, das, dbm, pca, interchange-intervention, controls]
related: [docs/usage/fit.md, docs/usage/featurizers.md, docs/usage/bundles.md, docs/usage/sweeps.md, docs/patterns/activation-patching.md]
sources: [clab/featurizers/subspace.py, clab/featurizers/gate.py, clab/featurizers/pca.py, clab/steps/fit.py]
---

# Trained Subspaces

Distributed alignment search (DAS) learns a rank-`k` subspace of one site's value such that swapping the counterfactual run's coordinates in that subspace into the base run moves the base's answer to the counterfactual's. Distributed boundless masking (DBM) learns which units to swap instead, as a mask. Both are an interchange intervention through a featurizer trained in a [fit](../usage/fit.md), and both mean something only next to controls of the same rank.

## DAS

`documents/das.json` trains `rot`, a `subspace` with `k: 8`, at layer 0's output at the last token. Its body is the experiment, pulled in from two root bundles:

```json
"interventions": {
  "cf_read": {"v_cf": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.rot"}]}},
  "das": {
    "patch": {"site": "target", "pos": -1, "ops": [
      {"op": "encode", "featurizer": "fit.rot"},
      {"op": "swap", "with": "counterfactual.v_cf"},
      {"op": "decode", "featurizer": "fit.rot"}
    ]},
    "logits": {"site": "head", "pos": -1}
  }
}
```

`counterfactual` reads the counterfactual prompt's coordinates; `patched` swaps them into the base prompt's and writes the change back, so only the `k` coordinates move and the rest of the vector stays the base's ([featurizers.md](../usage/featurizers.md)). The objective is the cross entropy of the patched logits against `train.cf_answer`; `early_stop` watches `iia`; the evals score the test split (`held_out`) and the training split (`train`).

In `das.json`, `iia` is a `logit_diff`, the counterfactual answer's logit less the base answer's, not an accuracy. For interchange intervention accuracy, add a `match` against `train.cf_answer` and a `mean` of it to the body; both then have a held-out value and a history.

The apply half, `documents/das_apply.json`, loads the saved `fit.rot` and runs the same interchange on the test split at the root, reproducing `fit.held_out.iia` to the bit. The workflow and its checks are in [bundles.md](../usage/bundles.md).

## DBM

`documents/dbm.json` is `das.json` with a `gate` (`mask`) in place of the rotation. The mask is soft, σ(θ/T), while the fit trains it and hard, θ > 0, whenever it is scored; `temperature` takes T from 1.0 to 0.05 linearly over the epochs. One more objective term pushes units off:

```json
"objective": {"ce": 1.0, "mask.mask": {"sweep": [0.0001, 0.002, 0.01, 0.05]}}
```

`mask.mask` is the mean of the soft mask. Swept, the document is a sparsity curve: the four fits keep 4, 2, 1 and 0 of the 16 units. The units a fit kept are where its saved `theta` is positive. Its bundle file holds one gate per point; to load one in another document, rerun at that one weight ([bundles.md](../usage/bundles.md#swept-training-runs)).

## Controls of matched rank

A trained subspace is evidence only where it beats subspaces nobody trained, at the same `k`, on the same rows, scored by the same op.

**Random subspaces.** A subspace drawn from a seed is uniformly random, and a fit at `epochs: 0` scores exactly that draw ([fit.md](../usage/fit.md#epochs-0-is-a-null-control)). So the control and the trained fit share one document. In `das.json`, set `rot`'s `k` to the rank you compare at, and change three values of the fit:

```json
"epochs": {"sweep": [0, 10]},
"seed": {"sweep": [0, 1, 2]},
"optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0}
```

`fit.held_out.iia` is then six values keyed `epochs=0,seed=0` through `epochs=10,seed=2`: the spread across seeds at `epochs=0` is the null each trained point has to clear. `documents/random_subspace.json` draws the same control without a fit, a declared subspace with its `seed` swept.

**Principal directions.** `documents/pca_harvest.json` reduces layer 0's output at every token after BOS to its top-4 directions; `documents/pca_control.json` loads them as a `pca` featurizer and runs `das_apply.json`'s interchange through them, under the same name `rot`, so the two documents differ only in what that featurizer is. A `pca` coordinate is centred on the harvest's mean, and the swap moves the counterfactual's centred coordinates in.

The corpus documents are not matched as they stand: `das.json` is rank 8, `pca_control.json` and `random_subspace.json` are rank 4, and `random_subspace.json` scores all four rows of `weekdays/train` while the others score the test split of `weekdays/data`. Set one `k` (the fit's `k`, or the `pca` op's), one dataset and one metric before comparing.

## Pitfalls

- **Few rows, few steps.** `das.json` trains on two pairs, one minibatch per epoch, so its ten epochs are ten optimizer steps, and `lr` decides how far those can turn the subspace. Count steps before reading a flat history ([fit.md](../usage/fit.md#epochs-minibatches-and-the-learning-rate)).
- **After the fit, use `fit.rot`.** A later step that names `rot` gets the declared, untrained draw ([fit.md](../usage/fit.md#members)).
- **`apply` is not an interchange.** A subspace's `apply` replaces the value with its projection onto the subspace, discarding the complement; the interchange is the `encode`, `swap`, `decode` bracket.
- **The fixture model is random.** Every number these documents give on the tiny Llama shows the mechanics; on it, the trained and random subspaces score alike.
