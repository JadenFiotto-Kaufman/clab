---
title: Trained Subspaces
one_liner: "Where is a variable stored? DAS and DBM, scored held out against PCA and random subspaces of matched rank."
tags: [patterns, subspaces, das, dbm, pca, interchange-intervention, controls]
related: [docs/usage/fit.md, docs/usage/featurizers.md, docs/usage/bundles.md, docs/usage/sweeps.md, docs/patterns/activation-patching.md, docs/patterns/head-masking.md]
sources: [clab/featurizers/subspace.py, clab/featurizers/gate.py, clab/featurizers/pca.py, clab/steps/fit.py]
---

# Trained Subspaces

Distributed alignment search (DAS) learns a rank-`k` subspace of one site's value such that swapping the counterfactual run's coordinates in that subspace into the base run moves the base's answer to the counterfactual's. Distributed boundless masking (DBM) learns which units to swap instead, as a mask. Both are an interchange intervention through a featurizer trained in a [fit](../usage/fit.md), and both mean something only next to controls of the same rank.

## DAS

`documents/das.json` trains `rot`, a `subspace` with `k: 8`, at layer 0's output at the last token.

**Its `iia` is a `logit_diff`**, the counterfactual answer's logit less the base answer's, not an accuracy. For interchange intervention accuracy, add a `match` and its `mean` to the body, which gives them a training value and a history, and the same two over `held_patched.logits` after the fit:

```json
"hit": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "train.cf_answer"},
"acc": {"kind": "op", "op": "mean", "of": "hit"}
```

What does not depend on the training runs once, before the fit: `counterfactual` harvests the counterfactual prompt's value at the site, `counterfactual.h`, a row per training pair. The body is what trains, and takes each minibatch's rows of that harvest by their ids:

```json
"v_cf": {"kind": "op", "op": "encode", "featurizer": "fit.rot", "of": "counterfactual.h"},
"patched": {"kind": "forward", "input": "train.input", "interventions": {
  "patch": {"site": "target", "pos": -1, "ops": [
    {"op": "encode", "featurizer": "fit.rot"},
    {"op": "swap", "with": "v_cf"},
    {"op": "decode", "featurizer": "fit.rot"}
  ]},
  "logits": {"site": "head", "pos": -1}}}
```

- **The bracket moves only the `k` coordinates.** `v_cf` is the harvest's coordinates in the rotation being trained, and `patched` swaps them into the base prompt's. `decode` writes the change back, so the rest of the vector stays the base's ([featurizers.md](../usage/featurizers.md#the-canonical-entry)).
- **The objective** is the cross entropy of the patched logits against `train.cf_answer`; `early_stop` watches `iia`.
- **Score the held-out split in steps after the fit**, through `fit.rot`: `held_counterfactual`, `held_patched` and `held_iia`. An eval cannot read `train` as `test` here, since the harvest is the training rows' ([fit.md](../usage/fit.md#evals)).

The apply half, `documents/das_apply.json`, loads the saved `fit.rot` and runs the same interchange on the test split, giving `held_iia`'s numbers. The workflow and its checks are in [bundles.md](../usage/bundles.md).

## DBM

`documents/dbm.json` is `das.json` with a `gate` (`mask`) in place of the rotation, the same harvest before the fit and the same held-out steps after it. The mask is soft, σ(θ/T), while the fit trains it and hard, θ > 0, whenever it is scored; `temperature` takes T from 1.0 to 0.05 linearly over the epochs. One more objective term pushes units off:

```json
"objective": {"ce": 1.0, "mask.mask": {"sweep": [0.0001, 0.002, 0.01, 0.05]}}
```

`mask.mask` is the mean of the soft mask. Swept, the document is a sparsity curve: the four fits keep 4, 2, 1 and 0 of the 16 units (`tests/snapshots/dbm.units.json`). The units a fit kept are where its saved `theta` is positive. Its bundle file holds one gate per point; to load one in another document, rerun at that one weight ([bundles.md](../usage/bundles.md#swept-training-runs)). A gate whose units are whole attention heads, every layer's at once, is [head-masking.md](head-masking.md).

## Controls of matched rank

A trained subspace is evidence only where it beats subspaces nobody trained, at the same `k`, on the same rows, scored by the same op. Each control's value is per row (`held_iia` is `[2, 1]` per point), so take a `mean` of it per point, `{"kind": "op", "op": "mean", "of": "held_iia"}`, before comparing.

- **Random subspaces, in the training document.** A subspace drawn from a seed is uniformly random, and a fit at `epochs: 0` scores exactly that draw ([fit.md](../usage/fit.md#epochs-0-is-a-null-control)). In `das.json`, set `rot`'s `k` to the rank you compare at, and sweep two values of the fit, `"epochs": {"sweep": [0, 10]}` and `"seed": {"sweep": [0, 1, 2]}`. The spread across seeds at `epochs=0` is the null each trained point has to clear.
- **The untrained draw beside one saved fit.** A swept fit writes one bundle per point, which `das_apply.json` cannot load ([bundles.md](../usage/bundles.md#swept-training-runs)). To save the rotation and carry its control, leave the fit unswept, give `rot` the fit's seed (`{"kind": "subspace", "k": 8, "seed": 0}`), and score the held-out split again with the bare `rot`, the declared draw ([fit.md](../usage/fit.md#members)). Its `ctl_iia` equals the swept document's `held_iia` at `epochs=0,seed=0`.
- **Random subspaces without a fit.** `documents/random_subspace.json` is a declared subspace with its `seed` swept.
- **Principal directions.** `documents/pca_harvest.json` reduces layer 0's output at every token after BOS to its top-4 directions. `documents/pca_control.json` loads them as a `pca` featurizer and runs `das_apply.json`'s interchange through them, under the same name `rot`, so the two documents differ only in what that featurizer is. A `pca` coordinate is centred on the harvest's mean, and the swap moves the counterfactual's centred coordinates in.

The unswept control's steps after the fit, beside `das.json`'s own held-out ones:

```json
"ctl_counterfactual": {"kind": "forward", "input": "test.counterfactual_inputs[0]",
                       "interventions": {"v_cf": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "rot"}]}}},
"ctl_patched": {"kind": "forward", "input": "test.input", "interventions": {
  "patch": {"site": "target", "pos": -1, "ops": [
    {"op": "encode", "featurizer": "rot"},
    {"op": "swap", "with": "ctl_counterfactual.v_cf"},
    {"op": "decode", "featurizer": "rot"}
  ]},
  "logits": {"site": "head", "pos": -1}}},
"ctl_iia": {"kind": "op", "op": "logit_diff", "of": "ctl_patched.logits", "a": "test.cf_answer", "b": "test.base_answer"}
```

On the corpus model the trained points sit inside the seeds' spread. `das.json` swept as above, with the `mean` step `held_mean`:

```text
clab table out/das_swept held_mean
at                      value
{"epochs":0,"seed":0}   -0.006963685154914856
{"epochs":0,"seed":1}   -0.002706684172153473
{"epochs":0,"seed":2}   0.001060575246810913
{"epochs":10,"seed":0}  -0.007296368479728699
{"epochs":10,"seed":1}  -0.0025233030319213867
{"epochs":10,"seed":2}  0.000816836953163147
```

The corpus documents are not matched as they stand: `das.json` is rank 8, `pca_control.json` and `random_subspace.json` are rank 4, and `random_subspace.json` scores all four rows of `weekdays/train` while the others score the test split of `weekdays/data`. Set one `k` (the fit's `k`, or the `pca` op's), one dataset and one metric before comparing.

## Pitfalls

- **The learning rate on a wide model.** A `subspace` trains `theta`, a width × width matrix whose `theta − thetaᵀ` gives the rotation through the Cayley transform (`clab/featurizers/subspace.py` `Subspace`). Adam moves every entry of it by about `lr` per step, so a rate that suits the corpus's 16-wide fixture can scramble a wide model's rotation and leave the curve flat at the floor. Start near 1e-3 and sweep `lr` before reading a flat curve as no subspace; count the optimizer steps too ([fit.md](../usage/fit.md#epochs-minibatches-and-the-learning-rate)).
- **`apply` is not an interchange.** A subspace's `apply` replaces the value with its projection onto the subspace, discarding the complement; the interchange is the `encode`, `swap`, `decode` bracket.
- **The fixture model is random.** Every number these documents give on the tiny Llama shows the mechanics, nothing about where a trained model keeps a variable.
