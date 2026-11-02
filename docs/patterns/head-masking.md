---
title: Head Masking
one_liner: "Which attention heads carry a behavior, learned as a mask: desiderata-based masking with one gate unit per head of every layer (over), a dropped head taking the corrupt run's value through apply with, the bit-exact null, an L1 sweep, and reading the heads kept."
tags: [patterns, heads, gate, dbm, desiderata-based-masking, circuits]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/patterns/trained-subspaces.md, docs/patterns/head-and-neuron-patching.md, docs/usage/bundles.md, docs/usage/output.md]
sources: [clab/featurizers/gate.py, clab/compile/uses.py, clab/ops/edits.py, clab/steps/fit.py, documents/head_gates.json]
---

# Head Masking

[Head patching](head-and-neuron-patching.md) moves one head at a time and asks what changes. Desiderata-based masking asks which set of heads, all at once, carries a behavior: a gate with one unit per head is trained so that the gated run keeps the clean run's metric while an L1 term turns heads off, and a head that is off takes its value on a corrupt prompt. It is [DBM](trained-subspaces.md#dbm) over heads rather than over the units of one value.

## The document

`documents/head_gates.json` runs it on the tiny GPT-2, 5 layers of 4 heads. One featurizer and one site:

```json
"featurizers": {"heads": {"kind": "gate", "over": ["layers", "num_heads"], "temperature": {"start": 1.0, "end": 0.05}}},
"sites": {"premix": {"component": "attention_premix", "layers": ":"}}
```

`over` lays `theta` out on the layers and heads of the site's selection, `[5, 4]`, one unit per head, broadcast over `head_dim` ([featurizers.md](../usage/featurizers.md#a-gate-over-named-axes)). The corrupt run is read once, before the fit, at every layer and all 12 positions (`corrupt.z`, `[layers, rows, pos, num_heads, head_dim]`), and the clean run's logit difference is a step (`clean_ld`). The body has one write at every head of every layer:

```json
"run": {"kind": "forward", "input": "pairs.input", "interventions": {
  "gated": {"site": "premix", "pos": {"index": "0:12"},
            "ops": [{"op": "apply", "featurizer": "fit.heads", "with": "corrupt.z"}]},
  "logits": {"site": "head", "pos": -1}}},
"ld": {"kind": "op", "op": "logit_diff", "of": "run.logits", "a": "pairs.base_answer", "b": "pairs.cf_answer"},
"diff": {"kind": "op", "op": "sub", "of": "ld", "with": "clean_ld"},
"sq": {"kind": "op", "op": "square", "of": "diff"}
```

At layer L the write takes row L of the mask and the slice of `corrupt.z` labelled L, and `apply` with `with` writes `h·m + corrupt·(1−m)`: the clean value where a head is on, the corrupt one where it is off. The objective is the squared change from the clean logit difference plus the swept L1 weight:

```json
"objective": {"sq": 1.0, "heads.mask": {"sweep": [0.00002, 0.0001, 0.0005, 0.003]}}
```

`heads.mask` is the mean of the soft mask over every unit, so each head kept costs λ divided by the number of heads: scale a sweep with the head count when moving it to another model. The four fits keep 16, 5, 1 and 0 of the 20 heads. The weights are random; the heads kept show the mechanics.

## The null and its mirror

An untrained gate has every θ at +1, and `apply` with `with` is the value itself to the bit where a unit is on. Set the fit's `"epochs": 0`: its `train` eval scores that draw ([fit.md](../usage/fit.md#epochs-0-is-a-null-control)), and `fit.train.ld` equals `clean_ld` to the bit, on every engine. Where a unit is off the write is `with` to the bit: a gate off at every head writes `corrupt.z` exactly, which saving the write's own value (`fit.train.run.gated`) shows. The logits are then not the corrupt run's, since the embeddings and the MLPs still see the clean prompt.

Write the interpolation with `with`. `[{"op": "sub", "with": "corrupt.z"}, {"op": "apply", "featurizer": "fit.heads"}, {"op": "add", "with": "corrupt.z"}]` means the same thing and computes `(h − c)·1 + c`, which rounds: its null moves the logits in their last bits.

## Reading the heads kept

The fit's scoring passes and its `train` eval use the hard mask, θ > 0, so each weight's `fit.train.ld` is the metric of the heads it kept. `fit.heads` holds each point's `theta`, laid out `[layers, num_heads]` as the site selects them:

```python
from clab import Plan

plan = Plan.load("out/head_gates")
for weight, gate in plan.results["fit.heads"].items():
    print(weight, (gate["theta"] > 0).nonzero().tolist())     # [layer, head] pairs
```

An index is the selection's: with a `select` on `num_heads`, head i of `theta` is the i-th head the site keeps. Each point writes its own bundle, whose header records `over`, the layers and `kept.num_heads`; to load one in another document, rerun at that one weight ([bundles.md](../usage/bundles.md#swept-training-runs)).

## Choosing the weight

The objective is a mean over the training rows, so score the kept heads on held-out rows after the fit, against a floor. Add a held-out harvest and gated forward through `fit.heads`, and a forward with every head corrupted (`swap` with the held-out harvest at the same site), and pick the largest weight whose held-out metric stays within 10% of clean, relative to that floor:

```json
"t_corrupt": {"kind": "forward", "input": "test.corrupt", "interventions": {"z": {"site": "premix", "pos": {"index": "0:16"}}}},
"t_run": {"kind": "forward", "input": "test.input", "interventions": {
  "gated": {"site": "premix", "pos": {"index": "0:16"},
            "ops": [{"op": "apply", "featurizer": "fit.heads", "with": "t_corrupt.z"}]},
  "logits": {"site": "head", "pos": -1}}},
"t_floor": {"kind": "forward", "input": "test.input", "interventions": {
  "gated": {"site": "premix", "pos": {"index": "0:16"}, "ops": [{"op": "swap", "with": "t_corrupt.z"}]},
  "logits": {"site": "head", "pos": -1}}}
```

## On GPT-2 small

The same document on GPT-2 small (12 layers of 12 heads, `theta` `[12, 12]`) over IOI prompts, 8 templates of 16 tokens with ABBA and BABA orders, each row's corruption three fresh names in the same template, 200 pairs to train and 200 held out; the metric is the IO name's logit less the S name's; 40 epochs, minibatch 50, Adam at 0.1, λ from 1 to 100. Clean held-out logit difference 3.389, floor −0.197:

| λ | heads kept | held-out LD | of clean − floor |
|---|---|---|---|
| 1 | 70 | 3.304 | 97.6% |
| 3 | 46 | 2.981 | 88.6% |
| 10 | 32 | 3.378 | 99.7% |
| 30 | 19 | 3.418 | 100.8% |
| 100 | 10 | 1.473 | 46.6% |

At λ 30, the largest within 10%, the 19 heads hold the three name movers (9.6, 9.9, 10.0), three of the four S-inhibition heads (7.9, 8.6, 8.10; not 7.3) and all nine induction, duplicate-token and previous-token heads of the published IOI circuit; 16 of the 19 are in it. A fit of a λ point takes about 10 seconds on one GPU (324 model calls).

## Pitfalls

- **A gate over axes serves `apply` alone.** `dbm.json`'s `[encode, swap, decode]` bracket is refused for it by name; the interchange through it is `apply` with `with`.
- **Every use selects the same heads.** A gate trained over every layer and applied at fewer, by a later step of the same document, is refused before any forward; a bundle loaded in another document may be applied at some of the layers its header records.
- **After the fit, `fit.heads`.** The bare `heads` is the untrained draw, every head on, and a forward through it is the clean run.
- **The corrupt read and the write meet position for position.** Read and write at one static window (`{"index": "0:12"}` on rows of 12 tokens), on prompts that tokenize to the same length, or the operand is refused ([operand-does-not-fit.md](../errors/operand-does-not-fit.md)).
- **The mean is kept, not every row.** The objective is a mean over rows: on GPT-2 small at λ 30 the training rows' mean logit difference is within 0.05 of clean while single rows move by 1.7 in root mean square. Save `fit.train.ld` and compare it with `clean_ld` row by row before reading the set as each row's circuit.
