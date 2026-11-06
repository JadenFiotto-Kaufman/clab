---
title: Head Masking
one_liner: "Which heads does a behavior need? A learned gate per head of every layer, its nulls, the L1 weight, and the heads kept."
tags: [patterns, heads, gate, dbm, desiderata-based-masking, circuits]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/patterns/trained-subspaces.md, docs/patterns/head-and-neuron-patching.md, docs/usage/bundles.md, docs/usage/output.md]
sources: [nnspec/featurizers/gate.py, nnspec/compile/uses.py, nnspec/ops/edits.py, nnspec/steps/fit.py, documents/head_gates.json]
---

# Head Masking

[Head patching](head-and-neuron-patching.md) moves one head at a time and asks what changes. Desiderata-based masking asks which set of heads, all at once, carries a behavior. A gate with one unit per head is trained so that the gated run keeps the clean run's metric while an L1 term turns heads off, and a head that is off takes its value on a corrupt prompt. It is [DBM](trained-subspaces.md#dbm) over heads rather than over the units of one value.

## The document

`documents/head_gates.json` runs it on the tiny GPT-2, 5 layers of 4 heads. One featurizer and one site:

```json
"featurizers": {"heads": {"kind": "gate", "over": ["layers", "num_heads"], "temperature": {"start": 1.0, "end": 0.05}}},
"sites": {"premix": {"component": "attention_premix", "layers": ":"}}
```

`over` lays `theta` out `[5, 4]`, one unit per head, broadcast over `head_dim` ([featurizers.md](../usage/featurizers.md#a-gate-over-named-axes)). The corrupt run is read once, before the fit, at every layer and all 12 positions (`corrupt.z`, `[layers, rows, pos, num_heads, head_dim]`), and the clean run's logit difference is a step (`clean_ld`). The body has one write at every head of every layer:

```json
"run": {"kind": "forward", "input": "pairs.input", "interventions": {
  "gated": {"site": "premix", "pos": {"index": "0:12"},
            "ops": [{"op": "apply", "featurizer": "fit.heads", "with": "corrupt.z"}]},
  "logits": {"site": "head", "pos": -1}}},
"ld": {"kind": "op", "op": "logit_diff", "of": "run.logits", "a": "pairs.base_answer", "b": "pairs.cf_answer"},
"diff": {"kind": "op", "op": "sub", "of": "ld", "with": "clean_ld"},
"sq": {"kind": "op", "op": "square", "of": "diff"}
```

At layer L the write takes row L of the mask and the slice of `corrupt.z` labelled L. `apply` with `with` writes `h·m + corrupt·(1−m)`: the clean value where a head is on, the corrupt one where it is off. The objective is the squared change from the clean logit difference plus the swept L1 weight:

```json
"objective": {"sq": 1.0, "heads.mask": {"sweep": [0.00002, 0.0001, 0.0005, 0.003]}}
```

`heads.mask` is the mean of the soft mask over every unit, so each head kept costs λ divided by the number of heads. Scale a sweep with the head count when moving it to another model. The four fits keep 16, 5, 1 and 0 of the 20 heads (`tests/snapshots/head_gates.heads.json`). The weights are random; the heads kept show the mechanics.

## The null and its mirror

- **The untrained gate is the clean run.** Every θ starts at +1, and `apply` with `with` is the value itself where a unit is on. Set the fit's `"epochs": 0`: its `train` eval scores that draw ([fit.md](../usage/fit.md#epochs-0-is-a-null-control)), and `fit.train.ld` equals `clean_ld` on every engine (`tests/test_fit.py` `test_an_untrained_head_gate_with_the_corrupt_value_is_the_clean_run_to_the_bit`).
- **A gate off everywhere writes `with` itself.** Where a unit is off the write is `corrupt.z` exactly, which saving the write's own value (`fit.train.run.gated`) shows.
- **That is the corrupt run only where the two prompts share the read's token.** With every head off at every position, the read's position takes nothing from the others but the corrupt run's attention. Its own embedding and MLPs see its own token, so the logits there are the corrupt run's wherever that token is the same on both prompts.
- **Measured on GPT-2** (`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU, float32), with every head of every layer off at all 14 positions: a `swap` of `corrupt.z`, read at `attention_premix` over `"layers": ":"` at `{"index": "0:14"}`, which is what a gate off everywhere writes. The logits were read at `lm_head_output`, `pos: -1`, in one batch. Two IOI pairs that differ only at the subject's second mention gave logits equal to the corrupt run's ([running.md](../usage/running.md#when-two-runs-give-the-same-numbers)):

  ```json
  {"input": "When Mary and John went to the store, John gave a drink to",
   "cf": "When Mary and John went to the store, Mary gave a drink to"},
  {"input": "When Tom and Anna went to the park, Tom gave a ball to",
   "cf": "When Tom and Anna went to the park, Anna gave a ball to"}
  ```

  The first pair with its corrupt prompt ending `a drink for` instead, which differs at the last token, the one read, gave logits up to 9.0 from the corrupt run's.
- **Write the interpolation with `with`.** `[{"op": "sub", "with": "corrupt.z"}, {"op": "apply", "featurizer": "fit.heads"}, {"op": "add", "with": "corrupt.z"}]` means the same thing and computes `(h − c)·1 + c`, which rounds. Its null moves the logits in their last bits (`tests/test_fit.py` `test_the_sub_apply_add_form_of_the_interpolation_is_not_exact`).

## Reading the heads kept

The fit's scoring passes and its `train` eval use the hard mask, θ > 0, so each weight's `fit.train.ld` is the metric of the heads it kept. `fit.heads` holds each point's `theta`, laid out `[layers, num_heads]` as the site selects them:

```python
from nnspec import Plan

plan = Plan.load("out/head_gates")
for weight, gate in plan.results["fit.heads"].items():
    print(weight, (gate["theta"] > 0).nonzero().tolist())     # [layer, head] pairs
```

An index is the selection's: with a `select` on `num_heads`, head i of `theta` is the i-th head the site keeps. The swept fit writes one file, `fit.heads.safetensors`, holding a bundle per point under keys prefixed by its coordinate (`heads.mask=0.0001/theta`), which no featurizer loads. To load one gate in another document, rerun at that one weight ([bundles.md](../usage/bundles.md#swept-training-runs)).

## Choosing the weight

The objective is a mean over the training rows, so choose the weight on held-out rows, after the fit. `head_gates.json` has one dataset; bring a held-out one of prompts that tokenize to the same length. In steps after the fit:

- **a harvest** of the held-out corrupt prompts at `premix`, at the same window as the body's;
- **a gated forward** through `fit.heads` with that harvest as `with`, which scores the kept heads;
- **a floor**, the same forward with a `swap` of the harvest at every head.

Pick the largest weight whose held-out metric stays within a tolerance of clean, relative to that floor, fixed before the sweep.

## Pitfalls

- **A gate over axes serves `apply` alone.** `dbm.json`'s `[encode, swap, decode]` bracket is refused for it by name; the interchange through it is `apply` with `with`.
- **Every use selects the same heads.** A gate trained over every layer and applied at fewer, by a later step of the same document, is refused before any forward. A bundle loaded in another document may be applied at some of the layers its header records.
- **The corrupt read and the write meet position for position.** Read and write at one static window (`{"index": "0:12"}` on rows of 12 tokens), on prompts that tokenize to the same length, or the operand is refused ([operand-does-not-fit.md](../errors/operand-does-not-fit.md)).
- **The mean is kept, not every row.** The objective is a mean over rows, so a mask can keep the mean while single rows move. Save `fit.train.ld` and compare it with `clean_ld` row by row before reading the set as each row's circuit.
