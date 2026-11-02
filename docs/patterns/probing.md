---
title: Probing
one_liner: "Linear probes: train a linear featurizer on a label column, score its classes with the same metrics as logits, sweep layers to see where a variable is decodable against the floor of an untrained probe, and train many probes in one fit."
tags: [patterns, probing, linear-probe, classes, decodability]
related: [docs/usage/fit.md, docs/usage/featurizers.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/positions.md, docs/usage/bundles.md, docs/patterns/trained-subspaces.md]
sources: [clab/featurizers/linear.py, clab/featurizers/__init__.py, clab/steps/fit.py]
---

# Probing

A linear probe is a `linear` featurizer: `h @ W + b`, one score per class, its classes the distinct values of a label column, sorted. A [fit](../usage/fit.md) trains it with `cross_entropy` against that column, and `match`, `token_prob`, `logit_diff` and the rest score its classes as they score logits against a column of tokens. Its fields are in [FORMAT.md](../../FORMAT.md#linear).

## A probe per layer

No corpus document trains a probe; this one is complete and runs on CPU in seconds. It reads which day a question starts from (`entity`, seven classes) out of each layer's output at the day's last token, trains on the 30 training rows and scores the 19 test rows:

```json
{
  "description": "A linear probe per layer for the day a question starts from: trained on the training split to read `entity` from layer L's output at the entity's last token, then scored on the test split with the metrics that score logits.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {
    "train": {"path": "natural_domains_arithmetic/data/weekdays", "split": "train"},
    "test": {"path": "natural_domains_arithmetic/data/weekdays", "split": "test"}
  },
  "sites": {"stream": {"component": "layer_output", "layers": {"sweep": [0, 1]}}},
  "featurizers": {"probe": {"kind": "linear", "classes": "train.entity"}},
  "steps": {
    "fit": {
      "kind": "fit",
      "train": ["probe"],
      "objective": {"ce": 1.0},
      "epochs": 50,
      "batch_size": 10,
      "seed": 0,
      "optimizer": {"name": "adam", "lr": 0.1},
      "eval": {"held_out": {"train": "test"}, "train": {}},
      "steps": {
        "read": {
          "kind": "forward",
          "input": "train.input",
          "interventions": {
            "scores": {
              "site": "stream",
              "pos": {"scope": {"column": "train.entity"}, "index": -1},
              "ops": [{"op": "encode", "featurizer": "fit.probe"}]
            }
          }
        },
        "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "train.entity"},
        "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "train.entity"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    }
  },
  "saves": ["fit.held_out.acc", "fit.train.acc", "fit.history.ce", "fit.held_out.hit", "fit.probe"]
}
```

```text
clab run documents/probe.json --out out/probe
clab table out/probe fit.held_out.acc
at            value
{"layers":0}  0.7368420958518982
{"layers":1}  0.7894737124443054
```

What each part does:

- **`classes: "train.entity"`** makes the class list `["Friday", "Monday", ..., "Wednesday"]`, sorted, and the probe's scores `[rows, pos, classes]`. The bundle's header records the list.
- **The sweep on the site's `layers`** makes two fits, each with its own probe, keyed `layers=0` and `layers=1`. A site whose `layers` picks both would train one probe shared by the two layers ([featurizers.md](../usage/featurizers.md#what-a-featurizer-cannot-do)).
- **`pos`** finds the row's `entity` in its prompt and takes its last token ([positions.md](../usage/positions.md)).
- **`ce` is the objective, `hit` and `acc` are reported.** `acc`, the `mean` of `hit`, is a body op, so it is both `fit.held_out.acc` and a per-epoch `fit.history.acc` ([fit.md](../usage/fit.md#history)).
- **The `held_out` eval** reads `test` wherever the body says `train`: the input, the scope's column and the target. The class list stays the training column's.

## Scoring classes

A metric scores a probe's `classes` axis against a column whose values are classes, exactly as it scores a `vocab_size` axis against a column of tokens ([ops.md](../usage/ops.md)): `match` (the highest score is the row's label), `cross_entropy`, `token_prob` (the softmax over classes), and `logit_diff` between two columns' classes. `argmax` and `top_k` rank `vocab_size` by default, so they need `"over": "classes"`; `top_k` then gives `probs`, a softmax over the classes, and its `ids` index the sorted class list.

Classes compare exactly, with no `strip` or `lower`: a column holding `yes` and `Yes` makes two classes. Two refusals guard the class list. Every value of the `classes` column is one type (all strings, all ints, all floats or all booleans):

```text
clab: featurizers.probe.classes: classes column 'train.entity' holds str (row 0 is 'a'), int (row 1 is 1); a probe's classes are one type, all strings, all ints, all floats or all booleans: write them in one
```

And a row scored against a value that is not a class is refused before any forward, here an eval on rows whose days the training split never had:

```text
clab: fit: eval 'held_out', with dataset 'train' read as 'test' (a row is the eval's dataset's): ce: target is 'entity' of 'test', whose row 0 is 'Saturday', not one of the classes of probe 'probe': ['Friday', 'Thursday']
```

## The floor

At `epochs: 0` the probe is untrained and its accuracy is the share of rows whose label sorts first ([fit.md](../usage/fit.md#epochs-0-is-a-null-control) says why). For this document that is 1 of 19 held-out rows (0.053) and 6 of 30 training rows (0.2). A probe is decoding something only where it clears its floor. Read at the prompt's last token (`"pos": -1`) instead of at the day, the same fit ends at exactly the floor on both layers of this model.

## Many probes in one fit

A sweep over layers runs a whole fit per point, each with its own forwards. Many probes can share one fit instead: one site per layer, one `linear` featurizer per (layer, position), all read in one forward of the body, each scored by its own `cross_entropy` term. A probe's parameters appear in its own term alone, so its gradient and Adam's step are those of the probe trained by itself: this is the separate fits side by side, at one forward per minibatch.

```json
{
  "description": "Three probes for the day a question starts from, trained side by side in one fit: at the day's last token in layers 0 and 1, and at the prompt's last token in layer 1. Each probe's parameters are in its own cross-entropy term alone.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {
    "train": {"path": "natural_domains_arithmetic/data/weekdays", "split": "train"},
    "test": {"path": "natural_domains_arithmetic/data/weekdays", "split": "test"}
  },
  "sites": {
    "l0": {"component": "layer_output", "layers": 0},
    "l1": {"component": "layer_output", "layers": 1}
  },
  "featurizers": {
    "day0": {"kind": "linear", "classes": "train.entity"},
    "day1": {"kind": "linear", "classes": "train.entity"},
    "last1": {"kind": "linear", "classes": "train.entity"}
  },
  "steps": {
    "fit": {
      "kind": "fit",
      "train": ["day0", "day1", "last1"],
      "objective": {"ce_day0": 1.0, "ce_day1": 1.0, "ce_last1": 1.0},
      "epochs": 50,
      "batch_size": 10,
      "seed": 0,
      "optimizer": {"name": "adam", "lr": 0.1},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "read": {
          "kind": "forward",
          "input": "train.input",
          "interventions": {
            "day0": {"site": "l0", "pos": {"scope": {"column": "train.entity"}, "index": -1},
                     "ops": [{"op": "encode", "featurizer": "fit.day0"}]},
            "day1": {"site": "l1", "pos": {"scope": {"column": "train.entity"}, "index": -1},
                     "ops": [{"op": "encode", "featurizer": "fit.day1"}]},
            "last1": {"site": "l1", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.last1"}]}
          }
        },
        "ce_day0": {"kind": "op", "op": "cross_entropy", "of": "read.day0", "target": "train.entity"},
        "ce_day1": {"kind": "op", "op": "cross_entropy", "of": "read.day1", "target": "train.entity"},
        "ce_last1": {"kind": "op", "op": "cross_entropy", "of": "read.last1", "target": "train.entity"},
        "hit_day0": {"kind": "op", "op": "match", "of": "read.day0", "expected": "train.entity"},
        "hit_day1": {"kind": "op", "op": "match", "of": "read.day1", "expected": "train.entity"},
        "hit_last1": {"kind": "op", "op": "match", "of": "read.last1", "expected": "train.entity"},
        "acc_day0": {"kind": "op", "op": "mean", "of": "hit_day0"},
        "acc_day1": {"kind": "op", "op": "mean", "of": "hit_day1"},
        "acc_last1": {"kind": "op", "op": "mean", "of": "hit_last1"}
      }
    }
  },
  "saves": ["fit.held_out.acc_day0", "fit.held_out.acc_day1", "fit.held_out.acc_last1",
            "fit.history.ce_day0", "fit.history.ce_day1", "fit.history.ce_last1",
            "fit.held_out.hit_day0", "fit.held_out.hit_day1", "fit.held_out.hit_last1",
            "fit.day0", "fit.day1", "fit.last1"]
}
```

Each probe's history, held-out scores and saved tensors equal those of the same probe trained alone, to the bit: `day0` and `day1` give the 0.737 and 0.789 of the swept document above, and `last1` sits at its floor, 0.053. On Llama-3.2-1B, 64 probes in one fit took 118 s against 289 s for a sweep of 4. `early_stop` watches one body op and ends the fit for every probe at once, so leave it out here, or give each probe its own fit.

## Loading a trained probe

`fit.probe` saved is a bundle ([bundles.md](../usage/bundles.md)). The loading document declares `{"kind": "linear", "classes": "<dataset>.<column>", "file_path": ...}`, and its column's class list must be the header's. A probe trained on this document's layer 1 alone, loaded with `classes` from a dataset that holds only two of the days:

```text
clab: featurizers.probe.file_path: documents/artifacts/fit.probe.safetensors: header 'classes' is '["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]'; this document expects '["Friday", "Thursday"]'
```

Saved from this swept document, the file holds one probe per layer, which no featurizer loads; fit the layer you want unswept first.

## Pitfalls

- **A word of several tokens.** `"index": -1` of the scope is the word's last token, and on this tokenizer Tuesday, Wednesday and Thursday all end in `day`. At the word's first token (`"index": 0`) the same probe reaches 1.0 on both layers. Choose the position for the question, and check `pos_token` in `clab table` ([positions.md](../usage/positions.md)).
- **Decodable is not used.** A probe that reads a variable says the information is there, not that the model uses it there. clab does not intervene along a probe's direction: a `linear` featurizer only encodes (a `decode` of it is refused, `it has no way back into the value`), and a featurizer is not an operand. Test use with an interchange through a trained subspace ([trained-subspaces.md](trained-subspaces.md)).
- **The fixture model is random.** The accuracies above show the mechanics on the tiny Llama; they say nothing about where a trained model keeps the day.
