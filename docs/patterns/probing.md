---
title: Probing
one_liner: "Is a variable linearly decodable, and at which layer? A linear probe per layer, its untrained floor, many probes in one fit."
tags: [patterns, probing, linear-probe, classes, decodability]
related: [docs/usage/fit.md, docs/usage/featurizers.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/positions.md, docs/usage/bundles.md, docs/patterns/trained-subspaces.md]
sources: [clab/featurizers/linear.py, clab/featurizers/__init__.py, clab/steps/fit.py, clab/ops/transforms.py, documents/probe_direction.json]
---

# Probing

A linear probe is a `linear` featurizer: `h @ W + b`, one score per class, its classes the distinct values of a label column, sorted. A [fit](../usage/fit.md) trains it with `cross_entropy` against that column. Every metric scores its classes as it scores logits against a column of tokens ([FORMAT.md](../../FORMAT.md#ops)). Its fields are in [FORMAT.md](../../FORMAT.md#linear).

## A probe per layer

No corpus document trains a probe; this one is complete. It reads which day a question starts from (`entity`, seven classes) out of each layer's output at the day's last token, trains on the 30 training rows and scores the 19 test rows:

```json
{
  "description": "A linear probe per layer for the day a question starts from, at the day's last token.",
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
        "read": {"kind": "forward", "input": "train.input", "interventions": {
          "scores": {"site": "stream", "pos": {"scope": {"column": "train.entity"}, "index": -1},
                     "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "train.entity"},
        "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "train.entity"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    }
  },
  "saves": ["fit.held_out.acc", "fit.train.acc", "fit.history.ce", "fit.probe"]
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
- **`pos`** finds the row's `entity` in its prompt and takes its last token ([positions.md](../usage/positions.md#words-and-occurrences)).
- **`ce` is the objective; `acc` is reported**, per epoch too, as `fit.history.acc` ([fit.md](../usage/fit.md#history)).
- **The `held_out` eval** reads `test` wherever the body says `train`: the input, the scope's column and the target. The class list stays the training column's.

## Scoring classes

`argmax` and `top_k` rank `vocab_size` by default, so over a probe's scores they need `"over": "classes"`. `top_k` then gives `probs`, a softmax over the classes, and its `ids` index the sorted class list.

Classes compare exactly, with no `strip` or `lower`: a column holding `yes` and `Yes` makes two classes. Two refusals guard the class list. Every value of the `classes` column is one type (all strings, all ints, all floats or all booleans):

```text
clab: featurizers.probe.classes: classes column 'train.entity' holds str (row 0 is 'a'), int (row 1 is 1); a probe's classes are one type, all strings, all ints, all floats or all booleans: write them in one
```

And a row scored against a value that is not a class is refused before any forward. Here a probe trained on the Thursday and Friday rows alone meets the test split:

```text
clab: fit.held_out.ce at layers=0: target is 'entity' of 'test', whose row 0 is 'Monday', not one of the classes of probe 'probe': ['Friday', 'Thursday']
```

## The floor

At `epochs: 0` the probe is untrained, and its accuracy is the share of rows whose label sorts first ([fit.md](../usage/fit.md#epochs-0-is-a-null-control) says why). For this document that is 1 of 19 held-out rows (0.053) and 6 of 30 training rows (0.2). A probe is decoding something only where it clears its floor.

A probe read at a position that does not hold the variable scores its floor. Read at the prompt's last token (`"pos": -1`) instead of at the day, the same fit ends at exactly the floor on both layers of this model.

## Many probes in one fit

A sweep over layers runs a whole fit per point, and its forwards ([sweeps.md](../usage/sweeps.md)). Many probes can share one fit instead, one forward per minibatch serving all of them:

- **one `linear` featurizer per probe**, a site per layer;
- **one entry per probe** in the body's one forward;
- **one `cross_entropy` term per probe** in the objective.

A probe's parameters appear in its own term alone, so its gradient and Adam's step are those of the probe trained by itself. Two probes, at the day's last token in layers 0 and 1:

```json
{
  "description": "Two probes for the day a question starts from, at the day's last token in layers 0 and 1, trained side by side in one fit.",
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
    "day1": {"kind": "linear", "classes": "train.entity"}
  },
  "steps": {
    "fit": {
      "kind": "fit",
      "train": ["day0", "day1"],
      "objective": {"ce_day0": 1.0, "ce_day1": 1.0},
      "epochs": 50,
      "batch_size": 10,
      "seed": 0,
      "optimizer": {"name": "adam", "lr": 0.1},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "read": {"kind": "forward", "input": "train.input", "interventions": {
          "day0": {"site": "l0", "pos": {"scope": {"column": "train.entity"}, "index": -1},
                   "ops": [{"op": "encode", "featurizer": "fit.day0"}]},
          "day1": {"site": "l1", "pos": {"scope": {"column": "train.entity"}, "index": -1},
                   "ops": [{"op": "encode", "featurizer": "fit.day1"}]}}},
        "ce_day0": {"kind": "op", "op": "cross_entropy", "of": "read.day0", "target": "train.entity"},
        "ce_day1": {"kind": "op", "op": "cross_entropy", "of": "read.day1", "target": "train.entity"},
        "hit_day0": {"kind": "op", "op": "match", "of": "read.day0", "expected": "train.entity"},
        "hit_day1": {"kind": "op", "op": "match", "of": "read.day1", "expected": "train.entity"},
        "acc_day0": {"kind": "op", "op": "mean", "of": "hit_day0"},
        "acc_day1": {"kind": "op", "op": "mean", "of": "hit_day1"}
      }
    }
  },
  "saves": ["fit.held_out.acc_day0", "fit.held_out.acc_day1", "fit.day0", "fit.day1"]
}
```

Each probe's history, held-out scores and saved tensors equal those of the same probe trained alone: `day0` and `day1` give the 0.737 and 0.789 of the swept document above. A document of many probes repeats a featurizer, an entry, a term and a `match` and `mean` per probe; build it in Python as a dict and pass it to `Plan(**fields)` ([running.md](../usage/running.md#from-python)). `early_stop` watches one body op and ends the fit for every probe at once, so leave it out here, or give each probe its own fit.

## Loading a trained probe

`fit.probe` saved is a bundle ([bundles.md](../usage/bundles.md)). The loading document declares `{"kind": "linear", "classes": "<dataset>.<column>", "file_path": ...}`, and its column's class list must be the header's. A probe trained on this document's layer 1 alone, loaded with `classes` from a dataset that holds only two of the days:

```text
clab: featurizers.probe.file_path: documents/artifacts/fit.probe.safetensors: header 'classes' is '["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]'; this document expects '["Friday", "Thursday"]'
```

Saved from the swept document, the file holds one probe per layer, which no featurizer loads; fit the layer you want unswept first.

## Use a probe's direction

A trained probe's tensors are values ([featurizers.md](../usage/featurizers.md#its-tensors-as-values)): `fit.probe.W`, `[hidden_size, classes]`, and `fit.probe.b`, `[classes]`, as the fit left them. A two-class probe's direction is `W[:, 1] - W[:, 0]`, the direction along which its second class's score grows against its first's. `documents/probe_direction.json` fits one on GPT-2 and projects that direction out of the stream it read, in one document:

```json
"positive": {"kind": "op", "op": "select", "of": "fit.probe.W", "select": {"classes": 1}},
"positive_column": {"kind": "op", "op": "sum", "of": "positive", "over": ["classes"]},
"negative": {"kind": "op", "op": "select", "of": "fit.probe.W", "select": {"classes": 0}},
"negative_column": {"kind": "op", "op": "sum", "of": "negative", "over": ["classes"]},
"direction": {"kind": "op", "op": "sub", "of": "positive_column", "with": "negative_column"},
"zero": {"kind": "op", "op": "mul", "of": "direction", "with": 0},
"projected": {"kind": "forward", "input": "reviews.input", "interventions": {
  "out": {"site": "stream", "pos": {"index": ":"}, "ops": [{"op": "project_out", "direction": "direction"}]},
  "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "reviews.answer", "b": "reviews.other"}]}}}
```

- **`select` keeps `classes` at size 1, labelled with the class it kept**, so the two columns pair with no other index under Ops' one rule. The `sum` over `classes` drops that axis, exactly, and the `sub` of the two `[hidden_size]` vectors is the direction ([ops.md](../usage/ops.md#transforms)).
- **The classes sort**, so `negative` is class 0 and `positive` class 1. Check the order in the bundle's header or the scores' table before naming a column.
- **The null is the direction scaled to zero.** `project_out` leaves every vector as it is along a zero direction, so `null`'s numbers equal the clean run's ([running.md](../usage/running.md#when-two-runs-give-the-same-numbers)).

On `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, float32 on the CPU, the probe reads the sentiment of twenty reviews off layer 10 at accuracy 0.85 after one epoch and 1.0 after fifty. `ld` is each row's logit for the word that matches its review less the other's, and its mean, `clean_margin`, is 1.340. With the probe's direction projected out of layer 10 at every position, `projected_margin` is 0.427; `null_margin` is the clean 1.340. The probe is trained and tested on the same rows, so this shows the mechanics and the size of one direction's share, not a held-out effect.

A probe over more than two classes has a span of directions, not one. Take the class differences the question needs, a `select` pair and a `sub` each, or train a `subspace` of the rank you want instead ([trained subspaces](trained-subspaces.md)).

## Pitfalls

- **A word of several tokens.** `"index": -1` of the scope is the word's last token, and a last token several classes share cannot separate them: on this tokenizer Tuesday, Wednesday and Thursday all end in `day`. At the word's first token (`"index": 0`) the same probe reaches 1.0 on both layers. Choose the position for the question, and check `pos_token` in `clab table`.
- **Decodable is not used.** A probe that reads a variable says the information is there, not that the model uses it there. A `linear` featurizer only encodes, but its span, an orthonormal basis of `W`'s columns written as a `subspace` bundle with that kind's header, loads as a `subspace` and can be intervened along ([bundles.md](../usage/bundles.md#file_path)). Test use with an interchange through a trained subspace ([trained-subspaces.md](trained-subspaces.md)).
- **The fixture model is random.** The accuracies above show the mechanics on the tiny Llama; they say nothing about where a trained model keeps the day.
