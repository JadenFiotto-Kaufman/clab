---
title: Writing a Document
one_liner: "The anatomy of a clab document: the root keys, a four-step patching experiment read top to bottom, how steps name each other's values, and what compile checks before anything runs."
tags: [usage, document]
related: [docs/usage/data.md, docs/usage/sites.md, docs/usage/positions.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/running.md, docs/usage/output.md, docs/concepts/plan-lifecycle.md, docs/concepts/names-and-sweeps.md]
sources: [clab/plan.py, clab/compile/__init__.py]
---

# Writing a Document

A clab document is one JSON object that declares a whole experiment: the checkpoint, the data, the steps that run in order, and the values to keep. `clab run` checks it, runs it on the model and writes the kept values to a directory. This page reads `documents/patching.json` from top to bottom, then covers how steps name each other's values and what is checked before the model loads.

## The root keys

| key | holds | page |
|---|---|---|
| `description` | prose saying what the experiment is | this page |
| `model` | the checkpoint: `key`, `revision` (a commit), `dtype` | [FORMAT.md](../../FORMAT.md#model), [engines.md](engines.md) |
| `data` | each dataset by name | [data.md](data.md) |
| `sites` | named locations, so several entries land on the same one | [sites.md](sites.md) |
| `interventions` | named bundles of entries that a step pulls in by name | [interventions.md](interventions.md) |
| `featurizers` | named spaces and functions that entries encode into or apply | [featurizers.md](featurizers.md) |
| `steps` | what runs, in the order written | below |
| `saves` | the values written to the output directory | below |

`model`, `data` and `steps` are required. Any other key at the root is refused, and the message lists the valid ones.

## The document, top to bottom

`documents/patching.json` reads the residual stream at the last token of a counterfactual prompt, swaps it into the same place in the base prompt, and scores what the model then predicts. [FORMAT.md](../../FORMAT.md#a-worked-document) prints it whole. Its head:

```json
"description": "The smallest real interchange: read the residual stream at the last token of the counterfactual prompt, swap it into the same place on the base prompt, and score what the model then says. ...",
"model": {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
          "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"},
"data": {"pairs": {"path": "weekdays/train"}},
"sites": {"target": {"component": "layer_output", "layers": 0}},
```

- `description` is for the reader. It is copied into the manifest and changes nothing in the run.
- `model` pins the weights: `revision` is a commit, so the document means the same model wherever it runs. `dtype` defaults to `float32`. Any other key, such as `"attn_implementation": "eager"`, goes to the engine's loader as written. Where the model runs is not the document's to say: that is `--device` at run time ([engines.md](engines.md)).
- `data.pairs` is the four rows of `data/weekdays/train.json`. Each row has an `input`, a list `counterfactual_inputs`, and the answers `base_answer` and `cf_answer`.
- `sites.target` names the residual stream after layer 0. The read and the write below both use the name, so they land on the same place.

The steps and the saves:

```json
"steps": {
  "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                     "interventions": {"v_cf": {"site": "target", "pos": -1}}},
  "patched": {"kind": "forward", "input": "pairs.input",
              "interventions": {
                "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
                "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}},
  "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
  "logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits",
                 "a": "pairs.cf_answer", "b": "pairs.base_answer"}
},
"saves": ["iia", "logit_diff"]
```

1. `counterfactual` is a `forward` over every row of `pairs.counterfactual_inputs[0]`, the first prompt of each row's list. Its one entry, `v_cf`, has no ops, so it only reads: the value at `target` at `pos: -1`, each row's last token. It binds `counterfactual.v_cf`, laid out `[rows, pos, hidden_size]`.
2. `patched` is a forward over `pairs.input`. The entry `patch` holds an edit (`swap`), so it writes: at `target`, last token, it replaces each row's value with that row of `counterfactual.v_cf`. The entry `logits` reads the head's output at the last token, with its site written inline. The head comes after layer 0 in the forward, so `logits` sees the patched run.
3. `iia` is an `op` step. `match` gives 1 for each row whose most likely next token is that row's `cf_answer`, else 0.
4. `logit_diff` is an op step: the logit of `cf_answer` minus the logit of `base_answer`.

`saves` brings `iia` and `logit_diff` home, one file each. Running it and reading it back:

```text
$ clab run documents/patching.json
patching on nnterp (batch size 8, device cpu) -> out/patching
  iia.safetensors              iia              [4, 1]
  logit_diff.safetensors       logit_diff       [4, 1]
  out/patching/manifest.json
$ clab table out/patching iia logit_diff
row  pos  pos_token  input                              cf_answer  iia  base_answer  logit_diff
0    10   ▁is        If today is Thursday, tomorrow is   Sunday    0.0   Friday      -0.027417995035648346
1    8    ▁is        If today is Friday, tomorrow is     Monday    0.0   Saturday    0.18085992336273193
...
```

The corpus checkpoint has random weights, so the numbers mean nothing; the wiring is the point. Row 0's last token sits at position 10 rather than 8 because ` Thursday` is three tokens on this tokenizer. Reading the table and the manifest is [output.md](output.md); running options are [running.md](running.md). Each entry's mechanics are [interventions.md](interventions.md), `pos` is [positions.md](positions.md), and the metrics are [ops.md](ops.md).

## How steps name each other's values

A reference is a thing, then a member, split at the first dot:

| written | means | example |
|---|---|---|
| `<step>.<entry>` | the value an entry of a `forward` or `generate` bound | `counterfactual.v_cf`, `patched.logits` |
| `<step>` | an `op` step's value, or a `generate` step's new token ids | `iia` |
| `<dataset>.<column>` | a column, read row by row | `pairs.cf_answer` |
| `<featurizer>` | a declared featurizer | `rot` |
| `<fit>.<member>` | a fit's trained featurizer, an eval's value, or its history | `fit.rot`, `fit.held_out.iia` (`documents/das.json`) |

- **Steps run in the order written, and a value is named only after the step that makes it.** An op step placed before the forward it reads is refused: `steps.iia.of: 'patched.logits' names no value produced before this step; known: counterfactual.v_cf`.
- **Names are one flat namespace per scope.** A step name has no dot and is not also a dataset or featurizer name. Entry names belong to their step, so two steps may each have a `logits`.
- **A write's value is readable too.** `patched.patch` is the value the write left at its selection, here each row of `counterfactual.v_cf`.
- **A `forward` step has no value of its own.** Save or use its entries; `"saves": ["patched"]` is refused.

The full rules, including a fit's own scope and how a swept name resolves, are in [FORMAT.md](../../FORMAT.md#names-and-scoping) and [names-and-sweeps.md](../concepts/names-and-sweeps.md).

## `saves`

Each name in `saves` is a step's value or a step's member: `iia`, `patched.logits`, `fit.rot`. Each becomes one file, and the manifest lists them in this order. A name saved twice, or one that names no value, is refused; the refusal lists every name that could be saved. With no `saves`, the run writes only the manifest.

Save what you will read. An entry's value leaves the model only when something saves or uses it, and saving `patched.logits` here keeps `[4, 1, 32000]` floats where the metric is `[4, 1]`.

## What is checked before the model loads

Most mistakes are refused before anything runs, each by the document path of what is wrong:

- **Reading the document** checks each record's shape: `steps.patched.interventions.logits: unknown field 'position'; valid: site, pos, ops`.
- **Compile** checks it against the data files and bundle headers: every reference, column, save and sweep. `steps.iia.expected: dataset 'pairs' has no column 'cf' on every row; columns: answer, base_answer, ...`

Both run without a model. To check a document without loading one:

```text
uv run python -c "from clab import Plan; Plan('my_experiment.json').compile()"
```

What needs the tokenizer or the model's sizes is checked after the model loads, before any forward: a position a row does not have (`counterfactual.v_cf: row 0: tokens 10 do not fit its 9 scope tokens`), a component the engine does not serve. A few things fail only when their step runs, such as a `generate` option the model's `generate` rejects. Any refusal leaves no output directory. Which stage refuses what is [plan-lifecycle.md](../concepts/plan-lifecycle.md); messages and their fixes are in [errors/index.md](../errors/index.md).

## Pitfalls

- **An entry with no ops reads; an entry with an edit writes.** There is no separate read or write key. A typo in an op name is refused, but a forgotten `ops` list silently turns an intended write into a read.
- **`description` is not checked against anything.** A document says what it does through its steps; keep the prose in step with them.
- **A layer past the model's last compiles.** Compile does not know the model, so `"layers": 5` on a two-layer model is refused only after the checkpoint loads (`counterfactual.v_cf: layer 5; the model has 2 layers`), which on a large model is the slow part. The same holds for positions, `select` indices and answer tokens.
