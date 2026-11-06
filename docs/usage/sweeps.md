---
title: Sweeps
one_liner: "How do I run one document at many settings? What a sweep may stand for, labels that zip or cross, each point's cost, and reading swept results."
tags: [usage, sweeps, grid, labels]
related: [docs/concepts/names-and-sweeps.md, docs/usage/output.md, docs/usage/sites.md, docs/usage/generate.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/logit-lens.md]
sources: [clab/format/sweeps.py, clab/compile/__init__.py, clab/plan.py, clab/output/table.py]
---

# Sweeps

Write `{"sweep": [v1, v2, ...]}` where a value goes, and the document runs once at each value, all on one model load. The rules are in [FORMAT.md#sweeps](../../FORMAT.md#sweeps). How a step's points follow from the values it uses is in [concepts/names-and-sweeps.md](../concepts/names-and-sweeps.md).

## Sweep one value

`documents/pos_sweep.json` is `documents/patching.json` with the patched position swept:

```json
"patch": {"site": "target", "pos": {"sweep": [-1, -2, -3]}, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}
```

- **A swept write's step runs one forward per point.** `patched` has three points, each a whole forward over every row.
- **A step that references a swept value inherits its points.** `iia` and `logit_diff` reference `patched`, so they have three points too.
- **A step that uses nothing swept runs once and serves every point.** The counterfactual read is one value.
- **A write swept against an unswept read writes the read's one value at every point.** At `pos=-2`, the counterfactual's last-token value goes onto the base's second-to-last token.
- **To patch each position from the same position, sweep the read's `pos` too.** Both sweeps are labelled `pos`, so their points pair up ([activation-patching.md](../patterns/activation-patching.md#a-layer-by-position-map)).

`clab show` lists the axis and the points' shape:

```text
result      step        dims       axes  shape
iia         iia         rows, pos  pos   3 points, each [4,1]
```

A sweep can stand in these places ([FORMAT.md#sweeps](../../FORMAT.md#sweeps) has the full rule):

- a field FORMAT.md types as "or a sweep": a site's `layers` and `select`, an entry's `pos` (its `frame`, its `index` or a scope's `occurrence`), and a fit's `epochs`, `seed`, objective weights and `lr`;
- an op's arguments, inside an entry or on an op step (`add`'s `scale`, a steering dose; `scale`'s `factor`; `top_k`'s `k`);
- a featurizer's fields (`k`, `seed`);
- a generate step's options (`max_new_tokens`).

A sweep is refused on `kind`, `op`, `site`, `component`, `input`, `of`, `featurizer`, `train`, `interventions`, `saves` and `file_path`. It is refused on a featurizer's `over`, a scope's `column` and a fit's optimizer `name`, and anywhere under `model` or `data`. What to write instead:

- **Two datasets:** write the steps twice, one set per dataset.
- **Two checkpoints or revisions:** a run loads one model, so write one document per revision and join their tables.
- **Two scope columns:** write one entry per column.
- **Two optimizers:** write one fit per optimizer.

Each value of a sweep is a number, a string, a boolean or a list of them. To vary a `pos` object, sweep a scalar inside it: `{"frame": 0, "index": {"sweep": [-1, -2]}}`.

```text
clab: sites.target.component: a sweep cannot stand for 'component'
clab: steps.forced.interventions.logits.pos.scope.column: {'sweep': ['days.base_answer', 'days.cf_answer']} is not a column reference, …; a scope's column takes no sweep (its occurrence does)
clab: steps.patched.interventions.patch.pos: a sweep's values are numbers, strings, booleans or lists of them, each point keyed by its value; …
```

## Labels: zip or cross

A sweep's label names its axis. Sweeps that share a label pair their points (they *zip*). Sweeps with different labels make a grid (they *cross*). One sweep reached from two places is one axis. With no `label`, the label is the field's own key, so two `layers` sweeps zip by default.

One label is one axis wherever it recurs in a step: two ops of one entry, two fields of one op, two entries, or an entry and the step's options. Where any of its sweeps changes the run, each point runs its own forward and every value of the step carries the axis. So one entry may project a swept direction out and take the cosine with the same sweep after the edit:

```json
"abl": {"site": "every", "pos": {"index": ":"}, "ops": [
  {"op": "project_out", "direction": {"sweep": ["contrast", "random"], "label": "fill"}},
  {"op": "cosine", "with": {"sweep": ["contrast", "random"], "label": "fill"}}]}
```

Each point projects out its own direction, and the entry's cosine and every read beside it equal the same ops written as two entries ([running.md](running.md#when-two-runs-give-the-same-numbers)).

This document reads the counterfactual at layer `src` and writes it into layer `dst`, over every pair of layers:

```json
{
  "description": "Read the counterfactual residual at layer src and swap it into the base run at layer dst, for every (src, dst).",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {
    "src": {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "src"}},
    "dst": {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "dst"}}
  },
  "steps": {
    "counterfactual": {
      "kind": "forward",
      "input": "pairs.counterfactual_inputs[0]",
      "interventions": {"v_cf": {"site": "src", "pos": -1}}
    },
    "patched": {
      "kind": "forward",
      "input": "pairs.input",
      "interventions": {
        "patch": {"site": "dst", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
        "logits": {"site": {"component": "lm_head_output"}, "pos": -1}
      }
    },
    "logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"}
  },
  "saves": ["logit_diff"]
}
```

- Crossed labels make the product of their values: `logit_diff` has four points, keyed `src=0,dst=0` through `src=1,dst=1`.
- Points that differ only in what they read share one forward: the counterfactual's two points run as one, and `patched` runs four.
- The same sweeps without labels are both labelled `layers` and zip: two points, `layers=0` and `layers=1`, equal to the grid's diagonal ([running.md](running.md#when-two-runs-give-the-same-numbers)).

The same rule decides every pair of sweeps:

| sweeps | labels | points |
|---|---|---|
| two `layers` sweeps over `[0, 1]`, no `label` | `layers`, `layers` | 2, zipped: layer L with layer L |
| the same sweeps with `"label": "src"` and `"label": "dst"` | `src`, `dst` | 4, crossed |
| one swept site used by two steps, or one swept bundle pulled into two steps | one | each step has the same 2 points |
| one label in two ops of one entry, two fields of one op, two entries, or an entry and the step's options | one | 2, zipped; a forward per point where any of them changes the run |
| `max_new_tokens` and `min_new_tokens` both over `[2, 6]`, no `label` | two field names | 4, crossed, including `max_new_tokens=2,min_new_tokens=6` |
| the same with `"label": "n"` on both | `n` | 2, zipped: `n=2`, `n=6` |
| a `load` of a value the run that saved it swept over `layers`, `[0, 1]` | `layers`, which the load declares where the document does not ([cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks)) | 2, zipped with a `layers` sweep here over the same values |
| one label over different values | — | refused |

Two sweeps under one label over different values are refused. Give one of them its own label to cross them, or the same values to pair them:

```text
clab: sites.dst.layers: label 'layers' is also the sweep at sites.src.layers, over [0, 1]; one label is one axis, …
```

A label is letters, digits and underscores, because a coordinate is written `<label>=<value>`. Before you trust a grid, check the `axes` column of `clab show`: a grid that zipped has one axis where you expected two.

## What a sweep costs

- A forward or generation depends on the step's options and its writes alone ([FORMAT.md#sweeps](../../FORMAT.md#sweeps)).
- Points that differ only in what they read share one forward, which visits every site any of them reads.
- Each point that changes a write, an operand or an option runs its own forward. A label that a write holds is the forward's, though a read, or an op after the write's last edit, holds it too.
- Each point's `schedule` in the manifest names the forward that served it. `clab show` prints one cost row per forward, with the points it served (`all 4 points`).
- A value is made once per distinct point on the axes it depends on (its `axes`). A read that does not depend on a step's sweep is one value, not identical copies.

`documents/head_patching.json` sweeps `select: {"num_heads": ...}` on the `target` site. The counterfactual step only reads that site, so its four points share one forward. `patched` writes there, so it runs four: five forwards for four heads. `documents/generate_probe.json` sweeps the frame of a read, so its three points share one generation, and its ids, `patched`, are one `[4, 3]` tensor.

A large grid costs outside the forwards too. The manifest holds every point's records, positions and schedule, so its size grows with the points and the rows. Compile time grows faster than the number of points. Split a very large grid into documents by layer range.

## Widen the read instead

When a sweep only chooses which part of one forward you read, the points share one forward either way. One wider read then gives the same numbers as one value with an axis:

| swept read | one wider read | the wider value |
|---|---|---|
| `"layers": {"sweep": [0, 1]}` (`documents/logit_lens.json`) | `"layers": ":"` (`documents/logit_lens_all.json`) | the metric `p_answer`: `[2, 4, 1]`, with a `layers` axis, where each point is `[4, 1]` |
| `"pos": {"sweep": [-2, -1]}` and `select: {"num_heads": {"sweep": [0, 1, 2, 3]}}` on `attention_premix` | `"pos": {"index": "-2:"}` and no `select` | the read: `[4, 2, 4, 4]`, where each point is `[4, 1, 1, 4]` |
| `{"frame": {"sweep": [0, 1, 2]}, "index": -1}` at the head in a generation | `{"frame": "0:3"}` | the metric `p_answer`: `[4, 3]`, where each point is `[4, 1]` |

Each pair gives equal values, from the same one forward or generation ([running.md](running.md#when-two-runs-give-the-same-numbers)). An op sees one point at a time, so it can reduce or compare along an axis but never across points. The wider read lets the document combine settings: a `mean` over `layers` works on the `layers: ":"` lens and is refused on the swept one:

```text
clab: steps.avg at layers=0: a reduction cannot remove 'layers'; the value is ['rows', 'pos']
```

Sweep when each point is a different intervention: the layer, head, position or dose a *write* acts at. A read used as the write's operand may leave out an axis the write has, and then broadcasts over it. An axis the operand keeps must hold the write's indices ([FORMAT.md#ops](../../FORMAT.md#ops) states the operand rule). So in head patching an unnarrowed read of every head does not serve a one-head swap. Sweep the counterfactual read's `select` with the write's; its points still share one forward:

```text
clab: steps.patched.interventions.patch.ops[0].with at num_heads=0: 'counterfactual.v_cf' holds every num_heads, and this write's selection holds [0]; …
```

## Reading swept results

A swept save holds one value per point, keyed by the point's coordinate, which is its *value* on each axis. The coordinate takes a different form in each place:

| where | key | example |
|---|---|---|
| `plan.results[name]` in Python | the value for one axis; a tuple in axis order for several; a list value as a tuple | `-1`, `(0, 1)`, `(0, 1, 2, 3)` |
| output files, `clab show`, `plan.run.costs`, `plan.run.tokens` | text, `label=value` joined by commas | `pos=-1`, `src=0,dst=1`, `intermediate_size=[0,1,2,3]` |
| `clab table` and `plan.table`, column `at` | an object | `{"src":0,"dst":1}` |
| a compiled step's `at` in `manifest.json` | the *index* on each axis, meaningful only inside its plan | `{"pos": 2}` for `pos=-3` |

```python
from clab import Plan
plan = Plan.load("out/pos_sweep")
plan.results["iia"][-2]          # the [rows, pos] tensor at pos=-2
```

`plan.sweeps` lists the axes by label, each with its values: `{'num_heads': Axis(values=[0, 1, 2, 3])}` for `documents/head_patching.json`. A coordinate tuple follows the order of `plan.sweeps`, which is the order of the `axes` column in `clab show`. Axes follow the format's root keys (`data`, `sites`, `interventions`, `featurizers`, `steps`), then document order inside each key. The order of keys in your file does not matter: a featurizer's `seed` sweep comes before a step's `pos` sweep even when `featurizers` is written after `steps`. `clab table` groups rows by point, in that order.

**A swept `pos`'s coordinate is not the table's `pos` column.** `at` holds the swept value; `pos` holds the position the tabled value's read landed on. On `documents/pos_sweep.json`, `clab table out iia` prints row 0 with `at` `{"pos":-2}` beside `pos` 10: the patch's position is in `at`, the logits read's in `pos`. Give a write's `pos` sweep its own label (`"label": "patch_pos"`) to keep the two names apart.

Reading the output files is covered in [usage/output.md](output.md).

## Pitfalls

- **Values swept over different axes cross in a table.** `clab table out iia m`, with `iia` swept over `pos` and `m` over `layers`, prints a row per example at every pair of points. Values that share an axis join on it. An unswept value's row repeats beside every point of a swept one ([output.md](output.md#clab-table)).
