---
title: Sweeps
one_liner: "Running one document at many settings: what a sweep can stand for, labels that zip or cross, what each point costs, when one wider read replaces a sweep, and reading swept results by coordinate."
tags: [usage, sweeps, grid, labels]
related: [docs/concepts/names-and-sweeps.md, docs/usage/output.md, docs/usage/sites.md, docs/usage/generate.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/logit-lens.md]
sources: [clab/format/sweeps.py, clab/compile/__init__.py, clab/plan.py, clab/output/table.py]
---

# Sweeps

Write `{"sweep": [v1, v2, ...]}` where a value goes, and the document runs once at each value, all on one model load. The rules are in [FORMAT.md#sweeps](../../FORMAT.md#sweeps). Why steps multiply the way they do, and how a step finds the right point of the value it uses, is in [concepts/names-and-sweeps.md](../concepts/names-and-sweeps.md).

## Sweep one value

`documents/pos_sweep.json` is `documents/patching.json` with the patched position swept:

```json
"patch": {"site": "target", "pos": {"sweep": [-1, -2, -3]}, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}
```

`patched` becomes three points, and each one is a whole forward over every row. `iia` and `logit_diff` reference `patched`, so they become three points too. The counterfactual step uses no swept value, so it runs once, and every point writes the counterfactual's last-token value: at `pos=-2` onto the base's second-to-last token. To patch each position from the same position, give the read's `pos` the same sweep; both are labelled `pos`, so their points pair up ([activation-patching.md](../patterns/activation-patching.md#a-layer-by-position-map)). `clab show` lists the axis and one shape per point:

```text
result      step        dims       axes  shape
iia         iia         rows, pos  pos   {"pos=-1":[4,1],"pos=-2":[4,1],"pos=-3":[4,1]}
```

A sweep can stand in these places:

- any field that FORMAT.md types as "or a sweep", for example a site's `layers` and `select`, an entry's `pos` or its `frame` and `index`, and a fit's `epochs`, `seed`, objective weights and `lr`;
- an op's arguments, inside an entry or on an op step (`scale`, `factor`, `top_k`'s `k`);
- a featurizer's fields (`k`, `seed`);
- a generate step's options (`max_new_tokens`).

A sweep cannot stand for a name (`site`, `input`, `of`, `featurizer`), a `component`, a `kind`, a `file_path`, or anything under `model` or `data`. To run the same steps on two datasets, write the steps twice. Each value of a sweep is a number, a string, a boolean or a list of them. To vary an object, sweep a field inside it:

```text
clab: sites.target.component: a sweep cannot stand for 'component'
clab: steps.patched.interventions.patch.pos.sweep: a sweep's values are numbers, strings, booleans or lists of them, each point keyed by its value; {'frame': 0, 'index': -1} is an object: sweep a scalar field inside it, {'frame': {'sweep': [0, ...]}}
```

## Labels: zip or cross

A sweep's label names its axis. Sweeps that share a label pair their points (they *zip*), and sweeps with different labels make a grid (they *cross*). With no `label`, the label is the field's own key, so two `layers` sweeps zip by default. This document reads the counterfactual at layer `src` and writes it into layer `dst`, over every pair of layers:

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

`logit_diff` has four points, keyed `src=0,dst=0` through `src=1,dst=1`. The counterfactual runs twice, once per `src`. With both labels removed, both sweeps are labelled `layers` and the run has two points, `layers=0` and `layers=1`, which give the grid's diagonal bit for bit. The same rule decides every pair of sweeps:

| sweeps | labels | points |
|---|---|---|
| two `layers` sweeps over `[0, 1]`, no `label` | `layers`, `layers` | 2, zipped: layer L with layer L |
| the same sweeps with `"label": "src"` and `"label": "dst"` | `src`, `dst` | 4, crossed |
| one swept site used by two steps, or one swept bundle pulled into two steps | one | each step has the same 2 points |
| `max_new_tokens` and `min_new_tokens` both over `[2, 6]`, no `label` | two field names | 4, crossed, including `max_new_tokens=2,min_new_tokens=6` |
| the same with `"label": "n"` on both | `n` | 2, zipped: `n=2`, `n=6` |
| one label over different values | — | refused |

```text
clab: sites.dst.layers: label 'layers' is also the sweep at sites.src.layers, over [0, 1]; one label is one axis, whose sweeps pair their points: give one of them its own label to cross them ({'sweep': [...], 'label': '<name>'}), or the same values to pair them
```

A label is letters, digits and underscores, because a coordinate is written `<label>=<value>`. Before you trust a grid, check the `axes` column of `clab show`: a grid that zipped has one axis where you expected two.

## What a sweep costs

Each point is a whole step over every row. A step that uses anything swept runs its forward or generation once per point, even when the only swept thing in it is a read:

- `documents/head_patching.json` sweeps `select: {"num_heads": ...}` on the `target` site, and the counterfactual step reads that site. The counterfactual therefore runs once per head along with the patch: eight forwards for four heads.
- `documents/generate_probe.json` sweeps the frame of a read, so the whole three-token generation runs three times.

A step that uses nothing swept runs once, and its value serves every point. A read that serves a write swept over layers and heads need not be swept over both: read it at every layer and sweep the heads alone ([head and neuron patching](../patterns/head-and-neuron-patching.md#layers-crossed-with-heads)).

A large grid costs outside the forwards too. The run record stores positions per point, entry and row, so on Qwen3-4B a grid of 2,300 points over 120 rows writes a manifest of about 32 MB; and compile time grows faster than the number of points, 12,742 points taking about 40 s to compile where 2,182 take 3 s. Split a very large grid into documents by layer range.

## Widen the read instead

When a sweep only chooses which part of one forward you read, read all of it once. You get the same numbers from fewer forwards, as one value with an axis:

| swept read | one wider read | forwards |
|---|---|---|
| `"layers": {"sweep": [0, 1]}` (`documents/logit_lens.json`) | `"layers": ":"` (`documents/logit_lens_all.json`), a `layers` axis | 2 → 1 |
| `"pos": {"sweep": [-2, -1]}` and `select: {"num_heads": {"sweep": [0, 1, 2, 3]}}` on `attention_premix` | `"pos": {"index": "-2:"}` and no `select` | 8 → 1 |
| `{"frame": {"sweep": [0, 1, 2]}, "index": -1}` at the head in a generation | `{"frame": "0:3"}` | 3 generations → 1 |

Each pair gives equal values bit for bit. A wider read also lets the document combine settings. An op sees one point at a time, so it can reduce or compare along an axis but never across points. A `mean` over `layers` works on the `layers: ":"` lens and is refused on the swept one:

```text
clab: steps.avg: a reduction cannot remove 'layers'; the value is ['rows', 'pos']
```

Sweep when each point is a different intervention: the layer, head, position or dose a *write* acts at. A read used as the write's operand must be narrowed the same way as the write. An unnarrowed read of every head does not serve a one-head swap, so in head patching the counterfactual read is swept along with the write:

```text
clab: patched.patch: operand 'counterfactual.v_cf' is 1 token(s) wide, narrowed to {}; this write's selection is 1 wide, narrowed to {'num_heads': 1}
```

## Reading swept results

A swept save holds one value per point, keyed by the point's coordinate, which is its *value* on each axis. The coordinate takes a different form in each place:

| where | key | example |
|---|---|---|
| `plan.results[name]` in Python | the value for one axis; a tuple in axis order for several; a list value as a tuple | `-1`, `(0, 1)`, `(0, 1, 2, 3)` |
| output files, `clab show`, `plan.run.positions`, `plan.run.tokens` | text, `label=value` joined by commas | `pos=-1`, `src=0,dst=1`, `intermediate_size=[0,1,2,3]` |
| `clab table` and `plan.table`, column `at` | an object | `{"src":0,"dst":1}` |
| a compiled step's `at` in `manifest.json` | the *index* on each axis | `{"pos": 2}` for `pos=-3` |

```python
from clab import Plan
plan = Plan.load("out/pos_sweep")
plan.results["iia"][-2]          # the [rows, pos] tensor at pos=-2
```

The tuple's order is the order of the `axes` column in `clab show`, which is the order of `plan.sweeps`. Axes follow the format's root keys, then document order, and not the order keys appear in your file ([concepts/names-and-sweeps.md](../concepts/names-and-sweeps.md#a-sweep-label-is-an-axis)). `clab table` groups rows by point, in that order. Reading the output files is covered in [usage/output.md](output.md).

## Pitfalls

- **Default labels zip.** Sweeping `layers` on a read and on a write with the same values gives the diagonal, not the grid. Give each sweep its own `label`.
- **Values swept over different axes do not join in a table.** `clab table out iia hit`, where `iia` is swept over `src, dst` and `hit` over `src` alone, prints no rows. A swept value does join an unswept one: the unswept row repeats beside every point. Table values with different axes one at a time.
- **A swept read re-runs its whole step.** See [What a sweep costs](#what-a-sweep-costs).
