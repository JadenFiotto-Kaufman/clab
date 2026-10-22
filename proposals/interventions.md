# Interventions: one list per step, and featurizers as ops

Ratified in principle 2026-10-22. This file fixes the spelling. Nothing is
built yet.

## 1. One list per step

`reads` and `writes` merge into the existing `interventions` key on
`forward` and `generate`. Its value is a name, a dict of entries, or a list
whose items are either: the reuse-or-inline pattern the key already has,
extended by one step. An entry is what a read or a write is today: a site,
a `pos`, and `ops`.

```
step.interventions := item | [item, ...]
item               := "<root bundle name>" | {<name>: entry, ...}
entry              := {"site": site, "pos"?: pos, "ops"?: [op, ...]}
root.interventions := {<bundle>: {<name>: entry, ...}}
```

A list is execution order, so the order between a reused bundle and the
step's own entries is written down, not implied.

Rules:

1. **In-place ops write back.** The edits group (`swap`, `add`, `sub`,
   `scale`, `noise`, `clamp`, `project_out`, `renormalize`) acts on the
   live value at the site. An entry containing one writes its result back
   to the model; an entry containing none does not touch the model. The
   op's group is the marker, nothing else.
2. **Every entry binds its name** to its value after its ops, writes
   included, so what a write wrote is readable as `<step>.<name>`.
3. **A value leaves the model only if used.** An entry's value is gathered
   when `saves` names it or a later step or entry consumes it. Writing costs
   no memory.
4. **An operand may name an earlier entry of the same step** (today's
   "a write may name a read of the same forward"); compile orders entries
   and preflight checks the order against the engine's site ranks, as now.
5. **Order is execution order**: items in the list, and entries within a
   dict, run in the order written; two entries at one site compose in
   that order (today's `steer` then `restore`).
6. **An edit after a transform is refused**: once `readout`, `mean`, a
   metric or `encode` has changed the layout, there is no live value to
   write. An edit must come before them, or, for `encode`, be bracketed by
   `decode` (§2).
7. The generation rules attach to entries with edits, unchanged: `frame`
   forms, `max_new_tokens` for open frames, online landing, `before`
   refused on a write beyond the prefill.

Root `interventions` keeps its meaning; a bundle's body is the same
`{<name>: entry}` dict, and a bare name in a step's list pulls it in.

## 2. Featurizers as ops

The `featurizer` and `select` arguments disappear from every edit op, and
the `featurizer` op with its `mode` goes. Three ops replace them:

| op | arguments | what |
|---|---|---|
| `encode` | `featurizer`, `select`? | the current value's coordinates in the featurizer's space; `select: {features: ...}` keeps a selector of them |
| `decode` | `featurizer` | undoes the most recent `encode` with that featurizer in this entry: the entry's value from before that `encode`, with the change in coordinates written back (the delta rule) |
| `apply` | `featurizer` | the featurizer's own version of the current value (a subspace's projection, an SAE's reconstruction without its error term, a LoRA's adapted value) |

Rules:

1. **`encode` and `decode` bracket.** Within an entry, each `decode` pairs
   with the most recent unmatched `encode` of the same featurizer. An
   unmatched `decode`, a `decode` of a different featurizer than the open
   one, or an entry that writes back with an `encode` still open, is
   refused at compile by name.
2. **Edits between the brackets act on coordinates.** `swap`, `add` and the
   rest do what they always do, on whatever the current value is. Their
   operands must have the current layout (a `source_z` read with the same
   `encode`s, or a number), checked at compile as now.
3. **Chains nest.** `encode a`, `encode b`, edit, `decode b`, `decode a`.
   `b`'s width is the width of the value it is applied to (`a`'s k), the
   same way a featurizer at a site takes the site's width today; a bundle
   records that width as it already does.
4. **A `select` on `encode` narrows the coordinates**; `decode` scatters the
   kept ones back and leaves the rest unchanged. This is today's
   `select: {features: [...]}`.
5. **The delta rule is what `decode` is.** An entry `[encode x, decode x]`
   is the identity to the bit; an edit of a coordinate that changes nothing
   changes nothing; an SAE's error term survives. `apply` is the only op
   that drops it.
6. **A read of coordinates is an entry that stops after `encode`.**
7. `fit` names, `fit.<featurizer>` and the trained-instance semantics are
   unchanged.

## 3. Before and after

### Patching (`documents/patching.json`)

Before:
```json
"counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
  "reads": {"v_cf": {"site": "target", "pos": -1}}},
"patched": {"kind": "forward", "input": "pairs.input",
  "writes": {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}},
  "reads": {"logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}
```
After:
```json
"counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
  "interventions": {"v_cf": {"site": "target", "pos": -1}}},
"patched": {"kind": "forward", "input": "pairs.input",
  "interventions": {
    "patch":  {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
    "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}
```
(a bare dict is one item, as a bare name is today)
```json
```
`patched.patch` is now also a value: the swapped activation, readable.

### Steering with renormalize (`documents/steer_renormalize.json`)

Before:
```json
"interventions": {
  "steer":   {"writes": {"steer":   {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "direction", "scale": 4.0}]}}},
  "restore": {"writes": {"restore": {"site": "target", "pos": -1, "ops": [{"op": "renormalize"}]}}}},
"steered_renormed": {"kind": "forward", "input": "pairs.input", "interventions": ["steer", "restore"],
  "reads": {"logits": {"site": "head", "pos": -1}, "resid": {"site": "target", "pos": -1}}}
```
After:
```json
"interventions": {
  "steer":   {"steer":   {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "direction", "scale": 4.0}]}},
  "restore": {"restore": {"site": "target", "pos": -1, "ops": [{"op": "renormalize"}]}}},
"steered_renormed": {"kind": "forward", "input": "pairs.input",
  "interventions": ["steer", "restore", {"logits": {"site": "head", "pos": -1}}]}
```
The separate `resid` read goes: `steered_renormed.restore` is the
renormalized activation, and `steered.steer` the steered one.

### DAS (`documents/das.json`)

Before:
```json
"cf_read": {"reads": {"v_cf": {"site": "target", "pos": -1,
  "ops": [{"op": "featurizer", "featurizer": "fit.rot", "mode": "encode"}]}}},
"das": {"writes": {"patch": {"site": "target", "pos": -1,
  "ops": [{"op": "swap", "with": "counterfactual.v_cf", "featurizer": "fit.rot"}]}},
  "reads": {"logits": {"site": "head", "pos": -1}}}
```
After:
```json
"cf_read": {"v_cf": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.rot"}]}},
"das": {
  "patch":  {"site": "target", "pos": -1, "ops": [
    {"op": "encode", "featurizer": "fit.rot"},
    {"op": "swap", "with": "counterfactual.v_cf"},
    {"op": "decode", "featurizer": "fit.rot"}]},
  "logits": {"site": "head", "pos": -1}}
```
MIB's "+PCA" (a gate over PCA coordinates), which was inexpressible:
```json
"patch": {"site": "target", "pos": -1, "ops": [
  {"op": "encode", "featurizer": "pc"}, {"op": "encode", "featurizer": "fit.mask"},
  {"op": "swap", "with": "counterfactual.v_cf"},
  {"op": "decode", "featurizer": "fit.mask"}, {"op": "decode", "featurizer": "pc"}]}
```

### SAE feature ablation (`documents/sae_feature_ablation.json`)

Before:
```json
"ablate": {"site": "target", "pos": -1, "ops": [
  {"op": "swap", "with": 0.0, "featurizer": "dictionary", "select": {"features": {"sweep": [0, 1, 2, 3]}}}]}
```
After:
```json
"ablate": {"site": "target", "pos": -1, "ops": [
  {"op": "encode", "featurizer": "dictionary", "select": {"features": {"sweep": [0, 1, 2, 3]}}},
  {"op": "swap", "with": 0.0},
  {"op": "decode", "featurizer": "dictionary"}]}
```
The splice (reconstruction without the error term) is `[{"op": "apply",
"featurizer": "dictionary"}]`, an edit, so it writes back.

### Generation (`documents/generated_answer.json`)

Before:
```json
"reads": {
  "at_said": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}},
  "at_last": {"site": "head", "pos": {"frame": -1}}}
```
After: the key is `interventions`; the entries are identical. A steering
write during decoding joins the same dict:
```json
"interventions": {
  "steer":   {"site": "target", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": "direction", "scale": 4.0}]},
  "at_said": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}},
  "at_last": {"site": "head", "pos": {"frame": -1}}}
```

## 4. Settled

1. **The step key.** `interventions` stays the one key; its list items are
   bundle names or inline entry dicts (ratified 2026-10-22).
2. **`apply` is an edit** (it writes back) but keeps the layout, so it can
   also end a read entry.
3. **Migration:** 54 documents, mechanical (`reads`/`writes` → one
   `interventions` item; `featurizer`/`select` args → `encode`/`decode`
   brackets; `featurizer` op → `encode`/`apply`). Compiled-plan snapshots
   change in spelling; every numeric snapshot stays byte-identical, since
   no numerics change.

## 5. What this removes

- the `reads`/`writes` split and the "a write's result is not readable" gap;
- the `featurizer` and `select` arguments on every edit op;
- the `featurizer` op and its `mode`;
- the "featurizers do not compose" gap (MIB's +PCA/+SAE rows).
