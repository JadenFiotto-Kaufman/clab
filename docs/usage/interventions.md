---
title: Interventions
one_liner: "Reading and writing at a site: entries, reads versus writes, the order ops run in, operands from earlier steps or the same forward, a gradient read with grad, writes across layers, named bundles of entries, and the null control every document should carry."
tags: [usage, interventions]
related: [docs/usage/ops.md, docs/usage/sites.md, docs/usage/positions.md, docs/usage/sweeps.md, docs/usage/featurizers.md, docs/patterns/activation-patching.md, docs/patterns/head-and-neuron-patching.md]
sources: [clab/compile/interventions.py, clab/compile/arguments.py, clab/steps/batches.py, clab/ops/edits.py]
---

# Interventions

An entry of a step's `interventions` names a site, the positions `pos` picks there, and the `ops` it runs on that selection. An entry with no edit among its ops reads; an entry with an edit writes its result back into the forward. The fields and grammar are in [FORMAT.md#an-entry-of-interventions](../../FORMAT.md#an-entry-of-interventions).

## Canonical example

The second forward of `documents/patching.json`:

```json
"patched": {
  "kind": "forward",
  "input": "pairs.input",
  "interventions": {
    "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
    "logits": {"site": {"component": "lm_head_output"}, "pos": -1}
  }
}
```

`patch` writes: `swap` replaces layer 0's residual stream at the last token with `counterfactual.v_cf`, which an earlier forward read on the counterfactual prompt. `logits` reads the head at the last token. Each entry binds `<step>.<name>` (`patched.patch`, `patched.logits`) for later steps and `saves`. A read leaves the model only when something names it; one nothing names costs nothing. A write binds what it wrote: `steered.steer` in `documents/steer_renormalize.json` is the steered activation.

## The order ops run in

Inside an entry, edits act on the live value as the forward reaches the site, and the value after the last edit is written back. Ops after the last edit run on that value once it is off the model: they shape what the entry binds, not what the model sees. So one entry can write and then score what it wrote:

```json
"x": {"site": {"component": "layer_output", "layers": ":"}, "pos": -1,
      "ops": [{"op": "add", "with": 0.5}, {"op": "readout"}, {"op": "token_prob", "token": "pairs.cf_answer"}]}
```

The model's logits are bit-identical to those of `[add]` alone, and `x` is one probability per layer, row and position. An edit after an op that changed the layout (a transform that changes it, a metric or a reduction) has no live value to write, and compile refuses it:

```text
steps.s.interventions.x.ops[1]: 'add' is an edit, which writes back, and it follows 'readout', which turned the value into ['rows', 'pos', 'vocab_size']: there is no live value to write; an edit comes before it, or put the transform in a later entry at this site
```

The elementwise transforms and the comparisons ([ops](ops.md#transforms)) keep the layout and are not edits, so they write nothing. In an entry with no edit they change only what it binds: `{"ops": [{"op": "abs"}]}` reads the absolute value and leaves the forward as it was. Before a write's last edit they act on the live value that edit writes back: `[{"op": "sign"}, {"op": "scale", "factor": 0.1}]` writes a tenth of each entry's sign, and a read listed after it at the site sees exactly that. A comparison read in another forward is a mask for a write: with `clean.positive` read at the write's selection as `{"op": "gt", "with": 0}`, a write of `{"op": "scale", "factor": "clean.positive"}` keeps the entries that were above zero, to the bit, and zeroes the rest. An edit in a read is still an edit: `[{"op": "sub", "with": "clean.h"}, {"op": "div", "with": "clean.h"}]` in an entry writes the difference back before `div` reads it, and `[{"op": "scale", "factor": "v"}, {"op": "sum", "over": ["hidden_size"]}]` writes the product, so a relative effect is `sub` then `div` as steps after the forward, and a product in a read is `mul`, which binds the same number and leaves the forward as it was. The reductions (`norm`, `dot`, `count`, ...) change the layout, so in a write they come after its last edit, on what it wrote.

Between entries, a step runs them in the order written (a list's items in turn, a bundle's entries where its name stands). Entries at different sites act as the forward reaches them; entries at one site compose in order. A read listed before a write at the same site sees the value before the write, a read listed after it sees the written value. Of two writes at one site, the later wins where they overlap: a freeze that should protect a patch is listed before it. Listed after it, the freeze puts the clean value back, every point equals the clean run, and the nulls still pass ([path patching](../patterns/head-and-neuron-patching.md#path-patching) is the recipe).

## Edit recipes

The signatures are in [FORMAT.md#edits](../../FORMAT.md#edits); this is which edit does which job.

| to | ops | in the corpus |
|---|---|---|
| put another run's value in (patching) | `{"op": "swap", "with": "<step>.<entry>"}` | `documents/patching.json` |
| put a constant in (zero ablation) | `{"op": "swap", "with": 0.0}` | `documents/zero_ablation.json` |
| put a mean in (mean ablation) | `{"op": "swap", "with": "mean"}`, `mean` an op step | `documents/mean_ablation.json` |
| add a direction at a dose (steering) | `{"op": "add", "with": "direction", "scale": 4.0}` | `documents/steer_renormalize.json` |
| subtract, or multiply | `{"op": "sub", "with": ...}`, `{"op": "scale", "factor": 0.5}` | |
| keep the entries where a comparison holds, zero the rest (a mask) | `{"op": "scale", "factor": "<a comparison's value>"}` | |
| add Gaussian noise | `{"op": "noise", "scale": 0.5, "seed": 7}` | `documents/noise_write.json` |
| bound every element | `{"op": "clamp", "min": -0.01, "max": 0.01}` | `documents/clamp_write.json` |
| remove a direction | `{"op": "project_out", "direction": "direction"}` | `documents/direction_ablation.json` |
| give each vector unit length | `{"op": "normalize"}` | |
| restore the norm the site had before this step's writes | `{"op": "renormalize"}` | `documents/steer_renormalize.json` |

`renormalize` rescales to the value at the site before any of the step's writes, so it goes after the write it corrects, in the same entry or a later one of the same step; listed before it, it changes nothing. On an attention pattern use `"norm": "sum"` ([attention](../patterns/attention.md)). `project_out` divides its `direction` by its norm first, so the direction's length does not matter: one 1000 times longer gives the same values to float rounding (5.6e-9 apart on the corpus). `decode` and `apply` are featurizer edits: [featurizers](featurizers.md). Every edit computes in float32 and rounds once to the value's dtype ([ops](ops.md#the-float32-rule)).

## Operands

An edit's operand (`with`, `direction`, `scale`, `factor`) is a number or a named value. A number broadcasts everywhere, and so does a named value with no axes (a `mean` of a metric). A named value may leave out any axis of the write's selection and broadcasts over it; every axis it has is the selection's, in its order, with the same label ([FORMAT.md](../../FORMAT.md#ops) states the rule). So one vector of features broadcasts over rows and positions, and a metric's `[rows, pos]` value as `scale`'s `factor` scales each position's vector by its own number. Three consequences:

- A value with a rows axis belongs to its dataset, not to its file: two datasets that read the same `path` are two. Written on another dataset's rows, or met with one in an op step (a `kl` and its `against`), it is refused: `'other.input' has a row per example of 'other' and 'harvest.acts' of 'pairs'; an op takes one dataset's rows`. A `mean` step removes the rows, so a mean harvested on one dataset can be written into any other.
- A read whose rows hold different numbers of positions (an `{"index": ":"}` over prompts of different lengths, a whole-entity scope) pairs with nothing in another forward; read it at a static index (`-1`, `{"index": "-6:"}`) or reduce it first.
- Positions pair in order: the operand's i-th with the write's i-th. Where the two prompts of a row differ in length, preflight refuses a pair that may be two different tokens and names them; locate the token in each prompt with a scope instead ([positions](positions.md)). An operand from a narrowed site holds the same heads or neurons in the same order ([head and neuron patching](../patterns/head-and-neuron-patching.md)).

**From the same forward.** An operand may name an earlier entry of its own step, at the same site or at one the forward reaches first. It is that entry's value in this batch at that moment, after any write listed before it:

```json
"interventions": {
  "src": {"site": {"component": "layer_output", "layers": 0}, "pos": -1},
  "dst": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
          "ops": [{"op": "swap", "with": "samefwd.src"}]}
}
```

In a step named `samefwd` this gives the same logits, to the bit, as reading `src` in a separate forward first, and saves that forward. Such an operand may be ragged, since each row pairs with itself: a self-swap at a whole-entity scope (`{"scope": {"column": "pairs.entity"}}`, three tokens on one row of `weekdays/train` and one on the others) leaves the logits bit-identical to the clean run, where the same read from another forward is refused. Compile refuses an operand naming a later entry, or an entry with ops after its edits (those run only once the forward is over), and an operand of the same forward given to an op after an entry's last edit, which runs once the value is off the model (take the two in an op step after the forward); preflight refuses one read at a site the forward reaches later (layer 1 into layer 0). Inside a `generate` step an operand cannot name an entry of the same step: read it in an earlier step.

An entry read only for such an operand is not kept: when nothing saves it, no other step names it and no `grad` entry differentiates it, the run hands each batch's read to the write that takes it and keeps none of it (`freed` in the forward's schedule), so it costs one batch's memory however many rows the forward runs: a reconstruction error over 10,000 rows of Gemma-2-2B in bfloat16 (an SAE's `apply` at layer 12, then a `sub` of the stream read there, every token after BOS) peaks at 13.8 GiB, model included, where keeping the read would add 39 GB. Save it, or name it in a later step, and its batches are joined like any read's.

## A gradient at a site

An entry whose first op is `grad` reads a gradient: of a later entry of its own forward, named bare in `of`, summed over every axis, with respect to the value at the entry's site, selected and positioned as the entry's `pos` and `select` say ([FORMAT.md#transforms](../../FORMAT.md#transforms)). It writes nothing: the ops after `grad`, edits too, act on the gradient once it is off the model, as the ops after a read's last edit do. The last forward of `documents/attribution_patching.json`:

```json
"h": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"},
                                          {"op": "scale", "factor": "difference"},
                                          {"op": "sum", "over": ["hidden_size"]}]},
"metric": {"site": {"component": "lm_head_output"}, "pos": -1,
           "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}
```

`patched.h` is each row's gradient of its own `metric` (rows do not mix in a forward, so the sum over rows leaves each row its own) at layer 0's last token, times `difference`, summed over `hidden_size`. The forward runs with gradients on and takes one backward per distinct `of` once it has reached its last site, and no gradient reaches the model's weights. At one site the order is a read's: a `grad` listed after a write differentiates the written value, and one listed before it the value the write acted on, so a write of `scale 2` between two makes the first exactly twice the second. The `of` is read at a site the forward reaches after the entry's, or preflight refuses it with the engine's ranks of both; `grad` is refused in a `generate`, in a fit's body and under `tp`, and so is an `of` that runs an op no gradient passes through (`top_k`, `argmax`, `match`, `contains`, `detokenize`, `sign`, the comparisons `eq` to `le`, `count`). A null for it is an `of` whose ops end in `{"op": "scale", "factor": 0}`, whose gradient is exactly zero. The technique, and the check that the gradient is right, are [attribution patching](../patterns/attribution-patching.md).

## Writes across layers

A site's `layers` as a slice or a list (`":"`, `"2:6"`, `[0, 3]`) makes the entry at each layer it picks, in one forward, each on that layer's live value. It is exactly the entries of one layer each it replaces: `documents/direction_ablation.json`'s one entry at `layers: ":"` and five entries at layers 0 to 4 give identical logits. An operand with a `layers` axis is sliced by label at each layer (a reversed list of layers takes the same slices); one that lacks a layer the write is made at is refused before any forward; one without a `layers` axis is the same at every layer. What the entry binds has a `layers` axis first, and no op inside the entry can reduce over it: reduce in an op step ([ops](ops.md)).

A `layers` selector is not a sweep. `layers: ":"` on a swap patches every layer at once; to patch one layer at a time, sweep it, `"layers": {"sweep": [0, 1]}`, and each point is its own forward ([sweeps](sweeps.md)).

## Named bundles

The root `interventions` holds named dicts of entries; a step pulls one in by name, alone or in a list with other bundles and dicts, in order:

```json
"steered_renormed": {"kind": "forward", "input": "pairs.input",
                     "interventions": ["steer", "restore", {"logits": {"site": "head", "pos": -1}}]}
```

That is from `documents/steer_renormalize.json`; `documents/two_observes.json` runs two experiments from two bundles against one baseline. A bundle is a template: its entries bind names in the step that pulls it in, and an operand that names one of the bundle's own entries bare (`"with": "src"`) means that entry in whichever step pulls it. Naming it as `<step>.src` is refused, and so are two entries of one name in one step. An operand that names another step's value (`"with": "counterfactual.v_cf"`) is looked up by that step name wherever the bundle is pulled in: `documents/two_observes.json`'s `patching` bundle needs a step named `counterfactual` before the step that pulls it in, or compile refuses `interventions.patching.patch.ops[0].with in steps.patched: 'counterfactual.v_cf' names no value produced before this step; known: ...`.

## The null control

Write back what was read, on the same input, at the same place. Nothing else moves the numbers, so the null's scores equal the clean run's to the bit, in float32 and in bfloat16, and the patched score minus the null's is the effect:

```json
{
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {
    "target": {"component": "layer_output", "layers": 0},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"scored": {"logits": {"site": "head", "pos": -1}}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input",
              "interventions": ["scored", {"v_self": {"site": "target", "pos": -1}}]},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                       "interventions": {"v_cf": {"site": "target", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input",
             "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "clean.v_self"}]}}]},
    "patched": {"kind": "forward", "input": "pairs.input",
                "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}]},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "patched_ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "effect": {"kind": "op", "op": "sub", "of": "patched_ld", "with": "null_ld"}
  },
  "saves": ["clean_ld", "null_ld", "patched_ld", "effect"]
}
```

If `null_ld` differs from `clean_ld`, the read and the write are not the same selection of the same input: fix the document before reading any effect. The other edits have nulls of the same kind, each bit-exact: `add` at `"scale": 0.0` (the first point of a dose sweep), `scale` at `"factor": 1.0`.

## Pitfalls

- A write's `pos` swept alone moves where the counterfactual's value lands, not where it was read; sweep the read's `pos` with it ([activation patching](../patterns/activation-patching.md#a-layer-by-position-map)).
- `normalize` leaves a zero vector as it is, so a direction that is zero (two harvests that agree) steers nothing at any dose and the sweep looks flat; check a direction's norm before reading a dose curve ([steering](../patterns/steering.md)).
- A write selects on one axis: `{"num_heads": [1], "head_dim": [0, 1]}` on a write is refused.
