---
title: Interventions
one_liner: "How do I read or write a value at a site? Entries, edits, the order they run in, operands, gradients, writes across layers, named bundles and the null control."
tags: [usage, interventions]
related: [docs/usage/ops.md, docs/usage/sites.md, docs/usage/positions.md, docs/usage/sweeps.md, docs/usage/featurizers.md, docs/patterns/activation-patching.md, docs/patterns/head-and-neuron-patching.md]
sources: [clab/compile/interventions.py, clab/compile/arguments.py, clab/steps/batches.py, clab/ops/edits.py]
---

# Interventions

An entry of a step's `interventions` names a site, the positions `pos` picks there, and the `ops` it runs on that selection. **An entry with an edit in its `ops` writes; one without only reads.** The edit writes its result back into the forward, and the model runs on from there. The fields and grammar are in [FORMAT.md#an-entry-of-interventions](../../FORMAT.md#an-entry-of-interventions).

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

The model's logits equal those of `[add]` alone, and `x` is one probability per layer, row and position. An edit after an op that changed the layout (a transform that changes it, a metric or a reduction) has no live value to write. Compile refuses it: `'add' is an edit, which writes back, and it follows 'readout', which turned the value into ['rows', 'pos', 'vocab_size']: there is no live value to write`. Put the edit first, or the transform in a later entry at this site.

Within one entry:

- **The elementwise transforms and the comparisons write nothing** ([ops](ops.md#transforms)). In an entry with no edit they change only what it binds: `{"ops": [{"op": "abs"}]}` reads the absolute value and leaves the forward as it was.
- **Before a write's last edit they act on the live value that edit writes back.** `[{"op": "sign"}, {"op": "scale", "factor": 0.1}]` writes a tenth of each entry's sign. A read listed after it at the site sees exactly that.
- **A comparison read in another forward is a mask for a write.** Read `clean.positive` at the write's selection as `{"op": "gt", "with": 0}`. A write of `{"op": "scale", "factor": "clean.positive"}` then keeps the entries that were above zero and zeroes the rest.
- **An edit in a read is still an edit.** `add`, `sub`, `scale` and `normalize` write back what they compute, whatever ops follow them. `[{"op": "sub", "with": "clean.h"}, {"op": "div", "with": "clean.h"}]` writes the difference back before `div` reads it. A relative effect is `sub` then `div` as steps after the forward; a product in a read is `mul`, which writes nothing.
- **The reductions change the layout** (`norm`, `dot`, `count`, ...), so in a write they come after its last edit, on what it wrote.

Between entries:

- A step runs them in the order written: a list's items in turn, a bundle's entries where its name stands.
- Entries at different sites act as the forward reaches them; entries at one site compose in order.
- A read listed before a write at the same site sees the value before the write; a read listed after it sees the written value.
- **Of two writes at one site, the later wins where they overlap.** A freeze that should protect a patch is listed before it. Listed after it, the freeze puts the clean value back, every point equals the clean run, and the nulls still pass ([path patching](../patterns/head-and-neuron-patching.md#path-patching) is the recipe).

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

`renormalize` rescales to the value at the site before any of the step's writes, so it goes after the write it corrects, in the same entry or a later one of the same step; listed before it, it changes nothing. On an attention pattern use `"norm": "sum"` ([attention](../patterns/attention.md)). `project_out` divides its `direction` by its norm first, so the direction's length does not matter: one 1000 times longer gives the same values to float rounding. `decode` and `apply` are featurizer edits: [featurizers](featurizers.md). Every edit computes in float32 and rounds once to the value's dtype ([ops](ops.md#the-float32-rule)).

## Operands

An edit's operand (`with`, `direction`, `scale`, `factor`) is a number or a named value. A number broadcasts everywhere, and so does a named value with no axes (a `mean` of a metric). A named value may leave out any axis of the write's selection and broadcasts over it. Every axis it has is the selection's, in its order, with the same label ([FORMAT.md](../../FORMAT.md#ops) states the rule). So one vector of features broadcasts over rows and positions, and a metric's `[rows, pos]` value as `scale`'s `factor` scales each position's vector by its own number. The consequences:

- **A value with a rows axis belongs to its dataset, not to its file.** Two datasets that read the same `path` are two. Written on another dataset's rows, or met with one in an op step (a `kl` and its `against`), it is refused: `'other.input' has a row per example of 'other' and 'harvest.acts' of 'pairs'; an op takes one dataset's rows`.
- **Keep a counterfactual in its base rows' dataset.** Put both prompts in columns of one dataset, as `data/weekdays/train.json` holds `input` and `counterfactual_inputs`, and run the counterfactual forward on that column (`pairs.counterfactual_inputs[0]`). A `mean` step removes the rows, so a mean harvested on one dataset can be written into any other.
- **An operand from another forward needs a width every row shares.** An open `pos` (`{"index": ":"}`, an `until` scope, several occurrences) pairs with nothing in another forward, whatever the prompts. One occurrence of a scope in the prefill pairs where every row's scope holds as many tokens ([positions](positions.md#static-width-or-ragged)). Otherwise read it at a static index (`-1`, `{"index": "-6:"}`) or reduce it first.
- **Positions pair in order**: the operand's i-th with the write's i-th. Where the two prompts of a row differ in length, a position pairs across prompts of different lengths only where every token between it and the end it is counted from is the same on both, whatever the two tokens it pairs. Preflight refuses any other pair and names its tokens; locate the token in each prompt with a scope instead ([positions](positions.md#static-width-or-ragged)).
- **An operand from a narrowed site** holds the same heads or neurons in the same order ([head and neuron patching](../patterns/head-and-neuron-patching.md)).

**From the same forward.** An operand may name an earlier entry of its own step, at the same site or at one the forward reaches first. It is that entry's value in this batch at that moment, after any write listed before it:

```json
"interventions": {
  "src": {"site": {"component": "layer_output", "layers": 0}, "pos": -1},
  "dst": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
          "ops": [{"op": "swap", "with": "samefwd.src"}]}
}
```

In a step named `samefwd` this gives the same logits as reading `src` in a separate forward first, and saves that forward. Such an operand may be ragged, since each row pairs with itself: a self-swap at a whole-entity scope (`{"scope": {"column": "pairs.entity"}}`) on `weekdays/train` leaves the logits equal to the clean run's, where the same read from another forward is refused. What is refused:

- an operand naming a later entry (compile);
- an operand naming an entry with ops after its edits, since those run only once the forward is over (compile);
- an operand of the same forward given to an op after an entry's last edit, which runs once the value is off the model (compile); take the two in an op step after the forward;
- an operand read at a site the forward reaches later, such as layer 1 into layer 0 (preflight);
- inside a `generate` step, an operand naming an entry of the same step; read it in an earlier step.

An entry read only for such an operand is not kept. When nothing saves it, no other step names it and no `grad` entry differentiates it, the run hands each batch's read to the write that takes it and keeps none of it (`freed` in the forward's schedule). It then costs one batch's memory however many rows the forward runs. Save it, or name it in a later step, and its batches are joined like any read's.

## A gradient at a site

An entry whose first op is `grad` reads a gradient: of a later entry of its own forward, named bare in `of`, summed over every axis. It is taken with respect to the value at the entry's site, selected and positioned as the entry's `pos` and `select` say ([FORMAT.md#ops](../../FORMAT.md#ops)). It writes nothing: the ops after `grad`, edits too, act on the gradient once it is off the model, as the ops after a read's last edit do. The last forward of `documents/attribution_patching.json`:

```json
"h": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"},
                                          {"op": "scale", "factor": "difference"},
                                          {"op": "sum", "over": ["hidden_size"]}]},
"metric": {"site": {"component": "lm_head_output"}, "pos": -1,
           "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}
```

`patched.h` is each row's gradient of its own `metric` at layer 0's last token, times `difference`, summed over `hidden_size`. Rows do not mix in a forward, so the sum over rows leaves each row its own gradient.

- **What runs.** The forward runs with gradients on and takes one backward per distinct `of` once it has reached its last site. No gradient reaches the model's weights.
- **At one site the order is a read's.** A `grad` listed after a write differentiates the written value, and one listed before it the value the write acted on. A write of `scale 2` between two makes the first exactly twice the second.
- **Where it is refused.** An `of` read at a site the forward reaches no later than the entry's is refused at preflight, with the engine's ranks of both. `grad` is refused in a `generate`, in a fit's body and under `tp`. An `of` that runs an op no gradient passes through is refused; [FORMAT.md#ops](../../FORMAT.md#ops) lists those ops.
- **The null.** A `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended to its ops must be exactly zero: a second entry at the head, `{"site": "head", "pos": -1, "ops": [<the metric>, {"op": "mul", "with": 0}]}`, and a `grad` of it. `mul` writes nothing, so the null can stand anywhere in the list.
- **Not `scale` 0.** After a metric it is refused, since it is an edit. Alone at the head it is a write, which zeroes the logits every entry listed after it reads, so the real `grad` is zero too.

The technique, and the check that the gradient is right, are [attribution patching](../patterns/attribution-patching.md).

## Writes across layers

A site's `layers` as a slice or a list (`":"`, `"2:6"`, `[0, 3]`) makes the entry at each layer it picks, in one forward, each on that layer's live value. It is exactly the entries of one layer each it replaces: `documents/direction_ablation.json`'s one entry at `layers: ":"` and five entries at layers 0 to 4 give equal logits.

- An operand with a `layers` axis is sliced by label at each layer; a reversed list of layers takes the same slices.
- An operand that lacks a layer the write is made at is refused before any forward.
- An operand without a `layers` axis is the same at every layer.
- What the entry binds has a `layers` axis first, and no op inside the entry can reduce over it: reduce in an op step ([ops](ops.md#a-step-or-an-entry)).

A `layers` selector is not a sweep. `layers: ":"` on a swap patches every layer at once. To patch one layer at a time, sweep it, `"layers": {"sweep": [0, 1]}`, and each point is its own forward ([sweeps](sweeps.md)).

## Named bundles

The root `interventions` holds named dicts of entries; a step pulls one in by name, alone or in a list with other bundles and dicts, in order:

```json
"steered_renormed": {"kind": "forward", "input": "pairs.input",
                     "interventions": ["steer", "restore", {"logits": {"site": "head", "pos": -1}}]}
```

That is from `documents/steer_renormalize.json`; `documents/two_observes.json` runs two experiments from two bundles against one baseline. A bundle is a template: its entries bind names in the step that pulls it in.

- An operand that names one of the bundle's own entries bare (`"with": "src"`) means that entry in whichever step pulls it. Naming it as `<step>.src` is refused, and so are two entries of one name in one step.
- An operand that names another step's value (`"with": "counterfactual.v_cf"`) is looked up by that step name wherever the bundle is pulled in. `documents/two_observes.json`'s `patching` bundle needs a step named `counterfactual` before the step that pulls it in, or compile refuses it: `'counterfactual.v_cf' names no value produced before this step`.

## The null control

Write back what was read, on the same input, at the same place. Nothing else moves the numbers, so the null's scores equal the clean run's, in float32 and in bfloat16 ([running](running.md#when-two-runs-give-the-same-numbers)). The patched score minus the null's is the effect:

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

If `null_ld` differs from `clean_ld`, the read and the write are not the same selection of the same input: fix the document before reading any effect. The other edits have nulls of the same kind, each equal to the clean run: `add` at `"scale": 0.0` (the first point of a dose sweep), `scale` at `"factor": 1.0`.

## Pitfalls

- A write's `pos` swept alone moves where the counterfactual's value lands, not where it was read; sweep the read's `pos` with it ([activation patching](../patterns/activation-patching.md#a-layer-by-position-map)).
- A write selects on one axis: `{"num_heads": [1], "head_dim": [0, 1]}` on a write is refused (`an entry that writes selects on one axis`).
