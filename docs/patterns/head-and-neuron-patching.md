---
title: Head and Neuron Patching
one_liner: "Which heads or neurons carry it? Patch one head or a few neurons at a time, head sets, and path patching."
tags: [patterns, heads]
related: [docs/patterns/activation-patching.md, docs/usage/sites.md, docs/usage/sweeps.md, docs/usage/interventions.md, docs/patterns/attention.md, docs/usage/engines.md, docs/concepts/values-and-layout.md]
sources: [clab/compile/interventions.py, clab/steps/batches.py, clab/format/address.py]
---

# Head and Neuron Patching

[Activation patching](activation-patching.md) of the residual stream moves everything a layer computed at a position. To ask which attention head or which MLP neurons carry it, patch a narrower site: a `select` keeps some indices of one axis of the component, and the write replaces those and leaves the rest of the layer alone. The documents are the canonical patching document with a different site.

## One head at a time

`documents/head_patching.json`:

```json
"target": {"component": "attention_premix", "layers": 0, "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}}
```

`attention_premix` is every head's result as the attention's output projection takes it, laid out `(rows, pos, num_heads, head_dim)`. A head's result is one index of `num_heads`, and the output projection is linear, so patching that index changes exactly that head's share of what attention adds to the stream. The read on the counterfactual prompt and the write on the base prompt use the same site, and the sweep makes one point per head, each with its own read and write. The counterfactual forward writes nothing, so its four points differ only in what they read and one forward serves them all; the patched forward runs once per head ([sweeps](../usage/sweeps.md#what-a-sweep-costs)):

```text
at               row  pos  pos_token  input                              cf_answer  base_answer  value
{"num_heads":0}  0    10   ▁is        If today is Thursday, tomorrow is   Sunday     Friday      -0.027985602617263794
...
{"num_heads":3}  3    8    ▁is        If today is Sunday, tomorrow is     Saturday   Monday      -0.19152364134788513
```

The projection's input sits at a module boundary, so both engines reach it, and `--engine hooks` gives the same numbers as nnterp. `attention_head_outputs` is the same tensor read inside the attention's forward. nnterp serves it under `attn_implementation: "eager"` only, and there it equals `attention_premix`. Switching a model from sdpa to eager moves the last bits of everything downstream, so compare runs under one attention implementation ([engines](../usage/engines.md)).

## A null per head

The [null control](../usage/interventions.md#the-null-control) writes back what was read at the same place, and its scores must equal the clean forward's. On a swept head, read the head's own value and write it back in one forward, both at the swept site. The read narrows like the write and zips with its sweep, and an earlier entry of the same forward is a valid operand ([interventions](../usage/interventions.md#operands)). Added to `documents/head_patching.json`:

```json
"clean": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
"null": {"kind": "forward", "input": "pairs.input", "interventions": {
  "v_self": {"site": "target", "pos": -1},
  "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "null.v_self"}]},
  "logits": {"site": "head", "pos": -1}}},
"null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"}
```

Every point's `null_ld` equals the clean forward's `logit_diff`. `clean` needs no read at the swept site, since the null forward reads its own head. A self-read in `clean` at the swept site would make `clean` a swept step, its four points served by one forward.

## Layers crossed with heads

Sweep `layers` on the patched site. Each sweep is labelled by its field (`layers`, `num_heads`), different labels cross, and the step gets a point per pair. The counterfactual may read the patched site itself: it writes nothing, so its eight points differ only in what they read and one forward serves them all. Or it reads every layer with the heads sweep alone, and each point of the patch takes the slice labelled with its layer, as any operand with a `layers` axis does ([values and layout](../concepts/values-and-layout.md#the-layers-axis)):

```json
"sites": {
  "target": {"component": "attention_premix", "layers": {"sweep": [0, 1]},
             "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}},
  "every_layer": {"component": "attention_premix", "layers": ":",
                  "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}}
}
```

The counterfactual reads `every_layer` and the patch writes `target`. The two `num_heads` sweeps share a label and zip, so the counterfactual holds one value per head, `[layers, rows, pos, num_heads, head_dim]`. Read at `target`, it would hold one per layer and head. Either way it is one forward, and the patched logits are equal.

The corpus Llama has 2 layers (`plan.run.sizes["num_layers"]`) of 4 heads, each `head_dim` 4. A sweep lists its values, so every head of a layer is `[0, 1, 2, 3]` here, and on another model as many as its config's `num_attention_heads` (preflight refuses a head past them).

- **Eight points**, keyed `layers=0,num_heads=0` to `layers=1,num_heads=3`. A `mean` step over `logit_diff` gives one number per point, which `clab table` lists with its `at`.
- **Rank the heads in Python.** No op compares across points ([ops](../usage/ops.md#what-is-not-an-op)), so sort `Plan.load(out).results["<mean step>"]`, a dict keyed `(layer, head)` ([sweeps](../usage/sweeps.md#reading-swept-results)).
- **`layers` on the patch is a sweep, not a selector** ([interventions](../usage/interventions.md#writes-across-layers)).
- **`documents/real/head_patching.json`** is the same grid on Llama-3.2-1B, three layers by 32 heads, 96 points. It reads its counterfactual at `every_layer`, one forward for its 32 points, and runs one patched forward per point. It scores the head inside the entry (`match` and `logit_diff` in the head's `ops`; [ops](../usage/ops.md#a-step-or-an-entry)) and averages each point in a step. It needs a GPU and a gated checkpoint.

## Head sets

A list keeps several heads, and one write patches them together:

```json
"select": {"num_heads": {"sweep": [[0, 1], [2, 3], [0, 1, 2, 3]]}}
```

Each point's coordinate is the list (`num_heads=[0,1]`). Heads that back each other up show an effect only together, so a set that moves the answer when no member does alone is a finding, not noise.

## Path patching

Patching head h measures everything h changes downstream. Path patching keeps one path: here h's effect on the logits that does not pass through another head. Every head's result at the position is held at its clean value, and h alone takes the counterfactual's. The MLPs and the residual stream recompute, so the path is h's write into the stream and whatever the MLPs after it make of that.

1. Read every head's `attention_premix` at the position on the clean prompt, `layers: ":"`.
2. In the patched forward, list a freeze entry that swaps that read back, then the entry that patches head h. Entries at one site compose in the order written ([interventions](../usage/interventions.md#the-order-ops-run-in)), so the freeze comes first and the patch overrides it at head h alone.

```json
{
  "description": "Path patching from each head to the logits: every head's result at the last token held at its clean value, then head h alone from the counterfactual; the MLPs recompute.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {
    "every_head": {"component": "attention_premix", "layers": ":"},
    "one_head": {"component": "attention_premix", "layers": {"sweep": [0, 1]},
                 "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}},
    "every_layer": {"component": "attention_premix", "layers": ":",
                    "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "freeze": {"freeze": {"site": "every_head", "pos": -1, "ops": [{"op": "swap", "with": "clean.every_head"}]}},
    "score": {"ld": {"site": "head", "pos": -1,
                     "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input",
              "interventions": [{"every_head": {"site": "every_head", "pos": -1}}, "score"]},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                       "interventions": {"v_cf": {"site": "every_layer", "pos": -1}}},
    "own": {"kind": "forward", "input": "pairs.input",
            "interventions": {"v": {"site": "every_layer", "pos": -1}}},
    "frozen": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze", "score"]},
    "path": {"kind": "forward", "input": "pairs.input", "interventions": [
      "freeze",
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}},
      "score"]},
    "path_null": {"kind": "forward", "input": "pairs.input", "interventions": [
      "freeze",
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "own.v"}]}},
      "score"]},
    "plain": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}},
      "score"]}
  },
  "saves": ["clean.ld", "frozen.ld", "path.ld", "path_null.ld", "plain.ld"]
}
```

The freeze is the same at every point, so it needs no sweep. It is one entry at `layers: ":"` whatever the model's depth, the patch's `layers` sweep pairs with it as it is, and `clean` runs once. `path`, `path_null` and `plain` each run one forward per point, layers times heads. The counterfactual is read as in [layers crossed with heads](#layers-crossed-with-heads). Three nulls hold on the corpus:

- `frozen.ld` equals `clean.ld`: the freeze alone writes back what was read.
- `path_null.ld` equals `clean.ld` at every point: the patch sourced from the clean prompt.
- At `layers=1`, the last layer, `path.ld` equals `plain.ld`: nothing after it is frozen. At layer 0 the two differ.

**Check that the patch alone moved.** The nulls still pass when the patch lands on the wrong head, since `frozen` and `path_null` write clean values. Read `every_head` again after the writes, `"after": {"site": "every_head", "pos": {"index": "-3:"}}`, in `path` after `freeze` and the patch and in `frozen` after `freeze`. Then subtract: `"moved": {"kind": "op", "op": "sub", "of": "path.after", "with": "frozen.after"}`. At every point `moved` is nonzero only at the patched layer, the last position and head h, and `path.after` there equals `counterfactual.v_cf`. Read a window of static width: a read at every position pairs with no other forward's value ([positions](../usage/positions.md#static-width-or-ragged)).

**To a later head's queries.** The path-patched forward reads the receiving head's query and a fresh forward on the clean prompt takes it in, so only the query moves. With the model block given `"attn_implementation": "eager"` ([attention](attention.md)), `one_head` at a layer before the receiver's, and `"receiver": {"component": "attention_queries", "layers": 1, "select": {"num_heads": [2]}}`:

```json
"path": {"kind": "forward", "input": "pairs.input", "interventions": [
  "freeze",
  {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
   "q": {"site": "receiver", "pos": -1}}]},
"received": {"kind": "forward", "input": "pairs.input", "interventions": [
  {"q": {"site": "receiver", "pos": -1, "ops": [{"op": "swap", "with": "path.q"}]}}, "score"]}
```

Its null sourced from the clean prompt equals the clean run as well.

## Select lists are positional

A narrowed axis keeps its indices in the order the list writes them, so `{"num_heads": [3, 1]}` holds head 3 at index 0. An operand must hold the write's heads in the write's order or preflight refuses it ([sites](../usage/sites.md#select) has the rule, [operand-does-not-fit](../errors/operand-does-not-fit.md#a-select-list-is-positional) the refusals). Declaring one site and naming it in both the read and the write, as the corpus does, makes the two agree by construction.

## Check where a head write landed

The write's record gives the token it landed on, its `positions` ([positions](../usage/positions.md#check-where-it-landed)). The layer and head it hit are the compiled entry's `site` at each point, in the manifest:

```python
from clab import Plan

plan = Plan.load("out/head_patching")
for point in plan.steps["patched"]:
    print(point.at, point.interventions["patch"].site)
```

```text
{'num_heads': 0} component='attention_premix' layers=0 select={'num_heads': 0}
...
{'num_heads': 3} component='attention_premix' layers=0 select={'num_heads': 3}
```

To see that only head h changed, read every head of the layer, `{"component": "attention_premix", "layers": 0}` at `pos: -1`, in `clean` and in `patched` listed after the write, which then sees the written value ([interventions](../usage/interventions.md#the-order-ops-run-in)). At each point the two differ at head h alone.

## Neurons

`documents/neuron_patching.json` narrows an MLP instead, four neurons at a time:

```json
"target": {"component": "mlp_activation", "layers": 0,
           "select": {"intermediate_size": {"sweep": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]}}}
```

Which MLP component is "the neuron" matters on a gated MLP (Llama's `down(act(gate x) * up x)`). `mlp_activation` is `act(gate x)`: patching it swaps the gate and keeps the base prompt's `up x`, so the neuron's output becomes the counterfactual's gate times the base's up projection. `mlp_neurons` is what the down projection sums, the product: patching it swaps the neuron's whole contribution. The two give different numbers on this document; choose by the question, and say which in the result. On a plain MLP, `mlp_activation` is `act(up x)`.

## Heads under grouped-query attention

`attention_keys` and `attention_values` are narrowed by `num_kv_heads` and read before they are repeated across query heads. On a model whose attention groups query heads, one key or value head serves several query heads, so patching it moves all of them; patch `attention_premix` for one query head's result.

## Pitfalls

- A head's result (`attention_premix`) and its pattern (`attention_probabilities`) answer different questions: what it writes versus where it looks ([attention](attention.md)).
