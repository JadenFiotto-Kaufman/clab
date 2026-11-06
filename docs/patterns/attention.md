---
title: Attention Experiments
one_liner: "Where does a head look? Patterns, previous-token and induction stripes, knocking out heads or keys, patching a pattern, queries after RoPE."
tags: [patterns, attention]
related: [docs/usage/sites.md, docs/usage/engines.md, docs/usage/interventions.md, docs/usage/positions.md, docs/usage/output.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/ablation.md, docs/concepts/positions.md, docs/concepts/values-and-layout.md]
sources: [clab/ops/edits.py, clab/compile/interventions.py, clab/steps/batches.py]
---

# Attention Experiments

The attention interior is the queries, keys and values, the scores entering the softmax, and the pattern leaving it. Patching or ablating a head's result ([head and neuron patching](head-and-neuron-patching.md)) asks what the head writes; these components ask where it looks and why. They are read inside the attention's forward, so the model block needs `"attn_implementation": "eager"`, and the hooks engine does not serve them ([sites](../usage/sites.md#attention-needs-eager)).

## Reading a pattern

`documents/attention_knockout.json` puts `"attn_implementation": "eager"` in its model block and reads the pattern at the last query position for every head of layer 0:

```json
"sites": {"pattern": {"component": "attention_probabilities", "layers": 0}},
"clean": {"kind": "forward", "input": "pairs.input", "interventions": {"pattern": {"site": "pattern", "pos": -1}}}
```

The value is `(rows, pos, num_heads, key)`, here `[4, 1, 4, 11]`. `key` is each row's own tokens in order, key 0 its BOS, as wide as the longest row and NaN past a shorter row's last token. A row's keys sum to 1 over the entries that are there ([the key axis](../concepts/values-and-layout.md#the-key-axis)). Eager and sdpa differ in the last bits, so compare runs under one implementation ([engines](../usage/engines.md#cross-checking-with-hooks)).

A previous-token or induction score reads the pattern at keys relative to each query: `select: {"key": {"offset": -1}}` ([sites](../usage/sites.md#select)).

### A key that moves by row

A `key` selector counted from the start picks the same keys on every row, and an offset follows the query; neither follows a word that sits at a different position on each row. Read the whole `key` axis, and add a marker entry scoped on the word: a read with no ops, which nothing names, so it costs nothing, while its record holds where it landed on each row (`positions`). That position is the word's key index into the saved pattern. Added to `clean` in `documents/attention_knockout.json`, beside its `pattern` entry:

```json
"entity": {"site": "pattern", "pos": {"scope": {"column": "pairs.entity"}, "index": -1}}
```

```python
from clab import Plan

plan = Plan.load("out/attention_knockout")
pattern, tokens = plan.results["clean.pattern"], plan.run.tokens["pairs.input"]
for row, (key,) in enumerate(plan.steps["clean"].interventions["entity"].positions["pairs"]):
    print(row, key, tokens[row][key], pattern[row, 0, :, key])    # each head's weight on the word
```

Row 0's entity ends at key 6 (`day`, of `▁Th urs day`), the other rows' at key 4 (`▁Friday`, `▁Saturday`, `▁Sunday`).

## Knocking out a head

The same document then zeroes one head's whole pattern at the last query, swept over heads:

```json
"one_head": {"component": "attention_probabilities", "layers": 0, "select": {"num_heads": {"sweep": [0, 1, 2, 3]}}},
"zero": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]}
```

A head mixes its values with its pattern, so a zero pattern makes it contribute nothing at that position. The logits are the same as those of zeroing the same head's `attention_premix` there.

## Knocking out keys

To stop the last query from attending to some keys and let the rest take up the weight, write through the scores or through the pattern. Through the scores, a large negative score leaves nothing for the softmax to give:

```json
"knock": {"site": {"component": "attention_scores", "layers": 0, "select": {"key": 0}}, "pos": -1,
          "ops": [{"op": "swap", "with": -1e9}]}
```

Through the pattern, zero the keys, then rescale each row back to the sum it had before the step's writes, in a later entry of the same step on the whole row:

```json
"knock": {"site": {"component": "attention_probabilities", "layers": 0, "select": {"key": 0}}, "pos": -1,
          "ops": [{"op": "swap", "with": 0.0}]},
"restore": {"site": {"component": "attention_probabilities", "layers": 0}, "pos": -1,
            "ops": [{"op": "renormalize", "norm": "sum"}]}
```

- **Both remove BOS for every head here**, and give the same pattern up to float32 rounding.
- **`renormalize` must say `"norm": "sum"`.** Its default rescales to the L2 norm, and the rows then sum to less than 1.
- **A `key` selector names the same token on every row only where the rows share it at that place**, as BOS or a common prefix. A write past a row's own keys changes nothing.
- **A write selects on one axis**, so a key knocked out through `select: {"key": 0}` is knocked out for every head. To knock it out for one head, restore the other heads with a second write at the same site, from a clean read of their scores at the same `pos`:

```json
"others": {"site": {"component": "attention_scores", "layers": 0, "select": {"num_heads": [1, 2, 3]}}, "pos": -1},
"keep_others": {"site": {"component": "attention_scores", "layers": 0, "select": {"num_heads": [1, 2, 3]}}, "pos": -1,
                "ops": [{"op": "swap", "with": "clean.others"}]}
```

With `others` read in `clean` and `keep_others` after `knock` in the knocked-out step, head 0's weight on BOS is 0 and heads 1 to 3 hold the clean pattern.

## Patching a pattern

`documents/attention_pattern_patching.json` makes one head attend at the last position the way it did on the counterfactual prompt and leaves its values alone, so it asks whether where the head looks carries the answer, apart from what it moves:

```json
"their_pattern": {"site": "one_head", "pos": -1},
"look_there": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.their_pattern"}]}
```

Key k of the swapped row is weight on the counterfactual's token k, written onto the base's token k, so both prompts of a row must tokenize to the same length. Preflight checks it row by row:

```text
patched.look_there: operand 'counterfactual.their_pattern' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there, so their key axes do not line up
```

The document therefore runs on `data/weekdays_aligned/data.json`, the pairs that tokenize alike. A real-model version is `documents/real/pattern_patching.json`, which needs a GPU and a gated checkpoint.

## Queries after RoPE

`documents/attention_query.json` swaps the counterfactual's query at the last token into the base prompt:

```json
"sites": {"q": {"component": "attention_queries", "layers": 0}},
"patch": {"site": "q", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}
```

`attention_queries` is laid out `(rows, pos, num_heads, qk_head_dim)` and taken after the projection, the head split and RoPE, so a query carries the position it was computed at. The evidence: at layer 0 every prompt here ends in the same token, and its last-token queries are identical across prompts of one length and rotated, with the same norm, across prompts of different lengths.

- **A query swapped between prompts of different lengths also moves the position its head looks from.** Nothing refuses it: the document runs on `weekdays/train`, whose rows differ in length. Patch between pairs that tokenize alike, or count the position change as part of the effect.
- **`attention_keys` are taken after RoPE as well**, and differ across such a pair the same way.
- **Keys and values are narrowed by `num_kv_heads`**, before they are repeated across query heads.
