---
title: Activation Patching
one_liner: "Interchange interventions: swap one position or a window from a counterfactual run into a base run, score with match and logit_diff, map layer by position with sweeps, and corrupt and restore with noise."
tags: [patterns, patching]
related: [docs/usage/interventions.md, docs/usage/ops.md, docs/usage/positions.md, docs/usage/sweeps.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/ablation.md]
sources: [clab/ops/edits.py, clab/ops/metrics.py, clab/compile/interventions.py]
---

# Activation Patching

Activation patching asks whether a component, at a position, carries what decides the answer. Read its value on a counterfactual prompt, swap it into the base prompt at the same place, and score the base run: if it now gives the counterfactual's answer, that value was enough to move it. In clab that is two forwards and a swap: one read, one write, and the metrics after.

## The canonical document

`documents/patching.json`, over `data/weekdays/train.json` (each row a base prompt such as "If today is Thursday, tomorrow is", one counterfactual prompt with another day, and both answers):

```json
"counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                   "interventions": {"v_cf": {"site": "target", "pos": -1}}},
"patched": {"kind": "forward", "input": "pairs.input", "interventions": {
  "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
  "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}},
"iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
"logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"}
```

```
uv run clab run documents/patching.json --out out/patching
uv run clab table out/patching iia logit_diff
```

The choices it makes:

- **Site**: `target` is `layer_output` at layer 0, the residual stream leaving block 0. At a position it holds everything computed there so far, so patching it moves the whole state, not one component's share; [head and neuron patching](head-and-neuron-patching.md) narrows it.
- **Position**: `-1`, the last token, where the next token is predicted. Patching there asks whether the prediction is sensitive to this layer; patching the entity's token asks whether the information sits there ([below](#by-content-not-by-count)).
- **Metrics**: `match` against the counterfactual answer is 1 where the patched model's top token is that answer, so its mean is the interchange accuracy. `logit_diff` (counterfactual answer less base answer) is the graded version and moves before any top token flips.
- **Null**: add the self-swap of [interventions](../usage/interventions.md#the-null-control); its scores equal the clean run's to the bit, and the effect is the difference.

The weights of the tiny checkpoint are random, so every number this corpus prints shows the mechanism and nothing about weekdays.

## Several positions

`documents/multi_position_patch.json` reads and writes `-4`, `-3` and `-2` as three entries each; `documents/window_patch.json` does it as one window:

```json
"patch": {"site": "target", "pos": {"index": "-4:-1"}, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}
```

The two give the same logits to the bit: a window is its positions patched together. An effect carried jointly by several positions can show in the window and in no single position.

## By content, not by count

Counted positions pair the counterfactual's i-th with the base's i-th. When a pair's prompts tokenize to different lengths (" Thursday" is three tokens on this tokenizer, " Saturday" one), the same count can name different words, and the run refuses it before any forward:

```text
patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11, so position 0 pairs '▁is' with 'urs'; use a scope to name the token in each row
```

`documents/entity_patch.json` names the token by what it says, on each side its own entity:

```json
"v_cf": {"site": "target", "pos": {"scope": {"column": "pairs.counterfactual_inputs_variables[0].entity"}, "index": -1}},
"patch": {"site": "target", "pos": {"scope": {"column": "pairs.entity"}, "index": -1},
          "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}
```

`index: -1` takes the entity's last token, one position on every row whatever the entity's length, so read and write pair up. [positions](../usage/positions.md) has the scope recipes.

## A layer by position map

`documents/pos_sweep.json` sweeps the write's `pos` and keeps the read at `-1`, so its points write the counterfactual's last-token value at the base's last three positions. To patch position p from position p, sweep both: sweeps with one label (here both default to `pos`) pair up, and a `layers` sweep on the shared site crosses them ([sweeps](../usage/sweeps.md)):

```json
{
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays_aligned/data"}},
  "sites": {
    "target": {"component": "layer_output", "layers": {"sweep": [0, 1]}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                       "interventions": {"v_cf": {"site": "target", "pos": {"sweep": [-1, -2, -3, -4]}}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "target", "pos": {"sweep": [-1, -2, -3, -4]}, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "logit_diff": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "mean_ld": {"kind": "op", "op": "mean", "of": "logit_diff"}
  },
  "saves": ["mean_ld"]
}
```

`data/weekdays_aligned/data.json` holds the pairs whose prompts tokenize to the same length, so every count names the same word on both sides. Eight points, `clab table <out> mean_ld`:

```
at                     value
{"layers":0,"pos":-1}  -0.0029188443440943956
{"layers":0,"pos":-2}  -0.0026546120643615723
...
{"layers":1,"pos":-1}  0.0026958249509334564
{"layers":1,"pos":-2}  -0.002695823786780238
```

Read the corners before the middle. At the last layer, a patch at the last position reproduces the counterfactual run's logits, and a patch at any earlier position leaves the base run's logits bit-identical, since nothing after the last block carries one position to another. So a layer sweep at the last position always ends at full effect: it measures how late the answer can still be swapped in, not where it is computed. The earlier layers at the other positions are where a map localizes. `documents/real/layer_sweep.json` is this experiment on a real checkpoint (it needs a GPU and a gated model).

## Scoring more than the answer

`documents/faithfulness.json` scores how far the patch moved the whole next-token distribution against the clean run on the same prompts: `kl` (KL(patched ‖ clean), in nats), `js` (symmetric, at most log 2) and `token_prob` of the counterfactual answer. `documents/cosine_metric.json` scores the residual stream instead: it reads layers 0, 2 and 4 of tiny GPT-2 at the last two tokens in the clean and the patched forward, with a write at layer 1's last token, and `cosine` between them is 1 to within rounding wherever the write cannot reach (layer 0, and the second-to-last token) and drops below 1 at the last token after layer 1.

## Corrupt and restore

The causal-tracing form patches clean states into a corrupted run. Corrupt the entity's embeddings with noise in a bundle, so every step that pulls it in draws the same noise, then restore the clean residual stream at one layer:

```json
"interventions": {"corrupt": {"noise": {"site": "embed", "pos": {"scope": {"column": "prompts.entity"}, "index": ":"},
                                         "ops": [{"op": "noise", "scale": 0.1, "seed": 1}]}}},
"steps": {
  "clean": {"kind": "forward", "input": "prompts.input", "interventions": {
    "resid": {"site": {"component": "layer_output", "layers": ":"}, "pos": -1},
    "logits": {"site": "head", "pos": -1}}},
  "corrupted": {"kind": "forward", "input": "prompts.input", "interventions": ["corrupt", {"logits": {"site": "head", "pos": -1}}]},
  "restored": {"kind": "forward", "input": "prompts.input", "interventions": ["corrupt", {
    "restore": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1]}}, "pos": -1,
                "ops": [{"op": "swap", "with": "clean.resid"}]},
    "logits": {"site": "head", "pos": -1}}]}
}
```

with the sites `embed` (`token_embeddings`) and `head` (`lm_head_output`) declared. `noise` draws from its `seed`, the row and the token position alone, so `corrupted` and `restored` add identical noise (`documents/noise_write.json` is the op alone). `clean` reads every layer once; each sweep point of `restore` takes the slice labelled with its layer. Restoring the last layer at the last position gives back the clean logits to the bit, which is the positive control; the effect at a layer is where between corrupted and clean its score lands.

## Two experiments, one baseline

`documents/two_observes.json` declares patching and mean ablation as bundles at the root and runs each in its own step against one shared clean run, its metrics named apart (`patching_ld`, `ablation_ld`) so `clab table` joins them row by row. `documents/gpt2_reach.json` is the canonical document on tiny GPT-2 at layer 2: the component names are the same on every model family.

## Pitfalls

- A `layers` selector on the swap (`":"`) patches every layer in one forward; a per-layer result needs `"layers": {"sweep": [...]}`.
- A write's `pos` swept alone moves where the counterfactual's value lands, not where it was read.
- `match` on a model that rarely puts the counterfactual answer on top is 0 everywhere; read `logit_diff` or `token_prob` beside it.
