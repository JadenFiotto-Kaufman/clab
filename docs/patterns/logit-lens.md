---
title: Logit Lens
one_liner: "The logit lens in one forward: readout over a layers selector, kl between lenses aligned by layer label, and top_k to see what each layer would say."
tags: [patterns, lens, logit-lens, readout, layers, kl, top_k]
related: [docs/usage/sites.md, docs/usage/ops.md, docs/concepts/values-and-layout.md, docs/usage/sweeps.md, docs/usage/positions.md]
sources: [clab/ops/transforms.py, clab/format/address.py, clab/compile/interventions.py, clab/engines/nnterp.py]
---

# Logit Lens

The logit lens decodes a layer's residual stream through the model's own final norm and head, as if that layer were the last, to see where in depth a prediction forms. In clab it is the `readout` op on a residual-stream site whose `layers` is a selector: every layer in one forward, one value with a `layers` axis.

## The document

`documents/logit_lens_all.json`:

```json
"sites": {"resid": {"component": "layer_output", "layers": ":"}},
"steps": {
  "lens": {
    "kind": "forward",
    "input": "prompts.input",
    "interventions": {"logits": {"site": "resid", "pos": -1, "ops": [{"op": "readout"}]}}
  },
  "p_answer": {"kind": "op", "op": "token_prob", "of": "lens.logits", "token": "prompts.base_answer"},
  "top1": {"kind": "op", "op": "match", "of": "lens.logits", "expected": "prompts.base_answer"}
}
```

`lens.logits` is laid out `layers, rows, pos, vocab_size`, and each metric of it keeps the `layers` axis: `p_answer` is the answer's probability at every layer, row and position. `clab table out/logit_lens_all p_answer top1` prints one line per layer and row, with the model's layer in the `layer` column. `documents/logit_lens.json` is the same lens as a sweep, one forward per layer; the numbers are equal to the bit ([sweeps.md](../usage/sweeps.md#widen-the-read-instead)).

`readout` takes a residual-stream component whole: `token_embeddings`, `layer_input`, `layer_mid`, `attention_output`, `mlp_output` or `layer_output`. Another component, or a site that selects part of `hidden_size`, is refused inside an entry:

```text
clab: steps.f.interventions.x.ops[0]: readout projects a residual-stream value; mlp_norm_output is not one
```

## Check the wiring, then trust the lens

At the last layer, `layer_output` through `readout` is what the model itself computes from it, so its lens must equal the model's `logits`. A `kl` of every layer's lens against `logits` checks that in the same forward, and adds the standard summary of the lens: how far each layer's prediction is from the final one. This complete document is that check, plus what each layer would say:

```json
{
  "description": "The logit lens at the last token of every layer, checked against the model's own logits, with each layer's three most likely tokens.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"prompts": {"path": "weekdays/train"}},
  "sites": {
    "resid": {"component": "layer_output", "layers": ":"},
    "head": {"component": "logits"}
  },
  "steps": {
    "lens": {
      "kind": "forward",
      "input": "prompts.input",
      "interventions": {
        "logits": {"site": "resid", "pos": -1, "ops": [{"op": "readout"}]},
        "final": {"site": "head", "pos": -1}
      }
    },
    "p_answer": {"kind": "op", "op": "token_prob", "of": "lens.logits", "token": "prompts.base_answer"},
    "to_final": {"kind": "op", "op": "kl", "of": "lens.logits", "against": "lens.final"},
    "top3": {"kind": "op", "op": "top_k", "of": "lens.logits", "k": 3},
    "p_by_layer": {"kind": "op", "op": "mean", "of": "p_answer"}
  },
  "saves": ["p_answer", "to_final", "top3", "p_by_layer"]
}
```

On the corpus model `to_final` is exactly `0.0` at layer 1, the last, on every row, and positive at layer 0. A nonzero value there means the engine's `readout` is not the model's own last computation, as where a model applies a logit softcap after its head: read `lm_head_output` beside `logits` to see which one the lens matches before reading any other layer. On the nnterp engine `readout` (`clab/engines/nnterp.py` `NnterpEngine.readout`) is nnterp's `project_on_vocab`: the final norm, the head, and what the model does after the head.

`lens.final` has no `layers` axis, so `kl` compares every layer with the same final distribution ([values-and-layout.md](../concepts/values-and-layout.md#the-layers-axis)).

## What each layer would say: `top_k`

`top3` above is, per layer, row and position, the three most likely ids, their `probs` (a float32 softmax over the vocabulary) and, because it is a step, their decoded `tokens`. It is saved as `top3.json`; `clab table --csv OUT top3` prints each cell whole:

```text
layer,row,pos,pos_token,input,value
0,0,10,▁is,"If today is Thursday, tomorrow is","{""ids"":[4598,17972,18312],""probs"":[4.240043926984072e-05,4.216024171910249e-05,4.213841384625994e-05],""tokens"":[""Handler"",""Fed"",""Finland""]}"
```

At the last layer it lists the same ids as `top_k` on the model's `logits` (`documents/readout.json`). `top_k` inside the entry, after `readout`, gives ids and `probs` without tokens.

## Comparing two lenses: `kl` by layer label

Two lenses compare layer by layer through their labels, whatever order each lists its layers in. A lens read on a patched run against the clean lens, as a fragment of `documents/patching.json` with a lens added to each forward:

```json
"clean": {"kind": "forward", "input": "pairs.input",
          "interventions": {"lens": {"site": "resid", "pos": -1, "ops": [{"op": "readout"}]}}},
"shift": {"kind": "op", "op": "kl", "of": "patched.lens", "against": "clean.lens"},
"shift_by_layer": {"kind": "op", "op": "mean", "of": "shift"}
```

With `patched.lens` read at `"layers": [1, 0]` and `clean.lens` at `":"`, `shift` pairs layer 1 with layer 1 and 0 with 0. An `against` lens that lacks one of the `of` lens's layers is refused by name. `shift_by_layer` is one number per layer, since `mean` keeps the `layers` axis unless `over` names it.

## Variations

- **Every position.** `"pos": {"index": ":"}` lenses every prompt token. A one-token answer column is then scored at every position of its row, a ragged value NaN past each row's own tokens.
- **Keep only the score.** `"ops": [{"op": "readout"}, {"op": "token_prob", "token": "prompts.base_answer"}]` inside the entry gives the same numbers, bit for bit, as the `token_prob` step, and holds no vocabulary-wide logits past each batch: `layers × rows × positions × vocab_size` adds up quickly on a real model.
- **Sublayers.** `attention_output` and `mlp_output` lens what one sublayer writes; `layer_mid` the stream between them.
- **One layer.** An int `layers` gives a lens without a `layers` axis.

## Pitfalls

- **An early layer's lens is not its information content.** The final norm and head were trained on the last layer's stream; a low probability at layer 2 can mean the answer is there in another basis. Compare layers with each other and with `to_final`, not against an absolute threshold.
- **`readout` as a step checks less.** `{"kind": "op", "op": "readout", "of": "f.x"}` over a saved read gives the entry form's numbers to the bit, but it runs on any value with a `hidden_size` axis, `mlp_norm_output` included, which then goes through the final norm a second time. Keep `readout` in the entry, where the site is checked.
