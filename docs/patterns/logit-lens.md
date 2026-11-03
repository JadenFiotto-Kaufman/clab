---
title: Logit Lens
one_liner: "The logit lens in one forward: readout over a layers selector, a kl check against the model's logits, kl between lenses aligned by layer label, and top_k to see what each layer would say."
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

`lens.logits` is laid out `layers, rows, pos, vocab_size`, and each metric of it keeps the `layers` axis: `p_answer` is the answer's probability at every layer, row and position. `clab table out/logit_lens_all p_answer top1` prints one line per layer and row, with the model's layer in the `layer` column. `documents/logit_lens.json` is the same lens as a sweep, a point per layer; its points differ only in what they read, so one forward serves both, and the numbers are equal to the bit ([sweeps.md](../usage/sweeps.md#widen-the-read-instead)).

`readout` takes a residual-stream component whole ([ops.md](../usage/ops.md#transforms) lists them). Another component, or a site that selects part of `hidden_size`, is refused inside an entry:

```text
clab: steps.f.interventions.x.ops[0]: readout projects a residual-stream value; mlp_norm_output is not one: read one of token_embeddings, layer_input, attention_output, layer_mid, mlp_output, layer_output
```

As a step, `{"kind": "op", "op": "readout", "of": "f.x"}` over a saved read gives the entry form's numbers to the bit on the CPU (on CUDA the two can differ in the last bits: the entry form projects the selected positions, a matmul of another shape, up to about 5e-5 in a logit on pythia-410m), and a read of another component is refused there the same way (`steps.r.of: 'f.x' is read at mlp_norm_output: ...`).

## Check the wiring, then trust the lens

At the last layer, `layer_output` through `readout` is what the model itself computes from it, so its lens must equal the model's `logits`: to the bit on the CPU, and on CUDA to the last bits of the selected-positions matmul (write both as `readout` steps for a bit-exact check there). A `kl` of every layer's lens against `logits` checks that in the same forward, and adds the standard summary of the lens: how far each layer's prediction is from the final one. This complete document is that check, plus what each layer would say:

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

On the corpus model `to_final` is exactly `0.0` at layer 1, the last, on every row, and positive at layer 0. A nonzero value there means the engine's `readout` is not the model's own last computation. `readout` is the one op each engine computes itself ([engines.md](../usage/engines.md#cross-checking-with-hooks)): on nnterp it is the model's own, so `to_final` is 0.0 at the last layer of the softcapped tiny Gemma-2 too; on hooks it is the final norm, the head and the config's softcap, and on a model that does more after its head it is not the model's: on the tiny Cohere, which scales its logits, the hooks lens at the last layer matches `lm_head_output` and not `logits`. Read `lm_head_output` beside `logits` to see which one the lens matches before reading any other layer.

`lens.final` has no `layers` axis, so `kl` compares every layer with the same final distribution ([values-and-layout.md](../concepts/values-and-layout.md#the-layers-axis)).

## What each layer would say: `top_k`

`top3` above is, per layer, row and position, the three most likely ids, their `probs` (a float32 softmax over the vocabulary) and, because it is a step, their decoded `tokens`. It is saved as `top3.json`; `clab table --csv OUT top3` prints each cell whole:

```text
layer,row,pos,pos_token,input,value
0,0,10,▁is,"If today is Thursday, tomorrow is","{""ids"":[4598,17972,18312],""probs"":[4.240043926984072e-05,4.216024171910249e-05,4.213841384625994e-05],""tokens"":[""Handler"",""▁Fed"",""▁Finland""]}"
```

At the last layer it lists the same ids as `top_k` on the model's `logits` (`documents/readout.json`). `top_k` inside the entry, after `readout`, gives ids and `probs` without tokens. Each token is the tokenizer's piece, spelled as `pos_token` is, so `▁Fed` above starts a word and `Handler` continues one ([ops.md](../usage/ops.md#transforms)).

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

- **The last few tokens.** `"pos": {"index": "-3:"}` on both `logits` and `final` lenses each row's last three tokens, the last included. `kl` pairs position with position, and `clab table OUT to_final` prints a line per layer, row and position, its `pos` and `pos_token` saying which token: 8, 9 and 10 (`▁tom`, `orrow`, `▁is`) on row 0, `to_final` 0.0 at layer 1 on each.
- **Every position.** `"pos": {"index": ":"}` lenses every prompt token. A one-token answer column is then scored at every position of its row, a ragged value NaN past each row's own tokens.
- **Keep only the score.** `"ops": [{"op": "readout"}, {"op": "token_prob", "token": "prompts.base_answer"}]` inside the entry gives the same numbers as the `token_prob` step, bit for bit on the CPU and to the last bits on CUDA, and holds no vocabulary-wide logits past each batch: `layers × rows × positions × vocab_size` adds up quickly on a real model.
- **Sublayers.** `attention_output` and `mlp_output` lens what one sublayer writes; `layer_mid` the stream between them.
- **One layer.** An int `layers` gives a lens without a `layers` axis.

## Pitfalls

- **An early layer's lens is not its information content.** The final norm and head were trained on the last layer's stream; a low probability at layer 2 can mean the answer is there in another basis. Compare layers with each other and with `to_final`, not against an absolute threshold.
