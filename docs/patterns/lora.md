---
title: LoRA
one_liner: "How do I train and apply a low-rank adapter? Through the frozen model, on in chosen steps, across sites."
tags: [patterns, lora, adapter, fine-tuning, with]
related: [docs/usage/fit.md, docs/usage/featurizers.md, docs/usage/bundles.md, docs/usage/interventions.md, docs/usage/generate.md, docs/usage/positions.md]
sources: [clab/featurizers/lora.py, clab/compile/uses.py, clab/steps/fit.py]
---

# LoRA

A `lora` featurizer is a low-rank adapter at one site. Its one op, `apply`, adds `(alpha/rank) (x A) B` to the live value, `x` being the value itself or, with `with`, a value read at another site of the same forward. A [fit](../usage/fit.md) trains `A` and `B` through the frozen model. Saved as a [bundle](../usage/bundles.md), the adapter is a named intervention that a step turns on by naming it. Its fields are in [FORMAT.md](../../FORMAT.md#lora).

## Train an adapter

`documents/lora_train.json` trains `{"kind": "lora", "rank": 4, "alpha": 8}` at layer 0's output to make the model say each training prompt's counterfactual answer:

```json
"adapted": {
  "kind": "forward",
  "input": "train.input",
  "interventions": {
    "adapt": {"site": "target", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "fit.lora"}]},
    "logits": {"site": "head", "pos": -1}
  }
},
"ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits", "target": "train.cf_answer"}
```

- **Every position, BOS included** (`{"index": ":"}`): an adapter stands for a change of the weights, which acts at every token.
- **It starts as the identity exactly.** `B` is drawn as zeros, so before the first optimizer step the adapted model is the model. At `epochs: 0` the held-out cross entropy equals a clean forward's, which makes it the null control ([fit.md](../usage/fit.md#epochs-0-is-a-null-control)).
- **The scale is `alpha / rank`**, 2 here. A loaded adapter takes `rank` and `alpha` from its bundle's header ([bundles.md](../usage/bundles.md#file_path)).

`fit.history.ce` is the training cross entropy after each of the 20 epochs, `fit.held_out.ce` and `fit.train.ce` the trained adapter on each split, and `fit.lora` the bundle.

## Apply it in some steps and not others

`documents/lora_apply.json` loads the bundle and declares the adapter once, as a root bundle of entries:

```json
"featurizers": {"lora": {"kind": "lora", "file_path": "documents/artifacts/fit.lora.safetensors"}},
"interventions": {
  "adapted": {
    "adapt": {"site": "target", "pos": {"frame": ":", "index": ":"}, "ops": [{"op": "apply", "featurizer": "lora"}]}
  }
},
"steps": {
  "clean": {"kind": "forward", "input": "test.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
  "adapted": {"kind": "forward", "input": "test.input", "interventions": ["adapted", {"logits": {"site": "head", "pos": -1}}]},
  "answer": {"kind": "generate", "input": "test.input", "interventions": "adapted", "max_new_tokens": 4, "do_sample": false}
}
```

A step that names `adapted` runs the adapted model; one that does not runs the model itself, so `clean` and `adapted` differ by the adapter alone. Run `lora_train.json` with `--out documents/artifacts` first.

**In a generation, write it at `"frame": ":"`.** A `pos` with no `frame` means the prefill only ([positions.md](../usage/positions.md)): the adapter changes the prompt's activations and their cache, and every decode call then runs unadapted. `":"` is the prefill and every decode call. On `lora_apply.json` the two give different generated tokens on both rows.

## Across sites: `with`

An adapter on a projection reads the projection's input and adds to its output. `with` names the input: an entry of the same forward, at a site the forward reaches first. No corpus document trains one; this one is complete. It adapts layer 0's attention output projection, reading `attention_premix` (every head's result, as the projection takes it) and adding at `attention_output`:

```json
{
  "description": "A rank-4 adapter on layer 0's attention output projection: it reads attention_premix and adds at attention_output.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"train": {"path": "weekdays/data", "split": "train"}, "test": {"path": "weekdays/data", "split": "test"}},
  "sites": {
    "premix": {"component": "attention_premix", "layers": 0},
    "o_proj": {"component": "attention_output", "layers": 0},
    "head": {"component": "lm_head_output"}
  },
  "featurizers": {"lora": {"kind": "lora", "rank": 4, "alpha": 8}},
  "steps": {
    "fit": {
      "kind": "fit",
      "train": ["lora"],
      "objective": {"ce": 1.0},
      "epochs": 20,
      "batch_size": 2,
      "seed": 0,
      "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "adapted": {"kind": "forward", "input": "train.input", "interventions": {
          "x": {"site": "premix", "pos": {"index": ":"}},
          "adapt": {"site": "o_proj", "pos": {"index": ":"},
                    "ops": [{"op": "apply", "featurizer": "fit.lora", "with": "adapted.x"}]},
          "logits": {"site": "head", "pos": -1}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits", "target": "train.cf_answer"}
      }
    }
  },
  "saves": ["fit.lora", "fit.held_out.ce"]
}
```

- **The input is flattened.** `adapted.x` is laid out `[rows, pos, num_heads, head_dim]`; `apply` flattens its axes after the positions into the adapter's width.
- **The header records where the input was read** (`with.component`, `with.layer`), and every later use is checked against it before any forward.
- **In the apply document, put both entries in the named bundle.** A bare `"with": "x"` there means that bundle's `x`, in whichever step pulls it in:

```json
"adapted": {
  "x": {"site": "premix", "pos": {"frame": ":", "index": ":"}},
  "adapt": {"site": "o_proj", "pos": {"frame": ":", "index": ":"},
            "ops": [{"op": "apply", "featurizer": "lora", "with": "x"}]}
}
```

A forward step that names it gives the fit's `fit.held_out.ce`. Leaving `with` out, reading `x` at `layer_input` instead, and reading it at layer 1 are each refused before any forward:

```text
clab: adapted.adapt: documents/artifacts/fit.lora.safetensors was trained reading its input at attention_premix layer 0 (with), and this use gives no 'with'; give 'with' an entry at attention_premix layer 0
clab: adapted.adapt: documents/artifacts/fit.lora.safetensors was trained reading its input at attention_premix layer 0 (with), and this use's with 'adapted.x' is layer_input layer 0
clab: adapted.adapt: operand 'adapted.x' is read at attention_premix layer 1, which the nnterp engine's forward reaches no earlier than this write's attention_output layer 0; an operand from the write's own forward is an earlier entry at this site, or at a site the forward reaches first
```

An adapter trained on its own site's value is refused a `with` too (`... header records no input site (with.component): it was trained on its own site's value`). A `generate` step does not serve an operand from its own entries, so a cross-site adapter runs in `forward` steps only; compile refuses it in a generate.

## Pitfalls

- **An adapter trained at a head applies at that head alone.** Its header records the site's `select`, and a use at another head or at the whole component is refused (`header 'select' is '{"num_heads": [1]}'; this document uses 'lora' at attention_premix layer 0, whose selection keeps num_heads [0]`). The checks are in [bundles.md](../usage/bundles.md#what-is-checked-and-when).
- **One adapter, many layers.** A site whose `layers` picks several applies one shared adapter at each; its header records the list. Sweep `layers` for one adapter per layer.
- **The answer is scored at the last position; the adapter acts everywhere.** Narrowing the adapter's `pos` to the last token makes it an activation edit there, not a weight change.
- **The fixture model is random.** `lora_train.json`'s cross entropy falls from about 10.46 to 10.16; the generated text in `lora_apply.json` shows that the adapter is on, nothing about what adapters learn.
