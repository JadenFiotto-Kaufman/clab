---
title: Engines
one_liner: "Choosing an engine: nnterp, hooks and ndif, what each can serve, device and device maps, engine options, running on an NDIF server or its local simulation, and what the server must have installed."
tags: [usage, engines, ndif, device, remote]
related: [docs/usage/running.md, docs/concepts/one-session.md, docs/concepts/plan-lifecycle.md, docs/usage/sites.md, docs/usage/output.md, docs/developing/engines.md, docs/reference/cli.md]
sources: [clab/engines/__init__.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/engines/standard.py, clab/cli.py]
---

# Engines

## What this is for

An engine loads the document's checkpoint and runs the compiled plan on it. clab ships three, chosen with `--engine` on the command line or `engine=` in `clab.run`. The engine changes where the model runs and what it can reach, never what the document means. `documents/patching.json` gives bit-identical saved tensors on nnterp, on hooks and on ndif through `remote=local`, on CPU in float32. Each engine's flags are in [FORMAT.md](../../FORMAT.md#engines-and-run-options); writing a new engine is [developing/engines.md](../developing/engines.md).

| engine | runs the model | serves | use it for |
|---|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model inside one nnsight session | every component nnterp serves on the checkpoint, the attention interior included | everything, by default |
| `hooks` | here, as a plain `AutoModelForCausalLM` with forward hooks, no nnsight at run time | components at a module boundary only | a second implementation to cross-check a result |
| `ndif` | on an NDIF server; the client holds a model with no weights | what nnterp serves on the server's model | models too large to load yourself |

```text
clab run documents/patching.json                                   # nnterp, on cpu
clab run documents/patching.json --engine hooks
clab run documents/patching.json --engine ndif -o remote=local     # the server trip, simulated here
clab run documents/patching.json --engine ndif                     # the NDIF server nnsight is configured for
```

## Cross-checking with hooks

The hooks engine reaches the model another way: `register_forward_hook` and `register_forward_pre_hook` on the native modules nnterp names for each component, with no nnsight at run time. Positions, edits and ops are clab's own on both engines, so when they agree, the result does not depend on how the model was reached. They agree to the bit on `documents/patching.json`, on `attention_premix` in `documents/head_patching.json`, and on the generation in `documents/generated_answer.json`, whose NaN rows also fall in the same places. Compare the two under one attention implementation: switching a model from `sdpa` to `eager` moves the last bits of everything downstream (about 2e-8 in `head_patching.json`'s logit differences).

What hooks cannot reach is a value read *inside* a module's forward: on a Llama, the attention interior (`attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities`, `attention_head_outputs`). Which components are interior depends on the model family; the refusal says where nnterp reads the value:

```text
$ clab run documents/attention_query.json --engine hooks
clab: counterfactual.v_cf: the hooks engine does not serve attention_queries: nnterp reads it inside a module's forward (source.attention_interface_1.inputs), not at a module boundary
```

Preflight raises it, after the model has loaded and before any forward. `attention_premix`, `mlp_activation` and the residual-stream components are module boundaries, and hooks serves them. Without `attn_implementation: "eager"` in the model block, both engines refuse the same six interior components, with a reason that asks for eager ([sites.md](sites.md)).

## Where the model goes: `--device`

`--device` is a run option, never a document key; it defaults to `cpu`.

- **nnterp and hooks** take one torch device (`cpu`, `cuda:0`) or a device map that transformers computes (`auto`, `balanced`, `balanced_low_0`, `sequential`). A device map may spread the model over several GPUs. Each batch's inputs go to the device of the model's first parameter, and transformers moves activations on from there.
- **ndif** with `remote=local` takes one torch device; a device map is refused by name. With `remote=true` the server places the model, and any `--device` other than the default `cpu` is refused:

  ```text
  clab: device: the ndif engine cannot honor 'cuda:0': the server places the model
  ```

The model block cannot place the model. `device`, `device_map` and `dispatch` there are refused when the engine loads, and so is `torch_dtype`, which is spelled `dtype`. Every other key goes to the loader as written ([FORMAT.md](../../FORMAT.md#model)):

```text
clab: model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device
```

nnterp and hooks hand the value to torch and transformers unchecked. An unknown device type (`gpu`), or `cuda:0` on a machine with no visible GPU, fails during loading with torch's own error and a traceback instead of a `clab:` line. So does a checkpoint whose `revision` is not in the local cache when the hub is offline (`HF_HUB_OFFLINE=1`): an `OSError` from transformers.

The run record keeps both sides. `device` is the value you asked for, and `devices` lists where the parameters actually were. `clab show` prints them as `device auto` and `served: float32 on cpu`. Under `--device auto` on a machine without a GPU, everything lands on `cpu`.

## Engine options: `-o key=value`

Options other than `--device` and `--batch-size` are the engine's own keyword parameters. `clab describe engines` lists them. nnterp and hooks have none; ndif has `remote`. Each value is read as JSON, or as text when it is not JSON: `-o remote=true` is the boolean, and `-o remote=local` is the string.

Options are checked before the model loads, and a bad one is refused by name:

```text
clab: the nnterp engine takes no 'remote'; its options: device
clab: remote: 'yes' is not one of True, 'local'
clab: -o device: it is given as --device
clab: -o 'remote': an option is KEY=VALUE
```

In Python the options are keywords: `clab.run(path, engine="ndif", remote="local")`. An engine you already loaded keeps the device and options it loaded with, and `clab.run` refuses new ones ([running.md](running.md)). `plan.run.options` records every option after its defaults, and `clab show` prints them (`remote "local"`).

## Running on NDIF

`--engine ndif` builds nnterp's model without its weights: a meta shell holding the module tree, the config and the tokenizer, so nothing else downloads. It then sends the whole run as one request. The request holds the engine, the compiled plan and the batch size. The server runs preflight and every step, and one dict comes back with the results and the run record. The client writes the output directory as usual. The server's address and API key are nnsight's settings (`NDIF_HOST`, `NDIF_API_KEY`, or `nnsight.CONFIG`). Why a document runs the same there is [one-session.md](../concepts/one-session.md).

Where a refusal comes from ([plan-lifecycle.md](../concepts/plan-lifecycle.md#where-each-stage-runs) has every stage):

- Compile and engine-loading refusals happen on the client, before anything is sent.
- Preflight and step refusals happen on the server, after the request has waited in its queue. They come back as nnsight's `RemoteError`, and `clab run` prints only its last line.

After a true remote run, `plan.run.device` is None and `clab show` prints `device placed by the server`.

### `remote=local`

`-o remote=local` makes the trip without a server. nnsight serializes the request exactly as it would for the server, deserializes it with non-installed modules hidden, and runs it in this process. The shell is placed on `--device` at the first trace, so the weights must fit here. The results match the nnterp engine's to the bit on `documents/patching.json` and on `documents/attention_query.json`, which reads inside attention. A passing local run shows that the request is self-contained. It does not show that a server has what the run needs.

### What the server must have

- **clab at your version**, installed with its dependencies (nnterp among them). Apart from the session block's own lines, the request carries no clab code: it names clab's engine, plan and steps by reference, and the server imports its own copies. A true remote run records the server's versions, not yours: compare the `versions:` line of `clab show` with your install.
- **`datasets`, when the document reads a hub dataset.** A hub dataset's rows are not in the request. Compile reads them on the client and records their digest. The server reads them again at the document's `revision` and refuses rows whose digest differs. File datasets and loaded bundles travel inside the plan.
- **The checkpoint deployed at the document's `key` and `revision`.** The server runs the dtype of its own deployment. Nothing compares that dtype with the document's `dtype`, so check the `served:` line of `clab show` (`plan.run.dtype`). A bfloat16 model's numbers differ from float32's ([ops.md](ops.md#the-float32-rule) has the float32 rule). Preflight runs on the server, so it checks the components served, the sizes and the tokenizer of the model the server holds.
