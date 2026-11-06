---
title: Engines
one_liner: "Which engine runs it? nnterp, hooks, ndif and vllm: what each serves, devices, options, quantized and sharded models."
tags: [usage, engines, ndif, vllm, device, remote]
related: [docs/usage/running.md, docs/concepts/one-session.md, docs/concepts/plan-lifecycle.md, docs/usage/sites.md, docs/usage/output.md, docs/developing/engines.md, docs/reference/cli.md]
sources: [clab/engines/__init__.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/engines/vllm.py, clab/engines/standard.py, clab/cli.py, clab/compile/interventions.py, clab/compile/preflight.py, clab/steps/__init__.py]
---

# Engines

## What this is for

An engine loads the document's checkpoint and runs the compiled plan on it. Choose it with `--engine` on the command line or `engine=` in `clab.run`. The engine changes where the model runs and what it can reach, never what the document means. On CPU in float32, `documents/patching.json` saves equal values on nnterp, on hooks and on ndif under `remote=local` ([running.md](running.md#when-two-runs-give-the-same-numbers)). On vllm it gives vLLM's kernels' numbers, which agree with those to vLLM's rounding ([below](#running-on-vllm)). Each engine's flags are in [FORMAT.md](../../FORMAT.md#engines-and-run-options); writing a new engine is [developing/engines.md](../developing/engines.md).

| engine | runs the model | serves | use it for |
|---|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model inside one nnsight session | every component nnterp serves on the checkpoint, the attention interior included | everything, by default |
| `hooks` | here, as a plain `AutoModelForCausalLM` with forward hooks, no nnsight at run time | components at a module boundary only | a second implementation to cross-check a result |
| `ndif` | on an NDIF server; the client holds a model with no weights | what nnterp serves on the server's model | models too large to load yourself |
| `vllm` | here, in vLLM's worker, as nnterp's standardized model over nnsight's vLLM engine: one request per row, no padding | what nnterp serves on vLLM: the stream, the sublayers, the attention interior (its pattern read-only), the head at the last position; no gradient | cross-checking nnterp against vLLM's kernels, or a run on the stack you serve from |

```text
clab run documents/patching.json                                   # nnterp, on cpu
clab run documents/patching.json --engine hooks
clab run documents/patching.json --engine ndif -o remote=local     # the server trip, simulated here
clab run documents/patching.json --engine ndif                     # the NDIF server nnsight is configured for
clab run documents/patching.json --engine vllm                     # vLLM's worker, on the GPUs the process sees
```

## Cross-checking with hooks

The hooks engine reaches the model another way: `register_forward_hook` and `register_forward_pre_hook` on the native modules nnterp names for each component, with no nnsight at run time. When the two engines agree, the result does not depend on how the model was reached.

- **Only `readout` differs.** Positions, edits and ops are clab's own on both engines. `readout` is each engine's own: on nnterp the model's `project_on_vocab`; on hooks the final norm, the head, then the text config's `final_logit_softcapping` as `cap * tanh(logits / cap)`. The two agree on the corpus and on the suite's softcapped tiny Gemma-2. They differ on a family that does more after the head, as the suite's tiny Cohere does by scaling its logits.
- **Where they agree.** `documents/patching.json`, `attention_premix` in `documents/head_patching.json` and the generation in `documents/generated_answer.json` give equal values on both, NaN rows in the same places (`tests/test_documents.py` and `tests/test_generate.py`, `test_the_engines_agree_to_the_bit`).
- **Compare their value files** to check that two runs gave the same numbers. The `cmp` recipe in [running.md](running.md#comparing-two-runs) compares nnterp with hooks and with ndif under `-o remote=local` in one loop.
- **Compare under one attention implementation.** Switching a model between `sdpa` and `eager` moves the last bits of everything downstream.

### Values inside a module's forward

Hooks cannot reach a value read *inside* a module's forward: on a Llama, the attention interior (`attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities`, `attention_head_outputs`). Which components are interior depends on the model family. Preflight refuses one after the model has loaded and before any forward, saying where nnterp reads the value:

```text
$ clab run documents/attention_query.json --engine hooks
clab: counterfactual.v_cf: the hooks engine does not serve attention_queries: nnterp reads it inside a module's forward (source.attention_interface_1.inputs), not at a module boundary
```

- **Module boundaries are served.** `attention_premix`, a dense MLP's `mlp_activation` and the residual-stream components are module boundaries, and hooks serves them.
- **Without `attn_implementation: "eager"`** in the model block, both engines refuse the interior components above. nnterp's reason asks for eager ([sites.md](sites.md#attention-needs-eager)); hooks refuses with the reason above under either attention implementation.
- **On a model whose blocks are of two kinds**, both engines serve a component at the blocks that have its module and refuse it at the others for the same reason. The refusal names the block (`the nnterp engine does not serve attention_output: no self_attn module on this block`).

### A mixture of experts on hooks

Hooks serves a mixture's values that nnterp reads at a module's output, laid out as nnterp lays them out (`[rows, pos, ...]`). There the two engines agree, reads and writes alike (`tests/test_moe.py`). It refuses the rest by name. The components are in [sites.md](sites.md#a-mixture-of-experts).

| component | on hooks |
|---|---|
| `routed_output` | served: the experts module's output |
| `shared_expert_output` | served where the shared expert is a module |
| `router_logits` | served where it is the router's projection's output (JetMoE); refused where nnterp reads it inside the router's forward (Mixtral, OLMoE, Qwen3-MoE) |
| `expert_outputs` | refused: each slot's output is inside the experts' forward |
| `expert_weights`, `expert_indices` | refused: nnterp reads them among the experts module's call arguments (`the hooks engine does not serve expert_weights: nnterp reads it among a module's call arguments (experts.inputs), ...`) |

A dense MLP's value at a mixture's block is refused on each engine, naming what that engine serves there.

## Where the model goes: `--device`

`--device` is a run option, never a document key. It defaults to the engine's own: `cpu`, or `cuda:0` for vllm.

The nnterp engine (at `tp` 1) loads as a meta shell: the module tree, the config and the tokenizer. It loads the weights before the first step of a run that calls the model: a forward, a generation, a `readout` or a `detokenize`. A document of `load` steps and other op steps loads none on one torch device, and its run record's `devices` say `meta`. Under a device map (`auto`, ...) it loads them anyway, since only the loaded weights say where the map puts the run's tensors. The hooks engine loads its weights when it loads.

- **nnterp and hooks** take one torch device (`cpu`, `cuda:0`) or a device map that transformers computes (`auto`, `balanced`, `balanced_low_0`, `sequential`). A device map may spread the model over several GPUs. Each batch's inputs go to the device of the model's first parameter, and transformers moves activations on from there.
- **ndif** with `remote=local` takes one torch device; a device map is refused by name. With `remote=true` the server places the model, and any `--device` other than the default `cpu` is refused:

  ```text
  clab: device: the ndif engine cannot honor 'cuda:0': the server places the model
  ```

The model block cannot place the model. `device`, `device_map` and `dispatch` there are refused when the engine loads, and so is `torch_dtype`, which is spelled `dtype`. Every other key goes to the loader as written ([FORMAT.md](../../FORMAT.md#model)):

```text
clab: model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device
clab: model.dispatch: the nnterp engine cannot honor it: the engine places its model itself
```

A device that cannot be is refused before the model loads: an unknown device type (`device 'gpu': not a torch device ...`), or `cuda:0` where no GPU is visible (`device 'cuda:0': no CUDA device is visible ...`). So is a checkpoint whose `revision` is not in the local cache when the hub is offline (`HF_HUB_OFFLINE=1`): `model.key '<key>' at revision '<revision>': the nnterp engine cannot load it: HF_HUB_OFFLINE is set ...` ([running.md](running.md#check-before-you-run)).

The run record keeps both sides. `device` is the value you asked for, and `devices` lists where the parameters actually were. `clab show` prints them as `device auto` and `served: float32 on cpu`. Under `--device auto` on a machine without a GPU, everything lands on `cpu`.

## Engine options: `-o key=value`

Options other than `--device`, `--batch-size`, `--padding` and `--readout` are the engine's own keyword parameters. `clab describe engines` lists each engine's, as [FORMAT.md](../../FORMAT.md#engines-and-run-options) does. Each value is read as JSON, or as text when it is not JSON: `-o remote=true` is the boolean, and `-o remote=local` is the string.

Options are checked before the model loads, and a bad one is refused by name:

```text
clab: the nnterp engine takes no 'remote'; its options: tp (where the model runs is `device`, given as --device)
clab: the hooks engine takes no 'tp'; its options: none (where the model runs is `device`, given as --device)
clab: remote: 'yes' is not one of True, 'local'
clab: -o device: it is given as --device
clab: -o 'remote': an option is KEY=VALUE
```

In Python the options are keywords: `clab.run(path, engine="ndif", remote="local")`. An engine you already loaded keeps the device and options it loaded with, and `clab.run` refuses new ones ([running.md](running.md)). `plan.run.options` records every option after its defaults, and `clab show` prints them (`remote "local"`).

## Quantized and sharded models

A `model.dtype` may name a quantization instead of a float dtype ([FORMAT.md](../../FORMAT.md#model)):

- `nf4` or `fp4` (bitsandbytes' 4 bits) or `int8` (its LLM.int8()), on the CPU or on CUDA with `bitsandbytes` installed;
- `fp8` (transformers' 8-bit floats), on a GPU of compute capability 8.9 or later.

A quantized model computes in `bfloat16` (`int8` in `float16`), and its values have that dtype. nnterp and hooks load it with the one quantization config; on `documents/patching.json` under `nf4` their value files are equal. `served:` lists the packed weights' dtype beside the compute one (`bfloat16, uint8 on cpu` under `nf4`, `float16, int8 on cpu` under `int8`). Under `-o remote=local` a quantized `dtype` runs as on nnterp. Refused by name:

```text
clab: model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; the nnterp engine's device is 'cpu', and no GPU is visible
clab: model.dtype 'nf4': the ndif engine cannot honor it: the server decides how a model's weights are held
```

`-o tp=N` shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under `torchrun --nproc-per-node=N` ([running.md](running.md#sharding-a-model--o-tp) has the command).

- **Every rank runs the whole plan**, from a seed rank 0 draws. Rank 0 alone writes and prints, and the run record lists every rank's devices.
- **Whole on every rank:** the residual stream, `attention_output`, `mlp_output`, a mixture's routing and `routed_output`, the head and the logits.
- **Refused as held in pieces** (`split`): the attention's interior but `attention_premix`, `mlp_activation`, a mixture's `expert_outputs` (computed inside its sharded experts' forward), and `token_embeddings` where the checkpoint's plan shards the embedding by vocabulary.
- **Refused by name too:** a bitsandbytes `dtype` under `tp`, a checkpoint nnsight cannot split that many ways, `tp` on hooks or ndif, and torchrun around an engine without `tp`. A `grad` entry is refused before the session opens (`<step>.<entry>: the nnterp engine takes no gradient under tp <n>, ...`).
- **The numbers are the unsharded ones to rounding**, not equal to them. A fit's trained values move further, since Adam amplifies the rounding. The ranks agree with each other exactly. In bfloat16 the rounding can flip a greedy generation's near-tie token.

## Running on NDIF

`--engine ndif` builds nnterp's model without its weights: a meta shell holding the module tree, the config and the tokenizer, so nothing else downloads. It then sends the whole run as one request, holding the engine, the compiled plan and the batch size. The server runs preflight and every step, and the results and the run record come back together. The client writes the output directory as usual. The server's address and API key are nnsight's settings (`NDIF_HOST`, `NDIF_API_KEY`, or `nnsight.CONFIG`). Why a document runs the same there is [one-session.md](../concepts/one-session.md).

- **A forward ends at its stop site on the server too.** The ndif engine runs the nnterp engine's forward under either `remote`, so a read at an early layer costs only the layers up to it (`tests/test_efficiency.py` `test_the_stand_in_server_stops_where_the_client_would`).
- **Preflight and step refusals arrive after the queue.** They come back as nnsight's `RemoteError`, and `clab run` prints only its last line. Compile, load and `execute`'s own checks refuse on the client, before anything is sent ([plan-lifecycle.md](../concepts/plan-lifecycle.md#which-stage-refused)).
- **The server places the model.** After a true remote run, `plan.run.device` is None and `clab show` prints `device placed by the server`.

### `remote=local`

`-o remote=local` makes the trip without a server. nnsight serializes the request exactly as it would for the server, deserializes it with non-installed modules hidden, and runs it in this process. The shell loads its weights on `--device` before the first step that calls the model, so the weights must fit here. On `documents/patching.json`, and on `documents/attention_query.json`, which reads inside attention, its value files equal the nnterp engine's. A passing local run shows that the request is self-contained. It does not show that a server has what the run needs.

### What the server must have

- **clab at your version**, installed with its dependencies (nnterp among them). The request names clab's code by reference, and the server imports its own ([one-session.md](../concepts/one-session.md#nothing-of-clab-ships-by-value-it-resolves-by-import-on-the-server)). A true remote run records the server's versions, not yours: compare the `versions:` line of `clab show` with your install.
- **`datasets`, when the document reads a hub dataset.** A hub dataset's rows are not in the request. Compile reads them on the client and records their digest. The server reads them again at the document's `revision` and refuses rows whose digest differs. File datasets, loaded bundles and a `load` step's value travel inside the plan, so the server reads no file of yours. A request carries at most 512 MiB of loaded values (`clab/steps/__init__.py` `LOADED`). A document that loads more is refused before anything is sent, under `-o remote=local` too: `<step>: the values this document loads hold <n> MiB of tensors ...` ([cached-harvest.md](../patterns/cached-harvest.md#under-ndif)).
- **The checkpoint deployed at the document's `key` and `revision`.** The server runs the dtype of its own deployment, and holds and shards the weights as its deployment does, so a quantized `dtype` and `tp` are refused for it. Nothing compares the deployment's float dtype with the document's `dtype`: check the `served:` line of `clab show` (`plan.run.dtype`). A bfloat16 model's numbers differ from float32's ([ops.md](ops.md#the-float32-rule)). Preflight checks the components, sizes and tokenizer of the model the server holds.

## Running on vLLM

`--engine vllm` runs the document on nnsight's vLLM engine through nnterp's `StandardizedVLLM`. Run a document on vllm to cross-check nnterp against vLLM's kernels, or where vLLM is the stack you serve from. Each row is its own request, run on its own at every decode step, so it is not the fast path for short generations.

- **Where it runs.** vLLM's worker holds the model on the GPUs the process sees (`CUDA_VISIBLE_DEVICES` names them), `-o tp=N` of them. `--device` is `cuda:0`, the first of them, where the run keeps its values; any other device is refused by name. It needs vLLM installed beside nnsight, at the version nnsight instruments (nnsight's vLLM guide names it; the run record's `versions` names the one that ran).
- **One request per row.** Every row of a batch is one request carrying its own token ids, so no padding reaches the model. `--batch-size` is how many rows one trace sends at once.
- **A run repeats its numbers.** Under the default `invariant`, a row's numbers depend on neither its batch companions nor `--padding`, and a null write is exact (`tests/test_vllm.py` `test_a_null_write_is_exact_and_a_run_repeats_to_the_bit_at_any_batch_size`).
- **Compare vllm with vllm.** vLLM computes the model with its own kernels. clab on vllm equals plain nnsight on nnterp's vLLM model, as `tests/test_vllm.py` `test_reads_and_writes_are_plain_nnsights_on_vllm_to_the_bit` checks, and nnterp or hooks only to vLLM's rounding. Compare vllm with nnterp on outcomes.
- **A Python script needs a guard.** Put the call under `if __name__ == "__main__":`, since vLLM starts its worker by spawning a process that imports the script.

```text
clab run documents/steer_renormalize.json --engine vllm --batch-size 64
clab run documents/steer_renormalize.json --engine vllm -o tp=2 -o memory=0.8 -o max_len=4096
clab run documents/steer_renormalize.json --engine vllm -o invariant=false     # vLLM's fastest kernels: bits that move between runs
```

| option | default | what it is |
|---|---|---|
| `tp` | `1` | how many GPUs vLLM shards the model over (its `tensor_parallel_size`) |
| `memory` | `0.9` | the fraction of each GPU's memory vLLM reserves for the weights and its cache (its `gpu_memory_utilization`). The run's own values (operands, the joined reads, the ops) live on `cuda:0` outside it: lower it to leave them room, or to share the GPU |
| `max_len` | the checkpoint's | the longest sequence, prompt and new tokens, the engine serves (its `max_model_len`). Lower it where the checkpoint's context would not fit the reserved memory. Preflight refuses a step whose longest row, plus a generation's `max_new_tokens`, does not fit (`answer: prompts.input's longest row has 700 tokens, ...`), since vLLM would cut the generation short with no EOS |
| `invariant` | `true` | vLLM's batch-invariant kernels, under which a run repeats its numbers and a null write is exact. `false` runs its fastest kernels, whose last bits move from one run to the next and with a row's batch companions |

The engine also has Triton run a float32 matmul in IEEE float32 (`TRITON_F32_DEFAULT=ieee`), where nothing set it. `--readout at-reads` runs the model's own readout in the worker (`project_on_vocab`). It agrees with `full` to vLLM's rounding, not exactly, since vLLM's forward fuses the final norm with the last residual add (`tests/test_vllm.py` `test_at_reads_is_full_to_vllms_last_bits`).

### What it serves, and what it refuses

The engine serves everything nnterp serves on vLLM, under the same names and layouts as on the other engines. That is the residual stream (`layer_input`, `layer_output`, `layer_mid`), the sublayers' contributions and the module boundaries between them, `attention_premix`, `mlp_neurons`, `token_embeddings`, `final_norm_output`, the attention interior (`attention_queries`, `attention_keys`, `attention_values`, `attention_head_outputs`), and the head. The interior needs no `attn_implementation: "eager"`: the engine drops that key, since vLLM's attention backend is its own.

Three things vLLM does not give, each refused by name before any forward:

- **The pattern and the scores are recomputed.** vLLM's attention kernel serves nothing between its inputs and its output, so nnterp recomputes `attention_scores` and `attention_probabilities` from the queries and keys. A read is served, in the prefill as on every engine. A write is refused by the engine's `execute` before the run starts (`patched.zero: the vllm engine does not write attention_probabilities: ...`). The attention knockout and pattern-patching patterns run on nnterp or hooks.
- **The head is the last position's.** vLLM computes `logits` for the prompt's last token alone. `lm_head_output` is served as the logits where the model does nothing after the head; on a Gemma-2 or a Cohere it is refused with what the model does. Preflight refuses a read or write of either at any other prefill position (`base.logits: the vllm engine serves logits at the prompt's last position alone ...`). A generation's head at its decode calls is each call's one position, as on every engine.
- **No gradient.** The engine's `execute` refuses a fit (`fit: a fit trains through the model's gradient, which vLLM's kernels do not take; ...`). `start_run` refuses a `grad` entry (`<step>.<entry>: a grad entry takes the model's gradient, ...`). Both come before vLLM's worker starts.

A quantized `dtype` is refused by name when the engine loads (`model.dtype 'nf4': the vllm engine cannot honor it: ...`). Preflight refuses by name a gated MLP's `mlp_activation` (nnterp's reason: vLLM fuses it) and a mixture's routing values (`vLLM runs the mixture of experts inside its fused MoE kernel, which serves no routing value; ...`).

A generate step's options map onto vLLM's sampling (`clab/engines/vllm.py` `sampling`). What the step leaves out is taken from the generation config, as transformers takes it:

| option | on vllm |
|---|---|
| `max_new_tokens` | required: vLLM's own default length is not the generation config's |
| `min_new_tokens` | passed under vLLM's name for it, `min_tokens` |
| `do_sample` | `false` is greedy; `true` samples with `temperature`, `top_p`, `top_k` and `min_p`, each the step's or the config's |
| `repetition_penalty` | applied under greedy decoding too, as transformers applies it |
| `eos_token_id` | the only ids a row stops on; without it, the config's EOS ids |
| `pad_token_id` | what the rows' ids are padded with; without it, the config's, else the first stop id |

- **Any other option fails the step by name before any step runs**, in the engine's `execute` (`answer: generate rejected 'penalty_alpha': the vllm engine's generate maps ...`). So does a generation config that sets a logits processor vLLM's sampling has no name for (`no_repeat_ngram_size`, `typical_p`, `bad_words_ids`, ...), which the other engines would apply.
- **A row stops on the other engines' set.** vLLM's own EOS is told to stand down, so a row stops on the step's `eos_token_id`, else the config's EOS ids, and on nothing else.
- **A row's new ids are what vLLM sampled**, its EOS included, padded past it to the batch's longest row, as transformers pads. A read at a decode call a row never made is missing there, as on the other engines.

The worker's memory is vLLM's own, so a step's `cost.memory` is null. `cost.tokens` counts the batch's padding, which the model never saw; `real` is what it fed. The run record's `devices` and `dtype` are the worker's, and `clab show` prints `served: float32 on cuda:0`. `plan.run.options` records `tp`, `memory`, `max_len` and `invariant`.
