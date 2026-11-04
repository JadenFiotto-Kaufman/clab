---
title: Engines
one_liner: "Choosing an engine: nnterp, hooks, ndif and vllm, what each can serve, device and device maps, engine options, quantized and sharded models, running on an NDIF server or its local simulation and what the server must have installed, and running on vLLM: what it serves, what it refuses, its bits and its cost."
tags: [usage, engines, ndif, device, remote]
related: [docs/usage/running.md, docs/concepts/one-session.md, docs/concepts/plan-lifecycle.md, docs/usage/sites.md, docs/usage/output.md, docs/developing/engines.md, docs/reference/cli.md]
sources: [clab/engines/__init__.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/engines/vllm.py, clab/engines/standard.py, clab/cli.py, clab/compile/interventions.py, clab/compile/preflight.py]
---

# Engines

## What this is for

An engine loads the document's checkpoint and runs the compiled plan on it. clab ships four, chosen with `--engine` on the command line or `engine=` in `clab.run`. The engine changes where the model runs and what it can reach, never what the document means. `documents/patching.json` gives bit-identical saved tensors on nnterp, on hooks and on ndif through `remote=local`, on CPU in float32; on vllm it gives vLLM's kernels' numbers, which agree with those to vLLM's rounding ([below](#running-on-vllm)). Each engine's flags are in [FORMAT.md](../../FORMAT.md#engines-and-run-options); writing a new engine is [developing/engines.md](../developing/engines.md).

| engine | runs the model | serves | use it for |
|---|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model inside one nnsight session | every component nnterp serves on the checkpoint, the attention interior included | everything, by default |
| `hooks` | here, as a plain `AutoModelForCausalLM` with forward hooks, no nnsight at run time | components at a module boundary only | a second implementation to cross-check a result |
| `ndif` | on an NDIF server; the client holds a model with no weights | what nnterp serves on the server's model | models too large to load yourself |
| `vllm` | here, in vLLM's worker, as nnterp's standardized model over nnsight's vLLM engine: one request per row, no padding | what nnterp serves on vLLM: the stream, the sublayers, the attention interior (its pattern read-only), the head at the last position; no gradient | cross-checking nnterp against vLLM's kernels, or a run on the stack you serve from; slower than nnterp on the card measured ([below](#the-cost)) |

```text
clab run documents/patching.json                                   # nnterp, on cpu
clab run documents/patching.json --engine hooks
clab run documents/patching.json --engine ndif -o remote=local     # the server trip, simulated here
clab run documents/patching.json --engine ndif                     # the NDIF server nnsight is configured for
clab run documents/patching.json --engine vllm                     # vLLM's worker, on the GPUs the process sees
```

## Cross-checking with hooks

The hooks engine reaches the model another way: `register_forward_hook` and `register_forward_pre_hook` on the native modules nnterp names for each component, with no nnsight at run time. Positions, edits and ops are clab's own on both engines, except `readout`, which each engine computes itself: on nnterp it is the model's own, nnterp's `project_on_vocab`; on hooks it is the standard form, the final norm and the head, then the text config's `final_logit_softcapping` as `cap * tanh(logits / cap)`. The two agree to the bit on the corpus and on a softcapped tiny Gemma-2; on a family that does more after the head (a tiny Cohere scales its logits) they differ. So when the engines agree, the result does not depend on how the model was reached. They agree to the bit on `documents/patching.json`, on `attention_premix` in `documents/head_patching.json`, and on the generation in `documents/generated_answer.json`, whose NaN rows also fall in the same places. Compare the two under one attention implementation: switching a model from `sdpa` to `eager` moves the last bits of everything downstream (about 2e-8 in `head_patching.json`'s logit differences).

### Comparing two runs to the bit

Two runs agree to the bit when their value files do. Every file in the output directory but `manifest.json` holds values or, in `tokens.json`, the tokens, which one checkpoint's tokenizer makes alike, so compare those byte for byte; the manifest always differs, since it records the engine and the time:

```text
for f in out/nnterp/*; do [ "${f##*/}" = manifest.json ] || cmp "$f" "out/hooks/${f##*/}"; done
```

`cmp` prints nothing where the files match, NaN rows and JSON values included. That every read and write landed alike is in the records: the manifests' `steps`, which hold each entry's `positions` and each forward's schedule, are equal (`json.load(open("out/nnterp/manifest.json"))["steps"] == json.load(open("out/hooks/manifest.json"))["steps"]`, as on `patching.json`, `head_patching.json` and `generated_answer.json`). Compare runs made at one batch size, padding and thread count ([running.md](running.md#batch-size-and-reproducibility)).

What hooks cannot reach is a value read *inside* a module's forward: on a Llama, the attention interior (`attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities`, `attention_head_outputs`). Which components are interior depends on the model family; the refusal says where nnterp reads the value:

```text
$ clab run documents/attention_query.json --engine hooks
clab: counterfactual.v_cf: the hooks engine does not serve attention_queries: nnterp reads it inside a module's forward (source.attention_interface_1.inputs), not at a module boundary
```

Preflight raises it, after the model has loaded and before any forward. Both engines also answer layer by layer from nnterp's one answer of what each block holds: on a hybrid whose blocks are of two kinds, a component is served at the blocks that have its module and refused at the others, by the same reason on both. On the tiny `yujiepan/qwen3-next-moe-tiny-random`, whose layers 0 to 2 are linear attention and layer 3 full attention, `attention_output` at layer 3 is byte-identical on nnterp and hooks, and at layer 0 both refuse it (`r.x: the nnterp engine does not serve attention_output: no self_attn module on this block`). `attention_premix`, a dense MLP's `mlp_activation` and the residual-stream components are module boundaries, and hooks serves them. Without `attn_implementation: "eager"` in the model block, both engines refuse the same six interior components: nnterp with a reason that asks for eager ([sites.md](sites.md#attention-needs-eager)), hooks with the one above, under either attention implementation.

### A mixture of experts on hooks

Of a mixture's six components ([sites.md](sites.md#a-mixture-of-experts)) hooks serves those nnterp reads at a module's output: `routed_output` (the experts module's), `shared_expert_output` where the shared expert is a module, and `router_logits` where it is the router's projection's (JetMoE). The model holds the routed sum and JetMoE's router logits flat over tokens, and hooks serves all three as nnterp does, `[rows, pos, ...]`; there the two engines agree to the bit, reads and writes alike. The rest it refuses by name: the router's logits where nnterp reads them inside the router's forward (Mixtral, OLMoE, Qwen3-MoE), each slot's output inside the experts' forward, and the routing pair, which nnterp reads among the experts module's call arguments (`the hooks engine does not serve expert_weights: nnterp reads it among a module's call arguments (experts.inputs), not at a module boundary`). A dense MLP's value at a mixture's block is refused on each engine naming what that engine serves there.

## Where the model goes: `--device`

`--device` is a run option, never a document key; it defaults to the engine's own: `cpu`, or `cuda:0` for vllm.

The nnterp engine (at `tp` 1) loads as a meta shell, the module tree, the config and the tokenizer, and loads the weights there before the first step of a run that calls the model: a forward, a generation, a `readout` or a `detokenize`. A document of `load` steps and other op steps loads none on one torch device, and its run record's `devices` say `meta`; under a device map (`auto`, ...) it loads them, since only the loaded weights say where the map puts the run's tensors. The hooks engine loads its weights when it loads.

- **nnterp and hooks** take one torch device (`cpu`, `cuda:0`) or a device map that transformers computes (`auto`, `balanced`, `balanced_low_0`, `sequential`). A device map may spread the model over several GPUs. Each batch's inputs go to the device of the model's first parameter, and transformers moves activations on from there.
- **ndif** with `remote=local` takes one torch device; a device map is refused by name. With `remote=true` the server places the model, and any `--device` other than the default `cpu` is refused:

  ```text
  clab: device: the ndif engine cannot honor 'cuda:0': the server places the model
  ```

The model block cannot place the model. `device`, `device_map` and `dispatch` there are refused when the engine loads, and so is `torch_dtype`, which is spelled `dtype`. Every other key goes to the loader as written ([FORMAT.md](../../FORMAT.md#model)):

```text
clab: model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device
```

A device that cannot be is refused before the model loads: an unknown device type (`device 'gpu': not a torch device ...`), or `cuda:0` where no GPU is visible (`device 'cuda:0': no CUDA device is visible ...`). So is a checkpoint whose `revision` is not in the local cache when the hub is offline (`HF_HUB_OFFLINE=1`), as `model.key '<key>' at revision '<revision>': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; fetch it online first` ([running.md](running.md#check-before-you-run)).

The run record keeps both sides. `device` is the value you asked for, and `devices` lists where the parameters actually were. `clab show` prints them as `device auto` and `served: float32 on cpu`. Under `--device auto` on a machine without a GPU, everything lands on `cpu`.

## Engine options: `-o key=value`

Options other than `--device`, `--batch-size`, `--padding` and `--readout` are the engine's own keyword parameters. `clab describe engines` lists them. nnterp has `tp` ([below](#quantized-and-sharded-models)), hooks has none, ndif has `remote`, and vllm has `tp`, `memory`, `max_len` and `invariant` ([below](#running-on-vllm)). Each value is read as JSON, or as text when it is not JSON: `-o remote=true` is the boolean, and `-o remote=local` is the string.

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

A `model.dtype` may name a quantization instead of a float dtype ([FORMAT.md](../../FORMAT.md#model)): `nf4` or `fp4` (bitsandbytes' 4 bits) or `int8` (its LLM.int8()), on the CPU or on CUDA with `bitsandbytes` installed, or `fp8` (transformers' 8-bit floats) on a GPU of compute capability 8.9 or later. A quantized model computes in `bfloat16` (`int8` in `float16`), and its values have that dtype. nnterp and hooks load it with the one quantization config and agree to the bit: on `documents/patching.json` under `nf4` their value files are byte-identical. `served:` lists the packed weights' dtype beside the compute one (`bfloat16, uint8 on cpu` under `nf4`, `float16, int8 on cpu` under `int8`). Refused by name:

```text
clab: model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; the nnterp engine's device is 'cpu', and no GPU is visible
clab: model.dtype 'nf4': the ndif engine cannot honor it: the server decides how a model's weights are held
```

Under `-o remote=local` a quantized `dtype` runs as on nnterp.

`-o tp=N` shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under `torchrun --nproc-per-node=N` ([running.md](running.md#sharding-a-model--o-tp) has the command). Every rank runs the whole plan from a seed rank 0 draws, and rank 0 alone writes and prints; the run record lists every rank's devices. A value the ranks hold in pieces is refused by name (`split`): the attention's interior but `attention_premix`, `mlp_activation`, a mixture's `expert_outputs` (computed inside its sharded experts' forward), and `token_embeddings` where the checkpoint's plan shards the embedding by vocabulary. The residual stream, `attention_output`, `mlp_output`, a mixture's routing and `routed_output`, the head and the logits are whole on every rank. A bitsandbytes `dtype` under `tp`, a checkpoint nnsight cannot split that many ways, `tp` on hooks or ndif, a `grad` entry, and torchrun around an engine without `tp` are refused by name too. Results under `tp` are the unsharded ones to rounding, not to the bit (on `documents/patching.json` at `tp` 2 on the CPU, `logit_diff` moves by 3e-8), and a fit's trained values move further, since Adam amplifies the rounding; the ranks agree with each other to the bit. In bfloat16 the rounding can flip a greedy generation's near-tie token.

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
- **`datasets`, when the document reads a hub dataset.** A hub dataset's rows are not in the request. Compile reads them on the client and records their digest. The server reads them again at the document's `revision` and refuses rows whose digest differs. File datasets, loaded bundles and a `load` step's value travel inside the plan, so the server reads no file of yours. A request carries at most 512 MiB of loaded values, and a document that loads more is refused before anything is sent ([cached-harvest.md](../patterns/cached-harvest.md#under-ndif)).
- **The checkpoint deployed at the document's `key` and `revision`.** The server runs the dtype of its own deployment, and holds and shards the weights as its deployment does, so a quantized `dtype` and `tp` are refused for it. Nothing compares the deployment's float dtype with the document's `dtype`, so check the `served:` line of `clab show` (`plan.run.dtype`). A bfloat16 model's numbers differ from float32's ([ops.md](ops.md#the-float32-rule) has the float32 rule). Preflight runs on the server, so it checks the components served, the sizes and the tokenizer of the model the server holds.

## Running on vLLM

`--engine vllm` runs the document on nnsight's vLLM engine through nnterp's `StandardizedVLLM`. vLLM's worker holds the model on the GPUs the process sees (`CUDA_VISIBLE_DEVICES` names them), `-o tp=N` of them, and batches the requests clab sends it. Every row of a batch is one request carrying its own token ids: no padding reaches the model, so under the default `invariant` a row's numbers depend on neither its batch companions nor `--padding`, and `--batch-size` is how many rows one trace sends at once. It needs vLLM installed beside nnsight, at the version nnsight instruments (nnsight's vLLM guide names it; the run record's `versions` names the one that ran), and a CUDA device. `--device` is `cuda:0`, the first of vLLM's GPUs, where the run keeps its values; any other device is refused by name. A Python script that loads this engine needs an `if __name__ == "__main__":` guard around the call, since vLLM starts its worker by spawning a process that imports the script.

```text
clab run documents/steer_renormalize.json --engine vllm --batch-size 64
clab run documents/steer_renormalize.json --engine vllm -o tp=2 -o memory=0.8 -o max_len=4096
clab run documents/steer_renormalize.json --engine vllm -o invariant=false     # vLLM's fastest kernels: bits that move between runs
```

| option | default | what it is |
|---|---|---|
| `tp` | `1` | how many GPUs vLLM shards the model over (its `tensor_parallel_size`); passed through, measured at 1 only |
| `memory` | `0.9` | the fraction of each GPU's memory vLLM reserves for the weights and its cache (its `gpu_memory_utilization`). The run's own values (operands, the joined reads, the ops) live on `cuda:0` outside it: lower it to leave them room, or to share the card |
| `max_len` | the checkpoint's | the longest sequence, prompt and new tokens, the engine serves (its `max_model_len`); lower it where the checkpoint's context would not fit the reserved memory. A step whose longest row, plus a generation's `max_new_tokens`, does not fit is refused by name before any forward (`answer: prompts.input's longest row has 700 tokens, and the step generates up to 12 more, past the 512 the vllm engine serves ...`), since vLLM would cut the generation short with no EOS |
| `invariant` | `true` | vLLM's batch-invariant kernels, under which a run reproduces to the bit; `false` runs its fastest kernels, whose last bits move from one run to the next |

### What it serves, and what it refuses

Everything nnterp serves on vLLM, under the same names and layouts as on the other engines: the residual stream (`layer_input`, `layer_output`, `layer_mid`), the sublayers' contributions and the module boundaries between them, `attention_premix`, `mlp_neurons`, `token_embeddings`, `final_norm_output`, the attention interior (`attention_queries`, `attention_keys`, `attention_values`, `attention_head_outputs`) without `attn_implementation: "eager"`, which the model block may still carry (the engine drops it: vLLM's attention backend is its own), and the head. Three things vLLM does not give, so preflight refuses them by name before any forward:

- **The pattern and the scores are recomputed.** vLLM's attention kernel serves nothing between its inputs and its output, so nnterp recomputes `attention_scores` and `attention_probabilities` from the queries and keys: a read is served (in the prefill, as on every engine: compile refuses a read of them at a decode call), and a write is refused (`patched.zero: the vllm engine does not write attention_probabilities: ...; write attention_queries, attention_keys or attention_head_outputs instead`). The attention knockout and pattern-patching patterns run on nnterp or hooks.
- **The head is the last position's.** vLLM computes `logits` for the prompt's last token alone, and `lm_head_output` is served as the logits where the model does nothing after the head (no softcap, no scale: the two are one value then; on a Gemma-2 or a Cohere it is refused with what the model does). A read or write of either at any other prefill position is refused (`base.logits: the vllm engine serves logits at the prompt's last position alone (...); this entry meets position [0] on a row of 7 tokens: name pos -1, or read layer_output at the last block and apply the readout op`); a generation's head at its decode calls is each call's one position, as on every engine.
- **No gradient.** A fit and a `grad` entry are refused (`fit: a fit trains through the model's gradient, which vLLM's kernels do not take; run it on the nnterp or hooks engine`).

Also refused: a quantized `dtype` (`model.dtype 'nf4': the vllm engine cannot honor it: vLLM holds a quantized model's weights in its own formats, not bitsandbytes', ...`), a gated MLP's `mlp_activation` (nnterp's reason: vLLM fuses it) and a mixture's routing values (`vLLM runs the mixture of experts inside its fused MoE kernel, which serves no routing value; the mixture's output is mlp_output`), and a generate option the engine cannot map onto vLLM's sampling. A generate step's `max_new_tokens` (required: vLLM's own default length is not the generation config's), `min_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `eos_token_id` and `pad_token_id` are mapped, what the step leaves out taken from the generation config as transformers takes it: `do_sample: false` is greedy, `true` samples with the temperature, top_p, top_k and min_p the step or the config gives, and the config's `repetition_penalty` applies either way; a row stops on the step's `eos_token_id`, else the config's EOS ids, and on nothing else (vLLM's own EOS is told to stand down, so the set is the one the other engines stop on). Any other option fails the step by name before anything runs (`answer: generate rejected 'penalty_alpha': the vllm engine's generate maps ...`), and so does a generation config that sets a logits processor vLLM's sampling has no name for (`no_repeat_ngram_size`, `typical_p`, `bad_words_ids`, ...), which the other engines would apply. A row's new ids are what vLLM sampled, its EOS included, padded past it to the batch's longest row with the config's pad token (else the first stop id), as transformers pads; a read at a decode call a row never made is missing there, as on the other engines.

### The bits

**A run reproduces to the bit, and a null is exact, under the default `invariant`.** vLLM's default kernels are not run-to-run stable: two identical request sets through SmolLM2-135M in float32, with no intervention at all, gave logits 5e-3 apart, and on Llama-3.1-8B in bfloat16 163 of 512 greedy generations differed between two unsteered runs. Under `VLLM_BATCH_INVARIANT=1`, which the engine sets as vLLM's worker starts, every read of every standard value is a no-op to the bit, writing back what was read changes nothing, and the model's values are the same bits at any batch size, so two runs of a document write byte-identical value files (on the 8B card, batch 64 and batch 256 agree to the bit in the forward and in 5,000 generations); an entry's own ops after its last edit still run per batch on the client's GPU, where a matmul-shaped op can pick its kernel by the batch's shape, as on every engine on CUDA. `-o invariant=false` buys vLLM's fastest kernels and gives up all of that: a row's numbers then move with its batch companions and from run to run. The engine also has Triton run a float32 matmul in float32 (`TRITON_F32_DEFAULT=ieee`, where nothing set it): Triton's default is TF32, under which the batch-invariant matmul moved SmolLM2's float32 logit differences by 6e-3 against transformers', and by 8e-6 once IEEE.

**The engines do not agree to the bit across vLLM.** vLLM computes the same model with other kernels (its attention, its fused norms and activations), so clab on vllm equals plain nnsight on `StandardizedVLLM` to the bit (the same requests in one trace, `tests/test_vllm.py`), and nnterp or hooks to vLLM's rounding: on `documents/patching.json` rewritten for SmolLM2-135M in float32, `iia` agrees and `logit_diff` differs by 8e-6; the readout's probabilities by 5e-6. In bfloat16 the differences are the dtype's, as between any two bfloat16 stacks ([ops.md](ops.md#the-float32-rule)). Compare vllm runs with vllm runs to the bit, and vllm with nnterp on outcomes. `--readout at-reads` is the model's own readout, run in the worker (`project_on_vocab`), and agrees with `full` to the last bits (2e-5 in the logits on SmolLM2 in float32), as it promises everywhere: it runs the final norm on the stream alone, where vLLM's forward fuses that norm with the last residual add, another kernel.

### The cost

**On the one card measured, the vllm engine is slower than nnterp.** Steering Llama-3.1-8B in bfloat16 through 5,000 64-token prompts on one RTX A6000 (the wave-3 card: the direction added at block 16 at every position, the top-5 next tokens read at the last; then 32 greedy tokens):

| version | forward, batch 64 | forward, batch 256 | generation, batch 64 | generation, batch 256 |
|---|---|---|---|---|
| clab, vllm engine | 95.7 s | 100.6 s | 236 s | 183 s |
| clab, nnterp engine | 53.6 s | 51.2 s | 146 s | 99.6 s |
| plain nnsight on vLLM, the steer as one `model.edit()` | 45.9 s | | 84.7 s | |
| plain vLLM, no intervention | 41.5 s | | 76.2 s | |

The numbers agree as two bfloat16 stacks do: against the nnterp engine the top token agrees on 98.4% of rows, the ordered top 5 on 82.6%, and 65.7% of the 32-token generations are identical (the same rates as plain nnsight on vLLM against transformers); batch 64 and batch 256 give the same bits; the null (the direction at scale 0) is exact in the top 5, their probabilities and the generations.

Where the time goes, by construction of nnsight's vLLM engine: every row is a request, and a request's block is serialized with the request (its own rows of every per-row operand, and every shared one, a direction or a featurizer, with each request) and runs in the worker on its own, once per step. A block that reads the head (`logits`, `lm_head_output`) or a generation's samples references the model's root, which travels by value with each request, about 9 ms a row (nnsight ships a block's layer envoys by reference, the root by value): some 50 s of the 96 s forward. At each decode step every request's block runs `visit_site` on its own row, a millisecond or so a request a step, of which about 40% is clab's block and 60% nnsight's per-request path against its `edit()` path (measured on Llama-3.2-1B: 64 rows by 32 tokens take 0.52 s with no block, 0.87 s as an edit, 1.12 s as one block per request, 1.55 s through this engine); the nnterp engine does each step's work once for the whole batch. So this engine pays per row and per step where the others pay per batch, and no batch size recovers it. What would change the picture is an nnsight that ships the root by reference, and running a batch as one `model.edit()`, both open items; until then run a document on vllm to cross-check the nnterp engine with another stack's kernels, or where vLLM is the stack you serve from.

The worker's memory is vLLM's own: a step's `cost.memory` is null (`nvidia-smi` says), and `cost.tokens` counts the batch's padding, which the model never saw (`real` is what it fed). The run record's `devices` and `dtype` are the worker's; `clab show` prints `served: float32 on cuda:0`. `plan.run.options` records `tp`, `memory`, `max_len` and `invariant`.
