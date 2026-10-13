# nnterp 2.0 cheat-sheet

nnterp standardizes transformer internals on top of nnsight: one envoy tree
and one set of value names across ~100 architectures. Installed editable from
`/home/localjadenfk/wd/nnterp` (branch `0.8-refactor`); its own docs are under
`docs/` there and are worth reading. nnsight is installed editable from
`/home/localjadenfk/wd/nnsight-dev-parent` (branch `dev`).

## Hard traps first

- **Import `nnterp` before anything from `transformers.models...`** or the
  process segfaults, silently.
- **Kernel routing is route-first**: on recurrent/hybrid architectures,
  `route_kernels(...)` must be called before the model is dispatched, traced
  or measured. (Irrelevant for Llama/GPT-2-class models.)
- The attention interior (queries/keys/values/scores/probabilities/
  head_outputs) is served **only under `attn_implementation="eager"`**; under
  sdpa those values report unavailable.
- **Never `nnsight.register(nnterp)`** or otherwise ship it by value; its
  descriptors do not pickle. Remote execution requires nnterp installed
  server-side at the client's version.

## Loading

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer(repo_id, revision=..., dtype=...,
                                attn_implementation="eager",
                                device_map="cpu")
```

- `dispatch=False, device_map=None` gives a **meta shell**: no weights, but
  tokenizer, sizes, `support()`, `order()`/`rank()` and layouts all answer.
  Useful for anything that must know about a checkpoint without loading it.
- `model.tokenizer` is one of nnsight's *persistent objects*: referenced
  inside a session it resolves to the serving side's own tokenizer — the
  correct way to tokenize inside a run (see charter law 3/4).
- The underlying module is `model._module` (e.g. for `.eval()`,
  `.requires_grad_(False)`).

## The tree and the values

Module names are standardized to Llama's spellings and containers are lifted:
`model.layers[i]`, `.layers[i].self_attn`, `.mlp`, `.input_layernorm`,
`.post_attention_layernorm`, `model.norm`, `model.lm_head`,
`model.embed_tokens`.

Standard values live on the envoys and are read/written inside a trace by
plain attribute access (tuple unwrapping and call-argument rebuilding are
handled for you):

```python
with model.trace({"input_ids": ids, "attention_mask": mask}) as tracer:
    x = model.layers[3].layer_output.save()          # read
    model.layers[3].layer_output[:, -1] = 0          # in-place write
    model.layers[4].layer_output = t                 # assign
    logits = tracer.result.logits
```

- Per layer: `layer_input`, `layer_output`; `self_attn.attention_output`,
  `.attention_queries`, `.attention_keys`, `.attention_values`,
  `.attention_scores`, `.attention_probabilities`, `.attention_head_outputs`;
  `mlp.mlp_output`. (MoE, recurrent-mixer and vision blocks have more — see
  the docs.)
- Whole-model: `logits` (the model's own output, after any final softcap),
  `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`.
- Plain module boundaries are nnsight's as-is: `model.norm.output`,
  `model.lm_head.output`, `model.layers[i].mlp.input`, etc.
- `model.project_on_vocab(hidden)` applies the model's own readout (final
  norm, head, softcap) to any residual-stream tensor — the logit-lens
  primitive.
- Generation: `model.generate(batch, max_new_tokens=...) as tracer`, with
  `tracer.iter[:n]` to visit decode steps and `tracer.result` for the ids.
- `model.session(remote=...)` wraps many traces in one request.

## Asking about a checkpoint

- **Availability:** `model.support()` returns `{dotted_name: reason_or_None}`
  where `None` means available on **every** block; `support(layer=i)` is one
  block's answer. Do not probe with `hasattr` (reading an unavailable value
  raises `Unavailable`). An op that moved between transformers versions
  raises `SourceNotAvailable` at trace time; an unknown architecture falls
  back to a `default` family with reduced coverage, or raises
  `UnsupportedFamily`.
- **Forward order:** `model.rank(name, layer=None) -> (side, r)` — `side` is
  the layer index, or `-1` / `num_layers` for whole-model values before/after
  the blocks; equal tuples mean either read order is legal.
  `model.order(layer=i)` gives one block's full ranking. Measured on first
  use by a probe (works on meta weights) and cached on the model. Note:
  nnsight traces require reads in forward order (`OutOfOrderError`
  otherwise), and a write must be installed before the read that observes it.
- **Layouts:** every value's axes are published as names:
  `type(envoy).attention_queries.dims` →
  `('batch', 'num_heads', 'seq', 'qk_head_dim')`. An axis that is a fact
  about the model is named exactly as the size attribute that holds it.
- **Sizes:** `nnterp.axis_size(envoy, axis)` resolves an axis name to its
  number on the serving envoy or the first envoy above it that declares it
  (`hidden_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`,
  `vocab_size`, `intermediate_size`); an axis no envoy declares (`batch`,
  `seq`, the attention pattern's `query`/`key`) raises — its size is the
  tensor's, not the model's. Root sizes (`model.hidden_size`,
  `model.num_layers`, ...) and per-module sizes
  (`model.layers[i].self_attn.num_heads`, `.mlp.intermediate_size`) are
  also plain attributes.
- A value a family lacks can be added or moved at load with `rename=` /
  `envoys=` (see `docs/extending/`).

## nnsight facts that will bite you

- A trace body executes as ordinary Python interleaved with the forward;
  touching a value parks until the forward reaches it.
- `backward()` works between traces inside a session; model weights arrive
  with `requires_grad=True` — freezing is your job.
- A `yield` cannot appear inside a traced/session block (bodies are
  recompiled), so context managers cannot wrap session opening from inside.
- Writes under gradients: prefer out-of-place (clone/scatter) edits; in-place
  edits on values you will differentiate through raise.
- `nnsight.save(obj)` at the top of a session is how results come home;
  locally it is the object you passed, remotely the copy shipped back — fill
  in whatever it returns.
