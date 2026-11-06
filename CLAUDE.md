# clab — Agent Guide

**Find the one page for your task in the table below**, or start at [docs/index.md](docs/index.md), the router for every docs section, and read that page before writing a document.

## By task

| I want to... | Read |
|---|---|
| Write my first document | [usage/writing-a-document.md](docs/usage/writing-a-document.md) |
| Load a dataset, inline rows or a chat, or cut a column from another's text | [usage/data.md](docs/usage/data.md) |
| Choose a component or layers, or read attention at keys relative to each query (previous-token, induction) | [usage/sites.md](docs/usage/sites.md), its [`select`](docs/usage/sites.md#select) for keys |
| Point at a token | [usage/positions.md](docs/usage/positions.md), [concepts/positions.md](docs/concepts/positions.md) |
| Read or write at a site | [usage/interventions.md](docs/usage/interventions.md) |
| Score or reduce a value, or compute on one (a norm, a ratio, a comparison) | [usage/ops.md](docs/usage/ops.md) |
| Run one document at many settings | [usage/sweeps.md](docs/usage/sweeps.md) |
| Generate text and score it | [usage/generate.md](docs/usage/generate.md), [patterns/generation-experiments.md](docs/patterns/generation-experiments.md) |
| Score what a generation says after a marker, or how long its reasoning is (`after`, `until`, `count`) | [patterns/generation-experiments.md](docs/patterns/generation-experiments.md#score-what-follows-a-text-after) |
| Train a featurizer, or load one | [usage/featurizers.md](docs/usage/featurizers.md), [usage/fit.md](docs/usage/fit.md), [usage/bundles.md](docs/usage/bundles.md) |
| Harvest once and use it in another document (`load`) | [patterns/cached-harvest.md](docs/patterns/cached-harvest.md) |
| Find a latent's max-activating examples (`top_k` over `["rows", "pos"]`) | [patterns/sae.md](docs/patterns/sae.md#find-a-latents-max-activating-examples) |
| Run a technique (activation, path and attribution patching; zero, mean, resample and optimal ablation; expert ablation; attention; logit and tuned lenses; steering; DAS and DBM; head masking; SAEs; probes; LoRA) | [patterns/index.md](docs/patterns/index.md) |
| Run a document and choose an engine (nnterp, hooks, ndif, vllm) | [usage/running.md](docs/usage/running.md), [usage/engines.md](docs/usage/engines.md) |
| Check a document before the model loads | [usage/running.md](docs/usage/running.md#check-before-you-run) |
| Feed fewer tokens, or compute the head only where it is read | [usage/running.md](docs/usage/running.md#batch-size-and-padding) |
| Load a quantized model, or shard one over GPUs with `tp` | [usage/engines.md](docs/usage/engines.md#quantized-and-sharded-models), [usage/running.md](docs/usage/running.md#sharding-a-model--o-tp) |
| Run many rows on GPUs with vLLM | [usage/engines.md](docs/usage/engines.md#running-on-vllm) |
| Run on an NDIF server | [usage/engines.md](docs/usage/engines.md#running-on-ndif), [concepts/one-session.md](docs/concepts/one-session.md) |
| Cross-check a result on a second engine | [usage/engines.md](docs/usage/engines.md#cross-checking-with-hooks) |
| Read a run back | [usage/output.md](docs/usage/output.md) |
| Fix a refusal, or find out why the numbers are wrong when nothing was refused | [errors/index.md](docs/errors/index.md), [errors/symptom-index.md](docs/errors/symptom-index.md) |
| Look up an exact field, op, component or option | [FORMAT.md](FORMAT.md), [reference/index.md](docs/reference/index.md) |
| Extend clab (an op, a featurizer kind, a component, a step kind, an engine) | [developing/extending.md](docs/developing/extending.md), [developing/engines.md](docs/developing/engines.md) |
| Understand why it behaves this way | [concepts/index.md](docs/concepts/index.md) |

## Before writing a clab document

Each mistake below runs without a refusal and costs a conclusion.

- **An entry without an edit in `ops` only reads**: a write that lost its `ops` runs and returns the clean numbers. See [interventions](docs/usage/interventions.md).
- **Write an answer column as the model continues the prompt, leading space included**: on a byte-level BPE tokenizer the other spelling is another one-token answer. See [data](docs/usage/data.md#answer-columns).
- **Add `strip: true` where `contains` or a scope searches generated text**: a SentencePiece tokenizer drops the first generated token's leading space when decoding. `contains` is a substring test; count whole words with a scope over the decode calls. See [generate](docs/usage/generate.md#the-first-generated-tokens-leading-space), [generation experiments](docs/patterns/generation-experiments.md#did-it-say-the-answer-contains-over-the-text).
- **Carry a null control**: writing back what was read must give the clean run's numbers. Compare two runs only at one batch size, padding, readout, device and thread count; a fit's body and evals run at the fit's own `batch_size`. See [interventions](docs/usage/interventions.md#the-null-control), [running](docs/usage/running.md#when-two-runs-give-the-same-numbers).
- **Carry a null for every gradient**: a `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended to its ops must be exactly zero. A `grad` of the wrong entry gives a plausible map. See [attribution patching](docs/patterns/attribution-patching.md#check-the-gradient).
- **Check a scoped read or write's `positions` in `clab show` on rows of each kind before trusting it**: `occurrence` counts matches in the whole row, the prompt's copies included. Scope a teacher-forced answer with `"occurrence": -1`. See [positions](docs/usage/positions.md#check-where-it-landed).
- **Map an index along a `select`ed axis back through the select list**: `argmax` and `top_k` count within it; `clab table` prints the index kept. See [sites](docs/usage/sites.md#select).
- **Give each of two sweeps of one field its own `label`**, or they zip into the diagonal instead of a grid. See [sweeps](docs/usage/sweeps.md#labels-zip-or-cross).
- **Sweep `layers` to patch one layer at a time**: a slice or list on a write patches every layer it picks at once. See [interventions](docs/usage/interventions.md#writes-across-layers).
- **Sweep a patch's read `pos` with its write `pos`**, or one read value lands at every swept position. See [activation patching](docs/patterns/activation-patching.md#a-layer-by-position-map).
- **Scope both sides of a patch between two inputs**: an unscoped `pos` pairs index with index, whatever tokens share an index on rows of equal length. See [symptom index](docs/errors/symptom-index.md#a-window-between-rows-of-equal-length).
- **List a freeze before the patch it protects at one site**: listed after, it overwrites the patch, and every point equals the clean run. See [interventions](docs/usage/interventions.md#the-order-ops-run-in).
- **After a fit, name the trained featurizer `<fit>.<name>`**: the bare name is the untrained draw. See [fit](docs/usage/fit.md#members).
- **Carry an `epochs: 0` point beside every fit**: a trained featurizer means something only where it beats its own draw on the same rows. See [fit](docs/usage/fit.md#epochs-0-is-a-null-control).
- **Write `do_sample: false` on every generate step**: options a step leaves out come from the checkpoint's generation config, which may sample, and nothing seeds sampling. See [generate](docs/usage/generate.md#options).
- **Count `alignment_missing` rows before reading a mean over a generation**: reductions skip them, and the number does not say so. See [alignment missing](docs/errors/alignment-missing.md#count-it-prevent-it-or-refuse-it).
- **State a published SAE's `centered`, `activation` and layer, inline in `site`, from its release.** A wrong `centered`, a TopK file stated as `relu`, a wrong layer, or a `site` naming the entries' site is not refused. Compare its mean L0 over every token after BOS with the release's, and for a TopK release count the tokens whose L0 exceeds its `k`. See [featurizers](docs/usage/featurizers.md#loading-a-published-sparse-autoencoder).
- **Recompute a few points in float32 before reading small effects from a bfloat16 run**: a null that equals the clean run in bfloat16 certifies the wiring, not small effects. See [ops](docs/usage/ops.md#the-float32-rule).

## Before changing clab

- **Base PRs on `master`; regenerate FORMAT.md with `uv run --no-sync clab describe --markdown` in the code's commit, and change the docs in the same PR.** See [contributing](docs/developing/contributing.md).
- **A format change lands only with before-and-after JSON the owner ratified.** See [contributing](docs/developing/contributing.md#changing-the-format).
- **A check that needs the model belongs in preflight, never in compile; a check from the plan and the engine's declarations alone goes in `start_run`.** See [compile and run](docs/developing/compile-and-run.md#where-a-check-belongs).
- **Leave `tests/conftest.py`'s thread pin alone**: the thread count sets the last bits, and the snapshots were made at the pinned one. See [testing](docs/developing/testing.md#running-the-suite).
