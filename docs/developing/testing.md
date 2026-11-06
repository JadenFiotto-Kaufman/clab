---
title: Testing
one_liner: "Running and extending the suite: offline CPU runs at the pinned thread count, snapshot pins and re-pinning, the GPU and torchrun tests, the FORMAT drift test, the docs test, pyright, and the CI workflows that run them."
tags: [developing, testing, snapshots]
related: [docs/developing/contributing.md, docs/developing/extending.md, docs/developing/engines.md]
sources: [tests/conftest.py, tests/test_documents.py, tests/test_plan.py, tests/test_real.py, tests/test_vllm.py, tests/test_format.py, tests/test_docs.py, pyproject.toml, .github/workflows/test.yml, .github/workflows/gpu.yml, documents/real/head_patching.json]
---

# Testing

The suite runs every corpus document on its checkpoint (tiny ones, and GPT-2 for one), offline and on CPU, and pins its numbers per engine; it also holds FORMAT.md and the docs to the code. This page is how to run it, how pins work and are re-pinned, and where a new test goes.

## Running the suite

```bash
uv sync --dev
uv pip install transformers==5.20.2
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/ -q
```

- **`uv sync --dev` needs nnterp's checkout** at the path `[tool.uv.sources]` in `pyproject.toml` names, at the commit [README.md](../../README.md#install) names.
- **Every `uv run` on this page carries `--no-sync`.** The lock resolves a newer transformers than the snapshots' 5.20.2, and a plain `uv run` syncs the venv back to it without a word. Exporting `UV_NO_SYNC=1`, as CI does, has the same effect.
- **CPU and offline by default.** `tests/conftest.py` sets `CUDA_VISIBLE_DEVICES` empty and `HF_HUB_OFFLINE=1` when they are unset. Setting either yourself overrides it; that is how the GPU tests run.
- **The thread count is pinned.** The BLAS splits a matmul's reduction by thread, so the thread count sets the last bits. `tests/conftest.py` pins torch to `THREADS`, the count at which the snapshots and the bit-equality tests were made. Pinned, `OMP_NUM_THREADS` and the host's core count do not decide whether the suite passes. At another thread count, the tests that hold two runs bit-equal (a row against its batch companions, a fit's history) can fail.
- **Floats are compared on the snapshots' CPU vendor or a GPU.** With no CUDA device, on a CPU whose vendor is not `tests/conftest.py` `SNAPSHOTS_VENDOR`, every bit-equality or snapshot comparison of floats skips, naming the vendor (`floats_compared`). CI's "What runs it" step prints the runner's vendor.
- **The snapshots are AVX2 bits.** CI sets `ATEN_CPU_CAPABILITY=avx2`, and so should a local run on a CPU with AVX-512.
- **One file, or tests by name:** `uv run --no-sync pytest tests/test_ops.py -q`, `uv run --no-sync pytest tests/test_ops.py -k clamp -q`. In the files that run the corpus, a module-scoped fixture (`filled`, `ran`) runs all of the file's documents for the first selected test that asks for it, so selecting one such test still runs them all.
- **Checkpoints and datasets** come from the local Hugging Face cache, at the revisions below. On a fresh cache, run once with `HF_HUB_OFFLINE=0` and hub access; `tests/test_hub.py`'s docstring has the command that fetches its dataset.
- **Each cached repository's `main` must be its pinned revision.** The meta load the hooks engine makes reads the tokenizer at `main`, whatever revision it is given. Write the revision into the repository's `refs/main` in the hub cache, as CI's "Fetch the checkpoints" step in `.github/workflows/test.yml` does.

`.github/workflows/test.yml` reads this table. It fetches each row at its revision: not a row that says `gated`; the configs named after `configs` where it says `a dataset`; `*.json` and `*.txt` alone where it says `tokenizer only`. It keys its cache by a hash of the rows. Keep the row shape and those words, or change `test.yml` in the same pull request; any edit to a row re-keys CI's cache.

| repository | revision | read by |
|---|---|---|
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | `9fb191250dd56d0ba7ec9785a025ed29c03d5998` | the corpus |
| `hf-internal-testing/tiny-random-gpt2` | `71034c5d8bde858ff824298bdedc65515b97d2b9` | `documents/gpt2_reach.json` and the other byte-level BPE documents and tests |
| `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | `documents/probe_direction.json`, the one corpus document on a trained model's weights |
| `trl-internal-testing/tiny-Qwen3ForCausalLM` | `25bb9b8963688237e94750f6f3e48f1947d3c7a3` | chat-template tests in `tests/test_positions.py` and `tests/test_findings.py` |
| `nyu-mll/glue`, a dataset, configs `wnli` and `ax` | `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | `tests/test_hub.py` |
| `Qwen/Qwen3-8B`, its tokenizer only | `b968826d9c46dd6066d109eabc6255188de91218` | one chat-template test in `tests/test_positions.py`, which skips without it |
| `hf-internal-testing/tiny-random-Gemma2ForCausalLM` | `de7c11b6c25d26ddd1bf4324fcf479b61d18e440` | the one tiny model that softcaps its logits: `tests/test_engines.py` and `tests/test_documents.py` |
| `yujiepan/qwen3-next-moe-tiny-random` | `ec636a93e953ed50d2ecebda99b04673b5f1e248` | a hybrid whose blocks are of two kinds: `tests/test_engines.py`, and `tests/test_tp.py` where it is cached |
| `trl-internal-testing/tiny-CohereForCausalLM` | `90bc56bd12c642f6bf5e710ebd7c626dd7c0201c` | one test in `tests/test_engines.py` (the readouts differing on a model that scales its logits), which skips without it |
| `yujiepan/llama-3.2-tiny-random` | `0bf108f80b7c6cdf6e7e85698f0464bfbf7f9ce5` | one test in `tests/test_plan.py` (a chat template that reads the clock), which skips without it |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | `ccb12fe2fc142cb752085506c3db22572290e90c` | `documents/expert_knockout.json`, `tests/test_moe.py` (as the next four are: tiny mixtures of experts, 4 experts routing each token to 2) and one test in `tests/test_tp.py` |
| `hf-internal-testing/tiny-random-OlmoeForCausalLM` | `99e4bc8eb1a024e46da438c410d7e707ed49d15a` | a softmax router that does not renormalize its top 2 |
| `trl-internal-testing/tiny-Qwen3MoeForCausalLM` | `6db57163fa56fa0e00d4b820169908ac0d3a219d` | a softmax router that renormalizes |
| `trl-internal-testing/tiny-Glm4MoeForCausalLM` | `19b382257595016963b60289ba295e33abad6834` | a dense block 0, and at block 1 a sigmoid router with a shared expert |
| `hf-tiny-v2/tiny-random-JetMoeForCausalLM` | `236ca328ccfedeb09989bfb2188b6fbb0cae5afc` | a router that computes the indices before the weights, whose logits are a module's output |
| `meta-llama/Llama-3.2-1B`, gated | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | `documents/real/`, through `tests/test_real.py` on a GPU, and the GPU case of `tests/test_quantized.py` |

The suite runs every document at `tests/conftest.py` `BATCH_SIZE` (2) and `THREADS`, and every pin of results records the batch size it was made at. Why both matter is [running.md](../usage/running.md#when-two-runs-give-the-same-numbers).

## Fixtures and helpers

`tests/conftest.py` holds what the files share:

- `engines`: each engine loaded once per model block and options for the whole session, `engines("nnterp", plan.model)`.
- `corpus`: a copy of `documents/`, its `data/` linked, whose `documents/artifacts/` holds stand-in bundles for the documents that load what another writes (`fit.rot`, `basis`, `fit.dictionary`, `fit.lora`). The docs test compiles fenced documents against it.
- `stand_in`: a plan run on the ndif engine against a stand-in server. The server serializes the real request, runs it with a loaded model, and returns what came home and what the trip carried (`by_reference`; `TRIP`: the results, the run record and the completed records).
- `tests/fixtures/sae.py` and `tests/fixtures/published_sae.py` write the committed SAE fixtures under `documents/artifacts/` from seeded generators, the same bytes on every run; rerun one from the repo root (`uv run --no-sync python tests/fixtures/sae.py`) only to change a fixture.
- Helpers: `plan_of(name)` compiles a corpus document, and `edited(name, change)` is one with `change` applied to its JSON, read as if from its own file. `on_each` runs a plan on each engine and keeps each run's results apart. `placed` is where every read and write resolved, per row, as the records hold it. `close` compares floats to 1e-6 and NaN to NaN; `within(a, b, ulps)` is equality but for floats within `ulps` float32 ulps (0: bit-equal); `same` is bit equality.

## Snapshot pins

`tests/conftest.py` `snapshot(name, value, holds)` returns the checked-in JSON under `tests/snapshots/`; with `CLAB_UPDATE_SNAPSHOTS=1` set, it writes `value` there first, unless `holds(checked_in)` says the checked-in one already holds it.

| snapshot, under `tests/snapshots/` | pinned by | holds, compared how |
|---|---|---|
| `<document>.<engine>.json` | `tests/test_documents.py` (forward documents), `tests/test_generate.py`, `tests/test_fit.py`, `tests/test_patching.py` | the saved results at `BATCH_SIZE`, to 1e-6; `softcapped_lens.*` also to one float32 ulp, its known drift between hosts (`DRIFT` in `tests/test_documents.py` says what it is) |
| `das.batch_size_1.json`, `dbm.units.json`, `head_gates.heads.json` | `tests/test_fit.py` | a fit's history at batch size 1, to 1e-6; the units and heads each L1 weight keeps, exactly |
| `plans/<document>.json` | `tests/test_plan.py` | the compiled plan, rows aside (every record's layout and `axes`, and which point each forward serves), exactly |
| `real/<document>.<engine>.json` | `tests/test_real.py` | GPU results to 1e-6, with the GPU, torch and transformers that made them (`made_with`), a record that is not compared |

Which file runs a document is decided in `tests/conftest.py`. One named in `FIT` (it trains or loads a featurizer) runs in `tests/test_fit.py`, one in `GENERATE` in `tests/test_generate.py`, and every other in `tests/test_documents.py`. Each runs on the nnterp and hooks engines; the hooks engine skips those in `INSIDE`, which it refuses. Beside the pins, the files check that the engines are bit-equal (`same`, and the same `placed`), and that the ndif engine under `remote="local"` equals the nnterp engine.

To re-pin after a change that means to move numbers, rerun the files whose pins move with the variable set, then read what changed:

```bash
CLAB_UPDATE_SNAPSHOTS=1 uv run --no-sync pytest tests/test_documents.py -q
git diff tests/snapshots/
```

The update writes every snapshot the run reaches and then passes, whatever moved, so the diff is the review: every changed number should be one the change was meant to change. Two kinds of file are left as they are where nothing moved past what they allow. One is `softcapped_lens.*`, whose floats are within one ulp of the run's. The other is the `real/` pins, whose values are bit-equal to the run's whatever their `made_with` says. A value that moved rewrites the file, `made_with` with it.

A new corpus document has no pins yet, so its tests fail with `FileNotFoundError` until this is run for it; the README's corpus count, which `tests/test_plan.py` checks, changes with it.

## GPU real-model tests

`documents/real/` holds experiments on `meta-llama/Llama-3.2-1B`, a gated checkpoint, in float32, over the 42 weekday pairs of `data/weekdays_full`. Each document's `description` states its finding, and `tests/test_real.py` checks it:

| document | what it finds |
|---|---|
| `documents/real/behavior.json` | the baseline: the model answers every prompt |
| `documents/real/layer_sweep.json` | where the day is, block by block, at its own token and at the last |
| `documents/real/head_patching.json` | each head of layers 10 to 12, one at a time: layer 12's head 28 alone moves the answer |
| `documents/real/attribution_patching.json` | every head of every layer at once by attribution patching: the same head comes first |
| `documents/real/pattern_patching.json` | that layer's attention patterns, which move nothing |
| `documents/real/das.json` | a trained subspace at layer 8, its held-out accuracy by rank |

They read the corpus's `data/` and want a GPU:

```text
clab run documents/real/head_patching.json --device cuda --batch-size 16
clab run documents/real/das.json --engine hooks --device cuda --batch-size 16
```

- **`pattern_patching.json` needs eager.** It reads the attention pattern inside the attention's forward, so its model block gives `"attn_implementation": "eager"`. The hooks engine refuses it by name, and so does an NDIF deployment that runs sdpa.
- **Through `--engine ndif`**, the server must serve the checkpoint at the document's revision, and in float32 for the numbers to be the ones the descriptions state ([engines.md](../usage/engines.md#what-the-server-must-have)).

`tests/test_real.py` runs these documents on the nnterp and hooks engines on `cuda` at batch size 16: its pins, the engines' agreement, and the findings each description states. It skips without a CUDA device and the checkpoint at the revision it pins in the local cache. It runs from a venv whose torch works with your CUDA driver, with clab installed without dependencies, from the repo root:

```bash
uv venv --python <a python whose torch runs on your driver> --system-site-packages <venv>
uv pip install --python <venv>/bin/python --no-deps -e .
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. <venv>/bin/python -m pytest tests/test_real.py -q
```

The other GPU tests skip under the suite's default of no CUDA device:

- `tests/test_device.py`: on a GPU, every tensor clab builds meets an activation on its device;
- the device test in `tests/test_featurizers.py`, the CUDA and Llama-3.2-1B cases of `tests/test_quantized.py`, and the two-GPU cases of `tests/test_tp.py`;
- `tests/test_vllm.py`, which also needs `vllm` importable beside nnsight and nnterp, and `HuggingFaceTB/SmolLM2-135M-Instruct` at the revision it pins in the cache. It holds the vllm engine bit-equal to plain nnsight on `StandardizedVLLM`, and checks its nulls, repeats, generation, readouts, refusals and run record. It holds the nnterp engine's numbers to vLLM's rounding. Run it with the command above, `tests/test_vllm.py` in place of `tests/test_real.py`, from a venv that has vLLM.

The rest of `tests/test_tp.py` starts its two ranks with torchrun on the CPU, in the suite. The tiny-Llama cases of `tests/test_quantized.py` (`nf4`, `fp4`, `int8`) run on the CPU with the dev group's `bitsandbytes`.

## The FORMAT drift test

`tests/test_format.py` fails when the committed FORMAT.md differs from what `clab/docs.py` `markdown` renders from the code, and when an engine option has no docstring line saying what it takes. Fix the first by regenerating it with `uv run --no-sync clab describe --markdown`, as [contributing.md](contributing.md#changing-the-format) says.

## The docs test

```bash
uv run --no-sync pytest tests/test_docs.py -q
```

It runs offline, with no model. It checks every page under `docs/` and the root guides:

- relative links and `#anchors` resolve inside the repo;
- front matter has `title`, `one_liner` and `tags`, and its `related` and `sources` exist;
- a repo path in backticks exists, and a symbol cited beside its `.py` path is defined there;
- nothing is cited by line number;
- no machine names, user names, home paths or temporary planning files appear;
- every fenced `json` block holding both `model` and `steps` compiles against the `corpus` fixture.

The standard it enforces is in [contributing.md](contributing.md#docs).

## pyright

```bash
uvx pyright
```

From the repo root; it reads `[tool.pyright]` in `pyproject.toml`, checks `clab/` and `tests/`, and runs clean.

## Continuous integration

`.github/workflows/test.yml` runs on every push to `master` and every pull request, on a GitHub-hosted `ubuntu-latest` runner (4 CPUs) with Python 3.12, within 20 minutes:

1. It checks clab out, and `ndif-team/nnterp` beside it at the commit the README names. It points the runner's copy of `pyproject.toml` at that checkout (the nnterp source and pyright's `extraPaths`). uv reads a path source from `pyproject.toml` alone: a `uv.toml` may not hold `sources`, and no environment variable sets one.
2. `uv sync --dev`, then `uv pip install transformers==5.20.2`, the transformers the snapshots were made with. `UV_NO_SYNC=1` keeps `uv run` from syncing back to the newest.
3. `uvx pyright`, then `tests/test_docs.py` and `tests/test_format.py`.
4. The checkpoints and the dataset of the table above, fetched with hub access where the cache lacks them, each repository's `main` set to its pinned revision. They are cached under a hash of the table's rows, so a new row fetches what is new. The gated checkpoint is not fetched, so the test of its tensor-parallel plan in `tests/test_tp.py` skips.
5. The suite, offline, at the pinned `THREADS`, with `ATEN_CPU_CAPABILITY=avx2` ([Running the suite](#running-the-suite)).

`.github/workflows/gpu.yml` runs `tests/test_real.py`, `tests/test_device.py` and `tests/test_vllm.py` (which skips where the runner's python has no vLLM) on a self-hosted runner labelled `gpu`, with the commands of [GPU real-model tests](#gpu-real-model-tests). It runs only when dispatched by hand with the repository variable `GPU_RUNNER` set to `true`; otherwise run the files by hand.

## Where a test goes

| file | covers |
|---|---|
| `tests/test_plan.py` | each document compiles to its snapshot; the plan as pure data; errors name the document path |
| `tests/test_engines.py` | a component means one thing on every engine; each engine's readout and its refusals layer by layer |
| `tests/test_moe.py` | a mixture of experts' components on the tiny mixtures: their layouts and identities, usage, the knockout and its null, what each engine serves |
| `tests/test_shared.py` | one forward shared by the points that differ only in what they read, and the schedule the engines execute |
| `tests/test_efficiency.py` | what a run costs (`run.costs`), and where a forward ends |
| `tests/test_refusals.py` | what cannot be served, or would change a document's meaning, refused by name before any forward |
| `tests/test_findings.py` | one test per silent failure mode or unclear refusal, each pinned as a refusal or a check, named by a short tag |
| `tests/test_documents.py` | forward documents on each engine; what padding, positions, scopes, windows, ragged reads and `layers` mean |
| `tests/test_padding.py` | every corpus document under `batch` and `sorted` padding: close to the exact run at a stated tolerance, the same rows missing and positions, the engines bit-equal, rows put back in order |
| `tests/test_patching.py` | `documents/patching.json` end to end, on every engine, and its output |
| `tests/test_generate.py` | generation, and what a position means across calls |
| `tests/test_fit.py` | the documents that train or load a featurizer, from `clab run` on every engine |
| `tests/test_positions.py` | every input token a position, numbered one way everywhere |
| `tests/test_answers.py` | answer columns lined up with reads: flags, sets, `before` scopes, list inputs |
| `tests/test_layers.py` | writes across layers, operands aligned by layer label, sweep labels |
| `tests/test_ops.py` | each op on plain tensors, and the machinery that runs a site's records |
| `tests/test_grad.py` | a gradient as a value: the central difference of the metric, the engines and a server alike, a constant's exactly zero, refusals by name |
| `tests/test_featurizers.py`, `tests/test_featurizer_ops.py` | featurizer math on any device and under autocast, published SAEs; `encode`, `decode` and `apply` per kind |
| `tests/test_lora.py` | the `lora` kind |
| `tests/test_hub.py` | hub datasets held by reference |
| `tests/test_load.py` | a value one run saved, loaded into another document: the numbers of the document that made it, bit-equal, on each engine, under `remote="local"` and through the stand-in server (which the value reaches inside the plan); what a load refuses |
| `tests/test_quantized.py` | quantized `dtype`s on both engines |
| `tests/test_tp.py` | the nnterp engine's `tp` under torchrun: its refusals, what every rank holds whole, rank 0 writing |
| `tests/test_device.py` | on a GPU, every tensor clab builds meets an activation on its device |
| `tests/test_vllm.py` | on a GPU with vLLM, the vllm engine is bit-equal to plain nnsight on `StandardizedVLLM`, exact in its nulls and repeats, and refuses what vLLM cannot do by name |
| `tests/test_cli.py` | `clab compile`, `clab show` and `clab table`, and what `clab run` prints and refuses |
| `tests/test_table.py` | `table`: values joined on the keys they share, and what has no rows to table |

What to add for a new op, kind, component or step kind is in [extending.md](extending.md).
