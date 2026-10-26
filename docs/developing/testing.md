---
title: Testing
one_liner: "Running and extending the suite: offline CPU runs, snapshot pins and re-pinning, the GPU real-model tests, the FORMAT drift test, the docs test, and pyright."
tags: [developing, testing, snapshots]
related: [docs/developing/contributing.md, docs/developing/extending.md, docs/developing/engines.md]
sources: [tests/conftest.py, tests/test_documents.py, tests/test_plan.py, tests/test_real.py, tests/test_format.py, tests/test_docs.py, pyproject.toml]
---

# Testing

The suite runs every corpus document on tiny checkpoints, offline and on CPU, and pins its numbers per engine; it also holds FORMAT.md and the docs to the code. This page is how to run it, how pins work and are re-pinned, and where a new test goes.

## Running the suite

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run pytest tests/ -q
```

- `tests/conftest.py` sets `CUDA_VISIBLE_DEVICES` empty and `HF_HUB_OFFLINE=1` when they are unset, so a plain `uv run pytest tests/` already runs on CPU and offline. Setting either yourself overrides it; that is how the GPU tests run.
- Run the suite at the machine's default thread count. Two pins depend on it: forcing `OMP_NUM_THREADS` to 2 fails the batch-companion test, and forcing 1 moves a fit's history pin past its tolerance. Cap threads for single documents, not for the suite.
- Checkpoints and datasets come from the local Hugging Face cache: the tiny Llama and tiny GPT-2 the corpus pins, a tiny Qwen3 for chat templates (`tests/test_positions.py`), and the hub fixture `tests/test_hub.py` reads, whose docstring has the command that fetches it. On a fresh cache, run once with `HF_HUB_OFFLINE=0` and hub access.
- One file, or tests by name: `uv run pytest tests/test_ops.py -q`, `uv run pytest tests/test_ops.py -k clamp -q`. In the files that run the corpus, a module-scoped fixture (`filled`, `ran`) runs all of the file's documents for the first selected test that asks for it, so selecting one such test still runs them all.

Results are reproducible to the bit only at a fixed batch size, so the suite runs every document at `tests/conftest.py` `BATCH_SIZE` (2), and every pin of results records the batch size it was made at.

## Fixtures and helpers

`tests/conftest.py` holds what the files share:

- `engines`: each engine loaded once per model block and options for the whole session, `engines("nnterp", plan.model)`.
- `corpus`: a copy of `documents/`, its `data/` linked, whose `documents/artifacts/` holds stand-in bundles for the documents that load what another writes (`fit.rot`, `basis`, `fit.dictionary`, `fit.lora`). The docs test compiles fenced documents against it.
- `stand_in`: a plan run on the ndif engine against a stand-in server, which serializes the real request, runs it with a loaded model and returns what came home and what the trip carried (`by_reference`, `TRIP`).
- Helpers: `plan_of(name)` compiles a corpus document; `edited(name, change)` is one with `change` applied to its JSON, read as if from its own file; `close` compares floats to 1e-6 and NaN to NaN; `same` is bit equality.

## Snapshot pins

`tests/conftest.py` `snapshot(name, value)` returns the checked-in JSON under `tests/snapshots/`; with `CLAB_UPDATE_SNAPSHOTS=1` set, it writes `value` there first.

| snapshot, under `tests/snapshots/` | pinned by | holds, compared how |
|---|---|---|
| `<document>.<engine>.json` | `tests/test_documents.py` (forward documents), `tests/test_generate.py`, `tests/test_fit.py`, `tests/test_patching.py` | the saved results at `BATCH_SIZE`, to 1e-6 |
| `plans/<document>.json` | `tests/test_plan.py` | the compiled plan, rows aside, exactly |
| `real/<document>.<engine>.json` | `tests/test_real.py` | GPU results, with the GPU, torch and transformers that made them |

Which file runs a document is decided in `tests/conftest.py`: one named in `FIT` (trains or loads a featurizer) runs in `tests/test_fit.py`, one in `GENERATE` in `tests/test_generate.py`, and every other in `tests/test_documents.py`, on the nnterp and hooks engines (the hooks engine skips those in `INSIDE`, which it refuses). Beside the pins, the files check that the engines agree to the bit (`same`, and the same `run.positions`) and that the ndif engine under `remote="local"` equals the nnterp engine.

To re-pin after a change that means to move numbers, rerun the files whose pins move with the variable set, then read what changed:

```bash
CLAB_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_documents.py -q
git diff tests/snapshots/
```

The update writes every snapshot the run reaches and then passes, whatever moved, so the diff is the review: every changed number should be one the change was meant to change. A new corpus document has no pins yet, so its tests fail with `FileNotFoundError` until this is run for it; the corpus count in `tests/test_plan.py` changes with it.

## GPU real-model tests

`tests/test_real.py` runs the documents under `documents/real/` (Llama-3.2-1B in float32) on the nnterp and hooks engines on `cuda` at batch size 16: its pins, the engines' agreement, and the findings each document's description states. It skips without a CUDA device and the checkpoint, a gated repository, at the revision the file pins in the local cache. It runs from a venv whose torch works with your CUDA driver, with clab installed without dependencies, from the repo root:

```bash
uv venv --python <a python whose torch runs on your driver> --system-site-packages <venv>
uv pip install --python <venv>/bin/python --no-deps -e .
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. <venv>/bin/python -m pytest tests/test_real.py -q
```

`tests/test_device.py` and the device test in `tests/test_featurizers.py` also need a CUDA device, and skip under the suite's default of none.

## The FORMAT drift test

`tests/test_format.py` fails when the committed FORMAT.md differs from what `clab/docs.py` `markdown` renders now, and when an engine option has no docstring line saying what it takes. Fix the first by regenerating it with `uv run clab describe --markdown` from the repo root, as [contributing.md](contributing.md#changing-the-format) says.

## The docs test

```bash
uv run pytest tests/test_docs.py -q
```

Offline, no model, a few seconds. It checks every page under `docs/` and the root guides: relative links and `#anchors` resolve inside the repo; front matter has `title`, `one_liner` and `tags`, and its `related` and `sources` exist; a repo path in backticks exists, and a symbol cited beside its `.py` path is defined there; nothing is cited by line number; no machine names, user names, home paths or temporary planning files appear; and every fenced `json` block holding both `model` and `steps` compiles against the `corpus` fixture. The standard it enforces is in [contributing.md](contributing.md).

## pyright

```bash
uvx pyright
```

From the repo root; it reads `[tool.pyright]` in `pyproject.toml`, checks `clab/` and `tests/`, and runs clean.

## Where a test goes

| file | covers |
|---|---|
| `tests/test_plan.py` | each document compiles to its snapshot; the plan as pure data; errors name the document path |
| `tests/test_refusals.py` | what cannot be served, or would change a document's meaning, refused by name before any forward |
| `tests/test_findings.py` | one test per silent failure mode or unclear refusal, each pinned as a refusal or a check, named by a short tag |
| `tests/test_documents.py` | forward documents on each engine; what padding, positions, scopes, windows, ragged reads and `layers` mean |
| `tests/test_patching.py` | `documents/patching.json` end to end, on every engine, and its output |
| `tests/test_generate.py` | generation, and what a position means across calls |
| `tests/test_fit.py` | the documents that train or load a featurizer, from `clab run` on every engine |
| `tests/test_positions.py` | every input token a position, numbered one way everywhere |
| `tests/test_answers.py` | answer columns lined up with reads: flags, sets, `before` scopes, list inputs |
| `tests/test_layers.py` | writes across layers, operands aligned by layer label, sweep labels |
| `tests/test_ops.py` | each op on plain tensors, and the machinery that runs a site's records |
| `tests/test_featurizers.py`, `tests/test_featurizer_ops.py` | featurizer math on any device and under autocast, published SAEs; `encode`, `decode` and `apply` per kind |
| `tests/test_lora.py` | the `lora` kind |
| `tests/test_hub.py` | hub datasets held by reference |
| `tests/test_device.py` | on a GPU, every tensor clab builds meets an activation on its device |
| `tests/test_cli.py` | `clab show` and `clab table` |

What to add for a new op, kind, component or step kind is in [extending.md](extending.md).
