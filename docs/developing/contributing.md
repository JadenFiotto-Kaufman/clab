---
title: Contributing
one_liner: "How changes land: branch and commit conventions, format changes with before-and-after JSON, one spelling per thing, the corpus runs unchanged, regenerating FORMAT.md, and the docs standard."
tags: [developing, contributing, docs-standard]
related: [docs/developing/testing.md, docs/developing/extending.md, docs/developing/architecture.md, docs/index.md]
sources: [FORMAT.md, clab/docs.py, clab/cli.py, tests/test_format.py, tests/test_docs.py, pyproject.toml]
---

# Contributing

How a change to clab lands: the branch and commit conventions, the rules a change to the format follows, what to run before a pull request, and the standard the docs are held to.

## Branches and commits

- Work on a branch and open the pull request into `master`.
- A commit message is one plain declarative sentence saying what is true after the commit, in the voice of the test names: `A ragged write lands only on each row's own tokens.`

## Changing the format

The format is everything a document can write: the records and their fields, the step kinds, the ops and their arguments, the components, the featurizer kinds and their fields, the engines' options. [FORMAT.md](../../FORMAT.md) lists all of it.

- **Proposed with before-and-after JSON.** A format change is proposed before it is built, as the same document fragment written today and as it would be written after the change. It lands only when the repository's owner ratifies it.
- **One spelling per thing.** No aliases, and no old spelling kept beside a new one. A spelling the format does not have is refused by name with the one to write instead. `reads` and `writes` on a step are refused because a step's reads and writes are one `interventions` value; a model block's `torch_dtype` is refused with ``the dtype is spelled `dtype` ``.
- **The corpus runs unchanged.** The documents under `documents/` are the format's semantics, and their pinned numbers say what each means ([testing.md](testing.md)). A change never edits a corpus document to make it pass. A change that would need one rewritten is a format change; a pin that moves is a change of meaning, made on purpose and visible in the snapshot diff. New behavior gets a new document or a new test.
- **FORMAT.md is regenerated in the same commit.** It is rendered from the code's docstrings and registries (`clab/docs.py` `markdown`), and `tests/test_format.py` fails when it drifts. Run `uv run --no-sync clab describe --markdown`. It writes `FORMAT.md` at the repo root itself, from any directory, and prints nothing (`--verbose` prints the path it wrote), so nothing is redirected into the file.

Adding an op, a featurizer kind, a component or a step kind is [extending.md](extending.md).

## Before a pull request

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/ -q   # after testing.md's install
uv run --no-sync clab describe --markdown        # when the format changed; commit FORMAT.md with the code
uvx pyright
```

`--no-sync` keeps `uv run` from syncing transformers past the one the snapshots were made with ([testing.md](testing.md#running-the-suite)).

These need GPUs and skip without them: `tests/test_real.py`, `tests/test_device.py`, `tests/test_vllm.py` (with vLLM installed), the device test in `tests/test_featurizers.py`, the CUDA cases of `tests/test_quantized.py` and the two-GPU cases of `tests/test_tp.py`. Run them on one when a change touches an engine or where tensors are placed ([testing.md](testing.md#gpu-real-model-tests)). Running, pinning and re-pinning are [testing.md](testing.md).

## What a pull request carries

- The code, and the tests that pin it ([extending.md](extending.md) lists them per kind of change).
- `FORMAT.md` regenerated, when the format changed, and for a format change the ratified before-and-after JSON in the description.
- Every docs page the change makes untrue, updated in the same pull request: docs change with the code that invalidates them.
- Re-pinned snapshots only for numbers the change means to move, each move said in the description.

## Docs

A page is for a reader, often an agent, who finds the one page for a task and applies it alone. Each change keeps these rules:

- **Findable.** `title` and `one_liner` say what the page answers, in a searcher's words. A new page gets a line in its section's `index.md`, and index pages only route.
- **One home per topic.** `usage/` owns a part of the format or a run stage, `patterns/` a technique, `concepts/` a mechanism, `errors/` a refusal or a symptom, `developing/` the code. Another page gives one sentence of context and a link. `CLAUDE.md` holds the by-task routes and the mistakes that run without a refusal, one line and a link each.
- **Routed.** Every op, featurizer kind, component, step kind and run option has a when-to-use sentence on the page that owns it. It also has a route from `CLAUDE.md` or a pattern page.
- **One rule per sentence**, under 25 words and never over 40. A recipe is a table row or a bullet; a trap is a bullet with a bold lead-in. No paragraph passes about 120 words.
- **Rules, not runs.** A walkthrough keeps only its rule, unless the number is the point. One example per rule: a corpus document by path, the fragment that matters quoted.
- **Numbers from the repository**: a snapshot, a test, a corpus document or its `clab show`. A figure from one machine, one card or one real model goes, and its mechanism stays.
- **"To the bit" has one home**, [usage/running.md](../usage/running.md#when-two-runs-give-the-same-numbers). Elsewhere say "equal" and link it; a test's assertion is "bit-equal" (`same`).
- **Current and general.** Present tense: no history, roadmap, version pins or issue references. No host or user names, home paths, one deployment, one CPU vendor's kernels or one card's timings.
- **FORMAT.md is the spec.** Link its section and add the task, the choice, the recipe or the trap. Full lists and counts live there alone. Use its spellings, one per thing, and quote a refusal by its first clause.
- **Own words, cited.** Never paste a docstring. Cite code by path and symbol (`clab/compile/positions.py` `resolve`), never by line number.
- **Examples compile.** A fenced `json` block with `model` and `steps` must compile against the corpus; a fragment omits `model`.
- **Front matter**: `title`, `one_liner` (quoted when it holds `: `), `tags`, and `related` and `sources` naming files that exist. Leaf pages run 60 to 200 lines.

`uv run --no-sync pytest tests/test_docs.py -q` checks links, anchors, front matter, cited paths and symbols, line-number citations, banned names and compiling examples ([testing.md](testing.md#the-docs-test)); the rest is review.
