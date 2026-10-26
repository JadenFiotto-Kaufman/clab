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
- **One spelling per thing.** No aliases, and no old spelling kept beside a new one. A spelling the format does not have is refused by name with the one to write instead: `reads` and `writes` on a step are refused because a step's reads and writes are one `interventions` value, and a model block's `torch_dtype` with ``the dtype is spelled `dtype` ``.
- **The corpus runs unchanged.** The documents under `documents/` are the format's semantics, and their pinned numbers say what each means ([testing.md](testing.md)). A change never edits a corpus document to make it pass. A change that would need one rewritten is a format change; a pin that moves is a change of meaning, made on purpose and visible in the snapshot diff. New behavior gets a new document or a new test.
- **FORMAT.md is regenerated in the same commit.** It is rendered from the code's docstrings and registries (`clab/docs.py` `markdown`), and `tests/test_format.py` fails when it drifts. Run `uv run clab describe --markdown` from the repo root: it writes `FORMAT.md` into the current directory itself, so do not redirect its output into the file, which overwrites the file's first line with the command's message.

Adding an op, a featurizer kind, a component or a step kind is [extending.md](extending.md).

## Before a pull request

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run pytest tests/ -q
uv run clab describe --markdown        # when the format changed; commit FORMAT.md with the code
uvx pyright
```

`tests/test_real.py` and `tests/test_device.py` need a GPU and skip without one; run them on one when a change touches an engine or where tensors are placed. Running, pinning and re-pinning are [testing.md](testing.md).

## What a pull request carries

- The code, and the tests that pin it ([extending.md](extending.md) lists them per kind of change).
- `FORMAT.md` regenerated, when the format changed, and for a format change the ratified before-and-after JSON in the description.
- Every docs page the change makes untrue, updated in the same pull request: docs change with the code that invalidates them.
- Re-pinned snapshots only for numbers the change means to move, each move said in the description.

## Docs

Docs are written for a reader, often an agent with clab installed, who finds the one page for a task and applies it without reading anything else. Every page is held to three requirements:

1. **Useful and findable.** A page answers a question someone has while using or developing clab; its title and `one_liner` say what it answers in the words a searcher would use. One topic has one home: a page that needs a topic another page owns gives one sentence of context and links it. Index pages route and do not teach, and a new page gets a line in its section's `index.md`.
2. **Compact.** State the rule, show it once, stop: no motivational introductions, no summaries that repeat the page, no enumeration where a pattern and one example suffice. Explain in your own words, completely enough that the reader does not need the source to finish the task, and never mirror the source (no attribute dumps, re-listed enums or walkthroughs of implementation lines).
3. **Nothing environment-specific.** No machine or host names, user names, home paths, personal environments or one operator's deployment details. Generalize the requirement, or leave it out.

And the rules particular to clab's docs:

- **FORMAT.md is the spec.** Link its section (`../../FORMAT.md#ops`) and add what it cannot say: the task, the choice, the recipe, the trap. Never restate its fields or ops.
- **Current truth only.** Present tense; no version pins, no history, no bug or issue references, no roadmap. Odd behavior is stated neutrally, with how to work with it.
- **Cite by path and symbol**, in backticks: `clab/compile/positions.py` `landing`. Never by line number.
- **Examples compile.** A fenced `json` document with both `model` and `steps` is compiled by the docs test, so it uses the corpus's pinned tiny checkpoint and datasets under `data/`; anything else is a fragment, without `model`. Prefer pointing at a corpus document by path and quoting the part that matters.
- **Front matter** on every page: `title`, `one_liner` (quoted when it holds `: `), `tags`, and `related` and `sources` naming repo files that exist. Leaf pages run 60 to 200 lines.

`uv run pytest tests/test_docs.py -q` checks links, anchors, front matter, cited paths and symbols, line-number citations, banned names and that complete documents compile; the rest is review.
