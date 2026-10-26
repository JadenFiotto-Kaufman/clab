---
title: Plan Lifecycle
one_liner: "What happens between a document and its results: Plan, compile with the document alone, preflight with the model, the run inside one session, results and the output directory, and how to tell from a refusal's message which stage raised it."
tags: [concepts, lifecycle, compile, preflight, refusals]
related: [docs/usage/running.md, docs/usage/engines.md, docs/concepts/one-session.md, docs/usage/output.md, docs/errors/index.md, docs/developing/compile-and-run.md]
sources: [clab/plan.py, clab/compile/__init__.py, clab/compile/preflight.py, clab/engines/__init__.py, clab/steps/__init__.py, clab/output/__init__.py]
---

# Plan Lifecycle

## The idea

A document reaches its results through six stages. Each stage knows more than the one before it, decides what that knowledge allows, and refuses whatever it is the first stage able to check. Once you know which stage refused, you know what was wrong and how much had already happened. `clab run` and `clab.run` go through every stage. The calls behind them, for stopping partway in Python, are in [running.md](../usage/running.md#the-four-calls-behind-clabrun); how each stage is built is [developing/compile-and-run.md](../developing/compile-and-run.md).

```text
document ─Plan(path)─▶ plan ─compile()─▶ compiled plan ─engines.load()─▶ engine
   ─execute()─▶ [ preflight ▸ every step ▸ results and run record ] ─dump()─▶ output directory ─Plan.load()─▶ filled plan
```

A plan's `repr` says how far it has gone: `Plan(patching, uncompiled)`, `Plan(patching, compiled, 1 point)`, `Plan(patching, run on nnterp, 2 results)`.

## 1. Plan: the document as records

`Plan(path)` reads the JSON and checks only each record's own shape: its fields and their types, the `pos` grammar, a site's narrowing, step kinds, names without dots. It also refuses `generate` options that would change the rows, such as `num_beams`. Nothing but the document is read.

```text
steps.iia: unknown step kind 'opp'; valid: forward, generate, op, fit
```

## 2. Compile: the document and its files

`plan.compile()` returns a new plan with `compiled` true. Compile knows the document, the data files, hub datasets and bundle files. It never needs the model. It:

- reads every dataset's rows. A file dataset's rows go into the plan. A hub dataset contributes only its digest and row count.
- inlines sites and named interventions.
- loads each bundle a featurizer names and checks its header against the document.
- expands every sweep into points.
- resolves every name, and works out every value's layout, the dataset its rows align to, and which entries the run must take off the model.

[FORMAT.md's derived fields](../../FORMAT.md#derived-fields) lists every field compile fills, and the two the run fills (`results` and `run`). Anything decidable from the document, the data and bundle headers is refused here, by document path:

```text
steps.patched.input: no dataset 'pears'; datasets: pairs
data.pairs.take: take 400, and the dataset has 4 rows; take keeps at most as many as there are
```

Compiling a compiled plan returns it unchanged. Compile runs before any engine loads, so a document it refuses never costs a model load. Under the `ndif` engine, such a document never reaches the server.

## 3. Load: the engine and its model

`engines.load(name, model, device, **options)` builds the engine. On the command line these are `--engine`, `--device` and `-o`. The engine loads the checkpoint the model block pins, or under `ndif` a shell with no weights. It refuses an option it does not declare, a model block key it cannot honor, and a key its loader rejects, each by name ([engines.md](../usage/engines.md)):

```text
the hooks engine takes no 'remote'; its options: device
model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device
```

## 4. Preflight: the model in hand, no forward yet

Preflight runs inside the run, before any forward. It sees the tokenizer, the model's sizes, the components the engine serves, and the order the engine's forward reaches them in. It:

- tokenizes every input and records the tokens in `plan.run.tokens`.
- resolves every prefill position per row.
- turns answer columns into token ids.
- resolves the indices each `select` keeps.
- fixes each featurizer's width.

It refuses by `<step>.<entry>`, or by `<step>` for an op step, without the `steps.` prefix of a document path. Typical causes: a component the engine does not serve, a layer or `select` outside the model's sizes, a position that does not fit some row, an answer that is not one token, an operand read later in the forward than the write that uses it, a bundle at the wrong width or layer.

```text
counterfactual.v_cf: layer 5; the model has 2 layers
patched.patch: row 0: tokens 40 do not fit its 11 scope tokens
counterfactual.v_cf: the hooks engine does not serve attention_queries: nnterp reads it inside a module's forward (source.attention_interface_1.inputs), not at a module boundary
```

A preflight refusal costs a model load and no forward.

## 5. The run: every step, in order

`engine.execute(plan)` first refuses a plan that is not compiled (`plan is not compiled; call plan.compile()`) and a plan whose model block differs from the one the engine loaded. It then hands the plan to `steps.run`. Under nnterp and ndif that call runs inside the engine's one nnsight session; under hooks it is plain Python.

At its start, before preflight, the run puts every dataset's rows in view. Rows that are not in the plan are read again: a hub dataset's always, a file dataset's after `Plan.load`. Rows whose digest differs from the one compile recorded are refused:

```text
data.pairs: data/weekdays/train.json is not what the run read (its rows' digest differs)
```

Each step and each point of it then runs in document order, without gradients except where a fit trains. The run decides whatever needs the model's values: where each generated row stopped, which positions beyond the prefill a read or write lands at, and every op. Few refusals are left this late. One is an option the model's `generate` rejects, which names the step:

```text
answer: generate rejected 'max_new_tokns': The following `model_kwargs` are not used by the model: ['max_new_tokns'] ...
```

## 6. Results, dump and load

After the last step, each saved value goes into `plan.results` as a detached CPU copy, keyed by coordinate when swept. `plan.run` records how the run went: the engine and its options, the versions that served it, the served dtype and devices, every input's tokens and every position each read and write resolved to. `plan.dump(out)` writes one file per save plus `manifest.json`, and `Plan.load(out)` reads them back as the filled plan. [output.md](../usage/output.md) covers reading them.

A loaded plan holds no rows from file datasets. `plan.table` reads them again and refuses if the file changed since the run, with the message shown under stage 5. An output directory is written only when the run completes, so a refusal at any stage leaves none.

## Where each stage runs

Under nnterp and hooks, every stage runs in your process. Under ndif, Plan, compile, load and dump run on the client. Preflight and the run happen on the server, inside the one session, and the results and run record come back together. A refusal from those two stages arrives only after the request has waited in the server's queue. Why this split keeps the numbers the same is [one-session.md](one-session.md).

## Which stage refused

| the message starts with | raised by | what had happened |
|---|---|---|
| a document path: `steps.<step>…`, `data.<name>…`, `saves[i]`, `featurizers.<name>…` | Plan or compile | nothing loaded |
| `the <engine> engine takes no …`, `model.<key>:`, `device:`, `remote:`, `-o …` | load, or the command line before it | nothing loaded |
| a step's name without `steps.`: `<step>.<entry>:`, `<step>:`, `<fit>.<step>.<entry>:` | preflight | the model loaded, no forward ran |
| `<step>: generate rejected '<option>'` | the run | earlier steps ran, nothing was written |
| `data.<name>:` saying the rows are not what compile read | the run, or `plan.table` on a loaded plan | the data changed after compile |

Each refusal, with its cause and fix, is listed by message in [errors/index.md](../errors/index.md).
