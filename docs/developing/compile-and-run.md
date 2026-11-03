---
title: Compile and Run
one_liner: "The two passes and the run: compile's checks with the document alone, preflight's with the model, the schedule each forward runs, how steps.run walks a plan, and what belongs to the engine."
tags: [developing, compile, preflight, run]
related: [docs/developing/architecture.md, docs/developing/engines.md, docs/concepts/plan-lifecycle.md, docs/errors/index.md]
sources: [clab/compile/__init__.py, clab/compile/preflight.py, clab/compile/schedule.py, clab/compile/state.py, clab/compile/interventions.py, clab/steps/__init__.py, clab/steps/batches.py, clab/steps/generate.py, clab/steps/fit.py, clab/plan.py]
---

# Compile and Run

A document is checked twice before anything runs: compile, with the document, its data files and bundle headers alone; and preflight, inside the run, with the model in hand. Then `steps.run` walks the plan step by step. This page says where each piece of that lives, so a new check or a new step lands in the right place. The stages from the user's side, and which refuses what, are [concepts/plan-lifecycle.md](../concepts/plan-lifecycle.md).

## Where a check belongs

Decide everything as early as the information allows.

| what it needs | where it runs | how it refuses |
|---|---|---|
| a record's own shape: fields, types, an op's arguments against its signature, a featurizer's fields | building the plan: the validators in `clab/plan.py` | `PlanError`, starting with the document path |
| the document, the data files, bundle headers | compile, `clab/compile/` | `fail(path, message)` in `clab/plan.py`: a `PlanError` starting with the document path |
| the tokenizer, a size of the model, what the engine serves, forward order | preflight, `clab/compile/preflight.py` (the schedule's sites, `clab/compile/schedule.py`) | `ValueError` starting with `<step>.<entry>` |

Two refusals of the patching document, one from each pass:

```text
steps.iia.of: 'patched.nothing' names no value produced before this step; known: counterfactual.v_cf, patched.logits, patched.patch
counterfactual.v_cf: layer 5; the model has 2 layers
```

The first is compile's: `of` names a value no step produced, and the message starts with the document path. The second is preflight's: `"layers": 5` compiles, because compile does not know the model has two layers, and the refusal names the entry. A refusal inside a fit's body is prefixed with the fit's name; one in an eval names its eval step (`fit.held_out.adapted.adapt: …`), and compile's names the eval (`steps.fit.eval.held_out: …`). `PlanError` is a `ValueError`, so a caller catches both as one, and `clab run` prints either as `clab: <message>` and exits 1.

## Compile

`clab/compile/__init__.py` `compile` returns a new plan with `compiled` true. In order:

1. Each dataset's rows are made once (`make_dataset_rows`: a file's rows in its `split`, inline rows, or a hub split; then `shuffle`, `take` and `columns`), and their digest and row count recorded. A hub dataset's rows are checked and left out of the plan; the run loads them again where it runs.
2. Sweeps are found, and each featurizer, then each step, is compiled at every point of the axes it uses or inherits from what it references (`expand`). A featurizer's bundle is loaded and checked key by key against the document (`compile_featurizer`).
3. Steps compile in document order (`compile_step` dispatches by kind). The records compiled so far are the symbol table (`Compiling.steps` and `Compiling.featurizers`, `clab/compile/state.py` `Compiling`): a later reference resolves to a record at the consumer's point (`clab/plan.py` `record_of`), and a name is referenced only after it exists. A fit's body compiles in a scope that holds the fit, each trained featurizer's bare name written `<fit>.<featurizer>`; then each eval compiles the body's specs again, renamed (`rename_spec`: each body name `<fit>.<eval>.<step>`, each dataset the eval maps read as the one it maps to), into ordinary steps held in the fit's `evals`, on the fit's axes, a forward with the fit's `batch_size` (`compile_fit`). `record_of` and `entry_of` reach an eval's value through the fit, and `clab/plan.py` `runnable` lists the steps that run, each fit's eval steps after it. Each value's record carries its labelled layout as far as the document decides it and the sweep axes it depends on (`clab/plan.py` `Labelled`, its `axes` by the dependency rule in [FORMAT.md](../../FORMAT.md#sweeps)), so every value's layout is known before any forward.
4. Saves are checked, and each entry is marked `gathered` when something uses its value (`mark_gathered`): a save, an op, a fit objective's scalar, or a gathered `grad` entry that differentiates it. The run takes off the model only what is gathered.
5. Sites and named interventions are inlined; the compiled plan holds neither.
6. Of the points of a forward or generate step that agree on the forward's axes, the first runs the forward and each other's `schedule` holds `serving`, that point's coordinates (`clab/compile/schedule.py` `share_forwards`).

The concepts compile resolves each have a module with their checks at compile and what they resolve to in the run: `clab/compile/answers.py`, `clab/compile/positions.py`, `clab/compile/arguments.py`, `clab/compile/interventions.py`, `clab/compile/schedule.py` and `clab/compile/uses.py`. A new check joins the module of its concept.

## Preflight

`clab/compile/preflight.py` `preflight` is the first thing `steps.run` does, so under the ndif engine it runs on the server. It:

- refuses every component the plan reads or writes that the engine does not serve, as the engine's `unserved` says; `clab/compile/interventions.py` `refuse_unserved` does it;
- per step (`prepare`): encodes each input once and records its tokens in `plan.run.tokens`; resolves each entry's `layers` and `select` among `engine.sizes()` and its positions per row; checks token columns against the tokenizer and each operand against Ops' one rule (`clab/compile/arguments.py` `mismatch`), and each write's operands from other entries (`check_operands`);
- writes each serving point's `schedule` once every point of the step has its records (`clab/compile/schedule.py` `write_schedule`): the sites the forward visits, sorted by `engine.rank`, the records done at each with every operand resolved to a producer's point or to a read of the same forward, the site it stops at, for a forward with `grad` entries each value it differentiates with the `grad` reads it serves, and for a generation the calls each record acts at; a write whose operand that forward reads at a site the engine reaches no earlier than the write's is refused there, and so is a `grad` entry whose `of` the engine reaches no later than the entry's site;
- prepares each fit's body over its training rows, into the same `Findings`, in the plan as the body sees it (`clab/plan.py` `body_view`), and each eval step after the fit as any step: `prepare` walks the steps `runnable` lists;
- finds each featurizer's width at its uses (`clab/compile/uses.py` `write_widths`).

It completes the plan's records in place (`layers`, `kept`, `positions` and `width` on every read and write, op steps' labels from their producers', the schedules, each featurizer's `width`) and returns a `Findings` (`clab/compile/state.py` `Findings`) with the rest the run needs: each input's encoded rows and padded width, room for what each generation's rows said, and answers.

## The run

`clab/steps/__init__.py` `run`:

1. `plan_with_rows`: the plan with every dataset's rows in it, a hub dataset's loaded here and refused unless their digest is the one compile recorded.
2. The versions and the model's sizes go into `plan.run`; then `preflight`.
3. `state_at_start`: each declared featurizer is made where the model is.
4. Under `torch.no_grad()`, every step in the order `runnable` lists them (document order, a fit's eval steps after it) at each point that runs (`clab/plan.py` `serving`: a forward's or generation's points that no other point's forward serves, an op's one per distinct point on its value's axes, a fit's every point), each through `KINDS[step.kind](engine, plan, name, step, state, batch_size, findings, cost)`, at the run's batch size unless the step's record holds one (an eval forward's, the fit's `batch_size`), its seconds and peak memory measured into its `cost` (`measure_cost`, which synchronizes the model's devices around the kinds that call the model and around no other, `NO_MODEL`) beside the model calls and tokens the kind adds. A CUDA out-of-memory stops the run there: the refusal (`out_of_memory_refusal`) is `plan.run.stopped`, and the steps that finished are recorded as below.
5. `record_results`: each step's costs into `plan.run.costs`, each saved value into `plan.results` as a detached CPU clone (`detached_cpu_clone`), one per distinct point on its own axes, then the tokens each generation's rows said, and the dtypes and devices of the model's parameters. Where each read and write resolved is on its record, which preflight and the steps complete in place.

`state` is a `ChainMap` in `run`'s frame (a fit's body runs in a child of it). `put` and `lookup` (`clab/format/sweeps.py`) keep each value under its name by its coordinates on its own axes (`projection`), so a consumer finds the value at its own point's coordinates there.

The step kinds:

- **forward** (`clab/steps/__init__.py` `forward`): the input in batches of dataset row ids, padded on the left as `plan.run.padding` says (`clab/steps/batches.py` `input_batches`: to the input's longest row, to the batch's own, or the rows longest first and each batch to its own). For each batch, `bind_schedule` binds the point's schedule to it: each site in order with its records, a write's ops bound to tensors at the operand points the schedule resolved, a read named `<entry>@<layer>` for each value the forward reads at the points it serves (`clab/compile/schedule.py` `values_read_by`). `engine.forward` executes it, under `torch.enable_grad()` with one backward after its stop site where the schedule differentiates (`differentiate`); `value_in_prefill` makes positions past a row's own NaN and runs the ops after the entry's last edit (`apply_ops_after_edit`); `stack_layers` puts a `layers` axis first; `join_batches` joins the batches along rows and puts the rows back in the run's order by id (`in_run_order`), which `sorted` padding changed. Each batch adds one model call, the tokens it fed and the rows' own of them to the cost.
- **op** (`op`): `ops.call` once over the whole value `of` names, its operands bound and aligned to `of`'s layers by label, the result put at the point's coordinates on the value's own axes.
- **generate** (`clab/steps/generate.py` `generate`): the schedule bound as a forward's; an entry past the prefill is bound in parts, over the prompt and at its decode `calls` (`online` for a write scoped to text), and its positions resolve after the generation from what each row said.
- **load** (`load`): the value the plan carries, which the compiled plan read from a run's file as it reads a bundle's (`clab/plan.py` `loaded_value`, through `clab/output/__init__.py` `read_saved_value`) and which travels with it, in a remote session's request too, put where the model is at the point's coordinates; the run reads no file. Preflight tokenizes the input of the read it carries, so `run.tokens` and the table label its positions.
- **fit** (`clab/steps/fit.py` `fit`): trains its featurizers over minibatches, running the body's steps through the same `KINDS` table (`run_body`) with gradients for the featurizers' tensors alone, each body forward given what it reads once for every run of the body, and ends with each trained featurizer scoring, under `<fit>.<featurizer>`, where the eval steps after it find it.

## What belongs to the engine

The engine loads and places the model, runs one batch through it (a forward, or a generation) with a bound schedule, and decides nothing: it visits the schedule's sites in the order written, hands each site's value to `clab/steps/batches.py` `visit_site` and sets back what that returns, and ends a forward right after the `stop` site where it declares that it `stops`. It also answers what preflight and the steps ask: its tokenizer, sizes, EOS ids, readout, what it does not serve, and where a component falls in its forward.

Everything else is `clab/steps/` in plain torch, the same code for every engine: padding, positions, binding operands, op order, NaN for missing positions, joins, metrics, featurizers and the fit loop. That is why the nnterp and hooks engines give the same results to the bit on every corpus document they both serve (`tests/test_documents.py` `test_the_engines_agree_to_the_bit`). The protocol in full is [engines.md](engines.md).
