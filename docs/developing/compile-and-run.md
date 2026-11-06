---
title: Compile and Run
one_liner: "The two passes and the run: compile's checks with the document alone, preflight's with the model, the schedule each forward runs, how steps.run walks a plan, and what belongs to the engine."
tags: [developing, compile, preflight, run]
related: [docs/developing/architecture.md, docs/developing/engines.md, docs/concepts/plan-lifecycle.md, docs/errors/index.md]
sources: [clab/compile/__init__.py, clab/compile/preflight.py, clab/compile/schedule.py, clab/compile/state.py, clab/compile/interventions.py, clab/steps/__init__.py, clab/steps/batches.py, clab/steps/generate.py, clab/steps/fit.py, clab/plan.py]
---

# Compile and Run

A document is checked before anything runs. Compile checks it with the document, its data files and bundle headers alone. The engine checks its model block as it loads, and the plan against its declarations in `execute`, before any session. Preflight, inside the run, checks it with the model in hand. Then `steps.run` walks the plan step by step. This page says where each piece of that lives, so a new check or a new step lands in the right place. The stages from the user's side, and which refuses what, are [concepts/plan-lifecycle.md](../concepts/plan-lifecycle.md).

## Where a check belongs

Decide everything as early as the information allows.

| what it needs | where it runs | how it refuses |
|---|---|---|
| a record's own shape: fields, types, an op's arguments against its signature, a featurizer's fields, a model block key no engine takes (`OWNED`) | building the plan: the validators in `clab/plan.py` | `PlanError`, starting with the document path |
| the document, the data files, bundle headers | compile, `clab/compile/` | `fail(path, message)` in `clab/plan.py`: a `PlanError` starting with the document path |
| an option, a device or a `dtype` the engine cannot honor; a model block key its loader rejects; a checkpoint or revision the loader cannot get (offline, one the cache lacks) | the engine's constructor, `clab/engines/__init__.py` `load` and `clab/engines/standard.py` `build` | `ValueError` naming the option or key |
| the plan against the engine's declarations (`gradless`, `LOADED`, the model block, the run options) | the engine's `execute`, through `clab/steps/__init__.py` `start_run`, on the client before the session | `ValueError` starting with `<step>.<entry>`, `<step>`, or naming the option |
| the tokenizer, a size of the model, what the engine serves, forward order | preflight, `clab/compile/preflight.py` (the schedule's sites, `clab/compile/schedule.py`) | `ValueError` starting with `<step>.<entry>` |

A check that needs the engine's declarations but not the model goes in `start_run`, or in an attribute the engine declares and `start_run` or preflight reads, as `gradless`, `last_only` and `longest` are ([engines.md](engines.md#what-stepsrun-asks-of-an-engine)). A check that needs the model goes in preflight.

Two refusals of the patching document, one from each pass:

```text
steps.iia.of: 'patched.nothing' names no value produced before this step; known: counterfactual.v_cf, patched.logits, patched.patch
counterfactual.v_cf: layer 5; the model has 2 layers
```

The first is compile's: `of` names a value no step produced, and the message starts with the document path. The second is preflight's: `"layers": 5` compiles, because compile does not know the model has two layers, and the refusal names the entry. A refusal inside a fit's body is prefixed with the fit's name. One in an eval names its eval step (`fit.held_out.adapted.adapt: …`), and compile's names the eval (`steps.fit.eval.held_out: …`). `PlanError` is a `ValueError`, so a caller catches both as one; `clab run` prints either as `clab: <message>` and exits 1.

## Compile

`clab/compile/__init__.py` `compile` returns a new plan with `compiled` true. In order:

1. Each dataset's rows are made once (`make_dataset_rows`: a file's rows in its `split`, inline rows, or a hub split; then `shuffle`, `take` and `columns`), and their digest and row count recorded. A hub dataset's rows are checked and left out of the plan; the run loads them again where it runs.
2. Sweeps are found, and each featurizer, then each step, is compiled at every point of the axes it uses or inherits from what it references (`expand`). A featurizer's bundle is loaded and checked key by key against the document (`compile_featurizer`).
3. Steps compile in document order; `compile_step` dispatches by kind.
   - The records compiled so far are the symbol table (`clab/compile/state.py` `Compiling`: `Compiling.steps`, `Compiling.featurizers`).
   - A name resolves only after it exists, to a record at the consumer's point (`clab/plan.py` `record_of`).
   - A fit's body compiles in a scope holding the fit; each trained featurizer is `<fit>.<featurizer>`.
   - Each eval recompiles the body renamed `<fit>.<eval>.<step>` (`rename_spec`), its datasets mapped, into ordinary steps in the fit's `evals`. They sit on the fit's axes, at the fit's `batch_size` (`compile_fit`).
   - `clab/plan.py` `runnable` lists each fit's eval steps after it; `record_of` and `entry_of` reach an eval's value through the fit.
   - Each value's record carries its labelled layout and its sweep axes (`clab/plan.py` `Labelled`, its `axes` by the dependency rule in [FORMAT.md](../../FORMAT.md#sweeps)), so every layout is known before any forward.
4. Saves are checked, and each entry is marked `gathered` when something uses its value (`mark_gathered`): a save, an op, a fit objective's scalar, or a gathered `grad` entry that differentiates it. The run takes off the model only what is gathered. A gathered entry used only as a live operand of a later write of its own forward (`Compiling.live`, beside `used`) is listed in its forward's schedule as `freed`, unless a save or a `grad` entry uses it. The run keeps none of a freed entry's batches.
5. Sites and named interventions are inlined; the compiled plan holds neither.
6. Of the points of a forward or generate step that agree on the forward's axes, the first runs the forward and each other's `schedule` holds `serving`, that point's coordinates (`clab/compile/schedule.py` `share_forwards`).

The concepts compile resolves each have a module with their checks at compile and what they resolve to in the run: `clab/compile/answers.py`, `clab/compile/positions.py`, `clab/compile/arguments.py`, `clab/compile/interventions.py`, `clab/compile/schedule.py` and `clab/compile/uses.py`. A new check joins the module of its concept.

## Preflight

`clab/steps/__init__.py` `run` calls `clab/compile/preflight.py` `preflight` before any step, after the rows are in view and the run record holds the versions, sizes and thread count. Under the ndif engine it runs on the server. It:

- refuses every component the plan reads or writes that the engine does not serve, as the engine's `unserved` says (`clab/compile/interventions.py` `refuse_unserved`);
- per step (`prepare`):
  - encodes each input once and records its tokens in `plan.run.tokens`;
  - resolves each entry's `layers` and `select` among `engine.sizes()`, and its positions per row;
  - checks token columns against the tokenizer, each operand against Ops' one rule (`clab/compile/arguments.py` `mismatch`), and each write's operands from other entries (`check_operands`);
  - refuses a step whose longest row does not fit the engine's `longest` (`check_longest`);
- writes each serving point's `schedule` once every point of the step has its records (`clab/compile/schedule.py` `write_schedule`):
  - the schedule lists the sites the forward visits, sorted by `engine.rank`;
  - each site lists the records done there; each operand is resolved to a producer's point or to a read of the same forward;
  - the schedule names the site the forward stops at;
  - a forward with `grad` entries lists each value it differentiates and the `grad` reads it serves;
  - a generation lists the calls each record acts at;
  - under `--readout at-reads`, a read of `logits` naming fewer than every position is done at the last block's output, marked `readout`, so the forward stops before the head (`clab/compile/schedule.py` `read_out_at_reads`);
  - it refuses a write whose same-forward operand is read at a site the engine reaches no earlier than the write;
  - it refuses a `grad` entry whose `of` the engine reaches no later than the entry's site;
- refuses, once a step's schedule is written, a prefill record at a component the engine serves at the last position alone that meets any other position (`check_last_only`);
- prepares each fit's body over its training rows, into the same `Findings`, in the plan as the body sees it (`clab/plan.py` `body_view`), and each eval step after the fit as any step: `prepare` walks the steps `runnable` lists;
- finds each featurizer's width at its uses (`clab/compile/uses.py` `write_widths`).

It completes the plan's records in place: `layers`, `kept`, `positions` and `width` on every read and write, op steps' labels from their producers', the schedules, each featurizer's `width`. It returns a `Findings` (`clab/compile/state.py` `Findings`) with the rest the run needs: each input's encoded rows and padded width, room for what each generation's rows said, and answers.

## The run

`clab/steps/__init__.py` `run`:

1. `plan_with_rows`: the plan with every dataset's rows in it, a hub dataset's loaded here and refused unless their digest is the one compile recorded.
2. The versions, the model's sizes and torch's thread count go into `plan.run` (`Run.threads`); then `preflight`.
3. The steps, under `torch.no_grad()`:
   - The run walks the steps `runnable` lists: document order, each fit's eval steps after it.
   - A step runs at each point `clab/plan.py` `serving` lists: a forward's or generation's points that no other point serves, an op's one per distinct point on its value's axes, a fit's every point.
   - Each run of a step is `KINDS[step.kind](engine, plan, name, step, state, batch_size, findings, cost)`.
   - A step runs at the run's batch size unless its record holds one; an eval forward holds the fit's `batch_size`.
   - `measure_cost` records seconds and peak memory into the step's `cost`, beside the model calls and tokens the kind adds. It synchronizes the model's devices only around kinds not in `NO_MODEL`.
   - Before each step, `make_featurizers` makes, on the model's device, every declared featurizer not yet made whose draw the state holds. That is every one before the first step, and one drawn from a value (a constant's `init`) before the first step after the one that made it.
   - A CUDA out-of-memory stops the run: the refusal (`out_of_memory_refusal`) is `plan.run.stopped`, and the finished steps are recorded as below.
4. `record_results`: each step's costs into `plan.run.costs`, and each saved value into `plan.results` as a detached CPU clone (`detached_cpu_clone`), one per distinct point on its own axes. Then the tokens each generation's rows said, and the dtypes and devices of the model's parameters. Where each read and write resolved is on its record, which preflight and the steps complete in place.

`state` is a `ChainMap` in `run`'s frame (a fit's body runs in a child of it). `put` and `lookup` (`clab/format/sweeps.py`) keep each value under its name by its coordinates on its own axes (`projection`), so a consumer finds the value at its own point's coordinates there.

The step kinds:

- **forward** (`clab/steps/__init__.py` `forward`):
  - It feeds its input in batches of dataset row ids, left-padded as `plan.run.padding` says (`clab/steps/batches.py` `input_batches`).
  - What the binding needs from the step alone is worked out once, before the batches (`schedule_binding`, through `op_binding` and `argument_binding`).
  - Each batch completes the binding with its tensors (`bind_schedule`, `bind_ops`, `bind_arguments`). Each read is named `<entry>@<layer>`, one per value the forward reads at the points it serves (`clab/compile/schedule.py` `values_read_by`).
  - `engine.forward` executes the bound schedule. Where the schedule differentiates, it runs under `torch.enable_grad()` with one backward after the stop site (`differentiate`).
  - `value_in_prefill` reads out a read marked `readout` (`engine.readout`, then its `select`), sets positions past a row's own to NaN, and runs the ops after the entry's last edit (`apply_ops_after_edit`).
  - `stack_layers` puts a `layers` axis first. `join_batches` joins the batches along rows and restores the run's row order (`in_run_order`), which `sorted` padding changed.
  - A value the schedule marks `freed` is kept from no batch.
  - Each batch adds one model call, its fed tokens and its rows' own tokens to the cost.
- **op** (`op`): `ops.call` once over the whole value `of` names, its operands bound and aligned to `of`'s layers by label, the result put at the point's coordinates on the value's own axes.
- **generate** (`clab/steps/generate.py` `generate`): the schedule bound as a forward's. An entry past the prefill is bound in parts, over the prompt and at its decode `calls` (`online` for a write scoped to text). Its positions resolve after the generation, from what each row said.
- **load** (`load`): the value the plan carries, put where the model is at the point's coordinates. The compiled plan read it from a run's file as it reads a bundle's (`clab/plan.py` `loaded_value`, through `clab/output/__init__.py` `read_saved_value`). It travels with the plan, in a remote session's request too, so the run reads no file. Preflight tokenizes the input of the read it carries, so `run.tokens` and the table label its positions.
- **fit** (`clab/steps/fit.py` `fit`): trains its featurizers over minibatches, running the body's steps through the same `KINDS` table (`run_body`) with gradients for the featurizers' tensors alone. Each body forward gets what it reads and its schedule's binding (`schedule_binding`) once for every run of the body. The fit ends with each trained featurizer scoring, under `<fit>.<featurizer>`, where the eval steps after it find it.

## What belongs to the engine

The engine loads and places the model and runs one batch through a bound schedule, deciding nothing; everything else is `clab/steps/` in plain torch, the same for every engine ([engines.md](engines.md#the-bound-schedule)).
