---
title: Engine Protocol
one_liner: "The engine protocol and how to add an engine: the constructor and options, what execute must do, the members steps.run and preflight read, a bound schedule site by site, where a forward stops, and where an engine refuses."
tags: [developing, engines, protocol]
related: [docs/usage/engines.md, docs/developing/compile-and-run.md, docs/developing/architecture.md, docs/concepts/one-session.md]
sources: [clab/engines/__init__.py, clab/engines/standard.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/engines/vllm.py, clab/steps/batches.py, clab/steps/__init__.py, clab/steps/generate.py, clab/compile/interventions.py, clab/compile/preflight.py]
---

# Engine Protocol

An engine is the one object that holds a model. It loads a checkpoint, runs batches through it, and executes the schedule preflight wrote, doing at each site what its records say and deciding nothing. Everything else a run does is `clab/compile/` and `clab/steps/`, the same for every engine. The protocol is `clab/engines/__init__.py` `Engine`; this page says what each part means and what a new engine must do. Choosing among the shipped engines is [usage/engines.md](../usage/engines.md).

An engine refuses what it cannot do at three seams, each by name and before any forward:

- its constructor and `load`: an option, a model block key, a device or a `dtype` it cannot honor;
- its `execute`, before and through `start_run`: what the plan asks that the engine cannot do;
- the members it declares, which `start_run` and preflight read: `gradless`, `unserved`, `last_only`, `longest`.

[Refusing by name](#refusing-by-name) maps each refusal to its seam.

## Registering and loading

An engine is a class decorated with `@engine("<name>")` (`clab/engines/__init__.py` `engine`), which sets the class's `name`. Its module is imported by the line at the end of `clab/engines/__init__.py`. That line's order is the order `clab describe engines` and [FORMAT.md](../../FORMAT.md#engines-and-run-options) list engines in.

Its constructor is `__init__(self, model, *, device="cpu", <options>)`:

- `model` is the plan's `model` block (`clab/plan.py` `Model`).
- `device` is its one placement argument, read however the engine likes.
- Every other keyword parameter is an option, with a default and a line `<name>: <what it takes>` in the constructor's docstring. `describe engines` prints that line, and `tests/test_format.py` `test_every_engine_option_says_what_it_takes` fails without it.

`load` (`clab/engines/__init__.py` `load`) checks options against that signature before anything loads. Under torchrun (`WORLD_SIZE` above 1) it also refuses an engine without a `tp` option, each of whose processes would run the whole plan.

```text
the nnterp engine takes no 'foo'; its options: tp (where the model runs is `device`, given as --device)
remote: 'yes' is not one of True, 'local'
```

The constructor, in order, with what `clab/engines/standard.py` shares:

1. Take the loader's dtype arguments from `loading`. It turns `dtype` into torch's dtype or a quantization's `quantization_config`, so every engine holds the same weights. It refuses a `device` no engine can place a model on.
2. Take the placement from `placing`.
3. Build the model with `build`. It passes the model block's other keys to the loader as they are, and refuses the keys in `OWNED` by name (`model.device_map: the hooks engine cannot honor it: ...`). It reports a checkpoint the loader cannot get as `model.key`, and a key the loader rejects as that key.
4. Freeze the model's parameters (`requires_grad_(False)`). That moves no value and keeps no graph for a weight's gradient, a fit's or a `grad` entry's.

A shell that loads its weights later loads them through `build` too, so a key the loader rejects is refused by name there. It freezes them once loaded. The nnterp engine does this in `clab/engines/nnterp.py` `NnterpEngine.ready`. That engine, and ndif's under `remote="local"`, run `ready` before the session, ahead of the first step of a run that calls the model (`clab/plan.py` `calls_model`). Under `remote=True`, `ready` runs on the server, which holds its own weights. `ready` also does once what the model would otherwise do at its first trace, so that no step's cost holds it:

- **It instruments a mixture of experts.** Each module of a mixture whose forward nnterp reads a value inside has that forward instrumented (its `.source` touched), at every block that serves the value. Without this, nnsight would do it at the first trace that reads the value. That is too late on JetMoE, whose router logits are its router's projection's output, read before the indices inside the router's forward. So one forward reads a mixture's values in forward order, on a fresh engine too.
- **It measures the forward order** (`rank`), which nnterp would do at the first site a step ranks.

An engine then holds these, and nothing else, since inside a session it is the one client-side object ([concepts/one-session.md](../concepts/one-session.md)):

- `name`, and `spec`, the model block;
- `model`: anything with `.parameters()`, whose dtypes and devices the run records;
- `device`: None where it does not place the model, as on a server;
- `options`: as given, after defaults;
- `remote`: its session's `remote=`, False for a run in this process;
- `stops`: whether it ends a forward at the schedule's stop site;
- `cuda`: the CUDA devices whose peak memory a step's cost measures, none for a remote run;
- what it measured at load, such as the model's sizes (`config` in each shipped engine).

It may also declare `gradless`, `last_only` and `longest` ([the table below](#what-stepsrun-asks-of-an-engine)).

## `execute`

`execute(plan, *, batch_size=8, padding="dataset", readout="full")` returns the filled plan; its batch size, padding and readout are keyword-only. It is the one method an engine that runs a plan without `clab.steps` must have. An engine that runs through `clab.steps`, as every shipped one does, does three things:

1. `clab/steps/__init__.py` `start_run`, on the client before any session opens. It refuses:
   - an uncompiled plan (`plan is not compiled; call plan.compile()`);
   - a batch size below one row;
   - a padding `clab/format/data.py` `PADDINGS` does not name, or a readout `clab/compile/schedule.py` `READOUTS` does not name;
   - a plan whose `model` block differs in any key from the one the engine loaded, the message naming each key with both values;
   - a step's `grad` entry where the engine declares `gradless` (`<step>.<entry>: <gradless>`);
   - where the engine's `remote` is set, a document whose loads hold more than `LOADED` bytes of tensors, which a request carries inside its plan.

   It then starts `plan.run` with the engine, batch size, padding, readout, device and options, and returns the start time.
2. `clab/steps/__init__.py` `run`, inside the engine's one session if it has one.
3. `plan.run.seconds`, set from `start_run`'s start time.

The hooks engine's `execute` is the shortest. Where the run calls the model, the nnterp and vllm engines load the weights (`ready`) after `start_run` and call `start_run` again, so that `seconds` leaves the load out. The vllm engine's `execute` first refuses three things by name: a fit, a write at `attention_scores` or `attention_probabilities`, and a generate step whose options `sampling` cannot map.

An engine that ignores `clab.steps` has everything `steps.run` does to do itself, preflight included. It loads a hub dataset's rows where it runs with `compile.dataset_rows` (as `steps.plan_with_rows` does), executes each serving point's schedule, and resolves live operands as `visit_site` does. It fills `plan.results`, `plan.run` and the records' `positions` the same way.

## What `steps.run` asks of an engine

| member | what it returns |
|---|---|
| `forward(batch, bound)` | one forward of the batch (`input_ids`, `attention_mask`, `position_ids`, already padded) with the bound schedule executed; each read's value by its name |
| `generate(batch, bound, options)` | the model's `generate` over the batch with the step's own keys as keyword arguments, the bound schedule executed at the forward calls its `loop` names; a pair: the new ids (the prompt cut off), then each read by name |
| `readout(hidden)` | a residual-stream value's logits: on nnterp the model's own (`project_on_vocab`); on hooks the final norm, the head and the config's `final_logit_softcapping`. They agree on the corpus |
| `unserved()` | each component in `COMPONENTS` mapped to None (served), the reason it is not, or `{layer: reason}`. The shipped engines start from nnterp's answer (`clab/engines/standard.py` `unserved`) and say it with `own`, which names what the engine serves where nnterp's reason points at its own `Moe` component |
| `rank(component, layer)` | any sortable value placing the component in the forward, by which preflight sorts the schedule's sites; sites at one module boundary may rank alike |
| `sizes()` | the model's sizes, the keys in `clab/engines/standard.py` `SIZES` (`num_experts` and `top_k` only on a model with a mixture of experts) |
| `tokenizer`, `eos` | a Hugging Face tokenizer (`PreTrainedTokenizerBase`), with which the steps tokenize inputs and answers, render chats with `apply_chat_template`, pad batches and name tokens with `convert_ids_to_tokens`; the ids generation stops on (`clab/engines/standard.py` `stops`) |
| `gradless` | optional, read by `start_run`. Why the engine refuses a `grad` entry, worded as the refusal says it after the entry's name, or None where it takes them. An engine without it takes them; the vllm engine declares it, and the nnterp engine under `tp`. A fit is not covered: the vllm engine refuses one in its `execute`, and the nnterp engine trains under `tp` |
| `last_only`, `longest` | optional, read by preflight. `last_only` maps each component the engine serves at the prompt's last position alone to the reason. Once a step's schedule is written, preflight refuses a prefill record there that meets another position on some row (`clab/compile/preflight.py` `check_last_only`); a read of `logits` that `at-reads` moved to the last block's output is not held to it. `longest` is the longest sequence, prompt and new tokens, the engine serves, or None; preflight refuses a step that does not fit (`check_longest`). An engine without them serves every position and any length; the vllm engine declares the head and vLLM's `max_model_len` |

Preflight uses `unserved`, `sizes`, `rank`, `last_only`, `longest` and the tokenizer before any forward; the steps use the rest. The batch's tensors go where the model's first parameter is, or to the engine's `device` while that parameter is on the meta device (`clab/steps/batches.py` `model_device`).

**Gradients.** The steps run with gradients off (`clab/steps/__init__.py` `run`). A fit runs its body's minibatches with gradients on (`clab/steps/fit.py` `run_body`) and differentiates through the engine's `forward` to the trained featurizers' tensors. A forward with `grad` entries differentiates inside it ([below](#the-bound-schedule)). So `forward` must not turn gradients off, run under `torch.inference_mode`, or detach a value it hands to `visit_site` or takes back from it.

## The bound schedule

`bound` is a forward's `schedule` bound to one batch (`clab/steps/batches.py` `bind_schedule`):

- `sites`, already in this engine's forward order: each its `component`, its `layer` (None for a model-level component) and its `records` in the order they are done there;
- `stop`, the site a forward ends at (None in a generation);
- `grad`, what the forward differentiates (empty where it takes no gradient);
- in a generation, `loop`, the forward calls to run the records at (None for every one), and `online`, whether a write is found call by call.

A record with `ops` is a write. One without is a read, and carries the `name` its value is returned under (`<entry>@<layer>`); a `grad` entry's read is marked `capture`. The rest of a record is for `visit_site` alone, and the engine passes it through. It holds the positions into the padded batch, the `select` indices, which positions are a row's own, a `key` axis's keys, and which run-supplied values its ops take (`takes`). A read's `readout` is for `clab/steps/batches.py` `value_in_prefill`, once the forward is done. What preflight writes into the schedule is [FORMAT.md](../../FORMAT.md#derived-fields) under `Forward`.

What the engine does with it, deciding nothing:

1. Visit the sites in the order listed.
2. At each site, take the value in the component's own layout (its `dims` in [FORMAT.md](../../FORMAT.md#components)), and call `clab/steps/batches.py` `visit_site(value, component, records, got)`. Where the model holds the value in another layout, which one is the engine's configuration, fixed at load and never read off the value. `visit_site` does the site's records in order. A write acts on the value as the earlier records left it, and a read stores it in `got`. A `capture` stores the value the forward goes on with, made a leaf that requires grad where nothing upstream does. It returns the value and whether to set it back: set it back only then (a record wrote, or captured it).
3. Where the engine `stops`, end the forward right after the `stop` site; the sites after it are not visited. An engine that runs every forward whole gets the same values, later, and declares `stops = False`.
4. Where the bound schedule's `grad` is not empty, run the forward under `torch.enable_grad()`. Once the `stop` site is done and before the forward ends, call `clab/steps/batches.py` `differentiate(got, bound["grad"])`: one backward per value differentiated, each `capture` in `got` replaced by its gradient as its record selects it, every value in `got` detached after. The nnterp engine calls it inside the trace before `tracer.stop()`; the hooks engine after `_Stopped`, the captured tensors keeping their graph.
5. Gather nothing a record does not ask for, and return `got`. A forward that ended before one of its reads fails in the step, naming the reads it lacks (`clab/steps/__init__.py` `forward`).

In a generation (both helpers below are in `clab/steps/batches.py`):

- Forward call 0 is the prefill, and call k is the decode step that takes generated token k-1 as input; the generation runs whole.
- At each call the schedule's `loop` names, the engine filters each site's records with `acting_at_call(records, call)`. A record without `calls` acts on the prefill only; one with `calls` acts on the decode calls listed (None for every one), a read appending one value per call.
- At a decode call the engine then passes the records through `stamped_at_call(records, seen, tokenizer, call)`, `seen` being each decode call's `input_ids` so far (read where the schedule says a write is `online`). It stamps each write with its call and narrows a write scoped to text to the rows it lands on.

The steps check that a read collected one value per decode call made (`clab/steps/generate.py` `run_engine_generate`); a read out of forward order shifts calls and fails there.

### The shipped engines

- **nnterp.** `clab/engines/nnterp.py` `NnterpEngine.forward` runs the bound schedule in one nnsight trace. `visited` reads and sets each component on nnterp's envoy up to the stop site, and `tracer.stop()` ends the trace there.
- **hooks.** `clab/engines/hooks.py` `HooksEngine.hooked` registers one hook per site, at the module boundary nnterp names.
  - Each boundary's layout is measured once at load, on a two-token forward (`order`).
  - A value nnterp holds flat over tokens (`[batch * seq, ...]`) is served as `[batch, seq, ...]`; a `[batch, seq, hidden]` boundary is split into heads for a per-head component.
  - A served boundary the two-token forward does not reach is refused by name at load (`<component> layer <i>: the hooks engine's two-token forward at load did not reach its module ...`).
  - The stop site's hook raises `_Stopped` after its records are done.
- **ndif.** It inherits the nnterp engine's forward and its stop, under either `remote`. A server runs the engine of the clab installed there, which decides its stop.
- **vllm.** `clab/engines/vllm.py` `VllmEngine.forward` opens one trace per batch, one invoke per row, each with the row's own token ids; no padding reaches vLLM.
  - Each invoke carries the row's part of the bound schedule (`clab/steps/batches.py` `schedule_for_row`). A write's operands with a rows axis are taken at the row (`bind_ops` marks them `rowed`); the rest are shared.
  - In vLLM's worker, `visit_request` pads the row's `[1, tokens, ...]` value on the left to the batch's width (`padded`), so the records' indices hold. It hands the value back unpadded where a record wrote.
  - The rows' reads come home joined along rows (`joined`), blank where a row made no such call.
  - It declares `stops = False`: nothing ends a request's forward in the worker.

## Live operands

A write's op argument `{"live": "<read name>"}` names a read the same forward call made earlier. `clab/steps/batches.py` `live_operand` takes that read's value from `got` for `visit_site`, and raises if the engine had not made the read when it reached the write. The schedule lists the read first: preflight refuses a write whose operand is read at a site the engine ranks no earlier than the write's (`clab/compile/schedule.py` `write_schedule`). So an engine whose `rank` matches the order its forward really reaches sites never meets that error. An engine that runs a plan without `clab.steps` resolves such an argument the same way.

## Refusing by name

| what the engine cannot do | where it is refused |
|---|---|
| an option it does not declare, or of another type; torchrun around an engine without `tp` | `load` |
| an option value it cannot honor | its constructor, a `ValueError` naming it (`device: the ndif engine cannot honor 'cuda:0': the server places the model`) |
| a model block key | `build` (`OWNED`), or the loader |
| a checkpoint the loader cannot get | `build`, as `model.key` |
| a `device` or a `dtype` it cannot place or hold | `loading` (`fp8` without a GPU of compute capability 8.9 or later; a bitsandbytes format without bitsandbytes) |
| anything else read off the plan alone | its `execute`, before `start_run`, naming the step (the vllm engine's fits, writes at the recomputed pattern, and generate options `sampling` cannot map) |
| a `grad` entry (under `tp`, whose ranks would hold a sharded value's gradient in pieces; on vLLM) | `gradless`, which `start_run` turns into `<step>.<entry>: <gradless>` before the session opens |
| more loaded tensors than a request carries, where its session is `remote` | `start_run`, before anything is sent |
| a component | `unserved`, which preflight turns into `<step>.<entry>: the <engine> engine does not serve <component>: <reason>` |
| a position or a length | `last_only` and `longest`, which preflight turns into `<step>.<entry>: the <engine> engine serves <component> at the prompt's last position alone ...` and `<step>: <input>'s longest row has <n> tokens ...` |

## Adding an engine

1. A module under `clab/engines/`, one class with `@engine("<name>")`, added to the import line at the end of `clab/engines/__init__.py`.
2. The constructor as above, each option documented on its own docstring line. Build the model with `build`, and reuse what `clab/engines/standard.py` shares (`loading`, `placing`, `unserved`, `own`, `envoy`, `inputs`, `stops`, `sizes`).
3. `execute` through `start_run` and `run`, the members in the table, and `remote`, `stops` and `cuda`. Declare `last_only` and `longest` only where the engine serves a position or a length in part, and `gradless` only where it refuses a `grad` entry.
4. Regenerate FORMAT.md (`uv run --no-sync clab describe --markdown`, [contributing.md](contributing.md#changing-the-format)) so its engines section lists the new one.
5. Prove agreement: the corpus documents on the new engine give values equal to the nnterp engine's (`tests/conftest.py` `same`). The engine loops to extend are `SERVED` in `tests/test_documents.py`, the `ran` fixture in `tests/test_fit.py` and the `filled` fixture in `tests/test_generate.py`. `tests/test_refusals.py` `test_a_loader_refuses_a_model_key_it_cannot_honor` lists the engines its refusals cover. Running the suite is [testing.md](testing.md).
