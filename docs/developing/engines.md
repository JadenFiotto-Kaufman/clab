---
title: Engine Protocol
one_liner: "The engine protocol and how to add an engine: what execute must do, what steps.run asks for, executing a bound schedule site by site, where a forward stops, live operands, and refusing by name."
tags: [developing, engines, protocol]
related: [docs/usage/engines.md, docs/developing/compile-and-run.md, docs/developing/architecture.md, docs/concepts/one-session.md]
sources: [clab/engines/__init__.py, clab/engines/standard.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/steps/batches.py, clab/steps/__init__.py, clab/steps/generate.py]
---

# Engine Protocol

An engine is the one object that holds a model: it loads a checkpoint, runs batches through it, and executes the schedule preflight wrote, doing at each site what its records say and deciding nothing. Everything else a run does is `clab/compile/` and `clab/steps/`, the same for every engine. The protocol is `clab/engines/__init__.py` `Engine`; this page says what each part means and what a new engine must do. Choosing among the shipped engines is [usage/engines.md](../usage/engines.md).

## Registering and loading

An engine is a class decorated with `@engine("<name>")` (`clab/engines/__init__.py` `engine`), which sets the class's `name`, and whose module is imported by the line at the end of `clab/engines/__init__.py`; that line's order is the order `clab describe engines` and [FORMAT.md](../../FORMAT.md#engines-and-run-options) list engines in.

Its constructor is `__init__(self, model, *, device="cpu", <options>)`. `model` is the plan's `model` block (`clab/plan.py` `Model`); `device` is its one placement argument, read however the engine likes; every other keyword parameter is an option, with a default and a line `<name>: <what it takes>` in the constructor's docstring, which `describe engines` prints (`tests/test_format.py` `test_every_engine_option_says_what_it_takes` fails without it). `load` (`clab/engines/__init__.py` `load`) checks options against that signature before anything loads:

```text
the nnterp engine takes no 'foo'; its options: tp (where the model runs is `device`, given as --device)
remote: 'yes' is not one of True, 'local'
```

Under torchrun (`WORLD_SIZE` above 1) `load` also refuses an engine without a `tp` option, each of whose processes would run the whole plan.

Take the loader's dtype arguments from `clab/engines/standard.py` `loading`, which turns `dtype` into torch's dtype or a quantization's `quantization_config`, so every engine holds the same weights, and refuses a `device` no engine can place a model on; take the placement from `placing`. Build the model with `build`: it passes the model block's other keys to the loader as they are, refuses the keys in `OWNED` by name (`model.device_map: the hooks engine cannot honor it: where the model runs is a run option: --device`), refuses a checkpoint the loader cannot get as `model.key`, and reports a key the loader rejects as that key.

An engine then holds `name`, `spec` (the model block), `model` (anything with `.parameters()`, whose dtypes and devices the run records), `device` (None where it does not place the model, as on a server), `options` (as given, after defaults), `remote` (its session's `remote=`, False for a run in this process), `stops` (whether it ends a forward at the schedule's stop site) and `cuda` (the CUDA devices whose peak memory a step's cost measures, none for a remote run), and nothing else: inside a session it is the one client-side object ([concepts/one-session.md](../concepts/one-session.md)).

## `execute`

`execute(plan, *, batch_size=8)`, its batch size keyword-only, is the single entry point, and the only method an engine that runs a plan without `clab.steps` must have; it returns the filled plan. An engine that runs through `clab.steps` (all three shipped ones do) does three things, the hooks engine's `execute` being the shortest:

1. `clab/steps/__init__.py` `begin`: refuses an uncompiled plan, a batch size below one row, or a plan whose `model` block differs in any key from the one the engine loaded (the message names each key that differs, with both values), and starts `plan.run` with the engine, batch size, device and options.
2. `clab/steps/__init__.py` `run`, inside the engine's one session if it has one.
3. `plan.run.seconds` set from `begin`'s start time.

An engine that ignores `clab.steps` has everything `steps.run` does to do itself, preflight included: it loads a hub dataset's rows where it runs with `compile.materialized` (as `steps.held` does), executes each serving point's schedule, resolves live operands as `visit` does, and fills `plan.results`, `plan.run` and the records' `positions` the same way.

## What `steps.run` asks of an engine

| member | what it returns |
|---|---|
| `forward(batch, bound)` | one forward of the batch (`input_ids`, `attention_mask`, `position_ids`, already padded) with the bound schedule executed; each read's value by its name |
| `generate(batch, bound, options)` | the model's `generate` over the batch with the step's own keys as keyword arguments, the bound schedule executed at the forward calls its `loop` names; a pair: the new ids (the prompt cut off), then each read by name |
| `readout(hidden)` | a residual-stream value's logits: on nnterp the model's own (`project_on_vocab`), on hooks the standard form, the final norm, the head and the config's `final_logit_softcapping`, which agree on the corpus |
| `unserved()` | each component in `COMPONENTS` mapped to None (served), the reason it is not, or `{layer: reason}`; the shipped engines start from nnterp's answer (`clab/engines/standard.py` `unserved`) |
| `rank(component, layer)` | any sortable value placing the component in the forward, which preflight reads to sort the schedule's sites; sites at one module boundary may rank alike |
| `sizes()` | the model's sizes, the keys in `clab/engines/standard.py` `SIZES` |
| `tokenizer`, `eos` | a Hugging Face tokenizer (`PreTrainedTokenizerBase`), with which the steps tokenize inputs and answers, render chats with `apply_chat_template`, pad batches and name tokens with `convert_ids_to_tokens`; the ids generation stops on (`clab/engines/standard.py` `stops`) |

Preflight uses `unserved`, `sizes`, `rank` and the tokenizer before any forward; the steps use the rest. The batch's tensors go where the model's first parameter is, or to the engine's `device` while that parameter is on `meta`, as in a shell that loads its weights at its first forward (`clab/steps/batches.py` `home`).

**Gradients.** The steps run with gradients off (`clab/steps/__init__.py` `run`), but a fit runs its body's minibatches with gradients on (`clab/steps/fit.py` `body_run`) and differentiates through the engine's `forward` to the trained featurizers' tensors. So `forward` must not turn gradients off, run under `torch.inference_mode`, or detach a value it hands to `visit` or takes back from it.

## The bound schedule

`bound` is a forward's `schedule` bound to one batch (`clab/steps/batches.py` `bind`): `sites`, already in this engine's forward order, each its `component`, its `layer` (None for a model-level component) and its `records` in the order they are done there; `stop`, the site a forward ends at (None in a generation); and for a generation `loop`, the forward calls to run the records at (None for every one), and `online`, whether a write is found call by call. A record with `ops` is a write; one without is a read and carries the `name` its value is returned under (`<entry>@<layer>`). The rest of a record (positions into the padded batch, the `select` indices, which positions are a row's own, a `key` axis's keys) is for `visit` alone, and the engine passes it through. What preflight writes into the schedule is [FORMAT.md](../../FORMAT.md#derived-fields) under `Forward`.

What the engine does with it, deciding nothing:

1. Visit the sites in the order listed.
2. At each site, take the value in the component's own layout (its `dims` in [FORMAT.md](../../FORMAT.md#components)) and call `clab/steps/batches.py` `visit(value, component, records, got)`. It does the site's records in turn, a write on the value as the records before it left it and a read taking that value into `got`, and returns the value and whether anything wrote. Set the value back only when it wrote.
3. Where the engine `stops`, end the forward right after the `stop` site; the sites after it are not visited. An engine that runs every forward whole gets the same values, later, and declares `stops = False`.
4. Gather nothing a record does not ask for, and return `got`. A forward that ended before one of its reads fails in the step, naming the reads it lacks (`clab/steps/__init__.py` `forward`).

In a generation, forward call 0 is the prefill and call k the decode step that takes generated token k-1 as input, and the generation runs whole. At each call the schedule's `loop` names, the engine filters each site's records with `now(records, call)` (it and `landed` are in `clab/steps/batches.py`): a record without `calls` acts on the prefill only, one with `calls` on the decode calls listed (None for every one), a read appending one value per call. At a decode call it then passes them through `landed(records, seen, tokenizer, call)`, `seen` being each decode call's `input_ids` so far (read where the schedule says a write is `online`), which stamps each write with its call and narrows a write scoped to text to the rows it lands on. The steps check that a read collected one value per decode call made (`clab/steps/generate.py` `generated`); a read out of forward order shifts calls and fails there.

The two shipped shapes: `clab/engines/nnterp.py` `NnterpEngine.visited` does this inside an nnsight trace, reading and setting the component on nnterp's envoy and ending the trace with nnsight's `tracer.stop()`; `clab/engines/hooks.py` `HooksEngine.hooked` registers one `hook` per site at the module boundary nnterp names, unflattening a flat `[batch, seq, hidden]` boundary into heads for a component laid out per head, the stop site's hook raising `_Stopped` once its records are done. The ndif engine inherits the nnterp engine's and declares `stops = False`: on a server its forwards run whole.

## Live operands

A write's op argument may be `{"live": "<read name>"}`: the value of a read the same forward call has already made, `got[<read name>]`. `visit` resolves it (`clab/steps/batches.py` `live`), and raises if the engine had not made that read when it reached the write. The schedule lists the read first: preflight refuses a write whose operand is read at a site the engine ranks no earlier than the write's (`clab/compile/schedule.py` `scheduled`), so an engine whose `rank` matches the order its forward really reaches sites never meets that error. An engine that runs a plan without `clab.steps` resolves such an argument the same way.

## Refusing by name

An engine refuses what it cannot do, by name, before any forward:

- an option it does not declare, or of another type, and torchrun around an engine without `tp`: `load`;
- a model-block key: `build` (`OWNED`), or the loader; a checkpoint the loader cannot get: `build`, as `model.key`;
- a `device` or a `dtype` it cannot place or hold: `loading` (`fp8` without a GPU of compute capability 8.9 or later, a bitsandbytes format without bitsandbytes);
- an option value it cannot honor: its constructor raises `ValueError` naming it (`device: the ndif engine cannot honor 'cuda:0': the server places the model`);
- a component: `unserved`, which preflight turns into `<step>.<entry>: the <engine> engine does not serve <component>: <reason>`, as the hooks engine does for a value nnterp reads inside a module's forward.

## Adding an engine

1. A module under `clab/engines/`, one class with `@engine("<name>")`, added to the import line at the end of `clab/engines/__init__.py`.
2. The constructor as above, each option documented on its own docstring line; build the model with `build`, and reuse what `clab/engines/standard.py` shares (`loading`, `placing`, `unserved`, `envoy`, `inputs`, `stops`, `SIZES`).
3. `execute` through `begin` and `run`, the members in the table, and `remote`, `stops` and `cuda`.
4. Regenerate FORMAT.md (`uv run clab describe --markdown`) so its engines section lists the new one.
5. Prove agreement: the corpus documents on the new engine give the nnterp engine's results to the bit. The engine loops to extend are `SERVED` in `tests/test_documents.py`, the `ran` fixture in `tests/test_fit.py` and the `filled` fixture in `tests/test_generate.py`; `tests/test_refusals.py` `test_a_loader_refuses_a_model_key_it_cannot_honor` lists the engines its refusals cover. Running the suite is [testing.md](testing.md).
