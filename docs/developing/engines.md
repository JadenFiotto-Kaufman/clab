---
title: Engine Protocol
one_liner: "The engine protocol and how to add an engine: what execute must do, what steps.run asks for, records by site, live operands, and refusing by name."
tags: [developing, engines, protocol]
related: [docs/usage/engines.md, docs/developing/compile-and-run.md, docs/developing/architecture.md, docs/concepts/one-session.md]
sources: [clab/engines/__init__.py, clab/engines/standard.py, clab/engines/nnterp.py, clab/engines/hooks.py, clab/engines/ndif.py, clab/steps/batches.py, clab/steps/__init__.py, clab/steps/generate.py]
---

# Engine Protocol

An engine is the one object that holds a model: it loads a checkpoint, runs batches through it, and does at each site what the run's records say. Everything else a run does is `clab/steps/`, the same for every engine. The protocol is `clab/engines/__init__.py` `Engine`; this page says what each part means and what a new engine must do. Choosing among the shipped engines is [usage/engines.md](../usage/engines.md).

## Registering and loading

An engine is a class decorated with `@engine("<name>")` (`clab/engines/__init__.py` `engine`), whose module is imported by the line at the end of `clab/engines/__init__.py`; that line's order is the order `clab describe engines` and [FORMAT.md](../../FORMAT.md#engines-and-run-options) list engines in.

Its constructor is `__init__(self, model, *, device="cpu", <options>)`. `model` is the plan's `model` block (`clab/plan.py` `Model`); `device` is its one placement argument, read however the engine likes; every other keyword parameter is an option, with a default and a line `<name>: <what it takes>` in the constructor's docstring, which `describe engines` prints (`tests/test_format.py` `test_every_engine_option_says_what_it_takes` fails without it). `load` (`clab/engines/__init__.py` `load`) checks options against that signature before anything loads:

```text
the nnterp engine takes no 'foo'; its options: device
remote: 'yes' is not one of True, 'local'
```

Build the model with `clab/engines/standard.py` `build`: it passes the model block's other keys to the loader as they are, refuses the keys in `OWNED` by name (`model.device_map: the hooks engine cannot honor it: where the model runs is a run option: --device`), and reports a key the loader rejects as that key.

An engine then holds `name`, `spec` (the model block), `model` (anything with `.parameters()`, whose dtypes and devices the run records), `device` (None where it does not place the model, as on a server) and `options` (as given, after defaults), and nothing else: inside a session it is the one client-side object ([concepts/one-session.md](../concepts/one-session.md)).

## `execute`

`execute(plan, batch_size=8)` is the single entry point and the only method an engine must have; it returns the filled plan. An engine that runs through `clab.steps` (all three shipped ones do) does three things, the hooks engine's `execute` being the shortest:

1. `clab/steps/__init__.py` `begin`: refuses an uncompiled plan, or one whose `model` block differs in any key from the one the engine loaded (the message names the key and revision alone, so a plan that differs only in `dtype` or a loader key reads as the same checkpoint on both sides), and starts `plan.run` with the engine, batch size, device and options.
2. `clab/steps/__init__.py` `run`, inside the engine's one session if it has one.
3. `plan.run.seconds` set from `begin`'s start time.

An engine that ignores `clab.steps` has everything `steps.run` does to do itself, preflight included: it loads a hub dataset's rows where it runs with `compile.materialized` (as `steps.held` does), resolves live operands as `visit` does, and fills `plan.results` and `plan.run` the same way.

## What `steps.run` asks of an engine

| member | what it returns |
|---|---|
| `forward(batch, records)` | one forward of the batch (`input_ids`, `attention_mask`, `position_ids`, already padded) with the records done; each read's value by its name |
| `generate(batch, records, options)` | the model's `generate` over the batch with the step's own keys as keyword arguments, the records done at every forward call; the new ids only (the prompt cut off) and each read by name |
| `readout(hidden)` | the model's own unembedding: final norm, head, and whatever the model does after the head |
| `unserved()` | each component in `COMPONENTS` mapped to None (served), the reason it is not, or `{layer: reason}` |
| `rank(component, layer)` | any sortable value placing the component in the forward |
| `sizes()` | the model's sizes, the keys in `clab/engines/standard.py` `SIZES` |
| `tokenizer`, `eos` | the tokenizer; the ids generation stops on (`clab/engines/standard.py` `stops`) |

Preflight uses `unserved`, `sizes`, `rank` and the tokenizer before any forward; the steps use the rest. The batch's tensors go where the model's first parameter is (`clab/steps/batches.py` `home`).

## Records, by site

`records` is one list in the order a step's entries run. A record with `ops` is a write; one without is a read and carries the `name` its value is returned under (`<entry>@<layer>`). Each names its site by `component` and `layer` (None for a model-level component); the rest of the record (positions into the padded batch, the `select` indices, which positions are a row's own, a `key` axis's keys) is for `visit` alone, and the engine passes it through.

What the engine does with them:

1. Group them by site (`clab/engines/standard.py` `by_site`, which keeps each site's records in listed order) and visit the sites in forward order: the nnterp engine sorts them by `rank`, and the hooks engine's hooks fire in that order by themselves.
2. At each site, take the value in the component's own layout (its `dims` in [FORMAT.md](../../FORMAT.md#components)) and call `clab/steps/batches.py` `visit(value, component, records, got)`. It does the site's records in turn, a write on the value as the records before it left it and a read taking that value into `got`, and returns the value and whether anything wrote. Set the value back only when it wrote.
3. Gather nothing a record does not ask for, and return `got`.

In a generation, forward call 0 is the prefill and call k the decode step that takes generated token k-1 as input. At each call the engine filters the records with `now(records, call)` (it and `landed` are in `clab/steps/batches.py`): a record without `calls` acts on the prefill only, one with `calls` on the decode calls listed (None for every one), a read appending one value per call. At a decode call it then passes them through `landed(records, seen, tokenizer)`, `seen` being each decode call's `input_ids` so far, which stamps each write with its call and narrows a write scoped to text (`online`) to the rows it lands on. The steps check that a read collected one value per decode call made (`clab/steps/generate.py` `generated`); a read out of forward order shifts calls and fails there.

The two shipped shapes: `clab/engines/nnterp.py` `NnterpEngine.visited` does this inside an nnsight trace, reading and setting the component on nnterp's envoy; `clab/engines/hooks.py` `hook` does it in a forward hook at the module boundary nnterp names, unflattening a flat `[batch, seq, hidden]` boundary into heads for a component laid out per head.

## Live operands

A write's op argument may be `{"live": "<read name>"}`: the value of a read the same forward call has already made, `got[<read name>]`. `visit` resolves it (`clab/steps/batches.py` `live`), and raises if the engine had not made that read when it reached the write. Preflight has already checked, by `rank`, that the read's site comes first, so an engine whose `rank` matches the order its forward really reaches sites never meets that error. An engine that runs a plan without `clab.steps` resolves such an argument the same way.

## Refusing by name

An engine refuses what it cannot do, by name, before any forward:

- an option it does not declare, or of another type: `load`;
- a model-block key: `build` (`OWNED`), or the loader;
- an option value it cannot honor: its constructor raises `ValueError` naming it (`device: the ndif engine cannot honor 'cuda:0': the server places the model`);
- a component: `unserved`, which preflight turns into `<step>.<entry>: the <engine> engine does not serve <component>: <reason>`, as the hooks engine does for a value nnterp reads inside a module's forward.

## Adding an engine

1. A module under `clab/engines/`, one class with `@engine("<name>")`, added to the import line at the end of `clab/engines/__init__.py`.
2. The constructor as above, each option documented on its own docstring line; build the model with `build`, and reuse what `clab/engines/standard.py` shares (`by_site`, `inputs`, `stops`, `SIZES`).
3. `execute` through `begin` and `run`, and the members in the table.
4. Regenerate FORMAT.md (`uv run clab describe --markdown` from the repo root) so its engines section lists the new one.
5. Prove agreement: the corpus documents on the new engine give the nnterp engine's results to the bit. The engine loops to extend are `SERVED` in `tests/test_documents.py`, the `ran` fixture in `tests/test_fit.py` and the `filled` fixture in `tests/test_generate.py`; `tests/test_refusals.py` `test_a_loader_refuses_a_model_key_it_cannot_honor` lists the engines its refusals cover. Running the suite is [testing.md](testing.md).
