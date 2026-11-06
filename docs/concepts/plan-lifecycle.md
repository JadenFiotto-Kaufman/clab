---
title: Plan Lifecycle
one_liner: "What happens between a document and its results, and which stage raised a refusal?"
tags: [concepts, lifecycle, compile, preflight, refusals]
related: [docs/usage/running.md, docs/usage/engines.md, docs/concepts/one-session.md, docs/usage/output.md, docs/errors/index.md, docs/developing/compile-and-run.md]
sources: [nnspec/plan.py, nnspec/compile/__init__.py, nnspec/compile/preflight.py, nnspec/engines/__init__.py, nnspec/engines/vllm.py, nnspec/steps/__init__.py, nnspec/output/__init__.py, nnspec/cli.py]
---

# Plan Lifecycle

## The idea

A document reaches its results through six stages. Each stage knows more than the one before it, decides what that knowledge allows, and refuses whatever it is the first stage able to check. Once you know which stage refused, you know what was wrong and how much had already happened. `nnspec run` and `nnspec.run` go through every stage. The calls behind them, for stopping partway in Python, are in [running.md](../usage/running.md#the-four-calls-behind-nnspecrun); how each stage is built is [developing/compile-and-run.md](../developing/compile-and-run.md).

```text
document ─Plan(path)─▶ plan ─compile()─▶ compiled plan ─engines.load()─▶ engine
   ─execute()─▶ [ checks ▸ preflight ▸ every step ▸ results and run record ] ─dump()─▶ output directory ─Plan.load()─▶ filled plan
```

A plan's `repr`, which `print` gives too, says how far it has gone: `Plan(patching, uncompiled)`, `Plan(patching, compiled, 1 point)`, `Plan(patching, run on nnterp, 2 results)`.

## 1. Plan: the document as records

`Plan(path)` reads the JSON and checks only each record's own shape: its fields and their types, the `pos` grammar, a site's narrowing, step kinds, names without dots. It also refuses `generate` options that would change the rows, such as `num_beams`. Nothing but the document is read.

```text
steps.iia: unknown step kind 'opp'; valid: forward, generate, op, fit, load
```

## 2. Compile: the document and its files

`plan.compile()` returns a new plan with `compiled` true. Compile knows the document, the data files, hub datasets and bundle files. It never needs the model. It:

- reads every dataset's rows. A file dataset's rows go into the plan. A hub dataset contributes only its digest and row count.
- inlines sites and named interventions.
- loads each bundle a featurizer names and checks its header against the document.
- expands every sweep into points.
- resolves every name, and works out every value's layout and labels, the dataset its rows align to, the sweep axes it depends on (`axes`), and which entries the run must take off the model.
- writes which points of each forward or generate step one forward serves: those that agree on the forward's axes, its options' and its writes' (each point's `schedule`, `serves` or `serving`).
- refuses a `generate` option of a type transformers' `GenerationConfig` refuses, and a parameter of `generate` itself.

[FORMAT.md's derived fields](../../FORMAT.md#derived-fields) lists every field compile fills, and the two the run fills (`results` and `run`). Anything decidable from the document, the data and bundle headers is refused here, by document path:

```text
steps.patched.input: no dataset 'pears'; datasets: pairs
data.pairs.take: take 400, and the dataset has 4 rows; take keeps at most as many as there are
```

`nnspec compile DOC` is this stage alone: it prints what would run, or the refusal ([running.md](../usage/running.md#check-before-you-run)). Compiling a compiled plan returns it unchanged. Compile runs before any engine loads, so a document it refuses never costs a model load. Under the `ndif` engine, such a document never reaches the server.

## 3. Load: the engine and its model

`engines.load(name, model, device, **options)` builds the engine. On the command line these are `--engine`, `--device` and `-o`.

- The hooks engine loads the checkpoint the model block pins.
- The nnterp engine (at `tp` 1), `ndif` and `vllm` build a shell with no weights. Each engine that runs the model here loads the weights before the first step of a run that calls the model ([engines.md](../usage/engines.md#where-the-model-goes---device)). A model block key the loader rejects is refused there.
- The engine refuses what it cannot honor by name, before any weight loads: an option, a device, a checkpoint or revision the offline cache lacks, a `dtype` ([engines.md](../usage/engines.md#engine-options--o-keyvalue)). A model block key no engine takes (`device`, `torch_dtype`) was refused at compile; an engine given such a block directly refuses it here.

```text
the hooks engine takes no 'remote'; its options: none (where the model runs is `device`, given as --device)
model.key 'hf-internal-testing/tiny-random-LlamaForCausalLM' at revision '0000000000000000000000000000000000000000': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; …
device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)
```

## 4. Preflight: the model in hand, no forward yet

`engine.execute(plan)` starts the run. Two things come before preflight:

- **`execute` checks the plan against the engine**, on the client, before any session opens (`nnspec/steps/__init__.py` `start_run`). It refuses an uncompiled plan, a model block other than the one the engine loaded, a `grad` entry the engine cannot take, and under ndif loads larger than a request carries. The vllm engine's `execute` first refuses what vLLM cannot do: a fit, a write at the recomputed attention pattern, a generate option its sampling cannot map. These checks read only the plan and the engine's declarations. Under nnterp and ndif the model is still a shell with no weights, and nothing has been sent. [developing/engines.md](../developing/engines.md#execute) lists them all.
- **The run puts every dataset's rows in view.** Rows that are not in the plan are read again: a hub dataset's always, a file dataset's after `Plan.load`. Rows whose digest differs from the one compile recorded are refused:

  ```text
  data.pairs: data/weekdays/train.json is not what the run read (its rows' digest differs)
  ```

Preflight then runs, inside the run and before any forward. It sees the tokenizer, the model's sizes, the components the engine serves, and the order the engine's forward reaches them in. It:

- tokenizes every input and records the tokens in `plan.run.tokens`.
- resolves every prefill position per row, onto each read's and write's record (`positions`).
- turns answer columns into token ids.
- resolves the indices each `select` keeps and the layers a slice picks.
- fixes each featurizer's width.
- completes each forward's `schedule` ([FORMAT.md](../../FORMAT.md#derived-fields)): the sites it visits in the engine's forward order, the reads and writes at each, and where it stops.

It refuses by `<step>.<entry>`, or by `<step>` for an op step, without the `steps.` prefix of a document path. Typical causes:

- a component the engine does not serve;
- a layer or `select` outside the model's sizes;
- a position that does not fit some row, or an answer that is not one token;
- an operand read later in the forward than the write that uses it;
- a bundle at the wrong width or layer;
- on vllm, the head at a position other than the prompt's last, or a step longer than the engine's `max_len`.

```text
counterfactual.v_cf: layer 5; the model has 2 layers
patched.patch: row 0: index 40 does not fit the row's 11 positions at this frame
counterfactual.v_cf: the hooks engine does not serve attention_queries: nnterp reads it inside a module's forward (source.attention_interface_1.inputs), not at a module boundary
```

A preflight refusal costs a model load and no forward.

## 5. The run: every step, in order

The run is `steps.run`. Under nnterp and ndif it runs inside the engine's one nnsight session. Under hooks and vllm it is plain Python; on vllm each batch is one trace.

Each step runs in document order, without gradients except where a fit trains or a forward has `grad` entries. A forward or generation runs once per forward its schedule names, serving every point it serves; an op runs once per distinct point of its value. The engines execute the schedule and decide nothing. The run decides whatever needs the model's values: where each generated row stopped, which positions beyond the prefill a read or write lands at, and every op. It records what each run of a step cost in `plan.run.costs`.

Few refusals are left this late. One is an option the model's `generate` rejects, which names the step (on vllm the engine's `execute` refuses an option it cannot map, before any step runs):

```text
answer: generate rejected 'max_new_tokns': The following `model_kwargs` are not used by the model: ['max_new_tokns'] ...
```

The other is a step that runs out of CUDA memory, which stops the run there: the refusal names the step and what the run holds, and the steps before it are saved ([running.md](../usage/running.md#when-a-step-runs-out-of-memory)).

## 6. Results, dump and load

After the last step, each saved value goes into `plan.results` as a detached CPU copy, keyed by coordinate on the sweep axes it depends on. `plan.run` records how the run went: the engine and its options, the versions that served it, the served dtype and devices, the model's sizes, torch's thread count, every input's tokens and what each step cost. The records hold every position each read and write resolved to, and each forward's schedule. `plan.dump(out)` writes one file per save plus `manifest.json` and, beside it, every input's tokens in `tokens.json`; `Plan.load(out)` reads them back as the filled plan. [output.md](../usage/output.md) covers reading them.

A loaded plan holds no rows from file datasets. `plan.table` reads them again and refuses if the file changed since the run, with the message shown under stage 4. An output directory is written only when the run completes, so a refusal at any stage leaves none, except an out-of-memory stop, which writes the steps before it.

## Where each stage runs

Under nnterp and hooks, every stage runs in your process. So it does under vllm, with the model in vLLM's worker. Under ndif with `remote=true`, Plan, compile, load, `execute`'s checks and dump run on the client. The rows' digest check, preflight and the run happen on the server, inside the one session, and the results and run record come back together. A refusal from the server arrives only after the request has waited in its queue. Why this split keeps the numbers the same is [one-session.md](one-session.md).

## Which stage refused

`nnspec run` prints a refusal as `nnspec: <where>: <what>`, and the `<where>` says which stage raised it. Some prefixes are shared between stages, so match on the first clause too.

| the message starts with | first clause, for example | raised by | what had happened |
|---|---|---|---|
| `no document at …`, `batch_size …`, `padding '…'`, `readout '…'` | `batch_size 0: a batch holds one row or more` | the command line or `nnspec.run`, before the document is read | nothing read |
| a document path: `steps.<step>…`, `data.<name>…`, `saves[i]`, `featurizers.<name>…`, `model: …` | `steps.iia: unknown step kind 'opp'` | Plan or compile | nothing loaded |
| `unknown engine …`, `the <engine> engine takes no …`, `model.key '<key>' at revision …`, `model.<key>: the <engine> engine …`, `device…`, `remote:`, `tp …`, `-o …` | `the hooks engine takes no 'tp'` | load | the document compiled; no weight loaded |
| `plan is not compiled`, `the plan's model is not the one this engine loaded`, a `grad` entry's `<step>.<entry>:`, a load's size `<step>: the values this document loads hold …`, and on vllm a fit, a pattern write and a generate option | `patched.h: a grad entry takes the model's gradient` | `execute`'s checks, on the client before the session | the engine loaded, under nnterp and ndif as a shell with no weights; nothing sent, no forward |
| a step's name without `steps.`: `<step>.<entry>:`, `<step>:`, `<fit>.<step>.<entry>:` | `counterfactual.v_cf: layer 5; the model has 2 layers` | preflight | the model loaded, no forward ran |
| `<step>: generate rejected '<option>'` | `answer: generate rejected 'max_new_tokns'` | the run; on vllm, `execute`, before any step | earlier steps ran, nothing was written |
| `steps.<step>: out of CUDA memory …` | | the run | the steps before it ran and are saved |
| `data.<name>:` saying the rows are not what compile read | `data.pairs: data/weekdays/train.json is not what the run read` | the run, or `plan.table` on a loaded plan | the data changed after compile |

Four forms add to a message:

- **At a sweep point**, compile adds ` at <label>=<value>` to the path: `steps.patched.interventions.patch.pos.frame at frame=1: frame 1: …`. Preflight names the point the same way in each refusal of a step's own checks or its schedule (`counterfactual.v_cf at layers=9: layer 9; the model has 2 layers`). So does every refusal of a featurizer's use, at compile (its bundle's header, a gate's `over`) and in preflight (its width, its layer, a features `select`). A refusal of a component the engine does not serve is of no point and names none.
- **A bundle's entry** is refused at its own path, then ` in <step path>`: `interventions.b.w.ops[0].with in steps.patched: 'nope.v' names no value produced before this step; …`.
- **Several document errors** print one per line, only the first prefixed `nnspec:` (`nnspec: model: Field required`, then `steps.iia: unknown step kind 'bogus'; …`). So do the `datasets` library's two-line messages.
- **`--batch-size`, `--padding` and `--readout` are refused before the document is read**, and so are an `-o` without `=` and one naming a flag (`-o device=cpu`). `--engine`, `--device` and every other `-o` key (`-o remote=local`) are refused after it compiles. A broken document run with `--batch-size 0` reports `batch_size 0: …`; run with `--device gpu`, it reports its own refusal.

Each refusal, with its cause and fix, is listed by message in [errors/index.md](../errors/index.md).
