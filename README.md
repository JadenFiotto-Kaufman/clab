# clab

clab runs mechanistic-interpretability experiments on transformer language models from declarative JSON
documents. A document names a pinned checkpoint, its datasets, the forwards and generations to run with
their interventions on the model's internals (each entry a site, its positions and the ops it runs there:
an entry with an edit among its ops writes back, and every entry's value is readable by name), the ops that
score or reduce what they take, and what to save. clab reads it
into a `Plan`, compiles the plan (rows loaded, names resolved, sweeps expanded into points), runs it inside one
nnsight session (so the same document runs locally or on a remote nnsight server), and writes each saved value
to a directory with a manifest.

Three engines ship: `nnterp` (primary: nnterp's standardized names on every architecture), `hooks` (plain
HuggingFace with forward hooks, for module-boundary components) and `ndif` (the nnterp engine over a
weightless model, its session run on the NDIF server nnsight is configured for).

## Install

```
uv sync
```

## Run a document

```
uv run clab run documents/patching.json                  # writes out/patching/
uv run clab run documents/patching.json --engine hooks --out out/patching-hooks --batch-size 2
uv run clab run documents/patching.json --engine ndif                # on NDIF (nnsight's CONFIG or env)
uv run clab run documents/patching.json --engine ndif -o remote=local # nnsight's in-process simulation of it
uv run clab run documents/patching.json --device cuda                 # run options: --engine, --batch-size,
uv run clab run documents/patching.json --device auto                 # --device, and -o for the engine's own
uv run clab run documents/generated_answer.json --batch-size 2
```

## Train a featurizer, then apply it

A `fit` step trains featurizers (a rotation's subspace, a mask over units, a probe) inside the run, through
the frozen model: its body is the experiment, run over minibatches, and what it trains is saved as a bundle
another document loads. `das.json` trains a rank-8 rotation and saves it as `fit.rot`; `das_apply.json`
loads that bundle and scores it on the held-out split, checking its header (model, site, kind, `k`, width)
key by key first:

```
uv run clab run documents/das.json --batch-size 2 --out documents/artifacts    # writes fit.rot.safetensors
uv run clab run documents/das_apply.json --batch-size 2                        # loads it
uv run clab table out/das_apply iia                                            # das's held-out iia, to the bit
```

`pca_control.json` loads what `pca_harvest.json` writes the same way, and `sae_train_ablation.json` the
sparse autoencoder `sae_train.json` trains on every token of the weekday prompts after BOS:

```
uv run clab run documents/sae_train.json --batch-size 2 --out documents/artifacts   # fit.dictionary.safetensors
uv run clab run documents/sae_train_ablation.json --batch-size 2                     # ablates latents 0 to 3
```

A LoRA is a featurizer kind too: `lora_train.json` trains a rank-4 adapter at layer 0's output to say each
training prompt's counterfactual answer, and `lora_apply.json` loads it and keeps it on as one named
intervention, `adapted`, which the steps that should run the adapted model name (a forward and a generation)
and the others leave out:

```
uv run clab run documents/lora_train.json --batch-size 2 --out documents/artifacts  # fit.lora.safetensors
uv run clab run documents/lora_apply.json --batch-size 2                            # clean vs adapted, and text
```

A featurizer is reached by three ops inside an entry: `encode` gives its coordinates of the value (`select:
{"features": ...}` keeps some), `decode` closes the most recent `encode` of it and writes the change of
coordinates back, and `apply` is its version of the value, an edit (`{"op": "apply", "featurizer":
"fit.lora"}`). The ops between an `encode` and its `decode` act on the coordinates, so `das.json`'s patch is
`[encode fit.rot, swap, decode fit.rot]`; brackets nest, so a gate trains over PCA coordinates as
`[encode pc, encode fit.mask, swap, decode fit.mask, decode pc]`. An adapter across sites (o_proj) reads
`attention_premix` in an earlier entry of the same forward and applies at `attention_output`, naming that
entry as `with`.

A fit's `eval` names its runs after training: `das.json`'s `{"held_out": {"train": "test"}, "train": {}}`
gives `fit.held_out.iia` and `fit.train.iia`. `documents/artifacts/sae.safetensors`,
the dictionary `sae_feature_ablation.json` loads, is a committed fixture (`uv run python tests/fixtures/sae.py`
writes it again). A fit's members are saved like any value; `fit.history.<op>` is a body op's mean on the
training rows after each epoch:

```python
plan = clab.run("documents/dbm.json", batch_size=2)
plan.results["fit.mask"][0.05]       # the gate trained at L1 weight 0.05: {"theta": tensor [16]}
(plan.results["fit.mask"][0.05]["theta"] > 0).sum()    # the units its hard mask keeps
plan.results["fit.held_out.iia"][0.05]                  # its iia on the test split
```

## Real models

`documents/real/` holds five experiments on `meta-llama/Llama-3.2-1B` in float32, over the 42 weekday pairs
of `data/weekdays_full`: `behavior` (the model answers every prompt), `layer_sweep` (where the day is, block
by block, at the day's token and at the last), `head_patching` (each head of layers 10 to 12: layer 12's head
28 alone moves the answer), `pattern_patching` (that layer's attention patterns, which move nothing) and `das`
(a trained subspace at layer 8, its held-out accuracy by rank). They are ordinary documents, which want a GPU:

```
uv run clab run documents/real/head_patching.json --device cuda --batch-size 16
uv run clab run documents/real/das.json --engine hooks --device cuda --batch-size 16
uv run clab run documents/real/layer_sweep.json --engine ndif --batch-size 16     # on the NDIF server nnsight names
```

Through `ndif`, the server must serve the checkpoint at the document's revision in float32. `pattern_patching`
reads the attention pattern, which needs eager attention: the hooks engine refuses it by name, and so does an
NDIF deployment that runs sdpa. A document in a directory under `documents/` reads the corpus's `data/` as
the others do. `tests/test_real.py` pins their numbers on a GPU, on both engines (its docstring has the
command); the CPU-only suite compiles them against their snapshots and runs none of them.

## Look at a run

```
uv run clab show out/generate_probe                      # each result: its step, dims, axes, shape; how it ran
uv run clab table out/generate_probe p_answer patched    # the saved values as one table, joined on row and step
uv run clab table out/generate_probe p_answer --csv      # the same as CSV on stdout, every cell whole
```

`show` prints the run's engine, device and options, batch size, start time and seconds, the served dtype and
devices, the versions, then one line per
saved value: the step that made it, its `dims`, its sweep axes and its shape (per coordinate when swept).
`table` prints `plan.table(name, ...)`: a row per example and position (`pos`/`pos_token`, and in a
generation the forward `call` that holds it), the point's coordinates `at` when the step is swept, the
dataset columns the step used, and one column per name; a row a read beyond the prefill did not reach says
`alignment_missing` in place of a value. Aligned output cuts long cells short;
`--csv` keeps them whole. Both are `Plan.load(out)` and one call, run from the corpus root (where the
dataset files are).

## From Python

```python
import clab
from clab import Plan

plan = clab.run("documents/patching.json", engine="nnterp", out="out/patching")
plan.results["iia"]                  # tensor [4, 1]: 1 where the patched model's top token is the answer
plan                                 # Plan(patching, run on nnterp, 2 results)

swept = clab.run("documents/pos_sweep.json")
swept.results["iia"][-2]             # the point where the patch landed at pos -2: tensor [4, 1]

back = Plan.load("out/patching")     # the filled plan again, from its directory
back.table("iia", "logit_diff")      # one row per example: row, pos, pos_token, iia, logit_diff, the columns used
```

`clab.run(plan_or_path, engine=..., out=None, batch_size=8, device="cpu", **options)` is the four steps of a
run (`options` are the engine's own: `clab describe engines` lists them), which are also there to take one at
a time:

```python
from clab.engines import load

plan = Plan("documents/pos_sweep.json")       # Plan(pos_sweep, uncompiled): the document as a model
compiled = plan.compile()                     # Plan(pos_sweep, compiled, 3 points)
engine = load("nnterp", compiled.model)       # loads the checkpoint; one engine serves many plans
filled = engine.execute(compiled, batch_size=8)
filled.dump("out/pos_sweep")
```

A plan's methods are views and serialization, never execution: `compile`, `dump`, `Plan.load`, `write` (the
plan as a JSON document, in its current state), `table`; `plan.run` says how it ran (engine, batch size,
device and every engine option; the versions, dtype and devices where the model was; timing; how each input
tokenized). `clab.describe()` prints the format's tables.

## Extending clab

Every extensible thing is a plain function or class under its name, registered by a decorator (ops: declared
by one, and listed in order at the end of `ops/__init__.py`), and its signature is its contract: what a document may write, checked by `inspect.signature` with the valid choices
named in every error.

- **Ops** (`clab/runtime/ops/`, one file per group): `@op(group, needs=..., removes=..., adds=..., bundle=None,
  check=None)` declares the function's group (`metrics`, `reductions`, `edits`, `transforms`), the axes the
  input must carry, the axes it removes (a tuple, or the name of the argument that lists them) and appends,
  when its value is a bundle a featurizer loads, that featurizer's kind, and a rule its arguments keep beyond
  their types; the ordered list at the end of `ops/__init__.py` registers it, and that order is the tables'.
  A parameter annotated `Value`, `Tokens` or
  `Featurizer` names a thing the run binds; `engine`, `original` and `dims` are filled by the run, never by
  a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is `"with"`).
- **Featurizer kinds** (`clab/runtime/featurizers/`, one module each) are classes on the `Kind` protocol,
  registered under their lowercase name: class attributes declare what their bundle holds and what a fit
  trains, the constructor's signature is the document's fields, and the ops it serves are its methods
  `encode`, `replace` (what `decode` runs) and `apply`, `OPS` saying which method each op runs.
- **Step kinds** (`clab/runtime/steps/`, `@kind`) and
  **engines** (`clab/engines/`, `@engine("<name>")`). An engine loads its checkpoint in `__init__` from a
  plan's `model` block, on `device`, with the options its constructor's keyword parameters declare (`-o`, or
  `clab.run`'s keywords); `execute(plan, batch_size=)` is its one entry point. `clab describe engines` lists
  each option with what its docstring says it takes (on nnterp and hooks `device` is one torch device, `cpu`
  or `cuda:0`, or a device map transformers computes, `auto`, `balanced`, ...).

## The output

One file per saved name: `<name>.safetensors` when the value is a tensor or a dict of tensors, `<name>.json`
otherwise. A swept value is one file keyed by coordinate (`pos=-1`, `pos=-2`, ...). `manifest.json` is the
plan itself, without the rows it read from files, with each result replaced by where it is and what it is:

```json
"results": {"iia": {"file": "iia.safetensors", "step": "iia", "dims": ["rows", "pos"], "axes": ["pos"],
                    "shape": {"pos=-1": [4, 1], "pos=-2": [4, 1], "pos=-3": [4, 1]}, "dtype": {...}}}
```

`step` names the step that made the value, and the manifest's `steps` hold that step as compiled (its entries
one dict in execution order, each marked `gathered` when the run takes its value off the model), so a result
links back to what computed it. `run.tokens` holds each input's tokens per row and `run.positions` the token
every entry resolved to, which is how `table` labels `pos_token`.

A row's positions are its input tokens, all of them, in order from 0: BOS where the tokenizer adds one (so
`pos: 0` is BOS on Llama, the first word on GPT-2), a chat template's markers, everything but padding, an
appended EOS included. A
document's `pos`, `run.positions`, a ragged value's `tok`, `table`'s `pos` and `key`, an attention pattern's
`key` axis (each row's own keys, NaN past a shorter row's) and every refusal that counts tokens number them
the same way.

## The format

`FORMAT.md` is the one-page reference: root keys, every record and its fields, step kinds, every op with its
arguments and axis behaviour (grouped as metrics, reductions and edits), every component, featurizer kinds,
the `pos` grammar (three stages: `frame` picks forward calls, the prefill or decode calls, `scope` text
within their tokens, `index` positions within that, every input token a position), names and scoping, sweeps and the output. It is generated from the code (`clab describe
--markdown`); a test fails when it is stale.
`documents/` holds the corpus: the 31 original documents and the example documents added since (44 in all),
and under `documents/real/` the five on a real checkpoint; `data/` their datasets.

## Tests

```
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run pytest tests/ -q
uvx pyright
```

`CLAB_UPDATE_SNAPSHOTS=1` rewrites the checked-in snapshots under `tests/snapshots/`. `tests/test_real.py` runs on a GPU only, with the
command its docstring gives.

Offline, the suite needs these checkpoints in the local Hugging Face cache, at the revisions the documents and
tests pin: `hf-internal-testing/tiny-random-LlamaForCausalLM` (`9fb19125`), `hf-internal-testing/tiny-random-gpt2`
(`71034c5d`) and `trl-internal-testing/tiny-Qwen3ForCausalLM` (`25bb9b89`, the chat cases); `Qwen/Qwen3-8B`'s
tokenizer (`b968826d`) where it is cached, which one test checks the tiny Qwen3's template against and skips
without; and, for `tests/test_real.py`, `meta-llama/Llama-3.2-1B` (`4e20de36`).

Numbers reproduce to the bit at a fixed `--batch-size`. Every batch of an input is padded to that input's
longest row, so a row's numbers do not depend on which rows share its batch; they do depend on the batch's
shape, and a trailing partial batch (a smaller matrix) can differ in the last bits. The snapshots pin the
batch size for that reason.
