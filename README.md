# clab

clab runs mechanistic-interpretability experiments on transformer language models from declarative JSON
documents. A document names a pinned checkpoint, its datasets, the forwards and generations to run with
their writes and reads on the model's internals, the ops that score or reduce what they read, and what to
save. clab reads it
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
uv run clab run documents/patching.json -o device_map=auto            # --device, and -o for the engine's own
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
sparse autoencoder `sae_train.json` trains on every token of the weekday prompts:

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

A featurizer is called by the `featurizer` op in one of its kind's modes: `encode` gives its coordinates and
`apply` its version of the value, `{"op": "featurizer", "featurizer": "fit.lora", "mode": "apply"}`. An
adapter across sites (o_proj) reads `attention_premix` in the same forward and applies at
`attention_output`, naming that read as `with`.

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
run (`options` are the engine's own, each routed by name to what it loads or runs with: `clab describe
engines` lists them), which
are also there to take one at a time:

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
  trains, the constructor's signature is the document's fields, and each mode a kind has (`encode`, `apply`)
  is its method of that name; the `featurizer` op reads the layout each mode produces from `MODES`.
- **Step kinds** (`clab/runtime/steps/`, `@kind`) and
  **engines** (`clab/engines/`, `@engine("<name>")`). An engine loads its checkpoint in `__init__` from a
  plan's `model` block, on `device`, with the options its constructor's keyword parameters declare (what it
  loads with); `execute(plan, batch_size=, ...)` is its one entry point, and its keyword parameters are what it
  runs with. One pool of options (`-o`, or `clab.run`'s keywords) goes to each by name; `clab describe engines`
  lists both.

## The output

One file per saved name: `<name>.safetensors` when the value is a tensor or a dict of tensors, `<name>.json`
otherwise. A swept value is one file keyed by coordinate (`pos=-1`, `pos=-2`, ...). `manifest.json` is the
plan itself, without the rows it read from files, with each result replaced by where it is and what it is:

```json
"results": {"iia": {"file": "iia.safetensors", "step": "iia", "dims": ["rows", "pos"], "axes": ["pos"],
                    "shape": {"pos=-1": [4, 1], "pos=-2": [4, 1], "pos=-3": [4, 1]}, "dtype": {...}}}
```

`step` names the step that made the value, and the manifest's `steps` hold that step as compiled, so a result
links back to what computed it. `run.tokens` holds each input's tokens per row and `run.positions` the token
every positioned read and write resolved to, which is how `table` labels `pos_token`.

## The format

`FORMAT.md` is the one-page reference: root keys, every record and its fields, step kinds, every op with its
arguments and axis behaviour (grouped as metrics, reductions and edits), every component, featurizer kinds,
the `pos` grammar (three stages: `frame` picks forward calls, the prefill or decode calls, `scope` text
within their tokens, `index` positions within that), names and scoping, sweeps and the output. It is generated from the code (`clab describe
--markdown`), and `schema.json` is the plan's JSON Schema (`clab schema`); a test fails when either is stale.
`documents/` holds the corpus: the 31 original documents and the example documents added since (35 in all);
`data/` their datasets.

## Tests

```
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run pytest tests/ -q
uvx pyright
```

`CLAB_UPDATE_SNAPSHOTS=1` rewrites the checked-in snapshots under `tests/snapshots/`.

Numbers reproduce to the bit at a fixed `--batch-size`. Every batch of an input is padded to that input's
longest row, so a row's numbers do not depend on which rows share its batch; they do depend on the batch's
shape, and a trailing partial batch (a smaller matrix) can differ in the last bits. The snapshots pin the
batch size for that reason.
