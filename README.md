# clab

clab runs interpretability experiments on transformer language models. An experiment is one JSON document:
the checkpoint, the data, the forwards and generations to run, the reads and writes on the model's internals,
the ops that score what comes out, and what to save. The same document runs unchanged on a model loaded
locally or on an [NDIF](https://ndif.us/) server. A run writes each saved value to a directory with a
manifest, which a person reads as a table and a script loads back as tensors. clab runs on
[nnsight](https://github.com/ndif-team/nnsight) and names a model's components in
[nnterp](https://github.com/ndif-team/nnterp)'s vocabulary.

## The idea

`documents/patching.json`, without its description, is a complete activation-patching experiment:

```json
{
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"pairs": {"path": "weekdays/train"}},
  "sites": {"target": {"component": "layer_output", "layers": 0}},
  "steps": {
    "counterfactual": {
      "kind": "forward",
      "input": "pairs.counterfactual_inputs[0]",
      "interventions": {"v_cf": {"site": "target", "pos": -1}}
    },
    "patched": {
      "kind": "forward",
      "input": "pairs.input",
      "interventions": {
        "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
        "logits": {"site": {"component": "lm_head_output"}, "pos": -1}
      }
    },
    "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
    "logit_diff": {
      "kind": "op",
      "op": "logit_diff",
      "of": "patched.logits",
      "a": "pairs.cf_answer",
      "b": "pairs.base_answer"
    }
  },
  "saves": ["iia", "logit_diff"]
}
```

`model` pins a checkpoint at a revision and a dtype, and `data` names a dataset under `data/`: four rows, each
a base prompt ("If today is Thursday, tomorrow is"), a counterfactual prompt with another day, and both
answers. `sites` names a place in the model, a component at some layers: `target` is the residual stream
leaving block 0. `steps` run in order and name each other's values: `counterfactual` reads `target` at the
last token of the counterfactual prompt, `patched` swaps that value into the base prompt at the same place and
reads the logits, and the two `op` steps score them. `saves` lists what the run writes.

```text
$ uv run clab run documents/patching.json
patching on nnterp (batch size 8, device cpu) -> out/patching
  iia.safetensors              iia              [4, 1]
  logit_diff.safetensors       logit_diff       [4, 1]
  out/patching/manifest.json
$ uv run clab table out/patching iia logit_diff
row  pos  pos_token  input                              cf_answer  iia  base_answer  logit_diff
0    10   ▁is        If today is Thursday, tomorrow is   Sunday    0.0   Friday      -0.027417995035648346
1    8    ▁is        If today is Friday, tomorrow is     Monday    0.0   Saturday    0.18085992336273193
2    8    ▁is        If today is Saturday, tomorrow is   Friday    0.0   Sunday      0.03514503687620163
3    8    ▁is        If today is Sunday, tomorrow is     Saturday  0.0   Monday      -0.19330616295337677
```

`iia` is 1 where the patched model's most likely next token is the counterfactual answer, and `logit_diff` is
the counterfactual answer's logit less the base answer's. The corpus checkpoint has random weights, so these
numbers show the mechanics, not a finding. `pos` and `pos_token` say where the scored read landed: "Thursday"
is three tokens, so row 0's last token is at position 10.

From Python, `clab.run` returns the filled plan:

```python
import clab
from clab import Plan

plan = clab.run("documents/patching.json")      # writes out/patching
plan                                            # Plan(patching, run on nnterp, 2 results)
plan.results["logit_diff"][:, 0]                # tensor([-0.0274,  0.1809,  0.0351, -0.1933])

swept = clab.run("documents/pos_sweep.json")    # the same patch written at positions -1, -2 and -3
list(swept.results["iia"])                      # [-1, -2, -3]: one [4, 1] tensor per point

Plan.load("out/patching").table("iia", "logit_diff")[0]
# {'row': 0, 'pos': 10, 'pos_token': '▁is', 'input': 'If today is Thursday, tomorrow is', 'cf_answer': ' Sunday', 'iia': 0.0, 'base_answer': ' Friday', 'logit_diff': -0.027417995035648346}
```

[docs/usage/running.md](docs/usage/running.md) has the run options and the four calls behind `clab.run`, and
[docs/usage/output.md](docs/usage/output.md) reads a run back.

## What you can do

Each technique has a page with documents that run:

- [Activation patching](docs/patterns/activation-patching.md): swap a position, a window or an entity's tokens from a counterfactual run, and map layer by position.
- [Head and neuron patching](docs/patterns/head-and-neuron-patching.md): patch one attention head or a few MLP neurons at a time.
- [Ablation](docs/patterns/ablation.md): zero, mean, resample, noise and direction ablation.
- [Attention](docs/patterns/attention.md): knock out a head or keys, patch a head's pattern, read queries and keys after RoPE.
- [Logit lens](docs/patterns/logit-lens.md): decode every layer's residual stream in one forward.
- [Steering](docs/patterns/steering.md): add a harvested direction at a dose, sweep the dose, keep it on through generation.
- [Generation experiments](docs/patterns/generation-experiments.md): score what the model says, its confidence as it says it, and teacher-forced answers.
- [Trained subspaces](docs/patterns/trained-subspaces.md): DAS and DBM, against PCA and random subspaces of matched rank.
- [Sparse autoencoders](docs/patterns/sae.md): splice in a published dictionary, ablate its features, or train one.
- [Probing](docs/patterns/probing.md): train a linear probe per layer on a label column.
- [LoRA](docs/patterns/lora.md): train a low-rank adapter through the frozen model and turn it on in chosen steps.
- [Vision-language models](docs/patterns/vision.md): an image in a chat, its tokens as positions, a region of it as keys, and gaze heads found and redirected.

Across techniques:

- [Sweeps](docs/usage/sweeps.md): `{"sweep": [...]}` where a value goes runs the document at each value on one model load; labels zip or cross two sweeps.
- [Generation](docs/usage/generate.md): a `generate` step reads and writes at the prefill and at each decode call.
- [Fits](docs/usage/fit.md): a `fit` step trains featurizers inside the run, through the frozen model.
- [Bundles](docs/usage/bundles.md): a trained featurizer is saved as a file that another document loads, its header checked against that document before anything runs.
- [Engines](docs/usage/engines.md): `nnterp` (the default), `hooks` (plain transformers with forward hooks, to cross-check) and `ndif`.

## Install

clab is not on PyPI. From a checkout of this repository:

```bash
uv sync
```

It needs Python 3.10 or later; `uv sync` installs nnsight, nnterp, torch and the rest into the checkout's
`.venv`, and `uv run clab ...` runs inside it. A GPU is optional: the documents under `documents/` run on CPU
on tiny random-weight checkpoints, and those under `documents/real/` run Llama-3.2-1B, a gated checkpoint, and
want one. The first run of a document downloads its checkpoint from the Hugging Face hub; after that,
`HF_HUB_OFFLINE=1` runs it from the local cache.

## Run on an NDIF server

`--engine ndif` sends the whole run as one request to the NDIF server nnsight is configured for (`NDIF_HOST`
and `NDIF_API_KEY`, or `nnsight.CONFIG`), and writes the output directory locally from what comes back. The
server needs clab installed at your version and the document's checkpoint deployed at its revision.
`-o remote=local` makes the same trip in this process without a server, and on `documents/patching.json` it
gives the nnterp engine's numbers to the bit:

```bash
uv run clab run documents/patching.json --engine ndif -o remote=local
```

[docs/usage/engines.md](docs/usage/engines.md) has what each engine serves, devices, and what a server must have.

## Documentation

[docs/index.md](docs/index.md) is the router: find the section, then the one page for the task.

- [docs/usage/](docs/usage/index.md): one page per part of the format and per stage of a run; start with [writing-a-document](docs/usage/writing-a-document.md).
- [docs/patterns/](docs/patterns/index.md): one technique per page, each with documents that run.
- [docs/concepts/](docs/concepts/index.md): why it behaves as it does: the plan's lifecycle, one session per run, positions, values and layout, names and sweeps.
- [docs/errors/](docs/errors/index.md): refusals by their message, and [the symptom index](docs/errors/symptom-index.md) for wrong numbers nothing refused.
- [docs/reference/](docs/reference/index.md): the CLI, the Python API and the glossary.
- [docs/developing/](docs/developing/index.md): how clab is built ([architecture](docs/developing/architecture.md)), how to add an op, a featurizer kind, a component or a step kind ([extending](docs/developing/extending.md)), the engine protocol, testing and contributing.
- [FORMAT.md](FORMAT.md): every field, step kind, op, component, featurizer kind and engine option, generated from the code by `clab describe --markdown`; `uv run clab describe ops` prints one of its tables.

## Agents

[CLAUDE.md](CLAUDE.md) is the agent guide for this repository: it starts at [docs/index.md](docs/index.md),
maps tasks to pages, and lists the mistakes that run without a refusal and cost a conclusion.
[AGENTS.md](AGENTS.md) points to it.

## Contributing

[docs/developing/contributing.md](docs/developing/contributing.md) has the conventions, the rules a format
change follows, and what to run before a pull request. Docs change in the same pull request as the code that
invalidates them.
