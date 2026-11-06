<p align="center"><img src="./nnspec_logo.png" alt="nnspec" width="300"></p>

<h3 align="center">Interpretability experiments as JSON documents, on top of nnsight and nnterp</h3>

<p align="center"><a href="./docs/index.md"><b>Documentation</b></a> | <a href="https://github.com/JadenFiotto-Kaufman/nnspec"><b>GitHub</b></a> | <a href="https://discuss.ndif.us/"><b>Forum</b></a></p>

nnspec runs interpretability experiments on transformer language models. An experiment is one JSON document: the
checkpoint, the data, the forwards and generations to run, the reads and writes on the model's internals, the ops
that score what comes out, and what to save. The same document runs unchanged on a model loaded locally or on an
[NDIF](https://ndif.us/) server. A run writes a directory that a person reads as a table and a script loads back as
tensors. nnspec runs on [nnsight](https://github.com/ndif-team/nnsight) and names a model's components in
[nnterp](https://github.com/ndif-team/nnterp)'s vocabulary.

## Install

nnspec is not on PyPI; install it from a checkout:

```bash
git clone https://github.com/JadenFiotto-Kaufman/nnspec.git
cd nnspec
uv sync
source .venv/bin/activate
```

It needs Python 3.10 or later; `uv sync` installs nnsight, torch and nnterp 2 into `.venv`. nnterp 2 is not on
PyPI either: `pyproject.toml`'s `[tool.uv.sources]` names a checkout of
[ndif-team/nnterp](https://github.com/ndif-team/nnterp)'s `v2.0.0` branch, so point that path at yours.
[testing.md](docs/developing/testing.md#running-the-suite) says which commit the suite pins, and how to run
`nnspec` from outside the venv.

## Quick start

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

`model` pins a checkpoint at a revision and a dtype. `data` names a dataset under `data/`: four rows, each a base
prompt ("If today is Thursday, tomorrow is"), a counterfactual prompt with another day, and both answers. `sites`
names a place: `target` is the residual stream leaving block 0. `steps` run in order and name each other's values:
`counterfactual` reads `target` at the counterfactual prompt's last token, `patched` swaps it into the base prompt
there and reads the logits, and the two `op` steps score them. `saves` lists what the run writes.

```text
$ nnspec run documents/patching.json
patching on nnterp (batch size 8, device cpu, tp 1) -> out/patching
  iia.safetensors         iia         [4,1]
  logit_diff.safetensors  logit_diff  [4,1]
  out/patching/manifest.json
$ nnspec table out/patching iia logit_diff
row  pos  pos_token  input                              cf_answer  iia  base_answer  logit_diff
0    10   ▁is        If today is Thursday, tomorrow is   Sunday    0.0   Friday      -0.027417995035648346
1    8    ▁is        If today is Friday, tomorrow is     Monday    0.0   Saturday    0.18085992336273193
2    8    ▁is        If today is Saturday, tomorrow is   Friday    0.0   Sunday      0.03514503687620163
3    8    ▁is        If today is Sunday, tomorrow is     Saturday  0.0   Monday      -0.19330616295337677
```

`iia` is 1 where the patched model's most likely next token is the counterfactual answer, and `logit_diff` is the
counterfactual answer's logit less the base answer's. The checkpoint has random weights, so these numbers show the
mechanics, not a finding. `pos` and `pos_token` say where the scored read landed: "Thursday" is three tokens, so row
0's last token is at position 10. `nnspec show out/patching` prints what each step cost, where each entry landed per
row, and each saved value's shape ([output.md](docs/usage/output.md)). From Python, `nnspec.run` returns the filled
plan:

```python
import nnspec
from nnspec import Plan

plan = nnspec.run("documents/patching.json")    # writes out/patching
plan                                            # Plan(patching, run on nnterp, 2 results)
plan.results["logit_diff"][:, 0]                # tensor([-0.0274,  0.1809,  0.0351, -0.1933])

Plan.load("out/patching").table("iia", "logit_diff")[0]
# {'row': 0, 'pos': 10, 'pos_token': '▁is', 'input': 'If today is Thursday, tomorrow is', 'cf_answer': ' Sunday', 'iia': 0.0, 'base_answer': ' Friday', 'logit_diff': -0.027417995035648346}
```

## The idea

**One document is the whole experiment.** It pins the checkpoint at a commit and a dtype, names its data (a file
under `data/`, inline rows, or a hub dataset at a commit), and holds the sites, steps and saves
([writing-a-document](docs/usage/writing-a-document.md)). That file is what you send someone and what you rerun.

**A document is checked before the model loads.** `nnspec compile` reads it with no model and names the document path
of each mistake it catches. Positions, layers and components are checked against the loaded model before the first
forward ([running.md](docs/usage/running.md#check-before-you-run)). [The errors index](docs/errors/index.md) gives
each refusal's cause and the page that fixes it. The controls are documents too: writing back what was read must give
the clean run's numbers ([null control](docs/usage/interventions.md#the-null-control)), and a `grad` with
`{"op": "mul", "with": 0}` appended must be exactly zero
([check the gradient](docs/patterns/attribution-patching.md#check-the-gradient)).

**The engine changes where a document runs, not what it means.** On CPU in float32, `documents/patching.json` saves
equal value files on nnterp, hooks and ndif under `-o remote=local`. On vllm it gives vLLM's kernels' numbers, which
agree with those to vLLM's rounding ([engines.md](docs/usage/engines.md)). A rerun at one batch size, padding,
readout, device and thread count writes byte-identical value files, and the run records each setting
([running.md](docs/usage/running.md#when-two-runs-give-the-same-numbers)).

**Sweeps and fits are part of the run.** `{"sweep": [...]}` where a value goes runs the document at each value on one
model load ([sweeps.md](docs/usage/sweeps.md)). A `fit` step trains featurizers through the frozen model, and the
steps after it read the trained ones as `<fit>.<name>` ([fit.md](docs/usage/fit.md)).

## What you can do

Each technique has a page with documents that run. The corpus under `documents/` (53 in all) runs on CPU on tiny
random-weight checkpoints, and the six under `documents/real/` want a GPU and the gated Llama-3.2-1B.

- [Activation patching](docs/patterns/activation-patching.md): swap a position, a window or an entity's tokens from a counterfactual run, and map layer by position.
- [Head and neuron patching](docs/patterns/head-and-neuron-patching.md): patch one attention head or a few MLP neurons at a time.
- [Attribution patching](docs/patterns/attribution-patching.md): estimate every component's patching effect from one backward, with `grad`.
- [Ablation](docs/patterns/ablation.md): zero, mean, resample (a read's rows permuted), noise, direction and optimal-constant ablation.
- [Expert ablation](docs/patterns/expert-ablation.md): knock out or force away one expert of a mixture of experts, and read its routing, usage and entropy.
- [Attention](docs/patterns/attention.md): knock out a head or keys, patch a head's pattern, read queries and keys after RoPE.
- [Logit lens](docs/patterns/logit-lens.md): decode every layer's residual stream in one forward, or train a tuned lens with a translator per layer.
- [Steering](docs/patterns/steering.md): add a harvested direction at a dose, sweep the dose, keep it on through generation.
- [Generation experiments](docs/patterns/generation-experiments.md): score what the model says, its confidence as it says it, and teacher-forced answers.
- [Trained subspaces](docs/patterns/trained-subspaces.md): DAS and DBM, against PCA and random subspaces of matched rank.
- [Head masking](docs/patterns/head-masking.md): learn which attention heads carry a behavior, one gate unit per head of every layer.
- [Sparse autoencoders](docs/patterns/sae.md): splice in a published dictionary, ablate its features, or train one.
- [Probing](docs/patterns/probing.md): train a linear probe per layer on a label column.
- [LoRA](docs/patterns/lora.md): train a low-rank adapter through the frozen model and turn it on in chosen steps.
- [Cached harvests](docs/patterns/cached-harvest.md): harvest once, save it, and load it in other documents to fit or score without a forward.

Across techniques:

- [Fits](docs/usage/fit.md): a `fit` step trains featurizers inside the run, through the frozen model.
- [Bundles](docs/usage/bundles.md): a trained featurizer is saved as a file that another document loads, its header checked against that document before anything runs.
- [Engines](docs/usage/engines.md): one document on `nnterp`, `hooks`, `ndif` or `vllm` ([below](#engines)).
- [Quantized and sharded models](docs/usage/engines.md#quantized-and-sharded-models): a `model.dtype` of `nf4`, `fp4`, `int8` or `fp8`, and `-o tp=N` under torchrun to shard a model over N ranks.

## Engines

`--engine` picks where a document runs ([engines.md](docs/usage/engines.md)):

- `nnterp`, the default, runs nnterp's standardized model inside one nnsight session, the attention interior included.
- `hooks` runs plain transformers with forward hooks and no nnsight, at module boundaries only, to cross-check a result.
- `ndif` sends the whole run as one request to the NDIF server nnsight is configured for, which needs [nnspec and the checkpoint](docs/usage/engines.md#what-the-server-must-have). `nnspec run documents/patching.json --engine ndif -o remote=local` makes the same trip in this process, with no server.
- `vllm` runs on vLLM's worker on a GPU, one request per row, without gradients.

A quantized `model.dtype` runs on nnterp and hooks, and `-o tp=N` shards a model on nnterp under torchrun
([quantized and sharded models](docs/usage/engines.md#quantized-and-sharded-models)).

## Documentation

[docs/index.md](docs/index.md) is the router: find the section, then the one page for the task.

- [docs/usage/](docs/usage/index.md): a page per part of the format and per stage of a run; start with [writing-a-document](docs/usage/writing-a-document.md).
- [docs/patterns/](docs/patterns/index.md): one technique per page, each with documents that run.
- [docs/concepts/](docs/concepts/index.md): why it behaves as it does: the plan's lifecycle, one session per run, positions, values and layout, names and sweeps.
- [docs/errors/](docs/errors/index.md): refusals by message, and [the symptom index](docs/errors/symptom-index.md) for wrong numbers nothing refused.
- [docs/reference/](docs/reference/index.md): the CLI, the Python API and the glossary.
- [docs/developing/](docs/developing/index.md): the architecture, extending nnspec, the engine protocol, testing and contributing.
- [FORMAT.md](FORMAT.md): every field, step kind, op, component, featurizer kind and engine option, from `nnspec describe --markdown`.

## Agents

The skills that teach an agent nnspec are in [ndif-team/skills](https://github.com/ndif-team/skills): `nnspec`,
`nnspec-patching`, `nnspec-ablation`, `nnspec-lenses-and-attention`, `nnspec-steering-and-generation`,
`nnspec-trained-featurizers` and `nnspec-debugging`. In Claude Code,
`/plugin marketplace add https://github.com/ndif-team/skills.git` then `/plugin install nnspec@ndif-team`; in Codex,
`$skill-installer install https://github.com/ndif-team/skills`. [CLAUDE.md](CLAUDE.md) is for agents working in this
repository, and [AGENTS.md](AGENTS.md) points to it.

## Contributing

[docs/developing/contributing.md](docs/developing/contributing.md) has the conventions and the rules a format change
follows. [docs/developing/testing.md](docs/developing/testing.md) runs the suite: offline, on CPU, on tiny
checkpoints, with the pins it needs. Docs change in the same pull request as the code that invalidates them.

## Citation

nnspec runs on nnsight; if you use it in your research, please cite nnsight:

```bibtex
@article{fiottokaufman2024nnsightndifdemocratizingaccess,
      title={NNsight and NDIF: Democratizing Access to Foundation Model Internals},
      author={Jaden Fiotto-Kaufman and Alexander R Loftus and Eric Todd and Jannik Brinkmann and Caden Juang and Koyena Pal and Can Rager and Aaron Mueller and Samuel Marks and Arnab Sen Sharma and Francesca Lucchetti and Michael Ripa and Adam Belfki and Nikhil Prakash and Sumeet Multani and Carla Brodley and Arjun Guha and Jonathan Bell and Byron Wallace and David Bau},
      year={2024},
      eprint={2407.14561},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14561},
}
```
