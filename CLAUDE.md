# clab — Agent Guide

**Start at [docs/index.md](docs/index.md)**, the router for every docs section. Find the one page for the task and read it before writing a document. Something broke? If the run finished, nothing was refused and the numbers look wrong, start at [docs/errors/symptom-index.md](docs/errors/symptom-index.md); for a refusal, look its message up in [docs/errors/index.md](docs/errors/index.md).

## Contributing

- **Base PRs on `master`.**
- Commit messages are one plain declarative sentence.
- A format change is proposed with before-and-after JSON and lands only when the owner ratifies it.
- `FORMAT.md` is generated: regenerate it with `uv run clab describe --markdown` from the repo root (it writes the file; do not redirect) in the same commit as the code.
- Docs change in the same PR as the code that invalidates them.
- Conventions: [docs/developing/contributing.md](docs/developing/contributing.md).

## By task

| I want to... | Read |
|---|---|
| Write my first document | [usage/writing-a-document.md](docs/usage/writing-a-document.md) |
| Load a dataset, inline rows or a chat | [usage/data.md](docs/usage/data.md) |
| Choose a component or layers | [usage/sites.md](docs/usage/sites.md) |
| Point at a token | [usage/positions.md](docs/usage/positions.md), [concepts/positions.md](docs/concepts/positions.md) |
| Read or write at a site | [usage/interventions.md](docs/usage/interventions.md) |
| Score or reduce a value | [usage/ops.md](docs/usage/ops.md) |
| Run one document at many settings | [usage/sweeps.md](docs/usage/sweeps.md) |
| Generate text and score it | [usage/generate.md](docs/usage/generate.md), [patterns/generation-experiments.md](docs/patterns/generation-experiments.md) |
| Train a featurizer, or load one | [usage/featurizers.md](docs/usage/featurizers.md), [usage/fit.md](docs/usage/fit.md), [usage/bundles.md](docs/usage/bundles.md) |
| Run a document and choose an engine | [usage/running.md](docs/usage/running.md), [usage/engines.md](docs/usage/engines.md) |
| Run on an NDIF server | [usage/engines.md](docs/usage/engines.md), [concepts/one-session.md](docs/concepts/one-session.md) |
| Read a run back | [usage/output.md](docs/usage/output.md) |
| Run a technique (patching, ablation, logit lens, steering, DAS, SAEs, probes, LoRA) | [patterns/index.md](docs/patterns/index.md) |
| Fix a refusal | [errors/index.md](docs/errors/index.md) |
| Find out why the numbers are wrong when nothing was refused | [errors/symptom-index.md](docs/errors/symptom-index.md) |
| Look up an exact field, op, component or option | [FORMAT.md](FORMAT.md), [reference/index.md](docs/reference/index.md) |
| Extend clab (an op, a featurizer kind, a component, a step kind, an engine) | [developing/extending.md](docs/developing/extending.md), [developing/engines.md](docs/developing/engines.md) |
| Understand why it behaves this way | [concepts/index.md](docs/concepts/index.md) |

## Before writing a clab document

- **An entry with an edit in `ops` writes; one without `ops` only reads.** A write whose `ops` list is lost runs and returns exactly the unpatched numbers. See [usage/interventions.md](docs/usage/interventions.md).
- **Write each answer column as the model continues the prompt, leading space included.** On a byte-level BPE tokenizer `a` and ` a` are two valid one-token answers, and the wrong one is scored without a refusal. See [usage/data.md](docs/usage/data.md#answer-columns).
- **Check `run.positions` before trusting a scoped read, and scope a teacher-forced answer with `"occurrence": -1`.** `occurrence` defaults to the first match, which is the prompt's copy when the prompt says the answer too. See [usage/positions.md](docs/usage/positions.md#check-where-it-landed).
- **Give each of two sweeps of one field its own `label`.** Unlabelled, they share the field's name and zip into the diagonal instead of a grid. See [usage/sweeps.md](docs/usage/sweeps.md#labels-zip-or-cross).
- **To patch one layer at a time, sweep `layers`; a slice or list on a write patches every picked layer in one forward.** Sweep a patch's read `pos` together with its write `pos`, or the last token's value lands at every position. See [usage/interventions.md](docs/usage/interventions.md#writes-across-layers), [patterns/activation-patching.md](docs/patterns/activation-patching.md#a-layer-by-position-map).
- **After a fit, name the trained featurizer `<fit>.<name>`.** The bare name is the untrained draw, and it runs without a refusal. See [usage/fit.md](docs/usage/fit.md#members).
- **Write `do_sample: false` on every generate step.** Options a step leaves out come from the checkpoint's generation config, which may sample, and nothing seeds sampling. See [usage/generate.md](docs/usage/generate.md#options).
- **Count `alignment_missing` rows before reading a mean over a generation.** Reductions skip the rows a read never reached, and nothing in the number says so. See [errors/alignment-missing.md](docs/errors/alignment-missing.md#count-it-prevent-it-or-refuse-it).
- **State a published SAE's `site`, `centered` and `activation` from its release, and compare the measured L0 with the release's.** The file says none of them, and a wrong value runs. See [usage/featurizers.md](docs/usage/featurizers.md#loading-a-published-sparse-autoencoder).
- **Compare runs to the bit only at one `--batch-size` and one thread count.** A different batch shape moves the last bits. See [usage/running.md](docs/usage/running.md).

## Before changing clab

- **A check that needs the model belongs in preflight, never in compile.** Compile runs on the client under `ndif`; preflight runs where the model is. See [developing/architecture.md](docs/developing/architecture.md#where-the-code-holds-the-five-constraints).
- **`@op` alone registers nothing: add the op to the ordered list at the end of `clab/ops/__init__.py`**, and a new featurizer kind to the import line at the end of `clab/featurizers/__init__.py`. See [developing/extending.md](docs/developing/extending.md).
- **Add a component only under a name nnterp's `support()` lists.** Otherwise the nnterp engine counts it as served and the hooks engine fails to load for every document. See [developing/extending.md](docs/developing/extending.md#adding-a-component).
- **Run the suite at the machine's default thread count.** Its pins were made there, and forcing `OMP_NUM_THREADS` to 1 or 2 moves two of them. See [developing/testing.md](docs/developing/testing.md).
