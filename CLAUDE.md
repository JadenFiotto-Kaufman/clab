# clab — Agent Guide

**Start at [docs/index.md](docs/index.md)**, the router for every docs section. Find the one page for the task and read it before writing a document. Something broke? If the run finished, nothing was refused and the numbers look wrong, start at [docs/errors/symptom-index.md](docs/errors/symptom-index.md); for a refusal, look its message up in [docs/errors/index.md](docs/errors/index.md).

## Contributing

- **Base PRs on `master`.**
- Commit messages are one plain declarative sentence.
- A format change is proposed with before-and-after JSON and lands only when the owner ratifies it.
- `FORMAT.md` is generated: regenerate it with `uv run clab describe --markdown > FORMAT.md` in the same commit as the code.
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

<!-- filled by the orchestrator after the content and usability phases; every line here must be a verified fact -->
