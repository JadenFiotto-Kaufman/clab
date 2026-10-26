---
title: Glossary
one_liner: clab's words, each in one sentence with the page that explains it.
tags: [reference, glossary]
related: [docs/reference/index.md, FORMAT.md, docs/usage/writing-a-document.md, docs/concepts/plan-lifecycle.md]
sources: [clab/plan.py, clab/docs.py]
---

# Glossary

Each word as clab uses it, in one sentence, with the page that explains it. Exact fields and signatures are in [FORMAT.md](../../FORMAT.md).

| word | meaning | explained in |
|---|---|---|
| `alignment_missing` | A row that a read or write beyond a generation's prefill did not reach (a call it never made, text it never generated): recorded so in `run.positions`, NaN in every value of it. | [alignment-missing.md](../errors/alignment-missing.md) |
| answer | A column a metric scores against, tokenized exactly as written, each of its tokens scored at one position of the read. | [data.md](../usage/data.md#answer-columns) |
| apply | The op that replaces a value with a featurizer's own version of it (a subspace's projection, a dictionary's reconstruction without its error term, an adapter's adapted value); an edit, so it writes back. | [featurizers.md](../usage/featurizers.md#the-canonical-entry) |
| at | A swept step's place on its sweep axes: its index on each axis in a compiled step, and the coordinate's values in the `at` column of a table. | [sweeps.md](../usage/sweeps.md#reading-swept-results) |
| axis | A named dimension of a value's layout (`rows`, `pos`, `layers`, `num_heads`, `hidden_size`, ...), listed in its `dims`; a sweep label is an axis of points. | [values-and-layout.md](../concepts/values-and-layout.md) |
| batch size | How many rows go through the model in one forward or generation call (`--batch-size`, default 8); results reproduce to the bit only at a fixed one. | [running.md](../usage/running.md#batch-size-and-reproducibility) |
| before | A scope flag that turns the scope into the positions that predict its tokens, the position before each one. | [positions.md](../usage/positions.md#the-token-before-before) |
| body | A fit's own `steps`, run over minibatches of its training rows in a scope of its own. | [fit.md](../usage/fit.md#the-body) |
| bracket | An `encode` and the `decode` that closes it within one entry; the ops between act on the featurizer's coordinates, and only their change is written back. | [featurizers.md](../usage/featurizers.md#brackets) |
| bundle | A safetensors file holding one featurizer's tensors and a header saying what made them, written by a saved fit member or `pca` op and loaded with `file_path`. | [bundles.md](../usage/bundles.md) |
| bundle of entries | A named dict of entries under the root `interventions`, pulled into a step by name, its names resolving in the step that pulls it in. | [interventions.md](../usage/interventions.md#named-bundles) |
| call | One forward pass of a generation: 0 the prefill, k ≥ 1 decode call k; the `call` column of `clab table`. | [generate.md](../usage/generate.md#frames-which-forward-call) |
| chat | An input whose value on each row is a list of `{"role", "content"}` messages, rendered through the checkpoint's own chat template. | [data.md](../usage/data.md#chat-inputs) |
| class | Every vocabulary token that compares the same as an answer token under `strip` or `lower`, scored together; a linear probe's classes are instead the distinct values of its `classes` column. | [data.md](../usage/data.md#answer-columns) |
| column | A field of a dataset's rows, referenced as `<dataset>.<column>` and read as an input, as text to find, or as an answer. | [data.md](../usage/data.md) |
| compile | The pass that checks a document against its data files and bundle headers, with no model, and returns the compiled plan. | [plan-lifecycle.md](../concepts/plan-lifecycle.md#2-compile-the-document-and-its-files) |
| compiled | A plan after compile: sites and bundles of entries inlined, sweeps expanded into points, every name resolved and every layout known. | [plan-lifecycle.md](../concepts/plan-lifecycle.md) |
| component | A named value of the model that a site reads or writes (`layer_output`, `attention_premix`, `logits`, ...), nnterp's name for it, with a fixed layout. | [sites.md](../usage/sites.md#choosing-a-component) |
| coordinate | A point's values on its sweep axes (`pos=-1`, `src=0,dst=1`), which key swept results, files, the run record and tables. | [sweeps.md](../usage/sweeps.md#reading-swept-results) |
| corpus | The documents under `documents/`, which run on CPU and whose pinned numbers are the format's semantics. | [contributing.md](../developing/contributing.md#changing-the-format) |
| corpus root | The nearest directory above a document that holds `data/`; data paths and bundle files resolve against it. | [running.md](../usage/running.md#the-corpus-root) |
| dataset | A named list of rows under `data`, read from a file, written inline, or loaded from the hub. | [data.md](../usage/data.md) |
| decode | The op that closes the entry's most recent open `encode` of the same featurizer and writes the change of coordinates back as a delta. | [featurizers.md](../usage/featurizers.md#the-canonical-entry) |
| decode call | A forward call of a generation after the prefill: decode call k takes generated token k−1 as its one input position. | [concepts/positions.md](../concepts/positions.md#a-generation-is-a-timeline-of-calls) |
| dims | The axis names of a value's layout, recorded on each entry and each result. | [values-and-layout.md](../concepts/values-and-layout.md#rows-and-positions-first) |
| document | One JSON object that declares a whole experiment: the checkpoint, the data, the steps that run in order, and the values to save. | [writing-a-document.md](../usage/writing-a-document.md) |
| edit | An op that keeps a value's layout and changes its contents (`swap`, `add`, `scale`, `noise`, ...); an entry with one writes. | [interventions.md](../usage/interventions.md#edit-recipes) |
| encode | The op that turns a value into a featurizer's coordinates, its last axis `features` (a probe's `classes`). | [featurizers.md](../usage/featurizers.md#the-canonical-entry) |
| engine | The object that loads a checkpoint and runs a compiled plan on it: `nnterp`, `hooks` or `ndif`. | [engines.md](../usage/engines.md) |
| entry | One item of a step's `interventions`: a site, a `pos` and the ops run there, binding `<step>.<entry>`. | [interventions.md](../usage/interventions.md) |
| eval | A named run of a fit's body after training, scored, with datasets read as others (`{"held_out": {"train": "test"}}`). | [fit.md](../usage/fit.md#evals) |
| featurizer | A space or a function declared under `featurizers` and reached from an entry by `encode`, `decode` and `apply`: drawn from a seed, trained by a fit, or loaded from a bundle. | [featurizers.md](../usage/featurizers.md) |
| fit | The step kind that trains featurizers inside the run, through the frozen model. | [fit.md](../usage/fit.md) |
| float32 rule | Metrics and the reductions `mean`, `sum` and `min` compute in float32 and return it whatever the model's dtype; an edit computes in float32 and rounds once to the value's dtype. | [ops.md](../usage/ops.md#the-float32-rule) |
| frame | The first stage of a `pos`: which forward calls, 0 the prefill and k ≥ 1 decode call k. | [generate.md](../usage/generate.md#frames-which-forward-call) |
| header | A bundle's record of what made it (the kind and its fields, the model, the site, the width, the writing document), checked against the loading document. | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| history | A fit's per-epoch record of each body op's mean and each trained featurizer's scalar, `<fit>.history.<name>`. | [fit.md](../usage/fit.md#history) |
| hooks | The engine that drives a plain Hugging Face model with forward hooks and reaches module-boundary components only; the cross-check for nnterp. | [engines.md](../usage/engines.md#cross-checking-with-hooks) |
| index | The third stage of a `pos`: which positions, counted over the tokens the frame and scope left. | [concepts/positions.md](../concepts/positions.md#three-stages-frame-scope-index) |
| input | The column, or list of columns and literals, that a `forward` or `generate` step runs over, one prompt per row. | [data.md](../usage/data.md#inputs) |
| kind | Which featurizer a declaration is (`subspace`, `gate`, `pca`, `sae`, `linear`, `lora`), or which a step is (`forward`, `generate`, `op`, `fit`). | [featurizers.md](../usage/featurizers.md) |
| label | The name of a sweep's axis: its `label`, else the swept field's own key; sweeps with one label zip, with different labels cross. | [sweeps.md](../usage/sweeps.md#labels-zip-or-cross) |
| layers axis | The leading axis a slice or list of `layers` gives a value, labelled with the model's layers it picks, by which ops align operands. | [values-and-layout.md](../concepts/values-and-layout.md#the-layers-axis) |
| manifest | `manifest.json` in an output directory: the compiled plan without rows read from files, bundle tensors or values, with the run record and each result's file, step, dims, axes, shape and dtype. | [output.md](../usage/output.md#the-directory) |
| member | The part of a reference after its first dot: a step's entry, a dataset's column, a featurizer's scalar, or a fit's output. | [names-and-sweeps.md](../concepts/names-and-sweeps.md#a-name-is-a-thing-then-a-member) |
| metric | An op that scores each example, one float32 number per row and position (`match`, `logit_diff`, `token_prob`, `kl`, ...). | [ops.md](../usage/ops.md#metrics) |
| missing | An entry of a value with no data (a row not reached, a position or key past a row's own): NaN in floats, -1 in ids. | [values-and-layout.md](../concepts/values-and-layout.md#missing-entries) |
| NaN rule | A missing entry stays NaN through every metric and edit and -1 through `argmax` and `top_k`, while `mean`, `sum` and `min` reduce over the entries that are there. | [alignment-missing.md](../errors/alignment-missing.md#how-it-propagates) |
| ndif | The engine that runs the whole plan in one session on an NDIF server, holding a weightless model on the client. | [engines.md](../usage/engines.md#running-on-ndif) |
| nnterp | The default engine, nnterp's standardized model inside one nnsight session; also the library whose names are clab's component names. | [engines.md](../usage/engines.md) |
| null control | A run that changes nothing by construction (a self-swap, an edit at its identity, a fit at `epochs: 0`), whose scores the effect is measured against. | [interventions.md](../usage/interventions.md#the-null-control) |
| objective | A fit's weighted sum of terms, each a body op or a trained featurizer's scalar, which the optimizer minimizes. | [fit.md](../usage/fit.md#the-objective) |
| occurrence | Which match of a scope's text the scope is: 0 the first (the default), -1 the last, `":"` every one, ordered by where each ends. | [positions.md](../usage/positions.md#words-and-occurrences) |
| op | A function over a value, run as an `op` step or inside an entry: a metric, a reduction, an edit or a transform. | [ops.md](../usage/ops.md) |
| operand | A value an op takes beside the one it acts on (`swap`'s `with`, `kl`'s `against`): a number, or a named value that must fit the value it meets. | [interventions.md](../usage/interventions.md#operands) |
| plan | The `Plan` a document becomes: its records as written, then compiled, then filled with results and the run record. | [plan-lifecycle.md](../concepts/plan-lifecycle.md) |
| point | One combination of the sweep axes a step depends on, run as a whole step. | [names-and-sweeps.md](../concepts/names-and-sweeps.md#points) |
| pos | An entry's positions, narrowed in three stages: frame, scope, index. | [positions.md](../usage/positions.md) |
| position | Where one input token sits in one row, numbered from 0 with BOS first, the same numbering everywhere a token is counted. | [concepts/positions.md](../concepts/positions.md) |
| prefill | Forward call 0, which holds every prompt token; a `forward` step's only call. | [concepts/positions.md](../concepts/positions.md#a-generation-is-a-timeline-of-calls) |
| preflight | The checks the run makes with the model loaded and before any forward: tokens, sizes, served components, positions, answers and widths. | [plan-lifecycle.md](../concepts/plan-lifecycle.md#4-preflight-the-model-in-hand-no-forward-yet) |
| published SAE | A safetensors file with no clab header holding `W_enc`, `W_dec`, `b_enc`, `b_dec` (and a JumpReLU's `threshold`), which an `sae` loads only with `activation`, `centered` and `site` stated. | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| ragged | A record whose rows hold different numbers of positions: as wide as its widest row, NaN past each row's own. | [positions.md](../usage/positions.md#static-width-or-ragged) |
| read | An entry with no edit among its ops: it binds the value at its selection and leaves the model alone. | [interventions.md](../usage/interventions.md) |
| readout | The transform that projects a residual-stream value through the model's final norm and head: the logit lens. | [logit-lens.md](../patterns/logit-lens.md) |
| reduction | An op that removes axes (`mean`, `sum`, `min`, `pca`), by default the rows and positions, a `layers` axis kept. | [ops.md](../usage/ops.md#reductions-and-over) |
| remote | The `ndif` engine's option: `true` for the NDIF server nnsight is configured for, `"local"` for nnsight's in-process simulation of the trip. | [engines.md](../usage/engines.md#remotelocal) |
| result | A saved value, in `plan.results` and as a file in the output directory; a swept one is a dict keyed by coordinate. | [output.md](../usage/output.md) |
| row | One example of a dataset, a JSON object; a step's values have one row per row of its dataset. | [data.md](../usage/data.md) |
| run option | A setting given when the document runs rather than in it: `--engine`, `--device`, `--batch-size` and `-o key=value`. | [running.md](../usage/running.md#from-the-command-line) |
| run record | `plan.run`: how the plan ran (engine, device, options, versions, dtype) and where every read and write landed (`tokens`, `positions`). | [output.md](../usage/output.md#the-run-record) |
| save | A name in `saves`, a step's value or member, which the run writes to the output directory as one file. | [writing-a-document.md](../usage/writing-a-document.md#saves) |
| scalar | A trained featurizer's own number that a fit's objective may name, `<featurizer>.<scalar>`: a gate's `mask`, an sae's `reconstruction` and `l1`. | [fit.md](../usage/fit.md#the-objective) |
| scope | The second stage of a `pos`, the tokens of a column's value, a literal text or a chat turn; for names, a namespace (the root, or a fit's body). | [positions.md](../usage/positions.md#words-and-occurrences) |
| select | A site's narrowing of a feature axis (`num_heads`, `intermediate_size`, `key`, ...) to some indices, in the order written, keeping the axis. | [sites.md](../usage/sites.md#select) |
| set | An answer row that is a list of strings, any member of which is the answer. | [data.md](../usage/data.md#answer-columns) |
| site | Where an entry reads or writes: a component, its `layers`, and an optional `select`. | [sites.md](../usage/sites.md) |
| static width | A record whose `pos` names the same number of positions on every row, fixed before any row is seen. | [positions.md](../usage/positions.md#static-width-or-ragged) |
| step | One item of `steps`, a `forward`, `generate`, `op` or `fit`, run in document order. | [writing-a-document.md](../usage/writing-a-document.md#the-document-top-to-bottom) |
| sweep | `{"sweep": [...]}` standing for a value, so that every step using it runs once per value. | [sweeps.md](../usage/sweeps.md) |
| table | The rows `clab table` and `plan.table` make of saved values, one per example and position, labelled with tokens and dataset columns. | [output.md](../usage/output.md#clab-table) |
| teacher forcing | A list input that appends each row's answer to its prompt, read at the positions before the answer with `before`. | [generation-experiments.md](../patterns/generation-experiments.md#a-fixed-answer-token-by-token-teacher-forcing) |
| template | A dataset's arguments to the checkpoint's `apply_chat_template` wherever a chat of it renders (`add_generation_prompt`, ...). | [data.md](../usage/data.md#chat-inputs) |
| transform | An op that turns each example's value into another layout (`readout`, `encode`, `argmax`, `top_k`, `detokenize`). | [ops.md](../usage/ops.md#transforms) |
| turn | A scope naming one message of a chat role, `{"turn": "user", "occurrence": 1}`. | [positions.md](../usage/positions.md#chat-turns) |
| write | An entry with an edit among its ops: its value after its last edit replaces the live value at its selection. | [interventions.md](../usage/interventions.md) |
