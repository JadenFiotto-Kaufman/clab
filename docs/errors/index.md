---
title: Errors Index
one_liner: Refusals keyed by their message, each with its cause and the page that fixes it; and the route to the symptom index when nothing was refused.
tags: [errors, refusals, index]
related: [docs/errors/symptom-index.md, docs/errors/operand-does-not-fit.md, docs/errors/position-not-in-row.md, docs/errors/answer-does-not-line-up.md, docs/errors/alignment-missing.md, docs/concepts/plan-lifecycle.md]
sources: [clab/plan.py, clab/compile/__init__.py, clab/compile/preflight.py, clab/cli.py, clab/engines/__init__.py]
---

# Errors Index

Up: [docs/index.md](../index.md) — all sections.

Find the message you saw. If the run finished, nothing was refused and the numbers look wrong, use [symptom-index.md](symptom-index.md) instead.

`clab run` prints a refusal as `clab: <where>: <what>`. A `<where>` that is a document path (`steps.patched.interventions.patch.pos`, `data.pairs.take`, `featurizers.rot.file_path`) came from reading or compiling the document, before any model loaded; the one exception is `data.<name>: … digest differs`, which the run raises. A step's name without `steps.` (`patched.patch`, `iia`) came from preflight: the model loaded and no forward ran. Inside a fit's body it starts with the fit (`fit.counterfactual.v_cf`), and under an eval with `fit: eval 'held_out', with dataset 'train' read as 'test' (…):`. [plan-lifecycle.md](../concepts/plan-lifecycle.md) says what each stage checks. In Python, Plan and compile raise `PlanError` (`clab/plan.py`), a `ValueError`; preflight raises `ValueError`.

The messages below drop the `<where>:` prefix. `<x>` stands for a name or value from your document.

## The document's own fields

| Message | Cause | Page |
|---|---|---|
| `unknown field '<f>'; valid: …` | a misspelled key, or one on the wrong record | [FORMAT.md](../../FORMAT.md#root-keys) |
| `Field required`, `Input should be …` | a required key is missing (`model`, `steps`), or a value has the wrong type; a path ending `.str` points at a column reference object with a key it does not take | [FORMAT.md](../../FORMAT.md#root-keys) |
| `unknown step kind '<k>'; valid: forward, generate, op, fit` | `kind` names no step kind | [FORMAT.md](../../FORMAT.md#step-kinds) |
| `'reads': a step's reads and writes are one 'interventions' value …` | entries go in one `interventions` dict; an entry with an edit writes | [interventions.md](../usage/interventions.md) |
| `a name has no dots …`; `'<n>' is also a dataset or a featurizer` | a step named with a dot, or like a dataset or featurizer | [writing-a-document.md](../usage/writing-a-document.md) |
| `unknown op '<op>'; valid: …`; `'<op>' takes no '<arg>'; its arguments: …`; `'<op>' needs '<arg>'`; `<arg>: <v> is not <type>`; `<arg>: <v> is not a named value or a number` | an op's arguments are checked against its signature | [FORMAT.md](../../FORMAT.md#ops) |
| `'clamp' needs 'min' or 'max', or both`; `'clamp' min <a> is greater than its max <b>`; `against <v> is not a name, and only a named value is a distribution …` | an op argument it cannot mean | [ops.md](../usage/ops.md) |
| `there is no 'featurizer' op …`; `'<op>' takes no 'featurizer' …`; `'<op>' takes no 'select' …`; `select <v>: an encode selects among a featurizer's coordinates …` | an op on a featurizer's coordinates goes between its `encode` and `decode`; only `encode` selects features | [featurizers.md](../usage/featurizers.md#brackets) |
| `unknown featurizer kind '<k>' …`; `'<kind>' takes no '<f>'; its arguments: …`; `an sae's 'k' is how many latents 'topk' keeps …`; `a 'jumprelu' sae's threshold is a published file's …`; `'topk' keeps k = <k> of <n> latents …` | a featurizer's fields | [featurizers.md](../usage/featurizers.md) |
| `epochs <n>: a fit runs 0 or more epochs …`; `batch_size <n>: a minibatch holds at least one row`; `a sweep cannot stand on a name, and the optimizer's name is one …` | a fit's counts and optimizer | [fit.md](../usage/fit.md) |
| `num_beams <n>; it stays 1: …` (also `num_return_sequences`, `return_dict_in_generate`, `use_cache`) | a generate option that would change the rows or the value | [generate.md](../usage/generate.md) |

## Sites, selectors, positions and sweeps as written

| Message | Cause | Page |
|---|---|---|
| `unknown component '<c>'; valid: …`; `<c> is a block's value and needs 'layers'`; `<c> is the model's own value and takes no layers`; `<c> has no axis '<a>' to select; its axes: …` | the site's component and narrowing | [sites.md](../usage/sites.md) |
| `select key <s>: the key axis is as long as each row, so a selector there picks a fixed number of keys counted from the start …` | a `key` selector counted from the end | [sites.md](../usage/sites.md) |
| `'<s>' is not a selector: …`; `picks [..] name <i> twice`; `'<s>' picks nothing, however many things there are` | the selector grammar: an int, a slice string, a list of distinct ints | [FORMAT.md](../../FORMAT.md#pos) |
| `a scope is one of {'column': …}, {'text': …}, {'turn': …}`; `a scope's list of columns or texts names at least one`; `a column's strip goes on its reference: …`; `a turn's messages are located as the chat template rendered them; it takes no <flag>`; `turn '<r>' is a role; …` | the scope grammar | [positions.md](../usage/positions.md) |
| `scope turn '<r>' is a chat turn of the prompt; it takes frame 0`; `index '<i>' picks none of the <n> tokens of frame '<f>''s decode calls, one each` | a frame and index that can never meet | [positions.md](../usage/positions.md) |
| `a sweep's values are numbers, strings, booleans or lists of them …`; `a sweep cannot stand for '<field>'`; `'<label>': a sweep's label is letters, digits and underscores …` | what a sweep may stand for | [sweeps.md](../usage/sweeps.md) |
| `label '<l>' is also the sweep at <path>, over [...]; one label is one axis …` | two sweeps share a label (a field's name is the default label) but not their values | [sweeps.md](../usage/sweeps.md) |

## Names, data and references (compile)

| Message | Cause | Page |
|---|---|---|
| `a dataset's rows come from one of 'path', inline 'rows' and a hub dataset 'hf' …`; `hf '<x>': a hub dataset is named '<owner>/<name>' …`; `'hf' needs 'revision' …`; `'revision' pins a hub dataset ('hf') …`; `'split' is which split of a file or a hub dataset …`; `take 0: …`; `template: … it takes no 'tokenize'` | the `data` block's fields | [data.md](../usage/data.md) |
| `no file data/<path>.json under the corpus root` | data paths resolve under the nearest directory above the document that holds `data/` | [running.md](../usage/running.md) |
| `the dataset has no rows [in split '<s>']`; `take <n>, and the dataset has <m> rows …`; `'<c>' is not a column of row <i> …`; `'<c>' is a column the rows keep, which this rename would overwrite …` | `split`, `take` and `columns` against the rows | [data.md](../usage/data.md) |
| `<hf> … has splits …; 'split' names the one to read`; `<hf> … is not in the cache, which holds it at <commit>`; `<hf> at revision <r>: Couldn't reach '<hf>' on the Hub (OfflineModeIsEnabled)` | a hub dataset as the hub or the offline cache has it | [data.md](../usage/data.md) |
| `data/<path>.json is not what the run read (its rows' digest differs)`; `<hf> gives other rows here than compile read …` | the rows changed between compile and the run, or since a loaded run | [data.md](../usage/data.md) |
| `no dataset '<d>'; datasets: …`; `dataset '<d>' has no column '<c>' on every row; columns: …`; `column '<c>' has no entry <i> on every row`; `'<ref>' is not on row <i> of '<d>'` | a column reference that is not on every row | [data.md](../usage/data.md) |
| `'<col>' is <type> on row <i> of '<d>'; a column read as text is a string, or a list of them (a set)`; `'<part>' is not text on row <r> …; a list input joins text` | a column read as text holds something else | [data.md](../usage/data.md) |
| `no step reads a chat column of '<d>', and a template renders a chat … only` | `template` on a dataset no step reads as a chat | [data.md](../usage/data.md) |
| `'<x>' names no value produced before this step; known: …` | a misspelled name, or its producer comes later in the document | [writing-a-document.md](../usage/writing-a-document.md) |
| `'<n>' is saved twice`; `'<n>' names no step's value; a save is a step or a step's member: …` | `saves` names a step or `<step>.<entry>` once | [writing-a-document.md](../usage/writing-a-document.md) |
| `no site '<s>'; declared: …`; `no intervention bundle '<b>'; declared: …`; `two entries named '<k>'; a step binds each name once` | a site or bundle name, or an entry name used twice in one step | [interventions.md](../usage/interventions.md) |
| `'<step>.<e>' names step '<step>''s entry '<e>', which is also this bundle's own entry; name the bundle's own entry bare …` | inside a bundle, its own entries are named bare | [interventions.md](../usage/interventions.md) |
| `'<x>' has a row per example of '<a>' and '<y>' of '<b>'; an op takes one dataset's rows` | an op mixes values or columns of two datasets; two datasets that read the same file are two | [interventions.md](../usage/interventions.md#operands) |

## Positions

| Message | Cause | Page |
|---|---|---|
| `frame <f>: a forward makes one call, the prefill …` | a decode frame on a `forward` step | [positions.md](../usage/positions.md) |
| `the input <i> is text, not a chat: a turn scope names a chat's messages …` | `turn` on a text input | [positions.md](../usage/positions.md) |
| `<c>'s key axis grows with every decode call; read it in the prefill` | an attention pattern or scores beyond frame 0 | [generate.md](../usage/generate.md) |
| `a write beyond the prefill lands as its call runs, before the token it would predict is chosen …` | `before` on a write in decode calls | [generate.md](../usage/generate.md) |
| `frame '<f>': a scoped write beyond the prefill is found call by call …`; `index '<i>': a scoped write lands at the call fed its text's last token …`; `occurrence <o> counts from the last occurrence …` | a scoped write in a generation: decode frames, index -1, occurrences from the first | [generate.md](../usage/generate.md) |
| `frame '<f>' with index <i> counts back from the row's last position …`; `… over a frame that holds the prompt a write's index is ':' …`; `… names calls up to the row's last; … the step gives max_new_tokens …`; `… names calls that depend on how far each row gets …` | a write in a generation must name its calls before the run | [generate.md](../usage/generate.md) |
| `row <i>: text '<t>' is not in its input`; `… occurs <n> times in the row's input; it has no occurrence <o>`; `row <i>: tokens <idx> do not fit its <n> scope tokens`; `row <i>: the write's positions hold no token of this row …`; `… begins the row, at position 0, which no position comes before`; `row <i>: the row has no '<role>' message` | a prefill position that names nothing on one row | [position-not-in-row.md](position-not-in-row.md) |

## Model sizes and engines (preflight)

| Message | Cause | Page |
|---|---|---|
| `layer <n>; the model has <m> layers`; `layers '<s>' pick none of the model's <m> layers` | `layers` outside the model | [sites.md](../usage/sites.md) |
| `select <axis> <s> outside the model's <n>`; `select <axis> '<s>' keeps none of the model's <n>`; `select key <k> outside the <n> tokens of the input's shortest row` | `select` outside the model's sizes, or a key past the shortest row | [sites.md](../usage/sites.md) |
| `the nnterp engine does not serve <c>: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'` | the attention interior needs eager attention | [sites.md](../usage/sites.md) |
| `the hooks engine does not serve <c>: nnterp reads it inside a module's forward (…), not at a module boundary` | hooks reaches module boundaries only | [engines.md](../usage/engines.md) |

## Entries, ops and operands

| Message | Cause | Page |
|---|---|---|
| `'<edit>' is an edit, which writes back, and it follows '<op>', which turned the value into […] …` | an edit after a transform, metric or reduction in one entry | [interventions.md](../usage/interventions.md) |
| `an entry that writes selects on one axis; this one selects <a>, <b>` | a write's site narrows two axes | [interventions.md](../usage/interventions.md) |
| `'<x>' is this entry or one this forward makes after it; an operand names an earlier entry`; `'<x>' is an entry of this generation; an operand from an entry of its own step is served in a forward …`; `'<x>' runs '<op>' once it is off the model …` | an operand from the write's own step is an earlier entry with no op after its edits, and only in a `forward` | [operand-does-not-fit.md](operand-does-not-fit.md) |
| `readout projects a residual-stream value; <c> is not one`; `readout projects the whole residual stream … keeps only some of hidden_size …` | `readout` on a non-stream component, or after a `select` on `hidden_size` | [ops.md](../usage/ops.md) |
| `'<op>' over 'layers': inside an entry an op acts at each layer …`; `'<op>' aggregates over rows, and inside an entry it sees one batch …`; `'<op>' restores the value a site had before its writes; it runs only inside an entry` | an op in the wrong place: reductions over rows or layers are steps | [ops.md](../usage/ops.md) |
| `a reduction cannot remove '<a>'; the value is […]`; `a reduction of a value laid out […] removes no axis of it …`; `'<op>' needs a value with a '<a>' axis …`; `it ranks the value's 'vocab_size' axis, and the value is laid out […] …` | the op's axes against the value's layout | [ops.md](../usage/ops.md) |
| `contains reads text, a detokenize's value (without tokens); '<x>' is not one` | `contains` on something other than detokenized text | [ops.md](../usage/ops.md) |
| `'<x>' is laid out […]; this write's value is […] here, so an operand is […]`; `'<x>' holds a different number of positions on each row …`; `'<x>' holds <n> position(s) per row and the value '<op>' scores holds <m> …`; `'<x>' is in the site's own layout; this bracket is in the coordinates of '<f>' …` | an operand's layout, width or coordinates against the value it meets | [operand-does-not-fit.md](operand-does-not-fit.md) |
| `operand '<x>' is <n> token(s) wide, narrowed to {…}; this write's selection is <m> wide …`; `operand '<x>' holds num_heads […], and this write's selection holds […] …`; `operand '<x>' holds layers […], and this op acts at layers […] …`; `operand '<x>' has a layers axis, and this op acts at no layer …`; `'<x>' is laid out […]: an op takes an operand's slice of each layer it acts at, and <c> is the model's own value, at no layer …` | an operand's positions, selected indices or layers against the write's | [operand-does-not-fit.md](operand-does-not-fit.md) |
| `operand '<x>' is read at <c> layer <l>, which the <engine> engine's forward reaches no earlier than this write's …`; `operand '<x>' was read on <input>, row <r> of which is <n> tokens; <input> has <m>, so position <i> pairs …`; `… so their key axes do not line up` | an operand read later in the same forward, or on another input of another length | [operand-does-not-fit.md](operand-does-not-fit.md) |
| `<column> row <r>: '<v>' is <k> tokens; the read has <n> position(s) on that row …`; `… stands for <n> ids (a set, or a class under strip or lower) …`; `… its members are … tokens; a set's members are one width …`; `… is no tokens once strip drops …`; `… position <p> predicts '<t>' in the input, which is not token <i> of '<v>' …`; `… a position each, and the value has no pos axis …` | a metric's answer column against the read it scores | [answer-does-not-line-up.md](answer-does-not-line-up.md) |

## Featurizers, bundles and fits

| Message | Cause | Page |
|---|---|---|
| `'<x>' is a featurizer, not a value …`; `'<x>' names no featurizer here; featurizers: …`; `'<f>' is a '<kind>' featurizer, which has no <op>: it serves …`; `a <kind>'s apply takes no 'with' …` | a featurizer named where a value goes, or an op its kind does not serve | [featurizers.md](../usage/featurizers.md) |
| `decode of '<f>' closes no encode …`; `decode of '<f>', and the open encode is of '<g>' …`; `the entry writes back here with the encode of '<f>' (ops[<j>]) still open …`; `'<op>' of '<f>' at ops[<i>] is inside the encode of '<f>' …`; `decode closes an encode of its own entry …; as a step there is none to close …` | encode and decode brackets | [featurizers.md](../usage/featurizers.md#brackets) |
| `a '<kind>' featurizer drawn without a bundle needs '<f>'; or load one with 'file_path'`; `site '<s>' is {…}; the site a featurizer was trained at is a component at one layer …` | a drawn featurizer's fields, or its `site` | [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) |
| `<file> is a published sae, which says neither how it encodes nor where it was trained; state …`; `<file> has no clab header and holds …`; `<file>: W_dec is […]; with W_enc […] it is […]`; `<file> has <n> latents; this document says <m>`; `<file> holds no threshold …`; `<file> holds a JumpReLU threshold; its activation is 'jumprelu', not '<a>'` | a published SAE file against the fields the document states | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file> does not exist under the corpus root; run the document that writes it first`; `<file> is not a safetensors file …`; `<file> holds one bundle per point of a sweep (…) …`; `<file> carries clab header keys (…) but its format is …`; `<file> holds …; a <kind> bundle holds …`; `<file> has no clab header; a <kind> bundle is one clab wrote` | the bundle file itself; a swept training run's file; from `clab show` or `clab table`, a relative `file_path` read from a working directory other than the run's corpus root | [bundles.md](../usage/bundles.md#what-is-checked-and-when), [swept runs](../usage/bundles.md#swept-training-runs), [reading back](../usage/bundles.md#reading-back-a-run-that-loads-a-bundle) |
| `<file>: header '<key>' is '<v>'; this document expects '<w>'`; `<file>: header 'layer' is '<l>'; this document uses '<f>' at <c> layer <m> (…)`; `<file>: header 'width' is '<w>'; this document uses '<f>' at <c>, whose width is <v>` | the bundle's header against the document and the model | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `featurizer '<f>' is used at width <a> here and <b> elsewhere …`; `featurizer '<f>' has <field> <n>, and it is used at width <w> …`; `<file> has no clab header and is <w> wide …`; `featurizer '<f>' was trained at <c> layer <l> (its site); this use is …`; `select features '<s>' keeps none of the <n> features of '<f>'` | a featurizer's width and site against where it is used | [featurizers.md](../usage/featurizers.md) |
| `<file> was trained reading its input at <c> layer <l> (with), and this use gives no 'with' …`; `… and this use's with '<x>' is <c> layer <m>`; `<file>'s header records no input site (with.component) …`; `operand '<x>' is read at positions … and this write lands at …` | a cross-site adapter's `with` | [lora.md](../patterns/lora.md#across-sites-with) |
| `pca of <n> vectors has no rank-<k> basis …`; `pca k <k> counts directions of a value <w> wide …` | `pca`'s `k` against its data and width | [trained-subspaces.md](../patterns/trained-subspaces.md) |
| `classes column '<c>' holds str (…), int (…) …`; `<arg> names a class of probe '<p>', which is compared exactly …`; `<arg> is '<c>' of '<d>', whose row <i> is '<v>', not one of the classes of probe '<p>': […]` | a probe's classes are the column's distinct values, one type, compared exactly | [probing.md](../patterns/probing.md#scoring-classes) |
| `no featurizer '<f>'; declared: …`; `'<f>' is a <kind>, which a bundle fixes …`; `'<f>' is loaded from a bundle, which fixes it …`; `a fit's body holds no fit`; `'<n>' is also a dataset or a featurizer, or has a dot …` | what a fit may train and name | [fit.md](../usage/fit.md) |
| `not a body op or a trained featurizer's scalar; body ops: …`; `'<m>' is not a body op; body ops: …`; `'<label>' names the fit's history or a featurizer it trains …` | `objective`, `early_stop` and `eval` | [fit.md](../usage/fit.md#the-objective) |
| `dataset '<a>' is read as '<b>', and their templates differ …` | an eval replaces a dataset with one of another `template`; a body that reads text gives neither one | [fit.md](../usage/fit.md#evals) |
| `the body uses '<x>', which has a row per example of the whole dataset …`; `the body's datasets have {…} rows …` | a body runs over minibatches of rows paired across its datasets | [fit.md](../usage/fit.md#the-body) |
| `'<f>' is trained but used at no site, so its width is unknown`; `an sae's scalars are over what the body's encodes of it saw; this body has none` | a trained featurizer the body never reaches, or a scalar with no `encode` to measure | [fit.md](../usage/fit.md#the-objective) |

## Loading, running and reading back

| Message | Cause | Page |
|---|---|---|
| `the <engine> engine takes no '<opt>'; its options: …`; `unknown engine '<e>'; valid: nnterp, hooks, ndif`; `remote: '<v>' is not one of True, 'local'` | an engine option | [engines.md](../usage/engines.md) |
| `device: the ndif engine cannot honor '<d>': the server places the model`; `device: the ndif engine places a local model on one torch device …` | `--device` under `ndif` | [engines.md](../usage/engines.md) |
| `model.<key>: the <engine> engine cannot honor it: …`; `model.<key>: the <engine> engine's loader rejects it: …` | a model-block key the loader does not take (`device_map`, `torch_dtype`, a bad `attn_implementation`) | [engines.md](../usage/engines.md) |
| `-o '<x>': an option is KEY=VALUE`; `-o device: it is given as --device`; `-o batch_size: it is given as --batch-size` | the command line's own options | [cli.md](../reference/cli.md) |
| `plan is not compiled; call plan.compile()`; `this engine loaded <key>@<rev>; the plan pins <key>@<rev>`; `the loaded <e> engine runs on <d> with {…}; device and options are given when an engine loads` | the Python calls out of order, or a loaded engine reused with another model block, device or options; a model block that differs only in `dtype` or a loader key shows the same `<key>@<rev>` on both sides | [python-api.md](../reference/python-api.md) |
| `<step>: generate rejected '<opt>': …` | the model's `generate` refused an option the step passed | [generate.md](../usage/generate.md) |
| `'<n>' was not saved; add it to saves`; `'<a>' has a row per example of '<d>' and '<b>' of '<e>'; a table joins the values of one dataset`; `a bundle has no rows to table …`; `[Errno 2] No such file or directory: '<out>/manifest.json'` | `clab table`, `clab show` and `plan.table` | [output.md](../usage/output.md) |

## Errors that are not clab's

These come as a Python traceback, not a `clab:` line.

| Error | Cause | Page |
|---|---|---|
| `FileNotFoundError: … '<document>.json'` | the document path given to `clab run` does not exist | [cli.md](../reference/cli.md) |
| `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.` | offline (`HF_HUB_OFFLINE=1`) with the checkpoint's `revision` not in the cache | [engines.md](../usage/engines.md) |
| `RuntimeError: Expected one of cpu, cuda, … device type at start of device string: gpu`, or torch's CUDA error | a `--device` torch does not know, or `cuda:0` with no visible GPU, under nnterp or hooks | [engines.md](../usage/engines.md#where-the-model-goes---device) |
| `TypeError: '<=' not supported between instances of 'str' and 'int'` (or another `TypeError` from `generate`) | a generate option of the wrong type, which the model's `generate` raises unnamed | [generate.md](../usage/generate.md) |
| `KeyError: '<name>'` from `plan.table` | the plan has not run; table a filled plan (`clab.run`, or `Plan.load(out)`) | [output.md](../usage/output.md) |

Under `--engine ndif`, preflight and the run happen on the server. With `-o remote=local`, nnsight's local simulation of that trip, `clab run` prints the same `clab:` line as under nnterp; [engines.md](../usage/engines.md) covers a real server.

## Pages in this section

- [symptom-index](symptom-index.md) — Something ran and the numbers are wrong: the silent cases, from indices counted within a select list to a published SAE stated at the wrong layer, each with what it is and where it is explained.
- [alignment-missing](alignment-missing.md) — Why a row is alignment_missing or a value is NaN, how missingness propagates through ops and tables, how to count it, and how to move the question into the prefill, where a miss is refused.
- [operand-does-not-fit](operand-does-not-fit.md) — Why a swap, add or metric refuses the value it is given: an operand meets the value position for position, head for head and layer for layer, and each message says which of those failed and how to fix it.
- [position-not-in-row](position-not-in-row.md) — Why a run refuses with 'row i: … is not in its input' or 'do not fit its N scope tokens': a prefill position that names nothing on one row, how to see that row's tokens, and the fix for each message.
- [answer-does-not-line-up](answer-does-not-line-up.md) — Why a metric refuses its answer column ('is 3 tokens; the read has 1 position', 'no one logit for logit_diff', 'predicts … in the input'): how answers are tokenized and lined up with the read, and the fix for each message.
