---
title: Errors Index
one_liner: Refusals keyed by their message, each with its cause and the page that fixes it; and the route to the symptom index when nothing was refused.
tags: [errors, refusals, index]
related: [docs/errors/symptom-index.md, docs/errors/operand-does-not-fit.md, docs/errors/position-not-in-row.md, docs/errors/answer-does-not-line-up.md, docs/errors/alignment-missing.md, docs/concepts/plan-lifecycle.md]
sources: [clab/plan.py, clab/compile/__init__.py, clab/compile/answers.py, clab/compile/interventions.py, clab/compile/preflight.py, clab/cli.py, clab/engines/__init__.py]
---

# Errors Index

Up: [docs/index.md](../index.md) — all sections.

Find the message you saw. If the run finished, nothing was refused and the numbers look wrong, use [symptom-index.md](symptom-index.md) instead.

A refusal prints as `clab: <where>: <what>`; find `<what>`'s first clause below, and see [plan-lifecycle.md](../concepts/plan-lifecycle.md#which-stage-refused) for which stage each `<where>` comes from.

Each row quotes a message's first clause without its `<where>:`. `<x>` stands for a name or value from your document, and `…` for the rest of the message.

## The document's own fields

| Message | Cause | Page |
|---|---|---|
| `unknown field '<f>'` | a misspelled key, or one on the wrong record | [FORMAT.md](../../FORMAT.md#root-keys) |
| `Field required` | a required key is missing (`model`, `steps`) | [FORMAT.md](../../FORMAT.md#root-keys) |
| `Input should be …` | a value of none of the field's forms; a field of several forms prints one line naming every form (`input: 5 should be a valid string, or a valid list`) | [FORMAT.md](../../FORMAT.md#root-keys) |
| `unknown step kind '<k>'` | `kind` names no step kind | [FORMAT.md](../../FORMAT.md#step-kinds) |
| `'<f>' is a derived field, which compile writes` | a field compile fills, set in a document: a step's `at` or `schedule`, `batch_size` on a forward or a generate, an entry's `positions`, `gathered`, `dims`, `width` or `space`, an op's `space` | [FORMAT.md](../../FORMAT.md#derived-fields) |
| `a compiled plan holds no sweep` | a document that sets `compiled`, a field the plan derives, beside a sweep; leave `compiled` out | [FORMAT.md](../../FORMAT.md#derived-fields) |
| `'reads': a step's reads and writes are one 'interventions' value` | entries go in one `interventions` dict; an entry with an edit writes | [interventions.md](../usage/interventions.md) |
| `an entry named '<k>': a step's reads and writes are one 'interventions' value, …` | a bundle's entry named `reads` or `writes`; a bundle is one dict of entries, and an entry with an edit writes | [interventions.md](../usage/interventions.md#named-bundles) |
| `a name has no dots` | a step named with a dot | [writing-a-document.md](../usage/writing-a-document.md) |
| `'<n>' is also a dataset or a featurizer` | a step named like a dataset or a featurizer | [writing-a-document.md](../usage/writing-a-document.md) |
| `unknown op '<op>'` | `op` names no op | [FORMAT.md](../../FORMAT.md#ops) |
| `'<x>' takes no '<arg>'` | an argument the op or the featurizer kind does not take; the message lists the ones it does | [FORMAT.md](../../FORMAT.md#ops) |
| `'<op>' needs '<arg>'` | a required argument left out | [FORMAT.md](../../FORMAT.md#ops) |
| `<arg>: <v> is not <type>` | an argument of another type (`p: 'inf' is not float`) | [FORMAT.md](../../FORMAT.md#ops) |
| `<arg>: <v> is not a named value or a number` | a value argument of another form | [FORMAT.md](../../FORMAT.md#ops) |
| `'<arg>' is no argument a document gives: the run supplies it` | an op argument under a name the run supplies the op (`dims`, `layers`, `rows`, `positions`, `eos`, `engine`, `original`: `clab/ops/__init__.py` `SUPPLIED`), on an op step or an entry's op | [FORMAT.md](../../FORMAT.md#ops) |
| `'clamp' needs 'min' or 'max', or both`; `'clamp' min <a> is greater than its max <b>` | `clamp`'s bounds | [FORMAT.md](../../FORMAT.md#edits) |
| `against <v> is not a name` | `kl`'s or `js`'s `against` is not a named value | [ops.md](../usage/ops.md#metrics) |
| `'norm' p <p> is not finite` | an infinite `p`; the largest absolute value is a `max` of `abs` | [ops.md](../usage/ops.md#reductions-and-over) |
| `'norm' p <p> is not greater than 0` | `p` of 0 or below | [ops.md](../usage/ops.md#reductions-and-over) |
| `there is no 'featurizer' op` | a featurizer written as an op: `encode` gives its coordinates, `apply` its version of the value | [featurizers.md](../usage/featurizers.md#brackets) |
| `'<op>' takes no 'featurizer'` | an op on a featurizer's coordinates goes between its `encode` and its `decode` | [featurizers.md](../usage/featurizers.md#brackets) |
| `'<op>' takes no 'select'`; `select <v>: an encode selects among a featurizer's coordinates` | only `encode` selects features, `{"features": <selector>}` | [featurizers.md](../usage/featurizers.md#selecting-features) |
| `unknown featurizer kind '<k>'` | `kind` names no featurizer kind | [FORMAT.md](../../FORMAT.md#featurizer-kinds) |
| `'k' is how many latents activation 'topk' keeps`; `activation 'topk' needs 'k', how many latents it keeps` | `k` without `"activation": "topk"`, or `topk` without `k` | [featurizers.md](../usage/featurizers.md) |
| `'topk' keeps k = <k> of <n> latents` | `k` outside 1 to `latents` | [featurizers.md](../usage/featurizers.md) |
| `a 'jumprelu' sae's threshold is a published file's` | a drawn `jumprelu` sae; load one with `file_path` | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `epochs <n>: a fit runs 0 or more epochs` | negative `epochs` | [fit.md](../usage/fit.md) |
| `batch_size <n>: a minibatch holds at least one row` | a fit's `batch_size` below 1 | [fit.md](../usage/fit.md) |
| `a sweep cannot stand on a name, and the optimizer's name is one` | a swept optimizer `name`; write one fit per optimizer | [fit.md](../usage/fit.md) |
| `<option> <v>; it stays <w>` | a generate option clab holds at one value (`num_beams 2; it stays 1`); the message names the option and says why | [generate.md](../usage/generate.md#options) |

## Sites, selectors, positions and sweeps as written

| Message | Cause | Page |
|---|---|---|
| `unknown component '<c>'` | `component` names nothing clab addresses; `clab describe components` lists them | [sites.md](../usage/sites.md#choosing-a-component) |
| `<c> is a block's value and needs 'layers'` | a block's component without `layers` | [sites.md](../usage/sites.md#layers) |
| `<c> is the model's own value and takes no layers` | `layers` on the model's own value (`logits`, `lm_head_output`) | [sites.md](../usage/sites.md#layers) |
| `<c> has no axis '<a>' to select` | `select` names an axis the component's value lacks | [sites.md](../usage/sites.md#select) |
| `select key <s>: the key axis is as long as each row` | a `key` selector counted from the end | [sites.md](../usage/sites.md#select) |
| `keys relative to the query are a read's` | a key `offset` on a write; a write selects keys counted from the start | [sites.md](../usage/sites.md#select) |
| `select <a> {'offset': <k>}: an offset is relative to the query position, on an attention pattern's key axis alone` | an `offset` on an axis other than `key` | [sites.md](../usage/sites.md#select) |
| `offsets <o> name at least one, each once` | an empty or repeated `offset` list | [sites.md](../usage/sites.md#select) |
| `'<x>' holds <keys>, and … holds <keys>; a key relative to the query pairs with the same offsets alone` | an operand and the value it meets at different keys, one of them at offsets from the query | [sites.md](../usage/sites.md#select) |
| `'<s>' is not a selector` | a selector outside the grammar: an int, a slice string, a list of distinct ints | [FORMAT.md](../../FORMAT.md#pos) |
| `picks <p> name <i> twice` | a list of picks that repeats an index | [FORMAT.md](../../FORMAT.md#pos) |
| `'<s>' picks nothing, however many things there are` | a slice empty at every size (`"3:1"`) | [FORMAT.md](../../FORMAT.md#pos) |
| `'<s>' is a string, which is not read as a number` | a `pos` written as a string (`"-1"`, `"-2:"`): an index is an int, and a slice goes inside an object, `{"index": "-2:"}` | [positions.md](../usage/positions.md#recipes) |
| `a scope is one of {'column': …}, {'text': …}, {'turn': …}` | a scope with none of the three, or two of them | [positions.md](../usage/positions.md) |
| `a scope's list of columns or texts names at least one` | an empty list in a scope | [positions.md](../usage/positions.md) |
| `a column's <flag> goes on its reference` | `strip` or `lower` beside a `column` scope; write `{"column": {"column": …, "strip": true}}` | [positions.md](../usage/positions.md#words-and-occurrences) |
| `<v> is an object; a text is a literal, and its strip and lower go on the scope beside it` | a scope's `text` written as a literal object | [positions.md](../usage/positions.md#words-and-occurrences) |
| `a turn's messages are located as the chat template rendered them; it takes no <flag>` | `strip` or `lower` on a `turn` scope | [positions.md](../usage/positions.md#chat-turns) |
| `turn '<r>' is a role; its messages are its occurrences` | a `turn` that is not a role's name; pick a message with `occurrence` | [positions.md](../usage/positions.md#chat-turns) |
| `scope turn '<r>' is a chat turn of the prompt; it takes frame 0` | a `turn` scope on a decode frame | [positions.md](../usage/positions.md#chat-turns) |
| `index '<i>' picks none of the <n> tokens of frame '<f>''s decode calls, one each` | a frame and an index that can never meet | [generate.md](../usage/generate.md#frames-which-forward-call) |
| `index '<i>' is <w> tokens wide and picks <k> of the <n> tokens of frame '<f>''s decode calls` | a frame and an index that can never meet | [generate.md](../usage/generate.md#frames-which-forward-call) |
| `a sweep's values are numbers, strings, booleans or lists of them, each point keyed by its value` | a sweep over objects or lists of lists. Sweep a scalar inside: a `pos`'s `frame`, `index` or scope `occurrence`; a `select` axis's selector. A scope's `text` takes no sweep: write one entry per text | [sweeps.md](../usage/sweeps.md) |
| `a sweep's values hold no sweep` | a sweep nested inside a sweep's values | [sweeps.md](../usage/sweeps.md) |
| `a sweep cannot stand for '<field>'` | a sweep at a field that takes none; under `model` it ends `a run loads one model: write one document per <field>` | [sweeps.md](../usage/sweeps.md) |
| `'<x>' is not a column reference, …; a scope's column takes no sweep (its occurrence does)` | a sweep at a scope's `column`; write one entry per column, or sweep `occurrence` | [sweeps.md](../usage/sweeps.md) |
| `'<label>': a sweep's label is letters, digits and underscores` | a `label` with other characters | [sweeps.md](../usage/sweeps.md#labels-zip-or-cross) |
| `label '<l>' is also the sweep at <path>, over […]; one label is one axis` | two sweeps share a label (a field's name is the default label) but not their values | [sweeps.md](../usage/sweeps.md#labels-zip-or-cross) |

## Names, data and references (compile)

| Message | Cause | Page |
|---|---|---|
| `a dataset's rows come from one of 'path', inline 'rows' and a hub dataset 'hf'` | a dataset with none of the three, or several | [data.md](../usage/data.md#where-rows-come-from) |
| `hf '<x>': a hub dataset is named '<owner>/<name>'` | an `hf` that is not `<owner>/<name>` | [data.md](../usage/data.md#where-rows-come-from) |
| `'hf' needs 'revision'` | a hub dataset without the commit to load | [data.md](../usage/data.md#where-rows-come-from) |
| `'revision' pins a hub dataset ('hf')`; `'config' is the subset of a hub dataset ('hf')` | `revision` or `config` on a file or inline rows | [data.md](../usage/data.md#where-rows-come-from) |
| `'split' is which split of a file or a hub dataset` | `split` on inline rows | [data.md](../usage/data.md#where-rows-come-from) |
| `take <n>: a dataset keeps at least one row`; `skip <n>: a dataset drops 0 rows or more from the start` | `take` below 1, or a negative `skip` | [data.md](../usage/data.md#narrowing-and-renaming-rows) |
| `template: a chat renders to text, which clab tokenizes; it takes no 'tokenize'` | `tokenize` in a `template` | [data.md](../usage/data.md#chat-inputs) |
| `no file data/<path>.json under the corpus root` | data paths resolve under the nearest directory above the document that holds `data/` | [running.md](../usage/running.md#the-corpus-root) |
| `the dataset has no rows` | an empty file, or an empty `split` of it | [data.md](../usage/data.md) |
| `skip <n>, and the dataset has <m> rows` | a `skip` that leaves no row | [data.md](../usage/data.md#narrowing-and-renaming-rows) |
| `take <n>, and the dataset has <m> rows` | a `take` past the rows left | [data.md](../usage/data.md#narrowing-and-renaming-rows) |
| `'<c>' is not a column of row <i>` | `columns` names a column a row lacks | [data.md](../usage/data.md#narrowing-and-renaming-rows) |
| `'<c>' is a column the rows keep, which this <rename> would overwrite` | a renamed or derived column given a name the rows hold | [data.md](../usage/data.md#narrowing-and-renaming-rows) |
| `row <i>: '<text>' is not in its '<c>'` | a derived column's `after` or `before` absent on a row | [data.md](../usage/data.md#a-column-cut-from-another) |
| `row <i>: '<c>' is <type>, …; a derived column is cut from text` | a derived column whose source is not text | [data.md](../usage/data.md#a-column-cut-from-another) |
| `'<s>' is a string; text written as it is is {'text': '<s>'}` | a derived column's `after` or `before` written as a bare string | [data.md](../usage/data.md#a-column-cut-from-another) |
| `<hf> (…) has splits <s>; 'split' names the one to read` | a hub dataset of several splits, none named | [data.md](../usage/data.md#where-rows-come-from) |
| `<hf> (…) has no such split; its splits: …` | a `split` the hub dataset lacks | [data.md](../usage/data.md#where-rows-come-from) |
| `<hf> (…) at revision <r> is not in the cache, which holds it at <commit>` | offline, a `revision` the cache lacks | [data.md](../usage/data.md#where-rows-come-from) |
| `<hf> (…) at revision <r>: <the datasets library's message>` | the `datasets` library refused the load (offline, a `config` the cache lacks); its text can run to a second line | [data.md](../usage/data.md#where-rows-come-from) |
| `<hf>: column '<c>' holds values that are not JSON (images, audio, bytes)` | a hub dataset with an image, audio or bytes column | [data.md](../usage/data.md#where-rows-come-from) |
| `data/<path>.json is not what the run read`; `<hf> gives other rows here than compile read` | the rows changed between compile and the run | [data.md](../usage/data.md) |
| `no dataset '<d>'` | a reference to a dataset `data` does not declare | [data.md](../usage/data.md) |
| `dataset '<d>' has no column '<c>' on every row` | a column some rows lack | [data.md](../usage/data.md) |
| `column '<c>' has no entry <i> on every row` | an `input` index past some row's list | [data.md](../usage/data.md#inputs) |
| `'<ref>' is not on row <i> of '<d>'` | a column reference that is not on every row | [data.md](../usage/data.md) |
| `'<x>' is not <dataset>.<column> or <dataset>.<column>[i]` | an `input` that names no column | [data.md](../usage/data.md#inputs) |
| `a list input names at least one column, whose dataset its rows are` | a list `input` of texts alone, or an empty one | [data.md](../usage/data.md#list-inputs-teacher-forcing) |
| `'<part>' is not text on row <r> of '<d>'; a list input joins text` | a list input's column that is not text on a row | [data.md](../usage/data.md#list-inputs-teacher-forcing) |
| `'<x>' is a column of '<d>', and the value it goes with has a row per example of '<e>'` | a list input's column, or a scope's `column`, from another dataset than the step's input | [data.md](../usage/data.md#list-inputs-teacher-forcing) |
| `'<x>' is not <dataset>.<column> over a dataset` | a column reference (an answer, a probe's `classes`) whose dataset is not declared; an answer the same on every row is `{"text": …}` | [data.md](../usage/data.md#answer-columns) |
| `'<x>' is not a column reference, '<dataset>.<column>'` | a dataset named without its column, as an answer or a scope's `column` | [data.md](../usage/data.md#answer-columns) |
| `<v> is not a column` | a column field holding a number or another type | [data.md](../usage/data.md#answer-columns) |
| `<v> is not a literal's text` | a literal whose `text` is not a string or a list of strings | [data.md](../usage/data.md#answer-columns) |
| `a literal takes no '<k>'` | a literal with a key other than `text`, `strip` and `lower` | [data.md](../usage/data.md#answer-columns) |
| `<v> is not a boolean` | a `strip` or `lower` that is not `true` or `false` | [data.md](../usage/data.md#answer-columns) |
| `'<col>' is <type> on row <i> of '<d>'` | a column read as text holds something else on a row | [data.md](../usage/data.md#answer-columns) |
| `no step reads a chat column of '<d>'` | `template` on a dataset no step reads as a chat | [data.md](../usage/data.md#chat-inputs) |
| `message <i> (<role>) '<content>' is not where the chat template put the messages before it` | a chat template that changes a message's text, or renders the messages out of order | [data.md](../usage/data.md#chat-inputs) |
| `'<x>' names no value produced before this step` | a misspelled name, or its producer comes later in the document | [writing-a-document.md](../usage/writing-a-document.md) |
| `'<n>' is saved twice` | a name listed twice in `saves` | [writing-a-document.md](../usage/writing-a-document.md) |
| `'<n>' names no step's value; a save is a step or a step's member` | `saves` names something no step makes | [writing-a-document.md](../usage/writing-a-document.md) |
| `'<n>' is the name of a run directory's own <n>.json` | a step saved whole named `manifest` or `tokens` | [writing-a-document.md](../usage/writing-a-document.md) |
| `no site '<s>'` | a site name `sites` does not declare; from `featurizers.<f>.site`, see [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) | [interventions.md](../usage/interventions.md) |
| `no intervention bundle '<b>'` | a bundle name `interventions` does not declare | [interventions.md](../usage/interventions.md#named-bundles) |
| `two entries named '<k>'; a step binds each name once` | an entry name used twice across the bundles and dicts of one step's `interventions` | [interventions.md](../usage/interventions.md#named-bundles) |
| `'<step>.<e>' names step '<step>''s entry '<e>', which is also this bundle's own entry` | inside a bundle, its own entries are named bare | [interventions.md](../usage/interventions.md#named-bundles) |
| `'<x>' has a row per example of '<a>' and '<y>' of '<b>'; an op takes one dataset's rows` | an op mixes values or columns of two datasets; two datasets that read the same file are two | [operand-does-not-fit.md](operand-does-not-fit.md#rows-of-another-dataset) |

## Loading a saved value

| Message | Cause | Page |
|---|---|---|
| `no run at <out> under the corpus root (no manifest.json)` | the `load`'s `from` holds no run | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `<out> ran another model than this document's` | the run's `model` block differs from this document's, a loader key included | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `<out> saved no '<n>'` | a `value` the run did not save | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `<out>/<file> is missing, which its manifest says holds '<n>'` | a saved file gone from the run directory | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `<out>/manifest.json is not a run this clab reads` | a manifest that does not parse as a run | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `<out>/manifest.json: a saved value's record holds no 'axes'`; `<out>/manifest.json: fit '<f>' has an 'eval' and no 'evals'` | the manifest's records lack a field a load reads; rerun the document that wrote it | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' was made over the rows of dataset '<d>', which this document does not declare` | the loading document gives the dataset another name, or none; declare it under the run's name | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' was made over the rows of dataset '<d>', and this document's '<d>' is not the rows <out> read, in their order` | other rows: it ends `<m> rows where it read <n>` after another `take` or `skip`, or `as many rows, whose digest differs` | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' was made over the rows of dataset '<d>', whose chats <out> rendered with template <t>` | the dataset's `template` differs from the run's | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' is swept over '<l>', […] in <out>, and this document sweeps '<l>' over […]` | this document sweeps a label the value depends on over other values | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' is a bundle, which a featurizer of its kind loads, with 'file_path'` | a `pca`'s value or a trained featurizer named in a `load` | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' is in the coordinates of '<f>'` | a value inside a featurizer's encode; load the value it encodes, and encode it here | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<n>' is a generation's value or holds its positions` | a generation's ids, or a read at its decode calls | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |
| `'<l>' loads '<x>.tokens', a top_k's tokens` | an op over a loaded `top_k` member that is text; save `<x>.ids` in the run that made it and load that | [ops.md](../usage/ops.md#transforms) |
| `'<l>' loads '<x>.ids', a top_k's ids over a list of axes` | an op over a loaded `ids` of a `top_k` over a list of axes, one tensor per axis; save the axis's (`<x>.ids.rows`) and load that | [ops.md](../usage/ops.md#transforms) |
| `a fit's body holds no load` | a `load` inside a fit; load before the fit | [cached-harvest.md](../patterns/cached-harvest.md#what-the-load-checks) |

## Positions

| Message | Cause | Page |
|---|---|---|
| `frame <f>: a forward makes one call, the prefill` | a decode frame on a `forward` step | [generate.md](../usage/generate.md#frames-which-forward-call) |
| `the input <i> is text, not a chat: a turn scope names a chat's messages` | `turn` on a text input | [positions.md](../usage/positions.md#chat-turns) |
| `<c>'s key axis grows with every decode call; read it in the prefill` | an attention pattern or scores beyond frame 0 | [generate.md](../usage/generate.md#frames-which-forward-call) |
| `a write beyond the prefill lands as its call runs, before the token it would predict is chosen` | `before` on a write in decode calls | [generate.md](../usage/generate.md#writes-at-generated-text-found-online) |
| `a write's positions beyond the prefill are decided call by call` | `until` on a write in decode calls | [positions.md](../usage/positions.md#up-to-a-text-until) |
| `frame '<f>' holds the prompt, and until with before starts before the frame's first token` | `until` with `before` over a frame that holds the prompt | [positions.md](../usage/positions.md#up-to-a-text-until) |
| `frame '<f>': a scoped write beyond the prefill is found call by call in the generated inputs` | a scoped write whose frame holds the prompt or counts from the row's last call; give decode calls (`"1:"`) | [generate.md](../usage/generate.md#writes-at-generated-text-found-online) |
| `index '<i>': a scoped write lands at the call fed its text's last token` | a scoped write in decode calls at an index other than -1 | [generate.md](../usage/generate.md#writes-at-generated-text-found-online) |
| `occurrence <o> counts from the last occurrence, which is known only when the row ends` | a scoped write in decode calls at an occurrence counted from the end | [generate.md](../usage/generate.md#writes-at-generated-text-found-online) |
| `frame '<f>' with index <i> counts back from the row's last position` | a write over the prompt and every decode call at an index counted from the end; write two entries | [generate.md](../usage/generate.md#writes-during-generation) |
| `frame '<f>' with index <i>: over a frame that holds the prompt a write's index is ':'` | a write over a frame holding the prompt at an index other than `":"` | [generate.md](../usage/generate.md#writes-during-generation) |
| `frame '<f>' with index <i> names calls up to the row's last` | a write over every decode call in a step without `max_new_tokens` | [generate.md](../usage/generate.md#writes-during-generation) |
| `frame '<f>' with index <i> names calls that depend on how far each row gets` | a write whose frame counts from the row's last call, or whose index counts from the end over several decode calls | [generate.md](../usage/generate.md#writes-during-generation) |
| `row <i>: text '<t>' is not in its input` | a `text` scope a row does not hold, on word boundaries | [position-not-in-row.md](position-not-in-row.md#-is-not-in-its-input) |
| `row <i>: column '<c>' ('<v>') is not in its input` | a `column` scope whose value a row does not hold | [position-not-in-row.md](position-not-in-row.md#-is-not-in-its-input) |
| `row <i>: … occurs <n> times in the row's input; it has no occurrence <o>` | an `occurrence` past the row's count | [position-not-in-row.md](position-not-in-row.md#-occurs-n-times--it-has-no-occurrence-k) |
| `row <i>: index <idx> does not fit the row's <n> positions at this frame`; `row <i>: index <idx> does not fit the row's <n> scope tokens` | a static index wider than a row, or than its scope | [position-not-in-row.md](position-not-in-row.md#index--does-not-fit-the-rows-) |
| `row <i>: the write's positions hold no token of this row` | a write at an open-ended index past a short row | [position-not-in-row.md](position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row) |
| `row <i>: … begins the row, at position 0, which no position comes before` | `before` on a scope at a row's first token | [position-not-in-row.md](position-not-in-row.md#-begins-the-row-at-position-0-which-no-position-comes-before) |
| `row <i>: the row has no '<role>' message` | a `turn` scope a row's chat does not hold | [position-not-in-row.md](position-not-in-row.md#the-row-has-no-role-message) |
| `this tokenizer has no offset mapping, which a scope in the prompt needs` | a prompt scope on a tokenizer without offsets (a slow tokenizer) | [positions.md](../usage/positions.md#words-and-occurrences) |

## Model sizes and engines (preflight)

| Message | Cause | Page |
|---|---|---|
| `layer <n>; the model has <m> layers` | a layer outside the model | [sites.md](../usage/sites.md#layers) |
| `layers '<s>' pick none of the model's <m> layers` | a `layers` selector that keeps no layer | [sites.md](../usage/sites.md#layers) |
| `select <axis> <s> outside the model's <n>` | a `select` past the model's size | [sites.md](../usage/sites.md#select) |
| `select <axis> '<s>' keeps none of the model's <n>` | a `select` that keeps nothing at the model's size | [sites.md](../usage/sites.md#select) |
| `select key <k> outside the <n> tokens of the input's shortest row` | a `key` past the shortest row | [sites.md](../usage/sites.md#select) |
| `select <axis>: picks <p> count one of these <n> twice` | two picks that name one index once counted (`-1` and `3` of 4) | [sites.md](../usage/sites.md#select) |
| `the nnterp engine does not serve <c>: read inside the eager attention forward, but this model runs 'sdpa'` | the attention interior needs eager attention | [sites.md](../usage/sites.md#attention-needs-eager) |
| `the hooks engine does not serve <c>: nnterp reads it inside a module's forward` | hooks reaches module boundaries only | [engines.md](../usage/engines.md#cross-checking-with-hooks) |
| `the hooks engine does not serve <c>: nnterp reads it among a module's call arguments` | hooks reaches a module's first input and its output only | [engines.md](../usage/engines.md#cross-checking-with-hooks) |
| `<c> layer <l>: the hooks engine's two-token forward at load did not reach its module` | a module the hooks engine's load-time forward never runs | [engines.md](../usage/engines.md#cross-checking-with-hooks) |
| `the <engine> engine does not serve <c>: block <i>'s MLP is a mixture of experts` | a dense MLP's value (`mlp_activation`, `mlp_neurons`) on a mixture-of-experts block; the message lists what that block serves | [sites.md](../usage/sites.md#a-mixture-of-experts) |
| `the <engine> engine does not serve <c>: no block of this model has a mixture of experts` | a mixture's value on a dense model | [sites.md](../usage/sites.md#a-mixture-of-experts) |
| `the <engine> engine does not serve <c>: no <c> value on this block's mlp` | a mixture's value at a dense block of a model with mixtures elsewhere | [sites.md](../usage/sites.md#a-mixture-of-experts) |
| `the <engine> engine does not serve shared_expert_output: this mixture has no shared expert` | `shared_expert_output` on a mixture without one | [sites.md](../usage/sites.md#a-mixture-of-experts) |
| `the <engine> engine does not serve <c>: nnterp has no family for this architecture` | a mixture on a checkpoint nnterp reads under its default family, which names none of a mixture's values | [sites.md](../usage/sites.md#a-mixture-of-experts) |
| `'<edit>' would compute on expert_indices, the ids the model indexes its num_experts with, as numbers` | an edit other than `swap` at `expert_indices` (at compile) | [expert-ablation.md](../patterns/expert-ablation.md#the-knockout) |
| `swap with <v>: expert_indices holds ids` | a constant that is no expert's id | [expert-ablation.md](../patterns/expert-ablation.md#the-knockout) |
| `writes <v> into expert_indices at row <r>, token <t>` | a named value's entry that is no id, as the write lands; a hole (`writes a hole, -1`) is a -1 in the operand: a step's value that holds one, or a position past a row's own. A generation's read on a row that stopped before its call is refused before the forward instead (`operand '<x>' holds no value on row <r>`) | [expert-ablation.md](../patterns/expert-ablation.md#the-knockout) |
| `the <engine> engine does not serve <c>: no self_attn module on this block` | a block without the component's module (the linear-attention blocks of a hybrid), refused layer by layer | [sites.md](../usage/sites.md) |
| `the nnterp engine does not serve <c>: under tp <n> …` | a value the ranks hold in pieces under tensor parallelism: `it is computed inside <module>`, or `<module> is a vocab-parallel embedding` | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `grad's of '<x>' is read at <c> layer <l>, which the <engine> engine's forward reaches no later than this entry's` | `of` is made before, or at, the gradient's own site; name a later entry, such as one at the head | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `the nnterp engine takes no gradient under tp <n>` | `grad` under tensor parallelism; run it at tp 1 | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `the checkpoint's chat template calls strftime_now, which writes today's date into the prompt` | a chat template that reads the clock; pin the date in `data.<d>.template`, `{"date_string": …}` | [data.md](../usage/data.md#chat-inputs) |
| `nnterp's support() does not name <name>, so nothing says whether it serves <component>` | a component added to `address.COMPONENTS` under a name nnterp's `support()` does not list; every document is refused | [extending.md](../developing/extending.md#adding-a-component) |


## Entries, ops and operands

| Message | Cause | Page |
|---|---|---|
| `'<edit>' is an edit, which writes back, and it follows '<op>'` | an edit after a transform, metric or reduction in one entry | [interventions.md](../usage/interventions.md#the-order-ops-run-in) |
| `an entry that writes selects on one axis` | a write's site narrows two axes | [interventions.md](../usage/interventions.md) |
| `'<x>' is this entry or one this forward makes after it` | an operand from the write's own forward that comes later | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `'<x>' is an entry of this generation` | an operand from an entry of the write's own `generate` step | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `'<x>' runs '<op>' once it is off the model, after this forward` | a same-forward operand with an op after its last edit | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `'<x>' is an entry of this forward, and '<op>' here runs once the value is off the model` | a same-forward operand given to an op that runs after the forward; take both in an op step | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `operand '<x>' is read at <c> layer <l>, which the <engine> engine's forward reaches no earlier than this write's` | a same-forward operand at a site the forward reaches later | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `operand '<x>' holds no position on row <r>` | a same-forward operand at an index a short row does not reach | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `operand '<x>' holds <n> position(s) on row <r> and this write <m>` | a same-forward operand whose width differs from the write's on a row | [operand-does-not-fit.md](operand-does-not-fit.md#an-operand-from-the-same-forward) |
| `'<x>' is laid out […]; '<op>' acts on a value laid out […]` | an operand with an axis the value lacks | [operand-does-not-fit.md](operand-does-not-fit.md#-an-operands-axes-are-among-the-values-) |
| `'<x>' has no width compile can fix (…), so it pairs with no other forward's value` | at compile: a side whose `pos` leaves its width open ([which forms do](../usage/positions.md#static-width-or-ragged)), or a `permute` of one | [operand-does-not-fit.md](operand-does-not-fit.md#-has-no-width-compile-can-fix) |
| `operand '<x>' holds <n> position(s) on rows … and <m> positions on row <r>; a scoped read pairs with another forward's value when every row's scope holds the same number of tokens` | a scope's rows hold different numbers of tokens; at compile, without `operand`, for a loaded read's scope | [operand-does-not-fit.md](operand-does-not-fit.md#-a-scoped-read-pairs-with-another-forwards-value-when-every-rows-scope-holds-the-same-number-of-tokens) |
| `operand '<x>' meets the value '<op>' acts on, which holds …` | the write's own scope holds different numbers of tokens by row | [operand-does-not-fit.md](operand-does-not-fit.md#-a-scoped-read-pairs-with-another-forwards-value-when-every-rows-scope-holds-the-same-number-of-tokens) |
| `'<x>' holds <n> position(s) per row and the value '<op>' … holds <m>` | two static widths that differ | [operand-does-not-fit.md](operand-does-not-fit.md#-holds-n-positions-per-row-and-the-value--holds-m) |
| `'<x>' has a pos axis, and this write lands beyond the prefill` | a per-position operand for a write over decode calls | [operand-does-not-fit.md](operand-does-not-fit.md#-has-a-pos-axis-and-this-write-lands-beyond-the-prefill-) |
| `'<x>' is read at decode call <k>, and '<g>' makes calls 0 to <m-1> (max_new_tokens <m>)` | at compile: a write's operand read at a fixed decode call at or past its generation's `max_new_tokens`, which no row makes | [alignment-missing.md](alignment-missing.md#how-it-propagates) |
| `operand '<x>' holds no value on row <r> (alignment_missing)`; `operand '<y>', which descends from '<x>', holds no value on row <r>` | in the run: a write whose operand is an earlier generation's read, or an op step's value made from one, on a row whose generation never made the call the read names | [alignment-missing.md](alignment-missing.md#how-it-propagates) |
| `'<x>' is in the site's own layout; this bracket is in the coordinates of '<f>'` | an operand inside an `encode` bracket, not encoded | [operand-does-not-fit.md](operand-does-not-fit.md#coordinates) |
| `'<x>' holds num_heads […], and this write's selection holds […]` | an operand's selected indices against the write's | [operand-does-not-fit.md](operand-does-not-fit.md#a-select-list-is-positional) |
| `'<x>' holds layers […], and this op acts at layers […]` | an operand without a slice for every layer the op acts at | [operand-does-not-fit.md](operand-does-not-fit.md#layers) |
| `'<x>' has a layers axis, and this op acts at no layer` | an operand with `layers` for a value at no layer | [operand-does-not-fit.md](operand-does-not-fit.md#layers) |
| `operand '<x>' was read on <input>, row <r> of which is <n> tokens` | an unscoped operand on another input of another length | [operand-does-not-fit.md](operand-does-not-fit.md#two-inputs-of-different-lengths) |
| `grad is taken in a forward, not a generate` | `grad` in a `generate` step | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `grad inside a fit's body is not served` | `grad` in a fit's body | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `grad is no step` | `grad` written as an op step | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `grad is ops[<i>] here, and it is the first op of its entry` | `grad` after another op | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `grad follows '<edit>', an edit, and an entry that takes a gradient writes nothing` | an edit before `grad`; write in an entry of its own listed before it at the site | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `'renormalize' restores the value the site had before its writes, and after grad` | `renormalize` after `grad` | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `decode closes an encode whose edits write back, and after grad nothing writes back` | an `encode` bracket after `grad` | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `'<x>' names no entry of this forward; grad's of is an entry of its own forward, by its bare name` | `of` names an entry elsewhere, or with its step's name (`"metric"`, not `"patched.metric"`) | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `'<x>' is a grad entry, whose value is a gradient the forward never computes` | `of` names another `grad` entry | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `'<x>' runs '<op>', which gives <what>, and no gradient passes through it` | `of` names an entry whose ops end in an op no gradient passes through; the message names it | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| `readout projects a residual-stream value; <c> is not one` | `readout` on a component that is not a stream | [ops.md](../usage/ops.md#transforms) |
| `readout projects the whole residual stream through the model's final norm and head, and <x> keeps …` | `readout` after a `select` on `hidden_size` | [ops.md](../usage/ops.md#transforms) |
| `'<op>' over 'layers': inside an entry an op acts at each layer` | a reduction over `layers` in an entry; make it a step | [ops.md](../usage/ops.md#a-step-or-an-entry) |
| `'<op>' aggregates over rows, and inside an entry it sees one batch` | a reduction over rows in an entry; make it a step | [ops.md](../usage/ops.md#a-step-or-an-entry) |
| `'<op>' restores the value a site had before its writes; it runs only inside an entry` | `renormalize` as a step | [ops.md](../usage/ops.md#a-step-or-an-entry) |
| `a top_k over a list of axes gives an index per axis of each pick, which an entry's batches do not join` | a `top_k` over several axes in an entry; make it a step | [ops.md](../usage/ops.md#a-step-or-an-entry) |
| `a reduction cannot remove '<a>'` | `over` names an axis the value lacks | [ops.md](../usage/ops.md#reductions-and-over) |
| `a reduction of a value laid out […] removes no axis of it` | `over` names none of the value's axes | [ops.md](../usage/ops.md#reductions-and-over) |
| `'<op>' needs a value with a '<a>' axis` | an op whose axis the value lacks | [ops.md](../usage/ops.md#reductions-and-over) |
| `it reduces over hidden_size unless 'over' names the axes` | `norm` or `dot` on a value without `hidden_size` | [ops.md](../usage/ops.md#reductions-and-over) |
| `it takes the softmax over vocab_size or num_experts unless 'over' names the axes` | `entropy` on a value with neither axis | [ops.md](../usage/ops.md#reductions-and-over) |
| `count counts along 'pos' unless 'over' names the axes it counts along` | `count` on a value without `pos` | [ops.md](../usage/ops.md#reductions-and-over) |
| `it ranks the value's '<a>' axis, and the value is laid out […]` | `top_k` or `argmax` on a value without its default axis | [ops.md](../usage/ops.md#transforms) |
| `over […]: a list of axes names each axis it ranks once` | `top_k` or `argmax` `over` an axis twice | [ops.md](../usage/ops.md#transforms) |
| `'<x>' is a top_k's value, several tensors` | an op over a whole `top_k` value (an op step's, a load's, an entry's whose last op is `top_k`), or an operand naming one; name a member (`<x>.ids`, `<x>.probs`), or make an entry's `top_k` a step after the forward | [ops.md](../usage/ops.md#transforms) |
| `'<op>' follows top_k, whose value is several tensors` | an op after a `top_k` in its entry; make the `top_k` a step after the forward and the op a step over its member | [ops.md](../usage/ops.md#transforms) |
| `'<x>.ids' is a top_k's ids over a list of axes, one tensor per axis it ranks` | an op over the `ids` of a `top_k` over a list of axes; name the axis's, `<x>.ids.rows` | [ops.md](../usage/ops.md#transforms) |
| `'<x>.tokens' is a top_k's tokens, its picks' token strings, which are text` | an op over a `top_k`'s `tokens`; name its ids, `<x>.ids` | [ops.md](../usage/ops.md#transforms) |
| `k <k>: a top_k picks at least one entry` | a `top_k` whose `k` is 0 or below | [ops.md](../usage/ops.md#transforms) |
| `k <k> is more than the <n> entries it ranks` | a `k` past the entries of an axis whose size is known: at compile where the document fixes it (a `select`, a static width, the dataset's rows, a probe's classes, a featurizer's coordinates, an earlier `top_k`'s `k`), in preflight where the model's sizes or the rows do | [ops.md](../usage/ops.md#transforms) |
| `contains reads text, a detokenize's value (without tokens)` | `contains` on something other than detokenized text | [generation-experiments.md](../patterns/generation-experiments.md#did-it-say-the-answer-contains-over-the-text) |
| `after keeps each row's text after an occurrence, and with tokens a row is its token strings` | `detokenize` with both `tokens` and `after` | [generation-experiments.md](../patterns/generation-experiments.md#score-what-follows-a-text-after) |
| `permute is no op of an entry` | `permute` in an entry; it is an op step over a read | [ablation.md](../patterns/ablation.md#resample-by-permuting-the-rows) |
| `'<x>' is a top_k's value, which permute does not reorder` | `permute` of a `top_k`'s or a `detokenize`'s value; permute the value it is over | [ablation.md](../patterns/ablation.md#resample-by-permuting-the-rows) |
| `'<x>' holds other rows' values (a permute's), so no one read's site says what it is` | a `pca` of a permuted value; take the read before it is permuted | [ablation.md](../patterns/ablation.md#resample-by-permuting-the-rows) |
| `'permute' with seed null is the identity, which moves no row` | `derangement` with `seed: null` | [ablation.md](../patterns/ablation.md#resample-by-permuting-the-rows) |
| `a derangement moves every row to another, and …` | a derangement over one row: a one-row dataset, or a fit's minibatch of one row | [ablation.md](../patterns/ablation.md#resample-by-permuting-the-rows) |
| `the value '<op>' scores keeps <n> of its vocab_size (select)` | a metric over a head narrowed on `vocab_size`; score the whole head, or rank within the selection with `argmax` or `top_k` | [ops.md](../usage/ops.md#metrics) |
| `<column> row <r>: '<v>' is <k> tokens; the read has <n> position(s) on that row` | an answer of several tokens against a read of other positions. At `pos: -1`, teacher-force it: append it to the input and read `before` its scope with `"occurrence": -1` ([teacher forcing](../patterns/generation-experiments.md#a-fixed-answer-token-by-token-teacher-forcing)). At a generated word, the model said one id that the column retokenizes into more ([its own text](../patterns/generation-experiments.md#scoring-the-models-own-text)) | [answer-does-not-line-up.md](answer-does-not-line-up.md#-is-k-tokens-the-read-has-n-positions-on-that-row) |
| `<column> row <r>: '<v>' is <k> tokens, a position each, and the value has no pos axis`; `an answer of several tokens scores a position each, and the value has no pos axis` | an answer of several tokens against a value reduced over `pos` | [answer-does-not-line-up.md](answer-does-not-line-up.md#-a-position-each-and-the-value-has-no-pos-axis) |
| `<column> row <r>: <v> stands for <n> ids (a set, or a class under strip or lower)` | a set or a class where the metric reads one id's logit | [answer-does-not-line-up.md](answer-does-not-line-up.md#-stands-for-n-ids--which-have-no-one-logit-for-logit_diff) |
| `<column> row <r>: its members are … tokens; a set's members are one width` | a set whose members tokenize to different lengths | [answer-does-not-line-up.md](answer-does-not-line-up.md#-its-members-are--tokens-a-sets-members-are-one-width) |
| `<column> row <r>: '<v>' is no tokens once strip drops the whitespace-only tokens at its end` | an answer of whitespace alone under `strip` | [answer-does-not-line-up.md](answer-does-not-line-up.md#-is-no-tokens-once-strip-drops-) |
| `<column> row <r>: position <p> predicts '<t>' in the input, which is not token <i> of '<v>'` | a teacher-forced input that says the answer with other tokens | [answer-does-not-line-up.md](answer-does-not-line-up.md#position-p-predicts--in-the-input-which-is-not-token-i-of-) |

## Featurizers, bundles and fits

| Message | Cause | Page |
|---|---|---|
| `'<x>' is a featurizer, not a value` | a featurizer named where a value goes; reach it with an op (`encode`, `apply`) | [featurizers.md](../usage/featurizers.md) |
| `'<x>' names no featurizer here` | a featurizer name `featurizers` does not declare | [featurizers.md](../usage/featurizers.md) |
| `'<f>' is a '<kind>' featurizer, which has no <op>` | an op the featurizer's kind does not serve; the message lists the ones it does | [featurizers.md](../usage/featurizers.md) |
| `a <kind>'s apply takes no 'with'` | `with` on an `apply` of a kind that is not an adapter or a gate | [featurizers.md](../usage/featurizers.md) |
| `<v> is not a name; 'with' names a value read at the positions the op acts on` | an adapter's `with` that is not a value's name | [lora.md](../patterns/lora.md#across-sites-with) |
| `decode of '<f>' closes no encode` | a `decode` with no open `encode` of it | [featurizers.md](../usage/featurizers.md#brackets) |
| `decode of '<f>', and the open encode is of '<g>'` | brackets closed out of order | [featurizers.md](../usage/featurizers.md#brackets) |
| `the entry writes back here with the encode of '<f>' (ops[<j>]) still open` | an entry that ends inside a bracket | [featurizers.md](../usage/featurizers.md#brackets) |
| `'<op>' of '<f>' at ops[<i>] is inside the encode of '<f>'` | the featurizer's own op inside its bracket | [featurizers.md](../usage/featurizers.md#brackets) |
| `decode closes an encode of its own entry, so it runs inside the entry that encodes` | `decode` as a step | [featurizers.md](../usage/featurizers.md#brackets) |
| `a '<kind>' featurizer drawn without a bundle needs '<f>'` | a drawn featurizer missing a field; or load one with `file_path` | [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) |
| `a <kind> loaded from file_path is its bundle's, and <field> draws one` | a loaded featurizer with a field that draws (a constant's `init`) | [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) |
| `site '<s>' is {…}; the site a featurizer was trained at is a component at one layer` | a featurizer's `site` with several layers or none | [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) |
| `<c>'s last axis is <axis>, which grows with the input; featurizer '<f>' needs a feature axis` | a featurizer at an attention pattern or its scores | [featurizers.md](../usage/featurizers.md) |
| `'<x>' is laid out […], …, and a constant's init is one vector of its width` | an `init` with a rows or `layers` axis, or more than the site's last axis; at a site of several heads, one vector serves every head | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `'<x>' is in another coordinate space than <entry>'s value` | an `init` in another featurizer's coordinates | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `'<x>' depends on the sweep axes […]` | an `init` that varies over a sweep | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `'<x>' is made by step '<s>', …, and '<f>' is drawn from it` | an `init` made after the featurizer's first use | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `'<x>' names no value a step makes` | an `init` no step makes | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `featurizer '<f>' is drawn from '<x>' (init), which keeps …` | before any forward: `init`'s selection against the use's | [ablation.md](../patterns/ablation.md#the-optimal-constant) |
| `<file> is a published sae, which says neither how it encodes nor where it was trained` | a published SAE without its stated fields | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file> has no clab header and holds …; a published sae holds …` | a file without an SAE's tensors | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file>: W_enc is […]; it is [width, latents]`; `<file>: W_dec is […]; with W_enc […] it is […]` | a published SAE's tensor shapes | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file> has <n> latents; this document says <m>`; `<file> has <n> latents; 'topk' keeps k = <k> of them` | `latents` or `k` against the file | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file> holds no threshold; a 'jumprelu' sae needs one`; `<file> holds a JumpReLU threshold; its activation is 'jumprelu', not '<a>'` | `activation` against the file's threshold | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| `<file> does not exist under the corpus root` | a `file_path` with no file; from `clab show` or `clab table`, a relative path read from outside the run's corpus root | [bundles.md](../usage/bundles.md#reading-back-a-run-that-loads-a-bundle) |
| `<file> is not a safetensors file` | a `file_path` in another format; convert it first | [bundles.md](../usage/bundles.md#file_path) |
| `<file> holds one bundle per point of a sweep (…)` | a swept training run's file | [bundles.md](../usage/bundles.md#swept-training-runs) |
| `<file> carries clab header keys (…) but its format is …` | a file with some of clab's header keys and another format | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file> holds …; a <kind> bundle holds …` | a bundle of another kind's tensors | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file> has no clab header; a <kind> bundle is one clab wrote` | a kind that loads only clab's bundles, given another file | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header '<key>' is '<v>'; this document expects '<w>'` | the bundle's header against the document | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header 'layer' is '<l>'; this document uses '<f>' at <c> layer <m>` | a bundle used at another layer | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header 'select' is '{…}'; this document uses '<f>' at <c> layer <m>, whose selection keeps …` | a bundle used at another selection | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header has no 'select'` | a bundle made at the whole component, used at a selection | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header 'width' is '<w>'; this document uses '<f>' at <c>, whose width is <v>` | a bundle used at another width | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: header records no '<field>', which a <kind> bundle records` | a hand-made clab header without its `width` or a field its kind's draw needs (a subspace's or a pca's `k`, an sae's `latents`, a lora's `rank` and `alpha`, a probe's `classes`) | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `<file>: '<t>' is […], and its header lays a <kind> out at width <w>` | a bundle tensor whose shape is not a draw's from its header (a subspace's `basis` `[width, k]`, a lora's `B` `[rank, width]`, a gate's `theta` `[width]`) | [bundles.md](../usage/bundles.md#what-is-checked-and-when) |
| `featurizer '<f>' is used at width <a> here and <b> elsewhere` | one featurizer used at two widths | [featurizers.md](../usage/featurizers.md) |
| `featurizer '<f>' has <field> <n>, and it is used at width <w>` | a rank or `k` larger than the width allows | [featurizers.md](../usage/featurizers.md) |
| `<file> has no clab header and is <w> wide` | a headerless file of another width | [featurizers.md](../usage/featurizers.md) |
| `featurizer '<f>' was trained at <c> layer <l> (its site); this use is …` | a featurizer used at another site than its `site` | [featurizers.md](../usage/featurizers.md#drawn-trained-or-loaded) |
| `select features '<s>' keeps none of the <n> features of '<f>'` | a `features` selector past the featurizer's coordinates | [featurizers.md](../usage/featurizers.md#selecting-features) |
| `over: […] names no axis` | a gate's `over` that is empty or repeats an axis | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `over: […] names <a>, an example's` | a gate `over` `rows` or `pos` | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `<use> encodes '<f>', and a gate over axes serves apply alone` | an `encode` of a gate over axes | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `<use> applies '<f>' inside the encode of '<g>'` | a gate over axes applied inside another featurizer's bracket | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `'<f>' is at <c> layer <l>, one layer, whose selection […] has no layers axis` | a gate `over` `layers` at a one-layer site | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `<use>'s selection at <c> is laid out […], which has no '<a>'` | a gate `over` an axis its site lacks | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `a gate's units are laid out on its axes in the selection's order, and <use>'s is […]` | `over` in another order than the site's layout | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `<c>'s <a> axis grows with the input; gate '<f>' lays its units on axes of a fixed size` | a gate `over` `key` | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `gate '<f>' is over <axes> here and … elsewhere` (an affine's: `affine '<f>' is over …`) | one gate or affine used at two selections of its axes | [featurizers.md](../usage/featurizers.md#a-gate-over-named-axes) |
| `<file>: header 'kept.<a>' is '[…]'; this document uses '<f>' at <c>, whose selection keeps …` | a gate's bundle used at another selection of its axes | [head-masking.md](../patterns/head-masking.md) |
| `<file>: 'theta' is […], and its header lays a gate over […] out on …` (an affine's: `'W' is […], …`) | a bundle's tensor against its header's axes | [head-masking.md](../patterns/head-masking.md) |
| `over: […]: an 'affine' is over ['layers'] or nothing` | an affine `over` another axis; a map is a matrix over the last axis | [featurizers.md](../usage/featurizers.md#an-affine-map) |
| `<file> was trained reading its input at <c> layer <l> (with), and this use gives no 'with'` | a cross-site adapter used without `with` | [lora.md](../patterns/lora.md#across-sites-with) |
| `<file> was trained reading its input at <c> layer <l> (with), and this use's with '<x>' is <c> layer <m>` | `with` read at another site than the training run's | [lora.md](../patterns/lora.md#across-sites-with) |
| `<file>'s header records no input site (with.component)` | `with` on an adapter trained on its own site's value | [lora.md](../patterns/lora.md#across-sites-with) |
| `operand '<x>' is read at positions … of row <r> and this write lands at …` | a same-forward `with` read at other positions than the write | [lora.md](../patterns/lora.md#across-sites-with) |
| `operand '<x>' is <n> wide per position, its axes after the positions flattened; this write's value is <m>` | `with` read at a site of another width per position | [lora.md](../patterns/lora.md#across-sites-with) |
| `pca of <n> vectors has no rank-<k> basis` | `pca`'s `k` against its data | [trained-subspaces.md](../patterns/trained-subspaces.md) |
| `pca k <k> counts directions of a value <w> wide` | `pca`'s `k` past the value's width | [trained-subspaces.md](../patterns/trained-subspaces.md) |
| `classes column '<c>' holds str (…), int (…)` | a probe's classes of several types | [probing.md](../patterns/probing.md#scoring-classes) |
| `<arg> names a class of probe '<p>', which is compared exactly` | `strip` or `lower` on a probe's class column | [probing.md](../patterns/probing.md#scoring-classes) |
| `<arg> is '<c>' of '<d>', whose row <i> is '<v>', not one of the classes of probe '<p>'`; `<arg> is the literal '<v>', not one of the classes of probe '<p>'` | an answer that is no class of the probe | [probing.md](../patterns/probing.md#scoring-classes) |
| `no featurizer '<f>'` | a fit trains a featurizer `featurizers` does not declare | [fit.md](../usage/fit.md) |
| `'<f>' is a <kind>, which a bundle fixes` | a fit trains a kind with nothing to train | [fit.md](../usage/fit.md) |
| `'<f>' is loaded from a bundle, which fixes it` | a fit trains a loaded featurizer | [fit.md](../usage/fit.md) |
| `a fit's body holds no fit` | a fit inside a fit | [fit.md](../usage/fit.md#the-body) |
| `'<n>' is also a dataset or a featurizer, or has a dot` | a body step's name | [fit.md](../usage/fit.md#the-body) |
| `not a body op or a trained featurizer's scalar` | an `objective` term that is neither | [fit.md](../usage/fit.md#the-objective) |
| `'<m>' is not a body op` | an `early_stop` metric that names no body op | [fit.md](../usage/fit.md#early-stopping) |
| `'<label>' names the fit's history or a featurizer it trains, or has a dot` | an `eval` name taken | [fit.md](../usage/fit.md#evals) |
| `none of its terms (<t>) depends on what the fit trains (<f>)` | an objective over values no trained featurizer reaches | [fit.md](../usage/fit.md#the-objective) |
| `'<s>' is over the coordinates the body reads of '<f>', an encode no decode closes` | an SAE's scalar in the objective of a body with no open `encode` of it | [fit.md](../usage/fit.md#the-objective) |
| `an sae's scalars are over what the body's encodes of it saw; this body has none` | an SAE's scalar with no `encode` to measure | [fit.md](../usage/fit.md#the-objective) |
| `'<f>' is trained but used at no site, so its width is unknown` | a trained featurizer the body never reaches | [fit.md](../usage/fit.md#the-objective) |
| `dataset '<a>' is read as '<b>', and their templates differ` | an eval replaces a dataset with one of another `template` | [fit.md](../usage/fit.md#evals) |
| `the body uses '<x>', made before the fit with a row per example of '<d>', which no forward of the body reads` | a harvest whose dataset no body forward reads | [fit.md](../usage/fit.md#the-body) |
| `the body's datasets have {…} rows` | body datasets of different lengths; a body pairs them row by row | [fit.md](../usage/fit.md#the-body) |
| `the body uses '<x>', made before the fit with a row per example of '<d>', and this eval reads '<d>' as '<e>'` | an eval that reads a harvest's dataset as another; score that split after the fit, through `<fit>.<featurizer>` | [fit.md](../usage/fit.md#evals) |
| `'<x>' is an op's value, which no one site made, and a featurizer's <op> acts on a value at its site` | an `encode` or `apply` step over an op step's value; encode a read, or inside the entry that reads | [featurizers.md](../usage/featurizers.md#as-a-step) |

## Loading, running and reading back

| Message | Cause | Page |
|---|---|---|
| `no document at '<path>'` | the document path given to `clab run` does not exist | [cli.md](../reference/cli.md) |
| `batch_size <n>: a batch holds one row or more` | a `--batch-size` below 1 | [running.md](../usage/running.md#batch-size-and-padding) |
| `padding '<p>': one of 'dataset', 'batch', 'sorted'` | a `--padding` clab does not have | [running.md](../usage/running.md#batch-size-and-padding) |
| `readout '<r>': one of 'full', 'at-reads'` | a `--readout` clab does not have | [running.md](../usage/running.md#batch-size-and-padding) |
| `-o '<x>': an option is KEY=VALUE` | an `-o` without `=` | [cli.md](../reference/cli.md) |
| `-o <key>: it is given as --<flag>` | `device`, `batch_size`, `padding` or `readout` given with `-o` | [cli.md](../reference/cli.md) |
| `device '<d>': not a torch device ('cpu', 'cuda:0') or a device map transformers computes` | a `--device` torch does not know | [engines.md](../usage/engines.md#where-the-model-goes---device) |
| `device '<d>': no CUDA device is visible` | a GPU device where none is visible, under nnterp or hooks | [engines.md](../usage/engines.md#where-the-model-goes---device) |
| `device '<d>': <n> CUDA devices visible, cuda:0 to cuda:<m>` | a GPU index past the visible ones | [engines.md](../usage/engines.md#where-the-model-goes---device) |
| `model.key '<k>' at revision '<r>': the <engine> engine cannot load it` | the loader could not load the checkpoint; offline, the `revision` is not in the cache | [engines.md](../usage/engines.md) |
| `model.<key>: the <engine> engine cannot honor it` | a model-block key the engine does not take (`device_map`, `torch_dtype`) | [engines.md](../usage/engines.md) |
| `model.<key>: the <engine> engine's loader rejects it` | a model-block value the loader refuses (a bad `attn_implementation`) | [engines.md](../usage/engines.md) |
| `<v> is not a value generate takes` | a generate option of a type the generation config refuses | [generate.md](../usage/generate.md#options) |
| `'<opt>' is a parameter of generate itself` | one of `generate`'s own parameters given as an option | [generate.md](../usage/generate.md#options) |
| `generate rejected '<opt>'` | the model's `generate` refused an option the step passed | [generate.md](../usage/generate.md#options) |
| `generate rejected its options {…}: <message>` | an option whose value is of a type the generation config takes at compile and `generate` does not (`min_length: "3"`); the message is `generate`'s | [generate.md](../usage/generate.md#options) |
| `unknown engine '<e>'` | `--engine` names no engine | [engines.md](../usage/engines.md) |
| `the <engine> engine takes no '<opt>'` | an `-o` option the engine does not take; the message lists its options | [engines.md](../usage/engines.md#engine-options--o-keyvalue) |
| `remote: '<v>' is not one of True, 'local'` | an ndif `remote` option of another value | [engines.md](../usage/engines.md#running-on-ndif) |
| `tp 0: tensor parallelism shards the model over at least one rank` | `-o tp=0` | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `tp <n> shards the model over <n> processes, one per rank, that torchrun starts` | `-o tp` outside torchrun | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `tp <n>: torchrun started <w> processes (WORLD_SIZE), each of which runs the whole plan` | torchrun's process count differs from `tp`; give both the one number | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `WORLD_SIZE <n>: torchrun started <n> processes, and the <engine> engine takes no 'tp'` | torchrun around an engine without `tp` | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `tp <n>: <key> cannot be sharded <n> ways` | a checkpoint that does not split that many ways | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `device '<d>': under tp <n> transformers puts each rank's shard …` | a `--device` under `tp` other than `cuda` (or `cpu` where no GPU is visible) | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `tp <n>: each rank holds its shard on its own GPU, and torchrun started <k> ranks on this node, which shows <m>` | fewer visible GPUs than ranks on a node | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `model.dtype 'fp8' needs a GPU of compute capability 8.9 or later` | `fp8` on an older GPU, or on none | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `model.dtype '<q>' holds the weights in bitsandbytes, which is not installed` | a quantized `dtype` without bitsandbytes | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `model.dtype '<q>': bitsandbytes' packed weights do not shard across ranks` | a quantized `dtype` under `tp` | [engines.md](../usage/engines.md#quantized-and-sharded-models) |
| `model.dtype '<q>': the ndif engine cannot honor it` | a quantized `dtype` under `ndif` | [engines.md](../usage/engines.md#running-on-ndif) |
| `device: the ndif engine cannot honor '<d>': the server places the model`; `device: the ndif engine places a local model on one torch device` | `--device` under `ndif` | [engines.md](../usage/engines.md#running-on-ndif) |
| `device: the vllm engine cannot honor '<d>'` | `--device` under `vllm` | [engines.md](../usage/engines.md#running-on-vllm) |
| `model.dtype '<q>': the vllm engine cannot honor it` | a bitsandbytes `dtype` under `vllm` | [engines.md](../usage/engines.md#running-on-vllm) |
| `tp <n>: vLLM shards the model over one GPU or more` | a vllm `tp` below 1 | [engines.md](../usage/engines.md#running-on-vllm) |
| `memory <m>: the fraction of each GPU's memory vLLM reserves is above 0 and at most 1` | a vllm `memory` outside (0, 1] | [engines.md](../usage/engines.md#running-on-vllm) |
| `max_len <n>: the longest sequence the engine serves is one token or more` | a vllm `max_len` below 1 | [engines.md](../usage/engines.md#running-on-vllm) |
| `the vllm engine does not write <c>` | on vllm, a write at the recomputed attention pattern or scores | [engines.md](../usage/engines.md#running-on-vllm) |
| `the <engine> engine serves <c> at the prompt's last position alone` | on vllm, the head at a prefill position but the last | [engines.md](../usage/engines.md#running-on-vllm) |
| `a fit trains through the model's gradient, which vLLM's kernels do not take`; `a grad entry takes the model's gradient, …` | a fit or a `grad` entry under `vllm` | [engines.md](../usage/engines.md#running-on-vllm) |
| `generate rejected '<opt>': the vllm engine's generate maps … onto vLLM's sampling, and has nothing to map '<opt>' to` | an option vLLM's sampling has no name for | [engines.md](../usage/engines.md#running-on-vllm) |
| `generate rejected 'max_new_tokens': the vllm engine needs the step to give it` | a vllm generate step without `max_new_tokens` | [engines.md](../usage/engines.md#running-on-vllm) |
| `generate rejected '<field>': the checkpoint's generation config sets <field> …` | a generation config with a logits processor vLLM lacks | [engines.md](../usage/engines.md#running-on-vllm) |
| `<input>'s longest row has <n> tokens …, past the <m> the vllm engine serves` | rows past the engine's `max_len` | [engines.md](../usage/engines.md#running-on-vllm) |
| `plan is not compiled; call plan.compile()` | `execute` on a plan not compiled | [python-api.md](../reference/python-api.md) |
| `the plan's model is not the one this engine loaded` | a loaded engine reused with another model block; the message names each field that differs | [python-api.md](../reference/python-api.md) |
| `the loaded <e> engine runs on <d> with {…}; device and options are given when an engine loads` | `clab.run` with a loaded engine and another device or options | [python-api.md](../reference/python-api.md) |
| `the plan has not run, so it has no results; execute it first` | the run record asked for before the run | [python-api.md](../reference/python-api.md) |
| `describe '<what>': nothing is described under that name` | `clab.describe` given a name that is none of its tables; the message lists them. On the command line argparse refuses it (`invalid choice`) | [python-api.md](../reference/python-api.md#clabdescribe) |
| `out of CUDA memory at batch size <n>` | a step ran out of CUDA memory; the run stops there and saves the steps that finished | [running.md](../usage/running.md#when-a-step-runs-out-of-memory) |
| `row <r> of dataset '<d>' is not among the <n> rows this value holds` | in the run: a value made over some of its dataset's rows (a fit body's minibatch) read at a row it does not hold; no document reaches it, so reaching it is a defect in clab | [fit.md](../usage/fit.md#the-body) |
| `the values this document loads hold <n> MiB of tensors (…), and a request carries at most 512 MiB inside its plan` | `load`s too large to send in a request, to an NDIF server or under `-o remote=local` | [cached-harvest.md](../patterns/cached-harvest.md#under-ndif) |
| `'<n>' was not saved; add it to saves` | `clab table` or `clab show` of an unsaved name, or of a run whose `--out` a later run reused, replacing its manifest | [output.md](../usage/output.md), [bundles.md](../usage/bundles.md#the-two-document-workflow) |
| `'<n>' has no result: the run stopped before its step` | a name whose step a stopped run never reached | [output.md](../usage/output.md) |
| `no run at '<out>' (no manifest.json)` | `clab show` or `clab table` of a directory without a run | [output.md](../usage/output.md) |
| `<out>/tokens.json is missing, and the manifest holds no tokens` | a run directory without its `tokens.json` | [output.md](../usage/output.md) |
| `'<a>' has a row per example of '<d>' and '<b>' of '<e>'; a table joins the values of one dataset` | names of different datasets in one table | [output.md](../usage/output.md#clab-table) |
| `'<a>' has a pos column counting positions of <i>, and '<b>' one counting positions of <j>` | names read on different inputs in one table; table them one at a time | [output.md](../usage/output.md#clab-table) |
| `'<n>' is a top_k over rows, whose rows are its picks, and a top_k over rows tables alone` | a `top_k` over `rows` beside another name | [output.md](../usage/output.md#clab-table) |
| `'<n>' is a trained featurizer's bundle, which has no rows to table` | a bundle in `clab table`; read it from `plan.results` | [output.md](../usage/output.md#clab-table) |
| `'<n>' has no rows to table: it is …` | a value with no rows axis | [output.md](../usage/output.md#clab-table) |
| `the plan holds none of its rows` | `preflight` or a step called from Python on a plan whose rows are not loaded; `clab.run` and an engine's `execute` load them first | [compile-and-run.md](../developing/compile-and-run.md#the-run) |


## Errors that are not clab's

A malformed command line (an unknown flag, a missing argument, a value outside a flag's choices) gets argparse's usage message and exit status 2, not a `clab:` line: `clab describe: error: argument what: invalid choice: 'bogus' (choose from …)`. An engine no one registered is a `clab:` line, `clab: unknown engine 'bogus'; valid: nnterp, hooks, ndif, vllm` ([cli.md](../reference/cli.md)).

Under `--engine ndif`, preflight and the run happen on the server. With `-o remote=local`, nnsight's local simulation of that trip, `clab run` prints the same `clab:` line as under nnterp; [engines.md](../usage/engines.md) covers a real server.

## Pages in this section

- [symptom-index](symptom-index.md) — Something ran and the numbers are wrong: the silent cases, from indices counted within a select list to a published SAE stated at the wrong layer, each with what it is and where it is explained.
- [alignment-missing](alignment-missing.md) — Why a row is alignment_missing or a value is NaN, how missingness propagates through ops and tables, how to count it, and how to move the question into the prefill, where a miss is refused.
- [operand-does-not-fit](operand-does-not-fit.md) — Why a swap, add, kl or cosine refuses its operand: layout, width, positions, selected heads, layers, coordinates, dataset, a same-forward entry, or two inputs of different lengths, with the fix for each message.
- [position-not-in-row](position-not-in-row.md) — Why a run refuses with 'row i: … is not in its input' or 'index … does not fit the row's …': a prefill position that names nothing on one row, how to see that row's tokens, and the fix for each message.
- [answer-does-not-line-up](answer-does-not-line-up.md) — Why a metric refuses its answer column ('is 3 tokens; the read has 1 position', 'no one logit for logit_diff', 'predicts … in the input'): how answers are tokenized and lined up with the read, and the fix for each message.
