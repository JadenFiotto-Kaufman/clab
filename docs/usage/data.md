---
title: Data and Inputs
one_liner: "How do I load rows and pick what a step runs over? File, inline and hub datasets, narrowing, renamed and derived columns, chats, teacher forcing and answer columns."
tags: [usage, data, inputs, chat]
related: [docs/usage/writing-a-document.md, docs/usage/positions.md, docs/usage/ops.md, docs/usage/generate.md, docs/usage/fit.md, docs/usage/running.md, docs/usage/output.md, docs/patterns/generation-experiments.md, docs/errors/answer-does-not-line-up.md]
sources: [clab/format/data.py, clab/plan.py, clab/compile/__init__.py, clab/compile/answers.py]
---

# Data and Inputs

The `data` block names a document's datasets. Each dataset is a list of rows, each row a JSON object, read from a file, written inline, or loaded from the Hugging Face hub. A step's `input` names the column it runs over, one row per example. Other fields name columns as text to search for or as answers to score. Every field is in [FORMAT.md](../../FORMAT.md#dataname); this page is how to use them.

## Canonical example

```json
{
  "model": {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"},
  "data": {
    "qa": {"rows": [
      {"example_id": "fri", "input": "If today is Friday, is tomorrow Saturday? Answer:", "answer": [" Yes", " True"]},
      {"example_id": "mon", "input": "If today is Monday, is tomorrow Sunday? Answer:", "answer": [" No", " False"]}
    ]},
    "days": {"path": "weekdays_full/data", "split": "train", "shuffle": true, "take": 4,
             "columns": {"prompt": "input"}}
  },
  "steps": {
    "ask": {"kind": "forward", "input": "qa.input",
            "interventions": {"logits": {"site": {"component": "lm_head_output"}, "pos": -1}}},
    "p_answer": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": "qa.answer"},
    "forced": {"kind": "forward", "input": ["days.prompt", "days.base_answer"],
               "interventions": {"logits": {"site": {"component": "lm_head_output"},
                                            "pos": {"scope": {"column": "days.base_answer", "occurrence": -1,
                                                              "before": true}}}}},
    "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.base_answer"}
  },
  "saves": ["p_answer", "nll"]
}
```

- `qa` is two inline rows whose `answer` is a set: `p_answer` is the summed probability of either member.
- `days` keeps 4 rows of the `train` split of `data/weekdays_full/data.json`, in a fixed shuffled order, with its `input` column renamed `prompt`.
- `forced` runs each prompt with its answer appended (teacher forcing) and reads the positions that predict the answer's tokens.
- **`"occurrence": -1` picks the appended answer.** Without it the scope takes the answer's first occurrence, which is in the prompt wherever the prompt says the answer too ([positions.md](positions.md#words-and-occurrences)).

## Where rows come from

A dataset has exactly one source; giving two is refused.

- **`path`** reads `data/<path>.json` under the corpus root, the nearest directory above the document that holds `data/` ([running.md](running.md#the-corpus-root)). The file is a JSON list of row objects.
- **`rows`** are written inline. They travel inside the plan and the manifest, so a run's table reads them back from anywhere.
- **`hf`** is a hub dataset, `<owner>/<name>`, with a required `revision` (a commit, as `model` pins one), its subset `config` where it has several, and `split`:

  ```json
  "glue": {"hf": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
           "config": "wnli", "split": "validation", "take": 3}
  ```

  Its rows never go into the plan or the manifest. Compile loads them to check the document and records their digest, the run loads them again where the model runs, and `clab table` loads them again to label its rows. Offline (`HF_HUB_OFFLINE=1`) they come from the `datasets` library's cache, which one online load of that commit fills (`tests/test_hub.py` gives the command). A hub dataset with several splits and no `split` is refused, and the message lists them.

## Narrowing and renaming rows

The five fields apply in this order: `split`, `shuffle`, `skip`, `take`, `columns`.

| field | does | refused |
|---|---|---|
| `split` | a file: keeps the rows whose `split` column equals it; the hub: which split | on inline rows; a split no row has (`data.pairs: the dataset has no rows in split 'train'`) |
| `shuffle: true` | one fixed random order, the same on every machine (below) | — |
| `skip: N` | drops the first N rows; with `take`, a shard of the order (`"skip": 20000, "take": 10000` is rows 20,000 to 29,999) | N that leaves no row; a negative N |
| `take: N` | keeps the first N of the rows left; after `shuffle`, a random sample of N | N beyond the rows there are (after `skip`) |
| `columns` | `{"<new name>": "<column as the source has it>"}` renames columns; `{"<new name>": {"from": "<column>", "after"?, "before"?, "strip"?}}` derives one ([below](#a-column-cut-from-another)) | a source the rows lack; a new name that is a column the rows keep; a derived column's text absent on a row |

- **A renamed column answers to its new name only.** In the example, `days.input` is refused and `days.prompt` is the column. Renaming lets one file serve under another dataset's column names, which a fit's `eval` substitution needs ([fit.md](fit.md)).
- **A swap, `{"a": "b", "b": "a"}`, is allowed**, since neither old name is kept.
- **A rename into a name the steps read changes what they read, unrefused.** On `weekdays/train`, `{"input": "entity", "prompt": "input"}` runs every `pairs.input` step on the entity alone (`<s> ▁Friday` on row 1). Check `run.tokens` after renaming.
- **`take` without `shuffle` keeps the first N rows.** On a file sorted by label or difficulty, that is not a sample.

The manifest records each dataset's `sha256` and `n_rows` after these fields apply, and a row's index in `clab table` counts these rows.

`shuffle`'s order ranks each row's index `i` by the 8-byte blake2b hash of `repr((0, i))` (`clab/format/data.py` `shuffled`), so it depends on the row count alone. Outside clab, with `n` the row count after `split`:

```python
order = sorted(range(n), key=lambda i: hashlib.blake2b(repr((0, i)).encode(), digest_size=8).digest())
```

`skip: S` then drops the first S of that order and `take: N` keeps the next N. A harvest too large for one run is made in shards this way ([cached-harvest.md](../patterns/cached-harvest.md#a-harvest-in-shards)).

### A column cut from another

A `columns` entry that is an object derives a column from another column's text, where a renamed one moves it. On `weekdays/train` the day each prompt names is

```json
"pairs": {"path": "weekdays/train",
          "columns": {"day": {"from": "input", "after": {"text": "today is "}, "before": {"text": ","}}}}
```

which gives `Thursday`, `Friday`, `Saturday` and `Sunday`.

- **`after`** keeps the text after the first occurrence of its text.
- **`before`** keeps the text before the first occurrence of its text, after `after`'s when both are given.
- **`strip`** drops the surrounding whitespace of what is left.
- **`from`** names the column as the source has it, which stays under its own name.
- **Each text is `{"text": ...}`**, one string compared exactly, with no `strip`, `lower` or set. A bare string is refused: `data.pairs.columns.day.after: 'today is ' is a string; …`.
- **Compile cuts every row** and refuses, naming the row, one whose `from` is not text or lacks one of its texts: `data.pairs.columns.day.after: row 0: 'yesterday was ' is not in its 'input', …`.
- **The derived column is part of the rows**, so the dataset's `sha256` covers it, and a hub dataset's rows are cut again where the run loads them. It is text cut from text before any forward, never a computation on values.

## Inputs

A `forward` or `generate` step's `input` takes three forms:

| form | example | each row's prompt |
|---|---|---|
| `<dataset>.<column>` | `"pairs.input"` | the column's text, or its chat |
| `<dataset>.<column>[i]` | `"pairs.counterfactual_inputs[0]"` | entry `i` of a list column; every row needs one |
| a list of columns and `{"text": ...}` texts | `["days.prompt", "days.base_answer"]` | the parts joined with nothing between |

Text is tokenized as the tokenizer does by default, with BOS where it adds one; how positions count those tokens is [positions.md](positions.md). The run records every input's tokens under its name in `run.tokens` ([output.md](output.md#the-run-record)). A list input's name joins its parts with ` + `, a text as JSON: `days.prompt + " and then" + days.base_answer`.

### List inputs: teacher forcing

`documents/forced_answer.json` writes `"input": ["days.input", "days.base_answer"]` to append each row's answer to its prompt. Nothing is inserted between parts, so the answer's own leading space separates the words. A text, `{"text": ...}`, one string alone, adds fixed text: `["days.input", {"text": " and then"}, "days.base_answer"]` gives `If today is Thursday, tomorrow is and then Friday`. All columns of a list come from one dataset, a list needs at least one column, and a chat column in a list is refused, since a list joins text. Reading the answer's positions is [positions.md](positions.md); scoring each token is [ops.md](ops.md).

**A column without the answer's leading space** (`data/weekdays/said.json`'s `said`) joins the prompt's last word when appended bare. `["p.input", "p.said"]` gives `issubstr`, and the scope on `p.said` is refused: `column 'p.said' ('substr') is not in its input; it occurs only inside 'issubstr', …`. Put the space in as a text: `"input": ["p.input", {"text": " "}, "p.said"]`. On the corpus Llama, where `substr` and ` substr` are the same tokens, that gives the same numbers as rows with the space written in. On a byte-level BPE such as GPT-2's, the bare target is other tokens than the input says. A metric of it is then refused (`'substr' is 4 tokens; the read has 3 positions on that row; …`), or scores the wrong tokens where the counts agree. There the space goes into the rows, since a derived column only cuts ([below](#answer-columns)).

### Chat inputs

A column whose value on each row is a list of `{"role", "content"}` messages is a chat. clab renders it through the checkpoint's own chat template and tokenizes the result without adding special tokens, since the template writes its own. `documents/chat_turn.json` reads `chats.chat` from `data/weekdays/chat.json`; on the corpus Llama the first row renders as:

```text
<s>[INST]▁Name▁a▁weekday.▁[/INST]▁Thursday.▁</s><s>[INST]▁If▁today▁is▁Thursday,▁tomorrow▁is▁[/INST]
```

The dataset's `template` is passed to the checkpoint's `apply_chat_template` as arguments, wherever a chat of that dataset renders:

```json
"chats": {"path": "weekdays/chat", "template": {"add_generation_prompt": true}}
```

- **Without a template** the prompt ends with the template's close of the last message, and `pos: -1` is that close's last token.
- **With `add_generation_prompt`** it ends with the assistant's header, so a `generate` continues as the assistant. What an argument does is the template's: on a Llama-2-style template, as above, `add_generation_prompt` changes nothing. Check the rendering in `run.tokens`.
- **A `template` on a dataset that no step reads as a chat is refused**, and so is a `tokenize` key in it. Positions inside a turn are [positions.md](positions.md#chat-turns).
- **A chat comes only from a column that already holds `{"role", "content"}` messages**; nothing in a document turns a text column into one. To run a text dataset as user turns, write the messages outside clab as inline `rows` (or a file), and record the hub source and its revision in `description`.

**A template can read the clock.** Llama-3.2's template adds a system header when the chat has none and writes `Today Date: <today>` in it (`strftime_now`). The same document would then render other tokens on another day, which a dataset's digest cannot say: it covers the rows, not how they render. A chat of such a template is refused before any forward unless the dataset's `template` pins the date, and the refusal gives the pin (`tests/test_plan.py` `test_a_chat_template_that_reads_the_clock_renders_only_with_its_date_pinned`):

```text
clab: blanked: chats.chat: the checkpoint's chat template calls strftime_now, which writes today's date into the prompt, so these rows render to other tokens on another day and nothing in the run record says so; pin it in data.chats.template, {"date_string": "26 Jul 2024"}
```

```json
"chats": {"path": "weekdays/chat", "template": {"add_generation_prompt": true, "date_string": "26 Jul 2024"}}
```

With the pin, the prompt carries that date; check it in `run.tokens`.

## Answer columns

Where a field takes a column, as an answer to score or as text to find (a metric's answer, `contains`'s text, a scope's `column`), write `"<dataset>.<column>"`, or the object with flags:

```json
"token": {"column": "qa.answer", "strip": true, "lower": true}
```

- **Flags say how text compares; they never rewrite the data.** `lower` compares lowercased; `strip` makes surrounding whitespace optional. Scored against logits, the answer's token then stands for every vocabulary token that compares the same: in `documents/answer_spellings.json`, ` Yes` under both flags sums the probability of `▁Yes`, `▁yes`, `Yes`, `yes`, `YES` and `▁YES`. What that does to each metric is [ops.md](ops.md#metrics).
- **A list of strings is a set**, any member of which is the answer: `"answer": [" Yes", " True"]` above. `clab table` prints it as JSON.
- **A fixed answer is a literal, not a column.** Where an op takes an answer or a text, `{"text": "</think>"}` stands in place of the column. It is the same text on every row, tokenized as written and compared under the same flags (`{"text": " yes", "strip": true, "lower": true}`; a list is a set). It gives the same numbers as a column holding that text on every row (`tests/test_answers.py` `test_a_literal_scores_as_the_constant_column_of_its_text_does`), so a constant column built only to be scored is not needed.
- **A bare string is always a column reference.** `"token": "</think>"` is refused with `'</think>' is not <dataset>.<column> over a dataset; …; text written as it is, the same on every row, is {'text': '</think>'}`.
- **A path reaches inside a column's entries**: `pairs.counterfactual_inputs_variables[0].entity` is the `entity` of each row's first variables object.
- **An answer is tokenized exactly as written, without special tokens.** Write the space the prompt implies. On a byte-level BPE tokenizer such as GPT-2's, `" Friday"` and `"Friday"` are different tokens; on the corpus Llama both are `▁Friday`.
- **Where both spellings are one token, the wrong one is scored without a refusal.** On `documents/gpt2_reach.json`'s GPT-2, `is` and ` is` are each one token, and a `token_prob` of either runs. Where the bare word is more tokens than the spaced one, it is refused ([answer-does-not-line-up.md](../errors/answer-does-not-line-up.md)); the symptom is in [symptom-index.md](../errors/symptom-index.md#answers-and-metrics).
- **A hub column without the space is fixed outside the document.** A derived column only cuts a column's text, and `strip` loosens the comparison, not the answer's tokens. Write the rows inline, or to a file, from a script that adds the space, and record the hub source and commit in `description`.
- **Text searched in a generation is decoded text**, in which the first generated token has lost its space: [generate.md](generate.md#the-first-generated-tokens-leading-space).

## `example_id`

A row's `example_id` column, when the dataset has one, leads every row of `clab table`. Give rows one when you shuffle or take, so a table row still says which example it is.

## The corpus data

The datasets under the repository's `data/`, by the `path` that reads them. Every file has a `split` column:

| `path` | rows | columns | `split` |
|---|---|---|---|
| `weekdays/train` | 4 | `input`, `counterfactual_inputs`, `counterfactual_inputs_variables`, `entity`, `answer`, `base_answer`, `cf_answer`, `label` | `all` |
| `weekdays/data` | 4 | those of `weekdays/train` | `train` 2, `test` 2 |
| `weekdays/said` | 4 | `example_id`, `input`, `said`, `base_answer`, `cf_answer` | `all` |
| `weekdays/chat` | 4 | `example_id`, `chat`, `entity`, `base_answer`, `cf_answer` | `all` |
| `weekdays_full/data` | 42 | `example_id`, `input`, `counterfactual_inputs`, `entity`, `cf_entity`, `base_answer`, `cf_answer` | `train` 30, `test` 12 |
| `weekdays_aligned/data` | 6 | those of `weekdays_full/data` | `all` |
| `weekdays_even/train` | 6 | `input`, `counterfactual_inputs`, `counterfactual_inputs_variables`, `entity`, `base_answer`, `cf_answer` | `all` |
| `counting/train` | 4 | `input`, `counterfactual_inputs`, `base_answer`, `cf_answer`, `label` | `all` |
| `pile/data/sample` | 3 | `example_id`, `input`, `resample_input`, `ml_token` | `sample` |
| `natural_domains_arithmetic/data/weekdays` | 49 | `input`, `counterfactual_inputs`, `counterfactual_inputs_variables`, `entity`, `number`, `result`, `label`, `base_answer`, `cf_answer`, `label_forms`, `base_answer_forms`, `cf_answer_forms`, `string_mode`, `scoring_digest` | `train` 30, `test` 19 |

`weekdays/train` is not a training split: every row's `split` is `all`, and it pairs each day with another counterfactual day than `weekdays/data` does. A fit's training and held-out rows are `weekdays/data`'s two splits ([fit.md](fit.md#the-canonical-fit)).

## Pitfalls

- **A file dataset is read again when you table a run.** `clab table` loads it from `data/` under the working directory and refuses rows that changed since the run ([output.md](output.md#tabling-needs-the-rows)). Inline rows do not have this problem.
