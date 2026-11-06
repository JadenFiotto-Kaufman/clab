---
title: Alignment Missing
one_liner: "Why a row is alignment_missing or a value is NaN, how missingness propagates through ops and tables, how to count it, and how to move the question into the prefill, where a miss is refused."
tags: [errors, alignment, alignment_missing, nan, generation]
related: [docs/concepts/positions.md, docs/usage/positions.md, docs/usage/generate.md, docs/concepts/values-and-layout.md, docs/usage/output.md, docs/usage/ops.md]
sources: [nnspec/compile/positions.py, nnspec/steps/generate.py, nnspec/output/table.py, nnspec/ops/transforms.py]
---

# Alignment Missing

A row is `alignment_missing` when a read or write names a position the row does not have. Beyond a generation's prefill that is a call the row did not make or text it did not generate; in the prefill too, an index with an open end that picks none of the row's tokens.

- **A read there is not refused.** The run finishes, the row is recorded as missing, and every value computed from it on that row is NaN, but a `count`, which is 0 ([below](#how-it-propagates)).
- **A write beyond the prefill is not refused either.** The row is not written, its `positions` say `alignment_missing`, and nothing else in the run says so.
- **A write in the prefill whose positions hold no token of a row is refused** ([position-not-in-row.md](position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row)).
- **A later write whose operand is missing on a row it lands on is refused**, naming the row and the call ([below](#how-it-propagates)).

## Symptom

`nnspec table` gives the row one line with `alignment_missing` true and no value, as in `documents/generated_answer.json`, where two of four rows never say their word:

```text
example_id  row  pos  pos_token  call  input                              cf_answer  value                   alignment_missing
substr      0    12   ▁substr    2     If today is Thursday, tomorrow is   Sunday    2.9666021873708814e-05
energ       1    12   calc       4     If today is Friday, tomorrow is     Monday    3.15390425384976e-05
Friday      2                          If today is Saturday, tomorrow is   Friday                            true
Saturday    3                          If today is Sunday, tomorrow is     Saturday                          true
```

The entry's record holds `"alignment_missing"` in place of those rows' position lists (`plan.steps["answer"].interventions["at_said"].positions`, which `nnspec show` prints as `[12], [12], —, —`), and the saved tensor is NaN on them.

## Causes

What a record names beyond the prefill depends on what each row generates, so a row that does not fit is a result, not an error ([concepts/positions.md](../concepts/positions.md#decided-by-the-data-or-a-result)). Each cause below was produced on the corpus model:

| cause | example | what to do |
|---|---|---|
| the row stopped before a call the frame names | `{"frame": 3}` on a row whose third token was EOS | `min_new_tokens` equal to `max_new_tokens`, or a frame that follows each row (`"1:"`, `{"frame": -1, "index": -1}`); `"1:"` still misses a row whose first token was EOS |
| the row never generated the scope's text | `generated_answer.json`, rows 2 and 3 | it is the finding: count it, or score the text by teacher forcing (below) |
| the word is the row's last generated token | ` toss` said as token 5 of 5 | read with `before`, or give the row room to go on |
| fewer occurrences than `occurrence` picks | `"occurrence": 1` of a word said once | pick from the first, or `":"` |
| an index with an open end that picks none of a row's tokens, in the prefill too | `{"index": "10:"}` read on prompts of 9 tokens and one of 11 | an index that every row holds (`-1`, `"-3:"`), or the rows that hold it |
| a `before` scope whose first token is position 0, beyond the prefill | a scope on the prompt's first word under a tokenizer that adds no BOS | nothing comes before position 0; scope a later token |
| a `before` read at the head on a prompt token of a `generate` | `{"frame": ":", "scope": {"text": " is", "before": true}}` at `lm_head_output` | the prefill computed the head at its last position only; read prompt positions in a `forward` |
| an online scoped write found no occurrence | a write at `{"frame": "1:", "scope": {"text": " toss"}, "index": -1}` on rows that never say it | nothing: the row is not written, and its generation is the unwritten one; count it |

**Check where a write landed, not only a read.** A steering write over decode calls that never lands on a row leaves that row's generation unsteered. A mean over the rows then dilutes the effect, and nothing in the number says so. Count the write's missing rows as the reads' [below](#count-it-prevent-it-or-refuse-it).

## In the prefill it is a refusal

The same misses of a scope or a static index in the prefill (`frame: 0`, every `forward`) are decided by the prompt alone, so they are refused before any forward, naming the row. The messages and their fixes are [position-not-in-row.md](position-not-in-row.md). An index with an open end is not among them: its read is `alignment_missing` on a row it picks nothing of, in the prefill too.

A frame and index of static widths that can never meet are refused at compile, before any row is seen: `index '1:3' is 2 tokens wide and picks 1 of the 2 tokens of frame '1:3''s decode calls, …`.

## How it propagates

- **The entry's value** is NaN at every entry of a missing row; ids (`argmax`, `top_k`'s `ids`) are -1 there and `top_k` gives no token text.
- **Metrics, edits, elementwise transforms and comparisons** of a hole are NaN. A comparison of ids is float32, NaN where either side is -1, so an agreement of two `argmax` steps is NaN where either has no id, never a 0 or 1 a `mean` takes. A write never lands on a row it did not reach.
- **A write whose operand is missing on a row it lands on is refused**, where it would write NaN. The operand is a generation's read at a call the row never made (`{"frame": 3, "index": 0}`), or an op step's value made from one as `of`, `with` or a `factor`.
- **The run refuses it on that row**, naming the row and the call: `operand 'g.v' holds no value on row 1 (alignment_missing)`. A forward's write is refused before its batch's forward, and a write at a decode call once the batch's generation says which rows made the call.
- **Compile refuses it where no row makes the call**, a decode call at or past the generation's `max_new_tokens`: `'g.v' is read at decode call 3, and 'g' makes calls 0 to 2 (max_new_tokens 3)`.
- **In a fit's body a generation records no positions**, so a write there with such an operand still writes NaN. The fit's evals, which run as root steps, refuse it.
- **Reductions skip holes**, an id's -1 as a NaN. A `mean` over rows averages the rows that were reached: a `mean` of `documents/generated_answer.json`'s `p_answer` is the mean of the 2 rows of 4 that said their word. An accuracy computed this way is the accuracy on the rows that got there, with nothing in the number to say so.
- **A `sum` or `min` over `pos` of a row with no value is NaN, and its `count` 0 in the saved tensor.** So `count` over `rows` of a scored read is the number of rows it scored. `nnspec table` still prints that row of the `count` flagged `alignment_missing` with no value, as it prints the row in every value made from the read.
- **Tables** give a missing row one line with `alignment_missing` true and no value, whatever the tensor holds there: a `count` of the read is 0 in its file and has no value in the table. When several names are tabled together, the row carries the flag. A read that no row reached tables to one flagged line per row too, and `nnspec show` prints its positions as `— on all 4 rows`. Its saved shape is `[4,0,32000]`, in the prefill as in a generation. Positions past a row's own on a ragged read are a different hole: they are left out of the table, with no flag.

What a hole is, and where the others come from (keys and ragged positions), is [values-and-layout.md](../concepts/values-and-layout.md#missing-entries).

## Count it, prevent it, or refuse it

**Count** the missing rows of every scored read before reading an average:

```python
from nnspec import Plan

plan = Plan.load("out/generated_answer")
rows = plan.steps["answer"].interventions["at_said"].positions["prompts"]    # per row, under the step's dataset
missing = [row for row, where in enumerate(rows) if where == "alignment_missing"]
assert not missing, f"rows {missing} never said their word"
```

```text
AssertionError: rows [2, 3] never said their word
```

Inside the document, `count` counts positions. Over `rows` and `pos` it is every position the read holds on every row, which is the number of rows it reached only where it holds one position per row. On `documents/generated_answer.json`, whose `at_said` takes `"index": -1`, `{"kind": "op", "op": "count", "of": "answer.at_said", "over": ["rows", "pos"]}` is 2. A scope read whole holds a position per token of its word on each row.

To count the rows a read reached whatever its width, count each row's positions, mark the rows that hold any, and sum the marks:

```json
"per_row": {"kind": "op", "op": "count", "of": "answer.at_said"},
"reached": {"kind": "op", "op": "gt", "of": "per_row", "with": 0},
"rows_reached": {"kind": "op", "op": "sum", "of": "reached"}
```

On GPT-2 (`openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU, float32), `answer` was a `generate` step of 8 tokens (`max_new_tokens` and `min_new_tokens` 8, `do_sample: false`) over four inline rows, each `said` two tokens:

```json
"p": {"rows": [{"input": "The capital of Japan is", "said": " United States"},
                {"input": "If today is Thursday, tomorrow is", "said": " next day"},
                {"input": "My favourite fruit is the", "said": " very sweet"},
                {"input": "The Eiffel Tower is located in the city of", "said": " Tokyo"}]}
```

Its `at_said` read `layer_output` layer 5 at `{"frame": "1:", "scope": {"column": "p.said"}}`. The first three rows said theirs and the last did not: `count` over `rows` and `pos` gave 6.0, the positions, and `rows_reached` 3.0.

A swept step is a list of points, each holding its own records, so count each point. For a dose sweep (`"scale": {"sweep": [0.0, 8.0, 32.0], "label": "dose"}`) of a steered generation with a read scoped to the word:

```python
for point in plan.steps["steered"]:
    rows = point.interventions["said"].positions["prompts"]
    dose = plan.sweeps["dose"].values[point.at["dose"]]
    print(dose, [row for row, where in enumerate(rows) if where == "alignment_missing"])
```

The rows a scope reached are not always the rows a `sum` of `contains` counts. `contains` finds the text anywhere in the decoded string, inside a longer word too, while a scope matches on word boundaries only. A row that says the word inside a longer one scores 1 under `contains` and is missing to the scope.

**Prevent** the causes that come from the generation's length:

- **Hold EOS off with `min_new_tokens`.** A row held past its EOS goes on as a different generation from there, not the first one extended ([generate.md](../usage/generate.md#rows-that-stop-early)).
- **Leave room after the answer**: `max_new_tokens` beyond its last token.
- **Read the positions that chose a word with `before`**, which include the row's last token.

**Refuse** instead. Beyond the prefill no option turns a missing row into an error, since only the generation decides it. In the prefill every miss of a scope or a static index is one. So when the question is whether the model gives an answer, ask it in the prefill by teacher forcing, as `documents/forced_answer.json` does ([positions.md](../usage/positions.md#the-token-before-before)). Every row is then scored and none is missing; a row whose input does not hold the text is refused before any forward. To stop on a missing generated row, check the entry's `positions` after the run, as above.
