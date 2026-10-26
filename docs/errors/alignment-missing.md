---
title: Alignment Missing
one_liner: "Why a row is alignment_missing or a value is NaN, how missingness propagates through ops and tables, how to count it, and how to move the question into the prefill, where a miss is refused."
tags: [errors, alignment, alignment_missing, nan, generation]
related: [docs/concepts/positions.md, docs/usage/positions.md, docs/usage/generate.md, docs/concepts/values-and-layout.md, docs/usage/output.md, docs/usage/ops.md]
sources: [clab/compile/positions.py, clab/steps/generate.py, clab/output/table.py, clab/ops/transforms.py]
---

# Alignment Missing

A row is `alignment_missing` when a read or write beyond a generation's prefill names a position the row never had: a call it did not make, or text it did not generate. Nothing is refused: the run finishes, the row is recorded as missing, and every value computed from it on that row is NaN.

## Symptom

`clab table` gives the row one line with `alignment_missing` true and no value, as in `documents/generated_answer.json`, where two of four rows never say their word:

```text
example_id  row  pos  pos_token  call  input                              cf_answer  value                   alignment_missing
substr      0    12   ▁substr    2     If today is Thursday, tomorrow is   Sunday    2.9666021873708814e-05
energ       1    12   calc       4     If today is Friday, tomorrow is     Monday    3.15390425384976e-05
Friday      2                          If today is Saturday, tomorrow is   Friday                            true
Saturday    3                          If today is Sunday, tomorrow is     Saturday                          true
```

`plan.run.positions["answer.at_said"]` holds `"alignment_missing"` in place of those rows' position lists, and the saved tensor is NaN on them.

## Causes

What a record names beyond the prefill depends on what each row generates, so a row that does not fit is a result, not an error ([concepts/positions.md](../concepts/positions.md#decided-by-the-data-or-a-result)). Each cause below was produced on the corpus model:

| cause | example | what to do |
|---|---|---|
| the row stopped before a call the frame names | `{"frame": 3}` on a row whose third token was EOS | `min_new_tokens` equal to `max_new_tokens`, or a frame that follows each row (`"1:"`, `{"frame": -1, "index": -1}`); `"1:"` still misses a row whose first token was EOS |
| the row never generated the scope's text | `generated_answer.json`, rows 2 and 3 | it is the finding: count it, or score the text by teacher forcing (below) |
| the word is the row's last generated token | ` toss` said as token 5 of 5 | read with `before`, or give the row room to go on |
| fewer occurrences than `occurrence` picks | `"occurrence": 1` of a word said once | pick from the first, or `":"` |
| a static frame and index that cannot fit | `{"frame": "1:3", "index": "1:3"}`: two calls, an index two wide that finds one | an index that fits the frame's calls, one position each (`"index": 1`) |
| a `before` scope whose first token is position 0, beyond the prefill | a scope on the prompt's first word under a tokenizer that adds no BOS | nothing comes before position 0; scope a later token |
| a `before` read at the head on a prompt token of a `generate` | `{"frame": ":", "scope": {"text": " is", "before": true}}` at `lm_head_output` | the prefill computed the head at its last position only; read prompt positions in a `forward` |
| an online scoped write found no occurrence | a write at `{"frame": "1:", "scope": {"text": " toss"}, "index": -1}` on rows that never say it | nothing: the row is not written, and its generation is the unwritten one |

## In the prefill it is a refusal

The same misses in the prefill (`frame: 0`, every `forward`) are decided by the prompt alone, so they are refused by name before any forward, naming the row:

```text
clab: f.x: row 0: column 'pairs.cf_answer' is not in its input
clab: f.x: row 0: text ' is' occurs 2 times in the row's input; it has no occurrence 2
clab: f.x: row 1: tokens 0:10 do not fit its 9 scope tokens
clab: f.x: row 0: text '<s>' begins the row, at position 0, which no position comes before
```

## How it propagates

- **The entry's value** is NaN at every entry of a missing row; ids (`argmax`, `top_k`'s `ids`) are -1 there and `top_k` gives no token text.
- **Metrics and edits** of a NaN are NaN. A write never lands on a row it did not reach.
- **`mean`, `sum` and `min` skip holes.** A `mean` over rows averages the rows that were reached: on six rows of which two stopped early, a mean of `token_prob` at `{"frame": 3}` is the mean of four. An accuracy computed this way is the accuracy on the rows that got there, with nothing in the number to say so. A `sum` or `min` over `pos` of a row with no value is NaN.
- **Tables** give a missing row one line with `alignment_missing` true and no value; when several names are tabled together, the row carries the flag. Positions past a row's own on a ragged read are a different hole: they are left out of the table, with no flag. A read that no row reached, whose `pos` width `clab show` gives as 0 (`[4,0,16]`), tables to nothing at all rather than one flagged line per row; count its rows in `run.positions`.

The layout of holes in general, keys and ragged positions included, is [values-and-layout.md](../concepts/values-and-layout.md#missing-entries).

## Count it, prevent it, or refuse it

**Count** the missing rows of every scored read before reading an average:

```python
from clab import Plan

plan = Plan.load("out/generated_answer")
missing = [row for row, where in enumerate(plan.run.positions["answer.at_said"]) if where == "alignment_missing"]
assert not missing, f"rows {missing} never said their word"
```

```text
AssertionError: rows [2, 3] never said their word
```

A swept step's positions are a dict keyed by coordinate, so count each point; from a dose sweep of a steered generation with a read scoped to the word:

```python
for point, rows in plan.run.positions["steered.said"].items():
    print(point, [row for row, where in enumerate(rows) if where == "alignment_missing"])
```

```text
dose=0.0 [0, 1, 2, 3]
dose=0.08 [1]
dose=0.32 [0, 1, 2, 3]
```

The rows a scope reached are not always the rows a `sum` of `contains` counts. `contains` finds the text anywhere in the decoded string, inside a longer word too (`Finland` in `Finlandony`), while a scope matches on word boundaries only, so such a row scores 1 under `contains` and is missing to the scope.

**Prevent** the causes that come from the generation's length: hold EOS off with `min_new_tokens` (a row held past its EOS goes on as a different generation from there, not the first one extended: [generate.md](../usage/generate.md#rows-that-stop-early)), leave room after the answer (`max_new_tokens` beyond its last token), or read the positions that chose a word with `before`, which include the row's last token.

**Refuse** instead: beyond the prefill no option turns a missing row into an error, since only the generation decides it, while in the prefill every miss is one. So when the question is whether the model gives an answer, ask it in the prefill by teacher forcing, as `documents/forced_answer.json` does: append the answer to the prompt (`"input": ["days.input", "days.base_answer"]`) and read the head with `{"scope": {"column": "days.base_answer", "occurrence": -1, "before": true}}` in a `forward`. Every row is then scored (`cross_entropy`, `match`) and none is missing; a row whose input does not hold the text is refused before any forward. To stop on a missing generated row, check `run.positions` after the run, as above.
