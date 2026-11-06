---
title: A Position Does Not Fit a Row
one_liner: "Why a run refuses with 'row i: … is not in its input' or 'do not fit its N scope tokens': a prefill position that names nothing on one row, how to see that row's tokens, and the fix for each message."
tags: [errors, positions, scope, preflight, refusals]
related: [docs/usage/positions.md, docs/concepts/positions.md, docs/errors/alignment-missing.md, docs/errors/index.md, docs/usage/data.md]
sources: [clab/compile/positions.py, clab/compile/preflight.py]
---

# A Position Does Not Fit a Row

In the prefill (frame 0, and the one call of every `forward` step), the data alone decides where a `pos` lands on each row. So clab resolves every row before any forward and refuses a text, an occurrence or a static index that a row does not have, naming the entry and the row (`clab/compile/positions.py` `prefill_positions`). An open-ended index that picks none of a short row's tokens is not refused for a read, nor is any miss beyond the prefill: the row is `alignment_missing` ([alignment-missing.md](alignment-missing.md)). A write there is refused.

## See the row's tokens

Each message names a row. A row's positions are its input tokens, BOS first and a chat template's markers included ([positions.md](../usage/positions.md#recipes)). A refusal writes no output directory, but the compiled plan already holds every input's tokens by then, so the step-by-step calls show the row that failed:

```python
from clab import Plan
from clab.engines import load

compiled = Plan("documents/my_experiment.json").compile()
engine = load("nnterp", compiled.model, "cpu")
try:
    engine.execute(compiled)
except ValueError as refusal:
    print(refusal)                                  # patched.patch: row 1: text 'Thursday' is not in its input
print(compiled.run.tokens["pairs.input"][1])        # ['<s>', '▁If', '▁today', '▁is', '▁Friday', ',', ...]
```

`run.tokens` is keyed by input (`pairs.input`, `pairs.counterfactual_inputs[0]`); [output.md](../usage/output.md#the-run-record) describes the run record.

## `… is not in its input`

```text
patched.patch: row 1: text 'Thursday' is not in its input
```

- **A `text` scope is one literal on every row.** A value that differs by row is a `column` scope, which searches each row for its own value; a list of texts finds any of them.
- **Text is found on token and word boundaries, as written.** [positions.md](../usage/positions.md#words-and-occurrences) says how case, whitespace, `lower` and `strip` are matched.
- **A text found only inside a word says so:** `row 0: text 'toda' is not in its input; it occurs only inside 'today', cutting a word, and an occurrence lies on word boundaries`.
- **A `column` scope's message quotes the row's value:** `row 0: column 'pairs.cf_answer' (' Sunday') is not in its input`.
- **The BOS covers no text,** so `{"text": "<s>"}` is never found; `pos: 0` is the BOS.

## `… occurs N times …; it has no occurrence K`

```text
patched.patch: row 0: text ' is' occurs 2 times in the row's input; it has no occurrence 2
```

The `occurrence` asks for more matches than the row holds; [positions.md](../usage/positions.md#words-and-occurrences) says how occurrences count. A list of picks (`[0, 2]`) needs every pick on every row.

## `tokens … do not fit its N scope tokens`

```text
patched.patch: row 0: tokens 40 do not fit its 11 scope tokens
```

An index of static width (an int, a list of picks, a slice like `"-4:-1"` or `"0:3"`) needs that many tokens on every row, counted in the whole prompt or, with a scope, in the scope's text. The shortest row decides.

In a `generate` step the prefill computes the head at the last position only, so an earlier prompt position of `lm_head_output` or `logits` does not fit:

```text
answer.logits: row 0: tokens -2: do not fit its 1 scope tokens; a generation's prefill computes lm_head_output at its last position only, so that is all call 0 holds of it
```

Read the logits of earlier prompt positions in a `forward` step on the same input.

## `the write's positions hold no token of this row`

```text
patched.patch: row 0: the write's positions hold no token of this row; a write needs one to write
```

An open-ended index (`"10:"`) has no static width, so on a row too short for it the write resolves to no position at all, and a write is refused. A read at the same index is not refused: those rows are `alignment_missing` and their values NaN ([alignment-missing.md](alignment-missing.md)).

## `… begins the row, at position 0, which no position comes before`

```text
patched.patch: row 0: text 'If' begins the row, at position 0, which no position comes before
```

A scope with `before` names the position before each of its tokens, and position 0 has none. It happens on tokenizers that add no BOS, such as GPT-2's, where the prompt's first word is position 0.

## `the row has no '<role>' message`

```text
blanked.blank: row 0: the row has no 'system' message
blanked.blank: row 0: the row has no 'system' message; the template wrote a 'system' header of its own, whose tokens are positions, not a message: locate its words with a {'text': ...} scope
```

A `turn` scope names a role of the chat in that row; check the role's spelling against the rows ([data.md](../usage/data.md#chat-inputs) covers chat inputs). The second form is a template that writes a header for a role the chat does not hold: those tokens are no message, so a `turn` scope never finds them, and a `text` scope does.

## Skipping a row instead

A refused row cannot be skipped: the data decided, so change the data (a `split`, a `take`, or a dataset of only the rows that fit).
