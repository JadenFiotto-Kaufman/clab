---
title: A Position Does Not Fit a Row
one_liner: "Why a run refuses with 'row i: … is not in its input' or 'do not fit its N scope tokens': a prefill position that names nothing on one row, how to see that row's tokens, and the fix for each message."
tags: [errors, positions, scope, preflight, refusals]
related: [docs/usage/positions.md, docs/concepts/positions.md, docs/errors/alignment-missing.md, docs/errors/index.md, docs/usage/data.md]
sources: [clab/compile/positions.py, clab/compile/preflight.py]
---

# A Position Does Not Fit a Row

In the prefill (frame 0, and the one call of every `forward` step), the data alone decides where a `pos` lands on each row. So clab resolves every row before any forward and refuses a `pos` that names nothing on some row, naming the entry and the row (`clab/compile/positions.py` `indices`). Beyond the prefill the same miss is not a refusal: the row is `alignment_missing`, as [alignment-missing.md](alignment-missing.md) explains.

## See the row's tokens

Each message names a row. A row's positions are its input tokens, BOS first and a chat template's markers included, as [positions.md](../usage/positions.md) explains. A refusal writes no output directory, but the compiled plan already holds every input's tokens by then, so the step-by-step calls show the row that failed:

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

`run.tokens` is keyed by input (`pairs.input`, `pairs.counterfactual_inputs[0]`); [output.md](../usage/output.md) describes the run record.

## `… is not in its input`

```text
patched.patch: row 1: text 'Thursday' is not in its input
```

A `text` or `column` scope is found on token and word boundaries, as written. `Sara` is not in `Sarah`. Case and surrounding whitespace compare exactly unless the scope sets `lower` or `strip` (on a column, on its reference). The BOS a tokenizer adds covers no text, so `{"text": "<s>"}` is never found; `pos: 0` is the BOS. The usual cause is a `text` that is the same on every row while the rows differ, as here: row 1 says Friday. A `column` scope searches each row for its own value; a list of texts finds any of them.

## `… occurs N times …; it has no occurrence K`

```text
patched.patch: row 0: text ' is' occurs 2 times in the row's input; it has no occurrence 2
```

`occurrence` counts from 0, in order of where each occurrence ends; `-1` is the last. A list of picks (`[0, 2]`) needs every pick on every row.

## `tokens … do not fit its N scope tokens`

```text
patched.patch: row 0: tokens -40:-1 do not fit its 11 scope tokens
```

An index with a static width (an int, a list of picks, a slice like `"-4:-1"` or `"0:3"`) needs that many tokens on every row, counted in the whole prompt or, with a scope, in the scope's text. The shortest row decides. `pos: 40` on an 11-token row is the same refusal (`tokens 40 do not fit …`).

In a `generate` step the prefill computes the head at the last position only, so an earlier prompt position of `lm_head_output` or `logits` does not fit:

```text
patched.logits: row 0: tokens -2: do not fit its 1 scope tokens; a generation's prefill computes lm_head_output at its last position only, so that is all call 0 holds of it
```

Read the logits of earlier prompt positions in a `forward` step on the same input.

## `the write's positions hold no token of this row`

```text
patched.patch: row 0: the write's positions hold no token of this row; a write needs one to write
```

An index with an open end (`"10:"`) has no static width, so on a row too short for it the write resolves to no position at all, and a write is refused. A read at the same index is not refused: it holds nothing on those rows, its values are NaN and `clab table` leaves the rows out ([symptom-index.md](symptom-index.md)).

## `… begins the row, at position 0, which no position comes before`

```text
patched.patch: row 0: text 'If' begins the row, at position 0, which no position comes before
```

A scope with `before` names the position before each of its tokens, and position 0 has none. It happens on tokenizers that add no BOS, such as GPT-2's, where the prompt's first word is position 0.

## `the row has no '<role>' message`

```text
blanked.blank: row 0: the row has no 'system' message
```

A `turn` scope names a role of the chat in that row; check the role's spelling against the rows ([data.md](../usage/data.md) covers chat inputs).

## Skipping a row instead

In the prefill a row cannot be skipped: the data decided, so change the data (a `split`, `take`, or a dataset of only the rows that fit). Beyond the prefill (decode frames of a `generate` step) a row the `pos` does not reach is `alignment_missing`: its values are NaN and reductions leave it out, which [symptom-index.md](symptom-index.md) lists among the cases that run quietly.

## Related

- [index.md](index.md): every refusal by message.
- [positions.md](../usage/positions.md): `pos` recipes; [concepts/positions.md](../concepts/positions.md): the position model.
- [alignment-missing.md](alignment-missing.md): the same miss beyond the prefill.
