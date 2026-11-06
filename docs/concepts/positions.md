---
title: The Position Model
one_liner: "How does clab count positions: prompt tokens, generation calls, where a write lands, static and ragged widths?"
tags: [concepts, positions, generation, alignment]
related: [docs/usage/positions.md, docs/usage/generate.md, docs/errors/alignment-missing.md, docs/concepts/values-and-layout.md, docs/usage/output.md]
sources: [clab/compile/positions.py, clab/format/selectors.py, clab/plan.py, clab/steps/generate.py, clab/steps/batches.py]
---

# The Position Model

## The idea

A position is where one input token sits in one row. Every token the model is fed is a position, numbered from 0 in the order it is fed. That is BOS where the tokenizer adds one, a chat template's markers and the prompt's words, then in a generation each generated token that is fed back. Padding is never a position. One numbering serves everywhere a token is counted: a document's `pos`, a record's `positions`, `clab table`'s `pos` and `key`, an attention pattern's `key` axis, and the token counts in refusals.

So a position names a token of *this row*, never a column of the padded batch. Position ids count real tokens, so `pos: -1` is each row's own last token whatever its length, and `pos: 0` its BOS, however the batch is padded ([usage/running.md](../usage/running.md#batch-size-and-padding)). Recipes for pointing at tokens are in [usage/positions.md](../usage/positions.md).

## Three stages: frame, scope, index

A `pos` narrows in three steps, each a selector over what the step before left ([FORMAT.md#pos](../../FORMAT.md#pos)):

1. **frame** picks forward calls. A `forward` makes one call, the prefill (0).
2. **scope** picks text among those calls' tokens: a column's value, a literal, or a chat turn, and which `occurrence` of it.
3. **index** counts positions within what the scope left, over every selected call in order.

So `{"frame": "0:3", "index": -1}` and `{"frame": 2, "index": -1}` land on the same position, and `{"scope": {"text": " is"}, "index": -1}` is the last token of the first ` is`, because `occurrence` picks before `index` counts.

## A generation is a timeline of calls

A generation runs the prefill and then one decode call per new token, under the cache:

| call | holds | its head predicts |
|---|---|---|
| 0, the prefill | every prompt token, positions 0 to P−1 | generated token 0, from position P−1 |
| decode call k ≥ 1 | one position, P+k−1: generated token k−1 as input | generated token k |

A row's calls end with the one that chose its last token, so the token a row stops on, EOS or the `max_new_tokens`-th, is never an input and has no position of its own to read at. The prefill of a generation computes the head (`lm_head_output`, `logits`) at its last position only, so at the head call 0 holds that one position. A frame counted from the end (`-1`) is resolved per row after the row stops: on a row whose first token was EOS, `-1` is the prefill, all P positions. Frames in use are in [usage/generate.md](../usage/generate.md#frames-which-forward-call).

## How text is found

In the prompt, a scope's text is located by the tokenizer's character offsets; in generated tokens, by decoding growing prefixes of what the row said. Either way it is the tokens that cover the text's characters. An occurrence must start where a token starts and end where one ends, with no letter or digit continuing a word it would cut: ` Sara` is not in ` Sarah`. Occurrences are ordered by where they end, then where they start. The prompt and the generated tokens are searched apart, so a text that straddles them is not found.

A read beyond the prefill searches the generated inputs only, and the last generated token is not one. A `before` read searches the tokens the calls chose, the last included. So a word at the end of one can be cut in the other. On a row whose last two tokens are `▁toss orter`, a read scoped to ` toss` finds `▁toss`; a `before` read of the same scope sees `tossorter` and finds no word.

## Where a write lands

In a forward, a write replaces the value at its positions as the forward reaches its site, and every later site and position computes from the written value.

In a generation, a write lands as its call runs. A write in the prefill changes the prompt's activations, and the generated tokens see it only through the cache. A write at decode call k changes generated token k onward and leaves tokens before k as they were. Because the call runs before the row's future is known, a write's positions must be decidable call by call: no frame counted from the row's last call, no index counted back over several calls, no `before`, no `until`. The refusals for each are in [usage/generate.md](../usage/generate.md#writes-during-generation).

A write with a `text` or `column` scope over decode calls is **found online** (`clab/compile/positions.py` `lands_at_call`). At each call clab decodes the inputs fed so far, and the write lands at the call fed an occurrence's last token, as that occurrence appears. It sees nothing past the token it is fed, so an occurrence's end counts as a word's end whatever the model says next. A read of the same `pos` sees the whole generation, so the two can resolve differently. On a row that generates `▁toss orter acters`, a write scoped to ` toss` lands on `▁toss`; the read, which sees `orter` follow, finds nothing. Compare both in the two entries' `positions`.

## Static widths and ragged rows

Positions are decided per row, so the number of positions a `pos` names can differ by row. When the `pos` alone fixes that number, the record has a **static width**: every row holds that many positions and the value is `(rows, width, ...)`. Otherwise compile leaves the width open. In the prefill, preflight, which tokenizes the prompts, then records the number every row's positions hold where they agree. Where they disagree the record is **ragged**: as wide as its widest row, each row's positions first in their order, NaN after them ([values-and-layout.md](values-and-layout.md#missing-entries)).

A static width is a promise the data must keep; a recorded one is what the data says. Two values of different reads pair only at a fixed width, whether one forward or two made them. `clab/compile/positions.py` `pending_width` names the two open widths the prefill's rows fix, which may pair with another read's value: an open index over the prompt (`{"index": ":"}`), and one occurrence of a scope, not an until scope. Only the tokenizer can count them, so preflight confirms them from the rows. Which forms pair is [usage/positions.md](../usage/positions.md#static-width-or-ragged); the refusals are [operand-does-not-fit.md](../errors/operand-does-not-fit.md).

## Decided by the data, or a result

What a record names in the prefill depends on the prompt alone, so preflight resolves it with the tokenizer before any forward (`clab/compile/positions.py` `resolve`). It refuses, naming the row:

- a row whose prompt lacks the scope's text, but for an `until` scope, which takes the whole frame there;
- a row with fewer occurrences than `occurrence` picks, with the same exception;
- a row too short for a static index;
- a scope another read's value pairs with, where its rows hold different numbers of positions, naming each count's rows;
- an unscoped position read on one input and written on another, on a row where the two differ in length: a position pairs across prompts of different lengths only where every token between it and the end it is counted from is the same on both, whatever the two tokens it pairs (`clab/compile/interventions.py` `check_window`).

The document cannot be what was meant.

Beyond the prefill, what a record names depends on what the model generates, which no check can know beforehand. So a row it does not fit is a result, not an error: the row is `alignment_missing`, recorded so in the entry's `positions`, and its value is NaN all the way through, but in a `count` of it, which holds 0 there. `clab table` prints that row flagged `alignment_missing`, with no value, for the read, for its `count` and for every other value made from it. The causes, how missing rows pass downstream, and how to keep them out are [errors/alignment-missing.md](../errors/alignment-missing.md).
