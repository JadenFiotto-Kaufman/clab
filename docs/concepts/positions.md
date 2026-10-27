---
title: The Position Model
one_liner: "The position model: every input token is a position, a generation is a timeline of calls, where a write lands, scoped writes found online, static widths versus ragged rows, and why some rows are alignment_missing."
tags: [concepts, positions, generation, alignment]
related: [docs/usage/positions.md, docs/usage/generate.md, docs/errors/alignment-missing.md, docs/concepts/values-and-layout.md, docs/usage/output.md]
sources: [clab/compile/positions.py, clab/format/selectors.py, clab/plan.py, clab/steps/generate.py, clab/steps/batches.py]
---

# The Position Model

## The idea

A position is where one input token sits in one row. Every token the model is fed is a position, numbered from 0 in the order it is fed: BOS where the tokenizer adds one, a chat template's markers, the prompt's words, then in a generation each generated token that is fed back. Padding is never a position. One numbering serves everywhere a token is counted: a document's `pos`, the run record's `positions`, `clab table`'s `pos` and `key`, an attention pattern's `key` axis, and the token counts in refusals.

So a position names a token of *this row*, never a column of the padded batch. Rows run in batches padded on the left to the input's longest row, with position ids that count real tokens; `pos: -1` is each row's own last token whatever its length, and `pos: 0` its BOS. Recipes for pointing at tokens are in [usage/positions.md](../usage/positions.md).

## Three stages: frame, scope, index

A `pos` narrows in three steps, each a selector over what the step before left ([FORMAT.md#pos](../../FORMAT.md#pos)):

1. **frame** picks forward calls. A `forward` makes one call, the prefill (0).
2. **scope** picks text among those calls' tokens: a column's value, a literal, or a chat turn, and which `occurrence` of it; or an image of the prompt, which `image` picks, and a `region` of it.
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

In the prompt, a scope's text is located by the tokenizer's character offsets; in generated tokens, by decoding growing prefixes of what the row said. Either way it is the tokens that cover the text's characters, and an occurrence must start where a token starts and end where one ends, with no letter or digit continuing a word it would cut: ` Sara` is not in ` Sarah`. Occurrences are ordered by where they end, then where they start. The prompt and the generated tokens are searched apart, so a text that straddles them is not found.

A read beyond the prefill searches the generated inputs only (the last generated token is not one) unless it is `before`, which searches the tokens the calls chose, the last included. The two can disagree about a word's end: when a row's last two tokens are `▁toss orter`, a read scoped to ` toss` finds it at the second-last token, since nothing it searched follows it, while a `before` read of the same scope sees `tossorter` and finds no word.

## Where a write lands

In a forward, a write replaces the value at its positions as the forward reaches its site, and every later site and position computes from the written value.

In a generation, a write lands as its call runs. A write in the prefill changes the prompt's activations, and the generated tokens see it only through the cache. A write at decode call k changes generated token k onward and leaves tokens before k as they were. Because the call runs before the row's future is known, a write's positions must be decidable call by call: no frame counted from the row's last call, no index counted back over several calls, no `before`. The refusals for each are in [usage/generate.md](../usage/generate.md#writes-during-generation).

A write with a `text` or `column` scope over decode calls is **found online** (`clab/compile/positions.py` `landing`): at each call clab decodes the inputs fed so far, and the write lands at the call fed an occurrence's last token, as that occurrence appears. It sees nothing past the token it is fed, so an occurrence's end counts as a word's end whatever the model says next. With the same `pos`, a write and a read can therefore resolve differently: on a row that generates `▁toss orter acters`, a write scoped to ` toss` lands on `▁toss`, and a read of the same scope, which sees `orter` follow, finds nothing. Compare both in `run.positions`.

## Static widths and ragged rows

Positions are decided per row, so the number of positions a `pos` names can differ by row. When the `pos` alone fixes that number, the record has a **static width** ([usage/positions.md](../usage/positions.md#static-width-or-ragged) lists the forms that do): every row holds that many positions and the value is `(rows, width, ...)`. Otherwise it is **ragged**: as wide as its widest row, each row's positions first in their order, NaN after them ([values-and-layout.md](values-and-layout.md#missing-entries)).

A static width is a promise the data must keep. A ragged read keeps none, so it cannot pair position by position with another forward's write; it can be reduced (a `mean` skips the NaN) or read at a static index.

## Decided by the data, or a result

What a record names in the prefill depends on the prompt alone, so preflight resolves it with the tokenizer before any forward (`clab/compile/positions.py` `resolve`). A row whose prompt lacks the scope's text, has fewer occurrences than `occurrence` picks, or is too short for a static index, is refused by name, with the row's number: the document cannot be what was meant.

Beyond the prefill, what a record names depends on what the model generates, which no check can know beforehand. So a row it does not fit is a result, not an error: the row is `alignment_missing`, recorded so in `run.positions`, and its value is NaN all the way through. The causes, how missing rows pass downstream, and how to keep them out are [errors/alignment-missing.md](../errors/alignment-missing.md).
