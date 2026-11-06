---
title: Positions
one_liner: "Which tokens does an entry touch? pos for the last token, a word, the token before it, a chat turn, text up to a marker and generated tokens, and checking where it landed."
tags: [usage, positions, pos, scope, occurrence, until, before]
related: [docs/concepts/positions.md, docs/usage/sites.md, docs/usage/generate.md, docs/usage/output.md, docs/errors/alignment-missing.md, docs/errors/operand-does-not-fit.md, docs/usage/data.md, docs/usage/interventions.md]
sources: [clab/compile/positions.py, clab/format/selectors.py, clab/plan.py]
---

# Positions

An entry's `pos` says which token positions of each row it reads or writes at its site. A row's positions are its input tokens counted from 0, BOS and a chat template's markers included, padding never. The fields and the grammar are in [FORMAT.md#pos](../../FORMAT.md#pos), and why it works this way is [concepts/positions.md](../concepts/positions.md). Leaving `pos` out means every prompt token.

## Check where it landed

Every read and write records the positions it resolved to on every row, saved or not. The record is the entry's `positions`, under the dataset whose rows they are. In the manifest, each entry of `steps` names its list by index in the top-level `positions`, which holds each distinct list once. `clab show` prints them one line per entry; look before you trust a number:

```python
from clab import Plan

plan = Plan.load("out/entity_patch")
tokens = plan.run.tokens["pairs.input"]
patch = plan.steps["patched"].interventions["patch"]
for row, where in enumerate(patch.positions["pairs"]):
    print(row, where, [tokens[row][p] for p in where])
```

```text
0 [6] ['day']
1 [4] ['▁Friday']
2 [4] ['▁Saturday']
3 [4] ['▁Sunday']
```

The positions are absolute, counted from each row's BOS. On prompts of different lengths a read and a write at the same `-1` print different numbers on one row (`[8]` and `[10]` on row 0 of `documents/patching.json`), so compare their tokens, or count from each row's end. For a saved value, `clab table OUT NAME` prints the same thing as its `pos` and `pos_token` columns. In a generation a position past the prompt indexes the generated tokens, `plan.run.tokens[<prompt input>][row] + plan.run.tokens[<step>][row]`. A row the entry did not reach holds `"alignment_missing"` instead of a list. The run record's other fields are in [output.md](output.md#the-run-record).

## Recipes

Row 0's prompt is `<s> ▁If ▁today ▁is ▁Th urs day , ▁tom orrow ▁is`, positions 0 to 10. `documents/chat_turn.json` holds the chat around it, and `documents/forced_answer.json` has rows of its own, so its row names two landings. A row that names a corpus document gives that document's record; the others are the same prompt read at that `pos`.

The last column says whether a read at that `pos` pairs, position by position, with a value of another forward: yes; no; or where rows agree, which means where every row's scope, or every prompt, holds as many tokens ([static width or ragged](#static-width-or-ragged)).

| you mean | `pos` | lands on row 0 (document) | pairs |
|---|---|---|---|
| the last prompt token | `-1` | 10 `▁is` (`documents/patching.json`) | yes |
| BOS | `0` | 0 `<s>` | yes |
| a window of the last tokens | `{"index": "-4:-1"}` | 7, 8, 9: `,` `▁tom` `orrow` (`documents/window_patch.json`) | yes |
| the last three tokens, the last included | `{"index": "-3:"}` | 8, 9, 10: `▁tom` `orrow` `▁is` | yes |
| a few positions, in this order | `{"index": [-1, 0]}` | 10 then 0 | yes |
| every prompt token | `{"index": ":"}`, or no `pos` | 0 to 10 | where rows agree |
| the last token of each row's own word | `{"scope": {"column": "pairs.entity"}, "index": -1}` | 6 `day` (`documents/entity_patch.json`) | yes |
| every token of that word | `{"scope": {"column": "prompts.entity"}}` | 4, 5, 6 (`documents/entity_mean_ablation.json`) | where rows agree |
| a fixed text, first occurrence | `{"scope": {"text": " is"}}` | 3 | where rows agree |
| a fixed text, last occurrence | `{"scope": {"text": " is", "occurrence": -1}}` | 10 | where rows agree |
| the positions that predict a word | `{"scope": {"column": "pairs.entity", "before": true}}` | 3, 4, 5 | where rows agree |
| an appended answer, scored token by token | `{"scope": {"column": "days.base_answer", "occurrence": -1, "before": true}}` | 8 on row 0 (` Monday`, one token); 10, 11, 12 on row 1 (` Wednesday`) (`documents/forced_answer.json`) | where rows agree |
| the last token of the second user message | `{"scope": {"turn": "user", "occurrence": 1}, "index": -1}` | 32 `▁is` (`documents/chat_turn.json`) | yes |
| where the model said a word | `{"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}` | 12 `▁substr` (`documents/generated_answer.json`) | no |
| the calls that chose that word | `{"frame": "1:", "scope": {"column": "prompts.said", "before": true}}` | 11 (`documents/generated_confidence.json`) | no |
| a generated word, its leading space optional ([why](generate.md#the-first-generated-tokens-leading-space)) | `{"frame": "1:", "scope": {"column": {"column": "prompts.said", "strip": true}}, "index": -1}` | 12 `▁substr` | no |
| every generated input up to and including a word, all of them where it is never said | `{"frame": "1:", "scope": {"column": "prompts.said", "until": true}}` | 11, 12: `Handler` `▁substr` | no |
| every prompt token up to and including a text | `{"scope": {"text": " is", "until": true}}` | 0 to 3 | no |

## Words and occurrences

A `scope` names text, and `index` then counts that text's tokens. `{"column": "<dataset>.<column>"}` is each row's own value of that column; `{"text": ...}` is the same literal on every row. A list of either finds every occurrence of each.

- **`occurrence` defaults to 0, the first.** `{"scope": {"text": " is"}, "index": -1}` is the last token of the *first* ` is`, position 3, not the prompt's last token. Write `"occurrence": -1` for the last one, `":"` for all of them. A word that also appears earlier (a name in the instructions, an answer in a few-shot example) is found there first. In `documents/chat_turn.json` the entity is in the assistant's answer and again in the second user message, and `{"column": "chats.entity", "occurrence": -1}` is the second.
- **A text matches on token and word boundaries only.** `tom` does not occur in `▁tom orrow`, and `Thurs` does not occur in `▁Th urs day`; in the prompt both are refused with `text 'tom' is not in its input`. Write the whole word as the tokenizer spaces it.
- **A leading space is optional one way only.** `is` finds `▁is`. A column whose text starts with a space does not find the word at the start of the prompt, where no space comes before it; `strip: true` finds both. `lower` and `strip` loosen the comparison as [data.md](data.md#answer-columns) describes.
- **Occurrences are ordered by where they end.** With `"text": [" today is Thursday", " is"]`, occurrence 0 is the ` is` at 3, occurrence 1 is ` today is Thursday` (2 to 6), occurrence 2 the ` is` at 10.
- **The BOS a tokenizer adds covers no text**, so no scope finds it; `0` is how to name it. A `<s>` that a chat template writes is text and can be found.

## Up to a text: `until`

- **`until: true` makes a scope every token of the frame** from its first token up to and including the last token of the occurrence it picks.
- **A row without the text gets every token of the frame**, and so does a row with fewer occurrences than `occurrence` picks. An `until` scope is never refused for an absent text, in the prefill either; the record's `positions` say per row where each ended.
- **`occurrence: -1` runs to the last occurrence.**
- **The scope searches its frame only.** Over `"frame": "2:"`, a word fed as an input at call 1 is not found, and the row gets the whole frame.
- **With `before`, it is the positions that predict those tokens**: in a generation, the calls that chose them.
- **`until` with `before` takes a frame of decode calls** (`"1:"`). Over a frame that holds the prompt it is refused at compile: `frame ':' holds the prompt, and until with before starts before the frame's first token, …`.
- **An `until` scope has no static width** and pairs with no other forward's value.
- **A write beyond the prefill cannot take an `until` scope**: `a write's positions beyond the prefill are decided call by call, …`. A read can, and so can a write in the prefill.
- **A `count` over `pos` of an `until` read is a length in tokens.** A thinking trace's length is that recipe ([generation experiments](../patterns/generation-experiments.md#a-reasoning-traces-length-count-of-an-until-read)).

## The token before: `before`

- **`before: true` makes a scope the position just before each of its tokens**, as many positions as tokens.
- **Read the head with `before` to score a word.** Each position is scored against the next token ([ops.md](ops.md#metrics)).
- **Teacher-force an answer by appending it** (`"input": ["days.input", "days.base_answer"]`) and scoping it with `"occurrence": -1`, so a prompt that also says the word is not picked. `documents/forced_answer.json` does this; its landings are the table's row.
- **A scope whose first token is position 0 has nothing before it.** In the prefill that is refused; beyond it the row is `alignment_missing`.
- **One occurrence of a `before` scope in the prefill pairs** with another forward's value where every row holds as many tokens, as the scope without `before` does.

## Chat turns

`{"scope": {"turn": "<role>"}}` is one message of that role, `occurrence` counting that role's messages (`":"` is all of them). Its tokens are the message's content as the template renders it. The template's markers (`[INST]`, `[/INST]`, `</s>`) belong to no turn, so `index: -1` of a turn is the message's own last token. Without a scope, `-1` is the template's close of the last message: 36 `]` in `documents/chat_turn.json`. A turn takes frame 0, and on an input that is text, not a chat, it is refused. A turn is a message the row holds: a system header that a template writes when the chat has none is positions but no turn, and `{"turn": "system"}` there is refused with `row 0: the row has no 'system' message`. Rendering chats is [data.md](data.md#chat-inputs).

## Generated tokens

In a `generate` step, `frame` picks forward calls; which token each call holds is [generate.md](generate.md#frames-which-forward-call). The reading recipes:

- **Every generated input**: `{"frame": "1:"}`. The row's last generated token is never an input, so it is not there.
- **The last call**: `{"frame": -1, "index": -1}`. Keep the `index`: on a row whose first generated token was EOS, the last call is the prefill, and `{"frame": -1}` alone is every prompt token.
- **Text the model generated**: a scope with `"frame": "1:"`, as in the table. A row that never says the text, or says it as its very last token, is `alignment_missing`; with `before` the last token counts. See [alignment-missing](../errors/alignment-missing.md#causes).
- **Prompt and generated tokens together**: `{"frame": ":"}` on a residual site is every prompt token followed by each generated input.

## Static width or ragged

A read pairs, position by position, with a value of another forward only where its width is fixed: by the `pos` at compile, or in preflight by the prefill's rows, for an open index or one scope occurrence. Which `pos` forms fix it:

| `pos` form | its width | pairs with another forward's value |
|---|---|---|
| a static index over a frame of a fixed number of calls: an int, picks, or a slice whose two ends count from one side (`-1`, `[-1, 0]`, `"-4:-1"`, `"-3:"`, `{"frame": -1, "index": -1}`, `{"frame": 3, "index": 0}`) | static: the `pos` fixes it, and compile writes it | yes |
| such an index on one occurrence of a scope in the prefill, not `until` (`{"scope": {"column": "pairs.entity"}, "index": -1}`) | static | yes |
| one occurrence of a scope in the prefill, not `until`, read whole, `before` or not | the count each row's scope holds, which preflight confirms | where every row's scope holds as many tokens (`documents/entity_patch_scope.json`) |
| an open index in the prefill: `{"index": ":"}`, `"2:"`, no `pos` | the count each row's prompt gives it, which preflight confirms | where every row holds as many positions (`documents/ablation_pair.json` at `":"` in place of `"0:12"`) |
| several occurrences (`"occurrence": ":"`) or an `until` scope, with an `index` or without; a scope beyond the prefill | open | no |
| a frame of no fixed number of calls (`"1:"`, `":"`), or a decode call with an open index (`{"frame": 3}`) | open | no, whatever the prompts |

- **Preflight records an open width where the rows agree.** In the prefill it resolves every row's positions before any forward and writes the count they share as the record's `width`. An open index or a scope occurrence there pairs at that width; a read whose width no row settles (`until`, several occurrences, beyond the prefill) pairs with nothing.
- **A read is ragged where its rows hold different numbers of positions**, such as an entity of one token on one row and three on another.
- **A ragged value is as wide as its widest row.** Row r's j-th position sits at index j, and past a row's own positions the value is NaN; `clab table` leaves those out.
- **Index j along `pos` is the read's j-th position on that row**, not position j. Index 0 of `{"index": "-4:-1"}` is a different absolute position on rows of different lengths ([values-and-layout.md](../concepts/values-and-layout.md#an-index-counts-along-the-value)).
- **To pair a ragged read**, read it at a static index (`documents/entity_patch.json`, `"index": -1`) or reduce it over `pos` first (`documents/entity_mean_ablation.json`, a `mean` over every entity token).
- **A pairing the table says no to is refused at compile; one the rows decide, in preflight.** An open index or a whole scope whose rows disagree is refused before any forward, naming each count's rows, and a loaded one at compile, since its positions came with it. A permuted read pairs as the read it permutes. The messages and their fixes are [operand-does-not-fit.md](../errors/operand-does-not-fit.md).
- **A static index that does not fit a row** is refused in the prefill (`row 1: index 0:10 does not fit the row's 9 positions at this frame`) and is `alignment_missing` beyond it.
- **An open index takes what each row has**, which can be nothing. A read with nothing on a row is `alignment_missing` there (`{"index": "10:"}` on a 9-token row, [alignment-missing](../errors/alignment-missing.md#causes)); a write is refused ([position-not-in-row.md](../errors/position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row)).

How an operand pairs with a write on prompts of different lengths is [interventions.md](interventions.md).
