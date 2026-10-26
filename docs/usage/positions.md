---
title: Positions
one_liner: "Pointing pos at the tokens you mean: the last token, an answer's tokens, a word by occurrence, the token before one, a chat turn, every token, the generated tokens, and reading back where each read and write landed."
tags: [usage, positions, pos, scope, occurrence, before]
related: [docs/concepts/positions.md, docs/usage/sites.md, docs/usage/generate.md, docs/usage/output.md, docs/errors/alignment-missing.md, docs/usage/data.md, docs/usage/interventions.md]
sources: [clab/compile/positions.py, clab/format/selectors.py, clab/plan.py]
---

# Positions

An entry's `pos` says which token positions of each row it reads or writes at its site. A row's positions are its input tokens counted from 0, BOS and a chat template's markers included, padding never; the fields and the grammar are in [FORMAT.md#pos](../../FORMAT.md#pos), and why it works this way is [concepts/positions.md](../concepts/positions.md). Leaving `pos` out means every prompt token.

## Check where it landed

Every read and write records the positions it resolved to on every row, saved or not, in `plan.run.positions` (the manifest's `run.positions`). Look before you trust a number:

```python
from clab import Plan

plan = Plan.load("out/entity_patch")
tokens = plan.run.tokens["pairs.input"]
for row, where in enumerate(plan.run.positions["patched.patch"]):
    print(row, where, [tokens[row][p] for p in where])
```

```text
0 [6] ['day']
1 [4] ['▁Friday']
2 [4] ['▁Saturday']
3 [4] ['▁Sunday']
```

For a saved value, `clab table OUT NAME` prints the same thing as its `pos` and `pos_token` columns. In a generation a position past the prompt indexes the generated tokens, `plan.run.tokens[<prompt input>][row] + plan.run.tokens[<step>][row]`, and a row the entry did not reach holds `"alignment_missing"` instead of a list. The run record's other fields are in [output.md](output.md#the-run-record).

## Recipes

Each recipe was checked against `run.positions` on row 0 of the corpus document named. Row 0's prompt in these documents is `<s> ▁If ▁today ▁is ▁Th urs day , ▁tom orrow ▁is`, positions 0 to 10 (in `documents/chat_turn.json`, the chat around it).

| you mean | `pos` | lands on row 0 (document) |
|---|---|---|
| the last prompt token | `-1` | 10 `▁is` (`documents/patching.json`) |
| BOS | `0` | 0 `<s>` |
| a window of the last tokens | `{"index": "-4:-1"}` | 7, 8, 9: `,` `▁tom` `orrow` (`documents/window_patch.json`) |
| the last three tokens, the last included | `{"index": "-3:"}` | 8, 9, 10: `▁tom` `orrow` `▁is` |
| a few positions, in this order | `{"index": [-1, 0]}` | 10 then 0 |
| every prompt token | `{"index": ":"}`, or no `pos` | 0 to 10 |
| the last token of each row's own word | `{"scope": {"column": "pairs.entity"}, "index": -1}` | 6 `day` (`documents/entity_patch.json`) |
| every token of that word | `{"scope": {"column": "prompts.entity"}}` | 4, 5, 6 (`documents/entity_mean_ablation.json`) |
| a fixed text, first occurrence | `{"scope": {"text": " is"}}` | 3 |
| a fixed text, last occurrence | `{"scope": {"text": " is", "occurrence": -1}}` | 10 |
| the positions that predict a word | `{"scope": {"column": "pairs.entity", "before": true}}` | 3, 4, 5 |
| an appended answer, scored token by token | `{"scope": {"column": "days.base_answer", "occurrence": -1, "before": true}}` | see below (`documents/forced_answer.json`) |
| the last token of the second user message | `{"scope": {"turn": "user", "occurrence": 1}, "index": -1}` | 32 `▁is` (`documents/chat_turn.json`) |
| where the model said a word | `{"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}` | 12 `▁substr` (`documents/generated_answer.json`) |
| the calls that chose that word | `{"frame": "1:", "scope": {"column": "prompts.said", "before": true}}` | 11 (`documents/generated_confidence.json`) |
| a generated word, its leading space optional ([why](generate.md#the-first-generated-tokens-leading-space)) | `{"frame": "1:", "scope": {"column": {"column": "prompts.said", "strip": true}}, "index": -1}` | 12 `▁substr` (`generated_answer.json` with `strip`) |

## Words and occurrences

A `scope` names text, and `index` then counts that text's tokens. `{"column": "<dataset>.<column>"}` is each row's own value of that column; `{"text": ...}` is the same literal on every row. A list of either finds every occurrence of each.

- **`occurrence` defaults to 0, the first.** `{"scope": {"text": " is"}, "index": -1}` is the last token of the *first* ` is`, position 3, not the prompt's last token. Write `"occurrence": -1` for the last one, `":"` for all of them. A word that also appears earlier (a name in the instructions, an answer in a few-shot example) is found there first: in `documents/chat_turn.json` the entity is in the assistant's answer and again in the second user message, and `{"column": "chats.entity", "occurrence": -1}` is the second.
- **A text matches on token and word boundaries only.** `tom` does not occur in `▁tom orrow`, and `Thurs` does not occur in `▁Th urs day`; in the prompt both are refused with `text 'tom' is not in its input`. Write the whole word as the tokenizer spaces it. A leading space is optional (`is` finds `▁is`), and `lower` and `strip` loosen the comparison as [data.md](data.md#answer-columns) describes.
- **Occurrences are ordered by where they end.** With `"text": [" today is Thursday", " is"]`, occurrence 0 is the ` is` at 3, occurrence 1 is ` today is Thursday` (2 to 6), occurrence 2 the ` is` at 10.
- **The BOS a tokenizer adds covers no text**, so no scope finds it; `0` is how to name it. A `<s>` that a chat template writes is text and can be found.

## The token before: `before`

`before: true` turns a scope into the positions that predict its tokens: the position just before each one, the same number of them. That is where to read the head to score a word: `documents/forced_answer.json` appends each row's answer to its prompt (`"input": ["days.input", "days.base_answer"]`) and reads the head at the positions before the appended answer. On a row whose answer is ` Wednesday` (`▁Wed nes day`), the read lands on 10, 11, 12 (`▁is ▁Wed nes`), each scored against the next token ([ops.md](ops.md)). `occurrence: -1` makes it the appended copy if the prompt says the word too. A scope whose first token is position 0 has nothing before it: in the prefill that is refused, beyond it the row is `alignment_missing`.

## Chat turns

`{"scope": {"turn": "<role>"}}` is one message of that role, `occurrence` counting that role's messages (`":"` is all of them). Its tokens are the message's content as the template renders it; the template's markers (`[INST]`, `[/INST]`, `</s>`) belong to no turn, so `index: -1` of a turn is the message's own last token. Without a scope, `-1` is the template's close of the last message: 36 `]` in `documents/chat_turn.json`. A turn takes frame 0, and on an input that is text, not a chat, it is refused. Rendering chats is [data.md](data.md#chat-inputs).

## Generated tokens

In a `generate` step, `frame` picks forward calls: 0 is the prefill, decode call k holds generated token k−1 as its input. The frames, the head's prefill and writes during generation are in [generate.md](generate.md#frames-which-forward-call); the reading recipes:

- **Every generated input**: `{"frame": "1:"}`. The row's last generated token is never an input, so it is not there.
- **The last call**: `{"frame": -1, "index": -1}`. Keep the `index`: on a row whose first generated token was EOS, the last call is the prefill, and `{"frame": -1}` alone is then every prompt token.
- **Text the model generated**: a scope with `"frame": "1:"`, as in the table. A row that never says the text, or says it as its very last token, is `alignment_missing`; with `before` the last token counts. See [alignment-missing](../errors/alignment-missing.md).
- **Prompt and generated tokens together**: `{"frame": ":"}` on a residual site is every prompt token followed by each generated input.

## Static width or ragged

A read has a **static width**, the same number of positions on every row, when its frame names a fixed number of calls (a forward's one call, an int, picks, `"1:3"`), its index a fixed number of positions (an int, picks, or a slice whose two ends count from the same side: `"-4:-1"`, `"-3:"`, `":2"`), and a text scope, if it has one, is one occurrence in the prefill. Anything else is **ragged**: an index of `":"` or `"2:"`, a whole scope (an entity of one token on one row and three on another), several occurrences, `"frame": "1:"`.

- In the prefill, a static index that does not fit a row is refused before any forward: `{"index": "0:10"}` on a 9-token row gives `row 1: tokens 0:10 do not fit its 9 scope tokens`. Beyond the prefill the row is `alignment_missing`. A ragged index takes what the row has, which can be nothing: on a row too short for `{"index": "10:"}` a read holds no position, its value is NaN there and `clab table` leaves the row out, while a write is refused ([position-not-in-row.md](../errors/position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row)).
- A ragged value is as wide as its widest row. Row r's j-th position sits at index j, and past a row's own positions the value is NaN; `clab table` leaves those out. So index j of `{"index": ":"}` is token j on every row, while index 0 of `{"index": "-4:-1"}` is a different absolute position on rows of different lengths.
- A ragged read cannot be another forward's operand, since its rows do not pair position by position. Read the operand at a static index (`documents/entity_patch.json`, `"index": -1`) or reduce it to one vector first (`documents/entity_mean_ablation.json`, a `mean` over every entity token). How an operand pairs with a write on prompts of different lengths is [interventions.md](interventions.md).
