---
title: Positions
one_liner: "Pointing pos at the tokens you mean: the last token, an answer's tokens, a word by occurrence, the token before one, a chat turn, every token, the generated tokens, and reading back where each read and write landed."
tags: [usage, positions, pos, scope, occurrence, until, before]
related: [docs/concepts/positions.md, docs/usage/sites.md, docs/usage/generate.md, docs/usage/output.md, docs/errors/alignment-missing.md, docs/usage/data.md, docs/usage/interventions.md]
sources: [clab/compile/positions.py, clab/format/selectors.py, clab/plan.py]
---

# Positions

An entry's `pos` says which token positions of each row it reads or writes at its site. A row's positions are its input tokens counted from 0, BOS and a chat template's markers included, padding never; the fields and the grammar are in [FORMAT.md#pos](../../FORMAT.md#pos), and why it works this way is [concepts/positions.md](../concepts/positions.md). Leaving `pos` out means every prompt token.

## Check where it landed

Every read and write records the positions it resolved to on every row, saved or not, on its own record: `positions`, under the dataset whose rows they are (in the manifest, each entry of `steps` names its list by index in the top-level `positions`, which holds each distinct list once). `clab show` prints them one line per entry; look before you trust a number:

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

For a saved value, `clab table OUT NAME` prints the same thing as its `pos` and `pos_token` columns. In a generation a position past the prompt indexes the generated tokens, `plan.run.tokens[<prompt input>][row] + plan.run.tokens[<step>][row]`, and a row the entry did not reach holds `"alignment_missing"` instead of a list. The run record's other fields are in [output.md](output.md#the-run-record).

## Recipes

Each recipe was checked against the entry's `positions` on row 0 of the corpus document named. Row 0's prompt in these documents is `<s> ▁If ▁today ▁is ▁Th urs day , ▁tom orrow ▁is`, positions 0 to 10 (in `documents/chat_turn.json`, the chat around it).

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
| every generated input up to and including a word, all of them where it is never said | `{"frame": "1:", "scope": {"column": "prompts.said", "until": true}}` | 11, 12 `Handler` `▁substr` (`generated_answer.json`) |
| every prompt token up to and including a text | `{"scope": {"text": " is", "until": true}}` | 0 to 3 |

## Words and occurrences

A `scope` names text, and `index` then counts that text's tokens. `{"column": "<dataset>.<column>"}` is each row's own value of that column; `{"text": ...}` is the same literal on every row. A list of either finds every occurrence of each.

- **`occurrence` defaults to 0, the first.** `{"scope": {"text": " is"}, "index": -1}` is the last token of the *first* ` is`, position 3, not the prompt's last token. Write `"occurrence": -1` for the last one, `":"` for all of them. A word that also appears earlier (a name in the instructions, an answer in a few-shot example) is found there first: in `documents/chat_turn.json` the entity is in the assistant's answer and again in the second user message, and `{"column": "chats.entity", "occurrence": -1}` is the second.
- **A text matches on token and word boundaries only.** `tom` does not occur in `▁tom orrow`, and `Thurs` does not occur in `▁Th urs day`; in the prompt both are refused with `text 'tom' is not in its input`. Write the whole word as the tokenizer spaces it. A leading space is optional (`is` finds `▁is`), and `lower` and `strip` loosen the comparison as [data.md](data.md#answer-columns) describes.
- **Occurrences are ordered by where they end.** With `"text": [" today is Thursday", " is"]`, occurrence 0 is the ` is` at 3, occurrence 1 is ` today is Thursday` (2 to 6), occurrence 2 the ` is` at 10.
- **The BOS a tokenizer adds covers no text**, so no scope finds it; `0` is how to name it. A `<s>` that a chat template writes is text and can be found.

## Up to a text: `until`

`until: true` turns a scope into every token the frame holds up to and including the last token of the occurrence it picks, rather than that occurrence's tokens. On a row that never says the text, or has fewer occurrences than `occurrence` picks, it is every token of the frame, so `until` is never refused for an absent text, in the prefill either; the record's `positions` say per row where each ended. That makes a length one read: in a thinking model's generation, `{"frame": "1:", "scope": {"text": "</think>", "until": true}}` is each row's generated inputs through its first `</think>`, and a `count` step over a read there is the thinking's length in tokens, a row that never closes it counting every input it was fed. On `documents/generated_answer.json` the rows say `substr` at generated token 1 and `energcalc` at tokens 2 and 3, and rows 2 and 3 never say their word: a read scoped `{"column": "prompts.said", "until": true}` lands on 11, 12; 9 to 12; and 9 to 13 on the last two, and its `count` is 2, 4, 5 and 5. `occurrence: -1` runs to the last occurrence. The scope is the frame's: over `"frame": "2:"` a word said at call 1 is not found, so the read is the whole frame, as on a row that never says it. With `before` it is the positions that predict those tokens, the calls that chose them in a generation, so its frame is decode calls: over a frame that holds the prompt it would start before the frame's first token, position 0 (at a generation's head, the prompt's last position, before which the prefill computes no head), and name nothing on any row, so compile refuses it (`frame ':' holds the prompt, and until with before starts before the frame's first token, …`). An until scope is ragged. A write beyond the prefill cannot take one: its positions there are decided call by call, as each runs, and `until` names positions up to an occurrence not yet seen then.

## The token before: `before`

`before: true` turns a scope into the positions that predict its tokens: the position just before each one, the same number of them. That is where to read the head to score a word: `documents/forced_answer.json` appends each row's answer to its prompt (`"input": ["days.input", "days.base_answer"]`) and reads the head at the positions before the appended answer. On a row whose answer is ` Wednesday` (`▁Wed nes day`), the read lands on 10, 11, 12 (`▁is ▁Wed nes`), each scored against the next token ([ops.md](ops.md)). `occurrence: -1` makes it the appended copy if the prompt says the word too. A scope whose first token is position 0 has nothing before it: in the prefill that is refused, beyond it the row is `alignment_missing`.

## Chat turns

`{"scope": {"turn": "<role>"}}` is one message of that role, `occurrence` counting that role's messages (`":"` is all of them). Its tokens are the message's content as the template renders it; the template's markers (`[INST]`, `[/INST]`, `</s>`) belong to no turn, so `index: -1` of a turn is the message's own last token. Without a scope, `-1` is the template's close of the last message: 36 `]` in `documents/chat_turn.json`. A turn takes frame 0, and on an input that is text, not a chat, it is refused. A turn is a message the row holds: a system header that the template writes when the chat has none (Llama-3.2's does) is positions but no turn, and `{"turn": "system"}` there is refused with `row 0: the row has no 'system' message`. Rendering chats is [data.md](data.md#chat-inputs).

## Generated tokens

In a `generate` step, `frame` picks forward calls: 0 is the prefill, decode call k holds generated token k−1 as its input. The frames, the head's prefill and writes during generation are in [generate.md](generate.md#frames-which-forward-call); the reading recipes:

- **Every generated input**: `{"frame": "1:"}`. The row's last generated token is never an input, so it is not there.
- **The last call**: `{"frame": -1, "index": -1}`. Keep the `index`: on a row whose first generated token was EOS, the last call is the prefill, and `{"frame": -1}` alone is then every prompt token.
- **Text the model generated**: a scope with `"frame": "1:"`, as in the table. A row that never says the text, or says it as its very last token, is `alignment_missing`; with `before` the last token counts. See [alignment-missing](../errors/alignment-missing.md).
- **Prompt and generated tokens together**: `{"frame": ":"}` on a residual site is every prompt token followed by each generated input.

## Static width or ragged

A read has a **static width**, the same number of positions on every row, when its frame names a fixed number of calls (a forward's one call, an int, picks, `"1:3"`), its index a fixed number of positions (an int, picks, or a slice whose two ends count from the same side: `"-4:-1"`, `"-3:"`, `":2"`), and a scope, if it has one, is one occurrence in the prefill, not `until`: the `pos` alone says the width, and compile writes it. Any other `pos` leaves the width open at compile. In the prefill, preflight resolves every row's positions before any forward and records the number they all hold where they agree (the record's `width`): a whole entity that is one token on every row, or `{"index": ":"}` over prompts of one length. Where the rows hold different numbers (an entity of one token on one row and three on another), the read is **ragged**. Only pairing with another forward's value needs the width to be one number (the last point below).

- In the prefill, a static index that does not fit a row is refused before any forward: `{"index": "0:10"}` on a 9-token row gives `row 1: tokens 0:10 do not fit its 9 scope tokens`. Beyond the prefill the row is `alignment_missing`. A ragged index takes what the row has, which can be nothing: on a row too short for `{"index": "10:"}` a read holds no position, the row is `alignment_missing` (NaN all the way, and one `clab table` line flagged `alignment_missing`), while a write is refused ([position-not-in-row.md](../errors/position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row)).
- A ragged value is as wide as its widest row. Row r's j-th position sits at index j, and past a row's own positions the value is NaN; `clab table` leaves those out. So index j of `{"index": ":"}` is token j on every row, while index 0 of `{"index": "-4:-1"}` is a different absolute position on rows of different lengths.
- A ragged read cannot be another forward's operand, since its rows do not pair position by position. Compile refuses the pairing wherever the `pos` leaves the width open, whatever the prompts hold (`{"index": ":"}` or `"2:"` without a scope, several occurrences, an `until` scope, a scope beyond the prefill, `"frame": "1:"`), but for a scope of one occurrence in the prefill, not `until`, which its rows decide: where they disagree it is refused before any forward, naming them (`'counterfactual.v_cf' holds 1 position on rows 0-1, 3 and 3 positions on row 2; …`), or at compile already for a loaded read, which carries its rows' positions; where every row's scope is one token it pairs as `"index": -1` does, to the bit (`documents/entity_patch_scope.json`). Read the operand at a static index (`documents/entity_patch.json`, `"index": -1`) or reduce it to one vector first (`documents/entity_mean_ablation.json`, a `mean` over every entity token). How an operand pairs with a write on prompts of different lengths is [interventions.md](interventions.md).
