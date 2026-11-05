---
title: Generation Experiments
one_liner: "Did the model say the answer, and how sure was it? contains or a whole-word scope, the text after a marker, a reasoning trace's length, teacher forcing, and chat turns."
tags: [patterns, generation, scoring, teacher-forcing, chat]
related: [docs/usage/generate.md, docs/usage/positions.md, docs/usage/ops.md, docs/usage/data.md, docs/errors/alignment-missing.md, docs/patterns/activation-patching.md]
sources: [clab/steps/generate.py, clab/compile/positions.py, clab/ops/metrics.py, clab/ops/transforms.py]
---

# Generation Experiments

## What this is for

These techniques score what a model says rather than its next-token distribution after a fixed prompt. Each technique below is a corpus document that runs on CPU. The checkpoint has random weights, so the numbers show the alignment and nothing else. The mechanics of a `generate` step (options, frames, writes, EOS) are in [usage/generate.md](../usage/generate.md).

## Did it say the answer: `contains` over the text

`documents/generated_contains.json` generates six greedy tokens per row, turns them into text, and scores whether the text contains the row's answer:

```json
"answer": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "min_new_tokens": 6, "do_sample": false},
"text": {"kind": "op", "op": "detokenize", "of": "answer"},
"said": {"kind": "op", "op": "contains", "of": "text", "text": "prompts.said"},
"acc": {"kind": "op", "op": "mean", "of": "said"}
```

```text
example_id  row  input                              text                                said
substr      0    If today is Thursday, tomorrow is  Handler substr energcalc tossorter  1.0
Friday      2    If today is Saturday, tomorrow is  Handler substr energcalc tossorter  0.0
```

This is plain generation accuracy, and no row is ever missing.

- **`contains` is a substring test.** A generated `Finlandony` contains `Finland`, so `contains` scores 1 on a row that never said the word. To count whole words, read with a scope over the decode calls ([below](#where-it-said-it-reads-in-the-generated-text)), where that row is `alignment_missing`.
- **Answer spellings** (`strip`, `lower`, a set of accepted answers) apply as they do for any column ([data.md](../usage/data.md#answer-columns)).
- **The first generated token loses its leading space when decoded** on a SentencePiece tokenizer. Give the column `strip: true` ([generate.md](../usage/generate.md#the-first-generated-tokens-leading-space)).
- **The token budget changes the result.** Swept as `{"sweep": [2, 6], "label": "n"}` on both `max_new_tokens` and `min_new_tokens`, row 1's `energcalc` is said by `n=6` and not by `n=2`.

## Score what follows a text: `after`

`contains` searches the whole text, so a thinking model's answer is found in its reasoning too. `after` cuts each row's text at the first occurrence of a column's text or a literal, and keeps what follows it:

```json
"answer_text": {"kind": "op", "op": "detokenize", "of": "answer", "after": {"text": "</think>"}},
"correct": {"kind": "op", "op": "contains", "of": "answer_text", "text": "test.gold"}
```

- **A row that never says the text keeps its whole text.** To tell it from a row that did, score a `contains` of the marker on the uncut text too.
- **`after` compares as `contains` does,** text on text, under the reference's `strip` and `lower`. Of a set, it cuts at the occurrence that ends first.
- **`after` does not take `tokens: true`,** which makes a row its token strings; the two together are refused.

On `documents/generated_contains.json`, `"after": "prompts.said"` keeps ` energcalc tossorter` on row 0 and the whole text on rows 2 and 3, which never say their word.

## A reasoning trace's length: `count` of an `until` read

A thinking trace's length in tokens is a `count` of a read at `{"frame": "1:", "scope": {"text": "</think>", "until": true}}` ([positions.md](../usage/positions.md#up-to-a-text-until)):

```json
"thinking": {"site": "head", "pos": {"frame": "1:", "scope": {"text": "</think>", "until": true}}},
"think_len": {"kind": "op", "op": "count", "of": "answer.thinking"}
```

The read runs through the first `</think>`, its last token included. A row that never closes it counts every input it was fed. No row of `documents/generated_contains.json`'s six-token generation says `</think>`, so there the count is 5 on every row.

## Where it said it: reads in the generated text

`documents/generated_answer.json` reads the head where each row said its own word:

```json
"at_said": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}},
"at_last": {"site": "head", "pos": {"frame": -1}}
```

`at_said` is the word's last token as an input to its call, and its logits are the distribution over the token after the word. A row that never says its word is a result, not an error:

```text
example_id  row  pos  pos_token  call  input                              cf_answer  value                   alignment_missing
substr      0    12   ▁substr    2     If today is Thursday, tomorrow is   Sunday    2.9666021873708814e-05
Friday      2                          If today is Saturday, tomorrow is   Friday                            true
```

A scope matches whole words. The fraction of rows that said the word as a whole word is the fraction not missing, and a `contains` count can be higher. How missing rows pass through ops, and how to count them, is in [errors/alignment-missing.md](../errors/alignment-missing.md#count-it-prevent-it-or-refuse-it).

## Confidence as it said the answer: `before`

`documents/generated_confidence.json` reads the positions that *chose* each token of the word:

```json
"logits": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said", "before": true}}},
"p":     {"kind": "op", "op": "token_prob",    "of": "answer.logits", "token": "prompts.said"},
"nll":   {"kind": "op", "op": "cross_entropy", "of": "answer.logits", "target": "prompts.said"},
"p_seq": {"kind": "op", "op": "sum", "of": "nll", "over": ["pos"]}
```

- A `before` scope over a word of n tokens reads the n positions that chose them.
- The first is the call fed the token before the word; for generated token 0, that is the prefill's last position.
- `sum` of `cross_entropy` over `pos` is the word's negative log-likelihood: 20.10 nats on row 1, whose `energcalc` is two tokens.

## A fixed answer, token by token: teacher forcing

To score an answer whatever the model would have said, append the answer to the prompt and read the positions that predict its tokens. This is a `forward`, not a generation. From `documents/forced_answer.json`:

```json
"forced": {
  "kind": "forward",
  "input": ["days.input", "days.base_answer"],
  "interventions": {
    "logits": {"site": "head", "pos": {"scope": {"column": "days.base_answer", "occurrence": -1, "before": true}}}
  }
},
"nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.base_answer"},
"seq_nll": {"kind": "op", "op": "sum", "of": "nll", "over": ["pos"]},
"hit": {"kind": "op", "op": "match", "of": "forced.logits", "expected": "days.base_answer"},
"exact": {"kind": "op", "op": "min", "of": "hit", "over": ["pos"]}
```

A list `input` joins the columns per row ([data.md](../usage/data.md#list-inputs-teacher-forcing)). Each of the answer's tokens is scored at its own position:

| value | per row | reads as |
|---|---|---|
| `nll`, `cross_entropy` | one per answer token: ` Wednesday` gives three, ` Monday` one | each token's surprisal |
| `seq_nll`, `sum` of `nll` over `pos` | one | the answer's negative log-likelihood |
| `exact`, `min` of `match` over `pos` | one | 1 where every token was the top prediction |
| `per_tok`, `mean` of `match` over `pos` | one | the fraction of tokens that were |

**Keep `occurrence: -1`.** The default, `occurrence: 0`, picks the first occurrence. That is in the prompt whenever the prompt says the answer too, and the run gives no warning ([positions.md](../usage/positions.md#words-and-occurrences)). clab does check that the scored column is what the input says at the read's positions. A row where it is not is refused before any forward:

```text
clab: nll: days.other row 0: position 13 predicts ' Monday' in the input, which is not token 0 of ' Friday' as it tokenizes alone ([' Friday']); …
```

### Scoring the model's own text

A list `input` takes dataset columns and texts only, so a `detokenize` step's text cannot be appended in the same document:

```text
clab: steps.forced.input: 'text' is not <dataset>.<column> or <dataset>.<column>[i]
```

Score it in two documents:

1. Generate, and save the `detokenize` step. Its file, `<name>.json`, holds one string per row.
2. Write each row's text into a column beside its prompt (a data file, or inline `rows`), with a leading space.
3. Teacher-force that column as above, scoped `"occurrence": -1`.

- **Without the leading space the text joins the prompt's last word,** and the scope refuses it: `column 'd.generated' ('Handler substr energcalc tossorter') is not in its input; it occurs only inside 'isHandler substr energcalc tossorter', cutting a word, …`.
- **The forced tokens are the text's own tokenization,** which can differ from the ids the model generated: the corpus checkpoint's `Handler` comes back as `▁Hand`, `ler`.
- **A gradient through generated text is taken this way too.** A generate step refuses `grad`: `grad is taken in a forward, not a generate, …`.

## A position inside a chat turn

`documents/chat_turn.json` gives a chat column (a list of `{"role", "content"}` messages), which is rendered through the checkpoint's chat template. It zeroes the last token of the second user message:

```json
"blank": {"site": "target", "pos": {"scope": {"turn": "user", "occurrence": 1}, "index": -1}, "ops": [{"op": "swap", "with": 0.0}]}
```

A `turn` scope holds the message's own tokens, and template markers belong to no turn. On row 0 the write lands at position 32, `▁is`, the question's own last token; the `[/INST]` that follows is no turn's. In a `generate` step over a chat, a `turn` scope reads or writes in the prefill (`frame: 0`). How the prompt ends, and whether a generation continues as the assistant, is the dataset's `template` ([data.md](../usage/data.md#chat-inputs)). `turn` scopes are in [usage/positions.md](../usage/positions.md).

## Does a patch reach later tokens

`documents/generate_probe.json` swaps the counterfactual's residual into the prompt's last position at the prefill, then reads the head at calls 0, 1 and 2. A patch at the prefill reaches later tokens only through the cache. To keep a write on at every generated token, write it at `{"frame": 0, "index": -1}` and at `{"frame": "1:"}` ([generate.md](../usage/generate.md#writes-during-generation)). The document's frame sweep is a sweep of a read, so one generation serves its three points, and `{"frame": "0:3"}` reads the same three distributions as one value ([usage/sweeps.md](../usage/sweeps.md#widen-the-read-instead)). The patching itself is covered in [activation-patching.md](activation-patching.md).

## Interpretation tips

- **Always set `do_sample: false`.** Otherwise the checkpoint's own config decides whether to sample, and a sampled run does not repeat ([generate](../usage/generate.md#options)).
- **Set `min_new_tokens` equal to `max_new_tokens`** when you compare runs. Otherwise rows that stop on EOS make fewer calls, and a read at a fixed call is missing on them ([generate](../usage/generate.md#rows-that-stop-early)).
- **Generated versus forced.** `contains` and reads in the generated text measure what this run said. Teacher forcing measures one fixed answer, in a single forward, whatever the model would have said. A model can give a forced answer a high probability and still not say it.
