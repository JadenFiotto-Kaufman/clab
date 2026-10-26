---
title: Generation Experiments
one_liner: "Scoring what the model says: generated answers, confidence before the answer, contains over decoded text, forced answers by teacher forcing, answer spellings, and positions inside chat turns."
tags: [patterns, generation, scoring, teacher-forcing, chat]
related: [docs/usage/generate.md, docs/usage/positions.md, docs/usage/ops.md, docs/usage/data.md, docs/errors/alignment-missing.md, docs/patterns/activation-patching.md]
sources: [clab/steps/generate.py, clab/compile/positions.py, clab/ops/metrics.py, clab/ops/transforms.py]
---

# Generation Experiments

## What this is for

These techniques score what a model says rather than its next-token distribution after a fixed prompt. They answer four questions: did it say the answer, how confident was it as it said it, how likely is a given answer token by token, and where does a chat put the tokens you care about. Each technique below is a corpus document that runs on CPU. The checkpoint has random weights, so the numbers show the alignment and nothing else. The mechanics of a `generate` step are in [usage/generate.md](../usage/generate.md).

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

This is plain generation accuracy, and no row is ever missing. The answer column's `strip`, `lower` and answer sets apply as they do for any column ([usage/data.md](../usage/data.md)). The token budget changes the result. Swept as `{"sweep": [2, 6], "label": "n"}` on both `max_new_tokens` and `min_new_tokens`, row 1's `energcalc` is said by `n=6` and not by `n=2`.

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

The fraction of rows that said the word is the fraction not missing. How missing rows pass through ops is in [errors/alignment-missing.md](../errors/alignment-missing.md).

## Confidence as it said the answer: `before`

`documents/generated_confidence.json` reads the positions that *chose* each token of the word. For generated token 0, that is the prefill's last position:

```json
"logits": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said", "before": true}}},
"p":     {"kind": "op", "op": "token_prob",    "of": "answer.logits", "token": "prompts.said"},
"nll":   {"kind": "op", "op": "cross_entropy", "of": "answer.logits", "target": "prompts.said"},
"p_seq": {"kind": "op", "op": "sum", "of": "nll", "over": ["pos"]}
```

Row 1's word `energcalc` is two tokens, so the row reads two positions (`▁substr` at call 2 chose `▁energ`, and `▁energ` at call 3 chose `calc`). `p_seq` is the word's negative log-likelihood, 20.10 nats on row 1. Use `before` here rather than reading at the word: a word that ends on the row's last generated token is never fed back as an input, so only `before` can read it.

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

A list `input` joins the columns per row. Each of the answer's tokens is scored at its own position, so ` Wednesday` (three tokens) gives its row three values and ` Monday` gives one. `sum` over `pos` is the answer's negative log-likelihood. `min` of `match` over `pos` says whether every token was the top prediction, and the document's `per_tok`, a `mean` over `pos`, gives the fraction that were.

Keep `occurrence: -1`, which picks the appended answer. The default, `occurrence: 0`, picks the first occurrence, and that is in the prompt whenever the prompt also contains the answer. For `"Sunday, then Monday. If today is Sunday, tomorrow is"` + `" Monday"`, occurrence 0 reads position 3, which predicts the prompt's own `▁Monday`, and the run gives no warning. clab does check that the scored column is what the input says at the read's positions. A row where it is not is refused before any forward:

```text
clab: nll: days.other row 0: position 13 predicts '▁Monday' in the input, which is not token 0 of ' Friday' as it tokenizes alone (['▁Friday']); the answer is scored as it tokenizes, so the input says it the same way
```

The rules for list inputs are in [usage/data.md](../usage/data.md), and how answers line up with positions is in [usage/ops.md](../usage/ops.md).

## Every spelling of the answer, and answer sets

`documents/answer_spellings.json` scores an answer as a class of spellings, and as any member of a set:

```json
"p_spelled": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": {"column": "qa.answer", "strip": true, "lower": true}},
"p_any":     {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "qa.accepted"}
```

Under `strip` and `lower`, ` Yes` stands for every token that reads `yes` once stripped and lowercased, and `token_prob` sums their probabilities. On row 0 that is 1.9e-4 against 3.4e-5 for ` Yes` alone. `qa.accepted` holds a list per row (`[" Yes", " True"]`), any member of which counts. `logit_diff` and `token_logit` refuse a class or a set, since neither has one logit. Column flags are covered in [usage/data.md](../usage/data.md) and metrics in [usage/ops.md](../usage/ops.md).

## A position inside a chat turn

`documents/chat_turn.json` gives a chat column (a list of `{"role", "content"}` messages), which is rendered through the checkpoint's chat template. It zeroes the last token of the second user message:

```json
"blank": {"site": "target", "pos": {"scope": {"turn": "user", "occurrence": 1}, "index": -1}, "ops": [{"op": "swap", "with": 0.0}]}
```

On row 0 this lands at position 32, `▁is`, the question's own last token. The `[/INST]` that follows belongs to no turn. To generate a reply from a chat, give the dataset `"template": {"add_generation_prompt": true}`. The same `turn` scope then reads or writes in the prefill (`frame: 0`). Rendering and templates are in [usage/data.md](../usage/data.md), and `turn` scopes are in [usage/positions.md](../usage/positions.md).

## Does a patch reach later tokens

`documents/generate_probe.json` swaps the counterfactual's residual into the prompt's last position at the prefill, then reads the head at calls 0, 1 and 2. The patch is made once, so it reaches later tokens only through the cache. The document sweeps the frame, which runs the generation three times. `{"frame": "0:3"}` reads the same three distributions in one generation, bit for bit ([usage/sweeps.md](../usage/sweeps.md#widen-the-read-instead)). The patching itself is covered in [activation-patching.md](activation-patching.md).

## Interpretation tips

- **Always set `do_sample: false`.** Otherwise the checkpoint's own config decides whether to sample, and a sampled run does not repeat ([generate](../usage/generate.md#options)).
- **Set `min_new_tokens` equal to `max_new_tokens`** when you compare runs. Otherwise rows that stop on EOS make fewer calls, and a read at a fixed call is missing on them ([generate](../usage/generate.md#rows-that-stop-early)).
- **Generated versus forced.** `contains` and reads in the generated text measure what this run said. Teacher forcing measures one fixed answer, in a single forward, whatever the model would have said. A model can give a forced answer a high probability and still not say it.
- **Leave room after the answer.** A word on the row's last generated token is missing to a read at the word itself.
