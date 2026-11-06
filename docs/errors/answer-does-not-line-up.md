---
title: An Answer Does Not Line Up
one_liner: "Why a metric refuses its answer column ('is 3 tokens; the read has 1 position', 'no one logit for logit_diff', 'predicts … in the input'): how answers are tokenized and lined up with the read, and the fix for each message."
tags: [errors, answers, metrics, tokens, refusals]
related: [docs/usage/ops.md, docs/usage/data.md, docs/patterns/generation-experiments.md, docs/patterns/probing.md, docs/errors/index.md, docs/errors/symptom-index.md]
sources: [nnspec/compile/answers.py, nnspec/ops/metrics.py]
---

# An Answer Does Not Line Up

A metric that scores a column (`match`, `token_prob`, `token_logit`, `logit_diff`, `cross_entropy`) tokenizes each row's answer as written, without special tokens, and scores answer token i at the read's position i. [data.md](../usage/data.md#answer-columns) says how answer columns and their flags are written, and [ops.md](../usage/ops.md#metrics) what each metric scores. Preflight checks every row against its read before any forward and refuses a row it cannot score, naming the op step, the column and the row (`nnspec/compile/answers.py` `check_tokens`). An answer that runs and scores the wrong spelling is not refused: [symptom-index.md](symptom-index.md#answers-and-metrics) lists those.

## `… is k tokens; the read has n position(s) on that row`

```text
iia: pairs.ans row 0: ' Tuesday' is 3 tokens; the read has 1 position on that row; token i of an answer scores the read's position i, so the read carries one per token (a scope with "before": true reads the positions that predict the answer's tokens)
```

- **The answer is several tokens and the read one position** (`pos: -1`). Teacher-force the answer, reading the positions before each of its tokens with a `before` scope ([generation-experiments.md](../patterns/generation-experiments.md#a-fixed-answer-token-by-token-teacher-forcing)), or give the column an answer of one token per row.
- **A missing leading space is a common cause on a byte-level BPE tokenizer.** GPT-2's tokenizer (`openai-community/gpt2`) writes `twelve` as `tw`, `elve`, and ` twelve` as one token. A column holding `twelve` is refused at `pos: -1` (`'twelve' is 2 tokens; the read has 1 position on that row`) where ` twelve` scores.
- **A hub column without the space cannot be fixed inside a document:** write its rows with the space added.

One ending names a whitespace-only token as the difference:

```text
nll: days.base_answer row 0: ' 42' is 3 tokens; the read has 2 positions on that row; …; its token 0, ' ', is whitespace only, which an answer keeps and a scope under strip does not cover: scope the column without strip
```

Some tokenizers write a number's space as its own token (the corpus Llama writes ` 42` as `▁`, `4`, `2`), which the answer keeps. A `strip` flag on the scope's column drops that token from the scope, so the read is a position short. Scope the column without `strip`: the scope then covers all three tokens, and the answer scores at each.

## `… a position each, and the value has no pos axis`

```text
tp: pairs.entity row 0: 'Thursday' is 3 tokens, a position each, and the value has no pos axis (['rows', 'vocab_size'])
```

The metric's `of` was reduced over `pos` before scoring. Score first, then reduce the scores.

## `… stands for N ids …, which have no one logit for logit_diff`

```text
logit_diff: pairs.cf row 0: [' Monday', ' Friday'] stands for 2 ids (a set, or a class under strip or lower), which have no one logit for logit_diff; token_prob scores them together
```

A set, or a token whose `strip` or `lower` class holds several ids, has no single logit, and `logit_diff` and `token_logit` read one. Use `token_prob` (their summed probability), `match` or `cross_entropy`; or give `logit_diff` a column of one spelling per row.

## `… its members are … tokens; a set's members are one width`

```text
iia: pairs.ans row 0: its members are ' Tuesday' 3, ' Wed' 1 tokens; a set's members are one width, the positions its read aligns
```

Every member of a set is lined up with the same positions, so the members tokenize to the same number of tokens. Split the alternatives into separate columns and score each, or keep only alternatives of one width.

## `… is no tokens once strip drops …`

```text
iia: pairs.ans row 0: ' ' is no tokens once strip drops the whitespace-only tokens at its end
```

Under `strip` an answer's whitespace-only tokens at its end are dropped, and an answer of whitespace alone leaves nothing to score. Fix the row's data.

## `position p predicts '…' in the input, which is not token i of …`

```text
nll: days.base_answer row 0: position 7 predicts ' 3' in the input, which is not token 0 of '3' as it tokenizes alone (['3']); the answer is scored as it tokenizes, so the input says it the same way
```

Teacher forcing reads where the input predicts each answer token, and the input must say the answer with the same tokens it has alone. A byte-level BPE tokenizer such as GPT-2's merges a prompt's trailing space into the token after it. So write the space on the answer (` 3`), not at the end of the prompt.

## A probe's classes

A metric over a linear probe's scores compares the column with the probe's classes exactly:

```text
fit.ce: target names a class of probe 'probe', which is compared exactly; strip and lower compare an answer's tokens
fit.held_out.ce: target is 'label' of 'te', whose row 0 is 'maybe', not one of the classes of probe 'probe': ['no', 'yes']
```

The classes are the distinct values of the probe's `classes` column. Drop the flags, and give every scored row one of those values. [probing.md](../patterns/probing.md#scoring-classes) covers probes.
