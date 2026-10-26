---
title: An Answer Does Not Line Up
one_liner: "Why a metric refuses its answer column ('is 3 tokens; the read has 1 position', 'no one logit for logit_diff', 'predicts … in the input'): how answers are tokenized and lined up with the read, and the fix for each message."
tags: [errors, answers, metrics, tokens, refusals]
related: [docs/usage/ops.md, docs/usage/data.md, docs/patterns/generation-experiments.md, docs/patterns/probing.md, docs/errors/index.md, docs/errors/symptom-index.md]
sources: [clab/compile/answers.py, clab/ops/metrics.py]
---

# An Answer Does Not Line Up

A metric that scores a column (`match`, `token_prob`, `token_logit`, `logit_diff`, `cross_entropy`) tokenizes each row's answer exactly as written, without special tokens, and scores answer token i at the read's position i. Preflight checks every row against its read before any forward (`clab/compile/answers.py` `check_tokens`) and refuses a row it cannot score, naming the op step, the column and the row.

## How an answer meets its read

- A column whose every row is one token scores that token at every position of the read, one score per position.
- A column with an answer of several tokens on any row lines every row up: token i at position i, so the read holds as many positions on that row as its answer has tokens. A scope with `before` reads exactly those positions: the ones that predict the answer's tokens.
- `logit_diff` and `token_logit` read one id's logit. A set (a row's list of strings) or a class (`strip`, `lower`) stands for several ids, which `match`, `token_prob` and `cross_entropy` score together.
- A set's members are tokenized alone and have the same number of tokens.

[ops.md](../usage/ops.md) covers metrics; [data.md](../usage/data.md#answer-columns) covers column references and their flags; [generation-experiments.md](../patterns/generation-experiments.md) uses forced answers and answer spellings in a technique.

`documents/forced_answer.json` is the canonical multi-token case: each prompt with its answer appended, the head read before each answer token, one score per token, summed per row:

```json
{"forced": {"kind": "forward", "input": ["days.input", "days.base_answer"],
            "interventions": {"logits": {"site": "head",
                                         "pos": {"scope": {"column": "days.base_answer", "occurrence": -1, "before": true}}}}},
 "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.base_answer"},
 "seq_nll": {"kind": "op", "op": "sum", "of": "nll", "over": ["pos"]}}
```

## `… is k tokens; the read has n position(s) on that row`

```text
iia: pairs.ans row 0: ' Tuesday' is 3 tokens; the read has 1 position on that row; token i of an answer scores the read's position i, so the read carries one per token (a scope with "before": true reads the positions that predict the answer's tokens)
```

The answer is several tokens and the read is one position (`pos: -1`). Teacher-force the answer as above, or give the column an answer of one token per row. On byte-level BPE tokenizers a missing leading space is a common cause: the corpus's GPT-2 tokenizer writes `three` as `th`, `ree` and ` three` as one token, so a column holding `three` is refused where ` three` would score. A hub column without the space cannot be fixed inside a document: write its rows with the space added ([data.md](../usage/data.md#answer-columns)).

One ending names a whitespace-only token as the difference:

```text
nll: days.base_answer row 0: ' 42' is 3 tokens; the read has 2 positions on that row; …; its token 0, ' ', is whitespace only, which an answer keeps and a scope under strip does not cover: scope the column without strip
```

Some tokenizers write a number's space as its own token, which the answer keeps. A `strip` flag on the scope's column drops it from the scope, so the read is a position short; scope the column without `strip`.

## `… a position each, and the value has no pos axis`

```text
tp: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens, a position each, and the value has no pos axis (['rows', 'vocab_size'])
```

The metric's `of` was reduced over `pos` before scoring. Score first, then reduce the scores, as `seq_nll` does.

## `… stands for N ids …, which have no one logit for logit_diff`

```text
logit_diff: pairs.cf row 0: [' Monday', ' Friday'] stands for 2 ids (a set, or a class under strip or lower), which have no one logit for logit_diff; token_prob scores them together
```

A set, or a token whose `strip` or `lower` class holds several ids, has no single logit. Use `token_prob` (their summed probability), `match` or `cross_entropy`; or give `logit_diff` a column of one spelling per row.

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
nll: days.base_answer row 0: position 11 predicts 'Ġ3' in the input, which is not token 0 of '3' as it tokenizes alone (['3']); the answer is scored as it tokenizes, so the input says it the same way
```

Teacher forcing reads where the input predicts each answer token, and the input must say the answer with the same tokens it has alone. A byte-level BPE tokenizer merges a space into the token after it: on GPT-2's, the prompt `next is ` followed by `3` tokenizes as `Ġ3`, a space and `3` in one token (`Ġ` is how GPT-2's tokenizer writes a leading space), while `3` alone is `3`. Write the space on the answer (` 3`) and not at the end of the prompt.

## A probe's classes

A metric over a linear probe's scores compares the column with the probe's classes exactly:

```text
fit.ce: target names a class of probe 'probe', which is compared exactly; strip and lower compare an answer's tokens
fit: eval 'held_out', with dataset 'tr' read as 'te' (a row is the eval's dataset's): ce: target is 'label' of 'te', whose row 0 is 'maybe', not one of the classes of probe 'probe': ['no', 'yes']
```

The classes are the distinct values of the probe's `classes` column. Drop the flags, and give every scored row one of those values. [probing.md](../patterns/probing.md) covers probes.

## Pitfalls that are not refused

An answer that is one token in either spelling is scored as written: on a byte-level BPE tokenizer `a` and ` a` are two different one-token answers, and `strip` or `lower` makes the answer a class whose probabilities add up. Both run, and [symptom-index.md](symptom-index.md#answers-and-metrics) lists them.
