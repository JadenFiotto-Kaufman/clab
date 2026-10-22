# Answers: multi-token, sets, normalization

A proposal for how a metric scores an answer that is several tokens, a set
of acceptable strings, or text that needs stripping or lowercasing, using
the grammar that exists. Nothing here is implemented. `piece` (PR 12) is
removed by this proposal.

## The three rules

### 1. `before` on a scope

A scope selects the tokens *of* the text it finds. With `"before": true` it
selects the positions that *predict* them: each shifted left by one, the same
width. That is the only shift anyone needs, so it is a boolean, not an
offset. A scope token with no predecessor (position 0) makes the row
missing, like any other unreachable position.

```
scope := {"column" | "text": one or a list, "occurrence"?: selector,
          "strip"?: bool, "lower"?: bool, "before"?: bool}
       | {"turn": "<role>", "occurrence"?: selector, "before"?: bool}
```

### 2. A metric aligns its column with the read

A token column tokenizes each row to *k* tokens; the read it scores must
carry *k* positions; token *i* scores against distribution *i*. Width 1 is
today's rule; any other mismatch is refused by name:

```
answer row 3: " New York" is 2 tokens; the read has 1 position
```

The metric keeps the `pos` axis. Aggregation is the reduction step that
already exists: `mean` over `pos`, plus `sum` (new) and `min` (new). No
kwargs on metrics.

### 3. Normalization lives on the column reference

Wherever a column is consumed *as text* (a scope's `column`, a metric's
answer column) it is written as the string `"days.answer"` or as the object

```json
{"column": "days.answer", "strip": true, "lower": true}
```

The same object in both places; scopes already take `strip`. The flags
define the *comparison*, they do not rewrite the data:

- On a scope, `lower` searches the input lowercased; positions come from the
  original.
- On an answer, `strip`/`lower` name an equivalence class of the vocabulary:
  every token whose text, stripped and lowercased, equals the answer's. The
  class is computed once from the vocabulary. `match` hits if the argmax is
  in the class; `token_prob` is the class's mass; `cross_entropy` minus its
  log; `token_logit` and `logit_diff` are refused when the class has more
  than one id, since a class has no logit. For a multi-token answer the
  widths come from the answer as written, and each position compares against
  the class of its token.

Spellings are a normalization problem, so they use the flags. A row's answer
may also be a list, for answers that are different words (any French city;
` Yes`, ` Sure`, ` Correct`). A list is an acceptable set; flags and list
compose, the members' classes unioned:

| metric | on a set |
|---|---|
| `match` | 1 if the argmax is any member |
| `token_prob` | the summed probability of the members (deduplicated after tokenizing) |
| `cross_entropy` | minus the log of that sum |
| `token_logit`, `logit_diff` | refused: a set has no logit; use `token_prob` |

Members must share one width, since one read aligns one span. A set of
single tokens is the case that matters.

## Examples

### A. One-token answer (unchanged)

```json
"answer": {"kind": "forward", "input": "days.input",
           "reads": {"logits": {"site": "head", "pos": -1}}},
"p":      {"kind": "op", "op": "token_prob", "of": "answer.logits", "token": "days.answer"},
"p_mean": {"kind": "op", "op": "mean", "of": "p"}
```

`p` is `[rows]`; `p_mean` a scalar. `pos: -1` is one position, the answer
one token: width 1 of rule 2.

### B. Multi-token answer, teacher forced

Row: input `"The Statue of Liberty stands in"`, answer `" New York"` (two
tokens). The same row as A. Teacher forcing appends the answer to the input
at the step, so the model has ` New` in context when it predicts ` York`:
`input` takes a list, joined per row (new).

```json
"forced":  {"kind": "forward", "input": ["days.input", "days.answer"],
            "reads": {"logits": {"site": "head",
                                 "pos": {"scope": {"column": "days.answer", "occurrence": -1, "before": true}}}}},
"nll":     {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.answer"},
"seq_nll": {"kind": "op", "op": "sum", "of": "nll", "over": "pos"},
"hit":     {"kind": "op", "op": "match", "of": "forced.logits", "expected": "days.answer"},
"exact":   {"kind": "op", "op": "min", "of": "hit", "over": "pos"},
"per_tok": {"kind": "op", "op": "mean", "of": "hit", "over": "pos"}
```

The input is `"The Statue of Liberty stands in New York"`. The scope finds
` New York` at positions 7 and 8 (`occurrence: -1`: the appended one, in
case the question mentions it earlier); `before` reads at 6 and 7 (`in`,
` New`): the distributions over ` New` and ` York`.

| value | layout | meaning |
|---|---|---|
| `nll` | `[rows, pos]` | per-token negative log-probability, labelled by the token predicted |
| `seq_nll` | `[rows]` | the answer's negative log-likelihood; `exp(-seq_nll)` is its probability |
| `hit` | `[rows, pos]` | 0/1 per token |
| `exact` | `[rows]` | 1 only if every token was the argmax |
| `per_tok` | `[rows]` | per-token accuracy |

### C. Spellings, and a set of alternatives

Spellings: the answer column holds ` positive`; the model's mass sits on
` Positive`, ` positive` and `positive` as separate ids.

```json
"p":   {"kind": "op", "op": "token_prob", "of": "answer.logits",
        "token": {"column": "days.label", "strip": true, "lower": true}},
"hit": {"kind": "op", "op": "match", "of": "answer.logits",
        "expected": {"column": "days.label", "strip": true, "lower": true}}
```

`p` is the mass on every spelling of positive; `hit` is 1 if the argmax is
any of them. "Positive versus negative" is `p` on each column and a `sub`
step; `logit_diff` on a class of several ids is refused.

Alternatives: the row's answer is `[" Paris", " Lyon", " Marseille"]` for
"name any French city". `match` hits on any member, `token_prob` sums them.

### D. Did it generate the answer

A text question, not a logits one: compare what came out with the answer.
`contains` (name open) is a text metric over a decoded generation: 1 if the
row's text contains the answer, else 0, a list answer meaning any, with
`strip`/`lower` on the column reference as elsewhere. No row is missing,
since every row generated something, so `acc` is plain generation accuracy
and the missing-row policy never enters.

```json
"gen":  {"kind": "generate", "input": "days.input", "max_new_tokens": 8,
         "reads": {"text": {"site": "head", "pos": {"frame": "1:"}, "ops": [{"op": "decode"}]}}},
"said": {"kind": "op", "op": "contains", "of": "gen.text", "text": "days.answer"},
"acc":  {"kind": "op", "op": "mean", "of": "said"}
```

Teacher forcing (B) and generation (D) answer different questions. B asks
whether each answer token is predicted given the correct prefix, in one
batched forward; under greedy decoding `exact = 1` is equivalent to "greedy
generation would produce the answer". D asks what the model produced.

### D'. How confident was it when it said it

The logits reads on a generation, with the answer scoped in the generated
text and `before`: the distributions at the calls that chose each answer
token. A row that never said it is missing.

```json
"gen":   {"kind": "generate", "input": "days.input", "max_new_tokens": 8,
          "reads": {"logits": {"site": "head",
                               "pos": {"frame": "1:", "scope": {"column": "days.answer", "before": true}}}}},
"p":     {"kind": "op", "op": "token_prob", "of": "gen.logits", "token": "days.answer"},
"nll":   {"kind": "op", "op": "cross_entropy", "of": "gen.logits", "target": "days.answer"},
"p_seq": {"kind": "op", "op": "sum", "of": "nll", "over": "pos"}
```

`p` is `[rows, pos]`, the probability the model gave each token as it chose
it; `p_seq` the sequence negative log-likelihood.

### E. Normalization at the point of use

```json
"pos":      {"scope": {"column": {"column": "days.city", "strip": true, "lower": true}, "before": true}},
"expected": {"column": "days.city", "strip": true, "lower": true}
```

A column value `"  New York "` finds `new york` in the input
case-insensitively, and tokenizes `new york` for scoring. Without the flags
both are exact. Today the rule everywhere is exact character match, and that
stays the default.

## What this removes and adds

Removed:

- `piece`.
- The one-token restriction as a special case: it is width 1 of rule 2.
- The sequence metric from the roadmap: it is example B.
- The row-filter question for rows that never said the answer, for
  generation accuracy: D has no missing rows.

Added:

- `before` on scopes.
- `input` as a list of columns and literals, joined per row.
- `lower` beside `strip` on scopes.
- The column-reference object `{"column", "strip", "lower"}` in the two
  places text is consumed.
- List answers, scored as sets with the table above.
- `sum` and `min` reductions.
- `contains`, a text metric over a decoded generation (name open).

Two things asserted rather than left open: a set's probability is the sum of
its members' (not max, not mean), and the metrics keep `pos` rather than
aggregating internally. Both follow from "one definition per metric;
aggregation is a step".
