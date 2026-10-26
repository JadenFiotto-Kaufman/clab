---
title: Symptom Index
one_liner: "Something ran and the numbers are wrong: the silent cases, from positional select lists to a published SAE at the wrong layer, each with what it is and where it is explained."
tags: [errors, symptoms, silent-failure, index]
related: [docs/errors/index.md, docs/errors/alignment-missing.md, docs/usage/positions.md, docs/usage/output.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/generate.md, docs/usage/featurizers.md]
sources: [clab/compile/positions.py, clab/compile/interventions.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/answers.py, clab/output/table.py, clab/featurizers/sae.py]
---

# Symptom Index

Up: [docs/index.md](../index.md), all sections.

Start here when the run finished, nothing was refused, and the numbers look wrong. For a refusal, look its message up in [index.md](index.md).

Most of these show up when you read back what the run recorded: where each read and write landed (`run.positions` in the manifest, the `pos` and `pos_token` columns of `clab table`), and how many rows each value holds ([output.md](../usage/output.md)).

## Positions and selections

| What you saw | What it is | Where |
|---|---|---|
| A read or write landed on the first mention of a word, not the one you meant | `occurrence` defaults to 0, the first, with occurrences ordered by where each ends; `-1` is the last and `":"` every one. Occurrences are whole words: `Sara` is not found inside `Sarah`. | [positions.md](../usage/positions.md) |
| A window swap between two inputs moved different tokens than you meant, and nothing was refused | The same unscoped `pos` on two inputs pairs position i with position i. Rows that tokenize to different lengths are checked; rows of equal length are not, whatever their tokens. Scope each side to the text it means. | [A window between rows of equal length](#a-window-between-rows-of-equal-length) |
| A read over every token includes `<s>` or a chat template's markers | Every input token is a position, BOS at 0 and a template's markers included, so `{"index": ":"}` takes them all. `documents/pca_harvest.json` reads `{"index": "1:"}` to leave BOS out. | [positions.md](../usage/positions.md) |
| `pos: -1` on a chat input reads a template marker (`]` of `[/INST]`) | The rendered chat ends with whatever the template writes last (on the corpus's checkpoint, the `]` that closes `[/INST]`), and `-1` is that token. A `turn` scope names the message's own tokens. | [data.md](../usage/data.md) |
| Some rows of a prefill read are NaN and missing from `clab table` | An index with an open end (`"10:"`) picks no token on rows too short for it. A write there is refused; a read is not. | [position-not-in-row.md](position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row) |
| `argmax` or `top_k` over heads gives 0 and 1 where you selected heads 3 and 1 | Along a `select`ed axis, values and indices are in the order of the `select` list, counted within it. | [Indices within a selection](#indices-within-a-selection) |

## Values, layers and reductions

| What you saw | What it is | Where |
|---|---|---|
| A `mean` or `sum` came back with one number per layer | A reduction removes the rows and positions by default and keeps a `layers` axis. Name `layers` in `over` to remove it. | [ops.md](../usage/ops.md) |
| A vector read at one layer changed every layer of a write across `layers` | An operand without a `layers` axis is the same at every layer the write is made at. Read the operand at the same `layers` to pair layer with layer, by label. | [interventions.md](../usage/interventions.md) |
| An accuracy or mean covers fewer rows than the dataset has | A row a read beyond the prefill does not reach (a scope the generation never said, a call after the row stopped) is `alignment_missing`: NaN, and `mean`, `sum` and `min` leave it out. Count them in `run.positions`, or as the table's `alignment_missing` rows. | [alignment-missing.md](alignment-missing.md) |
| After knocking keys out and `renormalize`, the pattern's rows do not sum to 1 | `renormalize` restores each vector's L2 norm unless told otherwise; `{"op": "renormalize", "norm": "sum"}` restores its sum. | [attention.md](../patterns/attention.md) |

## Answers and metrics

| What you saw | What it is | Where |
|---|---|---|
| `token_prob` is higher than the probability of the token you meant | `strip` or `lower` on the column makes each answer token a class (under `strip`, ` a` and `a`; under `lower`, ` a` and ` A`), and `token_prob` sums the class. | [ops.md](../usage/ops.md) |
| A metric scores a different token than the model says, on a BPE tokenizer | Answers are tokenized as written. On GPT-2 `a` and ` a` are two different one-token answers. Write the column as the model continues the prompt, leading space included. | [answer-does-not-line-up.md](answer-does-not-line-up.md) |
| A probe has more classes than you meant | Its classes are the distinct values of its `classes` column, compared exactly: `yes` and `Yes` are two. | [probing.md](../patterns/probing.md) |

## Featurizers

| What you saw | What it is | Where |
|---|---|---|
| A published SAE's latents, reconstruction or L0 look wrong, and nothing refused | A published file does not say where it was trained or how it encodes. Its `site`, `centered` and `activation` are the document's word: clab checks every use against the stated `site`, and checks in the file only its shapes and a `jumprelu`'s threshold. A `topk` file stated as `relu` runs, with an L0 of 17 per vector where 4 was meant. Take the fields from the release, and compare the measured L0 with the release's. | [featurizers.md](../usage/featurizers.md), [sae.md](../patterns/sae.md) |

## Generation

| What you saw | What it is | Where |
|---|---|---|
| Two runs of one `generate` step said different things | A step passes only the options it names to the model's `generate`; the rest come from the checkpoint's generation config, which may sample, and sampling is not seeded. Give `do_sample: false`. | [generate.md](../usage/generate.md) |
| `contains` never finds the stop token's text | A `detokenize`d row ends before its first EOS. The step's ids after that EOS are filler, not the row's words. | [generate.md](../usage/generate.md) |
| A read at a later decode call is NaN on some rows | Those rows stopped at an EOS before that call, so they are `alignment_missing` there. `min_new_tokens` holds every row to a count. | [generate.md](../usage/generate.md) |

## Tables and reproducibility

| What you saw | What it is | Where |
|---|---|---|
| `clab table` prints nothing for a saved read | No row reached it: `clab show` gives its `pos` width as 0 (`[4,0,16]`), and every row in `run.positions` is `alignment_missing`. | [alignment-missing.md](alignment-missing.md) |
| `clab table out a b` prints nothing, while each alone prints rows | A table joins its values on the axes they share; two reads at different positions share no `pos` value. Table them one at a time. | [output.md](../usage/output.md) |
| The last bits of a value differ between two runs | Batch shape: another `--batch-size`, or a trailing partial batch, is another matrix (about 1e-7 on the corpus's float32 checkpoint). Results reproduce to the bit at a fixed batch size. | [running.md](../usage/running.md) |

## Indices within a selection

A `select` list keeps the indices it names, in the order written, and the value's axis is those indices in that order. Anything that reports a place along that axis counts within the selection:

```json
{"sel": {"site": {"component": "attention_premix", "layers": 0, "select": {"num_heads": [3, 1]}}, "pos": -1}}
```

An `argmax` step `over` `num_heads` on `sel` gives 0 where head 3 is the larger of the two and 1 where head 1 is; on the unselected site the same op names heads by number, 0 to 3. `clab table` lists the selected heads inside `value` in the same order. Map an index back through the list. Two sites whose lists hold the same heads in another order are refused when one is an operand of the other ([operand-does-not-fit.md](operand-does-not-fit.md#a-select-list-is-positional)).

## A window between rows of equal length

A read and a write at the same unscoped `pos` on two inputs pair index with index. Preflight refuses a pair that may hold two different tokens on rows of different lengths, but on rows that tokenize to the same length every pair runs. With `{"index": "4:7"}` on both sides:

```text
pairs.a: <s> ▁If ▁today ▁is [▁Th urs day] , ▁tom orrow ▁is
pairs.b: <s> ▁If ▁today ▁is [▁Monday , ▁so] ▁then ▁tom orrow ▁is
```

The run swaps the three pieces of ` Thursday` onto ` Monday`, a comma and ` so`. Scope each side to its own text, as `documents/entity_patch.json` does, and check `run.positions` for both entries.

## Related

- [index.md](index.md): refusals, by message.
- [alignment-missing.md](alignment-missing.md): why a row is missing and how missingness travels.
- [output.md](../usage/output.md): reading `run.positions` and the table's columns.
