---
title: Symptom Index
one_liner: "Something ran and the numbers are wrong: the silent cases, from indices counted within a select list to a published SAE stated at the wrong layer, each with what it is and where it is explained."
tags: [errors, symptoms, silent-failure, index]
related: [docs/errors/index.md, docs/errors/alignment-missing.md, docs/usage/positions.md, docs/usage/output.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/generate.md, docs/usage/featurizers.md]
sources: [clab/compile/positions.py, clab/compile/interventions.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/answers.py, clab/output/table.py, clab/featurizers/sae.py]
---

# Symptom Index

Up: [docs/index.md](../index.md) — all sections.

Start here when the run finished, nothing was refused, and the numbers look wrong. For a refusal, look its message up in [index.md](index.md).

Most of these show up when you read back what the run recorded: where each read and write landed (`run.positions` in the manifest, the `pos` and `pos_token` columns of `clab table`), how many rows each value holds, and the sweep axes `clab show` lists ([output.md](../usage/output.md)).

## Positions and selections

| What you saw | What it is | Where |
|---|---|---|
| A read or write landed on the first mention of a word, not the one you meant | `occurrence` defaults to 0, the first, with occurrences ordered by where each ends; `-1` is the last and `":"` every one. Occurrences are whole words: `Sara` is not found inside `Sarah`. | [positions.md](../usage/positions.md#words-and-occurrences) |
| A teacher-forced answer was scored at an earlier mention of it in the prompt | The default `occurrence: 0` scopes the first occurrence, which is in the prompt whenever the prompt says the answer too. Write `"occurrence": -1` for the appended copy. | [generation-experiments.md](../patterns/generation-experiments.md#a-fixed-answer-token-by-token-teacher-forcing) |
| A window swap between two inputs moved different tokens than you meant | The same unscoped `pos` on two inputs pairs position i with position i. Rows that tokenize to different lengths are checked; rows of equal length are not, whatever their tokens. Scope each side to the text it means. | [below](#a-window-between-rows-of-equal-length) |
| A read over every token includes `<s>` or a chat template's markers | Every input token is a position, BOS at 0 and a template's markers included, so `{"index": ":"}` takes them all. `documents/pca_harvest.json` reads `{"index": "1:"}` to leave BOS out. | [positions.md](../usage/positions.md) |
| `pos: -1` on a chat input reads a template marker (`]` of `[/INST]`) | The rendered chat ends with whatever the template writes last, and `-1` is that token. A `turn` scope names the message's own tokens. | [data.md](../usage/data.md#chat-inputs) |
| Some rows of a prefill read are NaN and missing from `clab table` | An index with an open end (`"10:"`) picks no token on rows too short for it. A write there is refused; a read is not. | [position-not-in-row.md](position-not-in-row.md#the-writes-positions-hold-no-token-of-this-row) |
| `argmax` or `top_k` over heads gives 0 and 1 where you selected heads 3 and 1 | Along a `select`ed axis, values and indices follow the `select` list, counted within it: index 0 is the list's first pick. Map an index back through the list. | [values-and-layout.md](../concepts/values-and-layout.md#select-keeps-the-axis) |
| A query patched between two prompts moved where its head looks from | `attention_queries` and `attention_keys` are taken after RoPE, so a query read on a prompt of another length carries that prompt's position. Patch between prompts that tokenize alike. | [attention.md](../patterns/attention.md#queries-after-rope) |

## Sweeps and layers

| What you saw | What it is | Where |
|---|---|---|
| A grid of two sweeps came back as its diagonal | Two sweeps with no `label` on fields of one name are both labelled with that name, and one label is one axis, so they zip. Give each its own `label`, and check the `axes` column of `clab show`. | [sweeps.md](../usage/sweeps.md#labels-zip-or-cross) |
| A position sweep of a patch shows effects at positions the counterfactual was never read at | A sweep of the write's `pos` alone moves where the counterfactual's value lands; the read stays where it was. Sweep the read's `pos` over the same values, which pairs the two. | [activation-patching.md](../patterns/activation-patching.md#a-layer-by-position-map) |
| A patch meant for one layer at a time changed every layer at once | A `layers` slice or list on a write writes at every layer it picks, in one forward. One layer per point is `"layers": {"sweep": [...]}`. | [interventions.md](../usage/interventions.md#writes-across-layers) |
| A vector read at one layer changed every layer of a write across `layers` | An operand without a `layers` axis is the same at every layer the write is made at. Read the operand at the same `layers` to pair layer with layer, by label. | [interventions.md](../usage/interventions.md#writes-across-layers) |
| A `mean` or `sum` came back with one number per layer | A reduction removes the rows and positions by default and keeps a `layers` axis. Name `layers` in `over` to remove it. | [ops.md](../usage/ops.md#reductions-and-over) |

## Values and reductions

| What you saw | What it is | Where |
|---|---|---|
| An accuracy or mean covers fewer rows than the dataset has | A row a read beyond the prefill does not reach (a scope the generation never said, a call after the row stopped) is `alignment_missing`: NaN, and `mean`, `sum` and `min` leave it out. Count them in `run.positions`, or as the table's `alignment_missing` rows. | [alignment-missing.md](alignment-missing.md) |
| A mean over each row's entity favours rows whose entity has more tokens | A reduction over a ragged read averages every position present, so a three-token entity weighs three times a one-token one. Reduce `"over": ["pos"]` first. | [ablation.md](../patterns/ablation.md#a-mean-over-an-entitys-tokens) |
| After knocking keys out and `renormalize`, the pattern's rows do not sum to 1 | `renormalize` restores each vector's L2 norm unless told otherwise; `{"op": "renormalize", "norm": "sum"}` restores its sum. | [attention.md](../patterns/attention.md#knocking-out-keys) |
| A steered run is NaN at every dose, 0 included | `normalize` of a zero vector is NaN, and adding NaN at any dose is NaN. Check a direction's norm before normalizing it. | [steering.md](../patterns/steering.md#a-mean-difference-at-a-dose) |
| A mean-difference direction moves nothing in particular | A difference of means over the same prompts in another order is rounding noise (a norm of about 4e-9), and `normalize` makes it a unit vector pointing nowhere. Save the difference and read its norm. | [steering.md](../patterns/steering.md#a-mean-difference-at-a-dose) |

## Answers and metrics

| What you saw | What it is | Where |
|---|---|---|
| `token_prob` is higher than the probability of the token you meant | `strip` or `lower` on the column makes each answer token a class (under `strip`, ` a` and `a`; under `lower`, ` a` and ` A`), and `token_prob` sums the class. | [data.md](../usage/data.md#answer-columns) |
| A metric scores a different token than the model says | Answers are tokenized as written. On a byte-level BPE tokenizer such as GPT-2's, `a` and ` a` are two different one-token answers. Write the column as the model continues the prompt, leading space included. | [answer-does-not-line-up.md](answer-does-not-line-up.md) |
| A probe has more classes than you meant | Its classes are the distinct values of its `classes` column, compared exactly: `yes` and `Yes` are two. | [probing.md](../patterns/probing.md#scoring-classes) |

## Featurizers

| What you saw | What it is | Where |
|---|---|---|
| A step after a fit scores like an untrained featurizer | After the fit, a featurizer's bare name is the declared one as drawn at the run's start; the trained one is `<fit>.<name>`. Nothing refuses the bare name. | [fit.md](../usage/fit.md#members) |
| A published SAE's latents, reconstruction or L0 look wrong | A published file does not say where it was trained or how it encodes. Its `site`, `centered` and `activation` are the document's word, and only its shapes and a `jumprelu`'s threshold are checked in the file: a `topk` file stated as `relu` runs, with every positive latent active. Compare the measured L0 with the release's. | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder), [sae.md](../patterns/sae.md#check-its-l0) |
| Changing a site's `layers` moved a published SAE with it, unrefused | A `site` that names the site the entries use is checked against itself, so the claim moves with the site. Write the release's layer inline, `{"component": ..., "layers": <int>}`. | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| Splicing an SAE in moved the output with no edit at all | `apply` replaces the value with the reconstruction and drops the error term. An `encode` … `decode` bracket keeps it, so an edit there is the edit alone. | [sae.md](../patterns/sae.md#splice-a-published-dictionary-in) |
| An adapter or an always-on write changed the prompt and not the generated tokens | In a `generate` step a `pos` without a `frame` is the prefill only; every decode call runs without it. Write the entry at `"frame": ":"`. | [lora.md](../patterns/lora.md#apply-it-in-some-steps-and-not-others) |

## Generation

| What you saw | What it is | Where |
|---|---|---|
| Two runs of one `generate` step said different things | A step passes only the options it names to the model's `generate`; the rest come from the checkpoint's generation config, which may sample, and sampling is not seeded. Give `do_sample: false`. | [generate.md](../usage/generate.md#options) |
| `contains` never finds the stop token's text | A `detokenize`d row ends before its first EOS. The step's ids after that EOS are filler, not the row's words. | [generate.md](../usage/generate.md#rows-that-stop-early) |
| A read at a later decode call is NaN on some rows | Those rows stopped at an EOS before that call, so they are `alignment_missing` there. `min_new_tokens` holds every row to a count. | [generate.md](../usage/generate.md#rows-that-stop-early) |
| `{"frame": -1}` read every prompt token on some rows | On a row whose first generated token was EOS the last call is the prefill, and `{"frame": -1}` without an `index` is all of it. Write `{"frame": -1, "index": -1}`. | [positions.md](../usage/positions.md#generated-tokens) |
| A write and a read with the same scoped `pos` landed on different tokens | A write found online sees nothing past the token it is fed, so it takes an occurrence's end as a word's end; a read is resolved after the row stops and sees what follows. Compare both in `run.positions`. | [concepts/positions.md](../concepts/positions.md#where-a-write-lands) |

## Tables and reproducibility

| What you saw | What it is | Where |
|---|---|---|
| `clab table` prints nothing for a saved read | No row reached it: `clab show` gives its `pos` width as 0 (`[4,0,16]`), and every row in `run.positions` is `alignment_missing`. | [alignment-missing.md](alignment-missing.md#how-it-propagates) |
| `clab table out a b` prints nothing, while each alone prints rows | A table joins its values on what they share: two reads at different positions share no `pos`, and values swept over different axes (`src, dst` against `src` alone) share no point. Table them one at a time. | [output.md](../usage/output.md#clab-table), [sweeps.md](../usage/sweeps.md#pitfalls) |
| The last bits of a value differ between two runs | Batch shape: another `--batch-size`, or a trailing partial batch, is another matrix (about 1e-7 in float32); on several CPU threads, other rows sharing the batch can move a heavily padded row too. Results reproduce to the bit at a fixed batch size and row order. | [running.md](../usage/running.md#batch-size-and-reproducibility) |
| An apply document's numbers differ from its fit's eval in the last bits | A fit's body runs in batches of the fit's `batch_size`, the apply document in batches of `--batch-size`. Run the apply document at the fit's `batch_size` to compare to the bit. | [bundles.md](../usage/bundles.md#reproducing-a-fits-numbers-in-the-apply-document) |

## A window between rows of equal length

A read and a write at the same unscoped `pos` on two inputs pair index with index. Preflight refuses a pair that may hold two different tokens on rows of different lengths ([operand-does-not-fit.md](operand-does-not-fit.md#two-inputs-of-different-lengths)), but on rows that tokenize to the same length every pair runs. With `{"index": "4:7"}` on both sides:

```text
pairs.a: <s> ▁If ▁today ▁is [▁Th urs day] , ▁tom orrow ▁is
pairs.b: <s> ▁If ▁today ▁is [▁Monday , ▁so] ▁then ▁tom orrow ▁is
```

The run swaps the three pieces of ` Thursday` onto ` Monday`, a comma and ` so`. Scope each side to its own text, as `documents/entity_patch.json` does, and check `run.positions` for both entries.
