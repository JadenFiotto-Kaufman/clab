---
title: Symptom Index
one_liner: "Something ran and the numbers are wrong: the silent cases, from indices counted within a select list to a published SAE stated at the wrong layer, each with what it is and where it is explained."
tags: [errors, symptoms, silent-failure, index]
related: [docs/errors/index.md, docs/errors/alignment-missing.md, docs/usage/positions.md, docs/usage/output.md, docs/usage/interventions.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/generate.md, docs/usage/featurizers.md, docs/usage/data.md, docs/usage/bundles.md, docs/usage/running.md, docs/patterns/attribution-patching.md, docs/patterns/trained-subspaces.md]
sources: [clab/compile/positions.py, clab/compile/interventions.py, clab/ops/reductions.py, clab/ops/transforms.py, clab/ops/edits.py, clab/compile/answers.py, clab/output/table.py, clab/featurizers/sae.py, clab/steps/batches.py]
---

# Symptom Index

Up: [docs/index.md](../index.md) — all sections.

Start here when the run finished, nothing was refused, and the numbers look wrong. For a refusal, look its message up in [index.md](index.md).

Most of these show up in what the run recorded ([output.md](../usage/output.md)):

- where each read and write landed: each entry's `positions`, the entry lines of `clab show`, the `pos` and `pos_token` columns of `clab table`;
- how many rows each value holds;
- the sweep axes each result depends on, which `clab show` lists.

## Positions and selections

| What you saw | What it is | Where |
|---|---|---|
| A read or write landed on the first mention of a word, not the one you meant | `occurrence` defaults to 0, the first, with occurrences ordered by where each ends; `-1` is the last and `":"` every one. Occurrences are whole words: `Sara` is not found inside `Sarah`. | [positions.md](../usage/positions.md#words-and-occurrences) |
| A teacher-forced answer was scored at an earlier mention of it in the prompt | The default `occurrence: 0` scopes the first occurrence, which is in the prompt whenever the prompt says the answer too. Write `"occurrence": -1` for the appended copy. | [generation-experiments.md](../patterns/generation-experiments.md#a-fixed-answer-token-by-token-teacher-forcing) |
| A patch scoped to a repeated name with `"occurrence": -1` gives a plausible curve | The last occurrence is the query's copy on rows that query that name, and the context's copy on the others. Check the recorded `positions` on rows of each kind. | [positions.md](../usage/positions.md#words-and-occurrences) |
| A window swap between two inputs moved different tokens than you meant | The same unscoped `pos` on two inputs pairs position i with position i. Rows that tokenize to different lengths are checked; rows of equal length are not, whatever their tokens. Scope each side to the text it means. | [below](#a-window-between-rows-of-equal-length) |
| A read over every token includes `<s>` or a chat template's markers | Every input token is a position, BOS at 0 and a template's markers included, so `{"index": ":"}` takes them all. `documents/pca_harvest.json` reads `{"index": "1:"}` to leave BOS out. | [positions.md](../usage/positions.md#recipes) |
| `pos: -1` on a chat input reads a template marker (`]` of `[/INST]`) | The rendered chat ends with whatever the template writes last, and `-1` is that token. A `turn` scope names the message's own tokens. | [data.md](../usage/data.md#chat-inputs) |
| `argmax` or `top_k` over heads gives 0 and 1 where you selected heads 3 and 1 | Along a `select`ed axis, values and indices follow the `select` list, counted within it: index 0 is the list's first pick. Map an index back through the list. | [values-and-layout.md](../concepts/values-and-layout.md#select-keeps-the-axis) |
| A query patched between two prompts moved where its head looks from | `attention_queries` and `attention_keys` are taken after RoPE, so a query read on a prompt of another length carries that prompt's position. Patch between prompts that tokenize alike. | [attention.md](../patterns/attention.md#queries-after-rope) |

## Reads, writes and gradients

| What you saw | What it is | Where |
|---|---|---|
| A write's scores equal the clean run's exactly | An entry without an edit in its `ops` only reads, so a write whose `ops` were lost runs as a read. Check that the entry's `ops` hold its edit. | [interventions.md](../usage/interventions.md) |
| Every point of a patch equals the clean run, and the nulls pass | A later write at the same site overwrote it: entries at one site compose in the order written, so a freeze listed after the patch puts the clean value back. List the freeze first. | [interventions.md](../usage/interventions.md#the-order-ops-run-in) |
| A read changed the model's output, or a later read of the same forward | An entry whose `ops` include an edit (`add`, `sub`, `scale`, `normalize`, …) writes back. Compute on a read in an op step, or with a transform such as `mul`. | [interventions.md](../usage/interventions.md) |
| A `grad` entry is exactly zero on every row | Its `of` does not depend on the value at its site. A layer's `attention_scores` and `attention_probabilities` are made from its queries and keys, not its `attention_values`; on a parallel block (Cohere) a layer's MLP takes the layer's input, not its attention's output. Name an `of` the site feeds. | [interventions.md](../usage/interventions.md#a-gradient-at-a-site) |
| A gradient map is nonzero and plausible, and does not predict what patching does | `of` may name any differentiable entry of the forward, and a wrong one (another metric, another answer) runs and maps that entry. Check the map once against a central difference. | [attribution-patching.md](../patterns/attribution-patching.md#check-the-gradient) |
| After adding a gradient null, the metric is 0 on every row and every gradient is exactly zero | `{"op": "scale", "factor": 0}` alone at the head is a write, and listed before the metric's entry it zeroes the logits the metric reads. Write the null as a second head entry: the metric's ops with `{"op": "mul", "with": 0}` appended, which writes nothing. | [attribution-patching.md](../patterns/attribution-patching.md#check-the-gradient) |
| An attribution map misses a component that patching shows restores the answer, at the early layers of the token the counterfactual changes | Attribution patching is first order, and one patch that restores the whole answer is far outside where that holds. Confirm the components you report with real patching. | [attribution-patching.md](../patterns/attribution-patching.md#pitfalls) |

## Sweeps and layers

| What you saw | What it is | Where |
|---|---|---|
| A grid of two sweeps came back as its diagonal | Two sweeps with no `label` on fields of one name are both labelled with that name, and one label is one axis, so they zip. Give each its own `label`, and check the `axes` column of `clab show`. | [sweeps.md](../usage/sweeps.md#labels-zip-or-cross) |
| A position sweep of a patch shows effects at positions the counterfactual was never read at | A sweep of the write's `pos` alone moves where the counterfactual's value lands; the read stays where it was. Sweep the read's `pos` over the same values, which pairs the two. | [activation-patching.md](../patterns/activation-patching.md#a-layer-by-position-map) |
| A patch meant for one layer at a time changed every layer at once | A `layers` slice or list on a write writes at every layer it picks, in one forward. One layer per point is `"layers": {"sweep": [...]}`. | [interventions.md](../usage/interventions.md#writes-across-layers) |
| A vector read at one layer changed every layer of a write across `layers` | An operand without a `layers` axis is the same at every layer the write is made at. Read the operand at the same `layers` to pair layer with layer, by label. | [interventions.md](../usage/interventions.md#writes-across-layers) |
| A swept step's saved read is one tensor, not a dict keyed by point | A result is keyed by the sweep axes its own value depends on (`axes`, which `clab show` lists per result). In `documents/generate_probe.json`, `p_answer` is a dict by `frame` and `patched` one `[4, 3]` tensor. | [sweeps.md](../usage/sweeps.md#reading-swept-results) |
| A `mean` or `sum` came back with one number per layer | A reduction removes the rows and positions by default and keeps a `layers` axis. Name `layers` in `over` to remove it. | [ops.md](../usage/ops.md#reductions-and-over) |

## Values and reductions

| What you saw | What it is | Where |
|---|---|---|
| An accuracy or mean covers fewer rows than the dataset has, or some rows of a read are NaN | The read did not reach those rows, so they are `alignment_missing`: NaN, and left out of every reduction. | [alignment-missing.md](alignment-missing.md#causes) |
| A mean over each row's entity favours rows whose entity has more tokens | A reduction over a ragged read averages every position present, so a three-token entity weighs three times a one-token one. Reduce `"over": ["pos"]` first. | [ablation.md](../patterns/ablation.md#a-mean-over-an-entitys-tokens) |
| After knocking keys out and `renormalize`, the pattern's rows do not sum to 1 | `renormalize` restores each vector's L2 norm unless told otherwise; `{"op": "renormalize", "norm": "sum"}` restores its sum. | [attention.md](../patterns/attention.md#knocking-out-keys) |
| A steered run equals the unsteered one at every dose | `normalize` of a zero vector leaves it zero, and adding zero at any dose changes nothing. Check a direction's norm before normalizing it. | [steering.md](../patterns/steering.md#a-mean-difference-at-a-dose) |
| Small effects on every head of a bfloat16 sweep, early layers included | bfloat16 rounding of the model's own forward: a head with no effect in float32 can show a small nonzero mean in bfloat16. A null that equals the clean run does not rule it out. Recompute a few heads in float32. | [ops.md](../usage/ops.md#the-float32-rule) |
| A mean-difference direction moves nothing in particular | A difference of means over the same prompts in another order is rounding noise, a norm near zero, and `normalize` makes it a unit vector pointing nowhere. Save the difference and read its norm. | [steering.md](../patterns/steering.md#a-mean-difference-at-a-dose) |

## Answers and metrics

| What you saw | What it is | Where |
|---|---|---|
| `token_prob` is higher than the probability of the token you meant | `strip` or `lower` on the column makes each answer token a class (under `strip`, ` a` and `a`; under `lower`, ` a` and ` A`), and `token_prob` sums the class. | [data.md](../usage/data.md#answer-columns) |
| A metric scores a different token than the model says | Answers are tokenized as written. On a byte-level BPE tokenizer such as GPT-2's, `a` and ` a` are two different one-token answers. Write the column as the model continues the prompt, leading space included. | [data.md](../usage/data.md#answer-columns) |
| A probe has more classes than you meant | Its classes are the distinct values of its `classes` column, compared exactly: `yes` and `Yes` are two. | [probing.md](../patterns/probing.md#scoring-classes) |

## Featurizers

| What you saw | What it is | Where |
|---|---|---|
| A step after a fit scores like an untrained featurizer | After the fit, a featurizer's bare name is the declared one as drawn; the trained one is `<fit>.<name>`. Nothing refuses the bare name. | [fit.md](../usage/fit.md#members) |
| A published SAE's latents, reconstruction or L0 look wrong | A published file does not say where it was trained or how it encodes, so its `site`, `centered` and `activation` are the document's word. A wrong `centered`, a TopK file stated as `relu`, or a wrong layer runs. Compare the mean L0 over every token after BOS with the release's. | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder), [sae.md](../patterns/sae.md#check-its-l0) |
| Changing a site's `layers` moved a published SAE with it, unrefused | A `site` that names the site the entries use is checked against itself, so the claim moves with the site. Write the release's layer inline, `{"component": ..., "layers": <int>}`. | [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder) |
| A fit's history is flat at the floor for every rank and seed on a wide model | The learning rate is too large for a width × width rotation, every entry of which Adam moves by about `lr` per step. Start near 1e-3 and sweep `lr`. | [trained-subspaces.md](../patterns/trained-subspaces.md#pitfalls) |
| Splicing an SAE in moved the output with no edit at all | `apply` replaces the value with the reconstruction and drops the error term. An `encode` … `decode` bracket keeps it, so an edit there is the edit alone. | [sae.md](../patterns/sae.md#splice-a-published-dictionary-in) |
| An adapter or an always-on write changed the prompt and not the generated tokens | In a `generate` step a `pos` without a `frame` is the prefill only; every decode call runs without it. Write the entry at `"frame": ":"`. | [lora.md](../patterns/lora.md#apply-it-in-some-steps-and-not-others) |

## Generation

| What you saw | What it is | Where |
|---|---|---|
| Two runs of one `generate` step said different things | A step passes only the options it names to the model's `generate`; the rest come from the checkpoint's generation config, which may sample, and sampling is not seeded. Give `do_sample: false`. | [generate.md](../usage/generate.md#options) |
| `contains` scores 0 on a row whose text begins with the answer | On a SentencePiece tokenizer, decoding drops the leading space of a row's first generated token, so an answer written with its space is not found at token 0. Give the column `strip: true` where `contains` or a scope searches generated text. | [generate.md](../usage/generate.md#the-first-generated-tokens-leading-space) |
| `contains` never finds the stop token's text | A `detokenize`d row ends before its first EOS. The step's ids after that EOS are filler, not the row's words. | [generate.md](../usage/generate.md#rows-that-stop-early) |
| A read at a later decode call is NaN on some rows | Those rows stopped at an EOS before that call, so they are `alignment_missing` there. | [alignment-missing.md](alignment-missing.md#causes) |
| A generation with a read at the head at every call ran out of memory, or its memory grew with `max_new_tokens` | A per-call read with no ops, or whose first op waits for the whole value, is held whole until its batch's generation ends: `calls × vocab_size` values a row. Score it with an op that runs at each call (`token_prob`, `argmax`), or lower `--batch-size`. | [generate.md](../usage/generate.md#a-read-at-every-call) |
| `{"frame": -1}` read every prompt token on some rows | On a row whose first generated token was EOS the last call is the prefill, and `{"frame": -1}` without an `index` is all of it. Write `{"frame": -1, "index": -1}`. | [positions.md](../usage/positions.md#generated-tokens) |
| A write and a read with the same scoped `pos` landed on different tokens | A write found online sees nothing past the token it is fed, so it takes an occurrence's end as a word's end; a read is resolved after the row stops and sees what follows. Compare both entries' positions in `clab show`. | [concepts/positions.md](../concepts/positions.md#where-a-write-lands) |

## Tables and reproducibility

| What you saw | What it is | Where |
|---|---|---|
| `clab table out a b` prints nothing, while each alone prints rows | A table joins its values on the keys they share, and two reads of one input at different positions share no `pos`, so no row agrees on every key. Table them one at a time. | [output.md](../usage/output.md#clab-table), [sweeps.md](../usage/sweeps.md#pitfalls) |
| The last bits of a value differ between two runs | Another batch shape is another matrix: another `--batch-size`, `--padding`, `--readout` or thread count, or a trailing partial batch. A run's numbers reproduce at the settings its run record holds. | [running.md](../usage/running.md#when-two-runs-give-the-same-numbers) |
| One row's last bits moved when the dataset changed (a `take`, a `skip`, another split) | Under the default `--padding dataset` every batch is as wide as the input's longest row, so a dataset with another longest row gives every row another matrix. | [running.md](../usage/running.md#batch-size-and-padding) |
| `clab table` refuses a value the run saved, `'<x>' was not saved` | A second run into the same `--out` replaced `manifest.json`, which names only its own saves; the first run's files are still there. Give each run its own `--out`, and copy a bundle to where the loading document's `file_path` points. | [bundles.md](../usage/bundles.md#the-two-document-workflow) |
| An apply document's numbers differ from its fit's eval in the last bits | A fit's body runs in batches of the fit's `batch_size`, the apply document in batches of `--batch-size`. Run the apply document at the fit's `batch_size` for the same numbers; on a GPU a loaded bundle's can still differ in the last bits, so compare at a tolerance. | [bundles.md](../usage/bundles.md#reproducing-a-fits-numbers-in-the-apply-document) |

## A window between rows of equal length

A read and a write at the same unscoped `pos` on two inputs pair index with index. On rows of different lengths a position pairs across prompts of different lengths only where every token between it and the end it is counted from is the same on both, whatever the two tokens it pairs; preflight refuses any other ([operand-does-not-fit.md](operand-does-not-fit.md#two-inputs-of-different-lengths)). On rows that tokenize to the same length every pair runs, whatever tokens share an index. Both entries at `{"index": "4:7"}` on these two 11-token rows swap ` Monday , so` onto the three pieces of ` Thursday`:

```text
pairs.b: ['<s>', '▁If', '▁today', '▁is', '▁Monday', ',', '▁so', '▁then', '▁tom', 'orrow', '▁is']
pairs.a: ['<s>', '▁If', '▁today', '▁is', '▁Th', 'urs', 'day', ',', '▁tom', 'orrow', '▁is']
```

Scope each side to its own text, as `documents/entity_patch.json` does, and check both entries' positions in `clab show`.
