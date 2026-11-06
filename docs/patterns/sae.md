---
title: Sparse Autoencoders
one_liner: "How do I use a sparse autoencoder? Splice a published one in, check its L0, ablate features, find max-activating examples, train one."
tags: [patterns, sae, dictionary-learning, features, jumprelu, topk, max-activating]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/usage/bundles.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/usage/output.md, docs/errors/symptom-index.md]
sources: [clab/featurizers/sae.py, clab/featurizers/__init__.py, clab/featurizers/bundles.py, clab/ops/transforms.py]
---

# Sparse Autoencoders

An `sae` featurizer is a sparse autoencoder at one site. `encode` reads its latents. An `encode`, edits, `decode` bracket changes latents and keeps the error term (what the dictionary does not explain) exactly. `apply` replaces the value with the dictionary's reconstruction, error term dropped. Its fields are in [FORMAT.md](../../FORMAT.md#sae); the three a published file needs are in [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder).

The corpus documents load four dictionaries, each a random 16-to-32 fixture at layer 0's output (the mechanics are real, the latents mean nothing). A latent's id belongs to its file:

| document | dictionary | file |
|---|---|---|
| `documents/sae_published_jumprelu.json` | JumpReLU, laid out as Gemma Scope publishes one | `documents/artifacts/sae_gemma_scope.safetensors` |
| `documents/sae_published_topk.json` | TopK, laid out as SAELens publishes one | `documents/artifacts/sae_saelens.safetensors` |
| `documents/sae_feature_ablation.json` | a clab bundle, `relu`, centered | `documents/artifacts/sae.safetensors` |
| `documents/sae_train_ablation.json` | the one `documents/sae_train.json` trains | `documents/artifacts/fit.dictionary.safetensors` |

## Splice a published dictionary in

`sae_published_jumprelu.json` reads the latents in a clean forward, and in a second forward replaces layer 0's output at the last token with the reconstruction:

```json
"base": {
  "kind": "forward",
  "input": "prompts.input",
  "interventions": {
    "latents": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "dictionary"}]},
    "logits": {"site": "head", "pos": -1}
  }
},
"spliced": {
  "kind": "forward",
  "input": "prompts.input",
  "interventions": {
    "reconstruct": {"site": "target", "pos": -1, "ops": [{"op": "apply", "featurizer": "dictionary"}]},
    "logits": {"site": "head", "pos": -1}
  }
},
"kl": {"kind": "op", "op": "kl", "of": "spliced.logits", "against": "base.logits"}
```

- **`kl` measures what dropping the error term costs.** The splice's effect can exceed a large feature edit's. On the fixture the error term is about 1.9 times the norm of the residual stream it is cut from. Splicing moves `logit_diff` (base answer against counterfactual) by up to 0.26 on a row; zeroing the three most active latents inside a bracket moves it by at most 0.10.
- **`[encode, decode]` in place of `apply` gives `base.logits`**, since that bracket puts the error term back.
- **The direction.** As written, `kl` is KL(spliced ‖ base). SAEBench reports KL(clean ‖ spliced): to match it, put `{"op": "kl", "against": "base.logits", "reverse": true}` in `spliced`'s head entry ([ops](../usage/ops.md#metrics)).

`sae_published_topk.json` does the same with `"activation": "topk", "k": 4`: a vector's latents have at most four nonzero entries.

## Check its L0

A dictionary loaded at the wrong layer, with the wrong `centered`, or a TopK file stated as `relu`, runs and gives numbers ([featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder)). Compare its L0 (nonzero latents per token) and its splice `kl` with the numbers its release reports before using it. L0 as op steps after `base`, a comparison and a sum over `features`:

```json
"active": {"kind": "op", "op": "gt", "of": "base.latents", "with": 0.0},
"l0": {"kind": "op", "op": "sum", "of": "active", "over": ["features"]},
"mean_l0": {"kind": "op", "op": "mean", "of": "l0"}
```

- **What it counts.** `active` is 1 where a latent is above zero and 0 where it is not, so `l0` is each row and position's count. `clab table <out> l0` gives 10 on every row of `sae_published_jumprelu.json` and 4 on `sae_published_topk.json`.
- **Read every token after BOS for a dictionary-wide number** (`"pos": {"index": "1:"}`). Positions past a row's own stay missing through these steps, and `mean_l0` averages the rest: 9.94 on the JumpReLU fixture, as counting the saved latents in Python gives. Save `mean_l0` and print it with `clab table OUT mean_l0`, one row holding `value`; `clab show` takes no name.
- **Compare that mean with the release's, not one token's count.** Stated `centered: true`, the JumpReLU fixture still gives 10 at the last token of every row, while the mean after BOS moves to 10.65.
- **For a TopK release, count the tokens above its `k`.** Stated `relu`, a TopK file changes exactly the tokens with more than `k` positive pre-activations, and their L0 then exceeds `k`. After `l0`, `{"kind": "op", "op": "gt", "of": "l0", "with": 4}` and a `mean` of it give the share of tokens above `sae_published_topk.json`'s `k` of 4. It is exactly 0 under the right `activation`, and 1.0 with that file stated `relu` ([featurizers.md](../usage/featurizers.md#a-topk-file-stated-as-relu)).
- **A recomputation outside clab can differ on a tie.** A pre-activation that ties its JumpReLU threshold at float precision can count differently; check a recomputation against the mean, not token by token.

## Ablate features

`sae_feature_ablation.json` zeroes one latent per sweep point, at the last position, inside an `encode` with `select` and its `decode` ([featurizers.md](../usage/featurizers.md#selecting-features)). Because `decode` writes back only the change, a zeroed latent removes its decoder row times its value and nothing else; the error term and every other latent stay. Its saves:

- `base.latents`, the latents of the clean forward, which says which latents were active;
- `logit_diff` and `p_answer`, one entry per point, keyed `features=0` to `features=3`.

On its dictionary, latents 1 and 3 are zero at the last position of every row, so their points equal the clean forward's. The document reads no clean logits; add these to check it:

```json
"logits": {"site": "head", "pos": -1},
"clean_ld": {"kind": "op", "op": "logit_diff", "of": "base.logits", "a": "prompts.cf_answer", "b": "prompts.base_answer"}
```

The first is an entry of `base`, the second a step. Each point's change from the clean forward is one more step, zero at the points of latents 1 and 3:

```json
"delta": {"kind": "op", "op": "sub", "of": "logit_diff", "with": "clean_ld"}
```

Other edits go where the `swap` is: `{"op": "scale", "factor": 5.0}` on a selected latent amplifies it, `{"op": "add", "with": 1.0}` adds to it.

**Rank, then ablate.** `select` takes indices written in the document, never a value a step computed (`'top.ids' is not a selector`). So rank the latents in one run and write the ids you read into the next. On `sae_published_jumprelu.json`'s `base.latents`:

```json
"top": {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": "features"}
```

gives 31, 19 and 24 on every row. `{"op": "encode", "featurizer": "dictionary", "select": {"features": [19, 24, 31]}}` with its `swap` and `decode` zeroes the three together; a sweep over the ids zeroes one per point.

**A latent that never fires is the bracket's null.** Count how often each latent fires with `{"kind": "op", "op": "sum", "of": "active", "over": ["rows", "pos"]}`, `active` read at every token after BOS. On the JumpReLU fixture latents 10, 21 and 22 never do, and zeroing one of them gives the clean `logit_diff`. Recount whenever `centered` or `activation` changes: under `centered: true` the silent ones are 18, 21 and 22.

## Find a latent's max-activating examples

A latent's max-activating examples are a `top_k` step over `["rows", "pos"]` of its latents read at every token after BOS ([ops](../usage/ops.md#transforms)). With `sae_published_jumprelu.json`'s dictionary:

```json
"base": {"kind": "forward", "input": "prompts.input", "interventions": {
  "latents": {"site": "target", "pos": {"index": "1:"},
              "ops": [{"op": "encode", "featurizer": "dictionary", "select": {"features": [19, 24, 31]}}]}}},
"top": {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": ["rows", "pos"]}
```

`features` indexes the `select` list: 0, 1 and 2 are latents 19, 24 and 31. `clab table` prints each pick's row, position and token:

```text
clab table out/maxact top
features  k  row  pos  pos_token  input                              value
0         0  0    9    orrow      If today is Thursday, tomorrow is  0.04615912586450577
0         1  1    7    orrow      If today is Friday, tomorrow is    0.045574791729450226
0         2  2    7    orrow      If today is Saturday, tomorrow is  0.04504638537764549
1         0  3    4    ▁Sunday    If today is Sunday, tomorrow is    0.027981538325548172
1         1  3    8    ▁is        If today is Sunday, tomorrow is    0.021905064582824707
1         2  0    3    ▁is        If today is Thursday, tomorrow is  0.021579559892416
2         0  2    8    ▁is        If today is Saturday, tomorrow is  0.05331781134009361
2         1  0    3    ▁is        If today is Thursday, tomorrow is  0.05251988023519516
2         2  1    3    ▁is        If today is Friday, tomorrow is    0.05251988023519516
```

A position past a row's own is missing, and `top_k` ranks it below every entry that is there.

## Train a dictionary

`documents/sae_train.json` draws a 32-latent dictionary at layer 0's output (`{"kind": "sae", "latents": 32}`) and trains it on every token of the prompts after BOS. The tokens are harvested once, before the fit, and the body is the dictionary's `encode` of each minibatch's rows of the harvest, so the model runs once in all:

```json
"harvest": {"kind": "forward", "input": "prompts.input",
            "interventions": {"acts": {"site": "target", "pos": {"index": "1:"}}}},
```

and in the fit:

```json
"objective": {"dictionary.reconstruction": 1.0, "dictionary.l1": 0.01},
"steps": {
  "latents": {"kind": "op", "op": "encode", "featurizer": "fit.dictionary", "of": "harvest.acts"}
}
```

- **The objective is the dictionary's own scalars**: `reconstruction`, the mean squared error between each vector read and its decoding, and `l1`, the mean absolute latent. They are over the vectors an `encode` no `decode` closes saw, as here ([fit.md](../usage/fit.md#the-objective)).
- **After each optimizer step the decoder rows are put back to unit norm.**
- **The history.** `fit.history.dictionary.reconstruction` and `fit.history.dictionary.l1` hold one value per epoch; on the fixture, reconstruction falls about sevenfold over the 20 epochs while `l1` rises.
- **`"activation": "topk", "k": 4`** beside `latents` trains a TopK dictionary instead. `jumprelu` is refused for a drawn dictionary, since its threshold comes only from a published file.

`documents/sae_train_ablation.json` loads the saved `fit.dictionary` (run `sae_train.json` with `--out documents/artifacts` first; [bundles.md](../usage/bundles.md)) and ablates its latents as `sae_feature_ablation.json` does. Its `activation`, `centered` and `latents` come from the bundle's header.

## Pitfalls

- **`apply` is the splice, the bracket is the edit.** An effect measured through `apply` against the clean forward includes dropping the error term. Edit latents inside `encode` and `decode`, which keep it, and the difference from the clean forward is the edit's alone.
- **A latent id from one file means nothing in another.** `[19, 24, 31]` and the silent 10, 21 and 22 are the JumpReLU fixture's; `sae_feature_ablation.json` loads another dictionary.
