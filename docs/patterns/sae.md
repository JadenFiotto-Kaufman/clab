---
title: Sparse Autoencoders
one_liner: "Sparse autoencoders: splice a published JumpReLU or TopK dictionary in, ablate features through encode and decode, check its L0, and train a dictionary inside a run."
tags: [patterns, sae, dictionary-learning, features, jumprelu, topk]
related: [docs/usage/featurizers.md, docs/usage/fit.md, docs/usage/bundles.md, docs/usage/ops.md, docs/usage/sweeps.md, docs/errors/symptom-index.md]
sources: [clab/featurizers/sae.py, clab/featurizers/__init__.py, clab/featurizers/bundles.py]
---

# Sparse Autoencoders

An `sae` featurizer is a sparse autoencoder at one site. `encode` reads its latents; an `encode`, edits, `decode` bracket changes latents and keeps the error term (what the dictionary does not explain) exactly; `apply` replaces the value with the dictionary's reconstruction, error term dropped. A dictionary is loaded from a published file, trained by a fit, or loaded from the bundle a fit wrote. Its fields are in [FORMAT.md](../../FORMAT.md#sae); the three a published file needs are explained in [featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder).

## Splice a published dictionary in

`documents/sae_published_jumprelu.json` loads a file laid out as Gemma Scope publishes one (a random 16-to-32 fixture: the mechanics are real, the latents mean nothing), reads its latents in a clean forward, and in a second forward replaces layer 0's output at the last token with the reconstruction:

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

`kl` measures how far the next-token distribution moves when the error term is dropped. On the fixture that term is about 1.9 times the norm of the residual stream it is cut from, so splicing moves `logit_diff` (base answer against counterfactual) by up to 0.26 on a row, where zeroing the three most active latents inside a bracket moves it by at most 0.10. Replace `apply` with `[encode, decode]` and the logits are bit-identical to `base.logits`, since that bracket puts the error term back. `documents/sae_published_topk.json` does the same with a SAELens-style file, `"activation": "topk", "k": 4`: a vector's latents have at most four nonzero entries (four on every row of the fixture).

As written, `kl` is KL(spliced ‖ base). SAEBench reports KL(clean ‖ spliced), so to match a published number the spliced logits are `against` ([ops](../usage/ops.md#metrics)); on gemma-2-2b the two directions differ by 1.6× at a KL near 0.2. On a large model, score it inside a head entry rather than as a step over two saved full-vocabulary reads ([ops](../usage/ops.md#a-step-or-an-entry)): SAEBench's direction is then `{"op": "kl", "against": "spliced.logits"}` in the head entry of a second clean forward listed after `spliced`.

## Check its L0

A dictionary loaded at the wrong layer, with the wrong `centered`, or a TopK file stated as `relu`, runs and gives numbers; the file cannot say it was misused. Compare its L0 (nonzero latents per token) and its splice `kl` with the numbers its release reports before using it. L0 as op steps after `base`:

```json
"big": {"kind": "op", "op": "scale", "of": "base.latents", "factor": 1e30},
"active": {"kind": "op", "op": "clamp", "of": "big", "max": 1.0},
"l0": {"kind": "op", "op": "sum", "of": "active", "over": ["features"]},
"mean_l0": {"kind": "op", "op": "mean", "of": "l0"}
```

Latents are never negative (every activation ends in a relu), so after the scale every latent above 1e-30 clamps to 1 and a zero stays 0; the sum over `features` counts the active ones per row and position. `clab table <out> l0` gives 10 on every row of `sae_published_jumprelu.json` and 4 on `sae_published_topk.json`. For a dictionary-wide number, read every token after BOS (`"pos": {"index": "1:"}`): positions past a row's own stay missing through these steps, and `mean_l0` averages the rest (9.94 on the JumpReLU fixture, as counting the saved latents in Python gives). Compare that mean with the release's, not one token's count: stated `centered: true`, the same fixture still gives 10 at the last token of every row, while the mean after BOS moves to 10.65 and the last token's three most active latents change. A per-token L0 recomputed outside clab can differ from clab's on a token whose pre-activation ties its JumpReLU threshold at float precision (one token in 6,096 on gemma-2-2b's layer 12): check a recomputation against the mean, not token by token.

## Ablate features

`documents/sae_feature_ablation.json` zeroes one latent per sweep point, at the last position, inside an `encode` with `select` and its `decode` (the bracket is shown in [featurizers.md](../usage/featurizers.md#selecting-features)). Because `decode` writes back only the change, a zeroed latent removes its decoder row times its value and nothing else; the error term and every other latent stay. Its saves:

- `base.latents`, the latents of the clean forward, which says which latents were active;
- `logit_diff` and `p_answer`, one entry per point, keyed `features=0` to `features=3`.

On the fixture, latents 1 and 3 are zero at the last position of every row, and their points are bit-identical to the clean forward. Other edits go in the same place: `{"op": "scale", "factor": 5.0}` on a selected latent amplifies it, `{"op": "add", "with": 1.0}` adds to it.

**Rank, then ablate.** `select` takes indices written in the document, never a value a step computed (`'top.ids' is not a selector`). So rank the latents in one run and write the ids you read into the next. On `sae_published_jumprelu.json`'s `base.latents`:

```json
"top": {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": "features"}
```

gives 31, 19 and 24 on every row, and `{"op": "encode", "featurizer": "dictionary", "select": {"features": [19, 24, 31]}}` with its `swap` and `decode` zeroes the three together; a sweep over the ids zeroes one per point.

**A latent that never fires is the bracket's null.** The `active` step above, read at every token after BOS and summed with `"over": ["rows", "pos"]`, counts how often each latent fires. On the JumpReLU fixture latents 10, 21 and 22 never do, and zeroing one of them leaves `logit_diff` unchanged to the bit. Recount whenever `centered` or `activation` changes: under `centered: true` the silent ones are 18, 21 and 22.

## Train a dictionary

`documents/sae_train.json` draws a 32-latent dictionary at layer 0's output (`{"kind": "sae", "latents": 32}`) and trains it on every token of the prompts after BOS:

```json
"objective": {"dictionary.reconstruction": 1.0, "dictionary.l1": 0.01},
"steps": {
  "harvest": {
    "kind": "forward",
    "input": "prompts.input",
    "interventions": {
      "latents": {"site": "target", "pos": {"index": "1:"}, "ops": [{"op": "encode", "featurizer": "fit.dictionary"}]}
    }
  }
}
```

The objective is the dictionary's own scalars: `reconstruction`, the mean squared error between each vector read and its decoding, and `l1`, the mean absolute latent. They are computed over the vectors the body's open `encode` read, positions past a row's own left out, so the body reads the dictionary with an entry that stops after `encode` ([fit.md](../usage/fit.md#the-objective)). After each optimizer step the decoder rows are put back to unit norm. `fit.history.dictionary.reconstruction` and `fit.history.dictionary.l1` hold one value per epoch; on the fixture, reconstruction falls about sevenfold over the 20 epochs while `l1` rises.

`"activation": "topk", "k": 4` beside `latents` trains a TopK dictionary instead; `jumprelu` is refused for a drawn dictionary, since its threshold comes only from a published file.

`documents/sae_train_ablation.json` loads the saved `fit.dictionary` (run `sae_train.json` with `--out documents/artifacts` first; [bundles.md](../usage/bundles.md)) and ablates its latents as `sae_feature_ablation.json` does. Its `activation`, `centered` and `latents` come from the bundle's header.

## Pitfalls

- **`apply` is the splice, the bracket is the edit.** An effect measured through `apply` against the clean forward includes dropping the error term. Edit latents inside `encode` and `decode`, which keep it, and the difference from the clean forward is the edit's alone.
- **State the release's layer inline** in `site`, not the name of the site you use, so the check has something to check ([featurizers.md](../usage/featurizers.md#loading-a-published-sparse-autoencoder)).
- **An `sae` objective term needs an open `encode`.** A body whose only use of the dictionary is a closed bracket has no vectors for `reconstruction` or `l1`, and the run stops.
- **One dictionary per layer.** A published file is trained at one layer, and every use is checked against it: a site whose `layers` picks any other layer is refused before any forward.
