---
title: Featurizers
one_liner: "Declaring a featurizer and using it in an entry: the seven kinds, encode, decode and apply, nested brackets, selecting features, a gate over named axes (one unit per head), and loading a published sparse autoencoder with the fields the file cannot state."
tags: [usage, featurizers, encode, decode, apply, sae, subspace]
related: [docs/usage/fit.md, docs/usage/bundles.md, docs/usage/interventions.md, docs/patterns/trained-subspaces.md, docs/patterns/head-masking.md, docs/patterns/sae.md, docs/patterns/probing.md, docs/patterns/lora.md]
sources: [clab/featurizers/__init__.py, clab/featurizers/subspace.py, clab/featurizers/gate.py, clab/featurizers/pca.py, clab/featurizers/sae.py, clab/featurizers/linear.py, clab/featurizers/lora.py, clab/featurizers/affine.py, clab/compile/uses.py, clab/compile/__init__.py]
---

# Featurizers

A featurizer is a space or a function you declare once under the root key `featurizers`, by name, and reach from an entry's `ops` with `encode`, `decode` and `apply`. It is drawn from a seed, trained by a [fit](fit.md), or loaded from a [bundle](bundles.md); every kind's fields are in [FORMAT.md](../../FORMAT.md#featurizer-kinds).

| kind | what it is | where it comes from | ops |
|---|---|---|---|
| [`subspace`](../../FORMAT.md#subspace) | a rank-`k` subspace, the first `k` columns of a rotation | drawn from `seed` (a uniformly random subspace), trained, or a bundle | `encode`, `decode`, `apply` |
| [`gate`](../../FORMAT.md#gate) | a mask over the units of the last axis of its site's selection, or of the axes its `over` names | drawn with every unit on, trained, or a bundle | `encode`, `decode`, `apply`; over axes, `apply` |
| [`pca`](../../FORMAT.md#pca) | the top-`k` principal directions and their mean | only the bundle a `pca` op wrote | `encode`, `decode`, `apply` |
| [`sae`](../../FORMAT.md#sae) | a sparse autoencoder | drawn with `latents`, trained, a bundle, or a published file | `encode`, `decode`, `apply` |
| [`linear`](../../FORMAT.md#linear) | a probe whose coordinates are its classes | drawn at zero and trained, or a bundle | `encode` |
| [`lora`](../../FORMAT.md#lora) | a low-rank adapter | drawn as the identity and trained, or a bundle | `apply` |
| [`affine`](../../FORMAT.md#affine) | an affine map of the last axis onto itself, `h + h W + b` | drawn as the identity and trained, or a bundle | `apply` |

## The canonical entry

`documents/das_apply.json` declares a loaded rotation and swaps the counterfactual's coordinates into the base run:

```json
{
  "featurizers": {
    "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley",
            "file_path": "documents/artifacts/fit.rot.safetensors"}
  },
  "steps": {
    "counterfactual": {
      "kind": "forward",
      "input": "test.counterfactual_inputs[0]",
      "interventions": {"v_cf": {"site": "target", "pos": -1, "ops": [{"op": "encode", "featurizer": "rot"}]}}
    },
    "patched": {
      "kind": "forward",
      "input": "test.input",
      "interventions": {
        "patch": {"site": "target", "pos": -1, "ops": [
          {"op": "encode", "featurizer": "rot"},
          {"op": "swap", "with": "counterfactual.v_cf"},
          {"op": "decode", "featurizer": "rot"}
        ]}
      }
    }
  }
}
```

An entry uses a featurizer in one of three shapes:

- **`[encode f]` reads coordinates.** The entry's value is `[rows, pos, k]`, its last axis `features` (a probe's: `classes`). Nothing is written. `counterfactual.v_cf` above is one.
- **`[encode f, <edits>, decode f]` edits coordinates and writes the change back.** The ops between the pair act on the coordinates; `decode` adds the change of coordinates to the original value as a delta. So `[encode f, decode f]` with nothing between is the identity to the bit, and what the space does not hold (a subspace's complement, a sparse autoencoder's error term) passes through untouched.
- **`[apply f]` replaces the value with the featurizer's own version of it**: a subspace's projection, a gate's masked value, an adapter's adapted value, a dictionary's reconstruction without its error term. It is an edit, so it writes back. Splicing an `sae` in with `apply` moves the site's value by exactly its error term, and so the model's output, with no edit at all. A gate's `apply` with `with` interpolates toward that value instead ([below](#a-gate-over-named-axes)).

`[encode f, decode f]` leaves the logits bit-identical to a clean forward for every kind that decodes (a drawn subspace and gate, a loaded `pca`, three loaded `sae`s, one of them a published JumpReLU, at every position of the prompt), and so does an `sae` bracket holding `{"op": "scale", "factor": 1.0}`.

## As a step

`encode` and `apply` run as op steps too, over a read: an entry's value, at its one site. The step's value is in the featurizer's coordinates (`encode`) or its own version of the value (`apply`), and it counts as a use of the featurizer at the site of that read, whose width it is checked against. `documents/das.json`'s fit body encodes a harvest made before the fit this way, `{"kind": "op", "op": "encode", "featurizer": "fit.rot", "of": "counterfactual.h"}`, so the counterfactual forward runs once ([fit.md](fit.md#the-canonical-fit)). An op's value has no one site, and `decode` closes an `encode` of its own entry, so both are refused as steps:

```text
clab: steps.x.of: 'm' is an op's value, which no one site made, and a featurizer's encode acts on a value at its site: encode a read (an entry's value), or encode inside the entry that reads
clab: steps.x: decode closes an encode of its own entry, so it runs inside the entry that encodes, after the encode and the edits of its coordinates; as a step there is none to close (detokenize turns a generation's ids into text)
```

## Selecting features

`select` on the `encode` keeps some coordinates; the ops between the pair act on those alone, and `decode` scatters them back, every other coordinate changing by exactly zero. The selector is an int, a slice string or a list, as in [`pos`](../../FORMAT.md#pos), and it may be a sweep. `documents/sae_feature_ablation.json` zeroes one latent per sweep point:

```json
{"op": "encode", "featurizer": "dictionary", "select": {"features": {"sweep": [0, 1, 2, 3]}}},
{"op": "swap", "with": 0.0},
{"op": "decode", "featurizer": "dictionary"}
```

A latent that is already zero at a row's position changes nothing there, to the bit; save the plain `encode` read beside the ablation to see which those were. A pick outside the featurizer's coordinates is refused before any forward:

```text
clab: zeroed.ablate: select features 40 outside the 32 features of 'dictionary'
```

## Brackets

Each `decode` closes the most recent `encode` not yet closed. Taking `decode` out of `das_apply.json`'s `patch`, or the `encode` out of `v_cf`, is refused when the document compiles:

```text
clab: steps.patched.interventions.patch.ops[1]: the entry writes back here with the encode of 'rot' (ops[0]) still open; close it with {'op': 'decode', 'featurizer': 'rot'} after its edits
clab: steps.patched.interventions.patch.ops[1].with: 'counterfactual.v_cf' is in the site's own layout; this bracket is in the coordinates of 'rot', so an operand is encoded as it is
```

An operand between a pair must be in the same coordinates: a value read through the same encodes with the same `select`s, or a number. Compile also refuses a `decode` with no `encode` open, a `decode` of a featurizer other than the open one, and an `encode` or `apply` of a featurizer inside its own `encode`.

Pairs nest: `[encode a, encode b, <edits>, decode b, decode a]` edits `b`'s coordinates of `a`'s coordinates, so `b`'s width is `a`'s `k`, and a `b` with a larger `k` is refused before any forward. A nested pair with nothing between is the identity to the bit, as a single one is. An op that changes the layout (`readout`, a metric) may follow the closing `decode`, since by then the value is the site's again.

## A gate over named axes

A `gate`'s units are the last axis of the value it meets unless its `over` names others: the axes of its site's selection the units lie on, `layers` among them where the site picks several, written in the selection's order. `theta` is laid out on those axes, as many on each as the selection keeps. At `attention_premix` (`rows, pos, num_heads, head_dim`) over every layer, `documents/head_gates.json` declares one unit per head:

```json
"featurizers": {"heads": {"kind": "gate", "over": ["layers", "num_heads"], "temperature": {"start": 1.0, "end": 0.05}}},
"sites": {"premix": {"component": "attention_premix", "layers": ":"}}
```

On the tiny GPT-2's 5 layers of 4 heads `theta` is `[5, 4]`. The mask meets the value as an operand does under [Ops' one rule](../../FORMAT.md#ops): it broadcasts over the axes it leaves out (`head_dim`, the rows and positions), and at each layer the write is made at it takes that layer's slice, by the layer's id. `over: ["num_heads"]` at one layer is `[4]`; a `select` on `num_heads` keeps that many units, and `over` left out is the last axis, as `documents/dbm.json` has it.

`apply` with `with` writes `h·m + with·(1−m)`, the interpolation toward `with`: a value at the same selection, laid out and labelled as `swap`'s operand would be, or a number. Where a unit is on the result is `h` to the bit and where it is off `with` to the bit, so an untrained gate (every θ +1) leaves the logits of a clean forward bit-identical:

```json
"gated": {"site": "premix", "pos": {"index": "0:12"},
          "ops": [{"op": "apply", "featurizer": "fit.heads", "with": "corrupt.z"}]}
```

Written as `[sub c, apply, add c]`, the same interpolation computes `(h − c)·1 + c`, which rounds, and the null is off in the last bits. Scored, the hard mask selects rather than multiplies, so a NaN or an infinity in `with` at a unit that is on never reaches the result, and a zero keeps its sign on either side; while a fit trains the gate the soft mask interpolates. Without `with`, `apply` is `h·m`.

A gate's `apply` is elementwise: a missing element of the value stays missing and the rest are gated. At `attention_probabilities` or `attention_scores` the `key` axis is NaN past a shorter row's own keys, so on rows of different lengths every vector of a short row holds a NaN; a gate `over: ["num_heads"]` there with one head off writes each row's own keys back gated, and gives the logits of a `swap` with 0 at that head, to the bit.

A gate over axes serves `apply` alone: its units are not the last axis an `encode` makes its coordinates, so an `encode` of it is refused, and so is an `apply` inside another featurizer's `encode`. Every site it is used at selects the same indices on its axes; two that differ are refused before any forward, and `over` takes no sweep (declare one gate per layout). Its bundle's header records `over` and, for each axis but `layers` (which `layer` records), the indices kept, `kept.<axis>`; a loading document may leave `over` out, a `theta` not laid out as the header records is refused at compile, and every use is checked against the header ([bundles.md](bundles.md#what-is-checked-and-when)). The fit, the null and reading which heads were kept are [head-masking.md](../patterns/head-masking.md).

## An affine map

An `affine` is `h + h W + b` on the last axis of the value it meets: `W` `[width, width]` and `b` `[width]`, both zero as drawn, so an untrained one is the identity to the bit and an `epochs: 0` fit of it is a null control ([fit.md](fit.md#epochs-0-is-a-null-control)). It draws nothing at random, so it takes no `seed`, and its one op is `apply`, which takes no `with`. Where a `lora` is a low-rank update with no bias, an affine is the full matrix and the bias, `width² + width` numbers: a full-rank `lora` (`rank` = `alpha` = the width) is `h + h A B`, the same map without `b`.

## Drawn, trained or loaded

A declared featurizer with no `file_path` is drawn before the run's first step, from its `seed` (0 when unset) for the kinds that take one, and needs the fields its kind draws from: `k` for a subspace, `latents` for an sae, `rank` and `alpha` for a lora, `classes` for a probe. `documents/random_subspace.json` sweeps the seed to get three random rank-4 subspaces with no fit at all. A fit draws the featurizers it trains at its own start, from the fit's `seed` unless the featurizer has one; the names a fit gives them are in [fit.md](fit.md#members). With `file_path` the featurizer is loaded and fixed, and the fields the bundle's header records may be left out ([bundles.md](bundles.md)).

## Loading a published sparse autoencoder

An `sae` reads a published safetensors file with no clab header, holding `W_enc`, `W_dec`, `b_enc`, `b_dec` and, for a JumpReLU, `threshold` (SAELens and Gemma Scope lay theirs out this way). The file says nothing about how it encodes or where it was trained, so the document states three fields, which have no default for a published file. From `documents/sae_published_jumprelu.json`, with its `site` written as the release's layer inline, as the last paragraph of this section asks:

```json
"dictionary": {"kind": "sae", "file_path": "documents/artifacts/sae_gemma_scope.safetensors",
               "activation": "jumprelu", "centered": false, "site": {"component": "layer_output", "layers": 0}}
```

- `activation`: `jumprelu` when the file holds a `threshold`, else `relu`, or `topk` with `k` (and `k` only with `topk`). A `threshold` is checked both ways; nothing in a file tells `relu` from `topk`, so a TopK file stated as `relu` runs with every positive latent (17 or 18 per row on `documents/sae_published_topk.json`'s fixture, where `topk` keeps 4).
- `centered`: whether the encoder subtracts `b_dec` first. Gemma Scope's do not (`false`). Nothing checks it: a wrong value runs and shifts every latent.
- `site`: where the release says it was trained, a name from `sites` or `{"component": "layer_output", "layers": 12}` with one layer. Every use is checked against it, component and layer, before any forward.

The tensors' shapes are checked against each other and against the width of the site you use it at; `latents`, when given, against the file. A `.npz` (Gemma Scope's `params.npz`) is refused with a message saying to convert it with `numpy.load` and `safetensors.torch.save_file`.

The `site` check compares your use with your claim; it cannot check the claim. Write the release's layer inline, `{"component": ..., "layers": <int>}`, rather than the name of the site the entries use: with `"site": "target"`, changing `target`'s `layers` moves the claim with it, and the dictionary then runs at the new layer without a refusal. A dictionary at the wrong layer, or with the wrong `centered`, gives numbers; check its L0 and reconstruction against the release's before trusting them ([sae.md](../patterns/sae.md)).

## What a featurizer cannot do

- **Serve another kind's ops.** A `linear` probe only encodes; a `lora` and an `affine` only apply; compile refuses the rest by name (`'probe' is a 'linear' featurizer, which has no decode`).
- **Be drawn as a `pca`.** A `pca` featurizer exists only as the bundle a `pca` op wrote.
- **Have two widths.** Its width is the last axis of every site it is used at (inside an `encode`, that one's `k`), and one featurizer used at sites of two widths is refused before any forward. `k` is at most that width. A gate over axes has as many units as its axes keep, and every site it is used at selects the same ones.
- **Be one per layer by itself.** A featurizer used at a site whose `layers` picks several is one featurizer, applied at each; only a gate with `layers` in its `over` has units of its own at each layer. Sweep the site's `layers` for one per layer ([probing.md](../patterns/probing.md) does).
- **Be trained once loaded.** A fit refuses to train a featurizer with a `file_path`.
- **Read across sites in a generate.** A lora's `with` names an entry of the same forward, which only a `forward` step serves ([lora.md](../patterns/lora.md)).
