---
title: Featurizers
one_liner: "How do I use a learned or loaded space in an entry? Declaring each kind, encode, decode and apply, selecting features, its tensors as values, and a published SAE."
tags: [usage, featurizers, encode, decode, apply, sae, subspace, gate, affine, constant]
related: [docs/usage/fit.md, docs/usage/bundles.md, docs/usage/interventions.md, docs/usage/ops.md, docs/patterns/trained-subspaces.md, docs/patterns/head-masking.md, docs/patterns/sae.md, docs/patterns/probing.md, docs/patterns/lora.md, docs/patterns/logit-lens.md, docs/patterns/ablation.md]
sources: [clab/featurizers/__init__.py, clab/featurizers/subspace.py, clab/featurizers/gate.py, clab/featurizers/pca.py, clab/featurizers/sae.py, clab/featurizers/linear.py, clab/featurizers/lora.py, clab/featurizers/affine.py, clab/featurizers/constant.py, clab/ops/edits.py, clab/compile/uses.py, clab/compile/__init__.py, clab/plan.py]
---

# Featurizers

A featurizer is a space or a function you declare once under the root key `featurizers`, by name, and reach from an entry's `ops` with `encode`, `decode` and `apply`. It is drawn, trained by a [fit](fit.md), or loaded from a [bundle](bundles.md). Every kind's fields and ops are in [FORMAT.md](../../FORMAT.md#featurizer-kinds); this table says which task picks each:

| kind | pick it to | the pattern |
|---|---|---|
| [`subspace`](../../FORMAT.md#subspace) | swap or edit a rank-`k` share of a value: an interchange in a learned or random subspace | [trained subspaces](../patterns/trained-subspaces.md#das) |
| [`gate`](../../FORMAT.md#gate) | learn which units of a value, or which heads of every layer, carry a behavior | [DBM](../patterns/trained-subspaces.md#dbm), [head masking](../patterns/head-masking.md) |
| [`pca`](../../FORMAT.md#pca) | edit a value along the top principal directions of a harvest, a control of matched rank | [trained subspaces](../patterns/trained-subspaces.md#controls-of-matched-rank) |
| [`sae`](../../FORMAT.md#sae) | read or edit a sparse dictionary's latents, published or trained | [sparse autoencoders](../patterns/sae.md) |
| [`linear`](../../FORMAT.md#linear) | ask whether a label column is linearly decodable from a value | [probing](../patterns/probing.md) |
| [`lora`](../../FORMAT.md#lora) | learn a low-rank weight update at one site and turn it on in chosen steps | [LoRA](../patterns/lora.md) |
| [`affine`](../../FORMAT.md#affine) | learn a full linear map of a site onto itself, one per layer | [tuned lens](../patterns/logit-lens.md#a-tuned-lens) |
| [`constant`](../../FORMAT.md#constant) | replace a value by one vector, set or learned | [optimal constant](../patterns/ablation.md#the-optimal-constant) |

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
- **`[encode f, <edits>, decode f]` edits coordinates and writes the change back.** The ops between the pair act on the coordinates. `decode` adds the change of coordinates to the original value as a delta, so `[encode f, decode f]` with nothing between returns the value exactly. What the space does not hold (a subspace's complement, a sparse autoencoder's error term) passes through untouched.
- **`[apply f]` replaces the value with the featurizer's own version of it.** That is a subspace's or a `pca`'s projection, a gate's masked value, or a dictionary's reconstruction without its error term. For an adapter it is the adapted value, for an affine map `h + h W + b`, and for a constant its vector. It is an edit, so it writes back. Splicing an `sae` in with `apply` moves the site's value by exactly its error term, and so the model's output, with no edit at all.

## As a step

`encode` and `apply` run as op steps too, over a read: an entry's value, at its one site. The step's value is in the featurizer's coordinates (`encode`) or its own version of the value (`apply`). It counts as a use of the featurizer at the site of that read, whose width it is checked against. `documents/das.json`'s fit body encodes a harvest made before the fit this way, `{"kind": "op", "op": "encode", "featurizer": "fit.rot", "of": "counterfactual.h"}`, so the counterfactual forward runs once ([fit.md](fit.md#the-canonical-fit)).

An op's value has no one site, and `decode` closes an `encode` of its own entry, so both are refused as steps:

```text
clab: steps.x.of: 'm' is an op's value, which no one site made, and a featurizer's encode acts on a value at its site: encode a read (an entry's value), or encode inside the entry that reads
clab: steps.x: decode closes an encode of its own entry, so it runs inside the entry that encodes, after the encode and the edits of its coordinates; as a step there is none to close (detokenize turns a generation's ids into text)
```

## The dtype a featurizer computes in

A featurizer's tensors meet the value in the value's dtype, on its device, with autocast off. On a model loaded `"dtype": "bfloat16"`:

- **`encode` runs in bfloat16**, and its coordinates are bfloat16, in an entry or as a step.
- **A bracket's `encode` and `decode` run in bfloat16**; the edits between them compute in float32 and round once to the coordinates' dtype.
- **`apply` is an edit**: it computes in float32, the value and the featurizer's tensors upcast, and rounds once to bfloat16.
- **A metric or reduction over coordinates computes in float32**, as every one does ([ops.md](ops.md#the-float32-rule)).

The manifest's `results.<name>.dtype` records each value's ([values and layout](../concepts/values-and-layout.md#dtypes)).

## Selecting features

`select` on the `encode` keeps some coordinates. The ops between the pair act on those alone, and `decode` scatters them back, every other coordinate changing by exactly zero. The selector is an int, a slice string or a list, as in [`pos`](../../FORMAT.md#pos), and it may be a sweep. `documents/sae_feature_ablation.json` zeroes one latent per sweep point:

```json
{"op": "encode", "featurizer": "dictionary", "select": {"features": {"sweep": [0, 1, 2, 3]}}},
{"op": "swap", "with": 0.0},
{"op": "decode", "featurizer": "dictionary"}
```

A latent that is already zero at a row's position changes nothing there; save the plain `encode` read beside the ablation to see which those were. A pick outside the featurizer's coordinates is refused before any forward:

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

Pairs nest: `[encode a, encode b, <edits>, decode b, decode a]` edits `b`'s coordinates of `a`'s coordinates. So `b`'s width is `a`'s `k`, and a `b` with a larger `k` is refused before any forward. A nested pair with nothing between returns the value exactly, as a single one does. An op that changes the layout (`readout`, a metric) may follow the closing `decode`, since by then the value is the site's again.

## A gate over named axes

A `gate`'s units are the last axis of the value it meets, as in `documents/dbm.json`, unless its `over` names other axes of its site's selection. Training a head gate and reading the heads it kept is [head-masking.md](../patterns/head-masking.md). `documents/head_gates.json` declares one unit per head of every layer at `attention_premix` (`rows, pos, num_heads, head_dim`):

```json
"featurizers": {"heads": {"kind": "gate", "over": ["layers", "num_heads"], "temperature": {"start": 1.0, "end": 0.05}}},
"sites": {"premix": {"component": "attention_premix", "layers": ":"}}
```

- **`theta` is laid out on the `over` axes**, written in the selection's order, `layers` among them where the site picks several. It holds one unit per index the selection keeps on each: `[5, 4]` on the tiny GPT-2's 5 layers of 4 heads. `over: ["num_heads"]` at one layer is `[4]`, and a `select` on `num_heads` keeps that many units.
- **The mask meets the value as an operand does** under [Ops' one rule](../../FORMAT.md#ops). It broadcasts over the axes it leaves out (`head_dim`, the rows and positions), and at each layer a write takes that layer's slice, by the layer's id.
- **`apply` with `with` interpolates**: `h·m + with·(1−m)`, toward a value at the same selection, laid out and labelled as `swap`'s operand would be, or a number. It is exact at either end of the mask: `h` where a unit is on, `with` where it is off. Without `with`, `apply` is `h·m`. The nulls this gives are [head masking's](../patterns/head-masking.md#the-null-and-its-mirror).
- **Scored, the hard mask selects rather than multiplies.** A NaN or an infinity in `with` at a unit that is on never reaches the result, and a zero keeps its sign on either side. While a fit trains the gate, the soft mask interpolates.
- **`apply` is elementwise.** A missing element of the value stays missing and the rest are gated, so a gate `over: ["num_heads"]` at `attention_probabilities` knocks a head out on rows of different lengths as a `swap` with 0 does.
- **It serves `apply` alone.** Its units are not the last axis an `encode` makes its coordinates, so an `encode` of it is refused, and so is an `apply` of it inside another featurizer's `encode`.
- **Every use selects the same indices on its axes.** Two sites that differ are refused before any forward. `over` takes no sweep: declare one gate per layout.
- **Its bundle's header records `over` and the indices kept**, and a loading document may leave `over` out ([bundles.md](bundles.md#what-is-checked-and-when)).

## An affine map

An `affine` is `h + h W + b` on the last axis of the value it meets, a full linear map of the site onto itself:

- **It is drawn as the identity.** `W` `[width, width]` and `b` `[width]` are both zero, so an untrained one returns the value exactly ([fit.md](fit.md#epochs-0-is-a-null-control)). It takes no `seed`, and its one op is `apply`, which takes no `with`. Where a `lora` is a low-rank update with no bias, an affine is the full matrix and the bias, `width² + width` numbers.
- **`over: ["layers"]` holds one map per layer** of its site's selection (`{"kind": "affine", "over": ["layers"]}` in `documents/tuned_lens.json`), `W` `[layers, width, width]` and `b` `[layers, width]`, its width still the site's last axis. At each layer `apply` takes that layer's map by the layer's id, so every use selects the same layers, in any order. `over` names `layers` alone (`over: ['num_heads']: an 'affine' is over ['layers'] or nothing, …`), and the site must pick several layers.
- **Inside an entry it writes**, and the layers after it see the mapped value. To read a layer through it and leave the forward as it is, make it an `apply` step over a read.

Its bundle's header records `over` and the layers; a loading document may leave `over` out and is refused at a layer the header does not name ([bundles.md](bundles.md#what-is-checked-and-when)). Training every layer's map in one fit is the [tuned lens](../patterns/logit-lens.md#a-tuned-lens).

## Drawn, trained or loaded

A declared featurizer with no `file_path` is drawn before the run's first step, from the fields its kind draws from:

| kind | drawn from | takes a `seed` |
|---|---|---|
| `subspace` | `k`: a uniformly random rank-`k` subspace | yes |
| `gate` | nothing: every unit on | no |
| `pca` | never drawn: only the bundle a `pca` op wrote | no |
| `sae` | `latents` | yes |
| `linear` | `classes`, a label column; `W` and `b` zero | no |
| `lora` | `rank` and `alpha`; `B` zero, so the identity | yes |
| `affine` | nothing: the identity | no |
| `constant` | `init`: a number, or the name of a value one vector wide (a `mean`) | no |

- **The seed** is the featurizer's own `seed`, 0 when unset. `documents/random_subspace.json` sweeps it to get three random rank-4 subspaces with no fit.
- **A fit draws the featurizers it trains at its own start**, from the fit's `seed` unless the featurizer has one. The names a fit gives them are in [fit.md](fit.md#members).
- **A constant whose `init` names a value is drawn once the step that makes it has run.** That step must come before the constant's first use and before the fit that trains it.
- **That value is one vector laid out on the site's last axis, in the use's coordinates.** Compile refuses by name:
  - an `init` with rows, a `layers` axis or another axis;
  - one in another featurizer's coordinates, or on a sweep axis;
  - one a step makes at or after the constant's first use.

  One that keeps other indices of the last axis than the use does is refused before any forward, once the model's sizes resolve them.
- **With `file_path` the featurizer is loaded and fixed.** The fields its bundle's header records may be left out ([bundles.md](bundles.md#file_path)).

## Its tensors as values

A featurizer's tensors, those its bundle holds, are values by name: `<featurizer>.<tensor>` the declared one as drawn or loaded, `<fit>.<featurizer>.<tensor>` the one a fit trained ([fit.md](fit.md#members)). Name one where an op needs the featurizer's own numbers: a probe's direction to project out, a dictionary's decoder row to add, its decoder bias as a constant's `init`.

| kind | its tensors, laid out |
|---|---|
| `linear` | `W` `[hidden_size, classes]`, `b` `[classes]` |
| `subspace` | `basis` `[hidden_size, features]` |
| `sae` | `W_enc` `[hidden_size, features]`, `W_dec` `[features, hidden_size]`, `b_enc` `[features]`, `b_dec` `[hidden_size]`, and `threshold` `[features]` where a published file holds one |
| `gate` | `theta` `[hidden_size]`; over axes, on them (`[layers, num_heads]`) |
| `pca` | `basis` `[hidden_size, features]`, `mean` `[hidden_size]` |
| `lora` | `A` `[hidden_size, rank]`, `B` `[rank, hidden_size]` |
| `affine` | `b` `[hidden_size]`, over layers `[layers, hidden_size]`; `W` is refused, below |
| `constant` | `vector` `[hidden_size]` |

- **`hidden_size` here is the last axis of the first site a step before the name uses the featurizer at**, with the indices that site's `select` keeps: `intermediate_size` at `mlp_neurons`, and inside an `encode`, that encode's `features`. A name before any step that encodes or applies the featurizer is refused by name (`'probe.W' is a tensor of 'probe', laid out on the last axis of the site it is used at, …`).
- **A tensor whose last axis is the kind's coordinates is in its coordinates**, as an `encode`'s value is: a metric or `top_k` over `fit.probe.b` scores the probe's classes. `select` a column and reduce over that axis for a vector in the site's own layout.
- **Each is a copy, on the model's device, in the featurizer's dtype, with no gradient.** A trained one is the best epoch's, as its bundle saves it, one per point of a swept fit. Inside the fit that trains it, its tensors are no value yet, and the body naming one is refused.
- **A name the kind does not hold is refused, listing those it does** (`'fit.probe.U' names no tensor of 'fit.probe', a linear, whose tensors are W, b`). So is an `affine`'s `W`, `[hidden_size, hidden_size]`, two axes under one name that no op tells apart: save the fit's affine whole, as its bundle.
- **A `load` brings no featurizer's tensor.** Save the trained featurizer, load its bundle with `file_path`, and name its tensor there.

## Loading a published sparse autoencoder

An `sae` reads a published safetensors file with no clab header, holding `W_enc`, `W_dec`, `b_enc`, `b_dec` and, for a JumpReLU, `threshold` (SAELens and Gemma Scope lay theirs out this way). The file says nothing about how it encodes or where it was trained, so the document states three fields, which have no default for a published file. From `documents/sae_published_jumprelu.json`:

```json
"dictionary": {"kind": "sae", "file_path": "documents/artifacts/sae_gemma_scope.safetensors",
               "activation": "jumprelu", "centered": false, "site": {"component": "layer_output", "layers": 0}}
```

- `activation`: `jumprelu` when the file holds a `threshold`, else `relu`, or `topk` with `k` (and `k` only with `topk`).
- `centered`: whether the encoder subtracts `b_dec` first. A release whose encoder "subtracts the decoder bias before encoding" is `true`, `(h - b_dec) @ W_enc + b_enc`. `false` encodes `h @ W_enc + b_enc`, as Gemma Scope's do.
- `site`: where the release says it was trained, `{"component": "layer_output", "layers": 12}` with one layer, or a name from `sites`. Every use is checked against it, component and layer, before any forward.

Refused by name:

- a field left out (`... is a published sae, which says neither how it encodes nor where it was trained`);
- a wrong `activation` on a file with a `threshold` (`... holds a JumpReLU threshold; its activation is 'jumprelu', not 'relu'`), and `jumprelu` on one without;
- tensors whose shapes disagree with each other or with the site's width, and `latents` other than the file's;
- a use at another component or layer than `site`;
- a `.npz` (`... is not a safetensors file, which a file_path is read as`), which `numpy.load` and `safetensors.torch.save_file` convert.

Run without a refusal:

- a wrong `centered`, which shifts every latent;
- a TopK file stated as `relu`, which keeps every positive latent (17 or 18 per row on `documents/sae_published_topk.json`'s fixture, where `topk` keeps 4);
- a wrong layer written in `site`;
- a `site` written as the name of the entries' site.

The `site` check compares your use with your claim; it cannot check the claim. Write the release's layer inline, `{"component": ..., "layers": <int>}`. `documents/sae_published_topk.json` names its site by the entries' site, `"site": "target"`, which moves with `target`'s `layers`; the layer written inline, as `sae_published_jumprelu.json` writes it, does not. Compare the mean L0 over every token after BOS with the release's before trusting the numbers ([sae.md](../patterns/sae.md#check-its-l0)).

## What a featurizer cannot do

- **Serve another kind's ops.** A `linear` probe only encodes; a `lora`, an `affine` and a `constant` only apply. Compile refuses the rest by name (`'probe' is a 'linear' featurizer, which has no decode`).
- **Be drawn as a `pca`.** A `pca` featurizer exists only as the bundle a `pca` op wrote.
- **Have two widths.** Its width is the last axis of every site it is used at (inside an `encode`, that one's `k`), and one featurizer used at sites of two widths is refused before any forward. `k` is at most that width. A gate over axes has as many units as its axes keep.
- **Be one per layer by itself.** A featurizer used at a site whose `layers` picks several is one featurizer, applied at each. Only a gate or an affine with `layers` in its `over` has its own units or map at each layer. Sweep the site's `layers` for one per layer ([probing.md](../patterns/probing.md) does).
- **Be trained once loaded.** A fit refuses to train a featurizer with a `file_path`.
- **Read across sites in a generate.** A lora's `with` names an entry of the same forward, which only a `forward` step serves ([lora.md](../patterns/lora.md#across-sites-with)).
