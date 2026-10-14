"""Featurizer kinds: the fields each kind takes, what an instance is, and how a value moves in and out of its space.

A featurizer is a space a read or write happens in. Its kind's signature is its contract: the document fields
it takes, with `width` (the feature width of the sites it is used at) and `bundle` (the tensors its `file_path`
names) supplied by the run, and for a probe `labels` (its class list). An instance is a dict of tensors, never
an `nn.Module`, made where the model is: a declared featurizer before the run's first step, one a fit trains
at the fit's start. It is drawn from its `seed` (else the seed of the fit that trains it, else 0), or loaded
from its bundle, which compile reads into the plan and checks key by key against the document. Its coordinates of a value `h` are `encode(f, h)`; an op in its space runs on them, and the change is
written back as a delta (`replace`), so an op that changes nothing changes nothing to the bit, and a sparse
autoencoder's error term is kept exactly. A fit trains an instance's `TRAINABLE` tensors; its objective may
name a featurizer's scalar as `<featurizer>.<scalar>` (a gate's `mask` is the mean of its soft mask).

A bundle is a safetensors file: an instance's tensors, and a header saying what made them — the kind and its
fields, the model, the site it was used at (`component`, `layer`), its `width` and the document. A document
that loads one is checked against it key by key: the kind, its fields and the model when the plan compiles,
the site at every use, and `width` against the model's size before any forward.
"""

import inspect
import os
from typing import Any, Callable, Literal, TypedDict

import torch
from safetensors import safe_open
from torch import Tensor

from . import registry

SUPPLIED = ("width", "bundle", "labels")
FORMAT = "clab/1"
SCALARS = {"gate": ("mask",)}
TRAINABLE = {"subspace": ("theta",), "gate": ("theta",), "linear": ("W", "b")}
# The axis a kind's coordinates are on, where it is not `features`: a probe's are its classes.
AXIS = {"linear": "classes"}
# The tensor whose first axis is the width, per kind.
WIDE = {"subspace": "origin", "gate": "theta", "pca": "basis", "sae": "W_enc", "linear": "W"}

KINDS: dict[str, Callable] = {}
kind = registry.registers(KINDS)


class Temperature(TypedDict):
    start: float
    end: float


@kind
def subspace(width: int, k: int, parametrization: Literal["cayley"] = "cayley", seed: int | None = None,
             file_path: str | None = None, bundle: dict | None = None) -> dict:
    """A rank-`k` subspace: the first `k` columns of a rotation `origin @ cayley(theta)`. `origin` is a
    Haar-random orthogonal matrix drawn from `seed`, and `theta` (what a fit trains) starts at zero, so an
    untrained subspace is a uniformly random one and each seed draws a different one."""
    if bundle is not None:
        return {"kind": "subspace", "k": k, "origin": bundle["origin"].clone(), "theta": bundle["theta"].clone()}
    generator = torch.Generator().manual_seed(seed or 0)
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    return {"kind": "subspace", "k": k, "origin": (q * r.diagonal().sign()).float(),
            "theta": torch.zeros(width, width)}


@kind
def gate(width: int, temperature: Temperature | None = None) -> dict:
    """A mask over units, `theta` starting at +1 on every one: soft (σ(θ/T)) while a fit trains it, hard
    (θ > 0) whenever it is scored. `temperature: {"start", "end"}` sets T, linearly over the fit's epochs
    (default 1)."""
    return {"kind": "gate", "theta": torch.ones(width), "temperature": temperature,
            "T": temperature["start"] if temperature else 1.0, "training": False}


@kind
def pca(width: int, k: int, file_path: str, bundle: dict | None = None) -> dict:
    """The top-`k` principal directions (`basis`) and the `mean` a `pca` op saved, loaded from its bundle; its
    coordinates are centred on the mean."""
    assert bundle is not None
    return {"kind": "pca", "basis": bundle["basis"], "mean": bundle["mean"]}


@kind
def sae(width: int, file_path: str, bundle: dict | None = None) -> dict:
    """A sparse autoencoder (`W_enc`, `b_enc`, `W_dec`, `b_dec`), loaded from its bundle; its coordinates are
    the latents `relu((h - b_dec) @ W_enc + b_enc)`, and a write keeps its error term exactly."""
    assert bundle is not None
    return {"kind": "sae", **{name: bundle[name] for name in ("W_enc", "b_enc", "W_dec", "b_dec")}}


@kind
def linear(width: int, classes: str, labels: list) -> dict:
    """A probe, `h @ W + b` (both starting at zero), whose coordinates are its classes: the distinct values of
    the `classes` column, sorted. A metric scores them against a column of those values, as it scores logits
    against a column of tokens. It has no way back into the value, so it is read with `encode` and never
    written through."""
    return {"kind": "linear", "classes": labels, "W": torch.zeros(width, len(labels)), "b": torch.zeros(len(labels))}


def check(kind: str, fields: dict) -> None:
    """Featurizer kind `kind` exists and `fields` bind to its signature."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    registry.check(KINDS[kind], fields, SUPPLIED, repr(kind))


def make(kind: str, fields: dict, width: int, bundle: dict | None, seed: int | None, labels: list | None) -> dict:
    """An instance of `kind` from a document's `fields`, its seed defaulting to `seed`."""
    supplied = {"width": width, "bundle": bundle, "labels": labels}
    taken = inspect.signature(KINDS[kind]).parameters
    given = {**fields, "seed": seed if fields.get("seed") is None else fields["seed"]} if "seed" in taken else fields
    return KINDS[kind](**given, **{k: v for k, v in supplied.items() if k in taken})


# -- a value in and out of a featurizer's space ---------------------------------------------------------


def basis(f: dict) -> Tensor:
    """A subspace's or a pca's `[width, k]` basis."""
    if f["kind"] == "pca":
        return f["basis"]
    skew = f["theta"] - f["theta"].T
    eye = torch.eye(skew.shape[0], dtype=skew.dtype)
    return (f["origin"] @ torch.linalg.solve(eye - skew, eye + skew))[:, :f["k"]]


def mask(f: dict) -> Tensor:
    """A gate's mask: soft while a fit trains it, hard when scored."""
    return torch.sigmoid(f["theta"] / f["T"]) if f["training"] else (f["theta"] > 0).to(f["theta"].dtype)


def encode(f: dict, h: Tensor) -> Tensor:
    """`h`'s coordinates in `f`'s space, along its last axis."""
    if f["kind"] == "subspace":
        return h @ basis(f).to(h.dtype)
    if f["kind"] == "pca":
        return (h - f["mean"].to(h.dtype)) @ f["basis"].to(h.dtype)
    if f["kind"] == "gate":
        return h
    if f["kind"] == "sae":
        return torch.relu((h - f["b_dec"].to(h.dtype)) @ f["W_enc"].to(h.dtype) + f["b_enc"].to(h.dtype))
    return h @ f["W"].to(h.dtype) + f["b"].to(h.dtype)


def replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    """`h` with its coordinates `z` changed to `new`, as a delta on `h`: `new == z` gives `h` to the bit."""
    if f["kind"] in ("subspace", "pca"):
        return h + (new - z) @ basis(f).to(h.dtype).T
    if f["kind"] == "gate":
        return h + mask(f).to(h.dtype) * (new - h)
    if f["kind"] == "sae":
        return h + (new - z) @ f["W_dec"].to(h.dtype)
    raise ValueError("a probe's space has no way back into the value")


def within(f: dict, h: Tensor, fn: Callable[[Tensor], Tensor], features: Tensor | None) -> Tensor:
    """`h` after `fn` ran on its coordinates in `f`'s space, on the `features` it keeps (every one when None),
    and the change was written back."""
    z = encode(f, h)
    new = fn(z if features is None else z.index_select(-1, features)).to(z.dtype)
    return replace(f, h, z, new if features is None else z.index_copy(-1, features, new))


def size(f: dict) -> int:
    """How many coordinates `f`'s space has."""
    return encode(f, torch.zeros(1, f[WIDE[f["kind"]]].shape[0])).shape[-1]


def scalar(f: dict, name: str) -> Tensor:
    """A featurizer's scalar a fit's objective may name: a gate's `mask`, the mean of its soft mask."""
    assert name == "mask"
    return torch.sigmoid(f["theta"] / f["T"]).mean()


def anneal(f: dict, epoch: int, epochs: int) -> None:
    """A gate's temperature at `epoch` of `epochs`: linear from its start at the first to its end at the last."""
    if f["kind"] == "gate" and f["temperature"]:
        t = f["temperature"]
        f["T"] = t["start"] + (t["end"] - t["start"]) * (epoch / (epochs - 1) if epochs > 1 else 0.0)


def parameters(f: dict) -> list[Tensor]:
    """The tensors a fit trains."""
    return [f[name] for name in TRAINABLE[f["kind"]]]


def tensors(f: dict) -> dict[str, Tensor]:
    """An instance's tensors: what its bundle holds."""
    return {k: v.detach() for k, v in f.items() if isinstance(v, Tensor)}


def width(kind: str, tensors: dict[str, Tensor]) -> int:
    return tensors[WIDE[kind]].shape[0]


# -- bundles ---------------------------------------------------------------------------------------------

# The header keys a featurizer's own fields give, where its kind takes them.
FIELDS = ("k", "parametrization")


def header(kind: str, fields: dict, model: Any, site: Any = None, width: int | None = None,
           document: str | None = None) -> dict[str, str]:
    """A bundle's header: the kind and its fields (with their defaults), the model, and, when written, the site
    it was used at, its width and the document that wrote it."""
    params = inspect.signature(KINDS[kind]).parameters
    own = {k: str(fields.get(k, params[k].default)) for k in FIELDS if k in params}
    found = {"format": FORMAT, "kind": kind, **own, "model.key": model.key, "model.revision": model.revision,
             "model.dtype": model.dtype}
    if site is not None:
        found.update({"component": site.component, "layer": str(site.layers), "width": str(width),
                      "document": str(document)})
    return found


def bundle(root: str, kind: str, fields: dict, model: Any) -> tuple[dict[str, Tensor], dict[str, str]]:
    """The tensors and header of the bundle `fields["file_path"]` names under `root`, the header checked key by
    key against the featurizer that loads it and the plan's `model`."""
    file = os.path.join(root, fields["file_path"])
    if not os.path.exists(file):
        raise ValueError(f"{fields['file_path']} does not exist under the corpus root; run the document that "
                         "writes it first")
    with safe_open(file, "pt") as f:
        found = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    for key, value in header(kind, fields, model).items():
        if found.get(key) != value:
            raise ValueError(f"{fields['file_path']}: header {key!r} is {found.get(key)!r}; this document "
                             f"expects {value!r}")
    return tensors, found
