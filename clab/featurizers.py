"""Featurizer kinds: the fields each kind takes, and what it is.

A featurizer is a space a read or write happens in. Its kind's signature is
its contract: the document fields it takes, with `width` (the feature width
of the sites it is used at) supplied by the run. An instance is built where
the model is, at first use: drawn from its `seed` (else the fit's seed, else
0), or from the bundle a `file_path` names, which compile reads into the plan
and checks key by key against the document. A fit's objective may name a
featurizer's scalar as `<featurizer>.<scalar>`: a gate's `mask` is the mean of
its soft mask.
"""

import os
from typing import Any, Callable, Literal

from safetensors import safe_open
from torch import Tensor

from . import registry

SUPPLIED = ("width",)
FORMAT = "clab/1"
SCALARS = {"gate": ("mask",)}

KINDS: dict[str, Callable] = {}
kind = registry.registers(KINDS)


@kind
def subspace(width: int, k: int, parametrization: Literal["cayley"] = "cayley", seed: int | None = None,
             file_path: str | None = None) -> dict:
    """A rank-`k` subspace: a Haar-random rotation, Cayley-parametrized, its first `k` columns the basis."""
    raise NotImplementedError("M4")


@kind
def gate(width: int, temperature: dict | None = None) -> dict:
    """A mask over units, soft (σ(θ/T)) while a fit trains it and hard (θ > 0) when scored.

    `temperature: {"start", "end"}` anneals T linearly over the fit's epochs.
    """
    raise NotImplementedError("M4")


@kind
def pca(width: int, k: int, file_path: str) -> dict:
    """The top-`k` principal directions and mean a `pca` op saved, loaded from its bundle."""
    raise NotImplementedError("M4")


@kind
def sae(width: int, file_path: str) -> dict:
    """A sparse autoencoder with its error term kept exactly, loaded from its bundle."""
    raise NotImplementedError("M4")


@kind
def linear(width: int, classes: str) -> dict:
    """A probe, `h @ W + b`; its output axis is the distinct values of the `classes` column, sorted."""
    raise NotImplementedError("M4")


def check(kind: str, fields: dict) -> None:
    """Featurizer kind `kind` exists and `fields` bind to its signature."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    registry.check(KINDS[kind], fields, SUPPLIED, repr(kind))


def header(kind: str, fields: dict, model: Any) -> dict[str, str]:
    """A bundle's header, written beside its tensors: what made it, for the featurizer that loads it to check."""
    return {"format": FORMAT, "kind": kind, "model.key": model.key, "model.revision": model.revision,
            **{k: str(v) for k, v in fields.items() if k == "k"}}


def bundle(root: str, kind: str, fields: dict, model: Any) -> dict[str, Tensor]:
    """The tensors of the bundle `fields["file_path"]` names under `root`, its header checked key by key against
    the featurizer that loads it and the plan's `model`."""
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
    return tensors
