"""Featurizer kinds: each is a function from its document fields to an instance, a dict of tensors.

A featurizer is a space a read or write happens in. Its kind's signature is
its contract: the document fields it takes, with `width` (the feature width
of the sites it is used at) supplied by the run. An instance is built where
the model is, at first use: drawn from its `seed` (else the fit's seed, else
0), or from the bundle a `file_path` names, which compile reads into the plan
and checks key by key against the document. A fit's objective may name a
featurizer's scalar as `<featurizer>.<scalar>`: a gate's `mask` is the mean of
its soft mask.
"""

from typing import Callable, Literal

SUPPLIED = ("width",)
SCALARS = {"gate": ("mask",)}

KINDS: dict[str, Callable] = {}


def kind(fn: Callable) -> Callable:
    KINDS[fn.__name__] = fn
    return fn


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
