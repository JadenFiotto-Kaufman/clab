"""The `subspace` kind: a rank-`k` subspace, the first columns of a rotation a fit trains."""

from typing import Literal

import torch
from torch import Tensor

from . import APPLY, ENCODE, REPLACE, SETTLE, kind, of


@kind(saves=("basis",), trains=("theta",), header=("k", "parametrization"), drawn=("k",))
def subspace(width: int, k: int | None = None, parametrization: Literal["cayley"] = "cayley",
             seed: int | None = None, file_path: str | None = None, bundle: dict | None = None) -> dict:
    """A rank-`k` subspace: the first `k` columns of a rotation `origin @ cayley(theta)`. `origin` is a
    Haar-random orthogonal matrix drawn from `seed`, and `theta` (what a fit trains) starts at zero, so an
    untrained subspace is a uniformly random one and each seed draws a different one. Its bundle holds the
    `[width, k]` basis."""
    if bundle is not None:
        return {"kind": "subspace", "basis": bundle["basis"]}
    assert k is not None
    generator = torch.Generator().manual_seed(seed or 0)
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    f = {"kind": "subspace", "k": k, "origin": (q * r.diagonal().sign()).float(), "theta": torch.zeros(width, width)}
    subspace_settle(f)
    return f


@of(SETTLE, "subspace")
def subspace_settle(f: dict) -> None:
    skew = f["theta"] - f["theta"].T
    eye = torch.eye(skew.shape[0], dtype=skew.dtype)
    f["basis"] = (f["origin"] @ torch.linalg.solve(eye - skew, eye + skew))[:, :f["k"]]


@of(ENCODE, "subspace")
def subspace_encode(f: dict, h: Tensor) -> Tensor:
    return h @ f["basis"].to(h.dtype)


@of(APPLY, "subspace")
def subspace_apply(f: dict, h: Tensor) -> Tensor:
    """`h` projected onto the subspace."""
    basis = f["basis"].to(h.dtype)
    return h @ basis @ basis.T


@of(REPLACE, "subspace")
def subspace_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + (new - z) @ f["basis"].to(h.dtype).T
