"""Reductions: ops that remove axes. `mean`, `sum` and `min` compute in float32 whatever the value's dtype
(`ops.ABOUT`); `pca` works in double and returns the value's dtype."""

import json

import torch
from torch import Tensor

from clab.format.address import EXAMPLE_AXES
from clab.ops import op

# What a reduction without `over` removes: the rows and the positions. A `layers` axis says which layer each
# value is, as a feature axis says which unit, so it stays unless `over` names it.
DEFAULT = ("rows", "pos")


def reduced(dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """Removes the axes `over` names, by default the rows and positions: a `layers` axis stays unless `over`
    names it. One that would remove no axis of the value is refused."""
    over = args.get("over")
    for axis in over or []:
        if axis not in dims:
            raise ValueError(f"a reduction cannot remove {axis!r}; the value is {dims}")
    kept = [a for a in dims if a not in (DEFAULT if over is None else over)]
    if kept == dims:
        why = "'over' names none" if over is not None else "by default it removes the rows and positions"
        fix = f"; name the axes it removes in 'over' (\"over\": {json.dumps(dims[-1:])})" if dims else ""
        raise ValueError(f"a reduction of a value laid out {dims} removes no axis of it: {why}{fix}")
    return kept


@op("reductions", layout=reduced)
def mean(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The mean over the axes `over` names, of the values that are there (a missing one is NaN); by default over
    the rows and positions, a `layers` axis kept."""
    of, axes = of.float(), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    return of.nanmean(dim=axes) if of.isnan().any() else of.mean(dim=axes)


@op("reductions", layout=reduced)
def sum(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The sum over the axes `over` names, as `mean` takes them, of the values that are there; NaN where none
    is (a missing row)."""
    of, axes = of.float(), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    return of.nansum(dim=axes).masked_fill(of.isnan().all(dim=axes), float("nan"))


@op("reductions", layout=reduced)
def min(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The least value over the axes `over` names, as `mean` takes them, of the values that are there; NaN where
    none is."""
    of, axes = of.float(), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    if of.numel() == 0:                                  # nothing at all: missing, as sum and mean have it
        return torch.full_like(of.sum(dim=axes), float("nan"))
    return of.nan_to_num(nan=float("inf")).amin(dim=axes).masked_fill(of.isnan().all(dim=axes), float("nan"))


@op("reductions", needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",), bundle="pca")
def pca(of: Tensor, k: int, dims: list[str]) -> dict[str, Tensor]:
    """The top-`k` principal directions (`basis`, `[hidden_size, k]`, orthonormal columns, the largest variance
    first) and the `mean` they are centred on: the bundle a `pca` featurizer loads."""
    x = of.reshape(-1, of.shape[-1]).double()
    x = x[~x.isnan().any(-1)]
    if x.shape[0] <= k:
        raise ValueError(f"pca of {x.shape[0]} vectors has no rank-{k} basis; it needs more than {k}")
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    # An SVD's singular vectors are unique only up to sign; fix it so the largest entry of each is positive.
    basis = vh[:k].T
    basis = basis * basis.gather(0, basis.abs().argmax(0, keepdim=True)).sign()
    return {"basis": basis.to(of.dtype).contiguous(), "mean": mean.to(of.dtype)}
