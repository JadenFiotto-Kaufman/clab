"""Reductions: ops that remove the rows axis."""

import torch
from torch import Tensor

from ..address import EXAMPLE_AXES
from . import op


# What a reduction without `over` removes: the rows and the positions (a ragged value's entries are both). A
# `layers` axis says which layer each value is, as a feature axis says which unit, so it stays unless `over` names it.
DEFAULT = ("rows", "entries", "pos")


def reduced(dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """Removes the axes `over` names, by default the rows and positions (a ragged value's entries): a `layers`
    axis stays unless `over` names it. A ragged value's entries are its rows' positions: over `pos` it gives a
    value per row, over `rows` and `pos` one over every entry."""
    over = args.get("over")
    if over is None:
        return [a for a in dims if a not in DEFAULT]
    if "entries" in dims and "rows" in over:
        if "pos" not in over:
            raise ValueError(f"a ragged value's rows are reduced with their positions (over ['rows', 'pos']), or "
                             f"its positions alone, one value per row (over ['pos']); the value is {dims}")
        over = [a for a in over if a not in ("rows", "pos")] + ["entries"]
    elif "entries" in dims and "pos" in over:
        dims = [x for a in dims for x in (["rows", "pos"] if a == "entries" else [a])]
    for axis in over:
        if axis not in dims:
            raise ValueError(f"a reduction cannot remove {axis!r}; the value is {dims}")
    return [a for a in dims if a not in over]


@op("reductions", layout=reduced)
def mean(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The mean over the axes `over` names (a ragged value's `pos`: each row's entries), of the values that are
    there (a missing one is NaN); by default over the rows and positions, a `layers` axis kept."""
    over = over if over is not None else [a for a in dims if a in DEFAULT]
    axes = [dims.index(a) for a in over]
    return of.nanmean(dim=axes) if of.isnan().any() else of.mean(dim=axes)


@op("reductions", layout=reduced)
def sum(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The sum over the axes `over` names, as `mean` takes them, of the values that are there; NaN where none
    is (a missing row)."""
    over = over if over is not None else [a for a in dims if a in DEFAULT]
    axes = [dims.index(a) for a in over]
    gone = of.isnan().all(dim=axes) if axes else of.isnan()
    return of.nansum(dim=axes).masked_fill(gone, float("nan"))


@op("reductions", layout=reduced)
def min(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The least value over the axes `over` names, as `mean` takes them, of the values that are there; NaN where
    none is."""
    over = over if over is not None else [a for a in dims if a in DEFAULT]
    axes = [dims.index(a) for a in over]
    if of.numel() == 0:                                  # nothing at all: missing, as sum and mean have it
        return torch.full_like(of.sum(dim=axes), float("nan"))
    gone = of.isnan().all(dim=axes) if axes else of.isnan()
    return of.nan_to_num(nan=float("inf")).amin(dim=axes).masked_fill(gone, float("nan"))


@op("reductions", needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",), bundle="pca")
def pca(of: Tensor, k: int, dims: list[str]) -> dict:
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
