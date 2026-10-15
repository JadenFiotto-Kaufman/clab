"""Edits: ops that keep the value's layout and change its contents."""

import torch
from torch import Tensor

from ...registry import Value
from . import op


@op("edits")
def swap(of: Tensor, with_: Value) -> Tensor:
    """Replace the value with `with`, broadcast over its rows and positions."""
    return torch.as_tensor(with_, dtype=of.dtype, device=of.device).expand_as(of).clone()


@op("edits")
def add(of: Tensor, with_: Value, scale: float = 1.0) -> Tensor:
    """`of + scale * with`."""
    return of + scale * torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op("edits")
def sub(of: Tensor, with_: Value) -> Tensor:
    """`of - with`."""
    return of - torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op("edits")
def scale(of: Tensor, factor: float) -> Tensor:
    """`of * factor`."""
    return of * factor


@op("edits")
def renormalize(of: Tensor, original: Value) -> Tensor:
    """Each vector rescaled to the norm of `original`'s; inside a write, the value before the site's writes."""
    return of * (original.norm(dim=-1, keepdim=True) / of.norm(dim=-1, keepdim=True))


@op("edits")
def normalize(of: Tensor) -> Tensor:
    """Each vector divided by its norm."""
    return of / of.norm(dim=-1, keepdim=True)


@op("edits")
def project_out(of: Tensor, direction: Value) -> Tensor:
    """Each vector less its component along `direction`."""
    unit = direction / direction.norm(dim=-1, keepdim=True)
    return of - (of * unit).sum(-1, keepdim=True) * unit


@op("edits")
def noise(of: Tensor, scale: float, seed: int = 0) -> Tensor:
    """`of + scale * noise`, the noise standard normal, one draw per call: drawn on the CPU from a generator
    seeded by `seed`, then moved to the value's device and dtype, so a document draws the same bits on any
    device. A write calls it once per batch, so every batch draws the same noise."""
    drawn = torch.randn(of.shape, generator=torch.Generator().manual_seed(seed))
    return of + scale * drawn.to(device=of.device, dtype=of.dtype)


def either_bound(args: dict) -> None:
    if args.get("min") is None and args.get("max") is None:
        raise ValueError("'clamp' needs 'min' or 'max', or both")


@op("edits", check=either_bound)
def clamp(of: Tensor, min_: float | None = None, max_: float | None = None) -> Tensor:
    """Each element clamped to `[min, max]`; at least one of them given."""
    return of.clamp(min_, max_)
