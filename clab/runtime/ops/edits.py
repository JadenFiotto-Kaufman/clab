"""Edits: ops that keep the value's layout and change its contents."""

import hashlib
import itertools
from typing import Any, Callable

import torch
from torch import Tensor

from ...registry import Featurizer, Value
from .. import featurizers
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


def applies(args: dict, kind: str) -> Callable:
    return featurizers.KINDS[kind].method("apply")


@op("edits", options=applies)
def apply(of: Tensor, featurizer: Featurizer, **options: Any) -> Tensor:
    """`featurizer`'s own version of the value, in its layout: a subspace's projection, a dictionary's
    reconstruction without its error term, a gate's masked value, an adapter's adapted value (with the
    arguments its kind's `apply` takes beside: `with`, for a lora)."""
    return featurizer.call("apply", of, options)


@op("edits")
def renormalize(of: Tensor, original: Value) -> Tensor:
    """Each vector rescaled to the norm of `original`'s; inside a write, the value before the site's writes."""
    return of * (original.to(of.device).norm(dim=-1, keepdim=True) / of.norm(dim=-1, keepdim=True))


@op("edits")
def normalize(of: Tensor) -> Tensor:
    """Each vector divided by its norm."""
    return of / of.norm(dim=-1, keepdim=True)


@op("edits")
def project_out(of: Tensor, direction: Value) -> Tensor:
    """Each vector less its component along `direction`."""
    direction = torch.as_tensor(direction, device=of.device)
    unit = direction / direction.norm(dim=-1, keepdim=True)
    return of - (of * unit).sum(-1, keepdim=True) * unit


@op("edits")
def noise(of: Tensor, scale: float, dims: list[str], seed: int = 0, rows: Tensor | None = None,
          positions: Tensor | None = None) -> Tensor:
    """`of + scale * noise`, the noise standard normal and drawn per example: the noise at a row's position is
    one draw of the value's other axes from a CPU generator seeded by `seed`, the row's place in its dataset
    and the token's position alone (in a step over a value that holds no positions, its index along `pos`),
    then moved to the value's device and dtype. So batch size, sweep point and forward call never change it,
    no two rows or positions share one, and a document draws the same bits on any device. A value with no
    rows is one draw from `seed`."""
    lead = [i for i, a in enumerate(dims) if a in ("rows", "entries", "pos")]
    if rows is None or not lead:
        return of + scale * torch.randn(of.shape, generator=seeded(seed)).to(device=of.device, dtype=of.dtype)
    shape = [of.shape[i] for i in lead]
    rest = [n for i, n in enumerate(of.shape) if i not in lead]
    draws = []
    for index in itertools.product(*map(range, shape)):
        row = int(rows[index[0]])
        position = int(positions[index]) if positions is not None else index[-1] if len(index) > 1 else None
        draws.append(torch.randn(rest, generator=seeded(seed, row, position)))
    drawn = torch.stack(draws).reshape(*shape, *rest).movedim(list(range(len(lead))), lead)
    return of + scale * drawn.to(device=of.device, dtype=of.dtype)


def seeded(*key) -> torch.Generator:
    """A CPU generator seeded by `key` (a document's seed, a row, a position) and nothing else."""
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest, "big"))


def either_bound(args: dict) -> None:
    """`clamp`'s arguments: at least one bound, and a `min` no greater than its `max`."""
    low, high = args.get("min"), args.get("max")
    if low is None and high is None:
        raise ValueError("'clamp' needs 'min' or 'max', or both")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low > high:
        raise ValueError(f"'clamp' min {low} is greater than its max {high}")


@op("edits", check=either_bound)
def clamp(of: Tensor, min_: float | None = None, max_: float | None = None) -> Tensor:
    """Each element clamped to `[min, max]`; at least one of them given."""
    return of.clamp(min_, max_)
