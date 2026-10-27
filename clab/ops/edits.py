"""Edits: ops that keep the value's layout and change its contents."""

import hashlib
import itertools
from typing import Literal

import torch
from torch import Tensor

from clab.format.registry import Featurizer, Value
from clab.ops import op


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
def scale(of: Tensor, factor: Value) -> Tensor:
    """`of * factor`, broadcast as `add`'s `with` is: a number, or a named value (a dose measured by a step; a
    `[rows, pos]` factor scales each position's vector by its own number, over `hidden_size`)."""
    return of * torch.as_tensor(factor, device=of.device)


@op("edits")
def apply(of: Tensor, featurizer: Featurizer, with_: Value | None = None) -> Tensor:
    """`featurizer`'s own version of the value, in its layout: a subspace's projection, a dictionary's
    reconstruction without its error term, a gate's masked value, an adapter's adapted value. `with` is an
    adapter's input where that is read at another site of the same forward (a lora's); a kind whose `apply`
    takes none refuses it."""
    return featurizer.call("apply", of, {} if with_ is None else {"with_": with_})


def uncoded(dims: list[str], args: dict, kind: str | None) -> list[str]:
    """the layout its `encode` had"""
    raise ValueError("'decode' closes an encode of its entry, whose layout it returns to; nothing else lays it out")


@op("edits", layout=uncoded)
def decode(of: Tensor, featurizer: Featurizer) -> Tensor:
    """Closes the entry's most recent `encode` not yet closed, which is of `featurizer`: the value from before
    that `encode`, the change of its coordinates written back as a delta (`replace`), so an op between them
    that changes no coordinate changes nothing and a sparse autoencoder's error term survives. The pair runs as
    one, the ops between on the coordinates (`batches.edited`)."""
    raise ValueError("'decode' runs as the close of its encode, with the ops between (batches.edited), never alone")


def present(x: Tensor) -> Tensor:
    """`x` with its missing entries (NaN) as 0, so that a norm or a dot product over a vector takes the entries
    that are there: an attention pattern has none past a row's own keys. A vector missing whole stays missing."""
    return x.masked_fill(x.isnan(), 0)


@op("edits")
def renormalize(of: Tensor, original: Value, norm: Literal["l2", "sum"] = "l2") -> Tensor:
    """Each vector rescaled to `original`'s: under `norm` `l2`, to its L2 norm; under `sum`, to its sum, so a row
    of an attention pattern a write knocked keys out of is a distribution again. `original` is, inside an entry,
    the value at the site before the step's writes (inside an `encode`, its coordinates). Norms and sums take the
    entries that are there (`present`), and a vector whose norm or sum is zero is left as it is (a row whose
    every key was knocked out stays zero)."""
    def size(x: Tensor) -> Tensor:
        return x.sum(-1, keepdim=True) if norm == "sum" else x.norm(dim=-1, keepdim=True)
    now = size(present(of))
    zero = now == 0
    return of * torch.where(zero, 1.0, size(present(original.to(of.device))) / now.masked_fill(zero, 1))


@op("edits")
def normalize(of: Tensor) -> Tensor:
    """Each vector divided by its norm, of the entries that are there (`present`); a vector whose norm is zero is
    left as it is, as `renormalize` leaves one."""
    norm = present(of).norm(dim=-1, keepdim=True)
    return of / norm.masked_fill(norm == 0, 1)


@op("edits")
def project_out(of: Tensor, direction: Value) -> Tensor:
    """Each vector less its component along `direction`, which is normalised first (its length does not matter),
    over the entries both have (`present`); a zero `direction` has no component to remove, and leaves each vector
    as it is."""
    direction = torch.as_tensor(direction, device=of.device)
    norm = present(direction).norm(dim=-1, keepdim=True)
    unit = direction / norm.masked_fill(norm == 0, 1)
    return of - present(of * unit).sum(-1, keepdim=True) * unit


@op("edits")
def noise(of: Tensor, scale: Value, dims: list[str], seed: int = 0, rows: Tensor | None = None,
          positions: Tensor | None = None) -> Tensor:
    """`of + scale * noise`, `scale` a number or a named value broadcast as `add`'s `with` is, the noise
    standard normal and drawn per example: the noise at a row's position is one draw of the value's other axes
    from a CPU generator seeded by `seed`, the row's place in its dataset and the token's position alone (in a
    step over a value that holds no positions, its index along `pos`), then moved to the value's device. So
    batch size, sweep point and forward call never change it, no two rows or positions share one, a write across
    layers draws the same bits at each of them (as one entry per layer with one seed does), and a document draws
    the same bits on any device. A value with no rows is one draw from `seed`."""
    lead = [i for i, a in enumerate(dims) if a in ("rows", "pos")]
    scale = torch.as_tensor(scale, device=of.device)
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


def seeded(*key: int | None) -> torch.Generator:
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
