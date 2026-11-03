"""Reductions: ops that remove axes, of the entries that are there (`floated`), in float32 whatever the value's dtype;
`pca` works in double and returns float32, as every bundle is (`ops.ABOUT`)."""

import json
import math

import torch
from torch import Tensor

from clab.format.address import EXAMPLE_AXES
from clab.format.registry import Value
from clab.ops import op


def floated(x: Value, like: Tensor | None = None) -> Tensor:
    """`x` in float32 with what is missing from it NaN: a floating value's NaN as it is, and in ids (an integer value)
    the hole, -1, which `argmax`, `top_k` and a read past a row's own positions give, so that arithmetic and a
    reduction take a hole as missing, as a comparison does (`transforms.missing`). A number is a number, on `like`'s
    device where it is given, as a named value moves to it."""
    if not isinstance(x, Tensor):
        return torch.as_tensor(x, device=None if like is None else like.device).float()
    x = x if like is None else x.to(like.device)
    return x.float() if x.is_floating_point() else x.float().masked_fill(x == -1, math.nan)


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
    """The mean over the axes `over` names, of the values that are there (a missing one is NaN, in ids -1); by
    default over the rows and positions, a `layers` axis kept."""
    of, axes = floated(of), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    # With none missing it is the plain mean, whose bits the NaN-skipping mean's division does not always give. The
    # host asks the device whether any is missing: that one wait costs less than computing both means for the device
    # to pick between.
    return of.nanmean(dim=axes) if of.isnan().any() else of.mean(dim=axes)


@op("reductions", layout=reduced)
def sum(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The sum over the axes `over` names, as `mean` takes them, of the values that are there; NaN where none
    is (a missing row)."""
    of, axes = floated(of), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    return of.nansum(dim=axes).masked_fill(of.isnan().all(dim=axes), float("nan"))


@op("reductions", layout=reduced, positionwise=True)
def min(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The least value over the axes `over` names, as `mean` takes them, of the values that are there; NaN where
    none is."""
    of, axes = floated(of), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    if of.numel() == 0:                                  # nothing at all: missing, as sum and mean have it
        return torch.full_like(of.sum(dim=axes), math.nan)
    return of.masked_fill(of.isnan(), math.inf).amin(dim=axes).masked_fill(of.isnan().all(dim=axes), math.nan)


@op("reductions", layout=reduced, positionwise=True)
def max(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The greatest value over the axes `over` names, as `mean` takes them, of the values that are there; NaN where
    none is."""
    of, axes = floated(of), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    if of.numel() == 0:                                  # nothing at all: missing, as sum and mean have it
        return torch.full_like(of.sum(dim=axes), math.nan)
    return of.masked_fill(of.isnan(), -math.inf).amax(dim=axes).masked_fill(of.isnan().all(dim=axes), math.nan)


@op("reductions", layout=reduced)
def std(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The sample standard deviation over the axes `over` names, as `mean` takes them, of the n values that are
    there: `sqrt(sum((x - mean)^2) / (n - 1))`; NaN where fewer than two are."""
    of, axes = floated(of), [i for i, a in enumerate(dims) if a in (DEFAULT if over is None else over)]
    n = (~of.isnan()).sum(dim=axes)
    spread = (of - of.nanmean(dim=axes, keepdim=True)).square().nansum(dim=axes)
    return (spread / (n - 1)).sqrt().masked_fill(n < 2, math.nan)


# What `count` counts along unless `over` names other axes: the positions.
COUNTED = ("pos",)


def counted(dims: list[str], args: dict, kind: str | None) -> list[str]:
    """Removes the axes `over` names, by default `pos`, and with them every axis but `layers`, `rows` and `pos`: one
    count per position of those it keeps (a read's count over `pos` is one number per row)."""
    over = COUNTED if args.get("over") is None else args["over"]
    for axis in over:
        if axis not in dims:
            default = " unless 'over' names the axes it counts along" if args.get("over") is None else ""
            raise ValueError(f"count counts along {axis!r}{default}, and the value is laid out {dims}; 'over' names "
                             f"axes the value has (\"over\": {json.dumps(dims[-1:])})")
    if not over:
        raise ValueError(f"a reduction of a value laid out {dims} removes no axis of it: 'over' names none")
    return [a for a in dims if a not in over and a in EXAMPLE_AXES]


@op("reductions", layout=counted, positionwise=True)
def count(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """How many positions along the axes `over` names (by default `pos`) hold the value, in float32: a position is
    held where any of its entries is there (not NaN; in ids, not -1), its vector counted once, so a read's count is
    each row's own number of positions (a generation's read at `{"frame": "1:"}`, the decode calls the row made) and
    0 where it has none."""
    over = list(COUNTED if over is None else over)
    held = ~of.isnan() if of.is_floating_point() else of != -1
    features = [i for i, a in enumerate(dims) if a not in EXAMPLE_AXES and a not in over]
    if features:
        held = held.any(dim=features, keepdim=True)
    return held.sum(dim=[i for i, a in enumerate(dims) if a in over or i in features]).float()


# What `norm` and `dot` reduce over unless `over` names other axes: a vector's features.
FEATURES = ["hidden_size"]


def featured(dims: list[str], args: dict, kind: str | None) -> list[str]:
    """Removes the axes `over` names, by default `hidden_size`, which a value without one names in `over`."""
    if args.get("over") is None and FEATURES[0] not in dims:
        raise ValueError(f"it reduces over {FEATURES[0]} unless 'over' names the axes, and the value is laid out "
                         f"{dims}; name them in 'over' (\"over\": {json.dumps(dims[-1:])})")
    return reduced(dims, {**args, "over": FEATURES if args.get("over") is None else args["over"]})


def finite_positive(args: dict) -> None:
    """`norm`'s `p` is a finite number greater than 0."""
    p = args.get("p", 2.0)
    if isinstance(p, (int, float)) and not math.isfinite(p):
        raise ValueError(f"'norm' p {p} is not finite; an Lp norm takes a finite p > 0 (2, the length; 1, the sum of "
                         "absolute values), and the largest absolute value over the axes is a `max` of `abs`")
    if isinstance(p, (int, float)) and p <= 0:
        raise ValueError(f"'norm' p {p} is not greater than 0; an Lp norm takes p > 0 (2, the length; 1, the sum "
                         "of absolute values)")


@op("reductions", layout=featured, check=finite_positive)
def norm(of: Tensor, dims: list[str], over: list[str] | None = None, p: float = 2.0) -> Tensor:
    """The Lp norm over the axes `over` names, by default `hidden_size`: `sum(|x|^p)^(1/p)` of the entries that
    are there, as `sum` takes them (at `p` 2, `sqrt` of the `sum` of the `square`, to the bit); NaN where none is."""
    of = floated(of)
    powered = of.square() if p == 2 else of.abs().pow(p)
    total = sum(powered, dims, FEATURES if over is None else over)
    return total.sqrt() if p == 2 else total.pow(1 / p)


@op("reductions", layout=featured)
def dot(of: Tensor, with_: Value, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The dot product with `with` over the axes `over` names, by default `hidden_size`: the `sum` of `of * with`
    over the entries both have, as `sum` takes them (`mul` then `sum`, to the bit); NaN where none is."""
    product = floated(of) * floated(with_, of)
    return sum(product, dims, FEATURES if over is None else over)


# What `entropy` takes its softmax over unless `over` names other axes: the first of these the value has, the
# vocabulary's next tokens or a mixture's experts.
OUTCOMES = ("vocab_size", "num_experts")


def distributed(dims: list[str], args: dict, kind: str | None) -> list[str]:
    """Removes the axes `over` names, by default `vocab_size` where the value has it, else `num_experts`; a value with
    neither names them in `over`."""
    if args.get("over") is None and not set(OUTCOMES) & set(dims):
        raise ValueError(f"it takes the softmax over {' or '.join(OUTCOMES)} unless 'over' names the axes, and the "
                         f"value is laid out {dims}; name them in 'over' (\"over\": {json.dumps(dims[-1:])})")
    over = args.get("over")
    return reduced(dims, {**args, "over": [next(a for a in OUTCOMES if a in dims)] if over is None else over})


@op("reductions", layout=distributed)
def entropy(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The entropy in nats of the softmax over the axes `over` names, by default `vocab_size` where the value has it,
    else `num_experts`: `-sum(p log p)`, `p` the float32 softmax of the entries that are there (a missing one has no
    share, and a vector with none is NaN), a `p` of 0 adding 0; routing entropy is `entropy` of `router_logits` on a
    softmax router, ln of the experts' count where it routes evenly."""
    over = [next(a for a in OUTCOMES if a in dims)] if over is None else over
    axes, of = [i for i, a in enumerate(dims) if a in over], floated(of)
    rest = [i for i in range(of.dim()) if i not in axes]
    flat = of.permute(*rest, *axes).flatten(len(rest))         # the axes the softmax is over, as one last axis
    logp = flat.masked_fill(flat.isnan(), -math.inf).log_softmax(-1)
    p = logp.exp()
    return -(p * logp).masked_fill(p == 0, 0.0).sum(-1)


@op("reductions", needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",), bundle="pca")
def pca(of: Tensor, k: int, dims: list[str]) -> dict[str, Tensor]:
    """The top-`k` principal directions (`basis`, `[hidden_size, k]`, orthonormal columns, the largest variance
    first) and the `mean` they are centred on, float32 whatever the value's dtype, as a fit's trained featurizers
    are: the bundle a `pca` featurizer loads."""
    x = of.reshape(-1, of.shape[-1]).double()
    x = x[~x.isnan().any(-1)]
    if x.shape[0] <= k:
        raise ValueError(f"pca of {x.shape[0]} vectors has no rank-{k} basis; it needs more than {k}")
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    # An SVD's singular vectors are unique only up to sign; fix it so the largest entry of each is positive.
    basis = vh[:k].T
    basis = basis * basis.gather(0, basis.abs().argmax(0, keepdim=True)).sign()
    return {"basis": basis.float().contiguous(), "mean": mean.float()}
