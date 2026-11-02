"""Transforms: ops that turn each example's value into another, of a different layout or of the same one, or a
gradient. The model's own readout, `encode`, which turns a value into a featurizer's coordinates, `grad`, a gradient
in the value's place, the ranks along an axis (`top_k`, `argmax`, by default over logits), `detokenize`, which turns
generated ids into text, and the elementwise ops, which keep the layout: arithmetic on each entry in float32 (`abs`
to `pow`) and the comparisons (`eq` to `le`), 0 or 1 in the value's dtype (float32 on ids) and missing where either
side is. Unlike an edit, none of them writes back: inside an entry a transform before its last edit acts on the live
value that edit writes, and one after it, or in an entry with no edit, on the value off the model."""

import json
import math
from typing import TYPE_CHECKING, Annotated, Callable

import torch
from torch import Tensor

from clab import featurizers
from clab.format import address, selectors
from clab.format.registry import Featurizer, Value
from clab.ops import VOCABULARY, op
from clab.ops.metrics import ranked

if TYPE_CHECKING:       # the engines sit later in the chain, so only the checker sees them
    from clab.engines import Engine


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: "Engine") -> Tensor:
    """The engine's readout of a residual-stream value (the logit lens): the model's own on nnterp, the standard final
    norm, head and softcap on hooks."""
    return engine.readout(of)


def unstreamed(component: str) -> str | None:
    """Why `readout` cannot project a value read at `component`: the final norm takes the residual stream, and the
    component is not a residual-stream value (`stream` in Components); None where it is. The one check, which compile
    runs on a `readout` inside an entry and on a `readout` step over a read alike (a `mlp_norm_output` is normed
    already, and a readout of it would norm it twice)."""
    if address.COMPONENTS[component]["stream"]:
        return None
    streams = ", ".join(c for c, spec in address.COMPONENTS.items() if spec["stream"])
    return f"readout projects a residual-stream value; {component} is not one: read one of {streams}"


def coordinates(dims: list[str], args: dict, kind: str) -> list[str]:
    """the last axis becomes the featurizer kind's: `features`, a probe's `classes`"""
    return dims[:-1] + [featurizers.KINDS[kind].axis]


def selects(args: dict) -> None:
    """`encode`'s `select`: `{"features": <selector>}`, among the featurizer's coordinates."""
    given = args.get("select")
    if given is None:
        return
    if not isinstance(given, dict) or set(given) != {"features"}:
        raise ValueError(f"select {given!r}: an encode selects among a featurizer's coordinates, "
                         "{'features': <selector>}")
    if not (isinstance(given["features"], dict) and "sweep" in given["features"]):
        try:
            selectors.checked(given["features"])
        except ValueError as e:
            raise ValueError(f"select.features: {e}") from None


@op("transforms", layout=coordinates, check=selects)
def encode(of: Tensor, featurizer: Featurizer, select: Annotated[dict, "{features: selector}"] | None = None) -> Tensor:
    """The value's coordinates in `featurizer`'s space (a probe's: its scores, one per class); `select`
    (`{"features": <selector>}`, among them) keeps some. Inside an entry, a `decode` of the same featurizer
    closes it, and the ops between act on the coordinates."""
    z = featurizer.call("encode", of, {})
    if select is None:
        return z
    return z.index_select(-1, torch.tensor(selectors.select(select["features"], z.shape[-1]), device=z.device))


@op("transforms", needs=("new",), removes=("new",))
def detokenize(of: Tensor, engine: "Engine", eos: list[int], tokens: bool = False) -> list[str] | list[list[str]]:
    """Each row's generated ids as text, stopped before the row's first EOS (the id its generation stops on);
    with `tokens`, as its token strings."""
    tokenizer, stops = engine.tokenizer, set(eos)
    out = []
    for ids in of.tolist():
        ids = next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
        out.append(tokenizer.convert_ids_to_tokens(ids) if tokens else tokenizer.decode(ids))
    return out


@op("transforms")
def grad(of: Tensor, of_: str) -> Tensor:
    """The gradient, at the entry's selection, of the sum of `of`'s value over every axis it has (a missing row adds
    nothing) with respect to the value at this entry's site as the forward goes on with it there, after the writes
    listed before this entry at that site: `of` names, bare, an entry of the same forward at a site the forward
    reaches after this one, whose ops a gradient passes through. The first op of an entry of a `forward` (see
    above); what it meets is that gradient already, which it gives back as it is."""
    return of


# The ops whose value no gradient passes through, with what each gives: compile refuses a `grad` whose `of` runs one,
# by name. The registry cannot tell them apart: `match`, `contains` and `argmax` return a tensor, as `logit_diff`
# does, and remove an axis, as it does; `sign` and the comparisons keep the layout, as `abs` does, and `count` removes
# axes, as `sum` does, each with a gradient of exactly zero.
NO_GRADIENT = {"top_k": "ids with their probabilities or values as a dict", "argmax": "an index", "match": "0 or 1",
               "contains": "0 or 1", "detokenize": "text", "sign": "-1, 0 or 1",
               **{name: "0 or 1" for name in ("eq", "ne", "gt", "ge", "lt", "le")}, "count": "a count"}


# What `argmax` and `top_k` rank unless `over` names another axis: a vocabulary's tokens.
RANKED = "vocab_size"


def ranking(dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """removes the axis `over` names, which it ranks: by default `vocab_size`"""
    over = args.get("over", RANKED)
    if over not in dims:
        hint = f' ("over": {json.dumps(dims[-1])})' if dims else ""
        raise ValueError(f"it ranks the value's {over!r} axis, and the value is laid out {dims}; 'over' names the "
                         f"axis it ranks, one the value has{hint}")
    return [a for a in dims if a != over]


def ranking_k(dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """removes the axis `over` names, which it ranks (by default `vocab_size`), and adds `k`"""
    return ranking(dims, args) + ["k"]


def picked(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The ids of the `k` largest of the entries of `x`'s last axis that are there, a missing one (NaN) ranked
    below every other, and which of them pick no entry that is there: past a vector's own along a ragged axis,
    or every one of a vector that is missing whole."""
    ids = ranked(x.masked_fill(x.isnan(), -math.inf), k)
    return ids, x.gather(-1, ids).isnan()


@op("transforms", layout=ranking_k)
def top_k(of: Tensor, k: int, dims: list[str], over: str = RANKED) -> dict[str, Tensor]:
    """The `k` largest along the axis `over` names, the largest first: `ids`, their indices along it, and over a
    vocabulary (`vocab_size`, a probe's `classes`) `probs`, their float32 softmax over it, over another axis
    `values`, the ranked values as the value has them. By default the most likely tokens, ranked by their logits
    (as a step, its value also holds their `tokens`, decoded); over `features`, a featurizer's coordinates after
    an `encode`. Along a ragged axis (a pattern's `key`) the entries that are there are ranked, and a pick past
    them is id -1. An exact tie goes to the lower index (`ranked`), so `top_k`'s first is `argmax`'s."""
    x = of.movedim(dims.index(over), -1)
    ids, gone = picked(x, k)
    if over in VOCABULARY:
        second = {"probs": x.float().masked_fill(x.isnan(), -math.inf).softmax(-1).gather(-1, ids)}
    else:
        second = {"values": x.gather(-1, ids)}
    return {"ids": ids.masked_fill(gone, -1), **{k: v.masked_fill(gone, float("nan")) for k, v in second.items()}}


@op("transforms", layout=ranking)
def argmax(of: Tensor, dims: list[str], over: str = RANKED) -> Tensor:
    """The index of the largest along the axis `over` names: by default the id of the most likely token, ranked
    by its logit. Along a ragged axis the largest of the entries that are there; -1 where none is (a missing
    value). An exact tie goes to the lower index, as in `top_k`."""
    ids, gone = picked(of.movedim(dims.index(over), -1), 1)
    return ids[..., 0].masked_fill(gone[..., 0], -1)


# -- elementwise: each entry on its own, the layout kept -------------------------------------------------------------


def operand(x: Value, like: Tensor) -> Tensor:
    """An operand (a number, or a named value laid out as the value is) on `like`'s device, in float32 where it is
    floating."""
    x = torch.as_tensor(x, device=like.device)
    return x.float() if x.is_floating_point() else x


@op("transforms")
def abs(of: Tensor) -> Tensor:
    """`|of|`, each entry's absolute value, in float32."""
    return of.float().abs()


@op("transforms")
def exp(of: Tensor) -> Tensor:
    """`e ** of`, each entry's exponential, in float32 (`inf` past about 88.7)."""
    return of.float().exp()


@op("transforms")
def log(of: Tensor) -> Tensor:
    """`ln(of)`, each entry's natural logarithm, in float32: `-inf` at 0 and NaN below it."""
    return of.float().log()


@op("transforms")
def sign(of: Tensor) -> Tensor:
    """Each entry's sign, -1, 0 or 1, in float32; a missing (NaN) entry stays missing."""
    x = of.float()
    return x.sign().masked_fill(x.isnan(), math.nan)


@op("transforms")
def sqrt(of: Tensor) -> Tensor:
    """`of ** 0.5`, each entry's square root, in float32: NaN below 0."""
    return of.float().sqrt()


@op("transforms")
def square(of: Tensor) -> Tensor:
    """`of * of`, each entry squared, in float32."""
    return of.float().square()


@op("transforms")
def neg(of: Tensor) -> Tensor:
    """`-of`, each entry negated, in float32: a read's negation, where `scale` by -1 would write it back."""
    return -of.float()


@op("transforms")
def div(of: Tensor, with_: Value) -> Tensor:
    """`of / with`, in float32, `with` a number or a named value broadcast as `scale`'s `factor` is; NaN wherever
    `with` is 0, so a division by zero is a missing entry, never an infinite one."""
    by = operand(with_, of).float()
    return torch.where(by == 0, math.nan, of.float() / by)


@op("transforms")
def pow(of: Tensor, exponent: Value) -> Tensor:
    """`of ** exponent`, in float32, `exponent` a number or a named value broadcast as `scale`'s `factor` is: NaN
    where a negative entry meets a fractional exponent, `inf` where 0 meets a negative one."""
    return of.float().pow(exponent if not isinstance(exponent, Tensor) else operand(exponent, of).float())


def missing(x: Tensor) -> Tensor:
    """Where a value is missing: NaN, and in ids -1, the id `argmax` and `top_k` give a hole."""
    return x.isnan() if x.is_floating_point() else x == -1


def compared(of: Tensor, with_: Value, holds: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """1 where `holds(of, with)`, else 0: a floating value's in its dtype, each side compared in float32, and ids' in
    float32, compared as they are; NaN where either side is missing (`missing`; a number `with` only where it is
    NaN), so a comparison of a hole is a hole, never a 0 or 1 a `mean` takes."""
    a, b = (of.float() if of.is_floating_point() else of), operand(with_, of)
    gone = missing(a) | (missing(b) if isinstance(with_, Tensor) else b.isnan())
    return holds(a, b).to(of.dtype if of.is_floating_point() else torch.float32).masked_fill(gone, math.nan)


@op("transforms")
def eq(of: Tensor, with_: Value) -> Tensor:
    """1 where `of == with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids)."""
    return compared(of, with_, torch.eq)


@op("transforms")
def ne(of: Tensor, with_: Value) -> Tensor:
    """1 where `of != with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids), so `eq` and `ne` sum to 1 wherever both are there."""
    return compared(of, with_, torch.ne)


@op("transforms")
def gt(of: Tensor, with_: Value) -> Tensor:
    """1 where `of > with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids)."""
    return compared(of, with_, torch.gt)


@op("transforms")
def ge(of: Tensor, with_: Value) -> Tensor:
    """1 where `of >= with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids)."""
    return compared(of, with_, torch.ge)


@op("transforms")
def lt(of: Tensor, with_: Value) -> Tensor:
    """1 where `of < with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids)."""
    return compared(of, with_, torch.lt)


@op("transforms")
def le(of: Tensor, with_: Value) -> Tensor:
    """1 where `of <= with`, else 0, `with` a number or a named value broadcast as `scale`'s `factor` is: compared in
    float32 (ids as they are) and given in the value's dtype (float32 on ids), NaN where either side is missing (NaN, or
    a hole, -1, in ids)."""
    return compared(of, with_, torch.le)
