"""Transforms: ops that turn each example's value into another of a different layout. The model's own readout,
`encode`, which turns a value into a featurizer's coordinates, the ranks along an axis (`top_k`, `argmax`, by
default over logits), and `detokenize`, which turns generated ids into text."""

import json
import math
from typing import TYPE_CHECKING, Annotated

import torch
from torch import Tensor

from clab import featurizers
from clab.format import selectors
from clab.format.registry import Featurizer
from clab.ops import VOCABULARY, op
from clab.ops.metrics import ranked

if TYPE_CHECKING:       # the engines sit later in the chain, so only the checker sees them
    from clab.engines import Engine


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: "Engine") -> Tensor:
    """The engine's readout of a residual-stream value (the logit lens): the model's own on nnterp, the standard final
    norm, head and softcap on hooks."""
    return engine.readout(of)


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
