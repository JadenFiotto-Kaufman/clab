"""Transforms: ops that turn each example's value into another of a different layout. The model's own readout,
`encode`, which turns a value into a featurizer's coordinates, the ones over logits (`top_k`, `argmax`), and
`detokenize`, which turns generated ids into text."""

from typing import Annotated, Any

import torch
from torch import Tensor

from ...registry import Featurizer
from .. import featurizers, positions
from . import op
from .metrics import missing, ranked


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: Any) -> Tensor:
    """The model's own readout of a residual-stream value, its final norm and head (the logit lens)."""
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
            positions.checked(given["features"])
        except ValueError as e:
            raise ValueError(f"select.features: {e}") from None


# `encode`'s `select`, which its `check` reads.
Kept = Annotated[dict, "{features: selector}"]


@op("transforms", layout=coordinates, check=selects)
def encode(of: Tensor, featurizer: Featurizer, select: Kept | None = None) -> Tensor:
    """The value's coordinates in `featurizer`'s space (a probe's: its scores, one per class); `select`
    (`{"features": <selector>}`, among them) keeps some. Inside an entry, a `decode` of the same featurizer
    closes it, and the ops between act on the coordinates."""
    z = featurizer.call("encode", of, {})
    if select is None:
        return z
    return z.index_select(-1, torch.tensor(positions.select(select["features"], z.shape[-1]), device=z.device))


@op("transforms", needs=("new",), removes=("new",))
def detokenize(of: Tensor, engine: Any, eos: list[int], tokens: bool = False) -> list:
    """Each row's generated ids as text, stopped before the row's first EOS (the id its generation stops on);
    with `tokens`, as its token strings."""
    tokenizer, stops = engine.tokenizer, set(eos)
    out = []
    for ids in of.tolist():
        ids = next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
        out.append(tokenizer.convert_ids_to_tokens(ids) if tokens else tokenizer.decode(ids))
    return out


@op("transforms", needs=("vocab_size",), removes=("vocab_size",), adds=("k",))
def top_k(of: Tensor, k: int) -> dict:
    """The `k` most likely tokens, `{"ids", "probs"}`, most likely first; as a step, its value also holds their
    `tokens`, decoded. Tokens are ranked by their logits, and an exact tie goes to the lower id (`ranked`), so
    `top_k`'s first is `argmax`'s token."""
    ids = ranked(of, k)
    probs = of.softmax(-1).gather(-1, ids)
    gone = missing(of)[..., None]
    return {"ids": ids.masked_fill(gone, -1), "probs": probs.masked_fill(gone, float("nan"))}


@op("transforms", needs=("vocab_size",), removes=("vocab_size",))
def argmax(of: Tensor) -> Tensor:
    """The id of the most likely token (-1 where the value is missing), ranked by its logit; an exact tie goes to
    the lower id, as in `top_k`."""
    return ranked(of, 1)[..., 0].masked_fill(missing(of), -1)
