"""Transforms: ops that turn each example's value into another of a different layout. The model's own readout,
the `featurizer` op, which calls a featurizer in one of its kind's modes, and the ones over logits and
generated ids (`top_k`, `argmax`, `decode`)."""

from typing import Any, Callable

from torch import Tensor

from ...registry import Featurizer
from .. import featurizers
from . import op
from .metrics import missing


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: Any) -> Tensor:
    """The model's own readout of a residual-stream value, its final norm and head (the logit lens)."""
    return engine.readout(of)


def mode_layout(dims: list[str], args: dict, kind: str) -> list[str]:
    """its mode's (see Featurizer kinds)"""
    return featurizers.KINDS[kind].layout(args["mode"], dims)


def mode_function(args: dict, kind: str) -> Callable:
    return featurizers.KINDS[kind].mode(args["mode"])


@op("transforms", layout=mode_layout, options=mode_function)
def featurizer(of: Tensor, featurizer: Featurizer, mode: str, **options: Any) -> Tensor:
    """`featurizer` called on the value in `mode`, one of its kind's modes, with the mode's own arguments
    beside (`with`, for a lora's `apply`): `encode` gives its coordinates (a probe's: its scores, one per
    class), `apply` its version of the value. Its axes depend on an argument: they are the mode's, which the
    featurizer's kind declares (see Featurizer kinds), and the op declares them as that function."""
    return featurizer.call(mode, of, options)


@op("transforms", needs=("vocab_size",), removes=("vocab_size",), adds=("k",))
def top_k(of: Tensor, k: int) -> dict:
    """The `k` most likely tokens, `{"ids", "probs"}`, most likely first; as a step, its value also holds their
    `tokens`, decoded."""
    probs, ids = of.softmax(-1).topk(k, dim=-1)
    gone = missing(of)[..., None]
    return {"ids": ids.masked_fill(gone, -1), "probs": probs.masked_fill(gone, float("nan"))}


@op("transforms", needs=("vocab_size",), removes=("vocab_size",))
def argmax(of: Tensor) -> Tensor:
    """The id of the most likely token (-1 where the value is missing)."""
    return of.argmax(-1).masked_fill(missing(of), -1)


@op("transforms", needs=("new",), removes=("new",))
def decode(of: Tensor, engine: Any, eos: list[int], tokens: bool = False) -> list:
    """Each row's generated ids as text, stopped before the row's first EOS (the id its generation stops on);
    with `tokens`, as its token strings."""
    tokenizer, stops = engine.tokenizer, set(eos)
    out = []
    for ids in of.tolist():
        ids = next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
        out.append(tokenizer.convert_ids_to_tokens(ids) if tokens else tokenizer.decode(ids))
    return out
