"""Transforms that read a value through the model or a featurizer: the model's own readout, and the
`featurizer` op, which calls a featurizer in one of its kind's modes."""

from typing import Any, Callable

from torch import Tensor

from ...registry import Featurizer
from .. import featurizers
from . import op


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: Any) -> Tensor:
    """The model's own readout of a residual-stream value, its final norm and head (the logit lens)."""
    return engine.readout(of)


def mode_layout(dims: list[str], args: dict, kind: str) -> list[str]:
    """its mode's (see Featurizer kinds)"""
    return featurizers.layout(kind, args["mode"], dims)


def mode_function(args: dict, kind: str) -> Callable:
    return featurizers.mode(kind, args["mode"])


@op("transforms", layout=mode_layout, options=mode_function)
def featurizer(of: Tensor, featurizer: Featurizer, mode: str, **options: Any) -> Tensor:
    """`featurizer` called on the value in `mode`, one of its kind's modes, with the mode's own arguments
    beside (`with`, for a lora's `apply`): `encode` gives its coordinates (a probe's: its scores, one per
    class), `apply` its version of the value. Its axes depend on an argument: they are the mode's, which the
    featurizer's kind declares (see Featurizer kinds), and the op declares them as that function."""
    return featurizers.call(featurizer, mode, of, options)
