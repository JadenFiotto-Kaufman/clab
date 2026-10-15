"""The `gate` kind: a mask over units, soft while a fit trains it and hard when scored."""

from typing import TypedDict

import torch
from torch import Tensor

from . import ANNEAL, APPLY, ENCODE, REPLACE, SCALARS, kind, of


class Temperature(TypedDict):
    start: float
    end: float


@kind(saves=("theta",), trains=("theta",))
def gate(width: int, temperature: Temperature | None = None, file_path: str | None = None,
         bundle: dict | None = None) -> dict:
    """A mask over units, `theta` starting at +1 on every one: soft (σ(θ/T)) while a fit trains it, hard
    (θ > 0) whenever it is scored. `temperature: {"start", "end"}` sets T, linearly over the fit's epochs
    (default 1)."""
    theta = bundle["theta"] if bundle is not None else torch.ones(width)
    return {"kind": "gate", "theta": theta, "temperature": temperature,
            "T": temperature["start"] if temperature else 1.0, "training": False}


def mask(f: dict) -> Tensor:
    """A gate's mask: soft while a fit trains it, hard when scored."""
    return torch.sigmoid(f["theta"] / f["T"]) if f["training"] else (f["theta"] > 0).to(f["theta"].dtype)


@of(ENCODE, "gate")
def gate_encode(f: dict, h: Tensor) -> Tensor:
    return h


@of(APPLY, "gate")
def gate_apply(f: dict, h: Tensor) -> Tensor:
    """`h` times the mask: soft while a fit trains it, hard when scored."""
    return h * mask(f).to(h.dtype)


@of(REPLACE, "gate")
def gate_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + mask(f).to(h.dtype) * (new - h)


@of(ANNEAL, "gate")
def gate_anneal(f: dict, epoch: int, epochs: int) -> None:
    """Linear from the temperature's start at the first epoch to its end at the last."""
    if f["temperature"]:
        t = f["temperature"]
        f["T"] = t["start"] + (t["end"] - t["start"]) * (epoch / (epochs - 1) if epochs > 1 else 0.0)


SCALARS["gate"] = {"mask": lambda f: torch.sigmoid(f["theta"] / f["T"]).mean()}
