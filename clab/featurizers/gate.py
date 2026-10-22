"""The `gate` kind: a mask over units, soft while a fit trains it and hard when scored."""

from typing import TypedDict

import torch
from torch import Tensor

from . import Kind


class Temperature(TypedDict):
    start: float
    end: float


class Gate(Kind):
    """A mask over units, `theta` starting at +1 on every one: soft (σ(θ/T)) while a fit trains it, hard
    (θ > 0) whenever it is scored. `temperature: {"start", "end"}` sets T, linearly over the fit's epochs
    (default 1)."""
    saves = trains = ("theta",)
    scalars = ("mask",)

    def __init__(self, width: int, temperature: Temperature | None = None, file_path: str | None = None,
                 bundle: dict | None = None):
        self.theta: Tensor = bundle["theta"] if bundle is not None else torch.ones(width)
        self.temperature = temperature
        self.T = temperature["start"] if temperature else 1.0

    def mask(self) -> Tensor:
        """The mask: soft while a fit trains it, hard when scored."""
        return torch.sigmoid(self.theta / self.T) if self.training else (self.theta > 0).to(self.theta.dtype)

    def encode(self, h: Tensor) -> Tensor:
        return h

    def apply(self, h: Tensor) -> Tensor:
        """`h` times the mask: soft while a fit trains it, hard when scored."""
        return h * self.mask().to(h)

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + self.mask().to(h) * (new - h)

    def anneal(self, epoch: int, epochs: int) -> None:
        """Linear from the temperature's start at the first epoch to its end at the last."""
        if self.temperature:
            t = self.temperature
            self.T = t["start"] + (t["end"] - t["start"]) * (epoch / (epochs - 1) if epochs > 1 else 0.0)

    def scalar(self, name: str) -> Tensor:
        """`mask`: the mean of the soft mask."""
        return torch.sigmoid(self.theta / self.T).mean() if name == "mask" else super().scalar(name)
