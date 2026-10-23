"""The `linear` kind: a probe whose coordinates are its classes."""

import torch
from torch import Tensor

from clab.featurizers import Kind


class Linear(Kind):
    """A probe, `h @ W + b` (both starting at zero), whose coordinates are its classes: the distinct values of
    the `classes` column, sorted, which its bundle's header records. A metric scores them against a column of
    those values, as it scores logits against a column of tokens. It has no way back into the value, so it is
    read by `encode` and never decoded."""
    saves = trains = ("W", "b")
    axis, header = "classes", ("classes",)

    def __init__(self, width: int, classes: str, labels: list, file_path: str | None = None,
                 bundle: dict[str, Tensor] | None = None) -> None:
        self.classes = labels
        self.W = bundle["W"] if bundle is not None else torch.zeros(width, len(labels))
        self.b = bundle["b"] if bundle is not None else torch.zeros(len(labels))

    def encode(self, h: Tensor) -> Tensor:
        return h @ self.W.to(h) + self.b.to(h)
