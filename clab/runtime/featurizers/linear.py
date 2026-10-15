"""The `linear` kind: a probe whose coordinates are its classes."""

import torch
from torch import Tensor

from . import ENCODE, kind, of


@kind(saves=("W", "b"), trains=("W", "b"), axis="classes", header=("classes",))
def linear(width: int, classes: str, labels: list, file_path: str | None = None, bundle: dict | None = None) -> dict:
    """A probe, `h @ W + b` (both starting at zero), whose coordinates are its classes: the distinct values of
    the `classes` column, sorted, which its bundle's header records. A metric scores them against a column of
    those values, as it scores logits against a column of tokens. It has no way back into the value, so it is
    read in mode `encode` and never written through."""
    if bundle is not None:
        return {"kind": "linear", "classes": labels, "W": bundle["W"], "b": bundle["b"]}
    return {"kind": "linear", "classes": labels, "W": torch.zeros(width, len(labels)), "b": torch.zeros(len(labels))}


@of(ENCODE, "linear")
def linear_encode(f: dict, h: Tensor) -> Tensor:
    return h @ f["W"].to(h.dtype) + f["b"].to(h.dtype)
