"""The `gate` kind: a mask over units, soft while a fit trains it and hard when scored."""

import json
from typing import TypedDict

import torch
from torch import Tensor

from clab.featurizers import Kind
from clab.format.registry import Value


class Temperature(TypedDict):
    start: float
    end: float


class Gate(Kind):
    """A mask over units, `theta` starting at +1 on every one: soft (σ(θ/T)) while a fit trains it, hard
    (θ > 0) whenever it is scored. `temperature: {"start", "end"}` sets T, linearly over the fit's epochs
    (default 1). The units are the last axis of its site's selection, or with `over` the axes of the selection it
    names (`layers` among them where the site picks several), in the selection's order: `theta` is laid out on
    them, as many on each as the selection keeps, so `["layers", "num_heads"]` at `attention_premix` over every
    layer is one unit per head. That mask meets the value as an operand does under Ops' one rule, broadcast over
    the axes it leaves out, its slice of each layer the value is at taken by that layer's id. A gate over axes
    serves `apply` alone, since its units are not the last axis an `encode` makes its coordinates. `apply` with
    `with` (a value at the same selection, or a number, as `swap`'s) is the interpolation toward it, exact at either
    end of the mask. Its `apply` is elementwise, so a missing element of the value stays missing and the rest are
    gated (a head gate at `attention_probabilities` on rows of different lengths)."""
    saves = trains = ("theta",)
    header = ("over",)
    scalars = ("mask",)
    elementwise = True

    @classmethod
    def check(cls, fields: dict) -> None:
        """`over`, given, names one or more axes, each once, none of them an example's rows or positions."""
        super().check(fields)
        over = fields.get("over")
        if not isinstance(over, list):
            return                      # null: the last axis (a sweep is refused when the document is read)
        if not over or len(set(over)) < len(over):
            raise ValueError(f"over: {over} names {'no axis' if not over else 'an axis twice'}; a gate's units lie on "
                             "one or more axes of its site's selection, each named once")
        if set(over) & {"rows", "pos"}:
            raise ValueError(f"over: {over} names {', '.join(a for a in over if a in ('rows', 'pos'))}, an example's; "
                             "a gate's units lie on the axes of its site's selection the model has, layers among them")

    def __init__(self, width: int, over: list[str] | None = None, temperature: Temperature | None = None,
                 file_path: str | None = None, bundle: dict[str, Tensor] | None = None,
                 header: dict[str, str] | None = None, kept: dict[str, list[int]] | None = None) -> None:
        # A loading document may leave `over` to the header, which records it.
        self.over = over if over is not None or header is None or "over" not in header else json.loads(header["over"])
        self.kept = kept
        shape = [width] if self.over is None else [len(kept_of(kept)[axis]) for axis in self.over]
        self.theta = bundle["theta"] if bundle is not None else torch.ones(shape)
        self.temperature = temperature
        self.T = temperature["start"] if temperature else 1.0

    def mask(self, dims: list[str] | None = None, layers: int | list[int] | None = None) -> Tensor:
        """The mask: soft while a fit trains it, hard when scored. Over axes, laid out against a value laid out
        `dims` at `layers` as an operand is (`batches.bind_arguments`): at one layer its slice labelled with it, at
        several its slices in their order, then a singleton axis wherever the value has an axis it leaves out."""
        m = torch.sigmoid(self.theta / self.T) if self.training else (self.theta > 0).to(self.theta.dtype)
        if self.over is None or dims is None:
            return m
        axes = list(self.over)
        if "layers" in axes and layers is not None:
            labels = kept_of(self.kept)["layers"]
            if isinstance(layers, int):
                m, axes = m[labels.index(layers)], axes[1:]
            elif layers != labels:
                m = m[[labels.index(layer) for layer in layers]]
        return m.reshape([m.shape[axes.index(a)] if a in axes else 1 for a in dims])

    def encode(self, h: Tensor) -> Tensor:
        return h

    def apply(self, h: Tensor, with_: Value | None = None, dims: list[str] | None = None,
              layers: int | list[int] | None = None) -> Tensor:
        """`h` times the mask `m`, soft while a fit trains it and hard when scored; with `with`, the interpolation
        toward it, `h·m + with·(1−m)` while a fit trains it, and when scored `h` where a unit is on and `with` where
        it is off, each to the bit (a NaN or an infinity on the other side, and a zero's sign, included)."""
        m = self.mask(dims, layers).to(h)
        if with_ is None:
            return h * m
        other = torch.as_tensor(with_, dtype=h.dtype, device=h.device)
        return h * m + other * (1 - m) if self.training else torch.where(m > 0, h, other)

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + self.mask().to(h) * (new - h)

    def anneal(self, epoch: int, epochs: int) -> None:
        """Linear from the temperature's start at the first epoch to its end at the last."""
        if self.temperature:
            t = self.temperature
            self.T = t["start"] + (t["end"] - t["start"]) * (epoch / (epochs - 1) if epochs > 1 else 0.0)

    def scalar(self, name: str) -> Tensor:
        """`mask`: the mean of the soft mask."""
        return torch.sigmoid(self.theta / self.T).mean()


def kept_of(kept: dict[str, list[int]] | None) -> dict[str, list[int]]:
    """A gate over axes' `kept`, which the run supplies wherever it has `over` (`uses.write_widths`)."""
    assert kept is not None, "a gate over axes is made with the indices it keeps on each"
    return kept
