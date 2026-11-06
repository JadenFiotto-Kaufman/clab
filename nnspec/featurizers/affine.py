"""The `affine` kind: an affine map of the value onto its own layout, whose one op is `apply`."""

import json

import torch
from torch import Tensor

from nnspec.featurizers import Kind


class Affine(Kind):
    """An affine map of the last axis onto itself, `h + h W + b`, `W` `[width, width]` and `b` `[width]`: loaded
    from its bundle, or one a fit trains, drawn with both zero, so it starts as the identity exactly and an
    `epochs: 0` fit of it is a null to the bit. It draws nothing at random, so it takes no `seed`; it has no
    coordinates, so its one op is `apply`, which takes no `with`. With `over: ["layers"]` it holds one map per
    layer of its site's selection, `W` `[layers, width, width]` and `b` `[layers, width]`, its width still the
    last axis: at each layer the value is at, `apply` takes that layer's map by the layer's id. `over` names
    `layers` and no other axis, since a map is a matrix over the last axis. A tuned lens's translators are one
    affine over layers."""
    saves = trains = ("W", "b")
    laid = {"W": ("width", "width"), "b": ("width",)}
    header = ("over",)
    stacked = True

    @classmethod
    def check(cls, fields: dict) -> None:
        """`over`, given, is `["layers"]`."""
        super().check(fields)
        over = fields.get("over")
        if isinstance(over, list) and over != ["layers"]:
            raise ValueError(f"over: {over}: an 'affine' is over ['layers'] or nothing, one map per layer of its "
                             "site's selection: a map is a matrix over the value's last axis, and one per head or per "
                             "neuron would be another kind")

    def __init__(self, width: int, over: list[str] | None = None, file_path: str | None = None,
                 bundle: dict[str, Tensor] | None = None, header: dict[str, str] | None = None,
                 kept: dict[str, list[int]] | None = None) -> None:
        # A loading document may leave `over` to the header, which records it.
        self.over = over if over is not None or header is None or "over" not in header else json.loads(header["over"])
        assert self.over is None or kept is not None, "an affine over layers is made with the layers it keeps"
        self.layers = None if self.over is None or kept is None else kept["layers"]
        stack = [] if self.layers is None else [len(self.layers)]
        self.W = bundle["W"] if bundle is not None else torch.zeros(*stack, width, width)
        self.b = bundle["b"] if bundle is not None else torch.zeros(*stack, width)

    def apply(self, h: Tensor, dims: list[str] | None = None, layers: int | list[int] | None = None) -> Tensor:
        """`h + h W + b`; over layers, with the map of the layer the value is at, by its id, or on a value with a
        `layers` axis of its own (a step over a read at several layers), each layer's map on that layer's vectors."""
        W, b = self.W.to(h), self.b.to(h)
        if self.layers is None:
            return h + h @ W + b
        if isinstance(layers, int):
            at = self.layers.index(layers)
            return h + h @ W[at] + b[at]
        assert dims is not None and layers is not None, "an affine over layers meets a value at its layers"
        picked, axis = [self.layers.index(layer) for layer in layers], dims.index("layers")
        x = h.movedim(axis, 0)
        moved = (x.reshape(len(picked), -1, x.shape[-1]) @ W[picked]).reshape(x.shape)
        return (x + moved + b[picked].reshape(len(picked), *[1] * (x.dim() - 2), -1)).movedim(0, axis)
