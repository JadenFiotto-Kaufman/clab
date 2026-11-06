"""The `constant` kind: one vector written in place of the value, whose one op is `apply`."""

import torch
from torch import Tensor

from nnspec.featurizers import Kind
from nnspec.format.registry import Value


class Constant(Kind):
    """A vector of its site's width, `vector`, which `apply` writes in place of the value at every row and position:
    a constant ablation. The vector lies on the site's last axis and is shared across every other axis the site keeps,
    so at a site that keeps several heads each takes the same vector; a constant per head is a site per head, each with
    a constant of its own. `init` draws it: a number, every entry that number (0 a zero ablation), or the name of a
    value that is one vector of the site's width (laid out on the site's last axis alone, in the site's own
    coordinates, on no sweep axis, made by a step before the featurizer's first use: a `mean` of a read at the site
    over every other axis, a mean ablation), which compile checks and the run passes in where it makes the instance.
    A fit trains it from there: drawn as a mean and trained on an objective it is the optimal constant, and at
    `epochs: 0`, or untrained outside a fit, it writes `init` itself, to the bit. Loaded, it is its bundle's `vector`,
    and `init` beside `file_path` is refused. It has no coordinates, so its one op is `apply`."""
    saves = trains = ("vector",)
    laid = {"vector": ("width",)}
    drawn = ("init",)

    def __init__(self, width: int, init: Value | None = None, file_path: str | None = None,
                 bundle: dict[str, Tensor] | None = None) -> None:
        if bundle is not None:
            self.vector = bundle["vector"]
            return
        assert init is not None and not isinstance(init, str), "a named init is the value the run made"
        if not isinstance(init, Tensor):
            self.vector = torch.full((width,), float(init))
        elif tuple(init.shape) == (width,):
            self.vector = init.detach().to("cpu", torch.float32).clone()
        else:
            raise ValueError(f"a constant's init is one vector of its width, [{width}], and this one is "
                             f"{list(init.shape)}")

    def apply(self, h: Tensor) -> Tensor:
        """`vector` in place of `h`, broadcast over its rows, positions and every axis before its last, in `h`'s
        dtype and on its device: a copy of the vector, with no arithmetic."""
        return self.vector.to(h).expand_as(h).clone()
