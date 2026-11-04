"""The `affine` kind: an affine map of the value onto its own layout, whose one op is `apply`."""

import torch
from torch import Tensor

from clab.featurizers import Kind


class Affine(Kind):
    """An affine map of the last axis onto itself, `h + h W + b`, `W` `[width, width]` and `b` `[width]`: loaded
    from its bundle, or one a fit trains, drawn with both zero, so it starts as the identity exactly and an
    `epochs: 0` fit of it is a null to the bit. It draws nothing at random, so it takes no `seed`; it has no
    coordinates, so its one op is `apply`, which takes no `with`. A tuned lens's translator is one."""
    saves = trains = ("W", "b")

    def __init__(self, width: int, file_path: str | None = None, bundle: dict[str, Tensor] | None = None) -> None:
        self.W = bundle["W"] if bundle is not None else torch.zeros(width, width)
        self.b = bundle["b"] if bundle is not None else torch.zeros(width)

    def apply(self, h: Tensor) -> Tensor:
        """`h + h W + b`."""
        return h + h @ self.W.to(h) + self.b.to(h)
