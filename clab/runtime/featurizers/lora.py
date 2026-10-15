"""The `lora` kind: a low-rank adapter, whose one mode is `apply`."""

import torch
from torch import Tensor

from ...registry import Value
from . import Kind


class Lora(Kind):
    """A low-rank adapter, `A` `[width, rank]` and `B` `[rank, width]` at scale `alpha / rank`: loaded from its
    bundle (its rank and alpha from the bundle's header), or with `rank` and `alpha` one a fit trains, drawn
    from `seed`: `A` standard normal over sqrt(width), `B` zero, so it starts as the identity exactly. It has
    no coordinates, so its one mode is `apply`, and no edit runs in its space."""
    saves = trains = ("A", "B")
    header = drawn = ("rank", "alpha")

    def __init__(self, width: int, rank: int | None = None, alpha: float | None = None, seed: int | None = None,
                 file_path: str | None = None, bundle: dict | None = None, header: dict | None = None):
        if bundle is not None:
            assert header is not None
            self.A: Tensor = bundle["A"]
            self.B: Tensor = bundle["B"]
            self.scale = float(header["alpha"]) / int(header["rank"])
            return
        assert rank is not None and alpha is not None
        generator = torch.Generator().manual_seed(seed or 0)
        self.A, self.B = torch.randn(width, rank, generator=generator) / width ** 0.5, torch.zeros(rank, width)
        self.scale = alpha / rank

    def apply(self, h: Tensor, with_: Value | None = None) -> Tensor:
        """`h + (alpha/rank) (x A) B`: `x` is `h`, or `with` (a value at the same positions, from another site
        of the same forward: an o_proj adapter reads `attention_premix` and applies at `attention_output`), its
        axes after the positions flattened into the adapter's width. The addition lands on the live value."""
        x = h if with_ is None else with_.flatten(h.dim() - 1).to(h)
        return h + self.scale * ((x @ self.A.to(h)) @ self.B.to(h))
