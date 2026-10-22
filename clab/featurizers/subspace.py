"""The `subspace` kind: a rank-`k` subspace, the first columns of a rotation a fit trains."""

from typing import Literal

import torch
from torch import Tensor

from . import Kind


class Subspace(Kind):
    """A rank-`k` subspace: the first `k` columns of a rotation `origin @ cayley(theta)`. `origin` is a
    Haar-random orthogonal matrix drawn from `seed`, and `theta` (what a fit trains) starts at zero, so an
    untrained subspace is a uniformly random one and each seed draws a different one. Its bundle holds the
    `[width, k]` basis."""
    saves, trains, header, drawn, counts = ("basis",), ("theta",), ("k", "parametrization"), ("k",), ("k",)
    basis: Tensor
    origin: Tensor
    theta: Tensor

    def __init__(self, width: int, k: int | None = None, parametrization: Literal["cayley"] = "cayley",
                 seed: int | None = None, file_path: str | None = None, bundle: dict | None = None):
        if bundle is not None:
            self.basis = bundle["basis"]
            return
        assert k is not None
        generator = torch.Generator().manual_seed(seed or 0)
        threads = torch.get_num_threads()
        # A parallel QR's bits follow the thread count; on one thread they are the same on every host whose BLAS
        # runs the same kernels (the same build on the same instruction set), however many threads it has.
        torch.set_num_threads(1)
        try:
            q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
        finally:
            torch.set_num_threads(threads)
        self.k, self.origin, self.theta = k, (q * r.diagonal().sign()).float(), torch.zeros(width, width)
        self.settle()

    def settle(self) -> None:
        skew = self.theta - self.theta.T
        eye = torch.eye(skew.shape[0], dtype=skew.dtype, device=skew.device)
        self.basis = (self.origin @ torch.linalg.solve(eye - skew, eye + skew))[:, :self.k]

    def encode(self, h: Tensor) -> Tensor:
        return h @ self.basis.to(h)

    def apply(self, h: Tensor) -> Tensor:
        """`h` projected onto the subspace."""
        basis = self.basis.to(h)
        return h @ basis @ basis.T

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + (new - z) @ self.basis.to(h).T
