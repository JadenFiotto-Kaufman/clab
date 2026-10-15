"""The `pca` kind: the principal directions and mean a `pca` op saved."""

from torch import Tensor

from . import Kind


class Pca(Kind):
    """The top-`k` principal directions (`basis`) and the `mean` a `pca` op saved, loaded from its bundle; its
    coordinates are centred on the mean."""
    saves, header = ("basis", "mean"), ("k",)

    def __init__(self, width: int, file_path: str, k: int | None = None, bundle: dict | None = None):
        assert bundle is not None
        self.basis: Tensor = bundle["basis"]
        self.mean: Tensor = bundle["mean"]

    def encode(self, h: Tensor) -> Tensor:
        return (h - self.mean.to(h)) @ self.basis.to(h)

    def apply(self, h: Tensor) -> Tensor:
        """`h` projected onto the principal directions through the mean: `mean + (h - mean) B Bᵀ`."""
        mean, basis = self.mean.to(h), self.basis.to(h)
        return mean + (h - mean) @ basis @ basis.T

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + (new - z) @ self.basis.to(h).T
