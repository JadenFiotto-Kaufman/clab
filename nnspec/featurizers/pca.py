"""The `pca` kind: the principal directions and mean a `pca` op saved."""

import torch
from torch import Tensor

from nnspec.featurizers import Kind


class Pca(Kind):
    """The top-`k` principal directions (`basis`) and the `mean` a `pca` op saved, loaded from its bundle; its
    coordinates are centred on the mean. A document always loads one (`file_path`); without a bundle it is zeros,
    `[width, k]` and `[width]`, the shapes compile holds a loaded bundle to (`uses.check_bundle_shapes`)."""
    saves, header, counts = ("basis", "mean"), ("k",), ("k",)
    laid = {"basis": ("width", "features"), "mean": ("width",)}

    def __init__(self, width: int, file_path: str, k: int | None = None,
                 bundle: dict[str, Tensor] | None = None) -> None:
        if bundle is None:
            assert k is not None, "a pca without its bundle is drawn at its k"
            bundle = {"basis": torch.zeros(width, k), "mean": torch.zeros(width)}
        self.basis = bundle["basis"]
        self.mean = bundle["mean"]

    def encode(self, h: Tensor) -> Tensor:
        return (h - self.mean.to(h)) @ self.basis.to(h)

    def apply(self, h: Tensor) -> Tensor:
        """`h` projected onto the principal directions through the mean: `mean + (h - mean) B Bᵀ`."""
        mean, basis = self.mean.to(h), self.basis.to(h)
        return mean + (h - mean) @ basis @ basis.T

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + (new - z) @ self.basis.to(h).T
