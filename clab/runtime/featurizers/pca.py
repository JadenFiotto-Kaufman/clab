"""The `pca` kind: the principal directions and mean a `pca` op saved."""

from torch import Tensor

from . import APPLY, ENCODE, REPLACE, kind, of


@kind(saves=("basis", "mean"), header=("k",))
def pca(width: int, file_path: str, k: int | None = None, bundle: dict | None = None) -> dict:
    """The top-`k` principal directions (`basis`) and the `mean` a `pca` op saved, loaded from its bundle; its
    coordinates are centred on the mean."""
    assert bundle is not None
    return {"kind": "pca", "basis": bundle["basis"], "mean": bundle["mean"]}


@of(ENCODE, "pca")
def pca_encode(f: dict, h: Tensor) -> Tensor:
    return (h - f["mean"].to(h.dtype)) @ f["basis"].to(h.dtype)


@of(APPLY, "pca")
def pca_apply(f: dict, h: Tensor) -> Tensor:
    """`h` projected onto the principal directions through the mean: `mean + (h - mean) B Bᵀ`."""
    mean, basis = f["mean"].to(h.dtype), f["basis"].to(h.dtype)
    return mean + (h - mean) @ basis @ basis.T


@of(REPLACE, "pca")
def pca_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + (new - z) @ f["basis"].to(h.dtype).T
