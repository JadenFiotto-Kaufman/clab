"""The `sae` kind: a sparse autoencoder, loaded or trained, whose writes keep its error term."""

import torch
from torch import Tensor

from . import APPLY, ENCODE, PROJECT, REPLACE, SCALARS, kind, of


@kind(saves=("W_enc", "b_enc", "W_dec", "b_dec"), trains=("W_enc", "b_enc", "W_dec", "b_dec"), header=("latents",),
      drawn=("latents",))
def sae(width: int, latents: int | None = None, file_path: str | None = None, seed: int | None = None,
        bundle: dict | None = None) -> dict:
    """A sparse autoencoder (`W_enc`, `b_enc`, `W_dec`, `b_dec`), loaded from its bundle, or with `latents` a
    dictionary a fit trains: drawn from `seed`, `W_enc` `[width, latents]` standard normal over sqrt(width),
    `W_dec` `[latents, width]` with unit-norm rows (kept unit-norm by a projection after each optimizer step),
    both biases zero. Its coordinates are the latents `relu((h - b_dec) @ W_enc + b_enc)`, and a write keeps its
    error term exactly. Its scalars, over every vector the body's encodes of it saw in that run of the body:
    `reconstruction`, the mean of ‖h − (z @ W_dec + b_dec)‖², and `l1`, the mean |z|."""
    if bundle is not None:
        return {"kind": "sae", **bundle}
    assert latents is not None
    generator = torch.Generator().manual_seed(seed or 0)
    w_dec = torch.randn(latents, width, generator=generator)
    return {"kind": "sae", "W_enc": torch.randn(width, latents, generator=generator) / width ** 0.5,
            "b_enc": torch.zeros(latents), "W_dec": w_dec / w_dec.norm(dim=1, keepdim=True), "b_dec": torch.zeros(width)}


@of(PROJECT, "sae")
def sae_project(f: dict) -> None:
    f["W_dec"].div_(f["W_dec"].norm(dim=1, keepdim=True))


def seen(f: dict) -> tuple[Tensor, Tensor]:
    """The vectors (`[n, width]`) and latents the body's encodes of `f` saw in this run of the body."""
    if not f.get("seen"):
        raise ValueError("an sae's scalars are over what the body's encodes of it saw; this body has none")
    return (torch.cat([h.reshape(-1, h.shape[-1]) for h, _ in f["seen"]]),
            torch.cat([z.reshape(-1, z.shape[-1]) for _, z in f["seen"]]))


def sae_reconstruction(f: dict) -> Tensor:
    h, z = seen(f)
    return (h - (z @ f["W_dec"].to(h.dtype) + f["b_dec"].to(h.dtype))).pow(2).sum(-1).mean()


def sae_l1(f: dict) -> Tensor:
    return seen(f)[1].abs().mean()


SCALARS["sae"] = {"reconstruction": sae_reconstruction, "l1": sae_l1}


@of(ENCODE, "sae")
def sae_encode(f: dict, h: Tensor) -> Tensor:
    return torch.relu((h - f["b_dec"].to(h.dtype)) @ f["W_enc"].to(h.dtype) + f["b_enc"].to(h.dtype))


@of(APPLY, "sae")
def sae_apply(f: dict, h: Tensor) -> Tensor:
    """`h`'s reconstruction: the decoding of its latents, `z @ W_dec + b_dec`, without the error term."""
    return sae_encode(f, h) @ f["W_dec"].to(h.dtype) + f["b_dec"].to(h.dtype)


@of(REPLACE, "sae")
def sae_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + (new - z) @ f["W_dec"].to(h.dtype)
