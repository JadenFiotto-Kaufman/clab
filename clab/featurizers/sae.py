"""The `sae` kind: a sparse autoencoder, trained, loaded from a bundle, or published, whose writes keep its
error term."""

from typing import Any, Literal, TypedDict

import torch
from torch import Tensor

from clab.featurizers import Kind, unautocast


class TrainedAt(TypedDict):
    """The site a dictionary was trained at: a component and one layer."""
    component: str
    layers: int


# A published SAE's tensors (SAELens, Gemma Scope) and their shapes, for `width` and `latents`.
PUBLISHED = {"W_enc": ("width", "latents"), "W_dec": ("latents", "width"), "b_enc": ("latents",),
             "b_dec": ("width",), "threshold": ("latents",)}


class Sae(Kind):
    """A sparse autoencoder (`W_enc`, `b_enc`, `W_dec`, `b_dec`), loaded from its bundle, or with `latents` a
    dictionary a fit trains: drawn from `seed`, `W_enc` `[width, latents]` standard normal over sqrt(width),
    `W_dec` `[latents, width]` with unit-norm rows (kept unit-norm by a projection after each optimizer step),
    both biases zero. Its coordinates are the latents `act(pre)`, `pre = (h - b_dec) @ W_enc + b_enc`, or
    `h @ W_enc + b_enc` when not `centered` (Gemma Scope's are not), and a `decode` keeps its error term exactly.
    `activation` is `relu` (`relu(pre)`), `jumprelu` (`relu(pre) * (pre > threshold)`, with the `threshold` a
    published file holds) or `topk` (the relu of the `k` largest of `pre` in each vector, every other latent
    zero). Left unstated (`null`), `activation` and `centered` are a drawn dictionary's, `relu` and centered,
    which its bundle's header records and a document loading a clab bundle is checked against; a published file
    has no default for them. Its `file_path` may name a published SAE, a safetensors file with no clab header
    holding `W_enc`, `W_dec`, `b_enc`, `b_dec` and optionally `threshold`, as SAELens and Gemma Scope publish
    them: its shapes are checked against each other and the site's width, and nothing in it says what model it
    was trained on or how it encodes, so the document states what the file cannot: its `activation` and
    `centered`, and the `site` it was trained at (a name from `sites`, or `{"component", "layers"}` with one
    layer), which every site it is used at is, component and layer, at each layer the site picks (`layers: -1`
    counting as the layer it is). A `site` given for any sae is checked so. Its scalars, over every vector the
    body read its coordinates of in that run of the body: `reconstruction`, the mean of ‖h − (z @ W_dec +
    b_dec)‖², and `l1`, the mean |z|."""
    saves = trains = ("W_enc", "b_enc", "W_dec", "b_dec")
    header, drawn = ("latents", "activation", "k", "centered"), ("latents",)
    unstated = {"activation": "relu", "centered": True}
    scalars = ("reconstruction", "l1")
    threshold: Tensor       # a published JumpReLU's, set with the bundle's other tensors

    def __init__(self, width: int, latents: int | None = None, file_path: str | None = None, seed: int | None = None,
                 activation: Literal["relu", "jumprelu", "topk"] | None = None, k: int | None = None,
                 centered: bool | None = None, site: str | TrainedAt | None = None,
                 bundle: dict[str, Tensor] | None = None) -> None:
        self.activation = self.unstated["activation"] if activation is None else activation
        self.centered = self.unstated["centered"] if centered is None else centered
        self.k = k
        if bundle is not None:
            for name, tensor in bundle.items():
                setattr(self, name, tensor)
            return
        assert latents is not None
        generator = torch.Generator().manual_seed(seed or 0)
        w_dec = torch.randn(latents, width, generator=generator)
        self.W_enc = torch.randn(width, latents, generator=generator) / width ** 0.5
        self.b_enc, self.b_dec = torch.zeros(latents), torch.zeros(width)
        self.W_dec = w_dec / w_dec.norm(dim=1, keepdim=True)

    @classmethod
    def check(cls, fields: dict[str, Any]) -> None:
        """`k` comes with `topk` and only with it, and `jumprelu` loads the threshold a published file holds."""
        super().check(fields)
        activation, k = fields.get("activation"), fields.get("k")
        if activation is None:
            activation = cls.unstated["activation"]
        if isinstance(activation, dict):
            return                          # swept: checked per point
        if (activation == "topk") != (k is not None):
            raise ValueError("an sae's 'k' is how many latents 'topk' keeps; it is given with activation 'topk' "
                             "and only then")
        if activation == "jumprelu" and fields.get("file_path") is None:
            raise ValueError("a 'jumprelu' sae's threshold is a published file's; load one with 'file_path'")
        latents = fields.get("latents")
        if isinstance(k, int) and isinstance(latents, int) and not 0 < k <= latents:
            raise ValueError(f"'topk' keeps k = {k} of {latents} latents; k is from 1 to latents")

    @classmethod
    def published(cls, tensors: dict[str, Tensor], fields: dict[str, Any]) -> dict[str, Tensor]:
        """A published SAE's tensors: exactly `W_enc`, `W_dec`, `b_enc`, `b_dec` and optionally `threshold`,
        shaped for one width and one number of latents (and the document's `latents`, `k` and `activation`).
        Nothing in the file says how it encodes or where it was trained, so the document states `activation`,
        `centered` and `site`; a `threshold` is a JumpReLU's, and only a `jumprelu` loads one."""
        file = fields["file_path"]
        unstated = [k for k in ("activation", "centered", "site") if fields.get(k) is None]
        if unstated:
            raise ValueError(f"{file} is a published sae, which says neither how it encodes nor where it was trained; "
                             f"state {', '.join(map(repr, unstated))}, as its release names "
                             f"{'them' if len(unstated) > 1 else 'it'}")
        expected = set(cls.saves) | ({"threshold"} & set(tensors))
        if set(tensors) != expected:
            raise ValueError(f"{file} has no clab header and holds {', '.join(sorted(tensors))}; a published sae "
                             f"holds {', '.join(PUBLISHED)} (threshold optional)")
        if tensors["W_enc"].dim() != 2:
            raise ValueError(f"{file}: W_enc is {list(tensors['W_enc'].shape)}; it is [width, latents]")
        size = dict(zip(("width", "latents"), tensors["W_enc"].shape))
        for name, tensor in tensors.items():
            if list(tensor.shape) != [size[a] for a in PUBLISHED[name]]:
                raise ValueError(f"{file}: {name} is {list(tensor.shape)}; with W_enc {list(tensors['W_enc'].shape)} "
                                 f"it is [{', '.join(f'{a} {size[a]}' for a in PUBLISHED[name])}]")
        latents, k = fields.get("latents"), fields.get("k")
        if latents is not None and latents != size["latents"]:
            raise ValueError(f"{file} has {size['latents']} latents; this document says {latents}")
        if k is not None and not 0 < k <= size["latents"]:
            raise ValueError(f"{file} has {size['latents']} latents; 'topk' keeps k = {k} of them")
        if (fields["activation"] == "jumprelu") != ("threshold" in tensors):
            raise ValueError(f"{file} holds no threshold; a 'jumprelu' sae needs one" if "threshold" not in tensors
                             else f"{file} holds a JumpReLU threshold; its activation is 'jumprelu', not "
                                  f"{fields['activation']!r}")
        return tensors

    def project(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    def scalar(self, name: str) -> Tensor:
        """Over the vectors and latents the body's encodes saw in this run of the body, the missing ones left out."""
        if not self.seen:
            raise ValueError("an sae's scalars are over what the body's encodes of it saw; this body has none")
        h = torch.cat([h.reshape(-1, h.shape[-1]) for h, _ in self.seen])
        z = torch.cat([z.reshape(-1, z.shape[-1]) for _, z in self.seen])
        there = ~h.isnan().any(-1)
        h, z = h[there], z[there]
        if name == "l1":
            return z.abs().mean()
        with unautocast(h):
            return (h - (z @ self.W_dec.to(h) + self.b_dec.to(h))).pow(2).sum(-1).mean()

    def encode(self, h: Tensor) -> Tensor:
        x = h - self.b_dec.to(h) if self.centered else h
        pre = x @ self.W_enc.to(h) + self.b_enc.to(h)
        if self.activation == "jumprelu":
            return torch.relu(pre) * (pre > self.threshold.to(h))
        if self.activation == "topk":
            assert self.k is not None
            top = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, top.indices, torch.relu(top.values))
        return torch.relu(pre)

    def apply(self, h: Tensor) -> Tensor:
        """`h`'s reconstruction: the decoding of its latents, `z @ W_dec + b_dec`, without the error term."""
        return self.encode(h) @ self.W_dec.to(h) + self.b_dec.to(h)

    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
        return h + (new - z) @ self.W_dec.to(h)
