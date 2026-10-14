"""Featurizer kinds: the fields each kind takes, what an instance is, and how a value moves in and out of its space.

A featurizer is a space a read or write happens in, or a function on a value. Its kind's signature is its
contract: the document fields it takes, with `width` (the feature width of the sites it is used at), `bundle`
and `header` (the tensors and header of the bundle its `file_path` names) supplied by the run, and for a probe
`labels` (its class list). An instance is a dict of tensors, never an `nn.Module`, made where the model is: a
declared featurizer before the run's first step, one a fit trains at the fit's start. It is drawn from its
`seed` (else the seed of the fit that trains it, else 0), or loaded from its bundle, which compile reads into
the plan and checks key by key against the document; a loaded one is fixed. It is called by the `featurizer`
op in one of its kind's modes: `encode` gives its coordinates of a value `h` (the last axis becomes the
kind's axis: `features`, or a probe's `classes`), and `apply` its version of `h` in `h`'s own layout (a
subspace's projection, a dictionary's reconstruction, a gate's masked value, an adapter's adapted value). An
edit op in its space runs on its coordinates, and the change is written back as a delta (`replace`), so an op
that changes nothing changes nothing to the bit, and a sparse autoencoder's error term is kept exactly. A fit
trains an instance's `trains` tensors; its objective may name a featurizer's scalar as `<featurizer>.<scalar>`
(a gate's `mask` is the mean of its soft mask; an sae's `reconstruction` and `l1` are over what the body's
encodes of it saw in that run of the body, which the instance keeps while a fit trains it).

A bundle is a safetensors file: an instance's `saves` tensors (the first laid out `[width, ...]`) and a header
saying what made them: the kind and its `header` fields, the model, the site it was used at (`component`,
`layer`), its `width` and the document's name. Both a `pca` op's value and a fit's trained featurizer are
saved as one. A document that loads one is checked against it key by key: the tensors and the kind, its
fields and the model when the plan compiles, the site at every use, and `width` against the model's size
before any forward.

Each kind is one `@kind` function that makes an instance and declares the rest in one line; what it does to a
value is one function per operation, in the table for that operation: one per mode (`ENCODE`, `APPLY`, which
`MODES` names with the layout each produces), `REPLACE`, and where a kind has one, `SETTLE`, `PROJECT`,
`ANNEAL`, `SCALARS`. A new kind touches one place per operation it supports; a mode's function takes the
instance, the value and the mode's own arguments.
"""

import inspect
import json
import os
from typing import Any, Callable, Literal, TypedDict

import torch
from safetensors import safe_open
from torch import Tensor

from . import registry
from .registry import Value

SUPPLIED = ("width", "bundle", "header", "labels")
FORMAT = "clab/1"

KINDS: dict[str, Callable] = {}
DECLARED: dict[str, dict] = {}
ENCODE: dict[str, Callable] = {}
APPLY: dict[str, Callable] = {}
REPLACE: dict[str, Callable] = {}
SETTLE: dict[str, Callable] = {}
PROJECT: dict[str, Callable] = {}
ANNEAL: dict[str, Callable] = {}
SCALARS: dict[str, dict[str, Callable]] = {}

# The modes the `featurizer` op calls a featurizer in: each one's table (one function per kind that has the
# mode), what it produces, and the layout of what it produces on a value laid out `dims`.
MODES: dict[str, dict] = {
    "encode": {"table": ENCODE, "produces": "its coordinates, the last axis becoming {axis}",
               "layout": lambda kind, dims: dims[:-1] + [DECLARED[kind]["axis"]]},
    "apply": {"table": APPLY, "produces": "its version of the value, in the value's own layout",
              "layout": lambda kind, dims: list(dims)},
}


def kind(saves: tuple, trains: tuple = (), axis: str = "features", header: tuple = (), drawn: tuple = ()):
    """Register the decorated function as a featurizer kind under its own name, with what its bundle holds
    (`saves`), what a fit trains (`trains`), the axis its coordinates are on, the fields its bundle's header
    records, and the fields that draw one for a fit to train (`drawn`), which a `file_path` excludes."""
    def register(fn: Callable) -> Callable:
        DECLARED[fn.__name__] = {"saves": saves, "trains": trains, "axis": axis, "header": header, "drawn": drawn}
        return registry.registers(KINDS)(fn)
    return register


def of(table: dict, name: str) -> Callable:
    """A decorator that puts the function into `table` under kind `name`."""
    def add(fn: Callable) -> Callable:
        table[name] = fn
        return fn
    return add


def known(f: dict, table: dict, what: str) -> Callable:
    """`table`'s function for `f`'s kind; a kind that has none is refused by name."""
    if f["kind"] not in table:
        raise ValueError(f"a {f['kind']!r} featurizer has no {what}; kinds with one: {', '.join(table)}")
    return table[f["kind"]]


class Temperature(TypedDict):
    start: float
    end: float


# -- subspace ------------------------------------------------------------------------------------------


@kind(saves=("basis",), trains=("theta",), header=("k", "parametrization"))
def subspace(width: int, k: int, parametrization: Literal["cayley"] = "cayley", seed: int | None = None,
             file_path: str | None = None, bundle: dict | None = None) -> dict:
    """A rank-`k` subspace: the first `k` columns of a rotation `origin @ cayley(theta)`. `origin` is a
    Haar-random orthogonal matrix drawn from `seed`, and `theta` (what a fit trains) starts at zero, so an
    untrained subspace is a uniformly random one and each seed draws a different one. Its bundle holds the
    `[width, k]` basis."""
    if bundle is not None:
        return {"kind": "subspace", "basis": bundle["basis"]}
    generator = torch.Generator().manual_seed(seed or 0)
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    f = {"kind": "subspace", "k": k, "origin": (q * r.diagonal().sign()).float(), "theta": torch.zeros(width, width)}
    subspace_settle(f)
    return f


@of(SETTLE, "subspace")
def subspace_settle(f: dict) -> None:
    skew = f["theta"] - f["theta"].T
    eye = torch.eye(skew.shape[0], dtype=skew.dtype)
    f["basis"] = (f["origin"] @ torch.linalg.solve(eye - skew, eye + skew))[:, :f["k"]]


@of(ENCODE, "subspace")
def subspace_encode(f: dict, h: Tensor) -> Tensor:
    return h @ f["basis"].to(h.dtype)


@of(APPLY, "subspace")
def subspace_apply(f: dict, h: Tensor) -> Tensor:
    """`h` projected onto the subspace."""
    basis = f["basis"].to(h.dtype)
    return h @ basis @ basis.T


@of(REPLACE, "subspace")
def subspace_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + (new - z) @ f["basis"].to(h.dtype).T


# -- gate ------------------------------------------------------------------------------------------------


@kind(saves=("theta",), trains=("theta",))
def gate(width: int, temperature: Temperature | None = None, file_path: str | None = None,
         bundle: dict | None = None) -> dict:
    """A mask over units, `theta` starting at +1 on every one: soft (σ(θ/T)) while a fit trains it, hard
    (θ > 0) whenever it is scored. `temperature: {"start", "end"}` sets T, linearly over the fit's epochs
    (default 1)."""
    theta = bundle["theta"] if bundle is not None else torch.ones(width)
    return {"kind": "gate", "theta": theta, "temperature": temperature,
            "T": temperature["start"] if temperature else 1.0, "training": False}


def mask(f: dict) -> Tensor:
    """A gate's mask: soft while a fit trains it, hard when scored."""
    return torch.sigmoid(f["theta"] / f["T"]) if f["training"] else (f["theta"] > 0).to(f["theta"].dtype)


@of(ENCODE, "gate")
def gate_encode(f: dict, h: Tensor) -> Tensor:
    return h


@of(APPLY, "gate")
def gate_apply(f: dict, h: Tensor) -> Tensor:
    """`h` times the mask: soft while a fit trains it, hard when scored."""
    return h * mask(f).to(h.dtype)


@of(REPLACE, "gate")
def gate_replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    return h + mask(f).to(h.dtype) * (new - h)


@of(ANNEAL, "gate")
def gate_anneal(f: dict, epoch: int, epochs: int) -> None:
    """Linear from the temperature's start at the first epoch to its end at the last."""
    if f["temperature"]:
        t = f["temperature"]
        f["T"] = t["start"] + (t["end"] - t["start"]) * (epoch / (epochs - 1) if epochs > 1 else 0.0)


SCALARS["gate"] = {"mask": lambda f: torch.sigmoid(f["theta"] / f["T"]).mean()}


# -- pca ---------------------------------------------------------------------------------------------------


@kind(saves=("basis", "mean"), header=("k",))
def pca(width: int, k: int, file_path: str, bundle: dict | None = None) -> dict:
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


# -- sae ---------------------------------------------------------------------------------------------------


@kind(saves=("W_enc", "b_enc", "W_dec", "b_dec"), trains=("W_enc", "b_enc", "W_dec", "b_dec"), drawn=("latents",))
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


# -- linear -------------------------------------------------------------------------------------------------


@kind(saves=("W", "b"), trains=("W", "b"), axis="classes", header=("classes",))
def linear(width: int, classes: str, labels: list, file_path: str | None = None, bundle: dict | None = None) -> dict:
    """A probe, `h @ W + b` (both starting at zero), whose coordinates are its classes: the distinct values of
    the `classes` column, sorted, which its bundle's header records. A metric scores them against a column of
    those values, as it scores logits against a column of tokens. It has no way back into the value, so it is
    read in mode `encode` and never written through."""
    if bundle is not None:
        return {"kind": "linear", "classes": labels, "W": bundle["W"], "b": bundle["b"]}
    return {"kind": "linear", "classes": labels, "W": torch.zeros(width, len(labels)), "b": torch.zeros(len(labels))}


@of(ENCODE, "linear")
def linear_encode(f: dict, h: Tensor) -> Tensor:
    return h @ f["W"].to(h.dtype) + f["b"].to(h.dtype)


# -- lora --------------------------------------------------------------------------------------------------


@kind(saves=("A", "B"), trains=("A", "B"), header=("rank", "alpha"), drawn=("rank", "alpha"))
def lora(width: int, rank: int | None = None, alpha: float | None = None, seed: int | None = None,
         file_path: str | None = None, bundle: dict | None = None, header: dict | None = None) -> dict:
    """A low-rank adapter, `A` `[width, rank]` and `B` `[rank, width]` at scale `alpha / rank`: loaded from its
    bundle (its rank and alpha from the bundle's header), or with `rank` and `alpha` one a fit trains, drawn
    from `seed`: `A` standard normal over sqrt(width), `B` zero, so it starts as the identity exactly. It has
    no coordinates, so its one mode is `apply`, and no edit runs in its space."""
    if bundle is not None:
        assert header is not None
        return {"kind": "lora", "A": bundle["A"], "B": bundle["B"],
                "scale": float(header["alpha"]) / int(header["rank"])}
    assert rank is not None and alpha is not None
    generator = torch.Generator().manual_seed(seed or 0)
    return {"kind": "lora", "A": torch.randn(width, rank, generator=generator) / width ** 0.5,
            "B": torch.zeros(rank, width), "scale": alpha / rank}


@of(APPLY, "lora")
def lora_apply(f: dict, h: Tensor, with_: Value | None = None) -> Tensor:
    """`h + (alpha/rank) (x A) B`: `x` is `h`, or `with` (a value at the same positions, from another site
    of the same forward: an o_proj adapter reads `attention_premix` and applies at `attention_output`), its
    axes after the positions flattened into the adapter's width. The addition lands on the live value."""
    x = h if with_ is None else with_.flatten(h.dim() - 1).to(h.dtype)
    return h + f["scale"] * ((x @ f["A"].to(h.dtype)) @ f["B"].to(h.dtype))


# -- what every kind shares ------------------------------------------------------------------------------


def check(kind: str, fields: dict) -> None:
    """Featurizer kind `kind` exists and `fields` bind to its signature."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    registry.check(KINDS[kind], fields, SUPPLIED, repr(kind))


def labels(rows: dict[str, list[dict]], column: str) -> list:
    """A probe's class list: the distinct values of `<dataset>.<column>`, sorted."""
    dataset, _, name = column.partition(".")
    return sorted({row[name] for row in rows[dataset]})


def make(kind: str, fields: dict, width: int, bundle: dict | None, seed: int | None, labels: list | None,
         header: dict | None = None) -> dict:
    """An instance of `kind` from a document's `fields`, its seed defaulting to `seed`."""
    supplied = {"width": width, "bundle": bundle, "header": header, "labels": labels}
    taken = inspect.signature(KINDS[kind]).parameters
    given = {**fields, "seed": seed if fields.get("seed") is None else fields["seed"]} if "seed" in taken else fields
    return KINDS[kind](**given, **{k: v for k, v in supplied.items() if k in taken})


def mode(kind: str, name: str) -> Callable:
    """Featurizer kind `kind`'s function for mode `name`; a mode the kind lacks is refused by name."""
    table = MODES[name]["table"] if name in MODES else {}
    if kind not in table:
        raise ValueError(f"kind {kind!r} has no mode {name!r}; its modes: {', '.join(modes(kind))}")
    return table[kind]


def modes(kind: str) -> list[str]:
    """The modes a featurizer kind has."""
    return [m for m, spec in MODES.items() if kind in spec["table"]]


def layout(kind: str, name: str, dims: list[str]) -> list[str]:
    """The layout of what mode `name` of a `kind` produces on a value laid out `dims`."""
    return MODES[name]["layout"](kind, dims)


def call(f: dict, name: str, h: Tensor, options: dict) -> Tensor:
    """`f` called in mode `name` on `h`, with the mode's own arguments (`options`, by document spelling). An
    encode is kept beside `h` on an instance a fit watches (`seen`), for the scalars its objective names."""
    fn = mode(f["kind"], name)
    names = {k: p.name for k, p in registry.spelled(fn, ("f", "h")).items()}
    out = fn(f, h, **{names[k]: v for k, v in options.items()})
    if name == "encode" and "seen" in f:
        f["seen"].append((h, out))
    return out


def encode(f: dict, h: Tensor) -> Tensor:
    """`h`'s coordinates in `f`'s space, along its last axis: its mode `encode`."""
    return mode(f["kind"], "encode")(f, h)


def replace(f: dict, h: Tensor, z: Tensor, new: Tensor) -> Tensor:
    """`h` with its coordinates `z` changed to `new`, as a delta on `h`: `new == z` gives `h` to the bit."""
    return known(f, REPLACE, "way back into the value")(f, h, z, new)


def within(f: dict, h: Tensor, fn: Callable[[Tensor], Tensor], features: Tensor | None) -> Tensor:
    """`h` after `fn` ran on its coordinates in `f`'s space, on the `features` it keeps (every one when None),
    and the change was written back."""
    z = encode(f, h)
    new = fn(z if features is None else z.index_select(-1, features)).to(z.dtype)
    return replace(f, h, z, new if features is None else z.index_copy(-1, features, new))


def settle(f: dict) -> None:
    """Recompute what derives from a featurizer's trained tensors (a subspace's basis), once per optimizer
    step rather than at every use."""
    known(f, KINDS, "kind")
    if f["kind"] in SETTLE:
        SETTLE[f["kind"]](f)


def project(f: dict) -> None:
    """Put a featurizer back onto its constraint after an optimizer step (an sae's decoder rows at unit norm)."""
    known(f, KINDS, "kind")
    if f["kind"] in PROJECT:
        PROJECT[f["kind"]](f)


def anneal(f: dict, epoch: int, epochs: int) -> None:
    """A featurizer's schedule at `epoch` of `epochs` (a gate's temperature)."""
    known(f, KINDS, "kind")
    if f["kind"] in ANNEAL:
        ANNEAL[f["kind"]](f, epoch, epochs)


def scalar(f: dict, name: str) -> Tensor:
    """A featurizer's scalar a fit's objective may name: a gate's `mask`, the mean of its soft mask."""
    scalars: Any = known(f, SCALARS, "scalar")
    return scalars[name](f)


def size(f: dict) -> int:
    """How many coordinates `f`'s space has."""
    return encode(f, torch.zeros(1, f[DECLARED[f["kind"]]["saves"][0]].shape[0])).shape[-1]


def parameters(f: dict) -> list[Tensor]:
    """The tensors a fit trains."""
    return [f[name] for name in DECLARED[f["kind"]]["trains"]]


def tensors(f: dict) -> dict[str, Tensor]:
    """An instance's tensors its bundle holds."""
    return {name: f[name].detach() for name in DECLARED[f["kind"]]["saves"]}


def width(kind: str, tensors: dict[str, Tensor]) -> int:
    return tensors[DECLARED[kind]["saves"][0]].shape[0]


# -- bundles ---------------------------------------------------------------------------------------------


def header(kind: str, fields: dict, model: Any, labels: list | None = None, site: Any = None,
           width: int | None = None, document: str | None = None) -> dict[str, str]:
    """A bundle's header: the kind and its header fields (with their defaults; a probe's classes as its class
    list), the model, and, when written, the site it was used at, its width and the document's name."""
    params = inspect.signature(KINDS[kind]).parameters
    given = {**{k: p.default for k, p in params.items()}, **fields, "classes": labels}
    # A field a loading document leaves out (a lora's rank and alpha) is read from the header, not checked.
    own = {k: json.dumps(given[k]) if isinstance(given[k], list) else str(given[k]) for k in DECLARED[kind]["header"]
           if given[k] is not None}
    found = {"format": FORMAT, "kind": kind, **own, "model.key": model.key, "model.revision": model.revision,
             "model.dtype": model.dtype}
    if site is not None:
        found.update({"component": site.component, "layer": str(site.layers), "width": str(width),
                      "document": str(document)})
    return found


def bundle(root: str, kind: str, fields: dict, model: Any, labels: list | None = None
           ) -> tuple[dict[str, Tensor], dict[str, str]]:
    """The tensors and header of the bundle `fields["file_path"]` names under `root`: exactly the tensors the
    kind's bundle holds, and the header checked key by key against the featurizer that loads it and the
    plan's `model`."""
    file = os.path.join(root, fields["file_path"])
    if not os.path.exists(file):
        raise ValueError(f"{fields['file_path']} does not exist under the corpus root; run the document that "
                         "writes it first")
    with safe_open(file, "pt") as f:
        found = dict(sorted((f.metadata() or {}).items()))
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    points = sorted({k.rpartition("/")[0] for k in tensors if "/" in k})
    if points:
        raise ValueError(f"{fields['file_path']} holds one bundle per point of a sweep ({', '.join(points)}); a "
                         "featurizer loads one bundle, so run the document that writes it unswept")
    if set(tensors) != set(DECLARED[kind]["saves"]):
        raise ValueError(f"{fields['file_path']} holds {', '.join(sorted(tensors))}; a {kind} bundle holds "
                         f"{', '.join(DECLARED[kind]['saves'])}")
    for key, value in header(kind, fields, model, labels).items():
        if found.get(key) != value:
            raise ValueError(f"{fields['file_path']}: header {key!r} is {found.get(key)!r}; this document "
                             f"expects {value!r}")
    return tensors, found
