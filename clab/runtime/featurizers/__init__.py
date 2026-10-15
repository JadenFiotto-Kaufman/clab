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
from typing import Any, Callable

import torch
from torch import Tensor

from ...plan import registry

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
    records, and the fields one drawn without a `file_path` needs (`drawn`). A document that loads one from
    its `file_path` may state any of the header's fields, and each it states is checked against the header."""
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


# The kinds register in this order, and `describe` and FORMAT.md list them in it.
from .subspace import subspace  # noqa: E402,F401
from .gate import gate, mask  # noqa: E402,F401
from .pca import pca  # noqa: E402,F401
from .sae import sae  # noqa: E402,F401
from .linear import linear  # noqa: E402,F401
from .lora import lora  # noqa: E402,F401
from .bundle import bundle, header  # noqa: E402,F401
