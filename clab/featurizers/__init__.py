"""Featurizer kinds: the fields each kind takes, what an instance is, and how a value moves in and out of its space.

A featurizer is a space an entry's ops encode into, or a function on a value. Its kind's signature is its
contract: the document fields it takes, with `width` (the feature width of the sites it is used at), `bundle`
and `header` (the tensors and header of the bundle its `file_path` names) supplied by the run, and for a probe
`labels` (its class list). An instance holds plain tensors, never an `nn.Module`, made where the model is: a
declared featurizer before the run's first step, one a fit trains at the fit's start. It is drawn from its
`seed` (else the seed of the fit that trains it, else 0), or loaded from its bundle, which compile reads into
the plan and checks key by key against the document; a loaded one is fixed. Three ops reach it: `encode` gives
its coordinates of a value `h` (the last axis becomes the kind's axis: `features`, or a probe's `classes`),
`decode` writes a change of them back into `h` as a delta (`replace`), so an op between them that changes
nothing changes nothing to the bit, and a sparse autoencoder's error term is kept exactly; and `apply` gives
its version of `h` in `h`'s own layout (a subspace's projection, a dictionary's reconstruction, a gate's
masked value, an adapter's adapted value). A fit trains an instance's `trains` tensors; its objective may name
a featurizer's scalar as `<featurizer>.<scalar>` (a gate's `mask` is the mean of its soft mask; an sae's
`reconstruction` and `l1` are over the coordinates the body read of it in that run of the body, its
`encode`s no `decode` closes, which the instance keeps while a fit trains it). Its math runs on
the value's device and dtype, with autocast off: a model served in bfloat16 under autocast leaves a float32
featurizer's float32 math alone.

A bundle is a safetensors file: an instance's `saves` tensors (the first laid out `[width, ...]`) and a header
saying what made them: the kind and its `header` fields, the model, the site it was used at (`component`,
`layer`), its `width` and the document's name. Both a `pca` op's value and a fit's trained featurizer are
saved as one. A document that loads one is checked against it key by key: the tensors and the kind, its
fields and the model when the plan compiles, the site at every use, and `width` against the model's size
before any forward. A file is clab's when its header's `format` is `clab/...`; one that carries clab's header
keys under another format is refused. Any other file is a published one, its kind's to read or refuse: an
`sae` reads one (SAELens, Gemma Scope), whose shapes are checked against each other and the site's width:
nothing in it says what model or site it was made at, so the document states the layer, which every use is
checked against.

Each kind is one class, in a module of its own, registered under its name in lower case. It declares as class
attributes what its bundle holds (`saves`), what a fit trains (`trains`), the axis its coordinates are on
(`axis`), the fields its bundle's header records (`header`), the fields one drawn without a `file_path` needs
(`drawn`), the scalars a fit's objective may name (`scalars`) and the fields that count directions of the
value, so are at most its width (`counts`); its constructor makes an instance. The ops it serves are its
methods `encode(h)`, `replace(h, z, new)` (what `decode` runs, so a kind serves `decode` when it has both) and
`apply(h, ...)`, a kind defining only those it has; the rest is the protocol `Kind` declares, with a default for
what a kind need not have.
"""

import functools
import inspect
from contextlib import nullcontext
from typing import Any, Callable, ClassVar, Protocol, cast

import torch
from torch import Tensor

from clab.format import registry

SUPPLIED = ("width", "bundle", "header", "labels")
FORMAT = "clab/1"
KINDS: dict[str, type["Kind"]] = {}

# The ops that reach a featurizer: the method of the kind's each runs, and what it produces on a value, for a
# kind whose coordinates are on `axis`. A kind serves an op when it defines its method (one with `replace` has
# `encode` too).
OPS: dict[str, dict] = {
    "encode": {"method": "encode", "produces": "gives its coordinates, the last axis becoming {axis}"},
    "decode": {"method": "replace", "produces": "writes a change of its coordinates back into the value"},
    "apply": {"method": "apply", "produces": "gives its version of the value, in the value's own layout"},
}


def unautocast(h: Tensor):
    """Autocast off on `h`'s device, for the length of a `with`: a featurizer's math stays in its own dtype."""
    kind = h.device.type
    return torch.autocast(kind, enabled=False) if torch.amp.is_autocast_available(kind) else nullcontext()


def exact(math: Callable) -> Callable:
    """A kind's method on a value `h` (`encode`, `replace`, `apply`), run with autocast off."""
    @functools.wraps(math)
    def run(self, h: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        with unautocast(h):
            return math(self, h, *args, **kwargs)
    return run


class Kind:
    """The protocol every featurizer kind implements. An instance holds its tensors as plain attributes; every
    tensor it holds that meets a value `h` is moved to `h`'s device and dtype where it is used (`t.to(h)`),
    so the tensors themselves stay where they were made."""
    saves: ClassVar[tuple[str, ...]] = ()
    trains: ClassVar[tuple[str, ...]] = ()
    axis: ClassVar[str] = "features"
    header: ClassVar[tuple[str, ...]] = ()
    drawn: ClassVar[tuple[str, ...]] = ()
    scalars: ClassVar[tuple[str, ...]] = ()
    counts: ClassVar[tuple[str, ...]] = ()     # fields that count directions of the value: at most its width
    kind: ClassVar[str]
    # Set by the fit that trains the instance: whether this run of its body trains it (a gate is soft then),
    # and every value the body's encodes of it saw, beside its coordinates, for its scalars.
    training: bool = False
    seen: list[tuple[Tensor, Tensor]] | None = None

    def __init_subclass__(cls) -> None:
        cls.kind = cls.__name__.lower()
        for name in ("encode", "replace", "apply"):
            if name in vars(cls):
                setattr(cls, name, exact(vars(cls)[name]))
        KINDS[cls.kind] = cls

    # -- what the document and the plan see ------------------------------------------------------------------

    @classmethod
    def check(cls, fields: dict) -> None:
        """`fields` bind to the kind's signature."""
        registry.check(cls, fields, SUPPLIED, repr(cls.kind))

    @classmethod
    def ops(cls) -> list[str]:
        """The ops the kind serves: those whose method it defines."""
        return [op for op, o in OPS.items() if hasattr(cls, o["method"])]

    @classmethod
    def method(cls, op: str) -> Callable:
        """The kind's method op `op` runs; an op the kind does not serve is refused by name."""
        if op not in cls.ops():
            raise ValueError(f"a {cls.kind!r} featurizer has no {op}; it serves {', '.join(cls.ops())}")
        return getattr(cls, OPS[op]["method"])

    @classmethod
    def width(cls, tensors: dict[str, Tensor]) -> int:
        """The feature width of a bundle's tensors: the first axis of the first one it saves."""
        return tensors[cls.saves[0]].shape[0]

    @classmethod
    def published(cls, tensors: dict[str, Tensor], fields: dict) -> dict[str, Tensor]:
        """The tensors of a published file (one with no clab header), as an instance loads them; a kind that
        reads none refuses."""
        raise ValueError(f"{fields['file_path']} has no clab header; a {cls.kind} bundle is one clab wrote")

    # -- what the run calls ----------------------------------------------------------------------------------

    def call(self, op: str, h: Tensor, options: dict) -> Tensor:
        """Op `op` (`encode` or `apply`) of the instance on `h`, with the method's own arguments (`options`, by
        document spelling). A missing vector of `h` (NaN), and the same vector of an option laid out on its
        rows and positions, goes in as zeros and comes out missing, so the math, and a fit's gradients through
        it, see none of it. An encode is kept beside `h` while a fit watches the
        instance (`seen`)."""
        fn = self.method(op)
        names = {k: p.name for k, p in registry.spelled(fn, ("self", "h")).items()}
        gone = h.isnan().any(-1, keepdim=True)
        x = h
        if gone.any():
            x = h.masked_fill(gone, 0.0)
            options = {k: v.masked_fill(gone.reshape(*gone.shape[:-1], *[1] * (v.dim() - gone.dim() + 1)), 0.0)
                       if isinstance(v, Tensor) and v.shape[:gone.dim() - 1] == gone.shape[:-1] else v
                       for k, v in options.items()}
        out = fn(self, x, **{names[k]: v for k, v in options.items()})
        out = out.masked_fill(gone, float("nan")) if gone.any() else out
        if op == "encode" and self.seen is not None:
            self.seen.append((h, out))
        return out

    def within(self, h: Tensor, fn: Callable[[Tensor], Tensor], features: Tensor | None) -> Tensor:
        """`h` after `fn` ran on its coordinates, on the `features` it keeps (every one when None), and the change
        was written back: an `encode` and the `decode` that closes it, `fn` the ops between; `new == z` gives `h`
        to the bit."""
        space = cast(Writable, self)         # compile closes an encode only of a kind that serves `decode`
        with unautocast(h):
            z = space.encode(h)
            index = None if features is None else features.to(z.device)
            new = fn(z if index is None else z.index_select(-1, index)).to(z.dtype)
            return space.replace(h, z, new if index is None else z.index_copy(-1, index, new))

    def settle(self) -> None:
        """Recompute what derives from the trained tensors (a subspace's basis), once per optimizer step rather
        than at every use."""

    def project(self) -> None:
        """Put the instance back onto its constraint after an optimizer step (an sae's decoder rows at unit norm)."""

    def anneal(self, epoch: int, epochs: int) -> None:
        """The instance's schedule at `epoch` of `epochs` (a gate's temperature)."""

    def scalar(self, name: str) -> Tensor:
        """The scalar `name` a fit's objective may name (one of `scalars`); a kind refuses any other."""
        raise ValueError(f"a {self.kind!r} featurizer has no scalar {name!r}")

    def size(self) -> int:
        """How many coordinates the instance's space has."""
        return self.method("encode")(self, torch.zeros(1, self.width(self.tensors()))).shape[-1]

    def parameters(self) -> list[Tensor]:
        """The tensors a fit trains."""
        return [getattr(self, name) for name in self.trains]

    def tensors(self) -> dict[str, Tensor]:
        """The tensors its bundle holds."""
        return {name: getattr(self, name).detach() for name in self.saves}


class Writable(Protocol):
    """A kind with coordinates and a way back into the value: one that serves `decode`."""
    def encode(self, h: Tensor) -> Tensor: ...
    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor: ...


def check(kind: str, fields: dict) -> None:
    """Featurizer kind `kind` exists and `fields` are fields of it."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    KINDS[kind].check(fields)


def labels(rows: dict[str, list[dict]], column: str) -> list:
    """A probe's class list: the distinct values of `<dataset>.<column>`, sorted. They sort, and two compare
    as one class only when they are the same value, so they are of one type: all strings, all ints, all floats or
    all booleans (`1`, `1.0` and `true` would otherwise be one class, and `1` and `"yes"` do not sort)."""
    dataset, _, name = column.partition(".")
    values = [row[name] for row in rows[dataset]]
    types: dict[str, tuple[int, Any]] = {}
    for i, v in enumerate(values):
        types.setdefault(type(v).__name__, (i, v))
    if len(types) > 1 or set(types) - {"str", "int", "float", "bool"}:
        shown = ", ".join(f"{t} (row {i} is {v!r})" for t, (i, v) in types.items())
        raise ValueError(f"classes column {column!r} holds {shown}; a probe's classes are one type, all strings, all "
                         "ints, all floats or all booleans: write them in one")
    return sorted(set(values))


def make(kind: str, fields: dict, width: int, bundle: dict | None, seed: int | None, labels: list | None,
         header: dict | None = None, device: str = "cpu") -> Kind:
    """An instance of `kind` from a document's `fields`, its seed defaulting to `seed`, its tensors on `device`.
    It is drawn or loaded on the CPU, so its bits do not depend on the device, and then moved there, so that
    what a fit trains, its optimizer's state and its math live where the model is."""
    supplied = {"width": width, "bundle": bundle, "header": header, "labels": labels}
    taken = inspect.signature(KINDS[kind]).parameters
    given = {**fields, "seed": seed if fields.get("seed") is None else fields["seed"]} if "seed" in taken else fields
    instance = KINDS[kind](**given, **{k: v for k, v in supplied.items() if k in taken})
    for name, value in vars(instance).items():
        if isinstance(value, Tensor):
            setattr(instance, name, value.to(device))
    return instance


# The kinds register in this order, and `describe` and FORMAT.md list them in it.
from clab.featurizers import subspace, gate, pca, sae, linear, lora, bundles  # noqa: E402,F401,I001  (the kinds register in this order)
