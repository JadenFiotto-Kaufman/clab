"""Featurizer kinds: the fields each kind takes, what an instance is, and how a value moves in and out of its space.

A featurizer is a space a read or write happens in, or a function on a value. Its kind's signature is its
contract: the document fields it takes, with `width` (the feature width of the sites it is used at), `bundle`
and `header` (the tensors and header of the bundle its `file_path` names) supplied by the run, and for a probe
`labels` (its class list). An instance holds plain tensors, never an `nn.Module`, made where the model is: a
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
encodes of it saw in that run of the body, which the instance keeps while a fit trains it). Its math runs on
the value's device and dtype, with autocast off: a model served in bfloat16 under autocast leaves a float32
featurizer's float32 math alone.

A bundle is a safetensors file: an instance's `saves` tensors (the first laid out `[width, ...]`) and a header
saying what made them: the kind and its `header` fields, the model, the site it was used at (`component`,
`layer`), its `width` and the document's name. Both a `pca` op's value and a fit's trained featurizer are
saved as one. A document that loads one is checked against it key by key: the tensors and the kind, its
fields and the model when the plan compiles, the site at every use, and `width` against the model's size
before any forward. A file is clab's when its header's `format` is `clab/...`; one that carries clab's header
keys under another format is refused. Any other file is a published one, its kind's to read or refuse: an
`sae` reads one (SAELens, Gemma Scope), whose shapes alone are checked, against each other and the site's
width: nothing says what model or site it was made at, so neither is checked.

Each kind is one class, in a module of its own, registered under its name in lower case. It declares as class
attributes what its bundle holds (`saves`), what a fit trains (`trains`), the axis its coordinates are on
(`axis`), the fields its bundle's header records (`header`), the fields one drawn without a `file_path` needs
(`drawn`) and the scalars a fit's objective may name (`scalars`); its constructor makes an instance. Its modes
are the methods of the same names, `encode(h)` and `apply(h, ...)`, a kind defining only those it has; the
rest is the protocol `Kind` declares, with a default for what a kind need not have.
"""

import functools
import inspect
from contextlib import nullcontext
from typing import Any, Callable, ClassVar, Protocol, cast

import torch
from torch import Tensor

from ... import registry

SUPPLIED = ("width", "bundle", "header", "labels")
FORMAT = "clab/1"
KINDS: dict[str, type["Kind"]] = {}

# The two modes the `featurizer` op calls a featurizer in: what each produces, and the layout of what it
# produces on a value laid out `dims`, for a kind whose coordinates are on `axis`. A kind has a mode when it
# defines the method of that name.
MODES: dict[str, dict] = {
    "encode": {"produces": "its coordinates, the last axis becoming {axis}",
               "layout": lambda axis, dims: dims[:-1] + [axis]},
    "apply": {"produces": "its version of the value, in the value's own layout",
              "layout": lambda axis, dims: list(dims)},
}


def unautocast(h: Tensor):
    """Autocast off on `h`'s device, for the length of a `with`: a featurizer's math stays in its own dtype."""
    kind = h.device.type
    return torch.autocast(kind, enabled=False) if torch.amp.is_autocast_available(kind) else nullcontext()


def exact(math: Callable) -> Callable:
    """A kind's method on a value `h` (a mode, `replace`), run with autocast off."""
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
    kind: ClassVar[str]
    # Set by the fit that trains the instance: whether this run of its body trains it (a gate is soft then),
    # and every value the body's encodes of it saw, beside its coordinates, for its scalars.
    training: bool = False
    seen: list[tuple[Tensor, Tensor]] | None = None

    def __init_subclass__(cls) -> None:
        cls.kind = cls.__name__.lower()
        for name in (*MODES, "replace"):
            if name in vars(cls):
                setattr(cls, name, exact(vars(cls)[name]))
        registry.registers(KINDS, cls.kind)(cls)

    # -- what the document and the plan see ------------------------------------------------------------------

    @classmethod
    def check(cls, fields: dict) -> None:
        """`fields` bind to the kind's signature."""
        registry.check(cls, fields, SUPPLIED, repr(cls.kind))

    @classmethod
    def modes(cls) -> list[str]:
        """The modes the kind has."""
        return [m for m in MODES if hasattr(cls, m)]

    @classmethod
    def mode(cls, name: str) -> Callable:
        """The kind's method for mode `name`; a mode the kind lacks is refused by name."""
        if name not in cls.modes():
            raise ValueError(f"kind {cls.kind!r} has no mode {name!r}; its modes: {', '.join(cls.modes())}")
        return getattr(cls, name)

    @classmethod
    def writable(cls) -> bool:
        """Whether an edit op runs in the kind's space: it has coordinates (`encode`) and a way back (`replace`)."""
        return hasattr(cls, "encode") and hasattr(cls, "replace")

    @classmethod
    def layout(cls, mode: str, dims: list[str]) -> list[str]:
        """The layout of what mode `mode` produces on a value laid out `dims`."""
        return MODES[mode]["layout"](cls.axis, dims)

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

    def call(self, mode: str, h: Tensor, options: dict) -> Tensor:
        """The instance called in mode `mode` on `h`, with the mode's own arguments (`options`, by document
        spelling). An encode is kept beside `h` while a fit watches the instance (`seen`)."""
        fn = self.mode(mode)
        names = {k: p.name for k, p in registry.spelled(fn, ("self", "h")).items()}
        out = fn(self, h, **{names[k]: v for k, v in options.items()})
        if mode == "encode" and self.seen is not None:
            self.seen.append((h, out))
        return out

    def within(self, h: Tensor, fn: Callable[[Tensor], Tensor], features: Tensor | None) -> Tensor:
        """`h` after `fn` ran on its coordinates, on the `features` it keeps (every one when None), and the change
        was written back; `new == z` gives `h` to the bit."""
        space = cast(Writable, self)         # compile lets an edit run only in a `writable` kind's space
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
        return self.mode("encode")(self, torch.zeros(1, self.width(self.tensors()))).shape[-1]

    def parameters(self) -> list[Tensor]:
        """The tensors a fit trains."""
        return [getattr(self, name) for name in self.trains]

    def tensors(self) -> dict[str, Tensor]:
        """The tensors its bundle holds."""
        return {name: getattr(self, name).detach() for name in self.saves}


class Writable(Protocol):
    """A kind with coordinates and a way back into the value (`Kind.writable`)."""
    def encode(self, h: Tensor) -> Tensor: ...
    def replace(self, h: Tensor, z: Tensor, new: Tensor) -> Tensor: ...


def check(kind: str, fields: dict) -> None:
    """Featurizer kind `kind` exists and `fields` are fields of it."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    KINDS[kind].check(fields)


def labels(rows: dict[str, list[dict]], column: str) -> list:
    """A probe's class list: the distinct values of `<dataset>.<column>`, sorted."""
    dataset, _, name = column.partition(".")
    return sorted({row[name] for row in rows[dataset]})


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
from . import subspace, gate, pca, sae, linear, lora, bundles  # noqa: E402,F401
