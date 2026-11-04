"""Featurizer kinds: the fields each kind takes, what an instance is, and how a value moves in and out of its space.

A featurizer is a space an entry's ops encode into, or a function on a value. Its kind's signature is its
contract: the document fields it takes, with `width` (the feature width of the sites it is used at), `bundle`
and `header` (the tensors and header of the bundle its `file_path` names) supplied by the run, for a probe
`labels` (its class list), and for a featurizer over named axes `kept` (the indices it keeps on each of them: a
gate's units, an affine's layers).
An instance holds plain tensors, never an `nn.Module`, made where the model is: a declared featurizer before the
run's first step (one with a field that names a value, a constant's `init`, once the step that made the value has
run), one a fit trains at the fit's start. It is drawn from its `seed` (else the seed of the fit that
trains it, else 0), or loaded from its bundle, which compile reads into the plan; a loaded one is fixed. Three
ops reach it: `encode` gives
its coordinates of a value `h` (the last axis becomes the kind's axis: `features`, or a probe's `classes`),
`decode` writes a change of them back into `h` as a delta (`replace`), so an op between them that changes
nothing changes nothing to the bit, and a sparse autoencoder's error term is kept exactly; and `apply` gives
its version of `h` in `h`'s own layout (a subspace's projection, a dictionary's reconstruction, a gate's
masked value or, given `with`, its interpolation toward that value, an adapter's adapted value, an affine map's
`h + h W + b`). A fit's objective may name a featurizer's scalar as `<featurizer>.<scalar>` (a gate's
`mask` is the mean of its soft mask; an sae's `reconstruction` and `l1` are over the coordinates the body read of
it in that run of the body, its `encode`s no `decode` closes, which the instance keeps while a fit trains it). Its
math runs on the value's device and dtype, with autocast off: a model served in bfloat16 under autocast leaves a
float32 featurizer's float32 math alone.

A bundle is a safetensors file: an instance's `saves` tensors (the first laid out `[width, ...]`, but for a featurizer
over axes: a gate's `theta` on the axes it is over, an affine's `W` `[layers, width, width]`) and a header
saying what made them: the kind and its `header` fields, the model, the site it was used at (`component`,
`layer`: the one layer, or the list of them, it was made at, counted from the first, and, where the site
selects, `select`: the indices it kept on each axis it selects, a JSON object), its `width` and the document's
name, and for a gate over named axes the indices it keeps on each of them but `layers` (`kept.<axis>`). Both a
`pca` op's value and a fit's trained featurizer are saved as one. A document that loads one is checked against
it key by key: the tensors and the kind, its fields and the model when the plan compiles, the site at every use
(each layer it picks one that `layer` names, its selection the one `select` records, none for a header without
one, which was made at the whole component, and a gate's selection on each axis it is over the one
`kept.<axis>` names), and `width` against the model's size before any forward. A file is clab's when its
header's `format` is `clab/...`; one that carries clab's header keys under another format is refused. Any other
file is a published one, its kind's to read or refuse: an `sae` reads one (SAELens, Gemma Scope).

Each kind is one class, in a module of its own, registered under its name in lower case. It declares as class
attributes what its bundle holds (`saves`), what a fit trains (`trains`), the axis its coordinates are on
(`axis`), the fields its bundle's header records (`header`), the fields one drawn without a `file_path` needs
(`drawn`), what a header field left unstated is where the kind rather than its signature decides it (`unstated`:
an sae's `activation` and `centered`), the scalars a fit's objective may name (`scalars`; those of them over what
the body's encodes of it saw, `over_encodes`, which compile refuses on a body with no open encode of it) and the
fields that count directions of the value, so are at most its width (`counts`), whether its `apply`'s `with` is
its input (`adapts`: an adapter's) or an operand of the value (a gate's), and whether its `apply` is elementwise
(`elementwise`: a gate's), which keeps a missing element of the value missing where any other kind loses its
whole vector, and whether `over` stacks its tensors, one slice per index of the axes it names, its width the
value's last axis (`stacked`: an affine's), or lays its units out on those axes (a gate's); its constructor
makes an instance. The ops it serves are its methods `encode(h)`,
`replace(h, z, new)` (what `decode` runs, so a kind serves `decode` when it has both) and `apply(h, ...)`, a kind
defining only those it has; the rest is the protocol `Kind` declares, with a default for what a kind need not have.
"""

import functools
from contextlib import nullcontext
from typing import Any, Callable, ClassVar

import torch
from torch import Tensor

from clab.format import registry

SUPPLIED = ("width", "bundle", "header", "labels", "kept")
FORMAT = "clab/1"
KINDS: dict[str, type["Kind"]] = {}

# The ops that reach a featurizer: the method of the kind's each runs, and what it produces on a value, for a
# kind whose coordinates are on `axis`.
OPS: dict[str, dict[str, str]] = {
    "encode": {"method": "encode", "produces": "gives its coordinates, the last axis becoming {axis}"},
    "decode": {"method": "replace", "produces": "writes a change of its coordinates back into the value"},
    "apply": {"method": "apply", "produces": "gives its version of the value, in the value's own layout"},
}


def unautocast(h: Tensor) -> torch.autocast | nullcontext[None]:
    """Autocast off on `h`'s device, for the length of a `with`: a featurizer's math stays in its own dtype."""
    kind = h.device.type
    return torch.autocast(kind, enabled=False) if torch.amp.is_autocast_available(kind) else nullcontext()


def exact(math: Callable[..., Tensor]) -> Callable[..., Tensor]:
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
    unstated: ClassVar[dict[str, Any]] = {}
    scalars: ClassVar[tuple[str, ...]] = ()
    over_encodes: ClassVar[tuple[str, ...]] = ()
    counts: ClassVar[tuple[str, ...]] = ()
    # Whether `apply`'s `with` is the kind's input, which may be read at another site of the same forward (an
    # adapter's): held to the value's rows and positions alone, and recorded in its bundle's header. Otherwise a
    # `with` is an operand of the value under Ops' one rule, as `swap`'s is (a gate's, which it interpolates toward).
    adapts: ClassVar[bool] = False
    # Whether `apply` is elementwise, each element of its result made from that element of `h` (and of its
    # operands) alone (a gate's), so that it keeps NaN exactly where `h` had it (`call`).
    elementwise: ClassVar[bool] = False
    # Whether `over` stacks the kind's tensors, one slice per index the axes it names keep, each meeting the value's
    # last axis as the kind without `over` does (an affine's one map per layer), so that its width is that axis's;
    # otherwise `over` lays its units out on those axes, and its width is how many units it has (a gate's).
    stacked: ClassVar[bool] = False
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
    def check(cls, fields: dict[str, Any]) -> None:
        """`fields` bind to the kind's signature."""
        registry.check(cls, fields, SUPPLIED, repr(cls.kind))

    @classmethod
    def ops(cls) -> list[str]:
        """The ops the kind serves: those whose method it defines."""
        return [op for op, o in OPS.items() if hasattr(cls, o["method"])]

    @classmethod
    def method(cls, op: str) -> Callable[..., Tensor]:
        """The kind's method op `op` runs: compile refuses an op the kind does not serve (`ops`)."""
        return getattr(cls, OPS[op]["method"])

    @classmethod
    def width(cls, tensors: dict[str, Tensor]) -> int:
        """The feature width of a bundle's tensors: the first axis of the first one it saves (a published file's, which
        no `over` lays out on axes of its own)."""
        return tensors[cls.saves[0]].shape[0]

    @classmethod
    def published(cls, tensors: dict[str, Tensor], fields: dict[str, Any]) -> dict[str, Tensor]:
        """The tensors of a published file (one with no clab header), as an instance loads them; a kind that
        reads none refuses."""
        raise ValueError(f"{fields['file_path']} has no clab header; a {cls.kind} bundle is one clab wrote")

    # -- what the run calls ----------------------------------------------------------------------------------

    def call(self, op: str, h: Tensor, options: dict[str, Any]) -> Tensor:
        """Op `op` (`encode` or `apply`) of the instance on `h`, with the method's own arguments (`options`, by
        parameter name). A missing vector of `h` (NaN), and the same vector of an option laid out on its
        rows and positions, goes in as zeros and comes out missing, so the math, and a fit's gradients through
        it, see none of it: a kind's math reduces over the last axis (a projection, a dictionary, a probe), where
        one NaN would poison the whole vector. An `apply` that is elementwise (`elementwise`: a gate's) treats a
        missing element so, with the same element of an option laid out on the value's axes, and keeps NaN
        exactly where `h` had it: on a `key` axis, NaN past a shorter row's own keys, every vector of that row
        holds one, and would otherwise go back into the model missing whole. An encode is kept beside `h` while a
        fit watches the instance (`seen`). The masks apply whether or not anything is missing, an option masked where
        `h` is: one that marks nothing changes no bit, of the values or of a gradient through them, and asking the
        device whether it marks anything would make the host wait for it at every call. An integer result (a
        probe's math on ids) holds no NaN to mark a missing vector with, and comes out as the math gives it."""
        fn = self.method(op)
        each = op == "apply" and self.elementwise
        gone = h.isnan() if each else h.isnan().any(-1, keepdim=True)
        x = h.masked_fill(gone, 0.0)
        if each:            # an operand on the value's axes, a singleton on each it leaves out
            options = {k: v.to(h.device).where(~gone, 0.0) if isinstance(v, Tensor) and v.dim() == h.dim() else v
                       for k, v in options.items()}
        else:
            options = {k: v.to(h.device).masked_fill(
                           gone.reshape(*gone.shape[:-1], *[1] * (v.dim() - gone.dim() + 1)), 0.0)
                       if isinstance(v, Tensor) and v.shape[:gone.dim() - 1] == gone.shape[:-1] else v
                       for k, v in options.items()}
        out = fn(self, x, **options)
        out = out.masked_fill(gone, float("nan")) if out.is_floating_point() else out
        if op == "encode" and self.seen is not None:
            self.seen.append((h, out))
        return out

    def within(self, h: Tensor, fn: Callable[[Tensor], Tensor], features: Tensor | None) -> Tensor:
        """`h` after `fn` ran on its coordinates, on the `features` it keeps (every one when None), and the change
        was written back: an `encode` and the `decode` that closes it, `fn` the ops between; `new == z` gives `h`
        to the bit."""
        encode, replace = self.method("encode"), self.method("decode")
        with unautocast(h):
            z = encode(self, h)
            index = None if features is None else features.to(z.device)
            new = fn(z if index is None else z.index_select(-1, index)).to(z.dtype)
            return replace(self, h, z, new if index is None else z.index_copy(-1, index, new))

    def settle(self) -> None:
        """Recompute what derives from the trained tensors (a subspace's basis), once per optimizer step rather
        than at every use."""

    def project(self) -> None:
        """Put the instance back onto its constraint after an optimizer step (an sae's decoder rows at unit norm)."""

    def anneal(self, epoch: int, epochs: int) -> None:
        """The instance's schedule at `epoch` of `epochs` (a gate's temperature)."""

    def scalar(self, name: str) -> Tensor:
        """The scalar `name` a fit's objective may name: one of `scalars`, as compile checks."""
        raise ValueError(f"a {self.kind!r} featurizer has no scalar {name!r}")

    def size(self) -> int:
        """How many coordinates the instance's space has: the shape of its encode of one zero vector made where
        its tensors are, in their dtype, so that no tensor of it moves (an SAE's encoder on the GPU stays there)."""
        first = self.tensors()[self.saves[0]]
        return self.method("encode")(self, first.new_zeros(1, first.shape[0])).shape[-1]

    def tensors(self) -> dict[str, Tensor]:
        """The tensors its bundle holds."""
        return {name: getattr(self, name).detach() for name in self.saves}


def check(kind: str, fields: dict[str, Any]) -> None:
    """Featurizer kind `kind` exists and `fields` are fields of it."""
    if kind not in KINDS:
        raise ValueError(f"unknown featurizer kind {kind!r}; valid: {', '.join(KINDS)}")
    KINDS[kind].check(fields)


def labels(rows: dict[str, list[dict[str, Any]]], column: str) -> list:
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


def make(kind: str, fields: dict[str, Any], width: int, bundle: dict[str, Tensor] | None, seed: int | None,
         labels: list | None, header: dict[str, str] | None = None, device: str = "cpu",
         kept: dict[str, list[int]] | None = None) -> Kind:
    """An instance of `kind` from a document's `fields`, its seed defaulting to `seed`, its tensors on `device`.
    It is drawn or loaded on the CPU, so its bits do not depend on the device, and then moved there, so that
    what a fit trains, its optimizer's state and its math live where the model is."""
    supplied = {"width": width, "bundle": bundle, "header": header, "labels": labels, "kept": kept}
    taken = registry.signature(KINDS[kind]).parameters
    given = {**fields, "seed": seed if fields.get("seed") is None else fields["seed"]} if "seed" in taken else fields
    instance = KINDS[kind](**given, **{k: v for k, v in supplied.items() if k in taken})
    for name, value in vars(instance).items():
        if isinstance(value, Tensor):
            setattr(instance, name, value.to(device))
    return instance


# The kinds register in this order, and `describe` and FORMAT.md list them in it.
from clab.featurizers import subspace, gate, pca, sae, linear, lora, affine, constant, bundles  # noqa: E402,F401,I001
