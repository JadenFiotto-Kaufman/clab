"""The one computation vocabulary: every op is a plain function over tensors.

`@op(group, ...)` registers the function under its own name and declares it: its group, the axes the input
must carry (`needs`), the axes it removes (`removes`: a tuple, or the name of the argument that lists them),
the axes it appends (`adds`), and, when its value is a bundle a featurizer loads, that featurizer's kind
(`bundle`). An op whose axes and further arguments depend on what it is given declares them as functions
instead (`layout`, `options`), as the `featurizer` op does for its modes. Compile reads the axes to lay out
every value. A parameter's annotation says how a document fills it (`Value`, `Tokens`,
`Featurizer`, or a literal); `engine`, `original` and `dims` are filled by the
run, never by a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is
`"with"`).
"""

ABOUT = """An op runs in three places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a value earlier steps produced;
- inside a read, `"ops": [...]`, on the gathered selection before its name binds;
- inside a write, `"ops": [...]`, on the live selection at the site; what the last op returns is written back.

Inside a read or write an op sees one batch, so it must act per example: one that aggregates over rows is
refused there (make it a step after the forward).

A row a read beyond the prefill did not reach is missing all the way down: in a rectangular value it is NaN,
every metric and edit of it is NaN, `argmax` and `top_k` give it id -1 (`top_k` NaN probabilities and no
token text). Reductions skip it: `mean` is the mean over the rows that have a value, and `pca` is fitted on them.

Ops come in four groups: **metrics** score each example (one number per row and position: `match` is 0 or
1, and the accuracy is a `mean` step after it); **reductions** remove the rows axis; **edits** keep the
value's layout and change its contents; **transforms** turn each example's value into another of a
different layout (logits from a residual stream, tokens from logits, text from ids).

An argument is filled by a document as its type says: `Value` is a named value or a number; `Tokens` a
dataset column, resolved through the value's own vocabulary: one token per row against logits, or one of a
probe's classes against its scores (a metric scores a `vocab_size` or a `classes` axis); `Featurizer` a
featurizer's name; anything else is a literal. Inside a read or write, `featurizer` and `select` are
arguments of every op: the op runs on the value's coordinates in that featurizer's space (on the features
`{"select": {"features": [...]}}` keeps; an operand is laid out on them) and the change is written back as a
delta, so an op that changes no coordinate changes nothing to the bit. The `featurizer` op calls a featurizer
instead, in one of its kind's modes (`encode` gives the coordinates themselves, `apply` the featurizer's
version of the value). `renormalize`'s reference is the value at the site before the step's writes. Inside a
write, an operand may name a read of the same forward at a site the forward reaches before the write's
(`patched.premix` in `patched`): it is that read's value in this batch, at the read's positions.
"""

import inspect
from typing import Any, Callable

import torch
from torch import Tensor

from ... import registry
from ...address import EXAMPLE_AXES, pos_axis
from ...registry import Featurizer, Tokens, Value  # noqa: F401  (ops.Tokens, ops.Value)
from .. import featurizers

# `engine`: the engine running the document. `original`: inside a write, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
# `eos`: the ids the generation `of` came from stops on.
SUPPLIED = ("engine", "original", "dims", "eos")
# A metric scores a vocabulary axis: the tokenizer's, or a probe's classes.
VOCABULARY = ("vocab_size", "classes")

OPS: dict[str, Callable] = {}
GROUPS = ("metrics", "reductions", "edits", "transforms")
DECLARED: dict[str, dict] = {}


def op(group: str, needs: tuple = (), removes: tuple | str = (), adds: tuple = (), bundle: str | None = None,
       layout: Callable | None = None, options: Callable | None = None):
    """Register the decorated function as an op under its own name, with its declaration: its `group`
    (`metrics`, `reductions`, `edits` or `transforms`), its axes, and, for a value a featurizer loads, the
    featurizer kind its `bundle` is for. An op whose result's axes depend on its arguments declares
    `layout(dims, args, kind)` in place of the axes, and one whose `**` arguments do declares `options(args,
    kind)`, the function they go to, whose first two parameters (the featurizer and the value) the op fills;
    `kind` is that of the featurizer the op's `Featurizer` argument names."""
    assert group in GROUPS
    def register(fn: Callable) -> Callable:
        DECLARED[fn.__name__] = {"group": group, "needs": needs, "removes": removes, "adds": adds, "bundle": bundle,
                                 "layout": layout, "options": options}
        return registry.registers(OPS)(fn)
    return register


def parameters(name: str, args: dict | None = None, kind: str | None = None) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones; with `args` and the `kind`
    of its featurizer, also those of the function its `options` choose (an optional thing, `Value | None`, as
    the thing)."""
    found = registry.spelled(OPS[name], SUPPLIED + ("of",))
    fn = options(name, args, kind)
    if fn is not None:
        own = registry.spelled(fn, tuple(inspect.signature(fn).parameters)[:2])
        found.update({k: p.replace(annotation=registry.thing(p.annotation)) for k, p in own.items()})
    return found


def options(name: str, args: dict | None, kind: str | None) -> Callable | None:
    """The function op `name`'s `**` arguments go to, for these arguments and featurizer kind, if it has one."""
    chooses = DECLARED[name]["options"]
    return chooses(args, kind) if chooses is not None and args is not None and kind is not None else None


def featurizer_of(name: str, args: dict) -> Any:
    """Op `name`'s `Featurizer` argument in `args`, if it takes one: a name in a document, an instance in a run."""
    return next((args.get(k) for k, p in parameters(name).items() if p.annotation == Featurizer), None)


def kind(name: str, args: dict) -> str | None:
    """The kind of the featurizer instance bound as op `name`'s `Featurizer` argument, if any."""
    f = featurizer_of(name, args)
    return f.kind if isinstance(f, featurizers.Kind) else None


def check(name: str, args: dict) -> None:
    """Op `name` exists and `args` bind to its signature."""
    if name not in OPS:
        raise ValueError(f"unknown op {name!r}; valid: {', '.join(OPS)}")
    registry.check(OPS[name], args, SUPPLIED + ("of",), repr(name))


def call(name: str, value: Any, args: dict[str, Any], **supplied: Any) -> Any:
    """Run op `name` on `value` with `args` (document spelling, already bound to tensors); an argument the
    op's signature does not name goes to its `**` catch-all as spelled."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    kwargs = {names.get(k, k): v for k, v in args.items()}
    kwargs.update((k, v) for k, v in supplied.items() if k in names and k not in kwargs)
    return fn(value, **kwargs)


def out_dims(name: str, dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`: its `layout`'s, for a featurizer of `kind`,
    when it declares one."""
    axes = DECLARED[name]
    if axes["layout"] is not None:
        return axes["layout"](dims, args, kind)
    for axis in axes["needs"]:
        if not set(axis if isinstance(axis, tuple) else (axis,)) & set(dims):
            raise ValueError(f"{name!r} needs a value with a {' or '.join(map(repr, axis)) if isinstance(axis, tuple) else repr(axis)} "
                             f"axis; this one is {dims}")
    removes = axes["removes"]
    if isinstance(removes, str):
        removes = args.get(removes)
        for axis in removes or []:
            if axis not in dims:
                raise ValueError(f"{name!r} cannot remove {axis!r}; the value is {dims}")
        removes = removes if removes is not None else EXAMPLE_AXES
    return [a for a in dims if a not in removes] + list(axes["adds"])


def select(value: Tensor, component: str, record: dict) -> Tensor:
    """The record's positions and narrowing of `value`, laid out `(rows, pos, ...)`.

    `record["positions"]` is `[rows, width]` token indices; `record["select"]`
    maps a dimension of that layout to the indices kept along it: one list for
    every row, or (`[rows, n]`, on the `key` axis) each row's own.
    """
    x = value.movedim(pos_axis(component), 1)
    x = x[torch.arange(x.shape[0])[:, None], record["positions"]]
    for dim, index in record["select"].items():
        x = take(x, dim, index)
    return x


def take(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` at `index` along `dim`: the same indices on every row, or (`[rows, n]`) each row's own."""
    return x.index_select(dim, index) if index.dim() == 1 else x.gather(dim, per_row(x, dim, index))


def per_row(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Each row's `[rows, n]` indices expanded to `x`'s shape with `n` along `dim`, for gather and scatter."""
    shape = [1] * x.dim()
    shape[0], shape[dim] = index.shape
    size = list(x.shape)
    size[dim] = index.shape[1]
    return index.reshape(shape).expand(size)


def write(value: Tensor, component: str, writes: list[dict], got: dict | None = None) -> Tensor:
    """`value` with `writes` applied in listed order, out of place.

    Each write's ops run on its selection of the current value; `original` is
    the selection as it was before any of these writes. An argument bound to a read of the same forward
    (`{"live": <the read's name>}`) is that read's value in `got`, where the forward put it. An op with a
    `featurizer` (an instance) runs on the selection's coordinates in its space, on the features its `select`
    keeps, and the change is written back (`Kind.within`). A write narrows at most
    one dimension of its selection (compile refuses more). A ragged write's rows are padded to one
    width; only the positions its `keep` marks are written.
    """
    axis = pos_axis(component)
    before = value.movedim(axis, 1)
    x = before.clone()
    rows = torch.arange(x.shape[0])[:, None]
    for w in writes:
        pos = w["positions"]
        current, original = x[rows, pos], before[rows, pos]
        new = current
        for dim, index in w["select"].items():
            new, original = take(new, dim, index), take(original, dim, index)
        for o in w["ops"]:
            f, features = o["featurizer"], o.get("features")
            args = {k: live(got, v) if isinstance(v, dict) and "live" in v else v for k, v in o["args"].items()}
            if f is None:
                new = call(o["op"], new, args, original=original, dims=o["dims"])
                continue
            start = f.encode(original)
            start = start if features is None else start.index_select(-1, features.to(start.device))
            new = f.within(new, lambda z, args=args, o=o, start=start: call(
                o["op"], z, args, original=start, dims=o["dims"]), features)
        for dim, index in w["select"].items():
            new = new.to(current.dtype)
            new = current.index_copy(dim, index, new) if index.dim() == 1 else \
                current.scatter(dim, per_row(new, dim, index), new)
        keep = w.get("keep", torch.ones(pos.shape, dtype=torch.bool))
        x[rows.expand_as(pos)[keep], pos[keep]] = new.to(x.dtype)[keep]
    return x.movedim(1, axis)


def live(got: dict | None, operand: dict) -> Tensor:
    """A write's operand that names a read of its own forward: that read's value, which the engine made before
    it reached the write."""
    if operand["live"] not in (got or {}):
        raise ValueError(f"operand {operand['live']!r} is a read of this forward the engine had not made when it "
                         "reached the write; it reads it at a later site")
    return (got or {})[operand["live"]]


# The ops register in this order, and the tables (`describe`, FORMAT.md, schema.json) list them in it.
from .edits import add, normalize, project_out, renormalize, scale, sub, swap  # noqa: E402,F401
from .transforms import featurizer, readout  # noqa: E402,F401
from .reductions import mean, pca  # noqa: E402,F401
from .metrics import (argmax, cross_entropy, decode, js, kl, logit_diff, match, token_logit,  # noqa: E402,F401
                      token_prob, top_k)
