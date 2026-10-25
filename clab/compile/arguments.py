"""An op's arguments resolved against what compile knows: the names it references (a value, a column, a
featurizer), the dataset its rows align to, the layout its operands must have, and the layout of its result."""

import inspect
from dataclasses import replace

from clab import featurizers, ops
from clab.compile.answers import column_ref, text_ref
from clab.compile.state import Compiling, Known
from clab.format import address, data
from clab.plan import Op, Plan, arguments, fail, kind_of


def narrowed_readout(what: str) -> str:
    return (f"readout projects the whole residual stream through the model's final norm and head, and {what} keeps "
            "only some of hidden_size, which the norm cannot take; read the site whole, readout, and narrow what "
            "the logits give")


def space_of(cx: Compiling, encodes: list[Op]) -> list[dict]:
    """The coordinates a value is in: the encodes open around it, outermost first, each its featurizer (by the
    name it resolves to here: inside a fit's body `rot` and `<fit>.rot` are one) and `select`."""
    return [{k: cx.aliases.get(v, v) if k == "featurizer" else v for k, v in arguments(o).items()
             if k in ("featurizer", "select")} for o in encodes]


def said(space: list[dict]) -> str:
    """A value's coordinates as a message names them."""
    if not space:
        return "the site's own layout"
    return "the coordinates of " + " then ".join(repr(s["featurizer"]) + (f" (select {s['select']})" if "select" in s
                                                                       else "") for s in space)


def called(cx: Compiling, plan: Plan, path: str, name: str, args: dict) -> str | None:
    """The kind of the featurizer op `name` names, checked to exist and to serve the op, and an `apply`'s `with`
    to be a name its kind takes; None when the op takes no featurizer."""
    ref = args.get("featurizer")
    if ref is None:
        return None
    featurizer_ref(cx, ref, f"{path}.featurizer")
    kind = kind_of(plan, ref)
    serves = featurizers.KINDS[kind].ops()
    if name not in serves:
        fail(f"{path}.featurizer", f"{ref!r} is a {kind!r} featurizer, which has no {name}: it serves "
             f"{', '.join(serves)}" + ("; it has no way back into the value" if name == "decode" else ""))
    if name == "apply" and args.get("with") is not None:
        if "with_" not in inspect.signature(featurizers.KINDS[kind].method("apply")).parameters:
            fail(f"{path}.with", f"a {kind}'s apply takes no 'with', an adapter's input read at another site (a lora's)")
        if not isinstance(args["with"], str):
            fail(f"{path}.with", f"{args['with']!r} is not a name; 'with' names a value read at the positions the op "
                 "acts on")
    return kind


def references(cx: Compiling, path: str, name: str, args: dict, classes: bool) -> dict[str, Known]:
    """Op `name`'s arguments that name a value or a column, each resolved (`Known`, with `name` as written). A
    column read as text or scored against the vocabulary holds text on every row; one scored against a probe's
    `classes` holds its labels, whatever they are. A featurizer argument is checked to exist."""
    found = {}
    for key, p in ops.parameters(name).items():
        value, at = args.get(key), f"{path}.{key}"
        if p.annotation == ops.Value and isinstance(value, str):
            found[key] = replace(value_ref(cx, value, at), name=value)
        elif p.annotation in (ops.Tokens, ops.Text) and value is not None:
            ref = data.column_of(value)
            dataset = column_ref(cx, ref[0] if ref else value, at, None) if classes and p.annotation == ops.Tokens \
                else text_ref(cx, value, at, None)
            found[key] = Known(["rows"], dataset, "column", name=ref[0] if ref else value)
        elif p.annotation == ops.Featurizer and value is not None:
            featurizer_ref(cx, value, at)
    return found


def aligned(path: str, of: tuple[str | None, str], found: dict[str, Known]) -> str | None:
    """The one dataset an op's rows align to: that of the value it acts on (`of`: its dataset, and how to name
    it), of its operands with a rows axis, and of its columns."""
    datasets = {of[0]: of[1]} if of[0] else {}
    for info in found.values():
        if info.data is not None and "rows" in info.dims:
            datasets.setdefault(info.data, repr(info.name))
    if len(datasets) > 1:
        (a, x), (b, y) = list(datasets.items())[:2]
        fail(path, f"{x} has a row per example of {a!r} and {y} of {b!r}; an op takes one dataset's rows")
    return next(iter(datasets), None)


def paired(path: str, name: str, dims: list[str], width: int | None, found: dict[str, Known],
           step: bool = False) -> None:
    """A metric's operand (`kl`'s `against`, `cosine`'s `with`), and in a step an edit's (inside an entry
    `operands` checks those), is laid out as the value the op acts on, as many positions wide, or is one vector of
    its features, each at one layer: a `layers` axis of either aligns by label (`acts_at`). One without a static
    width lines up with nothing."""
    group = ops.DECLARED[name]["group"]
    if group != "metrics" and not (step and group == "edits"):
        return
    dims = address.unstacked(dims)
    one = [a for a in dims if a not in address.EXAMPLE_AXES]
    verb = "scores" if group == "metrics" else "acts on"
    for key, info in found.items():
        if info.kind != "value" or (name == "apply" and key == "with"):
            continue
        if "pos" in info.dims and info.width is None:
            fail(f"{path}.{key}", f"{info.name!r} holds a different number of positions on each row, which pair "
                 f"with no other value's; read both at a static index (pos -1, or {{'index': '-3:'}})")
        if address.unstacked(info.dims) not in (dims, one):
            fail(f"{path}.{key}", f"{info.name!r} is laid out {info.dims}; {name!r} {verb} a value laid out "
                 f"{dims}, so its {key!r} is laid out so too, or is one vector {one} (either with a layers axis "
                 "first)")
        if "pos" in dims and address.unstacked(info.dims) == dims and info.width != width:
            fail(f"{path}.{key}", f"{info.name!r} holds {info.width} position(s) per row and the value "
                 f"{name!r} {verb} holds {width}; position i {'scores against' if group == 'metrics' else 'meets'} "
                 "position i, so both are read at as many (a pos axis does not broadcast)")


def layout(path: str, name: str, args: dict, dims: list[str], where: str, kind: str | None) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims`. Inside an entry (`where`) the op sees one
    batch, so it may not aggregate over rows; `original` is only an entry's."""
    if where != "entry" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside an entry")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args}, kind)
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and "rows" in dims and "rows" not in out:
        fail(path, f"{name!r} aggregates over rows, and inside an entry it sees one batch; "
             "make it a step after the forward")
    return out


def value_ref(cx: Compiling, value: str, path: str) -> Known:
    """A named value produced before this point, noted as referenced and used."""
    info = cx.names.get(value)
    if info is not None and info.kind == "featurizer":
        fail(path, f"{value!r} is a featurizer, not a value; an op reaches it ({{'op': 'encode', 'featurizer': "
             f"{value!r}}} gives a value's coordinates, 'apply' its version of the value), and the value that entry "
             "binds is an operand")
    if info is None or info.kind != "value":
        known = sorted(k for k, v in cx.names.items() if v.kind == "value")
        fail(path, f"{value!r} names no value produced before this step; known: {', '.join(known) or 'none'}")
    cx.refs.append(value)
    cx.used.add(value)
    return info


def featurizer_ref(cx: Compiling, value: object, path: str) -> None:
    """A declared featurizer's name, noted as referenced."""
    if not (isinstance(value, str) and value in cx.names and cx.names[value].kind == "featurizer"):
        known = sorted(k for k, v in cx.names.items() if v.kind == "featurizer")
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")
    cx.refs.append(value)
