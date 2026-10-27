"""An op's arguments resolved against the plan's own records: the names it references (a value, a column, a
featurizer), the layout of its result, and the one rule an operand keeps against the value it meets (`mismatch`)."""

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from clab import featurizers, ops
from clab.compile.answers import column_ref, text_ref
from clab.compile.state import Compiling
from clab.format import address, data
from clab.format.sweeps import points
from clab.plan import Labelled, Op, OpStep, Plan, arguments, fail, kind_of, produced, record_of


def narrowed_readout(what: str) -> str:
    return (f"readout projects the whole residual stream through the model's final norm and head, and {what} keeps "
            "only some of hidden_size, which the norm cannot take; read the site whole, readout, and narrow what "
            "the logits give")


def space_of(aliases: dict[str, str], encodes: list[Op] | list[OpStep]) -> list[dict]:
    """The coordinates a value is in: the encodes open around it, outermost first (an entry's ops, or an encode
    step), each its featurizer (by the name it resolves to here: inside a fit's body `rot` and `<fit>.rot` are
    one, `aliases`) and `select`."""
    return [{k: aliases.get(v, v) if k == "featurizer" else v for k, v in arguments(o).items()
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


def references(name: str, args: dict, resolve: Callable[[str, Any, inspect.Parameter], Labelled | None]
               ) -> dict[str, tuple[str, Labelled]]:
    """Op `name`'s arguments that name a thing the run binds (a `Value`, a column scored or read as `Tokens` or
    `Text`, a `Featurizer`), each as written and resolved by `resolve(key, written, parameter)` to the record of
    its labelled layout (a column's: a row per example of its dataset; None for a featurizer, which is no
    value): the one loop over an op's named arguments, which compile runs with `resolver` and preflight with
    `plan.record_of`."""
    found = {}
    for key, p in ops.parameters(name).items():
        value = args.get(key)
        named = p.annotation in (ops.Tokens, ops.Text, ops.Featurizer) or p.annotation == ops.Value and isinstance(value, str)
        record = resolve(key, value, p) if named and value is not None else None
        if record is not None:
            ref = data.column_of(value) if p.annotation != ops.Value else None
            found[key] = (ref[0] if ref else str(value), record)
    return found


def resolver(cx: Compiling, path: str, classes: bool, at: dict[str, int],
             live: Mapping[str, Labelled] | None = None) -> Callable[[str, Any, inspect.Parameter], Labelled | None]:
    """How compile resolves an op's named arguments (`references`) at `path`, at the consumer's point `at`: a value
    to its record (`value_of`, an entry of the forward being compiled among `live`), a column to a row per example
    of its dataset once it is checked to hold text on every row, or, scored against a probe's `classes`, its
    labels, whatever they are (`classes`), and a featurizer to nothing, once it is checked to exist."""
    def resolve(key: str, value: Any, p: inspect.Parameter) -> Labelled | None:
        where = f"{path}.{key}"
        if p.annotation == ops.Value:
            return value_of(cx, value, where, at, live)
        if p.annotation == ops.Featurizer:
            featurizer_ref(cx, value, where)
            return None
        ref = data.column_of(value)
        dataset = column_ref(cx, ref[0] if ref else value, where, None) if classes and p.annotation == ops.Tokens \
            else text_ref(cx, value, where, None)
        return Labelled(dims=["rows"], dataset=dataset)
    return resolve


def layout(path: str, name: str, args: dict, dims: list[str], where: str, kind: str | None) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims`. Inside an entry (`where`: `write` up to the
    entry's last edit, where the op acts on the live value, else `entry`) the op sees one batch, so it may not
    aggregate over rows: a reduction of a read is a step after the forward, and one a write would write back is a
    step before it, written in; `original` is only an entry's."""
    if where == "step" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside an entry")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args}, kind)
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and "rows" in dims and "rows" not in out:
        fail(path, f"{name!r} aggregates over rows, and inside an entry it sees one batch; " + (
            "this entry writes it back, so make it an op step before the forward, over a read of the rows it "
            f"aggregates, and write that in ({{'op': 'swap', 'with': '<that step>'}})" if where == "write" else
            "make it a step after the forward"))
    return out


def carried(of: Labelled, dims: list[str]) -> tuple[list[int] | None, dict[str, list[int]], int | None]:
    """What an op's value keeps of the labels of the value it is over: its layers, unless the op removed a
    `layers` axis (then it is from no layer), the indices kept on each selected axis it still has, and its width
    where it still has a `pos` axis."""
    return (None if "layers" in (of.dims or []) and "layers" not in dims else of.layers,
            {a: v for a, v in of.kept.items() if a in dims}, of.width if "pos" in dims else None)


def trained(cx: Compiling, thing: str, member: str) -> bool:
    """Whether `<thing>.<member>` is a featurizer a compiled fit trains."""
    step = points(cx.steps[thing])[0] if thing in cx.steps else None
    return step is not None and step.kind == "fit" and member in step.train


def value_of(cx: Compiling, ref: str, path: str, at: dict[str, int],
             live: Mapping[str, Labelled] | None = None) -> Labelled:
    """The record of a named value produced before this point, at the consumer's point `at` (an entry of the
    forward being compiled, among `live`, as it is), noted as used."""
    thing, _, member = ref.partition(".")
    if live and ref in live:
        cx.used.add(ref)
        return live[ref]
    if thing in cx.featurizers or ref in cx.aliases.values() or trained(cx, thing, member):
        fail(path, f"{ref!r} is a featurizer, not a value; an op reaches it ({{'op': 'encode', 'featurizer': "
             f"{ref!r}}} gives a value's coordinates, 'apply' its version of the value), and the value that entry "
             "binds is an operand")
    known = produced(cx.steps, cx.featurizers)
    if ref not in known:
        fail(path, f"{ref!r} names no value produced before this step; known: {', '.join(sorted(known)) or 'none'}")
    cx.used.add(ref)
    return record_of(cx.steps, ref, at)


def featurizer_ref(cx: Compiling, value: object, path: str) -> None:
    """A declared featurizer's name, or one a fit trains (`<fit>.<featurizer>`, and inside its body its bare name)."""
    thing, _, member = str(value).partition(".")
    if not (isinstance(value, str) and (value in cx.featurizers or value in cx.aliases.values()
                                        or trained(cx, thing, member))):
        known = sorted(set(cx.featurizers) | set(cx.aliases.values())
                       | {f"{n}.{f}" for n, e in cx.steps.items() for s in points(e)[:1] if s.kind == "fit"
                          for f in s.train})
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")


# -- the one rule an operand keeps ---------------------------------------------------------------------


def narrowed(ours: Labelled, name: str, key: str, written: str) -> str | None:
    """Why the column `written`, argument `key` of op `name`, scored as `Tokens`, cannot score the value laid out
    `ours`: its vocabulary axis is narrowed, by a site's `select` on `vocab_size` (`kept`) or a probe's `encode`
    with a `select` (`space`), and the column resolves to ids of the whole vocabulary, where index i of the
    selection is not id i; None where the axis is whole, or the argument is no column. Run from `mismatch`, so
    compile checks it inside an entry and as a step alike."""
    if ops.parameters(name)[key].annotation != ops.Tokens:
        return None
    axis = next((a for a in ops.VOCABULARY if a in (ours.dims or []) and (a in ours.kept or
                                                                            ours.space and "select" in ours.space[-1])), None)
    if axis is None:
        return None
    kept = f"{len(ours.kept[axis])} of its {axis} (select)" if ours.kept.get(axis) else f"some of its {axis} (select)"
    return (f"the value {name!r} scores keeps {kept}, and it scores {written!r} by its id over the whole vocabulary, "
            "where index i of the selection is not id i: score the whole vocabulary (read it without the select, and "
            "narrow what the metric gives), or rank within the selection by index (argmax or top_k over it)")


def mismatch(ours: Labelled, theirs: Labelled, name: str, key: str, written: str, mine: str, write: bool,
             forward: bool) -> str | None:
    """Why the value `written` names, laid out `theirs`, cannot be operand `key` of op `name` on the value `mine`
    names, laid out `ours`, at one point; None where it can. The rule is Ops' one rule (`ops.ABOUT`): an operand
    may leave out any axis of the value it meets, which it then broadcasts over; every axis it has is the value's,
    in the value's order, with the same label (`layers` by model layer, `rows` by dataset, `pos` by the read whose
    positions both hold or by `width`, a selected axis by the indices kept in order, `features` by the encodes it is
    in); a `layers` axis it has covers the layers the op acts at and is aligned by id. A number broadcasts in any
    coordinates; an adapter's `with` is compared on its rows and positions alone. `write` says the op is one of an
    entry's up to its last edit (what the messages call the value), `forward` that the operand is an entry of the
    value's own forward (a ragged pair there is compared row by row by `check_operands`, once preflight has the
    rows). A label preflight has yet to resolve (`[]`) is compared then. The coordinates are compared first, since
    two values in different coordinates differ in their last axis too, and that is the reason. A column scored as
    `Tokens` is first held to a whole vocabulary axis (`narrowed`)."""
    if why := narrowed(ours, name, key, written):
        return why
    group = ops.DECLARED[name]["group"]
    verb = "scores" if group == "metrics" else "acts on"
    whose = "this write's selection" if write else "the value it acts on"
    adapts = name == "apply" and key == "with"
    own_axes, their_axes = ours.dims or [], theirs.dims or []
    if adapts:                  # an adapter's input: on the value's rows and positions, its features its own
        own_axes = [a for a in own_axes if a in address.EXAMPLE_AXES]
        their_axes = [a for a in their_axes if a in address.EXAMPLE_AXES]
    if not address.unstacked(their_axes):
        return None             # one number (at each layer), which broadcasts in any coordinates as a literal does
    featured = any(a not in address.EXAMPLE_AXES for a in their_axes)    # a column or a per-row dose is in none
    if featured and ours.space != theirs.space:
        where = ("this bracket" if ours.space else "this write") if write else f"the value {name!r} {verb}"
        return (f"{written!r} is in {said(theirs.space)}; {where} is in {said(ours.space)}, so an operand is "
                "encoded as it is")
    if [a for a in address.unstacked(own_axes) if a in their_axes] != address.unstacked(their_axes):
        return (f"{written!r} is laid out {theirs.dims}; {name!r} {verb} a value laid out {ours.dims}, and an "
                "operand's axes are among the value's, in its order (those it leaves out it broadcasts over)")
    if "layers" in their_axes:
        if ours.layers is None:
            return (f"{written!r} has a layers axis, and this op acts at no layer (on a model's own value, or one "
                    "reduced over its layers); read the operand at one layer, or reduce it over layers")
        missing = [layer for layer in ours.layers if layer not in theirs.layers] if theirs.layers else []
        if missing:
            return (f"{written!r} holds layers {theirs.layers}, and this op acts at layers {ours.layers}; at each "
                    f"layer it takes the operand's slice of that layer, and it holds none of {missing}: read it at "
                    "those layers too")
    if "rows" in their_axes and theirs.dataset != ours.dataset:
        return (f"{mine} has a row per example of {ours.dataset!r} and {written!r} of {theirs.dataset!r}; an op "
                "takes one dataset's rows")
    if "pos" in their_axes and theirs.read != ours.read:
        if (theirs.width is None or ours.width is None) and not forward:
            ragged = repr(written) if theirs.width is None else f"the value {name!r} {verb}"
            return (f"{ragged} holds a different number of positions on each row, which pair with no other "
                    "value's; read both at a static index (pos -1, or {'index': '-3:'})")
        if theirs.width is not None and ours.width is not None and theirs.width != ours.width:
            return (f"{written!r} holds {theirs.width} position(s) per row and the value {name!r} {verb} holds "
                    f"{ours.width}; position i {'scores against' if group == 'metrics' else 'meets'} position i, so "
                    "both are read at as many (a pos axis does not broadcast)")
    for axis in their_axes:
        own, held = ours.kept.get(axis), theirs.kept.get(axis)
        if axis in address.EXAMPLE_AXES or adapts or own == [] or held == [] or own == held:
            continue                                        # not a selected axis, pending, or the same
        if own is not None and held is not None and sorted(own) == sorted(held):
            why = "a select list is positional, so they would trade places"
        elif held is not None and len(held) == 1:
            why = "an operand broadcasts over rows and positions only"
        else:
            why = "an operand holds the same ones"
        fix = f"select {axis} {own if own is not None else 'every one'} at the read" if write \
            else f"select one {axis} list at both reads"
        return (f"{written!r} holds {f'every {axis}' if held is None else f'{axis} {held}'}, and {whose} holds "
                f"{'every one' if own is None else own}; {why}: {fix}")
    return None
