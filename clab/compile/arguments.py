"""An op's arguments resolved against the plan's own records: the names it references (a value, a column, a
featurizer), the layout of its result, and the one rule an operand keeps against the value it meets (`mismatch`)."""

import inspect
import json
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

from clab import featurizers, ops
from clab.featurizers import bundles
from clab.compile import positions
from clab.compile.answers import check_column_ref, check_text_ref, is_literal
from clab.compile.state import Compiling
from clab.compile.uses import tensor_refusal
from clab.format import address, data, registry, selectors
from clab.format.sweeps import agreeing_point, points
from clab.plan import (Featurizer, Intervention, Labelled, Load, Op, OpStep, Plan, arguments, fail, horizon, kind_of,
                       member_of, produced, record_of, tensor_of, top_k_members, top_k_of, trained_featurizers)


def narrowed_readout_refusal(what: str) -> str:
    return (f"readout projects the whole residual stream through the model's final norm and head, and {what} keeps "
            "only some of hidden_size, which the norm cannot take; read the site whole, readout, and narrow what "
            "the logits give")


def space_of(encodes: list[Op] | list[OpStep]) -> list[dict]:
    """The coordinates a value is in: the encodes open around it, outermost first (an entry's ops, or an encode
    step), each its featurizer and `select`."""
    return [{k: v for k, v in arguments(o).items() if k in ("featurizer", "select")} for o in encodes]


def coordinates_text(space: list[dict]) -> str:
    """A value's coordinates as a message names them."""
    if not space:
        return "the site's own layout"
    return "the coordinates of " + " then ".join(repr(s["featurizer"]) + (f" (select {s['select']})" if "select" in s
                                                                       else "") for s in space)


def featurizer_kind(cx: Compiling, plan: Plan, path: str, name: str, args: dict) -> str | None:
    """The kind of the featurizer op `name` names, checked to exist and to serve the op, and an `apply`'s `with`
    to be a name its kind takes; None when the op takes no featurizer."""
    ref = args.get("featurizer")
    if ref is None:
        return None
    check_featurizer_ref(cx, ref, f"{path}.featurizer")
    kind = kind_of(plan, ref)
    serves = featurizers.KINDS[kind].ops()
    if name not in serves:
        fail(f"{path}.featurizer", f"{ref!r} is {featurizers.article(kind)} {kind!r} featurizer, which has no {name}: "
             f"it serves {', '.join(serves)}" + ("; it has no way back into the value" if name == "decode" else ""))
    if name == "apply" and args.get("with") is not None:
        if "with_" not in registry.signature(featurizers.KINDS[kind].method("apply")).parameters:
            fail(f"{path}.with", f"{featurizers.article(kind)} {kind}'s apply takes no 'with', an adapter's input read "
                 "at another site (a lora's), or the value a gate interpolates toward")
        if featurizers.KINDS[kind].adapts and not isinstance(args["with"], str):
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
    named_arguments = {}
    for key, p in ops.parameters(name).items():
        value = args.get(key)
        named = p.annotation in (ops.Tokens, ops.Text, ops.Featurizer) or p.annotation == ops.Value and isinstance(value, str)
        record = resolve(key, value, p) if named and value is not None else None
        if record is not None:
            ref = data.column_of(value) if p.annotation != ops.Value else None
            named_arguments[key] = (ref[0] if ref else json.dumps(value, ensure_ascii=False) if data.literal_of(value)
                                    else str(value), record)
    return named_arguments


def resolver(cx: Compiling, path: str, classes: bool, at: dict[str, int],
             live: Mapping[str, Labelled] | None = None) -> Callable[[str, Any, inspect.Parameter], Labelled | None]:
    """How compile resolves an op's named arguments (`references`) at `path`, at the consumer's point `at`: a value
    to its record (`value_record`, an entry of the forward being compiled among `live`), a column to a row per example
    of its dataset once it is checked to hold text on every row, or, scored against a probe's `classes`, its
    labels, whatever they are (`classes`), a literal to a row per example of whatever dataset the value it goes with
    has rows of, the same text on each, once it is checked to be one (`is_literal`), and a featurizer to nothing, once
    it is checked to exist."""
    def resolve(key: str, value: Any, p: inspect.Parameter) -> Labelled | None:
        where = f"{path}.{key}"
        if p.annotation == ops.Value:
            record = value_record(cx, value, where, at, live)
            if why := whole_top_k(cx, value, record, at, "an operand"):
                fail(where, why)
            return record
        if p.annotation == ops.Featurizer:
            check_featurizer_ref(cx, value, where)
            return None
        if is_literal(value, where):
            return Labelled(dims=["rows"])          # an answer is laid against rows, a column's and a literal's alike
        ref = data.column_of(value)
        dataset = check_column_ref(cx, ref[0] if ref else value, where, None) \
            if classes and p.annotation == ops.Tokens else check_text_ref(cx, value, where, None)
        return Labelled(dims=["rows"], dataset=dataset)
    return resolve


def layout(path: str, name: str, args: dict, dims: list[str], where: str, kind: str | None) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims`. `where` is `step`, or inside an entry `live`
    up to the entry's last edit, where the op acts on the live value the entry writes back, and `entry` after it;
    inside an entry the op sees one batch, so it may not aggregate over rows: a reduction of a read is a step after
    the forward, and one a write would write back is a step before it, written in. `original` is only an entry's."""
    if where == "step" and "original" in registry.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside an entry")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args}, kind)
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and name == "select":
        fail(path, "inside an entry the value is its site's selection, which the site's own select narrows "
             "({'component': ..., 'select': {<axis>: <selector>}}); select a value in a step after the forward")
    if where != "step" and name == "top_k" and isinstance(args.get("over"), list):
        fail(path, "a top_k over a list of axes gives an index per axis of each pick, which an entry's batches do "
             "not join; make it a step after the forward")
    if where != "step" and "rows" in dims and "rows" not in out:
        fail(path, f"{name!r} aggregates over rows, and inside an entry it sees one batch; " + (
            "this entry writes it back, so make it an op step before the forward, over a read of the rows it "
            f"aggregates, and write that in ({{'op': 'swap', 'with': '<that step>'}})" if where == "live" else
            "make it a step after the forward"))
    return out


def select_kept(where: str, of: Labelled, args: dict, sizes: dict[str, int] | None,
                known: dict[str, tuple[int, str]], bounds: dict[str, tuple[int, str]] | None = None,
                compiled: dict[str, selectors.Kept] | None = None) -> dict[str, selectors.Kept]:
    """What a `select` step over the value labelled `of` keeps on each axis it narrows: the picks of its selector
    among the entries along that axis, each the index the value holds there (one a site's or an earlier select's
    `kept` names, else the axis's own), so selects compose. Their number is the record's or what the stage knows
    (`along`: at compile the counts the document fixes, `select_counts`; in preflight the model's sizes; in the run
    the value's own shape). Where none knows it, a selector counted from the start (`selectors.keyed`) picks the same
    indices whatever the number, held to the upper bound the document fixes (`bounds`: a generation's
    `max_new_tokens`, a fit's `epochs` under `early_stop`), and any other is `[]` until a stage that knows the number,
    the run at last. Where this stage knows no more than an earlier one that resolved the axis (`compiled`), its picks
    stand. A pick past the entries, or a selector that keeps none, is refused by name, a `ValueError` starting with
    `where`."""
    out: dict[str, selectors.Kept] = {}
    for axis, selector in args["select"].items():
        held, count = of.kept.get(axis), along(of, axis, sizes, known)
        if held == [] or count is None and (compiled or {}).get(axis):     # waits on preflight, or compile's stands
            out[axis] = (compiled or {}).get(axis, [])
            continue
        if count is not None:
            selectors.selected(where, axis, selector, count[0], f"the {count[0]} along {axis} ({count[1]})")
            picks = selectors.select(selector, count[0])
        elif selectors.keyed(selector):
            if axis in (bounds or {}):
                n, why = (bounds or {})[axis]
                selectors.selected(where, axis, selector, n, f"the {n} along {axis} at most ({why})")
            picks = selectors.keys(selector)
        else:
            out[axis] = []
            continue
        out[axis] = [held[i] for i in picks] if isinstance(held, list) else picks
    return out


def selected_layers(of: Labelled, kept: dict[str, selectors.Kept]) -> tuple[list[int] | None, dict[str, selectors.Kept]]:
    """A `select`'s value's layers and the indices it keeps (`select_kept`'s), the picks along `layers` taken out of the
    kept indices and mapped to the model layers of the value it is over (`[]` while either waits on a later stage);
    the value's own layers where it selects none."""
    if "layers" not in kept:
        return of.layers, kept
    picks, rest = kept["layers"], {a: v for a, v in kept.items() if a != "layers"}
    return ([cast(list, of.layers)[i] for i in cast(list, picks)] if picks and of.layers else []), rest


def select_counts(cx: Compiling, record: Labelled, ref: str, at: dict[str, int]
                  ) -> tuple[dict[str, tuple[int, str]], dict[str, tuple[int, str]]]:
    """What compile knows of the numbers of entries along the axes of value `ref` (labelled `record`) a `select`
    narrows, each with what says so: those `known_counts` knows, and those the document fixes on the value the axis
    comes from, through the op steps that keep it: a featurizer's tensor's own axes (a probe's `classes`, an sae's
    latents, a subspace's or a pca's `k`, a lora's `rank`: `coordinates_of`) and a fit's history's `epochs`; and,
    apart, the upper bounds it fixes where the run may hold fewer, a generation's `max_new_tokens` along `new` and
    the epochs of a fit that `early_stop` may end sooner."""
    exact, bounds = known_counts(cx, record, ref, at), {}
    name, here = ref, at
    while True:
        thing, _, member = name.partition(".")
        if (tensor := tensor_of(cx.steps, cx.featurizers, name)) is not None:
            f = points(cx.featurizers[thing if thing in cx.featurizers else tensor[0].partition(".")[2]])[0]
            n = coordinates_of(f)
            for axis in featurizers.KINDS[f.kind].laid.get(tensor[1], ()):
                if axis == "classes":
                    exact[axis] = (len(featurizers.labels(cx.rows, (f.model_extra or {})["classes"])), "the probe's")
                elif axis != "width" and n is not None:
                    exact[axis] = (n, f"the {f.kind}'s")
            break
        step = agreeing_point(cx.steps[thing], here) if thing in cx.steps else None
        if step is not None and step.kind == "fit" and member.partition(".")[0] == "history":
            stops = step.early_stop is not None
            (bounds if stops else exact)["epochs"] = (cast(int, step.epochs), "the fit's epochs" + (
                ", which early_stop may end sooner" if stops else ""))
        elif step is not None and step.kind == "generate" and not member and (most := horizon(step)) is not None:
            bounds["new"] = (most, "the generation's max_new_tokens, which a row that stops sooner leaves shorter")
        elif step is not None and step.kind == "op" and not member:
            name, here = step.of, step.at
            continue
        break
    dims = record.dims or []
    return {a: c for a, c in exact.items() if a in dims}, {a: c for a, c in bounds.items() if a in dims}


def kept_labels(of: Labelled, dims: list[str]) -> tuple[list[int] | None, dict[str, selectors.Kept], int | None]:
    """What an op's value keeps of the labels of the value it is over: its layers, unless the op removed a
    `layers` axis (then it is from no layer), the indices kept on each selected axis it still has, and its width
    where it still has a `pos` axis."""
    return (None if "layers" in (of.dims or []) and "layers" not in dims else of.layers,
            {a: v for a, v in of.kept.items() if a in dims}, of.width if "pos" in dims else None)


def value_record(cx: Compiling, ref: str, path: str, at: dict[str, int],
                 live: Mapping[str, Labelled] | None = None) -> Labelled:
    """The record of a named value produced before this point, at the consumer's point `at`, noted as used (`used`);
    an entry of the forward being compiled, among `live`, as it is, noted as a live operand instead (`live`); a
    featurizer's tensor (`plan.tensor_of`) as its first use before this point lays it out (`uses.tensor_refusal`
    says what it refuses)."""
    if live and ref in live:
        cx.live.add(ref)
        return live[ref]
    if tensor_of(cx.steps, cx.featurizers, ref) is not None:
        if why := tensor_refusal(cx.steps, cx.featurizers, ref, at, True):
            fail(path, why)
        cx.used.add(ref)
        return record_of(cx.steps, ref, at, cx.featurizers)
    if ref.partition(".")[0] in cx.featurizers or ref in trained_featurizers(cx.steps):
        fail(path, f"{ref!r} is a featurizer, not a value; an op reaches it ({{'op': 'encode', 'featurizer': "
             f"{ref!r}}} gives a value's coordinates, 'apply' its version of the value), and the value that entry "
             "binds is an operand")
    known = produced(cx.steps, cx.featurizers)
    if ref not in known:
        fail(path, f"{ref!r} names no value produced before this step; known: {', '.join(sorted(known)) or 'none'}")
    cx.used.add(ref)
    return record_of(cx.steps, ref, at, cx.featurizers)


def several(top: Op | OpStep) -> str:
    """What a top_k's value holds, as a refusal names it."""
    over = arguments(top).get("over", ops.transforms.RANKED)
    axes = over if isinstance(over, list) else [over]
    return ("its ids" + (" (one tensor per axis it ranks)" if isinstance(over, list) else "") +
            (", their probs" if set(axes) & set(ops.VOCABULARY) else ", their values") +
            (" and their tokens" if isinstance(top, OpStep) and ops.transforms.RANKED in axes else ""))


def whole_top_k(cx: Compiling, ref: str, record: Labelled, at: dict[str, int], takes: str) -> str | None:
    """Why value `ref` (its record `record`, at the consumer's point `at`) is no one tensor an op takes: a `top_k`'s
    whole value (`plan.top_k_of`), which is several tensors, whose members a step's names (`plan.top_k_members`); a
    top_k step's `ids` over a list of axes, one tensor per axis; its `tokens`, text. A load's value is each of these
    as the run that saved it recorded it, and the fix is the member to save there. None for any other value. `takes`
    names what takes one (an op, an operand)."""
    loaded = record.value if isinstance(record, Load) else None          # the name the run that saved it gave it
    found = member_of(cx.sources[record.from_]["plan"].steps, record.value, record.at) if isinstance(record, Load) \
        else member_of(cx.steps, ref, at)
    if found is not None:
        step, member = found
        named, over = loaded or ref, arguments(step).get("over")
        what, fix = (f"{ref!r} loads {loaded!r}, a top_k's", "save {} in the run that made it and load that") \
            if loaded else (f"{ref!r} is a top_k's", "name {}")
        if member == "ids" and isinstance(over, list):
            axes = ", ".join(repr(f"{named}.{a}") for a in over)
            return (f"{what} ids over a list of axes, one tensor per axis it ranks, and {takes} takes one: "
                    + fix.format(f"the axis's ({axes})"))
        if member == "tokens":
            return (f"{what} tokens, its picks' token strings, which are text, and {takes} takes a tensor: "
                    + fix.format(f"their ids ({named.removesuffix('.tokens') + '.ids'!r})"))
        return None
    top = top_k_of(record if not isinstance(record, Load) else
                   record_of(cx.sources[record.from_]["plan"].steps, record.value, record.at,
                             cx.sources[record.from_]["plan"].featurizers))
    if top is None:
        return None
    if isinstance(top, OpStep):
        named = ", ".join(repr(f"{loaded or ref}.{m}") for m in top_k_members(top)
                          if m != "tokens" and not (m == "ids" and isinstance(arguments(top).get("over"), list)))
        fix = (f"save the member it needs in the run that made it ({named}) and load that" if loaded else
               f"name the member it needs ({named})")
        fix += ", or rank what it needs with an op over the value the top_k ranks" + ("" if loaded else f", {top.of!r}")
    else:
        fix = ("rank what it needs in that entry in place of the top_k, or make the top_k a step after the forward, "
               "whose members a later step names (`<step>.ids`)")
    return f"{ref!r} is a top_k's value, several tensors ({several(top)}), and {takes} takes one: {fix}"


def k_past(name: str, args: dict, met: Labelled, sizes: dict[str, int] | None,
           known: dict[str, tuple[int, str]]) -> str | None:
    """Why a `top_k`'s `k` (op `name`'s arguments `args`) cannot pick among the entries of the axes it ranks on the
    value it meets, labelled `met`: below 1, which picks nothing, or more than there are, where their number is
    static. From the record: a `select`'s indices, the layers (at compile those written from the first, in preflight
    every pick), a width of `pos` (at compile a static one, in preflight one the rows confirm); from `known`, by axis
    with what says so, what the caller knows (at compile `known_counts`: the rows, a probe's classes, a featurizer's
    coordinates, an earlier top_k's picks; in preflight a probe's classes); and in preflight the model's sizes
    (`sizes`, None at compile). Along a ragged axis (`transforms.RAGGED`: a `pos` whose rows differ, the keys) there is
    no one number, and a pick past what is there is id -1. None where `k` fits, or where an axis's number is known to
    neither stage (a gate's coordinates, a generation's `new`, a fit's `epochs`, a load's `k`)."""
    if name != "top_k" or not isinstance(args.get("k"), int):
        return None
    if args["k"] < 1:
        return f"k {args['k']}: a top_k picks at least one entry, so k is 1 or more"
    over = args.get("over", ops.transforms.RANKED)
    counts: list[tuple[str, int, str]] = []
    for axis in over if isinstance(over, list) else [over]:
        if (count := along(met, axis, sizes, known)) is None:
            return None
        counts.append((axis, *count))
    n = math.prod(c for _, c, _ in counts)
    if args["k"] <= n:
        return None
    held = " by ".join(f"{c} along {axis} ({why})" for axis, c, why in counts)
    return (f"k {args['k']} is more than the {n} entr{'y' if n == 1 else 'ies'} it ranks, {held}, each of which it "
            f"picks once: k is at most {n}")


def along(met: Labelled, axis: str, sizes: dict[str, int] | None, known: dict[str, tuple[int, str]]
          ) -> tuple[int, str] | None:
    """How many entries the value labelled `met` holds along `axis`, with what says so, where its number is static and
    known: from the record, a `select`'s indices, the layers (at compile those written from the first, in preflight
    every pick), a width of `pos`; from `known`, what the caller knows (`known_counts` at compile); and in preflight
    the model's sizes (`sizes`, None at compile), along an axis no row holds a number of its own along
    (`transforms.RAGGED`). None where neither stage knows it, or it waits on preflight (a `select` it resolves)."""
    kept = met.kept.get(axis)
    if isinstance(kept, list) and kept:
        return len(kept), "those its select keeps"
    if kept is not None:
        return None                     # left to preflight, or keys relative to each query
    if axis == "layers" and met.layers:
        return len(met.layers), "the value's"
    if axis == "pos" and met.width is not None:
        return met.width, "each row's"
    if axis in known:
        return known[axis]
    if sizes is not None and axis in sizes and axis not in ops.transforms.RAGGED:
        return sizes[axis], "the model's"
    return None


def known_counts(cx: Compiling, record: Labelled, ref: str | None = None,
                 at: dict[str, int] | None = None) -> dict[str, tuple[int, str]]:
    """What compile knows of how many entries the value `ref` names (or an entry's, labelled `record`) holds along the
    axes `k_past` takes from the caller, each with what says so: its dataset's rows (in a fit's body a run's rows are
    fewer, which `compile_fit` holds a `top_k` to); the classes of the probe its `classes` axis is of (the last encode
    left open around it); the coordinates of the featurizer its `features` are in, as its field or bundle counts them
    (`coordinates_of`), as many as that encode's `select` keeps; and the picks of the `top_k` step whose member it is,
    or descends from through op steps, along `k`."""
    dims, out = record.dims or [], {}
    if "rows" in dims and record.dataset in cx.rows:
        out["rows"] = (len(cx.rows[cast(str, record.dataset)]), "the dataset's")
    if record.space and ("classes" in dims or "features" in dims):
        ref_f = record.space[-1]["featurizer"]
        spec = points(cx.featurizers[ref_f if ref_f in cx.featurizers else ref_f.partition(".")[2]])[0]
        if "classes" in dims:
            out["classes"] = (len(featurizers.labels(cx.rows, (spec.model_extra or {})["classes"])), "the probe's")
        elif (n := coordinates_of(spec)) is not None:
            select = record.space[-1].get("select")
            out["features"] = (len(selectors.select(select["features"], n)), "those its encode's select keeps") \
                if select else (n, f"the {spec.kind}'s coordinates")
    if "k" in dims and ref is not None and at is not None and (k := picks_of(cx, ref, at)) is not None:
        out["k"] = (k, "the picks of the top_k it ranks")
    return out


def coordinates_of(spec: Featurizer) -> int | None:
    """How many coordinates featurizer `spec` (a compiled record) encodes into, where compile knows: the field of its
    kind that counts them (`Kind.counts`, else one it is drawn from: a subspace's or a pca's `k`, an sae's
    `latents`), written or recorded by its bundle's header, else the last axis of the first tensor its bundle holds,
    the encode's (a published sae's `W_enc`, `[width, latents]`); None for a drawn featurizer with no such field (a
    gate's, as wide as its site)."""
    kind, fields = featurizers.KINDS[spec.kind], spec.model_extra or {}
    for key in dict.fromkeys((*kind.counts, *kind.drawn)):
        given = fields.get(key)
        if given is None and spec.header is not None and key in spec.header:
            given = bundles.value(spec.header[key])
        if isinstance(given, int) and not isinstance(given, bool):
            return given
    return None if spec.bundle is None else spec.bundle[kind.saves[0]].shape[-1]


def picks_of(cx: Compiling, ref: str, at: dict[str, int]) -> int | None:
    """The `k` of the `top_k` step whose member value `ref` is, or descends from through op steps; None for any other
    value (a load's `k` axis, whose top_k ran in another run)."""
    if (found := member_of(cx.steps, ref, at)) is not None:
        return arguments(found[0])["k"]
    record = record_of(cx.steps, ref, at, cx.featurizers)
    return picks_of(cx, record.of, record.at) if isinstance(record, OpStep) and "k" in (record.dims or []) else None


def check_featurizer_ref(cx: Compiling, value: object, path: str) -> None:
    """A declared featurizer's name, or one a fit trains (`<fit>.<featurizer>`)."""
    known = sorted(set(cx.featurizers) | set(trained_featurizers(cx.steps)))
    if value not in known:
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")


# -- the one rule an operand keeps ---------------------------------------------------------------------


def mismatch(ours: Labelled, theirs: Labelled, name: str, key: str, written: str, mine: str, write: bool,
             forward: bool, adapter: bool, reads: Callable[[str], Intervention]) -> str | None:
    """Why the value `written` names, laid out `theirs`, cannot be operand `key` of op `name` on the value `mine`
    names, laid out `ours`, at one point; None where it can. The rule is Ops' one rule (`ops.ABOUT`): an operand
    may leave out any axis of the value it meets, which it then broadcasts over; every axis it has is the value's,
    in the value's order, with the same label (`layers` by model layer, `rows` by dataset, a literal's, which names
    none, by any one, `pos` by the read whose positions both hold or by `width`, a selected axis by the indices kept
    in order, `features` by the encodes it is in); a `layers` axis it has covers the layers the op acts at and is
    aligned by id. Two values with no `read` (a permute's, whose rows hold other rows' positions) hold no read's
    positions in common, so they pair by width too, and a ragged one with none. A number broadcasts in any
    coordinates; an adapter's `with` (`adapter`: the op is an `apply` of one, `plan.adapts`) is compared on its rows
    and positions alone. `write` says the op is one of an
    entry's up to its last edit (what the messages call the value), `forward` that the operand is an entry of the
    value's own forward (a ragged pair there is compared row by row by `check_operands`, once preflight has the
    rows). A label preflight has yet to resolve (`[]`) is compared then, and so is a width: `reads` gives the entry a
    value's `read` names, and where its `pos` leaves the width to the rows (`positions.pending_width`, a scope's in
    the prefill), a width of None is pending until that entry holds its rows' `positions` (in preflight, or at
    compile already for a load's read, which carries them from its run), and then says the rows disagree. The
    coordinates are compared first, since two values in different coordinates differ in their last axis too, and
    that is the reason. A column scored as `Tokens` is first held to a whole vocabulary axis, which no site's
    `select` (`kept`) or probe's `encode` with a `select` (`space`) narrowed: its ids are of the whole vocabulary,
    where index i of a selection is not id i."""
    narrowed = next((a for a in ops.VOCABULARY if a in (ours.dims or []) and (
        a in ours.kept or ours.space and "select" in ours.space[-1])), None) \
        if ops.parameters(name)[key].annotation == ops.Tokens else None
    if narrowed is not None:
        kept = f"{len(ours.kept[narrowed])} of its {narrowed}" if ours.kept.get(narrowed) else \
            f"some of its {narrowed}"
        return (f"the value {name!r} scores keeps {kept} (select), and it scores {written!r} by its id over the whole "
                "vocabulary, where index i of the selection is not id i: score the whole vocabulary (read it without "
                "the select, and narrow what the metric gives), or rank within the selection by index (argmax or "
                "top_k over it)")
    group = ops.DECLARED[name]["group"]
    verb = "scores" if group == "metrics" else "acts on"
    whose = "this write's selection" if write else "the value it acts on"
    adapts = adapter and key == "with"
    own_axes, their_axes = ours.dims or [], theirs.dims or []
    if adapts:                  # an adapter's input: on the value's rows and positions, its features its own
        own_axes = [a for a in own_axes if a in address.EXAMPLE_AXES]
        their_axes = [a for a in their_axes if a in address.EXAMPLE_AXES]
    if not address.unstacked(their_axes):
        return None             # one number (at each layer), which broadcasts in any coordinates as a literal does
    featured = any(a not in address.EXAMPLE_AXES for a in their_axes)    # a column or a per-row dose is in none
    if featured and ours.space != theirs.space:
        where = ("this bracket" if ours.space else "this write") if write else f"the value {name!r} {verb}"
        return (f"{written!r} is in {coordinates_text(theirs.space)}; {where} is in {coordinates_text(ours.space)}, "
                "so an operand is encoded as it is")
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
    if "rows" in their_axes and theirs.dataset is not None and theirs.dataset != ours.dataset:
        return (f"{mine} has a row per example of {ours.dataset!r} and {written!r} of {theirs.dataset!r}; an op "
                "takes one dataset's rows")
    if "pos" in their_axes and (theirs.read != ours.read or theirs.read is None):   # a permute's pair by width
        # Each side whose width is open, the operand first: the value it meets is named as what the operand meets.
        for side, ragged, counted in ((theirs, repr(written), repr(written)),
                                      (ours, f"the value {name!r} {verb}",
                                       f"{written!r} meets the value {name!r} {verb}, which")):
            if side.width is not None or forward:
                continue
            read = reads(side.read) if side.read is not None else None
            if read is None or not positions.pending_width(positions.parse(read.pos)):
                return (f"{ragged} has no width compile can fix (an open index, an `until` scope, several "
                        "occurrences), so it pairs with no other forward's value; read both at a static index without a "
                        "scope ('index': -1, or '-3:'), at one "
                        "occurrence of a scope in the prefill, not until, whose rows give its width, or over decode "
                        "calls at a frame of a static number of them with an index of static width ({'frame': '1:4', "
                        "'index': ':3'})")
            resolved = read.positions.get(cast(str, read.dataset))
            if resolved is not None:                # the rows are known (preflight's, a load's), and they disagree
                return (f"{counted} holds {row_counts(resolved)}; a scoped read pairs with another forward's value "
                        "when every row's scope holds the same number of tokens: pick one with 'index' (-1, or '-2:'), "
                        "or reduce it")
        if theirs.width is not None and ours.width is not None and theirs.width != ours.width:
            return (f"{written!r} holds {theirs.width} position(s) per row and the value {name!r} {verb} holds "
                    f"{ours.width}; position i {'scores against' if group == 'metrics' else 'meets'} position i, so "
                    "both are read at as many (a pos axis does not broadcast)")
    for axis in their_axes:
        own, held = ours.kept.get(axis), theirs.kept.get(axis)
        if axis in address.EXAMPLE_AXES or adapts or own == [] or held == [] or own == held:
            continue                                        # not a selected axis, pending, or the same
        if isinstance(own, dict) or isinstance(held, dict):
            def keyed(k: selectors.Kept | None) -> str:
                return "every key" if k is None else f"the keys at offsets {k['offset']} from each query" \
                    if isinstance(k, dict) else f"keys {k}"
            return (f"{written!r} holds {keyed(held)}, and {whose} holds {keyed(own)}; a key relative to the query "
                    "pairs with the same offsets alone: select one key offset at both reads")
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


def row_counts(resolved: list) -> str:
    """How many positions each row holds, by count in the order the rows first hold it: `1 position on rows 0-7 and 2
    positions on row 8`, a row a read reached at no position holding 0, and past four runs of rows a count's rows
    as their first three runs and how many there are."""
    rows: dict[int, list[int]] = {}
    for row, p in enumerate(resolved):
        rows.setdefault(len(p) if isinstance(p, list) else 0, []).append(row)

    def named(of: list[int]) -> str:
        runs: list[list[int]] = []
        for row in of:
            if runs and row == runs[-1][-1] + 1:
                runs[-1].append(row)
            else:
                runs.append([row])
        spans = [f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs]
        if len(spans) > 4:
            spans = spans[:3] + [f"… ({len(of)} rows)"]
        return ("row " if len(of) == 1 else "rows ") + ", ".join(spans)
    counts = [f"{n} position{'s' * (n != 1)} on {named(of)}" for n, of in rows.items()]
    return ", ".join(counts[:-1]) + " and " + counts[-1] if len(counts) > 1 else counts[0]
