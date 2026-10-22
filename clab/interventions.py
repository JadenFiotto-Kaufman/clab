"""Interventions: a step's entries, the brackets their ops nest in, an entry compiled (its site inlined, its
positions checked against its step, its ops' layouts and operands), and what preflight checks of them against
the model."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

from . import address, data, featurizers, ops, positions, registry, selectors
from .answers import column_ref, text_ref
from .plan import (Intervention, Op, Plan, Site, agreeing, arguments, body_view, dims_of, edits, entries, entry_of,
                   fail, gathers, horizon, kind_of, parameters, walk, writes)
from .positions import decided
from .sweeps import lookup

if TYPE_CHECKING:
    from .compile import Compiling


class Unbracketed(ValueError):
    """A `decode` its entry has no encode open for, at index `at` among the entry's ops."""

    def __init__(self, at: int, message: str):
        super().__init__(message)
        self.at = at


def bracketed(listed: list[Op]) -> list[dict]:
    """An entry's ops as their brackets nest, the one place a bracket is read: each op `{"at": <its index>,
    "op": <the op>}` in turn, and an `encode` also with the ops after it as `inner`, nested the same way, up to
    the `decode` that closes it (`closed`, that decode's index; the decode is not a node of its own) or, when
    none does, to the end. A `decode` closes the most recent `encode` not yet closed; one with none open, or of
    another featurizer than the open one, is refused (`Unbracketed`)."""
    top: list[dict] = []
    opened: list[dict] = []
    for i, o in enumerate(listed):
        ref = arguments(o).get("featurizer")
        if o.op == "decode":
            if not opened:
                raise Unbracketed(i, f"decode of {ref!r} closes no encode: none is open in this entry here; put the "
                                     f"encode of {ref!r} before it, the edits of its coordinates between them")
            theirs = arguments(opened[-1]["op"])["featurizer"]
            if theirs != ref:
                raise Unbracketed(i, f"decode of {ref!r}, and the open encode is of {theirs!r} "
                                     f"(ops[{opened[-1]['at']}]); close {theirs!r} first: a decode closes the most recent encode not yet closed")
            opened.pop()["closed"] = i
            continue
        node = {"at": i, "op": o}
        (opened[-1]["inner"] if opened else top).append(node)
        if o.op == "encode":
            node.update(inner=[], closed=None)
            opened.append(node)
    return top


def visited(nodes: list[dict], around: tuple = ()):
    """`bracketed` nodes in the order their ops run, each `(node, around)`, `around` the encode nodes open
    around it, outermost first."""
    for node in nodes:
        yield node, around
        yield from visited(node.get("inner", []), around + (node,))


def items(plan: Plan, path: str, name: str, spec) -> list[tuple[str, str, Intervention]]:
    """A step's entries in the order written, each with its document path and name: the items of its
    `interventions` in turn (a dict is one item), a bundle's entries where its name stands, rescoped to the
    step (`scoped`)."""
    value = spec.interventions
    listed = value if isinstance(value, list) else [value]
    out = []
    for i, item in enumerate(listed):
        at = f"{path}.interventions" + (f"[{i}]" if isinstance(value, list) else "")
        if isinstance(item, dict):
            out += [(f"{at}.{k}", k, e) for k, e in item.items()]
            continue
        if item not in plan.interventions:
            fail(at, f"no intervention bundle {item!r}; declared: {', '.join(plan.interventions) or 'none'}")
        pullers = {n.rpartition(".")[2] for n, s in walk(plan.steps) if item in bundles_of(s)}
        out += [(f"interventions.{item}.{k}", k, scoped(plan, f"interventions.{item}.{k}", e, item, name, pullers))
                for k, e in plan.interventions[item].items()]
    return out


def bundles_of(step) -> list[str]:
    """The bundles a step's `interventions` pull in, by name."""
    value = getattr(step, "interventions", None)
    return [i for i in (value if isinstance(value, list) else [value]) if isinstance(i, str)]


def scoped(plan: Plan, path: str, spec: Intervention, bundle: str, step: str, pullers: set) -> Intervention:
    """A bundle's entry, its operands that name one of the bundle's own entries by its bare name (`"premix"`)
    naming that entry of the step it is pulled into (`"<step>.premix"`). An operand that names `<a step that
    pulls the bundle in>.<one of its entries>` is refused: pulled into another step it would still mean that
    step's entry, not the one this step makes. `pullers` are the steps that pull the bundle in."""
    own = plan.interventions[bundle]
    rewritten = []
    for i, o in enumerate(spec.ops):
        args = arguments(o)
        try:
            params = parameters(plan, o.op, args)
        except (KeyError, ValueError):          # an unknown featurizer or argument, which compile refuses by name
            params = ops.parameters(o.op)
        changes = {}
        for key, value in args.items():
            if key not in params or params[key].annotation != ops.Value or not isinstance(value, str):
                continue
            thing, _, member = value.partition(".")
            if thing in pullers and member in own:
                fail(f"{path}.ops[{i}].{key}", f"{value!r} names step {thing!r}'s entry {member!r}, which is also "
                     f"this bundle's own entry; name the bundle's own entry bare ({member!r}), which means it in "
                     "whichever step pulls the bundle in")
            if value in own:
                changes[key] = f"{step}.{value}"
        rewritten.append(Op(**{**o.model_dump(exclude_defaults=True), **changes}) if changes else o)
    return spec.model_copy(update={"ops": rewritten})


def entry(cx: Compiling, path: str, spec: Intervention, plan: Plan, step, dataset: str,
          live: dict[str, tuple[dict, Intervention]], own: set) -> dict:
    """One entry: its site inlined, its positions checked against the step, its ops' layouts. An entry that
    holds an edit writes: each op up to its last edit acts on the live value, so no edit follows an op that
    changed the layout, unless an `encode` changed it and its `decode` follows, and every `encode` before the
    write-back is closed by then; its operands stand for the value they meet (`operands`). `live` holds the
    earlier entries of a forward, by name, which an operand may name; `own` names every entry of the step."""
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    write = writes(spec)
    if write and address.stacks(site.layers):
        fail(f"{path}.site", f"an entry that writes names its layer, an int; layers {site.layers!r} stack several, "
             "which is a read's (or sweep the layers)")
    if write and len(site.select or {}) > 1:
        fail(f"{path}.site.select", f"an entry that writes selects on one axis; this one selects "
             f"{', '.join(site.select or {})}")
    pos = positions.parse(spec.pos)
    frame, scope = pos["frame"], pos["scope"] or {}
    if step.kind != "generate" and selectors.select(frame, 1) != [0]:
        fail(f"{path}.pos.frame", f"frame {frame!r}: a forward makes one call, the prefill; its frame picks that "
             "call (0, or ':', the whole timeline, which in a forward is that call)")
    if frame != 0 and "key" in address.COMPONENTS[site.component]["dims"]:
        fail(f"{path}.site", f"{site.component}'s key axis grows with every decode call; read it in the prefill")
    if write and scope.get("before") and not positions.prefill(pos):
        fail(f"{path}.pos.scope.before", "a write beyond the prefill lands as its call runs, before the token it "
             "would predict is chosen; a scope that is before is a read's there, or a write's in the prefill")
    if write and frame != 0 and step.kind == "generate":
        decided(f"{path}.pos", pos, horizon(step))
    if "turn" in scope and not all(isinstance(data.field(r, step.input), list) for r in cx.rows[dataset]):
        fail(f"{path}.pos.scope.turn", f"the input {data.input_name(step.input)} is text, not a chat: a turn scope "
             "names a chat's messages (a column of {'role', 'content'} lists, rendered through the template), and "
             "text has none; locate the message's words with a {'text': ...} or {'column': ...} scope")
    if "column" in scope:
        for i, column in enumerate(scope["column"] if isinstance(scope["column"], list) else [scope["column"]]):
            text_ref(cx, column, f"{path}.pos.scope.column" + (f"[{i}]" if isinstance(scope["column"], list) else ""),
                     dataset)
    ragged = positions.width(pos) is None
    dims = address.dims(site.component, site.layers, ragged)
    last = len(spec.ops) - len(gathers(spec)) - 1          # the last edit's index; -1 when none
    try:
        tree = bracketed(spec.ops)
    except Unbracketed as e:
        fail(f"{path}.ops[{e.at}]", str(e))
    around = {node["at"]: list(outer) for node, outer in visited(tree)}
    closes = {node["closed"]: node for node, _ in visited(tree) if node.get("closed") is not None}
    before: dict[int, list[str]] = {}                       # the layout each encode turned into coordinates
    changed = None                                          # the op that changed the layout, which no edit follows
    for i, o in enumerate(spec.ops):
        at, args = f"{path}.ops[{i}]", arguments(o)
        outer = around[closes[i]["at"]] if i in closes else around[i]
        if o.op == "readout" and not address.COMPONENTS[site.component]["stream"]:
            fail(at, f"readout projects a residual-stream value; {site.component} is not one")
        if o.op == "readout" and "hidden_size" in (site.select or {}) and "hidden_size" in dims:
            fail(at, narrowed_readout(f"{site.component} select hidden_size {(site.select or {})['hidden_size']!r}"))
        if edits(o) and changed is not None:
            fail(at, f"{o.op!r} is an edit, which writes back, and it follows {changed!r}, which turned the value "
                 f"into {dims}: there is no live value to write; an edit comes before it, or put the transform in "
                 "a later entry at this site")
        ref = args.get("featurizer")
        mine = next((n for n in outer if arguments(n["op"])["featurizer"] == ref), None) if i not in closes else None
        if mine is not None:
            fail(at, f"{o.op!r} of {ref!r} at ops[{i}] is inside the encode of {ref!r} (ops[{mine['at']}]), so it "
                 f"would act on {ref!r}'s own coordinates; put it before that encode or after the decode that closes it")
        kind = called(cx, plan, at, o.op, args)
        for key, value in args.items():
            if isinstance(value, str) and value in own and value not in live:
                fail(f"{at}.{key}", f"{value!r} is an entry of this generation; an operand from an entry of its own "
                     "step is served in a forward, not yet inside a generate" if step.kind == "generate" else
                     f"{value!r} is this entry or one this forward makes after it; an operand names an earlier "
                     "entry")
        found = references(cx, plan, at, o.op, args, "classes" in dims)
        aligned(at, (dataset, repr(step.input)), found)
        paired(at, o.op, dims, positions.width(pos), found)
        if i <= last:
            chosen = set(parameters(plan, o.op, args)) - set(ops.parameters(o.op))
            operands(at, found, dims, space_of(cx, [n["op"] for n in outer]), ragged, live, chosen)
        if i in closes:
            dims = before[closes[i]["at"]]
        elif o.op == "encode":
            before[i] = dims
            dims = layout(at, o.op, args, dims, "entry", kind)
        else:
            out = layout(at, o.op, args, dims, "entry", kind)
            changed = o.op if out != dims and changed is None else changed
            dims = out
    unclosed = [n for n, _ in visited(tree) if n["op"].op == "encode" and n["closed"] is None]
    if any(n["at"] < last for n in unclosed):
        j = max(n["at"] for n in unclosed if n["at"] < last)
        ref = arguments(spec.ops[j])["featurizer"]
        fail(f"{path}.ops[{last}]", f"the entry writes back here with the encode of {ref!r} (ops[{j}]) still open; "
             f"close it with {{'op': 'decode', 'featurizer': {ref!r}}} after its edits")
    record = {**spec.model_dump(exclude_defaults=True), "site": site.model_dump(exclude_defaults=True)}
    return {**record, "dims": dims, **({"space": space_of(cx, [n["op"] for n in unclosed])} if unclosed else {})}


def narrowed_readout(what: str) -> str:
    return (f"readout projects the whole residual stream through the model's final norm and head, and {what} keeps "
            "only some of hidden_size, which the norm cannot take; read the site whole, readout, and narrow what "
            "the logits give")


def space_of(cx: Compiling, encodes: list[Op]) -> list[dict]:
    """The coordinates a value is in: the encodes open around it, outermost first, each its featurizer, by the
    name it resolves to here (inside a fit's body `rot` and `<fit>.rot` are the one being trained), and the
    `select` it keeps; none for the site's own layout."""
    return [{k: cx.aliases.get(v, v) if k == "featurizer" else v for k, v in arguments(o).items()
             if k in ("featurizer", "select")} for o in encodes]


def said(space: list[dict]) -> str:
    """A value's coordinates as a message names them."""
    if not space:
        return "the site's own layout"
    return "the coordinates of " + " then ".join(repr(s["featurizer"]) + (f" (select {s['select']})" if "select" in s
                                                                       else "") for s in space)


def called(cx: Compiling, plan: Plan, path: str, name: str, args: dict) -> str | None:
    """The kind of the featurizer op `name`'s `Featurizer` argument names, checked to exist and to serve the op
    (`encode`, `decode`, `apply`); and the op's further arguments bound to the function its `options` choose for
    them, a `Value` among them a name. None when the op takes no featurizer."""
    ref = ops.featurizer_of(name, args)
    if ref is None:
        return None
    featurizer_ref(cx, ref, f"{path}.featurizer")
    kind = kind_of(plan, ref)
    serves = featurizers.KINDS[kind].ops()
    if name not in serves:
        fail(f"{path}.featurizer", f"{ref!r} is a {kind!r} featurizer, which has no {name}: it serves "
             f"{', '.join(serves)}" + ("; it has no way back into the value" if name == "decode" else ""))
    own = {k: v for k, v in args.items() if k not in ops.parameters(name)}
    try:
        fn = ops.options(name, args, kind)
        if fn is not None:
            article = "an" if kind[0] in "aeiou" or kind == "sae" else "a"        # "an sae", said letter by letter
            registry.check(fn, own, tuple(inspect.signature(fn).parameters)[:2], f"{article} {kind}'s `{fn.__name__}`")
    except ValueError as e:
        fail(path, f"featurizer {ref!r}: {e}")
    for key, p in ops.parameters(name, args, kind).items():
        if key in own and p.annotation == ops.Value and not isinstance(own[key], str):
            fail(f"{path}.{key}", f"{own[key]!r} is not a name; {key!r} names a value read at the positions the op "
                 "acts on")
    return kind


def references(cx: Compiling, plan: Plan, path: str, name: str, args: dict, classes: bool = False) -> dict[str, dict]:
    """Op `name`'s arguments that name a value or a column, each resolved to `{"dims", "data", "kind", "name"}`
    (a column is laid out `["rows"]`): a column read as text, or scored against the vocabulary, holds text on
    every row; one scored against a probe's `classes` holds its labels, whatever they are. A featurizer
    argument is checked to exist."""
    found = {}
    for key, p in parameters(plan, name, args).items():
        value, at = args.get(key), f"{path}.{key}"
        if p.annotation == ops.Value and isinstance(value, str):
            found[key] = {**value_ref(cx, value, at), "name": value}
        elif p.annotation in (ops.Tokens, ops.Text) and value is not None:
            ref = data.column_of(value)
            dataset = column_ref(cx, ref[0] if ref else value, at, None) if classes and p.annotation == ops.Tokens \
                else text_ref(cx, value, at, None)
            found[key] = {"dims": ["rows"], "data": dataset, "kind": "column", "name": ref[0] if ref else value}
        elif p.annotation == ops.Featurizer and value is not None:
            featurizer_ref(cx, value, at)
    return found


def aligned(path: str, of: tuple[str | None, str], found: dict) -> str | None:
    """The one dataset an op's rows align to: that of the value it acts on (`of`: its dataset, and how to name
    it), of its operands with a rows axis, and of its columns."""
    datasets = {of[0]: of[1]} if of[0] else {}
    for info in found.values():
        if info["data"] is not None and {"rows", "entries"} & set(info["dims"]):
            datasets.setdefault(info["data"], repr(info["name"]))
    if len(datasets) > 1:
        (a, x), (b, y) = list(datasets.items())[:2]
        fail(path, f"{x} has a row per example of {a!r} and {y} of {b!r}; an op takes one dataset's rows")
    return next(iter(datasets), None)


def paired(path: str, name: str, dims: list[str], width: int | None, found: dict) -> None:
    """A metric's operand (`kl`'s `against`, `cosine`'s `with`) is laid out as the value it scores, as many
    positions wide (`width`) when it has positions, or is one vector of its features, the same for every
    example; a ragged one's entries line up with nothing."""
    if ops.DECLARED[name]["group"] != "metrics":
        return
    one = [a for a in dims if a not in address.EXAMPLE_AXES]
    for key, info in found.items():
        if info["kind"] != "value":
            continue
        if "entries" in info["dims"]:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {name!r} scores a value laid out "
                 f"{dims}, and a ragged value's entries are its rows' positions, which pair with no other value's; "
                 "read both at a static index (pos -1, or {'index': '-3:'}), which makes them rectangular")
        if info["dims"] not in (dims, one):
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {name!r} scores a value laid out "
                 f"{dims}, so its {key!r} is laid out so too, or is one vector {one}")
        if "pos" in dims and info["dims"] == dims and info.get("width") != width:
            fail(f"{path}.{key}", f"{info['name']!r} holds {info.get('width')} position(s) per row and the value "
                 f"{name!r} scores holds {width}; position i scores against position i, so both are read at as "
                 "many (a pos axis broadcasts no more than a layers axis does)")


def operands(path: str, found: dict, own: list[str], space: list[dict], ragged: bool, live: dict, chosen: set) -> None:
    """Inside a write an operand stands for the value it meets, laid out `own` in the coordinates `space` (the
    encodes open around the op): in the same coordinates, encoded by the same featurizers with the same
    `select`, and that layout or one vector of its features; only the vector when the selection is ragged,
    unless the operand is an earlier entry of the same forward, which is gathered as the selection is. Such an
    entry is taken off the model as the forward reaches it, so its value is what its edits left (no op after its
    last edit), at one layer. An argument of the function an op's `options` choose (`chosen`: a lora's `with`)
    is laid out on the selection's rows and positions, whatever its features, which that function takes as it
    says."""
    lead = [a for a in own if a in address.EXAMPLE_AXES]
    for key, info in found.items():
        if info["kind"] != "value":
            continue
        if info["name"] in live:
            record, spec = live[info["name"]]
            if gathers(spec):
                fail(f"{path}.{key}", f"{info['name']!r} runs {gathers(spec)[0].op!r} once it is off the model, "
                     "after this forward; an operand from the same forward is an entry with no op after its edits")
            if address.stacks(record["site"].get("layers")):
                fail(f"{path}.{key}", f"{info['name']!r} is read at every layer; an operand from the write's own "
                     "forward is one layer's read")
        whole = not ragged or info["name"] in live
        if key in chosen:
            if [a for a in info["dims"] if a in address.EXAMPLE_AXES] != lead or not whole:
                fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {key!r} is "
                     f"laid out on this write's {lead}" + ("" if whole else ", and a ragged write's is an entry of "
                                                           "its own forward"))
            continue
        if info.get("space", []) != space:
            fail(f"{path}.{key}", f"{info['name']!r} is in {said(info.get('space', []))}; this "
                 f"{'bracket' if space else 'write'} is in {said(space)}, so an operand is encoded as it is")
        allowed = [[a for a in own if a not in address.EXAMPLE_AXES]] + ([own] if whole else [])
        if info["dims"] == own and not whole:
            fail(f"{path}.{key}", f"{info['name']!r} and this write are both ragged, {own}; what differs is that its "
                 "entries are another forward's positions, which need not pair with this one's: read both at a "
                 "static index (pos -1, {'index': '-3:'}), or reduce the operand to one vector")
        if "layers" in info["dims"] and "layers" not in own:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}: it keeps a layers axis, and this "
                 "write lands at one layer, so its operand has none; read the operand at the write's own layer")
        if info["dims"] not in allowed:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; this write's value is {own} here, "
                 f"so an operand is {' or '.join(map(str, allowed))}")


def layout(path: str, name: str, args: dict, dims: list[str], where: str, kind: str | None = None) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims` (for an op with a `layout`, for its
    featurizer's `kind`). Inside an entry (`where`) the op sees one batch, so it may not aggregate over rows;
    `original` is only an entry's."""
    if where != "entry" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside an entry")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args}, kind)
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and {"rows", "entries"} & (set(dims) - set(out)):
        fail(path, f"{name!r} aggregates over rows, and inside an entry it sees one batch; "
             "make it a step after the forward")
    return out


def made_at(site, n: int) -> list:
    """The layers a site is made at among the model's `n`: those its `layers` select, or `[None]` for a
    model's own value, which has none."""
    return [None] if site.layers is None else selectors.select(site.layers, n)


def picked(where: str, layers, n: int) -> list[int]:
    """The layers a site's `layers` select among the model's `n`, each one there: an int or a pick outside
    them, or a slice that picks none, is refused by name."""
    for i in [layers] if isinstance(layers, int) else layers if isinstance(layers, list) else []:
        if not -n <= i < n:
            raise ValueError(f"{where}: layer {i}; the model has {n} layers")
    try:
        got = selectors.select(layers, n)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    if not got:
        raise ValueError(f"{where}: layers {layers!r} pick none of the model's {n} layers")
    return got


def selected(where: str, axis: str, selector, n: int, what: str) -> None:
    """Refuses by name a `select` on an axis of `n` (`what` says which) with a pick outside it, or that keeps
    none of it."""
    written = [selector] if isinstance(selector, int) else selector if isinstance(selector, list) else []
    try:
        got = selectors.kept({axis: selector}, {axis: n})[axis]
    except ValueError as e:
        raise ValueError(f"{where}: select {axis}: {e}") from None
    if any(not -n <= i < n for i in written) or any(i >= n for i in got):
        raise ValueError(f"{where}: select {axis} {selector!r} outside {what}")
    if not got:
        raise ValueError(f"{where}: select {axis} {selector!r} keeps none of {what}")


def refuse(engine, plan: Plan) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name. `engine.unserved()`
    maps a component to None (served), why not, or why not per layer (nnterp's `support()` shape)."""
    reasons = engine.unserved()
    for name, step in walk(plan.steps):
        for key, record in entries(step).items():
            reason, layers = reasons.get(record.site.component), record.site.layers
            if isinstance(reason, dict):
                named = picked(f"{name}.{key}", layers, engine.sizes()["num_layers"])
                reason = next((r for layer, r in reason.items() if r and layer in named), None)
            if reason:
                raise ValueError(f"{name}.{key}: the {engine.name} engine does not serve {record.site.component}: "
                                 f"{reason}")


def check_operands(engine, plan: Plan, where: str, o, shape: tuple, step, site, found: dict, shapes: dict) -> None:
    """A write's operands that another entry selected: the same width and narrowing; on a `key` axis, inputs
    that line up token for token. One from an entry of the write's own forward is an earlier entry at the
    write's own site, or one at a site this engine reaches first (by site, not rank: the hooks engine ranks
    every component at one module boundary the same). An argument of the function an op's `options` choose (a
    lora's `with`) is as wide per position, its axes after the positions flattened, as the value the op acts on,
    and, from the same forward, read at the write's own positions."""
    args, sizes = arguments(o), engine.sizes()
    chosen = set(parameters(plan, o.op, args)) - set(ops.parameters(o.op))
    for key, p in parameters(plan, o.op, args).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        producer, record = entry_of(plan.steps, value, step.at)
        read = record.site
        same = value.partition(".")[0] == where.partition(".")[0]
        theirs_at, ours_at = ((s.component, made_at(s, sizes["num_layers"])[0]) for s in (read, site))
        if same and theirs_at != ours_at and engine.rank(*theirs_at) >= engine.rank(*ours_at):
            raise ValueError(f"{where}: operand {value!r} is read at {read.component} layer {read.layers}, which the "
                             f"{engine.name} engine's forward reaches no earlier than this write's {site.component} "
                             f"layer {site.layers}; an operand from the write's own forward is an earlier entry at "
                             "this site, or at a site the forward reaches first")
        if key in chosen:
            per, own = (math.prod(kept.get(a) or sizes[a] for a in address.dims(s.component, None, False)[2:])
                        for s, kept in ((read, theirs[1]), (site, shape[1])))
            if per != own:
                raise ValueError(f"{where}: operand {value!r} is {per} wide per position, its axes after the "
                                 f"positions flattened; this write's value is {own}")
            ours, at = (lookup(found["positions"], n, step.at) for n in (where, value)) if same else ([], [])
            for row, (a, b) in enumerate(zip(at, ours)):
                if a != b:
                    raise ValueError(f"{where}: operand {value!r} is read at positions {a} of row {row} and this "
                                     f"write lands at {b}; {key!r} from the write's own forward is read where it "
                                     "lands")
        if theirs[0] != shape[0] or (theirs[1] != shape[1] and key not in chosen):
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer, consumer = data.input_name(producer.input), data.input_name(step.input)
        if "key" in address.COMPONENTS[site.component]["dims"] and producer != consumer:
            for row, (a, b) in enumerate(zip(found["inputs"][producer]["rows"], found["inputs"][consumer]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {consumer} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
        written = positions.parse(entry_of(plan.steps, where, step.at)[1].pos)
        if producer != consumer and written["scope"] is None and positions.parse(record.pos) == written:
            windowed(engine, where, value, producer, consumer, found, step.at)


def windowed(engine, where: str, value: str, producer: str, consumer: str, found: dict, at: dict) -> None:
    """The same positions with no scope, read on one input and written on another, pair the operand's i-th
    position with the write's i-th (positions written differently pair what they say). On a row whose two prompts are as long, those count alike. On one where they are not, the
    check is conservative: each pair is the same distance from the start within the tokens the two prompts
    begin with alike, or from the end within those they end with alike, the first token past either counted
    in (both are the first token where the prompts part); any other pair may be two different tokens and is
    refused by name, as an attention pattern's keys are."""
    try:
        ours, theirs = lookup(found["positions"], where, at), lookup(found["positions"], value, at)
    except KeyError:
        return                  # beyond the prefill, where positions are calls
    tokens = engine.tokenizer.convert_ids_to_tokens
    for row, (a, b) in enumerate(zip(found["inputs"][producer]["rows"], found["inputs"][consumer]["rows"])):
        x, y = a["ids"], b["ids"]
        if len(x) == len(y):
            continue
        start = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        end = next((i for i, (p, q) in enumerate(zip(x[::-1], y[::-1])) if p != q), min(len(x), len(y)))
        for i, (p, q) in enumerate(zip(theirs[row], ours[row])):
            if not (p == q <= start or len(x) - p == len(y) - q <= end + 1):
                raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                 f"{len(x)} tokens; {consumer} has {len(y)}, so position {i} pairs "
                                 f"{tokens([x[p]])[0]!r} with {tokens([y[q]])[0]!r}; use a scope to name the token "
                                 "in each row")


def check_labels(plan: Plan, where: str, o, at: dict, ours: dict | None, dims: list[str], sizes: dict,
                 write: bool) -> None:
    """An op's operand holds, on each axis a read's site narrows (`num_heads`, `hidden_size`, ...), the indices the
    value the op acts on holds (`ours`, by axis, laid out `dims`: a write's selection, or what a read or an op
    step's `of` holds, `held`), in its order. A `select` list is positional: an operand read at heads `[1, 4]`
    swapped into a write at `[4, 1]`, or subtracted from a value read at `[4, 1]`, would mix the heads, and one
    head's value would broadcast over three. A number, or a value no read made, holds none to compare."""
    chosen = set(parameters(plan, o.op, arguments(o))) - set(ops.parameters(o.op))
    whose = "this write's selection" if write else "the value it acts on"
    for key, p in parameters(plan, o.op, arguments(o)).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or key in chosen:
            continue
        theirs, their_dims = held(plan, value, at, sizes), dims_of(plan, value)
        for axis, own in (ours or {}).items():
            if axis == "key" or axis not in dims or axis not in their_dims or theirs is None or axis not in theirs:
                continue
            if theirs[axis] == own:
                continue
            if sorted(theirs[axis]) == sorted(own):
                why = "a select list is positional, so they would trade places"
            elif len(theirs[axis]) == 1 and write:
                why = "an operand broadcasts over rows and positions only"
            else:
                why = "an operand holds the same ones"
            fix = f"select {axis} {own} at the read" if write else f"select one {axis} list at both reads"
            raise ValueError(f"{where}: operand {value!r} holds {axis} {theirs[axis]}, and {whose} holds "
                             f"{own}; {why}: {fix}")


def held(plan: Plan, name: str, at: dict, sizes: dict) -> dict[str, list[int]] | None:
    """The indices value `name` holds on each axis its read's site could narrow: what the site's `select` keeps,
    every index where it keeps them all; followed through op steps and into a fit's eval members. None for a
    value no read made (a fit's history, a generation's ids)."""
    thing, _, member = name.partition(".")
    step = agreeing(plan.steps[thing], at)
    if step.kind == "op":
        return held(plan, step.of, step.at, sizes)
    if step.kind == "fit":
        label, _, rest = member.partition(".")
        return held(body_view(plan, thing, step.at), rest, step.at, sizes) if label in step.eval and rest else None
    if not member:
        return None
    site = entry_of(plan.steps, name, at)[1].site
    chosen = selectors.kept(site.select, sizes)
    return {a: chosen[a] if a in chosen else list(range(sizes[a])) for a in address.selectable(site.component)
            if a in chosen or a in sizes}


def value_ref(cx: Compiling, value, path: str) -> dict:
    """A named value produced before this point."""
    info = cx.names.get(value) if isinstance(value, str) else None
    if info is not None and info["kind"] == "featurizer":
        fail(path, f"{value!r} is a featurizer, not a value; an op reaches it ({{'op': 'encode', 'featurizer': "
             f"{value!r}}} gives a value's coordinates, 'apply' its version of the value), and the value that entry "
             "binds is an operand")
    if info is None or info["kind"] != "value":
        known = sorted(k for k, v in cx.names.items() if v["kind"] == "value")
        fail(path, f"{value!r} names no value produced before this step; known: {', '.join(known) or 'none'}")
    cx.refs.append(value)
    cx.used.add(value)
    return info


def featurizer_ref(cx: Compiling, value, path: str) -> None:
    if not (isinstance(value, str) and cx.names.get(value, {}).get("kind") == "featurizer"):
        known = sorted(k for k, v in cx.names.items() if v["kind"] == "featurizer")
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")
    cx.refs.append(value)
