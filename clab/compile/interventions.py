"""Interventions: a step's entries, the brackets their ops nest in, an entry compiled (its site inlined, its
positions checked against its step, its ops' layouts and operands), and what preflight checks of them against
the model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from clab import ops
from clab.compile import positions
from clab.compile.answers import column_ref, text_ref
from clab.compile.arguments import aligned, called, layout, narrowed_readout, paired, references, said, space_of
from clab.compile.positions import decided
from clab.compile.state import Compiling, Found, Known
from clab.format import address, data, selectors
from clab.format.selectors import made_at, picked
from clab.format.sweeps import lookup
from clab.plan import (
    Forward,
    Generate,
    Intervention,
    Op,
    OpStep,
    Plan,
    Site,
    Step,
    Unbracketed,
    agreeing,
    arguments,
    body_view,
    boxes,
    bracketed,
    dims_of,
    edits,
    entries,
    entry_of,
    fail,
    gathers,
    horizon,
    visited,
    walk,
    writes,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def items(plan: Plan, path: str, name: str, spec: Forward | Generate) -> list[tuple[str, str, Intervention]]:
    """A step's entries in the order written, each with its document path and name: the items of its
    `interventions` in turn, a bundle's entries where its name stands, rescoped to the step (`scoped`)."""
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


def bundles_of(step: Step) -> list[str]:
    """The bundles a step's `interventions` pull in, by name."""
    value = getattr(step, "interventions", None)
    return [i for i in (value if isinstance(value, list) else [value]) if isinstance(i, str)]


def scoped(plan: Plan, path: str, spec: Intervention, bundle: str, step: str, pullers: set) -> Intervention:
    """A bundle's entry with an operand that names one of the bundle's own entries bare (`"premix"`) naming it in
    the step the bundle is pulled into (`"<step>.premix"`). Naming `<a step that pulls the bundle in>.<one of its
    entries>` is refused: pulled into another step it would still mean that step's entry."""
    own = plan.interventions[bundle]
    rewritten = []
    for i, o in enumerate(spec.ops):
        args, params, changes = arguments(o), ops.parameters(o.op), {}
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


def entry(cx: Compiling, path: str, spec: Intervention, plan: Plan, step: Forward | Generate, dataset: str,
          live: dict[str, tuple[dict, Intervention]], own: set[str]) -> tuple[dict, Known]:
    """One entry compiled, and what a name records of its value: its site inlined, its positions checked against
    the step, its ops walked for their layouts and operands. Up to its last edit each op acts on the live value,
    so no edit follows an op that changed the layout unless that was an `encode` its `decode` closes, and every
    encode before the write-back is closed by then. `live` holds the forward's earlier entries by name; `own`
    names every entry of the step."""
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    write = writes(spec)
    narrowed = [a for a, s in (site.select or {}).items() if not selectors.scoped(s)]
    if write and len(narrowed) > 1:
        fail(f"{path}.site.select", f"an entry that writes selects on one axis (beside keys a scope names); this one "
             f"selects {', '.join(narrowed)}")
    pos = positions.parse(spec.pos)
    frame, scope = pos["frame"], pos["scope"] or {}
    if step.kind != "generate" and selectors.select(frame, 1) != [0]:
        fail(f"{path}.pos.frame", f"frame {frame!r}: a forward makes one call, the prefill; its frame picks that "
             "call (0, or ':', the whole timeline, which in a forward is that call)")
    keyed = selectors.scoped((site.select or {}).get("key"))
    if frame != 0 and "key" in address.COMPONENTS[site.component]["dims"] and not (write and keyed):
        fail(f"{path}.site", f"{site.component}'s key axis grows with every decode call; " + (
            "a write beyond the prefill names the prompt's keys by a scope (select key {'scope': ...}), which keep "
            "their places there" if write else "read it in the prefill"))
    if write and scope.get("before") and not positions.prefill(pos):
        fail(f"{path}.pos.scope.before", "a write beyond the prefill lands as its call runs, before the token it "
             "would predict is chosen; a scope that is before is a read's there, or a write's in the prefill")
    if write and frame != 0 and step.kind == "generate":
        decided(f"{path}.pos", pos, horizon(step))
    scope_checks(cx, f"{path}.pos.scope", scope, step, dataset)
    if keyed:
        scope_checks(cx, f"{path}.site.select.key.scope",
                     positions.parse(cast(Site, site).select["key"])["scope"] or {}, step, dataset)  # type: ignore[index]
    width = positions.width(pos)                            # None when rows differ
    dims = address.dims(site.component)                     # every op acts at each layer the entry is made at
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
        if "layers" in (args.get("over") or []):
            fail(f"{at}.over", f"{o.op!r} over 'layers': inside an entry an op acts at each layer the entry is made "
                 "at, so it meets no layers axis; reduce over layers in a step after the forward, {'kind': 'op', "
                 f"'op': {o.op!r}, 'of': '<step>.<entry>', 'over': ['layers']}}")
        for key, value in args.items():
            if isinstance(value, str) and value in own and value not in live:
                fail(f"{at}.{key}", f"{value!r} is an entry of this generation; an operand from an entry of its own "
                     "step is served in a forward, not yet inside a generate" if step.kind == "generate" else
                     f"{value!r} is this entry or one this forward makes after it; an operand names an earlier "
                     "entry")
            if keyed and i <= last and isinstance(value, str) and key in ops.parameters(o.op) and \
                    ops.parameters(o.op)[key].annotation == ops.Value:
                fail(f"{at}.{key}", f"{value!r} names a value, and this write's keys are named by a scope, a "
                     "different number of them on each row; such a write takes a number as its operand")
        found = references(cx, at, o.op, args, "classes" in dims)
        aligned(at, (dataset, repr(step.input)), found)
        for key, info in found.items():
            if "layers" in info.dims and site.layers is None:
                fail(f"{at}.{key}", f"{info.name!r} is laid out {info.dims}: an op takes an operand's slice of each "
                     f"layer it acts at, and {site.component} is the model's own value, at no layer; read the "
                     "operand at one layer, or reduce it over layers")
        paired(at, o.op, dims, width, found)
        if i <= last:
            operands(at, found, dims, space_of(cx, [n["op"] for n in outer]), width is None, live, o.op == "apply")
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
    space = space_of(cx, [n["op"] for n in unclosed])
    dims = (["layers"] if address.stacks(site.layers) else []) + dims
    record = {**spec.model_dump(exclude_defaults=True), "site": site.model_dump(exclude_defaults=True), "dims": dims,
              **({"space": space} if space else {})}
    return record, Known(dims, dataset, space=space, select={a: v for a, v in (site.select or {}).items() if a in dims},
                         width=width if "pos" in dims else None)


def scope_checks(cx: Compiling, path: str, scope: dict, step: Forward | Generate, dataset: str) -> None:
    """What compile checks of a scope (an entry's `pos`'s, or a site's `select` on `key`) against the rows: a
    `turn` or an `image` needs a chat input, an `image` a chat with an image part on some row, and a column
    (`column`, or `region` naming one) is on every row, text or boxes."""
    chat = all(isinstance(data.field(r, step.input), list) for r in cx.rows[dataset])
    if "turn" in scope and not chat:
        fail(f"{path}.turn", f"the input {data.input_name(step.input)} is text, not a chat: a turn scope "
             "names a chat's messages (a column of {'role', 'content'} lists, rendered through the template), and "
             "text has none; locate the message's words with a {'text': ...} or {'column': ...} scope")
    if "image" in scope:
        if not chat:
            fail(f"{path}.image", f"the input {data.input_name(step.input)} is text, not a chat: an image comes "
                 "as a part of a chat message's content ({'type': 'image', 'image': '<file under data/>'}), and "
                 "text has none")
        if not any(data.image_parts(data.field(r, step.input)) for r in cx.rows[dataset]):
            fail(f"{path}.image", f"no message of {data.input_name(step.input)} has an image part on any row "
                 "({'type': 'image', 'image': '<file under data/>'} among a message's content parts)")
        region = scope.get("region")
        if isinstance(region, str):
            column_ref(cx, region, f"{path}.region", dataset)
            name, _, column = region.partition(".")
            for i, row in enumerate(cx.rows[name]):
                try:
                    boxes(data.lookup(row, column))
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    fail(f"{path}.region", f"{region!r} on row {i} of {name!r}: {e}")
    if "column" in scope:
        for i, column in enumerate(scope["column"] if isinstance(scope["column"], list) else [scope["column"]]):
            text_ref(cx, column, f"{path}.column" + (f"[{i}]" if isinstance(scope["column"], list) else ""), dataset)


def operands(path: str, found: dict[str, Known], own: list[str], space: list[dict], ragged: bool,
             live: dict[str, tuple[dict, Intervention]], adapts: bool) -> None:
    """Inside a write an operand stands for the value it meets at each layer, laid out `own` in the coordinates
    `space`, its own `layers` axis aside (`acts_at` says how it aligns): the same coordinates, and that layout or
    one vector of its features; only the vector when the selection is ragged, unless the operand is an earlier
    entry of the same forward, gathered as the selection is (no op after its last edit). An operand without a
    static width (its rows hold different numbers of positions) pairs with nothing but such an entry. An
    adapter's input (`apply`'s `with`, `adapts`) is laid out on the selection's rows and positions, whatever
    its features."""
    lead = [a for a in own if a in address.EXAMPLE_AXES]
    for key, info in found.items():
        if info.kind != "value":
            continue
        if info.name in live:
            _, spec = live[info.name]
            if gathers(spec):
                fail(f"{path}.{key}", f"{info.name!r} runs {gathers(spec)[0].op!r} once it is off the model, "
                     "after this forward; an operand from the same forward is an entry with no op after its edits")
        whole = not ragged or info.name in live
        if "pos" in info.dims and info.width is None and not (ragged and info.name in live):
            fail(f"{path}.{key}", f"{info.name!r} holds a different number of positions on each row, which "
                 + ("another forward's positions need not pair with" if ragged else "pair with no static width's")
                 + "; read it at a static index (pos -1, {'index': '-3:'}), or reduce it to one vector")
        dims = address.unstacked(info.dims)
        if adapts and key == "with":
            if [a for a in dims if a in address.EXAMPLE_AXES] != lead or not whole:
                fail(f"{path}.{key}", f"{info.name!r} is laid out {info.dims}; {key!r} is "
                     f"laid out on this write's {lead}" + ("" if whole else ", and a ragged write's is an entry of "
                                                           "its own forward"))
            continue
        if not dims:                        # one number (at each layer), broadcast in any coordinates as a literal
            continue
        if info.space != space:
            fail(f"{path}.{key}", f"{info.name!r} is in {said(info.space)}; this "
                 f"{'bracket' if space else 'write'} is in {said(space)}, so an operand is encoded as it is")
        allowed = [[a for a in own if a not in address.EXAMPLE_AXES]] + ([own] if whole else [])
        if dims not in allowed:
            fail(f"{path}.{key}", f"{info.name!r} is laid out {info.dims}; this write's value is {own} here, "
                 f"so an operand is {' or '.join(map(str, allowed))}" + (", beside a layers axis" if
                                                                         "layers" in info.dims else ""))


# -- what preflight checks of the entries against the model ---------------------------------------------


def refuse(engine: Engine, plan: Plan) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name (`engine.unserved()` maps
    a component to None, why not, or why not per layer)."""
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


def check_operands(engine: Engine, plan: Plan, where: str, o: Op, shape: tuple, step: Forward | Generate, site: Site,
                   found: Found, shapes: dict) -> None:
    """A write's operands that another entry selected: the same width and narrowing; on a `key` axis, inputs
    that line up token for token; from the write's own forward, read at a site this engine reaches first, at
    each layer the write is made at (where the operand stacks, its read at that layer). An adapter's input
    (`apply`'s `with`) is as wide per position as the value the op acts on, its axes after the positions
    flattened, and from the same forward is read at the write's own positions."""
    args, sizes, n = arguments(o), engine.sizes(), engine.sizes()["num_layers"]
    adapts = o.op == "apply"
    for key, p in ops.parameters(o.op).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        producer, record = entry_of(plan.steps, value, step.at)
        read = record.site
        same = value.partition(".")[0] == where.partition(".")[0]
        for layer in made_at(site, n) if same else []:
            made = layer if address.stacks(read.layers) else made_at(read, n)[0]
            if (read.component, made) != (site.component, layer) and \
                    engine.rank(read.component, made) >= engine.rank(site.component, layer):
                raise ValueError(f"{where}: operand {value!r} is read at {read.component} layer "
                                 f"{made if address.stacks(read.layers) else read.layers}, which the {engine.name} "
                                 f"engine's forward reaches no earlier than this write's {site.component} layer "
                                 f"{layer if address.stacks(site.layers) else site.layers}; an operand from the "
                                 "write's own forward is an earlier entry at this site, or at a site the forward "
                                 "reaches first")
        if adapts and key == "with":
            per, own = (math.prod(kept.get(a) or sizes[a] for a in address.dims(s.component)[2:])
                        for s, kept in ((read, theirs[1]), (site, shape[1])))
            if per != own:
                raise ValueError(f"{where}: operand {value!r} is {per} wide per position, its axes after the "
                                 f"positions flattened; this write's value is {own}")
            ours, at = (lookup(found.positions, n, step.at) for n in (where, value)) if same else ([], [])
            for row, (a, b) in enumerate(zip(at, ours)):
                if a != b:
                    raise ValueError(f"{where}: operand {value!r} is read at positions {a} of row {row} and this "
                                     f"write lands at {b}; {key!r} from the write's own forward is read where it "
                                     "lands")
        if theirs[0] != shape[0] or (theirs[1] != shape[1] and not (adapts and key == "with")):
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer, consumer = data.input_name(producer.input), data.input_name(step.input)
        if "key" in address.COMPONENTS[site.component]["dims"] and producer != consumer:
            for row, (a, b) in enumerate(zip(found.inputs[producer]["rows"], found.inputs[consumer]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {consumer} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
        written = positions.parse(entry_of(plan.steps, where, step.at)[1].pos)
        if producer != consumer and written["scope"] is None and positions.parse(record.pos) == written:
            windowed(engine, where, value, producer, consumer, found, step.at)


def windowed(engine: Engine, where: str, value: str, producer: str, consumer: str, found: Found, at: dict) -> None:
    """The same unscoped positions read on one input and written on another pair position i with position i. On
    a row whose two prompts differ in length that is checked conservatively: each pair must sit at one distance
    from the start within the tokens both prompts begin with alike, or from the end within those they end with
    alike (the first token past either counted in); any other pair may be two different tokens and is refused."""
    try:
        ours, theirs = lookup(found.positions, where, at), lookup(found.positions, value, at)
    except KeyError:
        return                  # beyond the prefill, where positions are calls
    tokens = engine.tokenizer.convert_ids_to_tokens
    for row, (a, b) in enumerate(zip(found.inputs[producer]["rows"], found.inputs[consumer]["rows"])):
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


def check_layers(plan: Plan, where: str, o: Op | OpStep, at: dict, layers: int | list[int] | None,
                 sizes: dict) -> None:
    """An op acts at `layers` (an entry's at each layer it is made at, a step's at `of`'s, `acts_at`) and takes an
    operand's slice of each where the operand has a `layers` axis: the operand holds every one of them, and an op
    that acts at no layer takes no such operand."""
    wanted = [layers] if isinstance(layers, int) else layers
    for key, p in ops.parameters(o.op).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or "layers" not in dims_of(plan, value):
            continue
        holds = (held(plan, value, at, sizes) or {}).get("layers")
        if wanted is None:
            raise ValueError(f"{where}: operand {value!r} has a layers axis, and this op acts at no layer (on a "
                             "model's own value, or one reduced over its layers); read the operand at one layer, or "
                             "reduce it over layers")
        missing = [layer for layer in wanted if layer not in (holds or [])]
        if missing:
            raise ValueError(f"{where}: operand {value!r} holds layers {holds}, and this op acts at layers {wanted}; "
                             f"at each layer it takes the operand's slice of that layer, and it holds none of "
                             f"{missing}: read it at those layers too")


def check_labels(plan: Plan, where: str, o: Op | OpStep, at: dict, ours: dict | None, dims: list[str], sizes: dict,
                 write: bool) -> None:
    """An op's operand holds, on each axis a read's site narrows, the same indices as the value the op acts on
    (`ours`, by axis: a write's selection, or what a read or an op step's `of` holds), in the same order: a
    `select` list is positional, so `[1, 4]` read into `[4, 1]` would trade the heads, and one head would
    broadcast over several. A number, or a value no read made, holds none to compare."""
    whose = "this write's selection" if write else "the value it acts on"
    for key, p in ops.parameters(o.op).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or (o.op == "apply" and key == "with"):
            continue
        theirs, their_dims = held(plan, value, at, sizes), dims_of(plan, value)
        for axis, own in (ours or {}).items():
            if axis in ("key", "layers") or axis not in dims or axis not in their_dims or theirs is None or \
                    axis not in theirs:
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
    """The indices value `name` holds on each axis its read's site could narrow: `layers`, the model layers its
    site picks (in order, the labels of a `layers` axis; none on a model's own value), and each axis a `select`
    narrows (every index where its `select` keeps them all); followed through op steps and into a fit's eval
    members; None for a value no read made."""
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
    layers = {} if site.layers is None else {"layers": cast(list[int], made_at(site, sizes["num_layers"]))}
    return {**layers, **{a: chosen[a] if a in chosen else list(range(sizes[a])) for a in address.selectable(
        site.component) if a in chosen or a in sizes}}



def acts_at(plan: Plan, name: str, at: dict, sizes: dict) -> int | list[int] | None:
    """The layers an op of value `name` acts at: the labels of its `layers` axis (`held`); the one layer a value
    without that axis was read at; None for a value at no layer (a model's own, or one reduced over its layers).
    An entry's op acts at each layer the entry is made at in turn."""
    picked = (held(plan, name, at, sizes) or {}).get("layers")
    if "layers" in dims_of(plan, name):
        return picked
    return picked[0] if picked is not None and len(picked) == 1 else None
