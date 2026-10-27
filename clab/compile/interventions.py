"""Interventions: a step's entries, the brackets their ops nest in, an entry compiled (its site inlined, its
positions checked against its step, its ops' layouts and operands, its labels written as far as the document
decides them), and what preflight checks of them against the model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from clab import ops
from clab.compile import positions
from clab.compile.answers import text_ref
from clab.compile.arguments import called, layout, mismatch, narrowed_readout, references, space_of
from clab.compile.positions import decided
from clab.compile.state import Compiling, Found
from clab.format import address, data, selectors
from clab.format.selectors import Selector, keys, picked
from clab.plan import (
    Forward,
    Generate,
    Intervention,
    Labelled,
    Op,
    Plan,
    Site,
    Step,
    Unbracketed,
    arguments,
    bracketed,
    edits,
    entries,
    entry_of,
    fail,
    gathers,
    horizon,
    record_of,
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
          live: dict[str, Intervention], own: set[str], where: str, at: dict[str, int]) -> Intervention:
    """One entry compiled: its site inlined, its positions checked against the step, its ops walked for their
    layouts (each op's recorded on it), its labels written as far as the document decides them (`layers_of`,
    `kept_of`), then each op's operands checked against what it meets (`met`, `arguments.mismatch`). Up to its
    last edit each op acts on the live value, so no edit follows an op that changed the layout unless that was an
    `encode` its `decode` closes, and every encode before the write-back is closed by then. `live` holds the
    forward's earlier entries by name (`<step>.<entry>`); `own` names every entry of the step; `where` is this
    one's name; `at` the step's point."""
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    write = writes(spec)
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
    laid: list[Op] = []                                     # each op with the layout of its value
    for i, o in enumerate(spec.ops):
        here, args = f"{path}.ops[{i}]", arguments(o)
        outer = around[closes[i]["at"]] if i in closes else around[i]
        if o.op == "readout" and not address.COMPONENTS[site.component]["stream"]:
            fail(here, f"readout projects a residual-stream value; {site.component} is not one")
        if o.op == "readout" and "hidden_size" in (site.select or {}) and "hidden_size" in dims:
            fail(here, narrowed_readout(f"{site.component} select hidden_size {(site.select or {})['hidden_size']!r}"))
        if edits(o) and changed is not None:
            fail(here, f"{o.op!r} is an edit, which writes back, and it follows {changed!r}, which turned the value "
                 f"into {dims}: there is no live value to write; an edit comes before it, or put the transform in "
                 "a later entry at this site")
        ref = args.get("featurizer")
        mine = next((n for n in outer if arguments(n["op"])["featurizer"] == ref), None) if i not in closes else None
        if mine is not None:
            fail(here, f"{o.op!r} of {ref!r} at ops[{i}] is inside the encode of {ref!r} (ops[{mine['at']}]), so it "
                 f"would act on {ref!r}'s own coordinates; put it before that encode or after the decode that closes it")
        kind = called(cx, plan, here, o.op, args)
        if "layers" in (args.get("over") or []):
            fail(f"{here}.over", f"{o.op!r} over 'layers': inside an entry an op acts at each layer the entry is made "
                 "at, so it meets no layers axis; reduce over layers in a step after the forward, {'kind': 'op', "
                 f"'op': {o.op!r}, 'of': '<step>.<entry>', 'over': ['layers']}}")
        for key, value in args.items():
            if isinstance(value, str) and value in own and value not in live:
                fail(f"{here}.{key}", f"{value!r} is an entry of this generation; an operand from an entry of its own "
                     "step is served in a forward, not yet inside a generate" if step.kind == "generate" else
                     f"{value!r} is this entry or one this forward makes after it; an operand names an earlier "
                     "entry")
        if i in closes:
            dims = before[closes[i]["at"]]
        elif o.op == "encode":
            before[i] = dims
            dims = layout(here, o.op, args, dims, "entry", kind)
        else:
            out = layout(here, o.op, args, dims, "entry", kind)
            changed = o.op if out != dims and changed is None else changed
            dims = out
        laid.append(Op(**o.model_dump(exclude_defaults=True), dims=dims))
    unclosed = [n for n, _ in visited(tree) if n["op"].op == "encode" and n["closed"] is None]
    if any(n["at"] < last for n in unclosed):
        j = max(n["at"] for n in unclosed if n["at"] < last)
        ref = arguments(spec.ops[j])["featurizer"]
        fail(f"{path}.ops[{last}]", f"the entry writes back here with the encode of {ref!r} (ops[{j}]) still open; "
             f"close it with {{'op': 'decode', 'featurizer': {ref!r}}} after its edits")
    record = Intervention(**spec.model_dump(exclude_defaults=True, exclude={"site", "ops"}), site=site, ops=laid,
                          dims=(["layers"] if address.stacks(site.layers) else []) + dims, dataset=dataset,
                          layers=layers_of(cast("Selector | None", site.layers)), read=where,
                          kept=kept_of(cast("dict[str, Selector] | None", site.select)),
                          space=space_of(cx.aliases, [n["op"] for n in unclosed]))
    for i, o in enumerate(laid):
        here, ours = f"{path}.ops[{i}]", met(record, i, cx.aliases)
        found = references(cx, here, o.op, arguments(o), "classes" in (ours.dims or []), at, live)
        for key, (written, theirs) in found.items():
            if written in live and gathers(live[written]):
                fail(f"{here}.{key}", f"{written!r} runs {gathers(live[written])[0].op!r} once it is off the model, "
                     "after this forward; an operand from the same forward is an entry with no op after its edits")
            # The forward's own entries are not among the compiled steps yet: their width is their `pos`'s.
            widths = (positions.width(pos) if "pos" in (ours.dims or []) else None,
                      positions.width(positions.parse(live[written].pos)) if written in live
                      else width_of(cx.steps, theirs, at, {}))
            why = mismatch(ours, theirs, o.op, key, written, repr(step.input), i <= last, written in live, widths)
            if why:
                fail(f"{here}.{key}", why)
    return record


def layers_of(layers: Selector | None) -> list[int] | None:
    """A site's `layers` as the value's label, as far as the document decides it: None for a model's own value,
    the layers a non-negative int or picks name, else `[]` until preflight resolves it among the model's."""
    if layers is None:
        return None
    picks = [layers] if isinstance(layers, int) else layers if isinstance(layers, list) else None
    return list(picks) if picks is not None and all(i >= 0 for i in picks) else []


def kept_of(select: dict[str, Selector] | None) -> dict[str, list[int]]:
    """A site's `select` as the indices it keeps on each axis, as far as the document decides them: a `key`
    selector's keys, a non-negative int's or picks' own, else `[]` until preflight resolves it among the model's."""
    return {axis: keys(s) if axis == "key" else layers_of(s) or [] for axis, s in (select or {}).items()}


def met(record: Intervention, i: int, aliases: dict[str, str]) -> Labelled:
    """The value op `i` of an entry meets, at each layer the entry is made at: what the op before it left (the
    site's selection, before the first), in the coordinates of the encodes open around it, with the entry's labels
    on the axes it has."""
    tree = bracketed(record.ops)
    outer = {node["at"]: around for node, around in visited(tree)}
    closes = {node["closed"]: node["at"] for node, _ in visited(tree) if node.get("closed") is not None}
    dims = record.ops[i - 1].dims if i else address.dims(cast(Site, record.site).component)
    assert dims is not None
    return Labelled(dims=dims, dataset=record.dataset, layers=record.layers, read=record.read,
                    kept={a: v for a, v in record.kept.items() if a in dims},
                    space=space_of(aliases, [n["op"] for n in outer[closes[i] if i in closes else i]]))


def operands(plan: Plan, name: str, args: dict, at: dict[str, int]) -> dict[str, tuple[str, Labelled]]:
    """Op `name`'s arguments that name a value or a column, each as written and resolved to its record at `at`,
    inside the run, where every name is known to resolve (`arguments.references` does the same when the plan
    compiles, checking each)."""
    found = {}
    for key, p in ops.parameters(name).items():
        value = args.get(key)
        if p.annotation == ops.Value and isinstance(value, str):
            found[key] = (value, record_of(plan.steps, value, at))
        elif p.annotation in (ops.Tokens, ops.Text) and value is not None:
            ref = data.column_of(value)
            column = ref[0] if ref else str(value)
            found[key] = (column, Labelled(dims=["rows"], dataset=column.partition(".")[0]))
    return found


def resolved(steps: dict, read: str, at: dict[str, int], swap: dict[str, str]) -> list | None:
    """The positions per row the read `read` holds at `at`, under the dataset the run reads it over (`swap`, a
    fit body's substitution), where they are known: the prefill's before any forward, a generation's decode
    calls' once it ran; else None."""
    record = record_of(steps, read, at)
    assert isinstance(record, Intervention), f"{read} is no read"
    return record.positions.get(swap.get(cast(str, record.dataset), cast(str, record.dataset)))


def width_of(steps: dict, record: Labelled, at: dict[str, int], swap: dict[str, str]) -> Any:
    """How many positions a value holds per row, its read's: a tuple per row once they are resolved, else the
    static width its `pos` names, else None (rows differ, and the run has yet to resolve them); None too for a
    value with no `pos` axis."""
    if record.read is None or "pos" not in (record.dims or []):
        return None
    got = resolved(steps, record.read, at, swap)
    if got is not None:
        return tuple(len(p) if isinstance(p, list) else 0 for p in got)
    return positions.width(positions.parse(cast(Intervention, record_of(steps, record.read, at)).pos))


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


def check_operands(engine: Engine, plan: Plan, where: str, o: Op, step: Forward | Generate, record: Intervention,
                   found: Found) -> None:
    """A write's operands that another entry selected, beyond Ops' one rule (`arguments.mismatch`): from the
    write's own forward, read at a site this engine reaches first, at each layer the write is made at (where the
    operand stacks, its read at that layer); an adapter's input (`apply`'s `with`) as wide per position as the
    value the op acts on, its axes after the positions flattened, and from the same forward read at the write's
    own positions; on a `key` axis, inputs that line up token for token, and an unscoped operand on another input
    `windowed`."""
    args, sizes, site = arguments(o), engine.sizes(), cast(Site, record.site)
    adapts = o.op == "apply"
    for key, p in ops.parameters(o.op).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str):
            continue
        theirs = record_of(plan.steps, value, step.at)
        if not isinstance(theirs, Intervention):
            continue                                # a step's value: no site, no positions of its own
        producer, read = entry_of(plan.steps, value, step.at)
        theirs_site = cast(Site, read.site)
        same = value.partition(".")[0] == where.partition(".")[0]
        for layer in (record.layers or [None]) if same else []:
            made = layer if address.stacks(theirs_site.layers) else (theirs.layers or [None])[0]
            if (theirs_site.component, made) != (site.component, layer) and \
                    engine.rank(theirs_site.component, made) >= engine.rank(site.component, layer):
                raise ValueError(f"{where}: operand {value!r} is read at {theirs_site.component} layer "
                                 f"{made if address.stacks(theirs_site.layers) else theirs_site.layers}, which the "
                                 f"{engine.name} engine's forward reaches no earlier than this write's "
                                 f"{site.component} layer {layer if address.stacks(site.layers) else site.layers}; an "
                                 "operand from the write's own forward is an earlier entry at this site, or at a site "
                                 "the forward reaches first")
        if adapts and key == "with":
            per, own = (math.prod(len(kept[a]) if a in kept else sizes[a] for a in address.dims(s.component)[2:])
                        for s, kept in ((theirs_site, theirs.kept), (site, record.kept)))
            if per != own:
                raise ValueError(f"{where}: operand {value!r} is {per} wide per position, its axes after the "
                                 f"positions flattened; this write's value is {own}")
            ours, at = (resolved(plan.steps, n, step.at, found.swap) or [] for n in (where, value)) if same \
                else ([], [])
            for row, (a, b) in enumerate(zip(at, ours)):
                if a != b:
                    raise ValueError(f"{where}: operand {value!r} is read at positions {a} of row {row} and this "
                                     f"write lands at {b}; {key!r} from the write's own forward is read where it "
                                     "lands")
        producer_input, consumer = data.input_name(producer.input), data.input_name(step.input)
        if "key" in address.COMPONENTS[site.component]["dims"] and producer_input != consumer:
            for row, (a, b) in enumerate(zip(found.inputs[producer_input]["rows"], found.inputs[consumer]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer_input}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {consumer} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
        written = positions.parse(record.pos)
        if producer_input != consumer and written["scope"] is None and positions.parse(read.pos) == written:
            windowed(engine, plan, where, value, producer_input, consumer, found, step.at)


def windowed(engine: Engine, plan: Plan, where: str, value: str, producer: str, consumer: str, found: Found,
             at: dict) -> None:
    """The same unscoped positions read on one input and written on another pair position i with position i. On
    a row whose two prompts differ in length that is checked conservatively: each pair must sit at one distance
    from the start within the tokens both prompts begin with alike, or from the end within those they end with
    alike (the first token past either counted in); any other pair may be two different tokens and is refused."""
    ours, theirs = (resolved(plan.steps, n, at, found.swap) for n in (where, value))
    if ours is None or theirs is None:
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
