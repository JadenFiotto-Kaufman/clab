"""Interventions: a step's entries, the brackets their ops nest in, an entry compiled (its site inlined, its
positions checked against its step, its ops' layouts and operands, its labels written as far as the document
decides them), and what preflight checks of them against the model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from clab import ops
from clab.compile import positions
from clab.compile.answers import check_text_ref, vocabulary
from clab.compile.arguments import (featurizer_kind, layout, mismatch, narrowed_readout_refusal, references, resolver,
                                    space_of)
from clab.compile.positions import check_calls_known
from clab.compile.state import Compiling, Findings
from clab.format import address, data, registry, selectors
from clab.format.selectors import Selector, keyed, keys, picked
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
    adapts,
    arguments,
    bracketed,
    documented,
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


def entries_with_paths(plan: Plan, path: str, name: str,
                       spec: Forward | Generate) -> list[tuple[str, str, Intervention]]:
    """A step's entries in the order written, each with its document path and name: the items of its
    `interventions` in turn, a bundle's entries where its name stands, rescoped to the step (`rescope_entry`)."""
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
        out += [(f"interventions.{item}.{k}", k,
                 rescope_entry(plan, f"interventions.{item}.{k}", e, item, name, pullers))
                for k, e in plan.interventions[item].items()]
    return out


def bundles_of(step: Step) -> list[str]:
    """The bundles a step's `interventions` pull in, by name."""
    value = getattr(step, "interventions", None)
    return [i for i in (value if isinstance(value, list) else [value]) if isinstance(i, str)]


def rescope_entry(plan: Plan, path: str, spec: Intervention, bundle: str, step: str, pullers: set) -> Intervention:
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


def compile_entry(cx: Compiling, path: str, spec: Intervention, plan: Plan, step: Forward | Generate, dataset: str,
                  live: dict[str, Intervention], own: set[str], where: str, at: dict[str, int]) -> Intervention:
    """One entry compiled: its site inlined, its positions checked against the step, its labels written as far as
    the document decides them (its layers, `kept_of`, the static width of its `pos`), and its ops walked once for
    their layouts and coordinates (each op's recorded on it) and for their operands, each checked against the value
    the op meets there (`arguments.mismatch`). Up to its last edit each op acts on the live value, so no edit
    follows an op that changed the layout unless that was an `encode` its `decode` closes, every encode before the
    write-back is closed by then, and on a component of ids (`address.COMPONENTS`' `ids`) the one edit is `swap`,
    whose ids preflight holds to the model's size and the run to it again as they land. `live` holds the forward's
    earlier entries by name (`<step>.<entry>`); `own` names every entry of the step; `where` is this one's name; `at`
    the step's point. Beyond a generation's prefill the leading ops after the last edit that are positionwise, keep
    `pos` and take no operand with a `pos` axis are counted into `per_call`, the ops the run applies at each call."""
    for written, record in [(path, spec)] + [(f"{path}.ops[{i}]", o) for i, o in enumerate(spec.ops)]:   # as written
        for key in sorted(record.model_fields_set & (set(type(record).model_fields) - set(documented(type(record))))):
            fail(f"{written}.{key}", f"{key!r} is a derived field, which compile writes; an entry sets none")
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    write = writes(spec)
    differentiates = [i for i, o in enumerate(spec.ops) if o.op == "grad"]
    if differentiates and step.kind == "generate":
        fail(f"{path}.ops[{differentiates[0]}]", "grad is taken in a forward, not a generate, whose calls feed it "
             "the tokens it chose, which no gradient passes through: take it in a forward over the prompt with the "
             "generated text appended (a list input)")
    for i in [i for i in differentiates if i > 0]:
        edit = next((o.op for o in spec.ops[:i] if edits(o)), None)
        fail(f"{path}.ops[{i}]", f"grad follows {edit!r}, an edit, and an entry that takes a gradient writes "
             "nothing: write in an entry of its own listed before this one at this site, and grad here "
             "differentiates what it wrote" if edit else
             f"grad is ops[{i}] here, and it is the first op of its entry: the entry's value is the gradient at "
             "its site, and the ops after grad act on that gradient")
    if write and len(site.select or {}) > 1:
        fail(f"{path}.site.select", f"an entry that writes selects on one axis; this one selects "
             f"{', '.join(site.select or {})}")
    if write and selectors.offset((site.select or {}).get("key")) is not None:
        fail(f"{path}.site.select.key", "keys relative to the query are a read's: a write selects keys counted "
             "from the start, the same at every query (0, [0, 2], \"0:3\")")
    pos = positions.parse(spec.pos)
    frame, scope = pos["frame"], pos["scope"] or {}
    if step.kind != "generate" and selectors.select(frame, 1) != [0]:
        fail(f"{path}.pos.frame", f"frame {frame!r}: a forward makes one call, the prefill; its frame picks that "
             "call (0, or ':', the whole timeline, which in a forward is that call)")
    if frame != 0 and "key" in address.COMPONENTS[site.component]["dims"]:
        fail(f"{path}.site", f"{site.component}'s key axis grows with every decode call; read it in the prefill")
    if scope.get("until") and scope.get("before") and not selectors.from_end(frame) and 0 in selectors.select(frame, 1):
        first = "at a generation's head the prompt's last, before which its prefill computes no head" \
            if step.kind == "generate" and site.component in positions.HEAD else "position 0, which none comes before"
        fail(f"{path}.pos.scope.before", f"frame {frame!r} holds the prompt, and until with before starts before the "
             f"frame's first token, {first}, so it names no position on any row; with before, until's frame is decode "
             "calls ('1:')")
    if write and scope.get("before") and not positions.prefill_only(pos):
        fail(f"{path}.pos.scope.before", "a write beyond the prefill lands as its call runs, before the token it "
             "would predict is chosen; a scope that is before is a read's there, or a write's in the prefill")
    if write and frame != 0 and step.kind == "generate":
        check_calls_known(f"{path}.pos", pos, horizon(step))
    if "turn" in scope and not all(isinstance(data.field(r, step.input), list) for r in cx.rows[dataset]):
        fail(f"{path}.pos.scope.turn", f"the input {data.input_name(step.input)} is text, not a chat: a turn scope "
             "names a chat's messages (a column of {'role', 'content'} lists, rendered through the template), and "
             "text has none; locate the message's words with a {'text': ...} or {'column': ...} scope")
    if "column" in scope:
        for i, column in enumerate(scope["column"] if isinstance(scope["column"], list) else [scope["column"]]):
            check_text_ref(cx, column,
                           f"{path}.pos.scope.column" + (f"[{i}]" if isinstance(scope["column"], list) else ""),
                           dataset)
    dims = address.dims(site.component)                     # every op acts at each layer the entry is made at
    last = len(spec.ops) - len(gathers(spec)) - 1          # the last edit's index; -1 when none
    try:
        tree = bracketed(spec.ops)
    except Unbracketed as e:
        fail(f"{path}.ops[{e.at}]", str(e))
    nodes = {node["at"]: node for node, _ in visited(tree)}
    around = {node["at"]: list(outer) for node, outer in visited(tree)}
    closes = {node["closed"]: node for node, _ in visited(tree) if node.get("closed") is not None}
    # The entry's labels, which every op's operand meets, and which its record carries: its `layers` None for a
    # model's own value, the layers a non-negative int or picks name, else `[]` until preflight resolves them.
    picks = [site.layers] if isinstance(site.layers, int) else site.layers if isinstance(site.layers, list) else None
    labels = {"dataset": dataset, "read": where,
              "layers": None if site.layers is None else list(picks) if picks and min(picks) >= 0 else [],
              "kept": kept_of(cast("dict[str, Selector] | None", site.select)), "width": positions.width(pos)}
    before: dict[int, list[str]] = {}                       # the layout each encode turned into coordinates
    changed = None                                          # the op that changed the layout, which no edit follows
    laid: list[Op] = []                                     # each op with the layout and coordinates of its value
    # The ops after the last edit that run at each call a generation reads the entry beyond its prefill: the leading
    # ones whose value at a position is its own to the bit, which keep `pos` and pair no operand by position.
    per_call, each = 0, step.kind == "generate" and not positions.prefill_only(pos)
    for i, o in enumerate(spec.ops):
        here, args = f"{path}.ops[{i}]", arguments(o)
        outer = around[closes[i]["at"]] if i in closes else around[i]
        if differentiates and i > 0 and o.op == "decode":
            fail(here, "decode closes an encode whose edits write back, and after grad nothing writes back: an "
                 "encode after grad reads the gradient's coordinates, and no decode closes it")
        if differentiates and i > 0 and "original" in registry.signature(ops.OPS[o.op]).parameters:
            fail(here, f"{o.op!r} restores the value the site had before its writes, and after grad the value is "
                 "a gradient off the model, which writes nothing back")
        if o.op == "readout" and (why := ops.transforms.unstreamed(site.component)):
            fail(here, why)
        if o.op == "readout" and "hidden_size" in (site.select or {}) and "hidden_size" in dims:
            fail(here, narrowed_readout_refusal(
                f"{site.component} select hidden_size {(site.select or {})['hidden_size']!r}"))
        if edits(o) and changed is not None:
            fail(here, f"{o.op!r} is an edit, which writes back, and it follows {changed!r}, which turned the value "
                 f"into {dims}: there is no live value to write; an edit comes before it, or put the transform in "
                 "a later entry at this site")
        if write and edits(o) and o.op != "swap" and (ids := address.COMPONENTS[site.component].get("ids")):
            fail(here, f"{o.op!r} would compute on {site.component}, the ids the model indexes its {ids} with, as "
                 "numbers, and the write would cast the result back to ids, truncated; an index takes a swap, "
                 "{'op': 'swap', 'with': <an id, or a named value of ids>}")
        ref = args.get("featurizer")
        mine = next((n for n in outer if arguments(n["op"])["featurizer"] == ref), None) if i not in closes else None
        if mine is not None:
            fail(here, f"{o.op!r} of {ref!r} at ops[{i}] is inside the encode of {ref!r} (ops[{mine['at']}]), so it "
                 f"would act on {ref!r}'s own coordinates; put it before that encode or after the decode that closes it")
        kind = featurizer_kind(cx, plan, here, o.op, args)
        if "layers" in (args.get("over") or []):
            fail(f"{here}.over", f"{o.op!r} over 'layers': inside an entry an op acts at each layer the entry is made "
                 "at, so it meets no layers axis; reduce over layers in a step after the forward, {'kind': 'op', "
                 f"'op': {o.op!r}, 'of': '<step>.<entry>', 'over': ['layers']}}")
        for key, value in args.items() if o.op != "grad" else ():       # grad's `of` is checked by its forward
            if isinstance(value, str) and value in own and value not in live:
                fail(f"{here}.{key}", f"{value!r} is an entry of this generation; an operand from an entry of its own "
                     "step is served in a forward, not inside a generate" if step.kind == "generate" else
                     f"{value!r} is this entry or one this forward makes after it; an operand names an earlier "
                     "entry")
        # What the op meets: the value as the op before it left it, in the coordinates of the encodes open here.
        ours = Labelled(dims=dims, space=space_of([n["op"] for n in outer]),
                        **{**labels, "kept": {a: v for a, v in labels["kept"].items() if a in dims},
                           "width": labels["width"] if "pos" in dims else None})
        operands = references(o.op, args, resolver(cx, here, "classes" in dims, at, live))
        for key, (written, theirs) in operands.items():
            if written in live and gathers(live[written]):
                fail(f"{here}.{key}", f"{written!r} runs {gathers(live[written])[0].op!r} once it is off the model, "
                     "after this forward; an operand from the same forward is an entry with no op after its edits")
            if written in live and i > last:
                fail(f"{here}.{key}", f"{written!r} is an entry of this forward, and {o.op!r} here runs once the value "
                     "is off the model, batch by batch, where an operand is a value made before this step; an "
                     "operand from the same forward is a write's: take the two in an op step after the forward")
            why = mismatch(ours, theirs, o.op, key, written, repr(step.input), i <= last, written in live,
                           adapts(plan, o))
            if why:
                fail(f"{here}.{key}", why)
        if i in closes:
            dims = before[closes[i]["at"]]
        elif o.op == "encode":
            before[i] = dims
            dims = layout(here, o.op, args, dims, "live" if i <= last else "entry", kind)
        else:
            out = layout(here, o.op, args, dims, "live" if i <= last else "entry", kind)
            changed = o.op if out != dims and changed is None else changed
            dims = out
        if i > last and each:
            each = ops.DECLARED[o.op]["positionwise"] and "pos" in dims and "rows" in dims and \
                not any("pos" in (theirs.dims or []) for _, theirs in operands.values())
            per_call += each
        open_after = outer + [nodes[i]] if o.op == "encode" else outer
        laid.append(Op(**o.model_dump(exclude_defaults=True), dims=dims,
                       space=space_of([n["op"] for n in open_after])))
    unclosed = [n for n, _ in visited(tree) if n["op"].op == "encode" and n["closed"] is None]
    if any(n["at"] < last for n in unclosed):
        j = max(n["at"] for n in unclosed if n["at"] < last)
        ref = arguments(spec.ops[j])["featurizer"]
        fail(f"{path}.ops[{last}]", f"the entry writes back here with the encode of {ref!r} (ops[{j}]) still open; "
             f"close it with {{'op': 'decode', 'featurizer': {ref!r}}} after its edits")
    return Intervention(**spec.model_dump(exclude_defaults=True, exclude={"site", "ops"}), site=site, ops=laid,
                        dims=(["layers"] if address.stacks(site.layers) else []) + dims, **labels,
                        space=space_of([n["op"] for n in unclosed]), per_call=per_call)


def kept_of(select: dict[str, Selector] | None) -> dict[str, selectors.Kept]:
    """A site's `select` as the indices it keeps on each axis, as far as the document decides them: those of a
    selector counted from the start (`keyed`: a non-negative int or picks, a front-bounded slice, every `key`
    selector), a key relative to the query's offsets (`selectors.relative`), else `[]` until preflight resolves it
    among the model's."""
    return {axis: selectors.relative(s) if selectors.offset(s) is not None else keys(s) if keyed(s) else []
            for axis, s in (select or {}).items()}


def read_positions(steps: dict, read: str, at: dict[str, int]) -> list | None:
    """The positions per row the read `read` holds at `at`, under its dataset, where they are known: the
    prefill's before any forward, a generation's decode calls' once it ran; else None."""
    record = record_of(steps, read, at)
    assert isinstance(record, Intervention), f"{read} is no read"
    return record.positions.get(cast(str, record.dataset))


# -- what preflight checks of the entries against the model ---------------------------------------------


def refuse_unserved(engine: Engine, plan: Plan) -> None:
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
                   findings: Findings) -> None:
    """A write's operands that another entry selected, beyond Ops' one rule (`arguments.mismatch`): from the
    write's own forward, where either holds no one width on every row, as many positions as the write on each row
    (that the forward reaches the operand's site first is checked where the schedule is written); an adapter's
    input (`apply`'s `with`) as wide per position as the value the op acts on, its axes after the positions
    flattened, and from the same forward read at the write's own positions; on a `key` axis, inputs that line up
    token for token, and an unscoped operand on another input, its window checked (`check_window`)."""
    args, sizes, site = arguments(o), engine.sizes(), cast(Site, record.site)
    adapter = adapts(plan, o)
    for key, p in ops.parameters(o.op).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str):
            continue
        theirs = record_of(plan.steps, value, step.at)
        if not isinstance(theirs, Intervention):
            continue                                # a step's value: no site, no positions of its own
        producer, read = entry_of(plan.steps, value, step.at)
        theirs_site = cast(Site, read.site)
        same = value.rpartition(".")[0] == where.rpartition(".")[0]
        if adapter and key == "with":
            per, own = (math.prod(selectors.width_of(kept[a]) if a in kept else sizes[a]
                                  for a in address.dims(s.component)[2:])
                        for s, kept in ((theirs_site, theirs.kept), (site, record.kept)))
            if per != own:
                raise ValueError(f"{where}: operand {value!r} is {per} wide per position, its axes after the "
                                 f"positions flattened; this write's value is {own}")
        if same and (adapter and key == "with" or None in (theirs.width, record.width)):
            ours, at = (read_positions(plan.steps, n, step.at) or [] for n in (where, value))
            for row, (a, b) in enumerate(zip(at, ours)):
                if adapter and key == "with" and a != b:
                    raise ValueError(f"{where}: operand {value!r} is read at positions {a} of row {row} and this "
                                     f"write lands at {b}; {key!r} from the write's own forward is read where it "
                                     "lands")
                if not isinstance(a, list):     # compile refused a ragged operand of another forward (Ops' one rule)
                    raise ValueError(f"{where}: operand {value!r} holds no position on row {row} ({positions.MISSING}: "
                                     f"its pos picks none of the row's tokens) and this write lands at {b} there; a "
                                     "write takes an operand read on every row it lands on")
                if len(a) != len(b):
                    raise ValueError(f"{where}: operand {value!r} holds {len(a)} position(s) on row {row} and this "
                                     f"write {len(b)}; position i meets position i, so an entry of the write's own "
                                     "forward without one width on every row is read at as many on each")
        producer_input, consumer = data.input_name(producer.input), data.input_name(step.input)
        if "key" in address.COMPONENTS[site.component]["dims"] and producer_input != consumer:
            for row, (a, b) in enumerate(zip(findings.inputs[producer_input]["rows"],
                                             findings.inputs[consumer]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer_input}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {consumer} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
        written = positions.parse(record.pos)
        if producer_input != consumer and written["scope"] is None and positions.parse(read.pos) == written:
            check_window(engine, plan, where, value, producer_input, consumer, findings, step.at)


def check_window(engine: Engine, plan: Plan, where: str, value: str, producer: str, consumer: str, findings: Findings,
                 at: dict) -> None:
    """The same unscoped positions read on one input and written on another pair position i with position i. On
    a row whose two prompts differ in length that is checked conservatively: each pair must sit at one distance
    from the start within the tokens both prompts begin with alike, or from the end within those they end with
    alike (the first token past either counted in); any other pair may be two different tokens and is refused."""
    ours, theirs = (read_positions(plan.steps, n, at) for n in (where, value))
    if ours is None or theirs is None:
        return                  # beyond the prefill, where positions are calls
    texts = vocabulary(engine.tokenizer)
    for row, (a, b) in enumerate(zip(findings.inputs[producer]["rows"], findings.inputs[consumer]["rows"])):
        x, y = a["ids"], b["ids"]
        if len(x) == len(y):
            continue
        start = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        end = next((i for i, (p, q) in enumerate(zip(x[::-1], y[::-1])) if p != q), min(len(x), len(y)))
        for i, (p, q) in enumerate(zip(theirs[row], ours[row])):
            if not (p == q <= start or len(x) - p == len(y) - q <= end + 1):
                raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                 f"{len(x)} tokens; {consumer} has {len(y)}, so position {i} pairs "
                                 f"{texts[x[p]]!r} with {texts[y[q]]!r}; use a scope to name the token "
                                 "in each row")
