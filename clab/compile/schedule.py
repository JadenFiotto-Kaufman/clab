"""The forward as a compiled schedule (`Forward.schedule`): which points of a step one forward serves, decided by
compile from the forward's axes (`shared`), and what that forward does, written by preflight with the model in
hand (`scheduled`): the sites it visits in the engine's forward order, the records done at each, the site it stops
at, and a generation's calls and loop. The engines execute it and decide nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from clab import ops
from clab.compile import positions
from clab.format import address
from clab.format.sweeps import agreeing, labelled, points, projected
from clab.plan import Forward, Intervention, OpStep, Plan, Site, arguments, entries, entry_of, gathers, horizon, record_of, writes

if TYPE_CHECKING:
    from clab.engines import Engine


def shared(plan: Plan) -> None:
    """Each step's `schedule` as compile decides who serves whom: the points of a forward or generate step that
    agree on the forward's axes (`axes`, which `compile.forward` wrote), or of an op step on its value's axes, are
    served by the first of them, which lists them in `serves`, itself first; each other names it in `serving`. A
    fit body's step is one point per fit point, which serves itself."""
    for entry in plan.steps.values():
        first: dict[tuple, Forward | OpStep] = {}
        for step in points(entry):
            if step.kind == "fit":
                for body in step.steps.values():
                    body.schedule = {**body.schedule, "serves": [body.at], "sites": [], "stop": None} \
                        if isinstance(body, Forward) else {"serves": [body.at]}
                continue
            if not isinstance(step, (Forward, OpStep)):
                continue
            key = tuple(projected(step.at, step.schedule["axes"] if isinstance(step, Forward) else step.axes).items())
            if key in first:
                first[key].schedule["serves"].append(step.at)
                step.schedule = {"serving": first[key].at}
            elif isinstance(step, Forward):
                step.schedule = {**step.schedule, "serves": [step.at], "sites": [], "stop": None}
                first[key] = step
            else:
                step.schedule = {"serves": [step.at]}
                first[key] = step


def scheduled(engine: Engine, plan: Plan, name: str, step: Forward, trained: dict[str, dict[str, int]]) -> None:
    """What the forward of `step` (a serving point of step `name`, every point's records completed) does, written
    into its `schedule`: `sites`, each `(component, layer)` the forward visits in the engine's forward order
    (`engine.rank`), holding the records done there in order, entry by entry: the write (the serving point's: the
    points it serves write alike) with its ops up to its last edit, each operand `resolved`, then a read for each
    value the entry is at the served points (`gathered`: once per distinct point on the entry's axes, `at`), with
    the name it feeds (`<entry>@<layer>`, the serving point's own; another value's with its coordinates after it;
    `#0` on the part over the prompt); `stop`, the last site, where a forward ends (null for a generation, which
    runs whole); and for a generation each record's part (`parts`), the `loop` of calls the engine iterates (null:
    every one) and whether a write is found `online`. A write whose operand is an entry of the same forward read at
    another site is refused by name unless the engine ranks that site strictly before the write's (two sites the
    engine ranks alike are refused, whichever the order lists first); one read at the write's own site is done
    first, in entry order. `trained` is each featurizer the enclosing fit trains, by both its names, with the fit's
    point."""
    reading = gathered(plan, name, step)
    sites: dict[tuple[str, int | None], list[dict]] = {}
    names: dict[str, dict[int | None, str]] = {}        # the serving point's reads by `<step>.<entry>`, per layer
    for key, record in entries(step).items():
        component = cast(Site, record.site).component
        if writes(record):
            for layer in record.layers or [None]:
                for part in parts(step, record)[0]:
                    sites.setdefault((component, layer), []).append(
                        {"write": key, "ops": resolved(plan, name, record, layer, names, trained, step.at), **part})
        for (k, _), (theirs, own, p) in reading.items():
            if k != key:
                continue
            for layer in theirs.layers or [None]:       # its own record: a value is read at its positions and calls
                for part in parts(step, theirs)[1]:
                    read = f"{key}@{layer}" + ("#0" if part.get("prompt") else "") + \
                        ("" if p is step else f"@{labelled(plan, own)}")
                    sites.setdefault((component, layer), []).append({"read": key, "at": own, "name": read, **part})
                    if p is step and not part.get("prompt"):
                        names.setdefault(f"{name}.{key}", {})[layer] = read
    order = sorted(sites, key=lambda site: engine.rank(*site))
    reached = {r["name"]: site for site in order for r in sites[site] if "read" in r}
    for site in order:
        for r in sites[site]:
            for o in r.get("ops", []):
                for operand in o.values():
                    theirs = reached[operand["live"]] if isinstance(operand, dict) and "live" in operand else site
                    if theirs != site and engine.rank(*theirs) >= engine.rank(*site):
                        raise ValueError(unreached(engine, plan, name, r["write"], operand["name"], site[1], step.at))
    step.schedule.update(sites=[{"component": c, "layer": layer, "records": sites[c, layer]} for c, layer in order],
                         stop=list(order[-1]) if order and step.kind == "forward" else None)
    if step.kind == "generate":
        asked = [r["calls"] for site in sites.values() for r in site if "calls" in r]
        step.schedule.update(loop=None if None in asked else sorted({0, *(k for a in asked for k in a)}),
                             online=any(r.get("online") for site in sites.values() for r in site))


def gathered(plan: Plan, name: str, step: Forward) -> dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]]:
    """The values the forward of `step` (a serving point of step `name`) reads, by entry and the value's
    coordinates on the entry's own axes: each gathered entry of each point the forward serves, once per distinct
    point on its axes, with the record it is at the first such point (its positions, its `kept`, its ops), those
    coordinates, and that point."""
    out: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]] = {}
    for at in step.schedule["serves"]:
        p = agreeing(plan.steps[name], at)
        for key, record in entries(p).items():
            if record.gathered:
                own = projected(p.at, record.axes)
                out.setdefault((key, tuple(own.items())), (record, own, p))
    return out


def parts(step: Forward, record: Intervention) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """How an entry's records part over a generation's calls, a write's and a read's: one record, as in a forward,
    for an entry in the prefill; beyond it, a part over the prompt (`prompt`, where its frame holds it) and a part
    at its decode `calls` (the frame's own, or None for every call, as a write with a text scope is, which the run
    finds `online`)."""
    pos = positions.parse(record.pos)
    if step.kind != "generate" or positions.prefill(pos):
        return [{}], [{}]
    holds, calls, _ = positions.reach(pos, horizon(step))
    prompt: list[dict[str, Any]] = [{"prompt": True}] if holds else []
    online = pos["scope"] is not None
    return prompt + [{"calls": None if online else calls, "online": online}], prompt + [{"calls": calls}]


def resolved(plan: Plan, name: str, record: Intervention, layer: int | None, names: dict[str, dict[int | None, str]],
             trained: dict[str, dict[str, int]], at: dict[str, int]) -> list[dict[str, Any]]:
    """A write's ops up to its last edit, each by name with its named operands resolved: an entry of the same
    forward to the name of its read at this `layer` (`live`: at the write's layer where the operand stacks layers,
    else its one read), a value or a featurizer to its name and the point of it this point takes (`at`: the point's
    own coordinates on the operand's axes, those its record says, a declared featurizer's own, a trained one's the
    fit's)."""
    out = []
    for o in record.ops[:len(record.ops) - len(gathers(record))]:
        params, listed = ops.parameters(o.op), cast(dict[str, Any], {"op": o.op})
        for key, value in arguments(o).items():
            if key not in params or params[key].annotation not in (ops.Value, ops.Featurizer) or not isinstance(value, str):
                continue
            if value in names:
                stacked = "layers" in (record_of(plan.steps, value, at).dims or [])
                listed[key] = {"name": value, "live": names[value][layer] if stacked else next(iter(names[value].values()))}
            elif value in trained:
                listed[key] = {"name": value, "at": trained[value]}
            elif value in plan.featurizers:
                listed[key] = {"name": value, "at": agreeing(plan.featurizers[value], at).at}
            else:
                listed[key] = {"name": value, "at": projected(at, record_of(plan.steps, value, at).axes)}
        out.append(listed)
    return out


def unreached(engine: Engine, plan: Plan, name: str, key: str, value: str, layer: int | None, at: dict[str, int]) -> str:
    """Why a write cannot take an operand read at a site the forward reaches no earlier than the write's."""
    write, read = (cast(Site, entry_of(plan.steps, n, at)[1].site) for n in (f"{name}.{key}", value))
    made = layer if address.stacks(read.layers) else read.layers
    return (f"{name}.{key}: operand {value!r} is read at {read.component} layer {made}, which the {engine.name} "
            f"engine's forward reaches no earlier than this write's {write.component} layer "
            f"{layer if address.stacks(write.layers) else write.layers}; an operand from the write's own forward is "
            "an earlier entry at this site, or at a site the forward reaches first")
