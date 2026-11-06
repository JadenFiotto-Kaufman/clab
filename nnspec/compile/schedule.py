"""The forward as a compiled schedule (`Forward.schedule`): which point's forward serves each point of a step,
decided by compile from the forward's axes (`share_forwards`), and what that forward does, written by preflight with the
model in hand (`write_schedule`): the sites it visits in the engine's forward order, the records done at each, the site
it stops at, and a generation's calls and loop. The engines execute it and decide nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from nnspec import ops
from nnspec.compile import positions
from nnspec.format import address, selectors
from nnspec.format.sweeps import agreeing_point, coordinate_text, points, projection
from nnspec.plan import (Forward, Intervention, Plan, Site, arguments, entries, entry_of, gathers, gradient_of, horizon,
                        record_of, writes)

if TYPE_CHECKING:
    from nnspec.engines import Engine


def share_forwards(plan: Plan) -> None:
    """Who serves whom, as compile decides it: of the points of a forward or generate step that agree on the
    forward's axes (`axes`, which `compile.compile_forward` wrote), the first runs the forward, and each other's
    `schedule` is `serving`, that point's coordinates, alone. A fit body's step, and a fit's eval step, is one point per
    fit point, which runs its own."""
    for entry in plan.steps.values():
        first: dict[tuple, Forward] = {}
        for step in points(entry):
            if isinstance(step, Forward):
                key = tuple(projection(step.at, step.schedule["axes"]).items())
                if key in first:
                    step.schedule = {"serving": first[key].at}
                else:
                    first[key] = step


# How a run computes a forward's `logits` (`plan.run.readout`, `nnspec run --readout`), each with what it trades.
READOUTS = {
    "full": "the model's head over every position, the default: a read of `logits` takes the model's own value at the "
            "positions it names",
    "at-reads": "a forward's reads of `logits` that name fewer than every position are read out from the last block's "
                "output at those positions alone, with the engine's `readout`, and the forward stops before the head: "
                "the head's compute and memory at the positions read only, and numbers that move in the last bits "
                "with the head's shape",
}


def write_schedule(engine: Engine, plan: Plan, name: str, step: Forward, served: list[Forward]) -> None:
    """What the forward of `step` (a serving point of step `name`, with the points it serves, `served`, as
    `plan.serving` lists them, every point's records completed) does, written into its `schedule`: `sites`, each
    `(component, layer)` the forward visits in the engine's forward order (`engine.rank`), holding the records done
    there in order, entry by entry: the write (the serving point's: the points it serves write alike) with its ops
    up to its last edit, each operand resolved (`resolve_operands`), then a read for each value the entry is at the
    served points (`values_read_by`: once per distinct point on the entry's axes, `at`), with the name it feeds
    (`<entry>@<layer>`, the serving point's own; another value's with its coordinates after it; `#0` on the part over
    the prompt; beyond a generation's prefill, each part with `per_call`, how many of the ops after the entry's last
    edit run at each call it reads, where its record runs any); `stop`, the last site, where a forward ends (null for a
    generation, which runs whole); and for a generation each record's part (`parts`), the `loop` of calls the engine
    iterates (null: every one) and whether a write is found `online`. A write whose operand is an entry of the same
    forward read at another site is refused by name unless the engine ranks that site strictly before the write's (two
    sites the engine ranks alike are refused, whichever the order lists first); one read at the write's own site is done
    first, in entry order. A forward with `grad` entries also holds `grad`: for each distinct value its gathered `grad`
    reads differentiate (the `of` entry at the point each serves), the names of that value's reads (`of`, one per layer)
    and of the `grad` reads it serves (`wrt`), each `grad` read refused by name unless the engine ranks every site its
    `of` is read at strictly after its own. Under the run's readout `at-reads` a forward's reads of `logits` are done at
    the last block's output where they can be (`read_out_at_reads`)."""
    values_read = values_read_by(served)
    sites: dict[tuple[str, int | None], list[dict]] = {}
    names: dict[str, dict[int | None, str]] = {}        # the serving point's reads by `<step>.<entry>`, per layer
    # each value's reads, by entry and coordinates, with sites
    reads_by_value: dict[tuple, list[tuple[str, tuple]]] = {}
    for key, record in entries(step).items():
        component = cast(Site, record.site).component
        if writes(record):
            for layer in record.layers or [None]:
                for part in parts(step, record)[0]:
                    sites.setdefault((component, layer), []).append(
                        {"write": key, "ops": resolve_operands(plan, record, layer, names, step.at), **part})
        for (k, _), (theirs, own, p) in values_read.items():
            if k != key:
                continue
            for layer in theirs.layers or [None]:       # its own record: a value is read at its positions and calls
                for part in parts(step, theirs)[1]:
                    read = f"{key}@{layer}" + ("#0" if part.get("prompt") else "") + \
                        ("" if p is step else f"@{coordinate_text(plan, own)}")
                    each = {"per_call": theirs.per_call} if theirs.per_call else {}
                    sites.setdefault((component, layer), []).append({"read": key, "at": own, "name": read, **part,
                                                                     **each})
                    reads_by_value.setdefault((key, tuple(own.items())), []).append((read, (component, layer)))
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
                        raise ValueError(unreached_site_refusal(engine, plan, name, r["write"], operand["name"],
                                                                site[1], step.at))
    at_reads = plan.run is not None and plan.run.readout == "at-reads"
    if at_reads and step.kind == "forward" and read_out_at_reads(engine, served, values_read, sites):
        order = sorted(sites, key=lambda site: engine.rank(*site))
    step.schedule.update(sites=[{"component": c, "layer": layer, "records": sites[c, layer]} for c, layer in order],
                         stop=list(order[-1]) if order and step.kind == "forward" else None)
    for p in served:
        for key, record in entries(p).items():
            if (of := gradient_of(record)) is None:
                continue
            for site in sites_of(record):
                for there in sites_of(entries(p)[of]):
                    if engine.rank(*there) <= engine.rank(*site):
                        raise ValueError(f"{name}.{key}: grad's of {of!r} is read at {site_text(there)}, which the "
                                         f"{engine.name} engine's forward reaches no later than this entry's "
                                         f"{site_text(site)} (rank {engine.rank(*there)} against "
                                         f"{engine.rank(*site)}); of is an entry at a site the forward reaches "
                                         "after this one")
    differentiated: dict[tuple, dict[str, list[str]]] = {}
    for (key, own), (theirs, _, p) in values_read.items():
        if (of := gradient_of(theirs)) is not None:
            value = (of, tuple(projection(p.at, entries(p)[of].axes).items()))
            item = differentiated.setdefault(value, {"of": [n for n, _ in reads_by_value[value]], "wrt": []})
            item["wrt"] += [n for n, _ in reads_by_value[key, own]]
    if differentiated:
        step.schedule["grad"] = list(differentiated.values())
    if step.kind == "generate":
        asked = [r["calls"] for site in sites.values() for r in site if "calls" in r]
        step.schedule.update(loop=None if None in asked else sorted({0, *(k for a in asked for k in a)}),
                             online=any(r.get("online") for site in sites.values() for r in site))


def read_out_at_reads(engine: Engine, served: list[Forward], values_read: dict, sites: dict) -> bool:
    """Whether a forward's reads of `logits` move to the last block's output, done there and marked `readout` (the run
    takes the stream at each read's positions and reads it out there with the engine's `readout`: the final norm, the
    head, and what the model does after the head), so that the head runs at the positions read alone and the forward can
    stop before it; `sites` is changed in place where they do. They move where every record at `logits` is a read that
    names fewer than every position of the prompt and that no `grad` entry takes or differentiates, where nothing writes
    at a site the engine reaches after the last block's output (the final norm's, the head's), whose write the readout
    would not see, and where the engine serves the last block's `layer_output`, at whose end they are done, after every
    record there."""
    reads, last = sites.get(("logits", None), []), engine.sizes()["num_layers"] - 1
    differentiated = {gradient_of(r) for p in served for r in entries(p).values()}
    refused = engine.unserved()["layer_output"]
    if not reads or isinstance(refused, str) or last in (refused or {}):
        return False
    end = engine.rank("layer_output", last)
    if any("write" in r for site, records in sites.items() if engine.rank(*site) > end for r in records):
        return False

    def narrow(r: dict) -> bool:
        if "read" not in r:
            return False
        record = values_read[r["read"], tuple(r["at"].items())][0]
        pos = positions.parse(record.pos)
        return gradient_of(record) is None and r["read"] not in differentiated and not (
            pos["scope"] is None and selectors.every(pos["index"]))
    if not all(map(narrow, reads)):
        return False
    sites.setdefault(("layer_output", last), []).extend({**r, "readout": True} for r in sites.pop(("logits", None)))
    return True


def values_read_by(served: list[Forward]) -> dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]]:
    """The values a forward reads, by entry and the value's coordinates on the entry's own axes: each gathered
    entry of each point it serves (`served`, as `plan.serving` lists them), once per distinct point on its axes,
    with the record it is at the first such point (its positions, its `kept`, its ops), those coordinates, and that
    point."""
    out: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]] = {}
    for p in served:
        for key, record in entries(p).items():
            if record.gathered:
                own = projection(p.at, record.axes)
                out.setdefault((key, tuple(own.items())), (record, own, p))
    return out


def parts(step: Forward, record: Intervention) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """How an entry's records part over a generation's calls, a write's and a read's: one record, as in a forward,
    for an entry in the prefill; beyond it, a part over the prompt (`prompt`, where its frame holds it) and a part
    at its decode `calls` (the frame's own, or None for every call, as a write with a text scope is, which the run
    finds `online`)."""
    pos = positions.parse(record.pos)
    if step.kind != "generate" or positions.prefill_only(pos):
        return [{}], [{}]
    holds, calls, _ = positions.reach(pos, horizon(step))
    prompt: list[dict[str, Any]] = [{"prompt": True}] if holds else []
    online = pos["scope"] is not None
    return prompt + [{"calls": None if online else calls, "online": online}], prompt + [{"calls": calls}]


def resolve_operands(plan: Plan, record: Intervention, layer: int | None, names: dict[str, dict[int | None, str]],
                     at: dict[str, int]) -> list[dict[str, Any]]:
    """A write's ops up to its last edit, each by name with its named operands resolved: an entry of the same
    forward to the name of its read at this `layer` (`live`: at the write's layer where the operand stacks layers,
    else its one read), a value or a featurizer to its name and the point of it this point takes (`at`: the point's
    own coordinates on the operand's axes, those its record says, a declared featurizer's own; a trained one's, as
    its record says, are the fit's)."""
    out = []
    for o in record.ops[:len(record.ops) - len(gathers(record))]:
        params, listed = ops.parameters(o.op), cast(dict[str, Any], {"op": o.op})
        for key, value in arguments(o).items():
            if key not in params or params[key].annotation not in (ops.Value, ops.Featurizer) or not isinstance(value, str):
                continue
            if value in names:
                stacked = "layers" in (record_of(plan.steps, value, at, plan.featurizers).dims or [])
                listed[key] = {"name": value, "live": names[value][layer] if stacked else next(iter(names[value].values()))}
            elif value in plan.featurizers:
                listed[key] = {"name": value, "at": agreeing_point(plan.featurizers[value], at).at}
            else:
                listed[key] = {"name": value, "at": projection(at, record_of(plan.steps, value, at,
                                                                             plan.featurizers).axes)}
        out.append(listed)
    return out


def sites_of(record: Intervention) -> list[tuple[str, int | None]]:
    """The sites an entry is made at: its component at each of its layers (None for a model's own value)."""
    return [(cast(Site, record.site).component, layer) for layer in record.layers or [None]]


def site_text(site: tuple[str, int | None]) -> str:
    """A site as a refusal names it: its component, and its layer where it has one."""
    return site[0] if site[1] is None else f"{site[0]} layer {site[1]}"


def unreached_site_refusal(engine: Engine, plan: Plan, name: str, key: str, value: str, layer: int | None,
                           at: dict[str, int]) -> str:
    """Why a write cannot take an operand read at a site the forward reaches no earlier than the write's."""
    write, read = (cast(Site, entry_of(plan.steps, n, at)[1].site) for n in (f"{name}.{key}", value))
    made = layer if address.stacks(read.layers) else read.layers
    return (f"{name}.{key}: operand {value!r} is read at {read.component} layer {made}, which the {engine.name} "
            f"engine's forward reaches no earlier than this write's {write.component} layer "
            f"{layer if address.stacks(write.layers) else write.layers}; an operand from the write's own forward is "
            "an earlier entry at this site, or at a site the forward reaches first")
