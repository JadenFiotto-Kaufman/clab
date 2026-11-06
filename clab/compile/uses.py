"""A featurizer's uses in a plan: every op that reaches it, the width it has at every site it is used at (a
featurizer over axes, the indices it keeps on each of them), the checks of a loaded bundle's header against those uses, the
instance made for the run, and the header a saved bundle carries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from clab.featurizers import KINDS, Kind, article, bundles, labels, make
from clab.format import address, registry, selectors
from clab.format.registry import Value
from clab.format.selectors import selected
from clab.format.sweeps import agreeing_point, lookup, points
from clab.plan import (
    Featurizer,
    Intervention,
    Op,
    OpStep,
    Plan,
    Site,
    arguments,
    declared,
    entry_of,
    fail,
    featurizer_uses,
    held_rows,
    kind_of,
    made_by,
    over_of,
    produced,
    producer,
    projections,
    record_of,
    runnable,
    scope,
    tensor_of,
    walk,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def uses(plan: Plan) -> dict[str, list[tuple[str, Intervention, dict, Op | OpStep, list[Op]]]]:
    """Every `encode` and `apply` of a declared featurizer, by featurizer: the entry's dotted name, the read it is
    of (the entry's own record; an op step's the read its `of` names, whose site and coordinates its featurizer is
    used at), its step's point, the op, and the encodes still open around it (outermost first), whose coordinates
    it is applied to (`plan.featurizer_uses`, in the order the steps are written)."""
    by_featurizer: dict[str, list] = {}
    for name, *use in featurizer_uses(plan.steps, plan.featurizers):
        by_featurizer.setdefault(name, []).append(tuple(use))
    return by_featurizer


def check_bundle_sites(plan: Plan) -> None:
    """A bundle's header names the component it was made at, the layers, counted from the first
    (`bundles.recorded`), and the indices its site's `select` kept (`bundles.selection`; a header without one was
    made at the whole component); every site its featurizer is used at is that component, each layer it picks is
    one of those, and its selection keeps those indices (`select_mismatch`). A layer counted from the last, a site
    that picks several, and a `select` the document alone does not resolve (counted from the end, a slice open at
    its end) are the model's to resolve, and preflight checks each layer and the selection. A use at a point of a
    sweep is refused naming the point."""
    from clab.compile.preflight import naming_point      # the stage whose refusals name their point
    for name, records in uses(plan).items():
        for where, read, at, _, _ in records:
            with naming_point(plan, at):
                site, f = cast(Site, read.site), agreeing_point(plan.featurizers[name], at)
                if f.header is None:
                    continue
                known = site.layers is None or isinstance(site.layers, int) and site.layers >= 0
                for key, ok in (("component", f.header.get("component") == site.component),
                                ("layer", not known or site.layers in bundles.recorded(f.header.get("layer")))):
                    if not ok:
                        fail(f"featurizers.{name}.file_path", f"{(f.model_extra or {})['file_path']}: header "
                             f"{key!r} is {f.header.get(key)!r}; this document uses {name!r} at {site.component} "
                             f"layer {site.layers} ({where})")
                why = select_mismatch(name, f, read) if all(read.kept.values()) else None
                if why:
                    fail(f"featurizers.{name}.file_path", f"{why} ({where})")


def select_mismatch(name: str, f: Featurizer, read: Intervention) -> str | None:
    """Why the bundle `f` loads may not be used at `read`: its header's `select` is not the indices `read`'s site
    keeps on each axis it selects (`Intervention.kept`), with no `select` the whole component; None where they are
    the same, and for a featurizer with no clab header. An axis a gate's header records as `kept.<axis>` is that
    record's to check (`check_gate_axes`). A bundle trained at one
    head applied at another would run and mean nothing."""
    gated = {key.partition(".")[2] for key in f.header or {} if key.startswith("kept.")}
    used = {a: v for a, v in read.kept.items() if a not in gated}
    if f.header is None or {a: v for a, v in bundles.selection(f.header).items() if a not in gated} == used:
        return None
    site, file = cast(Site, read.site), (f.model_extra or {})["file_path"]
    said = f"header 'select' is {f.header['select']!r}" if "select" in f.header else \
        "header has no 'select': it was made at the whole component"
    keeps = f"whose selection keeps {', '.join(f'{a} {v}' for a, v in used.items())}" if used else \
        "which selects nothing"
    return f"{file}: {said}; this document uses {name!r} at {site.component} layer {site.layers!r}, {keeps}"


def check_bundle_shapes(plan: Plan) -> None:
    """A loaded clab bundle's tensors have the shapes one drawn from its header has, every kind's: drawn at the
    header's `width` from the kind's fields it records (a subspace's `k`, an sae's `latents`, a lora's `rank`, a probe's
    `classes`, a pca's `k`; a field a draw takes that no header records, a constant's `init`, draws no other shape),
    and for a featurizer over axes at the indices it records (`layer`, `kept.<axis>`): a gate's `theta` as many units
    on each axis as it records indices there, or `[width]` on the last axis of its site's selection, an affine's `W`
    and `b` a map per layer at that width (`Kind.stacked`). The draw is made on the meta device, which holds no data,
    and a refusal names the tensor and the `file_path`. A header that records no `width`, or none of a field of its
    kind's header the draw needs (one it is drawn from or that counts its coordinates: a subspace's or a pca's `k`, an
    sae's `latents`, a lora's `rank` and `alpha`, a probe's `classes`), is refused by name, naming the file, the field
    and the kind: clab writes every one. A published file has no header, and its kind reads its shapes
    (`Kind.published`)."""
    for name, spec in plan.featurizers.items():
        for f in points(spec):
            if f.bundle is None or f.header is None:
                continue
            kind, over, file = KINDS[f.kind], over_of(f), (f.model_extra or {})["file_path"]
            needed = ["width", *(k for k in kind.header if k in (*kind.drawn, *kind.counts, "classes"))]
            if (lacking := next((k for k in needed if k not in f.header), None)) is not None:
                fail(f"featurizers.{name}.file_path", f"{file}: header records no {lacking!r}, which "
                     f"{article(f.kind)} {f.kind} bundle records")
            width = f.header["width"]
            fields = {k: bundles.value(f.header[k]) for k in kind.header if k in f.header and k != "classes"}
            fields.update({k: 0 for k in kind.drawn if k not in kind.header})     # a constant's init: any number
            taken = registry.signature(kind).parameters
            fields.update({"over": over} if "over" in taken else {}, **{k: (f.model_extra or {})[k] for k in
                                                                         ("classes", "file_path") if k in taken})
            keys = {axis: "layer" if axis == "layers" else f"kept.{axis}" for axis in over or []}
            recorded = [bundles.recorded(f.header.get("layer")) if axis == "layers" else
                        json.loads(f.header.get(key) or "[]") for axis, key in keys.items()]
            classes = bundles.value(f.header["classes"]) if "classes" in f.header else None
            with torch.device("meta"):
                drawn = make(f.kind, fields, int(width), None, None, classes, device="meta",
                             kept=cast("dict[str, list[int]]", dict(zip(over or [], recorded))) if over else None)
            for tensor in kind.saves:
                shape, laid_out = list(f.bundle[tensor].shape), list(getattr(drawn, tensor).shape)
                if shape == laid_out:
                    continue
                if over is not None:
                    said = ", ".join(f"{key} {f.header.get(key)}" for key in keys.values()) + \
                        (f", width {width}" if kind.stacked else "")
                    fail(f"featurizers.{name}.file_path", f"{file}: {tensor!r} is {shape}, and its header lays "
                         f"{article(f.kind)} {f.kind} over {over} out on {said}: {tensor!r} is {laid_out}, one "
                         f"{'map' if kind.stacked else 'unit'} per index it keeps on each axis")
                said = ", ".join([f"width {width}"] + [f"{k} {f.header[k]}" for k in kind.header
                                                        if k in f.header and k != "over"])
                fail(f"featurizers.{name}.file_path", f"{file}: {tensor!r} is {shape}, and its header lays "
                     f"{article(f.kind)} {f.kind} out at {said}: {tensor!r} is {laid_out}")


def check_gate_axes(plan: Plan) -> None:
    """A featurizer over axes (a gate's units, an affine's maps) is used by `apply` alone, on its site's own layout
    (inside no `encode`, whose coordinates are no axes of the site's), whose selection has each axis it names, in
    the selection's order, `layers` only where the site picks several. A refusal names the featurizer's `over` and
    the use, and its point where the use is at one of a sweep. A loaded one's tensors are checked against its header
    before (`check_bundle_shapes`)."""
    from clab.compile.preflight import naming_point      # the stage whose refusals name their point
    for name, records in uses(plan).items():
        for where, read, at, o, around in records:
            with naming_point(plan, at):
                over = over_of(agreeing_point(plan.featurizers[name], at))
                if over is None:
                    continue
                path, site = f"featurizers.{name}.over", cast(Site, read.site)
                # What the gate meets at a use made before any encode: inside an entry its site's selection, a `layers`
                # axis first where the site picks several; as a step, its `of`'s layout.
                laid = list(read.dims or []) if isinstance(o, OpStep) else address.dims(site.component, site.layers)
                if o.op != "apply":
                    fail(path, f"{over}: {where} encodes {name!r}, and a gate over axes serves apply alone: its units "
                         "are not the last axis, which an encode makes its coordinates; an interchange through it is "
                         f"{{'op': 'apply', 'featurizer': {arguments(o)['featurizer']!r}, 'with': <the other value>}}")
                if around:
                    fail(path, f"{over}: {where} applies {name!r} inside the encode of "
                         f"{arguments(around[-1])['featurizer']!r}, whose coordinates are no axes of its site's "
                         "selection; apply it outside the encode")
                for axis in over:
                    if axis == "layers" and axis not in laid:
                        fail(path, f"{over}: {where} is at {site.component} layer {site.layers!r}, one layer, whose "
                             f"selection {laid} has no layers axis; leave layers out of over, or pick the layers as a "
                             "list or a slice")
                    if axis not in laid:
                        fail(path, f"{over}: {where}'s selection at {site.component} is laid out {laid}, which has no "
                             f"{axis!r}")
                if [a for a in laid if a in over] != over:
                    fail(path, f"{over}: a gate's units are laid out on its axes in the selection's order, and "
                         f"{where}'s is {laid}: write over as {[a for a in laid if a in over]}")


def units(where: str, name: str, f: Featurizer, read: Intervention, over: list[str], sizes: dict[str, int]
          ) -> dict[str, list]:
    """The indices a featurizer over axes keeps on each of them at one use: the layers the use is made at, and on each
    other axis those its selection keeps (every one of an axis it keeps whole). A loaded one's are its header's
    (`layer`, and `kept.<axis>` for each other axis), which each use's other axes are checked against; each layer
    the use is made at is checked to be one `layer` names, as every loaded bundle's is."""
    kept = {}
    for axis in over:
        if axis == "layers":
            kept[axis] = list(read.layers or [])
        elif axis in read.kept:
            kept[axis] = list(read.kept[axis])
        elif sizes.get(axis) is None:
            raise ValueError(f"{where}: {cast(Site, read.site).component}'s {axis} axis grows with the input; "
                             f"gate {name!r} lays its units on axes of a fixed size")
        else:
            kept[axis] = list(range(sizes[axis]))
    if f.header is None:
        return kept
    file = (f.model_extra or {})["file_path"]
    recorded: dict[str, list] = {"layers": bundles.recorded(f.header.get("layer"))} if "layers" in over else {}
    for axis in over:
        if axis == "layers":
            continue
        written = f.header.get(f"kept.{axis}")
        recorded[axis] = json.loads(written) if written is not None else []
        if recorded[axis] != kept[axis]:
            raise ValueError(f"{where}: {file}: header 'kept.{axis}' is {written!r}; this document uses {name!r} "
                             f"at {cast(Site, read.site).component}, whose selection keeps {axis} {kept[axis]}")
    return recorded


def write_widths(engine: Engine, plan: Plan) -> None:
    """Each featurizer's width, written on its record (`Featurizer.width`, at every point): the size of the last
    axis of every site it is used at, which is one size, and the width its bundle's header says. A `select` on
    its features stays within them, and a featurizer a fit trains is used somewhere at each of the fit's points (a sweep
    in the body over featurizers it trains leaves the others unused at each point). A featurizer over axes keeps on
    each of them the indices every site it is used at selects, which are one selection, or that its bundle's header
    records (`units`), written on its record too (`Featurizer.kept`): a gate over axes is as wide as it has units,
    in one order at every use, and one that stacks its tensors over them (`Kind.stacked`: an affine over layers) as
    its sites' last axis, its uses listing those indices in any order, since it takes each one's slice by its id
    (the order of the first use is the order its tensors and its bundle's header lay them out in)."""
    from clab.compile.preflight import naming_point      # the stage whose refusals name their point
    sizes, out, laid, used_at = engine.sizes(), {}, {}, {}
    for name, used in uses(plan).items():
        for where, read, at, o, around in used:
            used_at.setdefault(name, []).append(at)
            with naming_point(plan, at):           # a use at a point of a sweep is refused naming it
                site, kept = cast(Site, read.site), read.kept
                f = agreeing_point(plan.featurizers[name], at)
                over = over_of(f)
                if over is not None:
                    units_kept, stacked = units(where, name, f, read, over, sizes), KINDS[f.kind].stacked
                    first = laid.setdefault(name, units_kept)
                    # A stacked kind takes each index's slice by its id, so a use may list them in any order; a gate's
                    # units on an axis but `layers` meet the value's in its selection's order.
                    if (first != units_kept if not stacked else
                            {a: sorted(v) for a, v in first.items()} != {a: sorted(v) for a, v in units_kept.items()}):
                        other = ", ".join(f"{a} {v}" for a, v in first.items())
                        lies = "it holds one map per index of" if stacked else "its units lie on"
                        raise ValueError(f"{where}: {f.kind} {name!r} is over "
                                         f"{', '.join(f'{a} {v}' for a, v in units_kept.items())} "
                                         f"here and {other} elsewhere; {lies} one selection of the axes it is "
                                         "over, so every site it is used at selects the same ones"
                                         + ("" if stacked else ", in the same order"))
                last = address.dims(site.component)[-1]
                width = math.prod(map(len, laid[name].values())) if over is not None and not KINDS[f.kind].stacked \
                    else selectors.width_of(kept[last]) if last in kept else sizes.get(last)
                if width is None:
                    raise ValueError(f"{where}: {site.component}'s last axis is {last}, which grows with the input; "
                                     f"featurizer {name!r} needs a feature axis")
                for outer in around:
                    ref = declared(plan.featurizers, arguments(outer)["featurizer"])
                    width = instance(plan, agreeing_point(plan.featurizers[ref], at), width, None).size()
                    chosen = (arguments(outer).get("select") or {}).get("features")
                    width = width if chosen is None else len(selectors.select(chosen, width))
                if out.setdefault(name, width) != width:
                    raise ValueError(f"{where}: featurizer {name!r} is used at width {width} here and {out[name]} "
                                     "elsewhere; a featurizer has one width")
                for field in KINDS[f.kind].counts:
                    count = (f.model_extra or {}).get(field)
                    if isinstance(count, int) and count > width:
                        raise ValueError(f"{where}: featurizer {name!r} has {field} {count}, and it is used at width "
                                         f"{width} ({site.component}), which has no more than {width} directions; "
                                         f"{field} is at most {width}")
                stated = (f.model_extra or {}).get("site")
                for layer in read.layers or [None]:
                    which = f"picks layer {layer}" if address.stacks(site.layers) else f"is layer {layer}"
                    if stated is not None and \
                            (stated["component"], selectors.select(stated["layers"], sizes["num_layers"])[0]) != \
                            (site.component, layer):
                        raise ValueError(f"{where}: featurizer {name!r} was trained at {stated['component']} layer "
                                         f"{stated['layers']} (its site); this use is {site.component} layer "
                                         f"{site.layers!r}"
                                         + (f", which {which}" if address.stacks(site.layers) else ""))
                    if f.header is not None and site.layers is not None and \
                            layer not in bundles.recorded(f.header.get("layer")):
                        raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'layer' is "
                                         f"{f.header.get('layer')!r}; this document uses {name!r} at {site.component} "
                                         f"layer {site.layers!r}, which {which}")
                    check_input_site(plan, where, at, f, o, layer)
                why = select_mismatch(name, f, read)
                if why:
                    raise ValueError(f"{where}: {why}")
                if f.header is not None and f.header.get("width") != str(width):
                    raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'width' is "
                                     f"{f.header.get('width')!r}; this document uses {name!r} at {site.component}, "
                                     f"whose width is {width}")
                if f.header is None and f.bundle is not None and KINDS[f.kind].width(f.bundle) != width:
                    raise ValueError(f"{where}: {(f.model_extra or {})['file_path']} has no clab header and is "
                                     f"{KINDS[f.kind].width(f.bundle)} wide; this document uses {name!r} at "
                                     f"{site.component}, whose width is {width}")
                features = (arguments(o).get("select") or {}).get("features")
                if features is not None:
                    n = instance(plan, f, width, None).size()
                    selected(where, "features", features, n, f"the {n} features of {name!r}")
    for name, step in walk(plan.steps):
        for f in step.train if step.kind == "fit" else []:
            if f not in out:
                raise ValueError(f"{name}: {f!r} is trained but used at no site, so its width is unknown")
            if not any(all(at.get(k) == i for k, i in step.at.items()) for at in used_at[f]):
                with naming_point(plan, step.at):
                    raise ValueError(f"{name}: {f!r} is trained but used at no site at this point, where a sweep in "
                                     "the body names another featurizer, so it would train on no gradient; write one "
                                     "fit per featurizer it trains, or sweep featurizers the fit does not train")
    for name, width in out.items():
        for f in points(plan.featurizers[name]):
            f.width, f.kept = width, laid.get(name, {})


def check_input_site(plan: Plan, where: str, at: dict, f: Featurizer, o: Op | OpStep, layer: int | None) -> None:
    """A bundle made reading its input at another site (a cross-site adapter's `with`, which its header records as
    `with.component` and `with.layer`) is used reading it there, at each layer the use is made at (`layer`; a
    `with` that stacks is read there at that layer), and one made without reads none: the width check alone would
    pass an o_proj adapter fed its own output, whose width can equal the premix's. A featurizer with no clab
    header (drawn, or a published file) records neither, and a kind whose `with` is no input (`Kind.adapts`: a
    gate's) reads none."""
    if f.header is None or not KINDS[f.kind].adapts:
        return
    given, recorded, recorded_layer = arguments(o).get("with"), f.header.get("with.component"), \
        f.header.get("with.layer")
    file = (f.model_extra or {})["file_path"]
    if recorded is None and given is not None:
        raise ValueError(f"{where}: {file}'s header records no input site (with.component): it was trained on its "
                         "own site's value; retrain it with this 'with'")
    if recorded is None:
        return
    if given is None:
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {recorded_layer} "
                         f"(with), and this use gives no 'with'; give 'with' an entry at {recorded} layer "
                         f"{recorded_layer}")
    read = entry_of(scope(plan, where), given, at)[1]
    site = cast(Site, read.site)
    made = layer if address.stacks(site.layers) else (read.layers or [None])[0]
    if site.component != recorded or made not in bundles.recorded(recorded_layer):
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {recorded_layer} "
                         f"(with), and this use's with {given!r} is {site.component} layer {made}")


def tensor_refusal(steps: dict, featurizers: dict, ref: str, at: dict[str, int], before: bool) -> str | None:
    """Why `ref`, which names a featurizer's tensor (`plan.tensor_of`; `featurizers` the plan's), is no value among
    `steps`: its kind holds no such tensor (`Kind.held`, which the refusal lists), no step among `steps` (those
    before the one naming it, where `before`) encodes or applies the featurizer at a site, whose last axis its tensors
    lie on, or it lays two of its axes out under one name (an affine's `W`, `[hidden_size, hidden_size]`; a
    subspace's basis inside another's coordinates), which no op tells apart. None where it is a value, at the point
    `at`, and for any other name."""
    named = tensor_of(steps, featurizers, ref)
    if named is None:
        return None
    featurizer, tensor = named
    name = featurizer if featurizer in featurizers else featurizer.partition(".")[2]
    f = points(featurizers[name])[0]
    held = KINDS[f.kind].held(f.bundle)
    if tensor not in held:
        return (f"{ref!r} names no tensor of {featurizer!r}, {article(f.kind)} {f.kind}, whose tensors are "
                f"{', '.join(held)}")
    if all(u[0] != name for u in featurizer_uses(steps, featurizers)):
        return (f"{ref!r} is a tensor of {featurizer!r}, laid out on the last axis of the site it is used at, and no "
                f"step {'before this one ' if before else ''}encodes or applies {featurizer!r} at a site; name its "
                "tensors after a step that does")
    dims = record_of(steps, ref, at, featurizers).dims or []
    if len(set(dims)) < len(dims):
        return (f"{ref!r} is laid out {dims}, two of its axes under one name, which no op tells apart; a fit's trained "
                f"featurizer saves whole, as the bundle that holds {tensor!r} (`<fit>.{name}`)")
    return None


def check_named_values(plan: Plan) -> None:
    """A featurizer's field that names a value (`named_values`: a constant's `init`) names one that a step makes before
    every use of the featurizer and before the fit that trains it (a tensor of one of the plan's featurizers, which no
    step makes, is there before the first), so that the run has it where it makes the instance;
    on no sweep axis, since the featurizer is one per point of its own fields; and one vector as each use meets the
    value: laid out `[<that value's last axis>]`, with no rows, `layers` or other axis, in the use's coordinates. That
    it keeps on that axis what the use keeps, and so is as wide, is preflight's (`check_named_widths`), once the
    model's sizes resolve both selections."""
    order, known = list(runnable(plan.steps)), produced(plan.steps, plan.featurizers)
    used = uses(plan)
    for name, spec in plan.featurizers.items():
        for f in points(spec):
            for field, ref in named_values(f).items():
                path, kind = f"featurizers.{name}.{field}", f.kind
                tensor = tensor_of(plan.steps, plan.featurizers, ref)
                if tensor is not None and (why := tensor_refusal(plan.steps, plan.featurizers, ref,
                                                                 points(producer(plan, ref))[0].at, False)):
                    fail(path, why)
                if ref not in known and tensor is None:
                    fail(path, f"{ref!r} names no value a step makes; known: {', '.join(known) or 'none'}")
                drawn = tensor[0] if tensor is not None and tensor[0] in plan.featurizers else None
                if drawn is not None and any(named_values(p) for p in points(plan.featurizers[drawn])):
                    fail(path, f"{ref!r} is a tensor of {drawn!r}, which is itself drawn from a value, so the run "
                         f"would make neither first; draw {name!r} from a step's value")
                made = made_by(plan, ref)
                fits = [n for n, step in plan.steps.items() if points(step)[0].kind == "fit"
                        and name in points(step)[0].train]
                firsts = [where.partition(".")[0] for where, *_ in used.get(name, [])] + fits
                first = min(firsts, key=order.index, default=None)
                if first is not None and made in order and order.index(first) <= order.index(made):
                    when = "which uses it too" if first == made else f"after step {first!r}, which uses it"
                    fail(path, f"{ref!r} is made by step {made!r}, {when}, and {name!r} is drawn from it: make the "
                         "value in a step before the featurizer's first use")
                record = record_of(plan.steps, ref, projections(plan, ref)[0][1].at, plan.featurizers)
                if record.axes:
                    fail(path, f"{ref!r} depends on the sweep axes {record.axes}, and {article(kind)} {kind} is drawn "
                         f"once per point of its own fields: {field} names a value on no sweep axis")
                for where, read, _, _, around in used.get(name, []):
                    space = [{k: v for k, v in arguments(o).items() if k in ("featurizer", "select")} for o in around]
                    last = KINDS[kind_of(plan, arguments(around[-1])["featurizer"])].axis if around else \
                        address.dims(cast(Site, read.site).component)[-1]
                    dims = record.dims or []
                    if dims != [last]:
                        why = "a rows axis, a value per row" if "rows" in dims else \
                            "a layers axis, a value per layer" if "layers" in dims else f"no {last!r} axis alone"
                        # an axis no site selects (a probe's classes, a featurizer's coordinates), which a select
                        # step narrows; one the site keeps several of, whose one vector all of them would share
                        other = [a for a in dims if a not in (last, *address.EXAMPLE_AXES)
                                 and not any(a in address.selectable(c) for c in address.COMPONENTS)]
                        shared = [a for a in dims if a not in (last, *address.EXAMPLE_AXES)
                                  and (read.kept.get(a) is None or len(read.kept[a]) > 1)]
                        fix = (f"take one of its {other[0]} in a step before the {kind}'s first use, a select of it "
                               f"summed over {other[0]} (a sum of one entry is that entry)") if other else \
                            (f"one vector is shared by every {shared[0]} the site keeps, so for each one's own, "
                             f"declare a site per {shared[0]} (a select of one) with {article(kind)} {kind} of its "
                             f"own, each {field} a mean of a read there over every axis but {last}") if shared else \
                            f"reduce it to that, a mean of a read at the site over every axis but {last}"
                        fail(path, f"{ref!r} is laid out {dims}, {why}, and {article(kind)} {kind}'s {field} is one "
                             f"vector of its width, laid out {[last]} as {where} meets the value: {fix}")
                    if record.space != space:
                        fail(path, f"{ref!r} is in another coordinate space than {where}'s value; {article(kind)} "
                             f"{kind}'s {field} is in the coordinates of the value it is applied to")


def check_named_widths(engine: Engine, plan: Plan) -> None:
    """Preflight's half of `check_named_values`: with every selection resolved among the model's sizes, a field that
    names a value keeps on its one axis the indices each use keeps there (the whole axis where the use's site selects
    none), so it is as wide as the featurizer (`Featurizer.width`) and lies unit for unit on the value it is written
    into; a use inside an `encode` is in the coordinates compile compared."""
    sizes, used = engine.sizes(), uses(plan)
    for name, spec in plan.featurizers.items():
        for f in points(spec):
            for field, ref in named_values(f).items():
                record = record_of(plan.steps, ref, projections(plan, ref)[0][1].at, plan.featurizers)
                last = (record.dims or [""])[-1]
                held = record.kept.get(last)
                for where, read, _, _, around in [] if last == "" else used.get(name, []):
                    kept = None if around else read.kept.get(last)
                    if held != kept:
                        wide = selectors.width_of(held) if held is not None else sizes.get(last)
                        raise ValueError(f"{where}: featurizer {name!r} is drawn from {ref!r} ({field}), which keeps "
                                         f"{last} {held if held is not None else 'whole'}, {wide} wide, and is applied "
                                         f"here to a value keeping {last} {kept if kept is not None else 'whole'}, "
                                         f"{f.width} wide; {article(f.kind)} {f.kind}'s {field} lies unit for unit on "
                                         f"the value it is written into: make {ref!r} from a read at this use's "
                                         "select")


def named_values(spec: Featurizer) -> dict[str, str]:
    """A featurizer point's fields that name a value, each to the name: those its kind's signature annotates `Value`
    and the document gives as a name (a constant's `init`); a number there is a literal."""
    params = registry.signature(KINDS[spec.kind]).parameters
    return {k: v for k, v in (spec.model_extra or {}).items()
            if isinstance(v, str) and k in params and registry.thing(params[k].annotation) == Value}


def instance(plan: Plan, spec: Featurizer, width: int, seed: int | None, device: str = "cpu",
             state: Mapping[str, Any] | None = None) -> Kind:
    """A featurizer point `spec` made on `device`: from its bundle, or drawn from its seed, else `seed`; a probe's
    class list from its `classes` column; and a field that names a value (`named_values`: a constant's `init`) as that
    value, which the run holds in `state` once the step that made it has run."""
    fields = spec.model_extra or {}
    dataset = fields["classes"].partition(".")[0] if "classes" in fields else None
    classes = labels({dataset: held_rows(plan, dataset)}, fields["classes"]) if dataset else None
    named = named_values(spec)
    if named:
        assert state is not None, "a field that names a value is drawn in the run, which holds it"
        fields = {**fields, **{k: lookup(state, ref, {}) for k, ref in named.items()}}
    return make(spec.kind, fields, width, spec.bundle, seed, classes, spec.header, device, spec.kept)
