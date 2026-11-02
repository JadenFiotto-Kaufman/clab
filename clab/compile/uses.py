"""A featurizer's uses in a plan: every op that reaches it, the width it has at every site it is used at (a gate
over axes, the indices it keeps on each of them), the checks of a loaded bundle's header against those uses, the
instance made for the run, and the header a saved bundle carries."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

from clab.featurizers import KINDS, Kind, bundles, labels, make
from clab.format import address, selectors
from clab.format.selectors import selected
from clab.format.sweeps import points
from clab.plan import (
    Featurizer,
    Intervention,
    Op,
    OpStep,
    Plan,
    Site,
    agreeing,
    arguments,
    bracketed,
    declared,
    entries,
    entry_of,
    fail,
    held_rows,
    record_of,
    scope,
    visited,
    walk,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def uses(plan: Plan) -> dict[str, list[tuple[str, Intervention, dict, Op | OpStep, list[Op]]]]:
    """Every `encode` and `apply` of a declared featurizer, by featurizer: the entry's dotted name, the read it is
    of (the entry's own record; an op step's the read its `of` names, whose site and coordinates its featurizer is
    used at), its step's point, the op, and the encodes still open around it (outermost first), whose coordinates
    it is applied to."""
    found: dict[str, list] = {}
    for name, step in walk(plan.steps):
        ref = arguments(step).get("featurizer") if step.kind == "op" else None
        if ref is not None:                 # a body's step sees the body's steps beside the root's
            fit = name.rpartition(".")[0]
            steps = {**plan.steps, **agreeing(plan.steps[fit], step.at).steps} if fit else plan.steps
            read = record_of(steps, cast(str, step.read), step.at)
            assert isinstance(read, Intervention), f"{name}: an encode or apply step is over a read"
            found.setdefault(declared(plan, ref), []).append((name, read, step.at, step,
                                                              [Op(op="encode", **s) for s in read.space]))
        for key, entry in entries(step).items():
            for node, around in visited(bracketed(entry.ops)):
                ref = arguments(node["op"]).get("featurizer")
                if ref is not None:
                    found.setdefault(declared(plan, ref), []).append((f"{name}.{key}", entry, step.at,
                                                                      node["op"], [a["op"] for a in around]))
    return found


def bundled_sites(plan: Plan) -> None:
    """A bundle's header names the component it was made at and the layers, counted from the first
    (`bundles.recorded`); every site its featurizer is used at is that component, and each layer it picks is one
    of those. A layer counted from the last, and a site that picks several, are the model's to resolve, and
    preflight checks each layer."""
    for name, records in uses(plan).items():
        for where, read, at, _, _ in records:
            site, f = cast(Site, read.site), agreeing(plan.featurizers[name], at)
            if f.header is None:
                continue
            known = site.layers is None or isinstance(site.layers, int) and site.layers >= 0
            for key, ok in (("component", f.header.get("component") == site.component),
                            ("layer", not known or site.layers in bundles.recorded(f.header.get("layer")))):
                if not ok:
                    fail(f"featurizers.{name}.file_path", f"{(f.model_extra or {})['file_path']}: header {key!r} is "
                         f"{f.header.get(key)!r}; this document uses {name!r} at {site.component} layer "
                         f"{site.layers} ({where})")


def over_of(f: Featurizer) -> list[str] | None:
    """The axes a gate's units lie on: its `over`, or, where a loading document leaves it out, its bundle's header's;
    None for the last axis of its site's selection, and for any other featurizer."""
    given = (f.model_extra or {}).get("over")
    return given if given is not None or f.header is None or "over" not in f.header else json.loads(f.header["over"])


def selection(read: Intervention, o: Op | OpStep) -> list[str]:
    """The layout of the value a featurizer's op meets at a use made before any encode: inside an entry its site's
    selection, a `layers` axis first where the site picks several; as a step, its `of`'s."""
    site = cast(Site, read.site)
    return list(read.dims or []) if isinstance(o, OpStep) else address.dims(site.component, site.layers)


def over_axes(plan: Plan) -> None:
    """A gate over axes is used by `apply` alone, on its site's own layout (inside no `encode`, whose coordinates
    are no axes of the site's), whose selection has each axis it names, in the selection's order, `layers` only
    where the site picks several. A refusal names the featurizer's `over` and the use."""
    for name, records in uses(plan).items():
        for where, read, at, o, around in records:
            over = over_of(agreeing(plan.featurizers[name], at))
            if over is None:
                continue
            path, site, laid = f"featurizers.{name}.over", cast(Site, read.site), selection(read, o)
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
    """The indices a gate over axes keeps on each of them at one use: the layers the use is made at, and on each
    other axis those its selection keeps (every one of an axis it keeps whole). A loaded gate's are its header's
    (`layer`, and `kept.<axis>` for each other axis), which each use's other axes are checked against; each layer
    the use is made at is checked to be one `layer` names, as every loaded bundle's is."""
    found = {}
    for axis in over:
        if axis == "layers":
            found[axis] = list(read.layers or [])
        elif axis in read.kept:
            found[axis] = list(read.kept[axis])
        elif sizes.get(axis) is None:
            raise ValueError(f"{where}: {cast(Site, read.site).component}'s {axis} axis grows with the input; "
                             f"gate {name!r} lays its units on axes of a fixed size")
        else:
            found[axis] = list(range(sizes[axis]))
    if f.header is None:
        return found
    file = (f.model_extra or {})["file_path"]
    recorded: dict[str, list] = {"layers": bundles.recorded(f.header.get("layer"))} if "layers" in over else {}
    for axis in over:
        if axis == "layers":
            continue
        written = f.header.get(f"kept.{axis}")
        recorded[axis] = json.loads(written) if written is not None else []
        if recorded[axis] != found[axis]:
            raise ValueError(f"{where}: {file}: header 'kept.{axis}' is {written!r}; this document uses {name!r} "
                             f"at {cast(Site, read.site).component}, whose selection keeps {axis} {found[axis]}")
    return recorded


def widths(engine: Engine, plan: Plan) -> None:
    """Each featurizer's width, written on its record (`Featurizer.width`, at every point): the size of the last
    axis of every site it is used at, which is one size, and the width its bundle's header says. A `select` on
    its features stays within them, and a featurizer a fit trains is used somewhere. A gate over axes is as wide as
    it has units, and keeps on each axis the indices every site it is used at selects, which are one selection,
    or that its bundle's header records (`units`), written on its record too (`Featurizer.kept`)."""
    sizes, out, laid = engine.sizes(), {}, {}
    for name, used in uses(plan).items():
        for where, read, at, o, around in used:
            site, kept = cast(Site, read.site), read.kept
            f = agreeing(plan.featurizers[name], at)
            over = over_of(f)
            if over is not None:
                found = units(where, name, f, read, over, sizes)
                if laid.setdefault(name, found) != found:
                    other = ", ".join(f"{a} {v}" for a, v in laid[name].items())
                    raise ValueError(f"{where}: gate {name!r} is over {', '.join(f'{a} {v}' for a, v in found.items())} "
                                     f"here and {other} elsewhere; its units lie on one selection of the axes it is "
                                     "over, so every site it is used at selects the same ones")
            last = address.dims(site.component)[-1]
            width = math.prod(map(len, laid[name].values())) if over is not None else \
                len(kept[last]) if last in kept else sizes.get(last)
            if width is None:
                raise ValueError(f"{where}: {site.component}'s last axis is {last}, which grows with the input; "
                                 f"featurizer {name!r} needs a feature axis")
            for outer in around:                # the coordinates of each encode it is inside, outermost first
                ref = declared(plan, arguments(outer)["featurizer"])
                width = instance(plan, agreeing(plan.featurizers[ref], at), width, None).size()
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
            for layer in read.layers or [None]:                       # each layer the use is made at
                which = f"picks layer {layer}" if address.stacks(site.layers) else f"is layer {layer}"
                if stated is not None and (stated["component"], at_layer(stated["layers"], sizes["num_layers"])) != \
                        (site.component, str(layer)):
                    raise ValueError(f"{where}: featurizer {name!r} was trained at {stated['component']} layer "
                                     f"{stated['layers']} (its site); this use is {site.component} layer "
                                     f"{site.layers!r}" + (f", which {which}" if address.stacks(site.layers) else ""))
                if f.header is not None and site.layers is not None and \
                        layer not in bundles.recorded(f.header.get("layer")):
                    raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'layer' is "
                                     f"{f.header.get('layer')!r}; this document uses {name!r} at {site.component} "
                                     f"layer {site.layers!r}, which {which}")
                read_with(plan, where, at, f, o, layer)
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
    for name, width in out.items():
        for f in points(plan.featurizers[name]):
            f.width, f.kept = width, laid.get(name, {})


def at_layer(layers: object, n: int) -> str:
    """A site's one layer counted from the first, as a header records it; other layers as written."""
    return str(selectors.select(layers, n)[0]) if isinstance(layers, int) else str(layers)


def read_with(plan: Plan, where: str, at: dict, f: Featurizer, o: Op | OpStep, layer: int | None) -> None:
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
                         "own site's value, or before bundles recorded a 'with'; retrain it with this 'with'")
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


def instance(plan: Plan, spec: Featurizer, width: int, seed: int | None, device: str = "cpu") -> Kind:
    """A featurizer point `spec` made on `device`: from its bundle, or drawn from its seed, else `seed`; a probe's
    class list from its `classes` column."""
    fields = spec.model_extra or {}
    dataset = fields["classes"].partition(".")[0] if "classes" in fields else None
    classes = labels({dataset: held_rows(plan, dataset)}, fields["classes"]) if dataset else None
    return make(spec.kind, fields, width, spec.bundle, seed, classes, spec.header, device, spec.kept)
