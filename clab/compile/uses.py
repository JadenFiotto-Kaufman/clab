"""A featurizer's uses in a plan: every op that reaches it, the width it has at every site it is used at, the
checks of a loaded bundle's header against those uses, the instance made for the run, and the header a saved
bundle carries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from clab.featurizers import KINDS, Kind, bundles, labels, make
from clab.format import address, selectors
from clab.format.selectors import Selector, made_at, selected
from clab.plan import (
    Featurizer,
    Op,
    OpStep,
    Plan,
    Site,
    agreeing,
    arguments,
    bracketed,
    declared,
    dims_in,
    entries,
    entry_of,
    fail,
    held_rows,
    scope,
    visited,
    walk,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def uses(plan: Plan) -> dict[str, list[tuple[str, Site, dict, Op | OpStep, list[Op]]]]:
    """Every `encode` and `apply` of a declared featurizer, by featurizer: the entry's dotted name, its site, its
    step's point, the op, and the encodes still open around it (outermost first), whose coordinates it is
    applied to. An op step's is at the site of the read its `of` comes from (`source`)."""
    found: dict[str, list] = {}
    for name, step in walk(plan.steps):
        ref = arguments(step).get("featurizer") if step.kind == "op" else None
        if ref is not None:
            site, around = source(plan, name, step)
            found.setdefault(declared(plan, ref), []).append((name, site, step.at, step, around))
        for key, entry in entries(step).items():
            for node, around in visited(bracketed(entry.ops)):
                ref = arguments(node["op"]).get("featurizer")
                if ref is not None:
                    found.setdefault(declared(plan, ref), []).append((f"{name}.{key}", entry.site, step.at,
                                                                      node["op"], [a["op"] for a in around]))
    return found


def source(plan: Plan, name: str, step: OpStep) -> tuple[Site, list[Op]]:
    """Where op step `name` (dotted, as `walk` names it) meets a site: that of the read its `of` comes from,
    followed through op steps (and into a fit's eval members), and the encodes its value is in, outermost first:
    those the read leaves open, then each encode step on the way. A value whose layout a step other than an
    encode changed on the way is no site's, and a featurizer acts on a site's value or its coordinates."""
    fit = name.rpartition(".")[0]
    steps = {**plan.steps, **agreeing(plan.steps[fit], step.at).steps} if fit else plan.steps
    value, around = step.of, []
    while True:
        thing, _, member = value.partition(".")
        producer = agreeing(steps[thing], step.at)
        if producer.kind == "fit":
            steps, value = {**steps, **producer.steps}, member.partition(".")[2]
            continue
        changed = producer.kind == "op" and producer.op != "encode" and \
            (producer.dims or [])[-1:] != dims_in(steps, producer.of)[-1:]
        if changed or producer.kind != "op" and not member:
            raise ValueError(f"{name}: {arguments(step)['featurizer']!r} is used on {step.of!r}, which comes from "
                             f"{value!r}, no site's value or its coordinates; a featurizer acts on one")
        if producer.kind != "op":
            entry = entries(producer)[member]
            return entry.site, [Op(op="encode", **s) for s in entry.space] + around
        if producer.op == "encode":
            around.insert(0, Op(op="encode", **arguments(producer)))
        value = producer.of


def bundled_sites(plan: Plan) -> None:
    """A bundle's header names the component it was made at and the layers, counted from the first
    (`bundles.recorded`); every site its featurizer is used at is that component, and each layer it picks is one
    of those. A layer counted from the last, and a site that picks several, are the model's to resolve, and
    preflight checks each layer."""
    for name, records in uses(plan).items():
        for where, site, at, _, _ in records:
            f = agreeing(plan.featurizers[name], at)
            if f.header is None:
                continue
            known = site.layers is None or isinstance(site.layers, int) and site.layers >= 0
            for key, ok in (("component", f.header.get("component") == site.component),
                            ("layer", not known or site.layers in bundles.recorded(f.header.get("layer")))):
                if not ok:
                    fail(f"featurizers.{name}.file_path", f"{(f.model_extra or {})['file_path']}: header {key!r} is "
                         f"{f.header.get(key)!r}; this document uses {name!r} at {site.component} layer "
                         f"{site.layers} ({where})")


def widths(engine: Engine, plan: Plan) -> dict[str, int]:
    """Each featurizer's width: the size of the last axis of every site it is used at, which is one size, and
    the width its bundle's header says. A `select` on its features stays within them, and a featurizer a fit
    trains is used somewhere."""
    sizes, out = engine.sizes(), {}
    for name, used in uses(plan).items():
        for where, site, at, o, around in used:
            last = address.dims(site.component)[-1]
            kept = selectors.kept(cast("dict[str, Selector] | None", site.select), sizes)
            width = len(kept[last]) if last in kept else sizes.get(last)
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
            f = agreeing(plan.featurizers[name], at)
            for field in KINDS[f.kind].counts:
                count = (f.model_extra or {}).get(field)
                if isinstance(count, int) and count > width:
                    raise ValueError(f"{where}: featurizer {name!r} has {field} {count}, and it is used at width "
                                     f"{width} ({site.component}), which has no more than {width} directions; "
                                     f"{field} is at most {width}")
            stated = (f.model_extra or {}).get("site")
            for layer in made_at(site, sizes["num_layers"]):         # each layer the use is made at
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
                read_with(plan, where, at, f, o, layer, sizes["num_layers"])
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
    return out


def at_layer(layers: object, n: int) -> str:
    """A site's one layer counted from the first, as a header records it; other layers as written."""
    return str(selectors.select(layers, n)[0]) if isinstance(layers, int) else str(layers)


def read_with(plan: Plan, where: str, at: dict, f: Featurizer, o: Op | OpStep, layer: int | None, n: int) -> None:
    """A bundle made reading its input at another site (a cross-site adapter's `with`, which its header records as
    `with.component` and `with.layer`) is used reading it there, at each layer the use is made at (`layer`; a
    `with` that stacks is read there at that layer), and one made without reads none: the width check alone would
    pass an o_proj adapter fed its own output, whose width can equal the premix's. A featurizer with no clab
    header (drawn, or a published file) records neither."""
    if f.header is None:
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
    site = entry_of(scope(plan, where), given, at)[1].site
    made = layer if address.stacks(site.layers) else made_at(site, n)[0]
    if site.component != recorded or made not in bundles.recorded(recorded_layer):
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {recorded_layer} "
                         f"(with), and this use's with {given!r} is {site.component} layer {made}")


def instance(plan: Plan, spec: Featurizer, width: int, seed: int | None, device: str = "cpu") -> Kind:
    """A featurizer point `spec` made on `device`: from its bundle, or drawn from its seed, else `seed`; a probe's
    class list from its `classes` column."""
    fields = spec.model_extra or {}
    dataset = fields["classes"].partition(".")[0] if "classes" in fields else None
    classes = labels({dataset: held_rows(plan, dataset)}, fields["classes"]) if dataset else None
    return make(spec.kind, fields, width, spec.bundle, seed, classes, spec.header, device)
