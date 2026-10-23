"""A featurizer's uses in a plan: every op that reaches it, the width it has at every site it is used at, the
checks of a loaded bundle's header against those uses, the instance made for the run, and the header a saved
bundle carries."""

from typing import Any

from clab.featurizers import KINDS, Kind, labels, make
from clab.format import address, selectors
from clab.format.plan import (
    Op,
    Plan,
    agreeing,
    arguments,
    bracketed,
    declared,
    entries,
    entry_of,
    fail,
    scope,
    visited,
    walk,
)
from clab.format.selectors import selected


def uses(plan: Plan) -> dict[str, list[tuple[str, Any, dict, Op, list[Op]]]]:
    """Every `encode` and `apply` of a declared featurizer, by featurizer: the entry's dotted name, its site, its
    step's point, the op, and the encodes still open around it (outermost first), whose coordinates it is
    applied to."""
    found: dict[str, list] = {}
    for name, step in walk(plan.steps):
        for key, entry in entries(step).items():
            for node, around in visited(bracketed(entry.ops)):
                ref = arguments(node["op"]).get("featurizer")
                if ref is not None:
                    found.setdefault(declared(plan, ref), []).append((f"{name}.{key}", entry.site, step.at,
                                                                      node["op"], [a["op"] for a in around]))
    return found


def bundled_sites(plan: Plan) -> None:
    """A bundle's header names the site it was made at, its layer counted from the first; every site its
    featurizer is used at is that one. A layer counted from the last is the model's to resolve, and preflight
    checks it."""
    for name, records in uses(plan).items():
        for where, site, at, _, _ in records:
            f = agreeing(plan.featurizers[name], at)
            layer = [] if isinstance(site.layers, int) and site.layers < 0 else [("layer", str(site.layers))]
            for key, value in [("component", site.component), *layer]:
                if f.header is not None and f.header.get(key) != value:
                    fail(f"featurizers.{name}.file_path", f"{(f.model_extra or {})['file_path']}: header {key!r} is "
                         f"{f.header.get(key)!r}; this document uses {name!r} at {site.component} layer "
                         f"{site.layers} ({where})")


def widths(engine, plan: Plan) -> dict[str, int]:
    """Each featurizer's width: the size of the last axis of every site it is used at, which is one size, and
    the width its bundle's header says. A `select` on its features stays within them, and a featurizer a fit
    trains is used somewhere."""
    sizes, out = engine.sizes(), {}
    for name, used in uses(plan).items():
        for where, site, at, o, around in used:
            last = address.dims(site.component)[-1]
            kept = selectors.kept(site.select, sizes)
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
            layer = selectors.select(site.layers, sizes["num_layers"]) if isinstance(site.layers, int) else None
            stated = (f.model_extra or {}).get("site")
            if stated is not None and (stated["component"], at_layer(stated["layers"], sizes["num_layers"])) != \
                    (site.component, at_layer(site.layers, sizes["num_layers"])):
                raise ValueError(f"{where}: featurizer {name!r} was trained at {stated['component']} layer "
                                 f"{stated['layers']} (its site); this use is {site.component} layer {site.layers}")
            if f.header is not None and layer and f.header.get("layer") != str(layer[0]):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'layer' is "
                                 f"{f.header.get('layer')!r}; this document uses {name!r} at {site.component} "
                                 f"layer {site.layers}, which is layer {layer[0]}")
            if f.header is not None:
                read_with(plan, where, at, f, o, sizes["num_layers"])
            if f.header is not None and f.header.get("width") != str(width):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'width' is "
                                 f"{f.header.get('width')!r}; this document uses {name!r} at {site.component}, "
                                 f"whose width is {width}")
            if f.header is None and f.bundle is not None and KINDS[f.kind].width(f.bundle) != width:
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']} has no clab header and is "
                                 f"{KINDS[f.kind].width(f.bundle)} wide; this document uses {name!r} at "
                                 f"{site.component}, whose width is {width}")
            features: Any = (arguments(o).get("select") or {}).get("features")
            if features is not None:
                n = instance(plan, f, width, None).size()
                selected(where, "features", features, n, f"the {n} features of {name!r}")
    for name, step in walk(plan.steps):
        for f in step.train if step.kind == "fit" else []:
            if f not in out:
                raise ValueError(f"{name}: {f!r} is trained but used at no site, so its width is unknown")
    return out


def at_layer(layers, n: int) -> str:
    """A site's one layer counted from the first, as a header records it; other layers as written."""
    return str(selectors.select(layers, n)[0]) if isinstance(layers, int) else str(layers)


def read_with(plan: Plan, where: str, at: dict, f, o, n: int) -> None:
    """A bundle made reading its input at another site (a cross-site adapter's `with`, which its header records as
    `with.component` and `with.layer`) is used reading it there, and one made without reads none: the width check
    alone would pass an o_proj adapter fed its own output, whose width can equal the premix's."""
    given, recorded, layer = arguments(o).get("with"), f.header.get("with.component"), f.header.get("with.layer")
    file = (f.model_extra or {})["file_path"]
    if recorded is None and given is not None:
        raise ValueError(f"{where}: {file}'s header records no input site (with.component): it was trained on its "
                         "own site's value, or before bundles recorded a 'with'; retrain it with this 'with'")
    if recorded is None:
        return
    if given is None:
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {layer} (with), and "
                         f"this use gives no 'with'; give 'with' an entry at {recorded} layer {layer}")
    site = entry_of(scope(plan, where), given, at)[1].site
    if (site.component, at_layer(site.layers, n)) != (recorded, layer):
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {layer} (with), and "
                         f"this use's with {given!r} is {site.component} layer {at_layer(site.layers, n)}")


def instance(plan: Plan, spec, width: int, seed: int | None, device: str = "cpu") -> Kind:
    """A featurizer point `spec` made on `device`: from its bundle, or drawn from its seed, else `seed`; a probe's
    class list from its `classes` column."""
    fields = spec.model_extra or {}
    classes = labels({n: d.rows or [] for n, d in plan.data.items()}, fields["classes"]) \
        if "classes" in fields else None
    return make(spec.kind, fields, width, spec.bundle, seed, classes, spec.header, device)
