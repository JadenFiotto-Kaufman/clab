"""A featurizer's uses in a plan: every op that reaches it, the width it has at every site it is used at, the
checks of a loaded bundle's header against those uses, the instance made for the run, and the header a saved
bundle carries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import address, ops, selectors
from ..answers import column_ref
from ..interventions import bracketed, selected, visited
from ..plan import Op, Plan, agreeing, arguments, declared, entries, entry_of, fail, kind_of, scope, walk
from ..plan import name as named
from ..sweeps import coordinate, labelled
from ..table import materialized, provenance
from . import KINDS, Kind, bundles, labels, make

if TYPE_CHECKING:
    from ..compile import Compiling


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


def featurizer(cx: Compiling, plan: Plan, name: str) -> dict:
    """A declared featurizer at one point, with the bundle its `file_path` names, checked key by key against
    the document."""
    f, path = plan.featurizers[name], f"featurizers.{name}"
    assert not isinstance(f, list)
    fields = f.model_extra or {}
    if fields.get("classes") is not None:
        column_ref(cx, fields["classes"], f"{path}.classes", None)
        try:
            labels(cx.rows, fields["classes"])
        except ValueError as e:
            fail(f"{path}.classes", str(e))
    missing = [k for k in KINDS[f.kind].drawn if fields.get(k) is None]
    if fields.get("file_path") is None and missing:
        fail(path, f"a {f.kind!r} featurizer drawn without a bundle needs {' and '.join(map(repr, missing))}; or "
             "load one with 'file_path'")
    record = f.model_dump(exclude_defaults=True)
    if isinstance(fields.get("site"), str):         # the site it was trained at, inlined as an entry's is
        if fields["site"] not in plan.sites:
            fail(f"{path}.site", f"no site {fields['site']!r}; declared: {', '.join(plan.sites) or 'none'}")
        named = plan.sites[fields["site"]]
        if not isinstance(named.layers, int) or named.select:
            fail(f"{path}.site", f"site {fields['site']!r} is {named.model_dump(exclude_defaults=True)}; the site a "
                 "featurizer was trained at is a component at one layer, {'component', 'layers': <int>}")
        record["site"] = fields["site"] = {"component": named.component, "layers": named.layers}
    if fields.get("file_path") is None:
        return record
    try:
        tensors, header = bundles.load(cx.root, f.kind, fields, plan.model,
                                                   labels(cx.rows, fields["classes"]) if "classes" in fields
                                                   else None)
    except ValueError as e:
        fail(f"{path}.file_path", str(e))
    return {**record, "bundle": tensors, "header": header}


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


def headed(plan: Plan, name: str, entry: Any) -> dict[str, str] | None:
    """The header of a saved bundle: its one point's, or a swept one's every point's, each key after the point's
    coordinate and a `/` as its tensors are (`k=4/k`), so the file describes each bundle it holds."""
    if not isinstance(entry, list):
        return header(plan, name, plan.results[name], entry)
    found: dict[str, str] = {}
    for p in entry:
        point = header(plan, name, plan.results[name][coordinate(plan, p.at)], p)
        if point is None:
            return None
        found["format"] = point.pop("format")
        found.update({f"{labelled(plan, p.at)}/{k}": v for k, v in point.items()})
    return found


def header(plan: Plan, name: str, value: Any, step: Any) -> dict[str, str] | None:
    """A bundle's header, when the value is one (`value`, at `step`, the point that made it): an op step's whose op
    declares the featurizer kind its bundle is for, made at the site of the read it was taken over, or a fit's
    trained featurizer, made at the first site it is used at."""
    member = name.partition(".")[2]
    if step.kind == "op" and not member and ops.DECLARED[step.op]["bundle"]:
        kind, fields = ops.DECLARED[step.op]["bundle"], arguments(step)
        site = entry_of(plan.steps, provenance(plan, name)[0] or "", step.at)[1].site
        reads = None
    elif step.kind == "fit" and member in step.train:
        kind, fields = kind_of(plan, member), agreeing(plan.featurizers[member], step.at).model_extra or {}
        used = uses(plan)[member]
        where, site, _, o, _ = next((u for u in used if all(step.at.get(k) == i for k, i in u[2].items())), used[0])
        given = arguments(o).get("with")
        reads = entry_of(scope(plan, where), given, step.at)[1].site if given else None
    else:
        return None
    classes = labels({n: materialized(plan, n) for n in plan.data}, fields["classes"]) \
        if "classes" in fields else None
    site, reads = (counted(plan, s) for s in (site, reads))
    return bundles.header(kind, fields, plan.model, classes, site, KINDS[kind].width(value),
                                      named(plan), reads)


def counted(plan: Plan, site: Any) -> Any:
    """A site whose one layer is counted from the first, as a header records it."""
    if site is not None and isinstance(site.layers, int) and plan.run is not None and plan.run.num_layers is not None:
        return site.model_copy(update={"layers": selectors.select(site.layers, plan.run.num_layers)[0]})
    return site
