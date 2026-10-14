"""compile: a plan resolved for running, every name checked, every sweep expanded into points.

Compile knows the document and the files; the run knows the model. Anything decided from the document, the
data files and bundle headers is checked here; anything that needs the tokenizer or a size of the model is
checked by `steps.preflight`, inside the run. The records' own shape (fields, types, the pos grammar, a
site's narrowing) is the models' to check, when the plan is built. How names resolve is `step`'s docstring.
"""

import inspect
import itertools
from dataclasses import dataclass, replace
from typing import Any

from . import address, data, featurizers, ops, positions
from .plan import Plan, Site, arguments, corpus_root, fail, find_sweeps, space


def compile(plan: Plan) -> Plan:
    """The plan compiled: a new `Plan`, `compiled=True`."""
    if plan.compiled:
        return plan
    root = corpus_root(plan)
    doc = plan.model_dump(exclude_defaults=True, exclude={"run", "document"})
    rows = {}
    for name, d in plan.data.items():
        if d.hf is not None:
            fail(f"data.{name}.hf", "hub datasets land later")
        try:
            rows[name] = d.rows if d.path is None else data.rows(root, d.path, d.split)
        except ValueError as e:
            fail(f"data.{name}.path", str(e))
        if not rows[name]:
            fail(f"data.{name}", "the dataset has no rows" + (f" in split {d.split!r}" if d.split else ""))
    sweeps = find_sweeps(doc)
    values = {label: get(doc, path)["sweep"] for path, label in sweeps.items()}
    cx = Compiling(doc=doc, root=root, rows=rows, sweeps=sweeps, values=values, names={}, axes_of={}, refs=[])
    compiled = {"featurizers": {}, "steps": {}}
    for name in plan.featurizers:
        compiled["featurizers"][name] = expand(cx, name, ("featurizers", name),
                                               lambda point, name=name: featurizer(cx, point, name))
        cx.names[name] = {"dims": [], "data": None, "kind": "featurizer"}
        for scalar in featurizers.SCALARS.get(points_kind(plan, name), ()):
            cx.names[f"{name}.{scalar}"] = {"dims": [], "data": None, "kind": "scalar"}
    for name in plan.steps:
        compiled["steps"][name] = expand(cx, name, ("steps", name), lambda point, name=name: step(
            cx, f"steps.{name}", name, point.steps[name], point))
    saveable = sorted(n for n, v in cx.names.items() if n.partition(".")[0] in plan.steps and v["kind"] != "scalar")
    for i, name in enumerate(plan.saves):
        if name in plan.data:
            fail(f"saves[{i}]", f"{name!r} is a dataset; hub datasets and dataset saves land later")
        if name in plan.saves[:i]:
            fail(f"saves[{i}]", f"{name!r} is saved twice")
        if name not in saveable:
            fail(f"saves[{i}]", f"{name!r} names no step's value; a save is a step or a step's member: "
                 f"{', '.join(saveable)}")
    return Plan(**{**doc, **compiled, "sites": {}, "interventions": {}, "compiled": True, "document": plan.document,
                   "sweeps": {label: {"path": ".".join(map(str, path)), "values": values[label]}
                              for path, label in sweeps.items()},
                   "data": {n: {**doc["data"][n], "rows": rows[n], "sha256": data.digest(rows[n]),
                                "n_rows": len(rows[n])} for n in rows}})


@dataclass
class Compiling:
    """What compiling an entry needs beside the entry, passed to every function that resolves a name.

    Read only: the document as a dict (`doc`), the corpus root, each dataset's rows, each sweep's label by
    document path (`sweeps`) and its values by label (`values`). Filled as entries compile, in document
    order: each name produced so far, `{dims, data, kind}` (`names`), each entry's sweep axes (`axes_of`),
    and the names the entry being compiled references (`refs`), which decide the axes it inherits. A fit's
    body compiles in a copy whose `names` is its own scope; it shares `refs`, since what the body references
    is what the fit references, and `axes_of`."""
    doc: dict
    root: str
    rows: dict
    sweeps: dict
    values: dict
    names: dict
    axes_of: dict
    refs: list


def get(doc: dict, path: tuple):
    for key in path:
        doc = doc[key]
    return doc


# -- sweeps: each entry compiled at every point of its axes -------------------------------------------


def expand(cx: Compiling, name: str, where: tuple, build) -> dict | list:
    """An entry compiled at every point of its axes: those in what it uses (its own spec, and the sites,
    interventions and featurizers it names), and those of every value it references."""
    own = {label for path, label in cx.sweeps.items() if any(path[:len(p)] == p for p in footprint(cx, where))}
    cx.refs.clear()
    first = build(point(cx, {label: 0 for label in own}))
    used = own | {a for ref in cx.refs for a in cx.axes_of.get(ref.partition(".")[0], [])}
    labels = [label for label in cx.values if label in used]
    cx.axes_of[name] = labels
    entries = []
    for index in itertools.product(*(range(len(cx.values[label])) for label in labels)):
        at = dict(zip(labels, index))
        # The first point is the document every axis puts at its first value, which `first` was built from.
        entry = first if not any(index) else build(point(cx, at))
        for e in [entry, *entry.get("steps", {}).values()]:
            e["at"] = at
        entries.append(entry)
    return entries if labels else entries[0]


def point(cx: Compiling, at: dict) -> Plan:
    """The document with every sweep `at` names replaced by its value there."""
    def substitute(value: Any, path: tuple) -> Any:
        if cx.sweeps.get(path) in at:
            return value["sweep"][at[cx.sweeps[path]]]
        if isinstance(value, dict):
            return {k: substitute(v, path + (k,)) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v, path + (i,)) for i, v in enumerate(value)]
        return value
    return Plan(**substitute(cx.doc, ()))


def footprint(cx: Compiling, where: tuple) -> list[tuple]:
    """The document paths an entry reads: its own, and the sites, interventions and featurizers it names."""
    found, doc = [where], cx.doc

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                named = v if isinstance(v, list) else [v]
                if k == "site" and isinstance(v, str):
                    found.append(("sites", v))
                if k == "interventions":
                    for n in named:
                        found.append(("interventions", n))
                        walk(doc.get("interventions", {}).get(n))
                if k in ("featurizer", "train"):
                    found.extend(("featurizers", n) for n in named if n in doc.get("featurizers", {}))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(get(doc, where))
    return found


# -- steps -------------------------------------------------------------------------------------------


def step(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    """Names are one flat namespace per scope: no dots, and no step also a dataset or a featurizer.

    A reference is "thing, then member", split at the first dot: `counterfactual.v_cf` is step
    `counterfactual`'s read `v_cf`; `mean` is an op step's value; `pairs.cf_answer` is a dataset's column; `rot`
    is a declared featurizer and `mask.mask` its scalar; `fit.rot` is a fit's member. Values are produced in
    document order and referenced only after they exist. A fit's body is a scope of its own: it sees every
    value produced before the fit, and nothing leaves it except through `<fit>.<member>`. A named intervention
    is a template, so the names inside it resolve where a step pulls it in. Under a sweep, a name means the
    producer's point that agrees with the consumer's coordinates on the producer's axes. Every value with a
    rows axis knows the dataset its rows are aligned to (a forward's input; an op's `of`, operands and
    columns), and an op takes one dataset's rows.
    """
    return {"forward": forward, "generate": forward, "op": op_step, "fit": fit}[spec.kind](cx, path, name, spec, plan)


def forward(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    dataset, column, index = data.parse_input(spec.input)
    if dataset not in cx.rows:
        fail(f"{path}.input", f"no dataset {dataset!r}; datasets: {', '.join(cx.rows)}")
    column_ref(cx, f"{dataset}.{column}", f"{path}.input", dataset)
    rows = cx.rows[dataset]
    if index is not None and any(not isinstance(r[column], list) or index >= len(r[column]) for r in rows):
        fail(f"{path}.input", f"column {column!r} has no entry {index} on every row")
    given: dict[str, list] = {"writes": [], "reads": []}
    for n in [spec.interventions] if isinstance(spec.interventions, str) else spec.interventions or []:
        if n not in plan.interventions:
            fail(f"{path}.interventions", f"no intervention {n!r}; declared: {', '.join(plan.interventions) or 'none'}")
        for side in given:
            declared = getattr(plan.interventions[n], side)
            given[side] += [(f"interventions.{n}.{side}.{k}", k, a) for k, a in declared.items()]
    for side in given:
        given[side] += [(f"{path}.{side}.{k}", k, a) for k, a in getattr(spec, side).items()]
        seen = [k for _, k, _ in given[side]]
        for k in seen:
            if seen.count(k) > 1:
                fail(f"{path}.{side}", f"two {side[:-1]}s named {k!r}")
    records = {side: {k: access(cx, p, a, plan, spec, dataset, side == "writes") for p, k, a in given[side]}
               for side in given}
    for k, r in records["reads"].items():
        cx.names[f"{name}.{k}"] = {"dims": r["dims"], "data": dataset, "kind": "value"}
    if spec.kind == "generate":
        cx.names[name] = {"dims": ["rows", "new"], "data": dataset, "kind": "value"}
    return {**spec.model_dump(exclude_defaults=True, exclude={"interventions"}), **records, "dataset": dataset}


def access(cx: Compiling, path: str, spec, plan: Plan, step, dataset: str, write: bool) -> dict:
    """One write or read: its site inlined, its positions checked against the step, its ops' layouts."""
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    if write and site.layers == "all":
        fail(f"{path}.site", "a write names its layer; layers 'all' is a read's (or sweep the layers)")
    if write and len(site.select or {}) > 1:
        fail(f"{path}.site.select", f"a write selects on one axis; this one selects {', '.join(site.select or {})}")
    pos = positions.parse(spec.pos)
    frame, scope = pos["frame"], pos["scope"] or {}
    if frame != 0 and step.kind != "generate":
        fail(f"{path}.pos.frame", "a forward makes one call, the prefill; frame is 0")
    if frame != 0 and "key" in address.COMPONENTS[site.component]["dims"]:
        fail(f"{path}.site", f"{site.component}'s key axis grows with every decode call; read it in the prefill")
    if write and frame != 0:
        decided(f"{path}.pos", pos)
    if "column" in scope:
        text_ref(cx, scope["column"], f"{path}.pos.scope.column", dataset)
    ragged = positions.width(pos) is None
    dims = selection = address.dims(site.component, site.layers, ragged)
    for i, o in enumerate(spec.ops):
        at = f"{path}.ops[{i}]"
        if o.op == "readout" and not address.COMPONENTS[site.component]["stream"]:
            fail(at, f"readout projects a residual-stream value; {site.component} is not one")
        if space(o) is not None:
            featurizer_ref(cx, space(o), f"{at}.featurizer")
        found = references(cx, at, o.op, arguments(o))
        aligned(at, (dataset, repr(step.input)), found)
        if write:
            operands(at, found, selection, space(o), ragged)
        dims = layout(at, o.op, arguments(o), dims, "write" if write else "read")
        if write and dims != selection:
            fail(at, f"{o.op!r} turns {selection} into {dims}; a write's ops keep the value's layout")
    record = {**spec.model_dump(exclude_defaults=True), "site": site.model_dump(exclude_defaults=True)}
    return record if write else {**record, "dims": dims}


def op_step(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    of = value_ref(cx, spec.of, f"{path}.of")
    dataset = aligned(path, (of["data"], repr(spec.of)), references(cx, path, spec.op, arguments(spec)))
    dims = layout(path, spec.op, arguments(spec), of["dims"], "step")
    cx.names[name] = {"dims": dims, "data": dataset, "kind": "value"}
    return {**spec.model_dump(exclude_defaults=True), "dataset": dataset, "dims": dims}


def references(cx: Compiling, path: str, name: str, args: dict) -> dict[str, dict]:
    """Op `name`'s arguments that name a value or a column, each resolved to `{"dims", "data", "kind", "name"}`
    (a column is laid out `["rows"]`); a featurizer argument is checked to exist."""
    found = {}
    for key, p in ops.parameters(name).items():
        value, at = args.get(key), f"{path}.{key}"
        if p.annotation == ops.Value and isinstance(value, str):
            found[key] = {**value_ref(cx, value, at), "name": value}
        elif p.annotation == ops.Tokens and value is not None:
            found[key] = {"dims": ["rows"], "data": column_ref(cx, value, at, None), "kind": "column", "name": value}
        elif p.annotation == ops.Featurizer and value is not None:
            featurizer_ref(cx, value, at)
    return found


def aligned(path: str, of: tuple[str | None, str], found: dict) -> str | None:
    """The one dataset an op's rows align to: that of the value it acts on (`of`: its dataset, and how to name
    it), of its operands with a rows axis, and of its columns."""
    datasets = {of[0]: of[1]} if of[0] else {}
    for info in found.values():
        if info["data"] is not None and info["dims"][:1] in (["rows"], ["entries"], ["layers"]):
            datasets.setdefault(info["data"], repr(info["name"]))
    if len(datasets) > 1:
        (a, x), (b, y) = list(datasets.items())[:2]
        fail(path, f"{x} has a row per example of {a!r} and {y} of {b!r}; an op takes one dataset's rows")
    return next(iter(datasets), None)


def operands(path: str, found: dict, selection: list[str], featurizer: str | None, ragged: bool) -> None:
    """Inside a write an operand stands for the selection: its own layout (in a featurizer's space, its
    features), or one vector of its features; only the vector when the selection is ragged."""
    own = selection[:-1] + ["features"] if featurizer is not None else selection
    allowed = [[a for a in own if a not in address.EXAMPLE_AXES]] + ([] if ragged else [own])
    for key, info in found.items():
        if info["kind"] == "value" and info["dims"] not in allowed:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; this write's selection is {own}, "
                 f"so an operand is {' or '.join(map(str, allowed))}")


def layout(path: str, name: str, args: dict, dims: list[str], where: str) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims`. Inside a read or write (`where`) the op sees
    one batch, so it may not aggregate over rows; `original` is only a write's."""
    if where != "write" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside a write")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args})
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and {"rows", "entries"} & (set(dims) - set(out)):
        fail(path, f"{name!r} aggregates over rows, and inside a read or write it sees one batch; "
             "make it a step after the forward")
    return out


def fit(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    for i, f in enumerate(spec.train):
        if f not in plan.featurizers:
            fail(f"{path}.train[{i}]", f"no featurizer {f!r}; declared: {', '.join(plan.featurizers) or 'none'}")
    within = replace(cx, names=dict(cx.names))
    for f in spec.train:
        within.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
    body = {n: step(within, f"{path}.steps.{n}", n, s, plan) for n, s in spec.steps.items()}
    inner = within.names
    body_ops = [n for n, s in body.items() if s["kind"] == "op"]
    for term in spec.objective:
        thing, _, scalar = term.partition(".")
        if term not in body_ops and not (thing in spec.train and scalar in featurizers.SCALARS.get(
                points_kind(plan, thing), ())):
            fail(f"{path}.objective.{term}", f"not a body op or a trained featurizer's scalar; body ops: "
                 f"{', '.join(body_ops)}")
    if spec.early_stop is not None and spec.early_stop.metric not in body_ops:
        fail(f"{path}.early_stop.metric",
             f"{spec.early_stop.metric!r} is not a body op; body ops: {', '.join(body_ops)}")
    for a, b in (spec.eval.data.items() if spec.eval else []):
        for d in (a, b):
            if d not in cx.rows:
                fail(f"{path}.eval.data", f"no dataset {d!r}; datasets: {', '.join(cx.rows)}")
    for f in spec.train:
        cx.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
    for n, s in body.items():
        if s["kind"] == "op":
            cx.names[f"{name}.{n}"] = inner[n]
            cx.names[f"{name}.history.{n}"] = {"dims": ["epochs"], "data": None, "kind": "value"}
        for read in s.get("reads", {}):
            cx.names[f"{name}.{n}.{read}"] = inner[f"{n}.{read}"]
    return {**spec.model_dump(exclude_defaults=True), "steps": body}


def points_kind(plan: Plan, name: str) -> str:
    f = plan.featurizers[name]
    return (f[0] if isinstance(f, list) else f).kind


# -- references --------------------------------------------------------------------------------------


def value_ref(cx: Compiling, value, path: str) -> dict:
    """A named value produced before this point."""
    info = cx.names.get(value) if isinstance(value, str) else None
    if info is None or info["kind"] != "value":
        known = sorted(k for k, v in cx.names.items() if v["kind"] == "value")
        fail(path, f"{value!r} names no value produced before this step; known: {', '.join(known) or 'none'}")
    cx.refs.append(value)
    return info


def column_ref(cx: Compiling, value, path: str, dataset: str | None) -> str:
    """`<dataset>.<column>`, a column present on every row; of `dataset` when one is given. Returns the dataset."""
    name, _, column = value.partition(".") if isinstance(value, str) else ("", "", "")
    if name not in cx.rows:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx.rows)}")
    if any(column not in r for r in cx.rows[name]):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(cx.rows[name][0])}")
    if dataset is not None and name != dataset:
        fail(path, f"{value!r} is a column of {name!r}, and the value it goes with has a row per example of "
             f"{dataset!r}")
    return name


def decided(path: str, pos: dict) -> None:
    """A write beyond the prefill lands at each call it selects as that call runs, so it takes no scope (the
    text a scope names may not be generated yet), and the calls it lands at must be known before the run
    (`positions.reach`)."""
    if pos["scope"] is not None:
        fail(f"{path}.scope", "a write beyond the prefill lands as its call runs, before the text a scope names "
             "is all generated; it takes no scope")
    if not positions.reach(pos)[2]:
        fail(path, f"frame {pos['frame']!r} with index {pos['index']!r} names calls that depend on how far each row "
             "gets; a write lands as its call runs, so it names a call, a span of calls or 'generated' with an "
             "index counted from their start (or -1 of a single call), or the prefill with them and index 'all'")


def text_ref(cx: Compiling, value, path: str, dataset: str) -> None:
    """`<dataset>.<column path>` naming text on every row of `dataset` (the step's)."""
    name, _, column = value.partition(".")
    column_ref(cx, f"{name}.{column.split('[')[0].split('.')[0]}", path, dataset)
    for i, row in enumerate(cx.rows[name]):
        try:
            found = data.lookup(row, column)
        except (KeyError, IndexError, TypeError):
            fail(path, f"{value!r} is not on row {i} of {name!r}")
        if not isinstance(found, str):
            fail(path, f"{value!r} is {type(found).__name__} on row {i} of {name!r}; a scope's column is text")


def featurizer_ref(cx: Compiling, value, path: str) -> None:
    if not (isinstance(value, str) and cx.names.get(value, {}).get("kind") == "featurizer"):
        known = sorted(k for k, v in cx.names.items() if v["kind"] == "featurizer")
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")
    cx.refs.append(value)


# -- featurizers -------------------------------------------------------------------------------------


def featurizer(cx: Compiling, plan: Plan, name: str) -> dict:
    """A declared featurizer at one point, with the bundle its `file_path` names, checked key by key against
    the document."""
    f, path = plan.featurizers[name], f"featurizers.{name}"
    assert not isinstance(f, list)
    fields = f.model_extra or {}
    if fields.get("classes") is not None:
        column_ref(cx, fields["classes"], f"{path}.classes", None)
    record = f.model_dump(exclude_defaults=True)
    if fields.get("file_path") is None:
        return record
    try:
        return {**record, "bundle": featurizers.bundle(cx.root, f.kind, fields, plan.model)}
    except ValueError as e:
        fail(f"{path}.file_path", str(e))
