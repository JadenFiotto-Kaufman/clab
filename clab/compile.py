"""compile: a plan resolved for running, every name checked, every sweep expanded into points.

Compile knows the document and the files; the run knows the model. Anything decided from the document, the
data files and bundle headers is checked here; anything that needs the tokenizer or a size of the model is
checked by `steps.preflight`, inside the run. The records' own shape (fields, types, the pos grammar, a
site's narrowing) is the models' to check, when the plan is built. How names resolve is `step`'s docstring.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

from . import data, featurizers, positions
from .answers import column_ref
from .featurizers.uses import bundled_sites, featurizer
from .fit import compile_fit
from .interventions import aligned, called, entry, items, layout, narrowed_readout, paired, references, value_ref
from .plan import Intervention, Plan, arguments, corpus_root, fail, kind_of
from .sweeps import find_sweeps


def compile(plan: Plan) -> Plan:
    """The plan compiled: a new `Plan`, `compiled=True`."""
    if plan.compiled:
        return plan
    root = corpus_root(plan)
    doc = plan.model_dump(exclude_defaults=True, exclude={"run", "document"})
    rows = {}
    for name, d in plan.data.items():
        try:
            rows[name] = d.rows if d.path is None else data.rows(root, d.path, d.split)
        except ValueError as e:
            fail(f"data.{name}.path", str(e))
        if not rows[name]:
            fail(f"data.{name}", "the dataset has no rows" + (f" in split {d.split!r}" if d.split else ""))
    sweeps = find_sweeps(doc)
    values = {label: get(doc, path)["sweep"] for path, label in sweeps.items()}
    cx = Compiling(doc=doc, root=root, rows=rows, sweeps=sweeps, values=values, names={}, axes_of={}, refs=[],
                   used=set())
    compiled = {"featurizers": {}, "steps": {}}
    for name in plan.featurizers:
        compiled["featurizers"][name] = expand(cx, name, ("featurizers", name),
                                               lambda point, name=name: featurizer(cx, point, name))
        cx.names[name] = {"dims": [], "data": None, "kind": "featurizer"}
        for scalar in featurizers.KINDS[kind_of(plan, name)].scalars:
            cx.names[f"{name}.{scalar}"] = {"dims": [], "data": None, "kind": "scalar"}
    for name in plan.steps:
        compiled["steps"][name] = expand(cx, name, ("steps", name), lambda point, name=name: step(
            cx, f"steps.{name}", name, point.steps[name], point))
    saveable = sorted(n for n, v in cx.names.items() if n.partition(".")[0] in plan.steps and v["kind"] != "scalar")
    for i, name in enumerate(plan.saves):
        if name in plan.saves[:i]:
            fail(f"saves[{i}]", f"{name!r} is saved twice")
        if name not in saveable:
            fail(f"saves[{i}]", f"{name!r} names no step's value; a save is a step or a step's member: "
                 f"{', '.join(saveable)}")
    gathered(compiled["steps"], cx.used | set(plan.saves))
    done = Plan(**{**doc, **compiled, "sites": {}, "interventions": {}, "compiled": True, "document": plan.document,
                   "sweeps": {label: {"path": ".".join(map(str, path)), "values": values[label]}
                              for path, label in sweeps.items()},
                   "data": {n: {**doc["data"][n], "rows": rows[n], "sha256": data.digest(rows[n]),
                                "n_rows": len(rows[n])} for n in rows}})
    bundled_sites(done)
    return done


@dataclass
class Compiling:
    """What compiling an entry needs beside the entry, passed to every function that resolves a name.

    Read only: the document as a dict (`doc`), the corpus root, each dataset's rows, each sweep's label by
    document path (`sweeps`) and its values by label (`values`). Filled as entries compile, in document
    order: each name produced so far, `{dims, data, kind}` and, for a value, what `value_of` records of it
    (`names`), each entry's sweep axes (`axes_of`), the names the entry being compiled references (`refs`),
    which decide the axes it inherits, and every value any entry references (`used`), which decides what the run
    gathers. A fit's body compiles in a copy whose
    `names` is its own scope; it shares `refs`, since what the body references is what the fit references,
    `axes_of` and `used`, and maps each featurizer it trains to the name it resolves to there (`aliases`:
    `rot` is `<fit>.rot`, the instance being trained)."""
    doc: dict
    root: str
    rows: dict
    sweeps: dict
    values: dict
    names: dict
    axes_of: dict
    refs: list
    used: set
    aliases: dict = field(default_factory=dict)


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
                        if isinstance(n, str):
                            found.append(("interventions", n))
                            walk(doc.get("interventions", {}).get(n))
                if k in ("featurizer", "train"):
                    found.extend(("featurizers", n) for n in named if isinstance(n, str) and n in doc.get("featurizers", {}))
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
    `counterfactual`'s entry `v_cf`; `mean` is an op step's value; `pairs.cf_answer` is a dataset's column; `rot`
    is a declared featurizer and `mask.mask` its scalar; `fit.rot` is a fit's member. Values are produced in
    document order and referenced only after they exist. A fit's body is a scope of its own: it sees every
    value produced before the fit, and nothing leaves it except through `<fit>.<member>`. A bundle of entries
    is a template, so the names inside it resolve where a step pulls it in. Under a sweep, a name means the
    producer's point that agrees with the consumer's coordinates on the producer's axes. Every value with a
    rows axis knows the dataset its rows are aligned to (a forward's input; an op's `of`, operands and
    columns), and an op takes one dataset's rows.
    """
    return {"forward": forward, "generate": forward, "op": op_step, "fit": compile_fit}[spec.kind](cx, path, name, spec, plan)


def forward(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    joined = not isinstance(spec.input, str)
    dataset = None
    for i, part in enumerate(spec.input if joined else [spec.input]):
        if not isinstance(part, str):
            continue
        at = f"{path}.input" + (f"[{i}]" if joined else "")
        name_, column, index = data.parse_input(part)
        if name_ not in cx.rows:
            fail(at, f"no dataset {name_!r}; datasets: {', '.join(cx.rows)}")
        dataset = dataset or name_
        column_ref(cx, f"{name_}.{column}", at, dataset)
        rows = cx.rows[dataset]
        if index is not None and any(not isinstance(r[column], list) or index >= len(r[column]) for r in rows):
            fail(at, f"column {column!r} has no entry {index} on every row")
        if joined:
            for r, row in enumerate(rows):
                if not isinstance(data.field(row, part), str):
                    fail(at, f"{part!r} is not text on row {r} of {dataset!r}; a list input joins text")
    assert dataset is not None
    given = items(plan, path, name, spec)
    keys = [k for _, k, _ in given]
    for k in keys:
        if keys.count(k) > 1:
            fail(f"{path}.interventions", f"two entries named {k!r}; a step binds each name once")
    own = {f"{name}.{k}" for k in keys}
    records: dict[str, dict] = {}
    live: dict[str, tuple[dict, Intervention]] = {}
    made = {}
    for p, k, e in given:
        records[k] = entry(cx, p, e, plan, spec, dataset, live, own)
        made[k] = value_of(records[k], e, dataset)
        # A forward's entries may name the ones before them, which the engine makes as the forward reaches them;
        # a generation's land call by call, so they may not yet.
        if spec.kind == "forward":
            cx.names[f"{name}.{k}"] = made[k]
            live[f"{name}.{k}"] = (records[k], e)
    cx.names.update({f"{name}.{k}": v for k, v in made.items()})
    if spec.kind == "generate":
        cx.names[name] = {"dims": ["rows", "new"], "data": dataset, "kind": "value"}
    return {**spec.model_dump(exclude_defaults=True, exclude={"interventions"}), "interventions": records,
            "dataset": dataset}


def value_of(record: dict, spec: Intervention, dataset: str) -> dict:
    """What a name records of an entry's value: its layout, its dataset, its coordinates (`space`), its site's
    `select` on the axes it still has, and how many positions it holds on every row (`width`, None when
    ragged)."""
    dims = record["dims"]
    return {"dims": dims, "data": dataset, "kind": "value", "space": record.get("space", []),
            "select": {a: v for a, v in (record["site"].get("select") or {}).items() if a in dims},
            "width": positions.width(positions.parse(spec.pos)) if "pos" in dims else None}


def op_step(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    if spec.op == "decode":
        fail(path, "decode closes an encode of its own entry, so it runs inside the entry that encodes, after the "
             "encode and the edits of its coordinates; as a step there is none to close (detokenize turns a "
             "generation's ids into text)")
    of = value_ref(cx, spec.of, f"{path}.of")
    if spec.op == "contains" and not of.get("text"):
        fail(f"{path}.of", f"contains reads text, a detokenize's value (without tokens); {spec.of!r} is not one")
    kind = called(cx, plan, path, spec.op, arguments(spec))
    found = references(cx, plan, path, spec.op, arguments(spec), "classes" in of["dims"])
    dataset = aligned(path, (of["data"], repr(spec.of)), found)
    paired(path, spec.op, of["dims"], of.get("width"), found)
    if spec.op == "readout" and "hidden_size" in of.get("select", {}):
        fail(path, narrowed_readout(repr(spec.of)))
    dims = layout(path, spec.op, arguments(spec), of["dims"], "step", kind)
    cx.names[name] = {"dims": dims, "data": dataset, "kind": "value",
                      "text": spec.op == "detokenize" and not arguments(spec).get("tokens"),
                      "space": of.get("space", []) if dims[-1:] == of["dims"][-1:] else [],
                      "select": {a: v for a, v in of.get("select", {}).items() if a in dims},
                      "width": of.get("width") if "pos" in dims else None}
    return {**spec.model_dump(exclude_defaults=True), "dataset": dataset, "dims": dims}


# -- references --------------------------------------------------------------------------------------


def gathered(steps: dict, used: set) -> None:
    """Marks each compiled entry the run takes off the model (`gathered`): one `used` names (a value something
    references, a save, or the entries a fit's objective scalar is over), and in a fit's body one a
    `<fit>.<eval>.<step>.<entry>` member names. Any other entry only writes, and an entry that neither writes
    nor is gathered does nothing."""
    for name, entry in steps.items():
        for point in entry if isinstance(entry, list) else [entry]:
            if point["kind"] == "fit":
                inner = {member.partition(".")[2] for member in (u.removeprefix(f"{name}.") for u in used
                                                                if u.startswith(f"{name}."))
                         if member.partition(".")[0] in point.get("eval", {})}
                gathered(point["steps"], used | inner)
            for k, e in point.get("interventions", {}).items():
                e["gathered"] = f"{name}.{k}" in used


# -- featurizers -------------------------------------------------------------------------------------
