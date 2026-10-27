"""What a document means, decided before any forward in two passes: here, with the document alone, a plan
resolved for running, every name checked, every sweep expanded into points; and in `preflight`, with the model
in hand, everything the model decides (tokens, positions per row, answers' ids, selects and widths against its
sizes), collected into what the run needs (`state.Found`).

The package holds the two drivers and the concepts they resolve: `sweeps` and `data` are the format's own; `answers`
(a column scored against a vocabulary), `positions` (`pos` resolved per row), `arguments` (an op's arguments
against what compile knows), `interventions` (an entry, its brackets and operands) and `uses` (a featurizer's uses in
a plan) are here, each with its checks at compile and what it resolves to in the run.

Compile knows the document and the files; the run knows the model. Anything decided from the document, the
data files and bundle headers is checked here; anything that needs the tokenizer or a size of the model is
checked by `preflight.preflight`, inside the run. The records' own shape (fields, types, the pos grammar, a
site's narrowing) is the models' to check, when the plan is built. How names resolve is `step`'s docstring.
"""

import itertools
from dataclasses import replace
from typing import Any, Callable

from clab import featurizers
from clab.compile.answers import column_ref
from clab.compile.arguments import aligned, called, layout, narrowed_readout, paired, references, space_of, value_ref
from clab.compile.interventions import entry, items
from clab.compile.state import Compiling, Known
from clab.compile.uses import bundled_sites
from clab.featurizers import KINDS, bundles, labels
from clab.format import data
from clab.format.sweeps import find_sweeps
from clab.plan import (
    Data,
    Entry,
    Fit,
    Forward,
    Generate,
    Intervention,
    OpStep,
    Plan,
    arguments,
    corpus_root,
    fail,
    gathers,
    kind_of,
    points,
)


def compile(plan: Plan) -> Plan:
    """The plan compiled: a new `Plan`, `compiled=True`."""
    if plan.compiled:
        return plan
    root = corpus_root(plan)
    doc = plan.model_dump(exclude_defaults=True, exclude={"run", "document"})
    rows = {name: dataset(root, name, d) for name, d in plan.data.items()}
    sweeps = find_sweeps(doc)
    values = {label: get(doc, path)["sweep"] for path, label in sweeps.items()}
    cx = Compiling(doc=doc, root=root, rows=rows, sweeps=sweeps, values=values, names={}, axes_of={}, refs=[],
                   used=set())
    compiled = {"featurizers": {}, "steps": {}}
    for name in plan.featurizers:
        compiled["featurizers"][name] = expand(cx, name, ("featurizers", name),
                                               lambda point, name=name: featurizer(cx, point, name))
        cx.names[name] = Known(kind="featurizer")
        for scalar in featurizers.KINDS[kind_of(plan, name)].scalars:
            cx.names[f"{name}.{scalar}"] = Known(kind="scalar")
    for name in plan.steps:
        compiled["steps"][name] = expand(cx, name, ("steps", name), lambda point, name=name: step(
            cx, f"steps.{name}", name, point.steps[name], point))
    saveable = sorted(n for n, v in cx.names.items() if n.partition(".")[0] in plan.steps and v.kind != "scalar")
    for i, name in enumerate(plan.saves):
        if name in plan.saves[:i]:
            fail(f"saves[{i}]", f"{name!r} is saved twice")
        if name not in saveable:
            fail(f"saves[{i}]", f"{name!r} names no step's value; a save is a step or a step's member: "
                 f"{', '.join(saveable)}")
    for name, d in plan.data.items():
        if d.template is not None and name not in cx.chats:
            fail(f"data.{name}.template", f"no step reads a chat column of {name!r}, and a template renders a chat "
                 "(a column of {'role', 'content'} messages) only")
    gathered(compiled["steps"], cx.used | set(plan.saves))
    done = Plan(**{**doc, **compiled, "sites": {}, "interventions": {}, "compiled": True, "document": plan.document,
                   "sweeps": {label: {"paths": [".".join(map(str, path)) for path, of in sweeps.items() if of == label],
                                      "values": values[label]} for label in values},
                   "data": {n: {**doc["data"][n], **({} if plan.data[n].hf is not None else {"rows": rows[n]}),
                                "sha256": data.digest(rows[n]), "n_rows": len(rows[n])} for n in rows}})
    bundled_sites(done)
    return done


def dataset(root: str, name: str, d: Data) -> list[dict]:
    """Dataset `name`'s rows as materialized, which its `sha256` and `n_rows` describe: its file's (under `root`),
    its inline rows or its hub split's (`data.hub`), then a file's `split`, `shuffle`, `take` and `columns`, in
    that order (`Data`). The one way a dataset's rows are made: compile's, and `materialized`'s where they are not
    in the plan."""
    try:
        rows = data.hub(d) if d.hf is not None else d.rows if d.path is None else data.rows(root, d.path, d.split)
    except ValueError as e:
        fail(f"data.{name}.{'path' if d.hf is None else 'hf'}", str(e))
    if not rows:
        fail(f"data.{name}", "the dataset has no rows" + (f" in split {d.split!r}" if d.split else ""))
    rows = data.shuffled(rows) if d.shuffle else rows
    if d.take is not None and d.take > len(rows):
        fail(f"data.{name}.take", f"take {d.take}, and the dataset has {len(rows)} rows"
             + (f" in split {d.split!r}" if d.split else "") + "; take keeps at most as many as there are")
    rows = rows[:d.take]
    for new, column in (d.columns or {}).items():
        lacking = next((i for i, r in enumerate(rows) if column not in r), None)
        if lacking is not None:
            fail(f"data.{name}.columns.{new}", f"{column!r} is not a column of row {lacking}; its columns: "
                 f"{', '.join(rows[lacking])}")
        if new not in (d.columns or {}).values() and any(new in r for r in rows):
            fail(f"data.{name}.columns.{new}", f"{new!r} is a column the rows keep, which this rename would "
                 "overwrite; rename it too, or name the new column otherwise")
    return data.renamed(rows, d.columns) if d.columns else rows


def materialized(plan: Plan, name: str) -> list[dict[str, Any]]:
    """A dataset's rows: the plan's, or, where they are not in it (a hub dataset's, always; a file's, after
    `Plan.load`), materialized again by `dataset` and checked against the digest compile recorded."""
    d = plan.data[name]
    if d.rows is not None:
        return d.rows
    rows = dataset(corpus_root(plan), name, d)
    if data.digest(rows) != d.sha256:
        fail(f"data.{name}", f"{data.hub_name(d)} gives other rows here than compile read (their digest is not "
             "the one it recorded)" if d.hf is not None else f"data/{d.path}.json is not what the run read (its rows' "
             "digest differs)")
    return rows


def get(doc: dict, path: tuple) -> Any:
    for key in path:
        doc = doc[key]
    return doc


# -- sweeps: each entry compiled at every point of its axes -------------------------------------------


def expand(cx: Compiling, name: str, where: tuple, build: Callable[[Plan], dict]) -> dict | list[dict]:
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


def point(cx: Compiling, at: dict[str, int]) -> Plan:
    """The document with every sweep `at` names replaced by its value there, built and validated once per
    distinct `at` (`cx.points`): every step of an unswept document compiles against the one plan."""
    key = tuple(sorted(at.items()))
    if key not in cx.points:
        def substitute(value: Any, path: tuple) -> Any:
            if cx.sweeps.get(path) in at:
                return value["sweep"][at[cx.sweeps[path]]]
            if isinstance(value, dict):
                return {k: substitute(v, path + (k,)) for k, v in value.items()}
            if isinstance(value, list):
                return [substitute(v, path + (i,)) for i, v in enumerate(value)]
            return value
        cx.points[key] = Plan(**substitute(cx.doc, ()))
    return cx.points[key]


def footprint(cx: Compiling, where: tuple) -> list[tuple]:
    """The document paths an entry reads: its own, and the sites, interventions and featurizers it names."""
    found, doc = [where], cx.doc

    def walk(value: Any) -> None:
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


def step(cx: Compiling, path: str, name: str, spec: Entry, plan: Plan) -> dict:
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
    assert not isinstance(spec, list), "a point holds one step under each name"
    return {"forward": forward, "generate": forward, "op": op_step, "fit": compile_fit}[spec.kind](cx, path, name, spec, plan)


def forward(cx: Compiling, path: str, name: str, spec: Forward | Generate, plan: Plan) -> dict:
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
        if not joined and not isinstance(data.field(rows[0], part), str):
            cx.chats.add(dataset)
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
        records[k], made[k] = entry(cx, p, e, plan, spec, dataset, live, own)
        # A forward's entries may name the ones before them, which the engine makes as the forward reaches them;
        # a generation's land call by call, so they may not yet.
        if spec.kind == "forward":
            cx.names[f"{name}.{k}"] = made[k]
            live[f"{name}.{k}"] = (records[k], e)
    cx.names.update({f"{name}.{k}": v for k, v in made.items()})
    if spec.kind == "generate":
        cx.names[name] = Known(["rows", "new"], dataset)
    return {**spec.model_dump(exclude_defaults=True, exclude={"interventions"}), "interventions": records,
            "dataset": dataset}


def op_step(cx: Compiling, path: str, name: str, spec: OpStep, plan: Plan) -> dict:
    if spec.op == "decode":
        fail(path, "decode closes an encode of its own entry, so it runs inside the entry that encodes, after the "
             "encode and the edits of its coordinates; as a step there is none to close (detokenize turns a "
             "generation's ids into text)")
    of, args = value_ref(cx, spec.of, f"{path}.of"), arguments(spec)
    if spec.op == "contains" and not of.text:
        fail(f"{path}.of", f"contains reads text, a detokenize's value (without tokens); {spec.of!r} is not one")
    kind = called(cx, plan, path, spec.op, args)
    found = references(cx, path, spec.op, args, "classes" in of.dims)
    dataset = aligned(path, (of.data, repr(spec.of)), found)
    paired(path, spec.op, of.dims, of.width, found, step=True)
    if spec.op == "readout" and "hidden_size" in of.select:
        fail(path, narrowed_readout(repr(spec.of)))
    dims = layout(path, spec.op, args, of.dims, "step", kind)
    # An encode's value is in its featurizer's coordinates, as an entry's is that leaves its encode open.
    space = of.space + space_of(cx, [spec]) if spec.op == "encode" else of.space if dims[-1:] == of.dims[-1:] else []
    cx.names[name] = Known(dims, dataset, text=spec.op == "detokenize" and not args.get("tokens"), space=space,
                           select={a: v for a, v in of.select.items() if a in dims},
                           width=of.width if "pos" in dims else None)
    return {**spec.model_dump(exclude_defaults=True), "dataset": dataset, "dims": dims}


# -- references --------------------------------------------------------------------------------------


def gathered(steps: dict, used: set[str]) -> None:
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


def featurizer(cx: Compiling, plan: Plan, name: str) -> dict:
    """A declared featurizer at one point, with the bundle its `file_path` names, checked key by key against
    the document."""
    f, path = plan.featurizers[name], f"featurizers.{name}"
    assert not isinstance(f, list)
    fields, classes = f.model_extra or {}, None
    if fields.get("classes") is not None:
        column_ref(cx, fields["classes"], f"{path}.classes", None)
        try:
            classes = labels(cx.rows, fields["classes"])
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
        tensors, header = bundles.load(cx.root, f.kind, fields, plan.model, classes)
    except ValueError as e:
        fail(f"{path}.file_path", str(e))
    return {**record, "bundle": tensors, "header": header}


def compile_fit(cx: Compiling, path: str, name: str, spec: Fit, plan: Plan) -> dict:
    """A fit's body runs over minibatches of its datasets' rows, the same rows of each: so its datasets have
    as many rows as each other (under each of `eval`'s substitutions too), a value made before the fit that the body
    uses with a row per example is of a dataset a forward of the body reads (with no forward, of one dataset), and
    no eval reads such a value's dataset as another. It trains featurizers that have something to train, its
    objective depends on one of them, and it holds no fit."""
    for i, f in enumerate(spec.train):
        if f not in plan.featurizers:
            fail(f"{path}.train[{i}]", f"no featurizer {f!r}; declared: {', '.join(plan.featurizers) or 'none'}")
        trainable = [k for k, c in featurizers.KINDS.items() if c.trains]
        if kind_of(plan, f) not in trainable:
            fail(f"{path}.train[{i}]", f"{f!r} is a {kind_of(plan, f)}, which a bundle fixes; it has nothing to "
                 f"train (trainable kinds: {', '.join(trainable)})")
        if any((p.model_extra or {}).get("file_path") for p in points(plan.featurizers[f])):
            fail(f"{path}.train[{i}]", f"{f!r} is loaded from a bundle, which fixes it; a fit trains a featurizer "
                 "it draws")
    within = replace(cx, names=dict(cx.names), aliases={f: f"{name}.{f}" for f in spec.train})
    for f in spec.train:
        within.names[f"{name}.{f}"] = Known(kind="featurizer")
    body, harvests = {}, {}         # harvests: each value made before the fit, with rows, that the body uses, by user
    trained = set(spec.train) | set(within.aliases.values())    # what the fit trains, by either of its names
    for n, s in spec.steps.items():
        if isinstance(s, Fit):
            fail(f"{path}.steps.{n}", "a fit's body holds no fit")
        if n in plan.data or n in plan.featurizers or "." in n:
            fail(f"{path}.steps.{n}", f"{n!r} is also a dataset or a featurizer, or has a dot; a body's names are "
                 "a root's")
        start = len(cx.refs)
        body[n] = step(within, f"{path}.steps.{n}", n, s, plan)
        for r in cx.refs[start:]:
            if within.names[r] is cx.names.get(r) and "rows" in within.names[r].dims:
                harvests.setdefault(r, n)
        if any(r in trained or r.partition(".")[0] in trained for r in cx.refs[start:]):
            trained.add(n)                  # a step whose value depends on what the fit trains
    inner = within.names
    body_ops = [n for n, s in body.items() if s["kind"] == "op"]
    for term in spec.objective:
        thing, _, scalar = term.partition(".")
        if term not in body_ops and not (thing in spec.train and scalar in featurizers.KINDS[
                kind_of(plan, thing)].scalars):
            fail(f"{path}.objective.{term}", f"not a body op or a trained featurizer's scalar; body ops: "
                 f"{', '.join(body_ops)}")
        if term not in body_ops:            # a scalar is over what the body's reads of its coordinates saw
            for n, s in body.items():
                for k, e in s.get("interventions", {}).items():
                    if any(o.op == "encode" and arguments(o)["featurizer"] in (thing, f"{name}.{thing}")
                           for o in gathers(Intervention(**e))):
                        cx.used.add(f"{n}.{k}")
    if not any(t.partition(".")[0] in trained for t in spec.objective):
        fail(f"{path}.objective", f"none of its terms ({', '.join(spec.objective)}) depends on what the fit trains "
             f"({', '.join(spec.train)}): each is a body op over values no trained featurizer reaches, so there is "
             "no gradient to step on; make a term of an op over a value the trained featurizer's encode, decode or "
             "apply made, or of its scalar")
    if spec.early_stop is not None and spec.early_stop.metric not in body_ops:
        fail(f"{path}.early_stop.metric",
             f"{spec.early_stop.metric!r} is not a body op; body ops: {', '.join(body_ops)}")
    for label, swap in spec.eval.items():
        where = f"{path}.eval.{label}"
        if label == "history" or label in spec.train or "." in label:
            fail(where, f"{label!r} names the fit's history or a featurizer it trains, or has a dot; an eval's name "
                 "is its own")
        for a, b in swap.items():
            for d in (a, b):
                if d not in cx.rows:
                    fail(where, f"no dataset {d!r}; datasets: {', '.join(cx.rows)}")
            for ref in harvests:
                if cx.names[ref].data == a:
                    fail(where, f"the body uses {ref!r}, made before the fit with a row per example of {a!r}, and "
                         f"this eval reads {a!r} as {b!r}, over whose rows no step made it; evaluate on {b!r} in "
                         "steps after the fit, which reach what it trained as "
                         f"{', '.join(repr(f'{name}.{f}') for f in spec.train)}")
            if plan.data[a].template != plan.data[b].template:
                fail(where, f"dataset {a!r} is read as {b!r}, and their templates differ ({a!r}: "
                     f"{plan.data[a].template}, {b!r}: {plan.data[b].template}); a chat renders with its own "
                     "dataset's template, so give the two the same one")
            if a in cx.chats:                   # read as `a`, its chats render with the template they share
                cx.chats.add(b)
    for label, side in {"": {}, **spec.eval}.items():
        counts = {d: len(cx.rows[side.get(d, d)]) for d in {s.get("dataset") for s in body.values()} - {None}}
        if len(set(counts.values())) > 1:
            fail(f"{path}.steps", f"the body's datasets have {counts} rows" + (f" under eval {label!r}" if label else "")
                 + "; it runs over minibatches of the same rows of each, paired row by row, so they have as many rows: "
                 "repeat the smaller dataset's rows (or cut the larger's) to the same count, or fit on one")
    # The rows a minibatch's ids are: of the datasets the body's forwards read, or, with none, its first harvest's.
    own = {s["dataset"] for s in body.values() if s["kind"] != "op"} or {cx.names[r].data for r in list(harvests)[:1]}
    for ref, n in harvests.items():
        if cx.names[ref].data not in own:
            d, theirs = cx.names[ref].data, " and ".join(map(repr, sorted(own)))
            fail(f"{path}.steps.{n}", f"the body uses {ref!r}, made before the fit with a row per example of {d!r}, "
                 f"which no forward of the body reads; the body runs over the rows of {theirs}, so a minibatch would "
                 f"take rows of {d!r} at the ids of its rows of {theirs}, pairing the two row by row: read {d!r} in a "
                 f"forward of the body, or make {ref!r} over {theirs}")
    for f in spec.train:
        cx.names[f"{name}.{f}"] = Known(kind="featurizer")
        for scalar in featurizers.KINDS[kind_of(plan, f)].scalars:
            cx.names[f"{name}.history.{f}.{scalar}"] = Known(["epochs"])
    for n, s in body.items():
        if s["kind"] == "op":
            cx.names[f"{name}.history.{n}"] = Known(["epochs"])
        members = ([n] if s["kind"] == "op" else []) + [f"{n}.{k}" for k in s.get("interventions", {})]
        for label, swap in spec.eval.items():
            for member in members:
                info = inner[member]
                cx.names[f"{name}.{label}.{member}"] = replace(info, data=swap[info.data] if info.data in swap
                                                               else info.data)
    return {**spec.model_dump(exclude_defaults=True), "steps": body}
