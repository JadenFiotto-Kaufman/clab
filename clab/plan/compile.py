"""compile: a plan resolved for running, every name checked, every sweep expanded into points.

Compile knows the document and the files; the run knows the model. Anything decided from the document, the
data files and bundle headers is checked here; anything that needs the tokenizer or a size of the model is
checked by `steps.preflight`, inside the run. The records' own shape (fields, types, the pos grammar, a
site's narrowing) is the models' to check, when the plan is built. How names resolve is `step`'s docstring.
"""

import inspect
import itertools
from dataclasses import dataclass, field, replace
from typing import Any

from .. import address, registry
from ..runtime import data, featurizers, ops, positions
from pydantic import ValidationError

from .model import (Column, Intervention, Op, Plan, Site, Unbracketed, agreeing, arguments, bracketed, corpus_root,
                    edits, fail, find_sweeps, gathers, horizon, kind_of, parameters, points, uses, visited, walk,
                    writes)


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
    return {"forward": forward, "generate": forward, "op": op_step, "fit": fit}[spec.kind](cx, path, name, spec, plan)


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


def items(plan: Plan, path: str, name: str, spec) -> list[tuple[str, str, Intervention]]:
    """A step's entries in the order written, each with its document path and name: the items of its
    `interventions` in turn (a dict is one item), a bundle's entries where its name stands, rescoped to the
    step (`scoped`)."""
    value = spec.interventions
    listed = value if isinstance(value, list) else [value]
    out = []
    for i, item in enumerate(listed):
        at = f"{path}.interventions" + (f"[{i}]" if isinstance(value, list) else "")
        if isinstance(item, dict):
            out += [(f"{at}.{k}", k, e) for k, e in item.items()]
            continue
        if item not in plan.interventions:
            fail(at, f"no intervention bundle {item!r}; declared: {', '.join(plan.interventions) or 'none'}")
        pullers = {n.rpartition(".")[2] for n, s in walk(plan.steps) if item in bundles_of(s)}
        out += [(f"interventions.{item}.{k}", k, scoped(plan, f"interventions.{item}.{k}", e, item, name, pullers))
                for k, e in plan.interventions[item].items()]
    return out


def bundles_of(step) -> list[str]:
    """The bundles a step's `interventions` pull in, by name."""
    value = getattr(step, "interventions", None)
    return [i for i in (value if isinstance(value, list) else [value]) if isinstance(i, str)]


def scoped(plan: Plan, path: str, spec: Intervention, bundle: str, step: str, pullers: set) -> Intervention:
    """A bundle's entry, its operands that name one of the bundle's own entries by its bare name (`"premix"`)
    naming that entry of the step it is pulled into (`"<step>.premix"`). An operand that names `<a step that
    pulls the bundle in>.<one of its entries>` is refused: pulled into another step it would still mean that
    step's entry, not the one this step makes. `pullers` are the steps that pull the bundle in."""
    own = plan.interventions[bundle]
    rewritten = []
    for i, o in enumerate(spec.ops):
        args = arguments(o)
        try:
            params = parameters(plan, o.op, args)
        except (KeyError, ValueError):          # an unknown featurizer or argument, which compile refuses by name
            params = ops.parameters(o.op)
        changes = {}
        for key, value in args.items():
            if key not in params or params[key].annotation != ops.Value or not isinstance(value, str):
                continue
            thing, _, member = value.partition(".")
            if thing in pullers and member in own:
                fail(f"{path}.ops[{i}].{key}", f"{value!r} names step {thing!r}'s entry {member!r}, which is also "
                     f"this bundle's own entry; name the bundle's own entry bare ({member!r}), which means it in "
                     "whichever step pulls the bundle in")
            if value in own:
                changes[key] = f"{step}.{value}"
        rewritten.append(Op(**{**o.model_dump(exclude_defaults=True), **changes}) if changes else o)
    return spec.model_copy(update={"ops": rewritten})


def entry(cx: Compiling, path: str, spec: Intervention, plan: Plan, step, dataset: str,
          live: dict[str, tuple[dict, Intervention]], own: set) -> dict:
    """One entry: its site inlined, its positions checked against the step, its ops' layouts. An entry that
    holds an edit writes: each op up to its last edit acts on the live value, so no edit follows an op that
    changed the layout, unless an `encode` changed it and its `decode` follows, and every `encode` before the
    write-back is closed by then; its operands stand for the value they meet (`operands`). `live` holds the
    earlier entries of a forward, by name, which an operand may name; `own` names every entry of the step."""
    if isinstance(spec.site, Site):
        site = spec.site
    elif spec.site in plan.sites:
        site = plan.sites[spec.site]
    else:
        fail(f"{path}.site", f"no site {spec.site!r}; declared: {', '.join(plan.sites) or 'none'}")
    write = writes(spec)
    if write and address.stacks(site.layers):
        fail(f"{path}.site", f"an entry that writes names its layer, an int; layers {site.layers!r} stack several, "
             "which is a read's (or sweep the layers)")
    if write and len(site.select or {}) > 1:
        fail(f"{path}.site.select", f"an entry that writes selects on one axis; this one selects "
             f"{', '.join(site.select or {})}")
    pos = positions.parse(spec.pos)
    frame, scope = pos["frame"], pos["scope"] or {}
    if step.kind != "generate" and positions.select(frame, 1) != [0]:
        fail(f"{path}.pos.frame", f"frame {frame!r}: a forward makes one call, the prefill; its frame picks that "
             "call (0, or ':', the whole timeline, which in a forward is that call)")
    if frame != 0 and "key" in address.COMPONENTS[site.component]["dims"]:
        fail(f"{path}.site", f"{site.component}'s key axis grows with every decode call; read it in the prefill")
    if write and scope.get("before") and not positions.prefill(pos):
        fail(f"{path}.pos.scope.before", "a write beyond the prefill lands as its call runs, before the token it "
             "would predict is chosen; a scope that is before is a read's there, or a write's in the prefill")
    if write and frame != 0 and step.kind == "generate":
        decided(f"{path}.pos", pos, horizon(step))
    if "turn" in scope and not all(isinstance(data.field(r, step.input), list) for r in cx.rows[dataset]):
        fail(f"{path}.pos.scope.turn", f"the input {data.input_name(step.input)} is text, not a chat: a turn scope "
             "names a chat's messages (a column of {'role', 'content'} lists, rendered through the template), and "
             "text has none; locate the message's words with a {'text': ...} or {'column': ...} scope")
    if "column" in scope:
        for i, column in enumerate(scope["column"] if isinstance(scope["column"], list) else [scope["column"]]):
            text_ref(cx, column, f"{path}.pos.scope.column" + (f"[{i}]" if isinstance(scope["column"], list) else ""),
                     dataset)
    ragged = positions.width(pos) is None
    dims = address.dims(site.component, site.layers, ragged)
    last = len(spec.ops) - len(gathers(spec)) - 1          # the last edit's index; -1 when none
    try:
        tree = bracketed(spec.ops)
    except Unbracketed as e:
        fail(f"{path}.ops[{e.at}]", str(e))
    around = {node["at"]: list(outer) for node, outer in visited(tree)}
    closes = {node["closed"]: node for node, _ in visited(tree) if node.get("closed") is not None}
    before: dict[int, list[str]] = {}                       # the layout each encode turned into coordinates
    changed = None                                          # the op that changed the layout, which no edit follows
    for i, o in enumerate(spec.ops):
        at, args = f"{path}.ops[{i}]", arguments(o)
        outer = around[closes[i]["at"]] if i in closes else around[i]
        if o.op == "readout" and not address.COMPONENTS[site.component]["stream"]:
            fail(at, f"readout projects a residual-stream value; {site.component} is not one")
        if o.op == "readout" and "hidden_size" in (site.select or {}) and "hidden_size" in dims:
            fail(at, narrowed_readout(f"{site.component} select hidden_size {(site.select or {})['hidden_size']!r}"))
        if edits(o) and changed is not None:
            fail(at, f"{o.op!r} is an edit, which writes back, and it follows {changed!r}, which turned the value "
                 f"into {dims}: there is no live value to write; an edit comes before it, or put the transform in "
                 "a later entry at this site")
        ref = args.get("featurizer")
        mine = next((n for n in outer if arguments(n["op"])["featurizer"] == ref), None) if i not in closes else None
        if mine is not None:
            fail(at, f"{o.op!r} of {ref!r} at ops[{i}] is inside the encode of {ref!r} (ops[{mine['at']}]), so it "
                 f"would act on {ref!r}'s own coordinates; put it before that encode or after the decode that closes it")
        kind = called(cx, plan, at, o.op, args)
        for key, value in args.items():
            if isinstance(value, str) and value in own and value not in live:
                fail(f"{at}.{key}", f"{value!r} is an entry of this generation; an operand from an entry of its own "
                     "step is served in a forward, not yet inside a generate" if step.kind == "generate" else
                     f"{value!r} is this entry or one this forward makes after it; an operand names an earlier "
                     "entry")
        found = references(cx, plan, at, o.op, args, "classes" in dims)
        aligned(at, (dataset, repr(step.input)), found)
        paired(at, o.op, dims, positions.width(pos), found)
        if i <= last:
            chosen = set(parameters(plan, o.op, args)) - set(ops.parameters(o.op))
            operands(at, found, dims, space_of(cx, [n["op"] for n in outer]), ragged, live, chosen)
        if i in closes:
            dims = before[closes[i]["at"]]
        elif o.op == "encode":
            before[i] = dims
            dims = layout(at, o.op, args, dims, "entry", kind)
        else:
            out = layout(at, o.op, args, dims, "entry", kind)
            changed = o.op if out != dims and changed is None else changed
            dims = out
    unclosed = [n for n, _ in visited(tree) if n["op"].op == "encode" and n["closed"] is None]
    if any(n["at"] < last for n in unclosed):
        j = max(n["at"] for n in unclosed if n["at"] < last)
        ref = arguments(spec.ops[j])["featurizer"]
        fail(f"{path}.ops[{last}]", f"the entry writes back here with the encode of {ref!r} (ops[{j}]) still open; "
             f"close it with {{'op': 'decode', 'featurizer': {ref!r}}} after its edits")
    record = {**spec.model_dump(exclude_defaults=True), "site": site.model_dump(exclude_defaults=True)}
    return {**record, "dims": dims, **({"space": space_of(cx, [n["op"] for n in unclosed])} if unclosed else {})}


def narrowed_readout(what: str) -> str:
    return (f"readout projects the whole residual stream through the model's final norm and head, and {what} keeps "
            "only some of hidden_size, which the norm cannot take; read the site whole, readout, and narrow what "
            "the logits give")


def space_of(cx: Compiling, encodes: list[Op]) -> list[dict]:
    """The coordinates a value is in: the encodes open around it, outermost first, each its featurizer, by the
    name it resolves to here (inside a fit's body `rot` and `<fit>.rot` are the one being trained), and the
    `select` it keeps; none for the site's own layout."""
    return [{k: cx.aliases.get(v, v) if k == "featurizer" else v for k, v in arguments(o).items()
             if k in ("featurizer", "select")} for o in encodes]


def said(space: list[dict]) -> str:
    """A value's coordinates as a message names them."""
    if not space:
        return "the site's own layout"
    return "the coordinates of " + " then ".join(repr(s["featurizer"]) + (f" (select {s['select']})" if "select" in s
                                                                       else "") for s in space)


def called(cx: Compiling, plan: Plan, path: str, name: str, args: dict) -> str | None:
    """The kind of the featurizer op `name`'s `Featurizer` argument names, checked to exist and to serve the op
    (`encode`, `decode`, `apply`); and the op's further arguments bound to the function its `options` choose for
    them, a `Value` among them a name. None when the op takes no featurizer."""
    ref = ops.featurizer_of(name, args)
    if ref is None:
        return None
    featurizer_ref(cx, ref, f"{path}.featurizer")
    kind = kind_of(plan, ref)
    serves = featurizers.KINDS[kind].ops()
    if name not in serves:
        fail(f"{path}.featurizer", f"{ref!r} is a {kind!r} featurizer, which has no {name}: it serves "
             f"{', '.join(serves)}" + ("; it has no way back into the value" if name == "decode" else ""))
    own = {k: v for k, v in args.items() if k not in ops.parameters(name)}
    try:
        fn = ops.options(name, args, kind)
        if fn is not None:
            article = "an" if kind[0] in "aeiou" or kind == "sae" else "a"        # "an sae", said letter by letter
            registry.check(fn, own, tuple(inspect.signature(fn).parameters)[:2], f"{article} {kind}'s `{fn.__name__}`")
    except ValueError as e:
        fail(path, f"featurizer {ref!r}: {e}")
    for key, p in ops.parameters(name, args, kind).items():
        if key in own and p.annotation == ops.Value and not isinstance(own[key], str):
            fail(f"{path}.{key}", f"{own[key]!r} is not a name; {key!r} names a value read at the positions the op "
                 "acts on")
    return kind


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


def references(cx: Compiling, plan: Plan, path: str, name: str, args: dict, classes: bool = False) -> dict[str, dict]:
    """Op `name`'s arguments that name a value or a column, each resolved to `{"dims", "data", "kind", "name"}`
    (a column is laid out `["rows"]`): a column read as text, or scored against the vocabulary, holds text on
    every row; one scored against a probe's `classes` holds its labels, whatever they are. A featurizer
    argument is checked to exist."""
    found = {}
    for key, p in parameters(plan, name, args).items():
        value, at = args.get(key), f"{path}.{key}"
        if p.annotation == ops.Value and isinstance(value, str):
            found[key] = {**value_ref(cx, value, at), "name": value}
        elif p.annotation in (ops.Tokens, ops.Text) and value is not None:
            ref = data.column_of(value)
            dataset = column_ref(cx, ref[0] if ref else value, at, None) if classes and p.annotation == ops.Tokens \
                else text_ref(cx, value, at, None)
            found[key] = {"dims": ["rows"], "data": dataset, "kind": "column", "name": ref[0] if ref else value}
        elif p.annotation == ops.Featurizer and value is not None:
            featurizer_ref(cx, value, at)
    return found


def aligned(path: str, of: tuple[str | None, str], found: dict) -> str | None:
    """The one dataset an op's rows align to: that of the value it acts on (`of`: its dataset, and how to name
    it), of its operands with a rows axis, and of its columns."""
    datasets = {of[0]: of[1]} if of[0] else {}
    for info in found.values():
        if info["data"] is not None and {"rows", "entries"} & set(info["dims"]):
            datasets.setdefault(info["data"], repr(info["name"]))
    if len(datasets) > 1:
        (a, x), (b, y) = list(datasets.items())[:2]
        fail(path, f"{x} has a row per example of {a!r} and {y} of {b!r}; an op takes one dataset's rows")
    return next(iter(datasets), None)


def paired(path: str, name: str, dims: list[str], width: int | None, found: dict) -> None:
    """A metric's operand (`kl`'s `against`, `cosine`'s `with`) is laid out as the value it scores, as many
    positions wide (`width`) when it has positions, or is one vector of its features, the same for every
    example; a ragged one's entries line up with nothing."""
    if ops.DECLARED[name]["group"] != "metrics":
        return
    one = [a for a in dims if a not in address.EXAMPLE_AXES]
    for key, info in found.items():
        if info["kind"] != "value":
            continue
        if "entries" in info["dims"]:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {name!r} scores a value laid out "
                 f"{dims}, and a ragged value's entries are its rows' positions, which pair with no other value's; "
                 "read both at a static index (pos -1, or {'index': '-3:'}), which makes them rectangular")
        if info["dims"] not in (dims, one):
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {name!r} scores a value laid out "
                 f"{dims}, so its {key!r} is laid out so too, or is one vector {one}")
        if "pos" in dims and info["dims"] == dims and info.get("width") != width:
            fail(f"{path}.{key}", f"{info['name']!r} holds {info.get('width')} position(s) per row and the value "
                 f"{name!r} scores holds {width}; position i scores against position i, so both are read at as "
                 "many (a pos axis broadcasts no more than a layers axis does)")


def operands(path: str, found: dict, own: list[str], space: list[dict], ragged: bool, live: dict, chosen: set) -> None:
    """Inside a write an operand stands for the value it meets, laid out `own` in the coordinates `space` (the
    encodes open around the op): in the same coordinates, encoded by the same featurizers with the same
    `select`, and that layout or one vector of its features; only the vector when the selection is ragged,
    unless the operand is an earlier entry of the same forward, which is gathered as the selection is. Such an
    entry is taken off the model as the forward reaches it, so its value is what its edits left (no op after its
    last edit), at one layer. An argument of the function an op's `options` choose (`chosen`: a lora's `with`)
    is laid out on the selection's rows and positions, whatever its features, which that function takes as it
    says."""
    lead = [a for a in own if a in address.EXAMPLE_AXES]
    for key, info in found.items():
        if info["kind"] != "value":
            continue
        if info["name"] in live:
            record, spec = live[info["name"]]
            if gathers(spec):
                fail(f"{path}.{key}", f"{info['name']!r} runs {gathers(spec)[0].op!r} once it is off the model, "
                     "after this forward; an operand from the same forward is an entry with no op after its edits")
            if address.stacks(record["site"].get("layers")):
                fail(f"{path}.{key}", f"{info['name']!r} is read at every layer; an operand from the write's own "
                     "forward is one layer's read")
        whole = not ragged or info["name"] in live
        if key in chosen:
            if [a for a in info["dims"] if a in address.EXAMPLE_AXES] != lead or not whole:
                fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; {key!r} is "
                     f"laid out on this write's {lead}" + ("" if whole else ", and a ragged write's is an entry of "
                                                           "its own forward"))
            continue
        if info.get("space", []) != space:
            fail(f"{path}.{key}", f"{info['name']!r} is in {said(info.get('space', []))}; this "
                 f"{'bracket' if space else 'write'} is in {said(space)}, so an operand is encoded as it is")
        allowed = [[a for a in own if a not in address.EXAMPLE_AXES]] + ([own] if whole else [])
        if info["dims"] == own and not whole:
            fail(f"{path}.{key}", f"{info['name']!r} and this write are both ragged, {own}; what differs is that its "
                 "entries are another forward's positions, which need not pair with this one's: read both at a "
                 "static index (pos -1, {'index': '-3:'}), or reduce the operand to one vector")
        if "layers" in info["dims"] and "layers" not in own:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}: it keeps a layers axis, and this "
                 "write lands at one layer, so its operand has none; read the operand at the write's own layer")
        if info["dims"] not in allowed:
            fail(f"{path}.{key}", f"{info['name']!r} is laid out {info['dims']}; this write's value is {own} here, "
                 f"so an operand is {' or '.join(map(str, allowed))}")


def layout(path: str, name: str, args: dict, dims: list[str], where: str, kind: str | None = None) -> list[str]:
    """The layout of op `name`'s result on a value laid out `dims` (for an op with a `layout`, for its
    featurizer's `kind`). Inside an entry (`where`) the op sees one batch, so it may not aggregate over rows;
    `original` is only an entry's."""
    if where != "entry" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(path, f"{name!r} restores the value a site had before its writes; it runs only inside an entry")
    try:
        out = ops.out_dims(name, dims, {**{k: p.default for k, p in ops.parameters(name).items()}, **args}, kind)
    except ValueError as e:
        fail(path, str(e))
    if where != "step" and {"rows", "entries"} & (set(dims) - set(out)):
        fail(path, f"{name!r} aggregates over rows, and inside an entry it sees one batch; "
             "make it a step after the forward")
    return out


def fit(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    """A fit's body runs over minibatches of its datasets' rows, the same rows of each: so its datasets have
    as many rows as each other (under each of `eval`'s substitutions too), and a value with a row per example it uses
    is one the body makes. It trains featurizers that have something to train, and holds no fit."""
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
        within.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
    before = len(cx.refs)
    body = {}
    for n, s in spec.steps.items():
        if s.kind == "fit":
            fail(f"{path}.steps.{n}", "a fit's body holds no fit")
        if n in plan.data or n in plan.featurizers or "." in n:
            fail(f"{path}.steps.{n}", f"{n!r} is also a dataset or a featurizer, or has a dot; a body's names are "
                 "a root's")
        body[n] = step(within, f"{path}.steps.{n}", n, s, plan)
    inner = within.names
    for ref in cx.refs[before:]:
        if inner.get(ref) is cx.names.get(ref) and {"rows", "entries"} & set(cx.names[ref]["dims"]):
            fail(f"{path}.steps", f"the body uses {ref!r}, which has a row per example of the whole dataset; the "
                 "body runs over minibatches of its rows, so such a value is one it makes")
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
    for label, side in {"": {}, **spec.eval}.items():
        counts = {d: len(cx.rows[side.get(d, d)]) for d in {s.get("dataset") for s in body.values()} - {None}}
        if len(set(counts.values())) > 1:
            fail(f"{path}.steps", f"the body's datasets have {counts} rows" + (f" under eval {label!r}" if label else "")
                 + "; it runs over minibatches of the same rows of each, paired row by row, so they have as many rows: "
                 "repeat the smaller dataset's rows (or cut the larger's) to the same count, or fit on one")
    for f in spec.train:
        cx.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
        for scalar in featurizers.KINDS[kind_of(plan, f)].scalars:
            cx.names[f"{name}.history.{f}.{scalar}"] = {"dims": ["epochs"], "data": None, "kind": "value"}
    for n, s in body.items():
        if s["kind"] == "op":
            cx.names[f"{name}.history.{n}"] = {"dims": ["epochs"], "data": None, "kind": "value"}
        members = ([n] if s["kind"] == "op" else []) + [f"{n}.{k}" for k in s.get("interventions", {})]
        for label, swap in spec.eval.items():
            for member in members:
                info = inner[member]
                cx.names[f"{name}.{label}.{member}"] = {**info, "data": swap.get(info["data"], info["data"])}
    return {**spec.model_dump(exclude_defaults=True), "steps": body}


# -- references --------------------------------------------------------------------------------------


def value_ref(cx: Compiling, value, path: str) -> dict:
    """A named value produced before this point."""
    info = cx.names.get(value) if isinstance(value, str) else None
    if info is not None and info["kind"] == "featurizer":
        fail(path, f"{value!r} is a featurizer, not a value; an op reaches it ({{'op': 'encode', 'featurizer': "
             f"{value!r}}} gives a value's coordinates, 'apply' its version of the value), and the value that entry "
             "binds is an operand")
    if info is None or info["kind"] != "value":
        known = sorted(k for k, v in cx.names.items() if v["kind"] == "value")
        fail(path, f"{value!r} names no value produced before this step; known: {', '.join(known) or 'none'}")
    cx.refs.append(value)
    cx.used.add(value)
    return info


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


def decided(path: str, pos: dict, most: int | None) -> None:
    """A generation's write beyond the prefill lands at each call it selects as that call runs, so what it
    selects must be known by then. Without a scope its calls must be known before the run, up to the step's
    `max_new_tokens` (`most`) when it gives one (`positions.reach`). With a text scope it is found online,
    call by call, in the inputs fed so far, and lands at the call fed the first occurrence's last token: so
    its frame is decode calls only, and its index -1, since the occurrence's earlier tokens were fed before
    the text was there to find; and which occurrences it means count from the first."""
    frame, index = pos["frame"], pos["index"]
    if pos["scope"] is not None:
        if positions.reach({**pos, "index": ":"})[0]:
            fail(f"{path}.frame", f"frame {frame!r}: a scoped write beyond the prefill is found call by call in the "
                 "generated inputs, so its frame names decode calls only, not counted from the row's last call "
                 "(a call, picks, or a slice from 1 such as '1:')")
        if index != -1:
            fail(f"{path}.index", f"index {index!r}: a scoped write lands at the call fed its text's last token, the "
                 "first call that can know the text is there; earlier tokens were already fed, so its index is -1")
        which = pos["scope"].get("occurrence", 0)
        if positions.from_end(which):
            fail(f"{path}.scope.occurrence", f"occurrence {which!r} counts from the last occurrence, which is known "
                 "only when the row ends; a write lands as its call runs, so it counts from the first")
        return
    if positions.reach(pos, most)[2]:
        return
    holds = not positions.from_end(frame) and 0 in positions.select(frame, 1)
    if holds and positions.covers(frame) and positions.from_end(index):
        fail(path, f"frame {frame!r} with index {index!r} counts back from the row's last position, known only when "
             "the row stops, and a write lands as each call runs; for the prompt's last positions and every decode "
             f"call, write two entries: {{'frame': 0, 'index': {index!r}}} and {{'frame': '1:'}}")
    if holds:
        fail(path, f"frame {frame!r} with index {index!r}: over a frame that holds the prompt a write's index is "
             "':', since any other counts on through the prompt, whose length differs by row, into the decode "
             "calls, and a write lands as each call runs; write the prompt's positions you mean as an entry at "
             "frame 0 and the decode calls you mean as another at their own frame ('1:' is every one)")
    if most is None and positions.reach(pos, positions.enough(pos))[2]:
        fail(path, f"frame {frame!r} with index {index!r} names calls up to the row's last; a write lands at calls "
             "listed before the run, so the step gives max_new_tokens, which bounds them")
    fail(path, f"frame {frame!r} with index {index!r} names calls that depend on how far each row gets; a write "
         "lands as its call runs, so its frame is not counted from the row's last call, and its index "
         "names the same calls on a row that stops early as on one that runs on: ':' over the prompt, and not "
         "counted from the end over several decode calls; or it takes a text scope")


def text_ref(cx: Compiling, value, path: str, dataset: str | None) -> str:
    """A column reference (`"<dataset>.<column path>"`, or `{"column", "strip"?, "lower"?}`) naming text on
    every row, of `dataset` (the step's) when one is given: a string, or a non-empty list of them (a set).
    Returns its dataset."""
    if isinstance(value, dict):
        try:
            Column(**value)
        except ValidationError as e:
            fail(path, "; ".join(err["msg"] if not err["loc"] else f"{err['loc'][0]}: {err['msg']}"
                                 for err in e.errors()))
    ref = data.column_of(value)
    if ref is None:
        fail(path, f"{value!r} is not a column: '<dataset>.<column>' or {{'column': '<dataset>.<column>', "
                   "'strip'?: bool, 'lower'?: bool}")
    name, _, column = ref[0].partition(".")
    column_ref(cx, f"{name}.{column.split('[')[0].split('.')[0]}", path, dataset)
    for i, row in enumerate(cx.rows[name]):
        try:
            found = data.lookup(row, column)
        except (KeyError, IndexError, TypeError):
            fail(path, f"{ref[0]!r} is not on row {i} of {name!r}")
        if not (isinstance(found, str) or isinstance(found, list) and found and all(isinstance(f, str) for f in found)):
            fail(path, f"{ref[0]!r} is {type(found).__name__} on row {i} of {name!r}; a column read as text is a "
                       "string, or a list of them (a set)")
    return name


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
        try:
            featurizers.labels(cx.rows, fields["classes"])
        except ValueError as e:
            fail(f"{path}.classes", str(e))
    missing = [k for k in featurizers.KINDS[f.kind].drawn if fields.get(k) is None]
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
        tensors, header = featurizers.bundles.load(cx.root, f.kind, fields, plan.model,
                                                   featurizers.labels(cx.rows, fields["classes"]) if "classes" in fields
                                                   else None)
    except ValueError as e:
        fail(f"{path}.file_path", str(e))
    return {**record, "bundle": tensors, "header": header}
