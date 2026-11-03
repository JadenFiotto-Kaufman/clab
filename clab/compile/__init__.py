"""What a document means, decided before any forward in two passes: here, with the document alone, a plan
resolved for running, every name checked, every sweep expanded into points, every value's record carrying its
labelled layout as far as the document decides it; and in `preflight`, with the model in hand, everything the
model decides (tokens, positions per row, answers' ids, the layers and indices a slice or a negative selector
picks among its sizes), completed into the same records, and what the run needs beside them (`state.Findings`).

The package holds the two drivers and the concepts they resolve: `sweeps` and `data` are the format's own; `answers`
(a column scored against a vocabulary), `positions` (`pos` resolved per row), `arguments` (an op's arguments
against the plan's records, and the one rule an operand keeps), `interventions` (an entry, its brackets and operands),
`schedule` (the forward as a compiled schedule: which points one forward serves, and what it does) and `uses` (a
featurizer's uses in a plan) are here, each with its checks at compile and what it resolves to in the run. The sweep
axes each value depends on (`Labelled.axes`, the dependency rule `sweeps.find_sweeps` states) are computed here as
the records are built (`compile_forward`, `compile_op`).

Compile knows the document and the files; the run knows the model. Anything decided from the document, the
data files and bundle headers is checked here; anything that needs the tokenizer or a size of the model is
checked by `preflight.preflight`, inside the run. The records' own shape (fields, types, the pos grammar, a
site's narrowing) is the models' to check, when the plan is built. How names resolve is `compile_step`'s docstring; the
compiled records themselves are the symbol table (`plan.record_of`), each name resolved at the consumer's point.
"""

import itertools
import os
import re
from dataclasses import replace
from typing import Any, Callable, cast

from clab import featurizers, ops
from clab.compile.answers import check_column_ref
from clab.compile.arguments import (featurizer_kind, kept_labels, layout, mismatch, narrowed_readout_refusal,
                                    references, resolver, space_of, value_record)
from clab.compile.interventions import bundles_of, compile_entry, entries_with_paths
from clab.compile.schedule import share_forwards
from clab.compile.state import Compiling
from clab.compile.uses import check_bundle_sites, check_gate_axes
from clab.featurizers import KINDS, bundles, labels
from clab.format import data, registry
from clab.format.sweeps import agreeing_point, coordinate_spelling, find_sweeps
from clab.plan import (
    Data,
    Entry,
    Featurizer,
    Fit,
    Forward,
    Generate,
    Intervention,
    Labelled,
    Load,
    OpStep,
    Plan,
    PlanError,
    Site,
    adapts,
    arguments,
    corpus_root,
    documented,
    entries,
    entry_of,
    fail,
    gathers,
    gradient_of,
    kind_of,
    made_by,
    open_encodes,
    points,
    produced,
    record_of,
    runnable,
    writes,
)


def compile(plan: Plan) -> Plan:
    """The plan compiled: a new `Plan`, `compiled=True`."""
    if plan.compiled:
        return plan
    root = corpus_root(plan)
    doc = plan.model_dump(exclude_defaults=True, exclude={"run", "document"})
    rows = {name: make_dataset_rows(root, name, d) for name, d in plan.data.items()}
    sweeps = find_sweeps(doc)
    values = {label: get(doc, path)["sweep"] for path, label in sweeps.items()}
    cx = Compiling(doc=doc, root=root, rows=rows, sweeps=sweeps, values=values, featurizers={}, steps={}, axes_of={},
                   used=set())
    for name in plan.featurizers:
        cx.featurizers[name] = expand(cx, name, ("featurizers", name),
                                      lambda point, at, name=name: compile_featurizer(cx, point, name, at))
    for name in plan.steps:
        cx.steps[name] = expand(cx, name, ("steps", name), lambda point, at, name=name: compile_step(
            cx, f"steps.{name}", name, point.steps[name], point, at))
    saveable = sorted(produced(cx.steps, cx.featurizers)
                      + [f"{n}.{f}" for n, e in cx.steps.items() for s in points(e)[:1] if s.kind == "fit" for f in s.train])
    for i, name in enumerate(plan.saves):
        if name in plan.saves[:i]:
            fail(f"saves[{i}]", f"{name!r} is saved twice")
        if name not in saveable:
            fail(f"saves[{i}]", f"{name!r} names no step's value; a save is a step or a step's member: "
                 f"{', '.join(saveable)}")
        if name in ("manifest", "tokens"):
            fail(f"saves[{i}]", f"{name!r} is the name of a run directory's own {name}.json, which a save beside it "
                 "would be written over or write over; give the step another name")
    for name, d in plan.data.items():
        if d.template is not None and name not in cx.chats:
            fail(f"data.{name}.template", f"no step reads a chat column of {name!r}, and a template renders a chat "
                 "(a column of {'role', 'content'} messages) only")
    mark_gathered(cx.steps, cx.used | set(plan.saves))
    done = Plan(**{**doc, "featurizers": cx.featurizers, "steps": cx.steps, "sites": {}, "interventions": {},
                   "compiled": True, "document": plan.document,
                   "sweeps": {label: {"values": values[label]} for label in values},
                   "data": {n: {**doc["data"][n], **({} if plan.data[n].hf is not None else {"rows": rows[n]}),
                                "sha256": data.digest(rows[n]), "n_rows": len(rows[n])} for n in rows}})
    check_bundle_sites(done)
    check_gate_axes(done)
    share_forwards(done)
    return done


def make_dataset_rows(root: str, name: str, d: Data) -> list[dict]:
    """Dataset `name`'s rows as materialized, which its `sha256` and `n_rows` describe: its file's (under `root`),
    its inline rows or its hub split's (`data.hub`), then a file's `split`, `shuffle`, `skip`, `take` and
    `columns`, in that order (`Data`). The one way a dataset's rows are made: compile's, and `dataset_rows`'s where
    they are not in the plan."""
    try:
        rows = data.hub(d) if d.hf is not None else d.rows if d.path is None else data.rows(root, d.path, d.split)
    except ValueError as e:
        fail(f"data.{name}.{'path' if d.hf is None else 'hf'}", str(e))
    held = f" in split {d.split!r}" if d.split else ""
    if not rows:
        fail(f"data.{name}", f"the dataset has no rows{held}")
    rows = data.shuffled(rows) if d.shuffle else rows
    if d.skip is not None and d.skip >= len(rows):
        fail(f"data.{name}.skip", f"skip {d.skip}, and the dataset has {len(rows)} rows{held}; skip leaves at least "
             "one")
    rows = rows[d.skip:]
    if d.take is not None and d.take > len(rows):
        fail(f"data.{name}.take", f"take {d.take}, and the dataset has {len(rows)} rows{held}"
             + (f" after skip {d.skip}" if d.skip else "") + "; take keeps at most as many as there are")
    rows = rows[:d.take]
    moved = {c for c in (d.columns or {}).values() if isinstance(c, str)}
    for new, column in (d.columns or {}).items():
        where, source = f"data.{name}.columns.{new}", column if isinstance(column, str) else column.source
        lacking = next((i for i, r in enumerate(rows) if source not in r), None)
        if lacking is not None:
            fail(where + ("" if isinstance(column, str) else ".from"), f"{source!r} is not a column of row "
                 f"{lacking}; its columns: {', '.join(rows[lacking])}")
        if new not in moved and any(new in r for r in rows):
            fail(where, f"{new!r} is a column the rows keep, which this "
                 f"{'rename' if isinstance(column, str) else 'derived column'} would overwrite; name the new column "
                 "otherwise")
        if isinstance(column, str):
            continue
        for i, r in enumerate(rows):
            if not isinstance(r[source], str):
                fail(f"{where}.from", f"row {i}: {source!r} is {type(r[source]).__name__}, {r[source]!r}; a derived "
                     "column is cut from text")
            if data.cut(r[source], column) is None:
                after = column.after.text if column.after is not None else None
                side = "after" if after is not None and after not in r[source] else "before"
                fail(f"{where}.{side}", f"row {i}: {getattr(column, side).text!r} is not in its {source!r}"
                     + (f" after the first {after!r}" if side == "before" and after is not None else "")
                     + f", {r[source]!r}")
    return data.renamed(rows, d.columns) if d.columns else rows


def dataset_rows(plan: Plan, name: str) -> list[dict[str, Any]]:
    """A dataset's rows: the plan's, or, where they are not in it (a hub dataset's, always; a file's, after
    `Plan.load`), materialized again by `make_dataset_rows` and checked against the digest compile recorded."""
    d = plan.data[name]
    if d.rows is not None:
        return d.rows
    rows = make_dataset_rows(corpus_root(plan), name, d)
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


def expand(cx: Compiling, name: str, where: tuple, build: Callable[[Plan, dict[str, int]], Any]) -> Any:
    """An entry compiled at every point of its axes: those in what it uses (its own spec, and the sites,
    interventions and featurizers it names: `footprint`), those of every value it references, read off its
    text before it compiles (`referenced_names`), and a load's, those its saved value depends on, read off the run's
    manifest, so that each point is built knowing its coordinates and resolves every name at them; a refusal at one
    point of a sweep names the point after the document path (`index="-2:"`). A load declares each sweep its value
    depends on that the document does not, with the label and values the run swept it over (`cx.values`, which the
    compiled plan's `sweeps` are), and is refused by name where the document sweeps that label over other values."""
    spec = get(cx.doc, where)
    inherited = {a for ref in referenced_names(cx, spec) for a in cx.axes_of.get(ref.partition(".")[0], [])}
    if where[0] == "steps" and spec["kind"] == "load":          # a saved value's own axes, which its run recorded
        src, d = saved_value_record(cx, f"steps.{name}", spec["from"], spec["value"])
        for label in d["axes"]:
            theirs = src["plan"].sweeps[label].values
            if cx.values.setdefault(label, theirs) != theirs:
                fail(f"steps.{name}.value", f"{spec['value']!r} is swept over {label!r}, {theirs} in {spec['from']}, "
                     f"and this document sweeps {label!r} over {cx.values[label]}; a load declares the sweeps its "
                     "value depends on, and a sweep of this document under the same label has the same values")
        inherited |= set(d["axes"])
    labels = in_axis_order(cx, labels_under(cx, *footprint(cx, where)) | inherited)
    cx.axes_of[name] = labels
    built = []
    for index in itertools.product(*(range(len(cx.values[label])) for label in labels)):
        at = dict(zip(labels, index))
        document = document_at_point(cx, at)
        try:
            built.append(build(document, at))
        except PlanError as e:              # a refusal at one point of a sweep names the point
            if not at:
                raise
            path, _, why = str(e).partition(": ")
            raise PlanError(f"{path} at {coordinate_spelling(cx.values, at)}: {why}") from None
    return built if labels else built[0]


def document_at_point(cx: Compiling, at: dict[str, int]) -> Plan:
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


def path_tuple(path: str) -> tuple:
    """A document path as text (`steps.x.interventions[0].k`) as the tuple `get` walks (`("steps", "x",
    "interventions", 0, "k")`): the names between dots, a bracketed index an int."""
    out: list = []
    for part in path.split("."):
        head, *indices = part.replace("]", "").split("[")
        out.append(head)
        out += [int(i) for i in indices]
    return tuple(out)


def labels_under(cx: Compiling, *paths: tuple) -> set[str]:
    """The labels of the sweeps standing under any of the document `paths`."""
    return {label for path, label in cx.sweeps.items() if any(path[:len(p)] == p for p in paths)}


def in_axis_order(cx: Compiling, labels: set[str]) -> list[str]:
    """Axis labels in axis order."""
    return [label for label in cx.values if label in labels]


def named_axes(cx: Compiling, ref: str, at: dict[str, int]) -> set[str]:
    """The axes a named value or featurizer varies on, at the consumer's point `at`: a value's as its record says;
    a declared featurizer's its own; a featurizer a fit trains (`<fit>.<f>`) the fit's, as its record says."""
    thing = ref.partition(".")[0]
    if thing in cx.featurizers or thing not in cx.steps:
        return set(cx.axes_of.get(thing, []))
    return set(record_of(cx.steps, ref, at).axes)


def operand_axes(cx: Compiling, listed: list, at: dict[str, int], live: dict[str, set[str]]) -> set[str]:
    """The axes of every value or featurizer the ops `listed` name as an operand, at `at`: an entry of the same
    forward (`live`, by its dotted name) those recorded for it there."""
    axes: set[str] = set()
    for o in listed:
        params = ops.parameters(o.op)
        for key, value in arguments(o).items():
            if key in params and params[key].annotation in (ops.Value, ops.Featurizer) and isinstance(value, str):
                axes |= live[value] if value in live else named_axes(cx, value, at)
    return axes


def footprint(cx: Compiling, where: tuple) -> list[tuple]:
    """The document paths an entry reads: its own, and the sites, interventions and featurizers it names."""
    paths, doc = [where], cx.doc

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                named = v if isinstance(v, list) else [v]
                if k == "site" and isinstance(v, str):
                    paths.append(("sites", v))
                if k == "interventions":
                    for n in named:
                        if isinstance(n, str):
                            paths.append(("interventions", n))
                            walk(doc.get("interventions", {}).get(n))
                if k in ("featurizer", "train"):
                    paths.extend(("featurizers", n) for n in named
                                 if isinstance(n, str) and n in doc.get("featurizers", {}))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(get(doc, where))
    return paths


def referenced_names(cx: Compiling, spec: Any) -> set[str]:
    """The names a document entry references, read off its text before it compiles: an op step's `of` (not a
    `grad`'s, an entry of its own forward), an op's arguments that name a value or a featurizer (by its signature),
    a fit's `train`, and those of the bundles a step's `interventions` pull in; a sweep standing for one, each of
    its values."""
    referenced: set[str] = set()

    def names(value: Any) -> list[str]:
        listed = value["sweep"] if isinstance(value, dict) and "sweep" in value else \
            value if isinstance(value, list) else [value]
        return [n for n in listed if isinstance(n, str)]

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            op = value.get("op")
            params = ops.parameters(op) if isinstance(op, str) and op in ops.OPS else {}
            for k, v in value.items():
                if k == "train" or k == "of" and op != "grad" or \
                        k in params and params[k].annotation in (ops.Value, ops.Featurizer):
                    referenced.update(names(v))
                if k == "interventions":
                    for n in names(v):
                        walk(cx.doc.get("interventions", {}).get(n))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(spec)
    return referenced


# -- steps -------------------------------------------------------------------------------------------


def compile_step(cx: Compiling, path: str, name: str, spec: Entry, plan: Plan, at: dict[str, int]) -> Any:
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
    derived = set(type(spec).model_fields) - set(documented(type(spec)))     # compile writes them over what was set
    for key in sorted(spec.model_fields_set & derived):
        fail(f"{path}.{key}", f"{key!r} is a derived field, which compile writes; a step sets none")
    return {"forward": compile_forward, "generate": compile_forward, "op": compile_op, "fit": compile_fit,
            "load": compile_load}[spec.kind](cx, path, name, spec, plan, at)


def compile_forward(cx: Compiling, path: str, name: str, spec: Forward | Generate, plan: Plan,
                    at: dict[str, int]) -> Forward | Generate:
    """Compiles a forward or generate step."""
    joined = not isinstance(spec.input, str)
    dataset = None
    for i, part in enumerate(spec.input if joined else [spec.input]):
        if not isinstance(part, str):
            continue
        here = f"{path}.input" + (f"[{i}]" if joined else "")
        name_, column, index = data.parse_input(part)
        if name_ not in cx.rows:
            fail(here, f"no dataset {name_!r}; datasets: {', '.join(cx.rows)}")
        dataset = dataset or name_
        check_column_ref(cx, f"{name_}.{column}", here, dataset)
        rows = cx.rows[dataset]
        if index is not None and any(not isinstance(r[column], list) or index >= len(r[column]) for r in rows):
            fail(here, f"column {column!r} has no entry {index} on every row")
        if not joined and not isinstance(data.field(rows[0], part), str):
            cx.chats.add(dataset)
        if joined:
            for r, row in enumerate(rows):
                if not isinstance(data.field(row, part), str):
                    fail(here, f"{part!r} is not text on row {r} of {dataset!r}; a list input joins text")
    assert dataset is not None
    for key, value in (spec.model_extra or {}).items() if spec.kind == "generate" else ():
        check_generate_option(f"{path}.{key}", key, value)
    given = entries_with_paths(plan, path, name, spec)
    keys = [k for _, k, _ in given]
    for k in keys:
        if keys.count(k) > 1:
            fail(f"{path}.interventions", f"two entries named {k!r}; a step binds each name once")
    own = {f"{name}.{k}" for k in keys}
    records: dict[str, Intervention] = {}
    live: dict[str, Intervention] = {}
    for p, k, e in given:
        try:
            records[k] = compile_entry(cx, p, e, plan, spec, dataset, live, own, f"{name}.{k}", at)
        except PlanError as refused:        # a bundle's entry is refused where it is pulled in, which the path names
            if not p.startswith("interventions."):
                raise
            where, _, why = str(refused).partition(": ")
            raise PlanError(f"{where} in {path}: {why}") from None
        # A forward's entries may name the ones before them, which the engine makes as the forward reaches them;
        # a generation's land call by call, so they may not yet.
        if spec.kind == "forward":
            live[f"{name}.{k}"] = records[k]
    for p, k, e in given:               # a gradient's `of`, which may be listed after it
        of = gradient_of(records[k])
        if of is None:
            continue
        where = f"{p}.ops[0].of" + (f" in {path}" if p.startswith("interventions.") else "")
        if of not in records:
            fail(where, f"{of!r} names no entry of this forward; grad's of is an entry of its own forward, by its "
                 f"bare name: {', '.join(keys)}")
        if gradient_of(records[of]) is not None:
            fail(where, f"{of!r} is a grad entry, whose value is a gradient the forward never computes; of names an "
                 "entry whose value the forward makes")
        held = next((o.op for o in records[of].ops if o.op in ops.transforms.NO_GRADIENT), None)
        if held is not None:
            fail(where, f"{of!r} runs {held!r}, which gives {ops.transforms.NO_GRADIENT[held]}, and no gradient passes "
                 "through it; of names an entry whose value is a differentiable number: a read, or a metric such "
                 "as logit_diff")
    # The dependency rule (`sweeps.find_sweeps`): the forward's axes are the step's options' and each write's
    # before the forward (`pre`: its site, its positions and its ops up to its last edit, with their operands); an
    # entry's are its own sweeps, its operands' and the forward's.
    pre: dict[str, set[str]] = {}
    for p, k, e in given:
        where, record = path_tuple(p), records[k]
        last = len(record.ops) - len(gathers(record))
        after = labels_under(cx, *(where + ("ops", i) for i in range(last, len(record.ops))))
        site = labels_under(cx, ("sites", e.site)) if isinstance(e.site, str) else set()
        pre[f"{name}.{k}"] = (labels_under(cx, where) - after) | site | operand_axes(cx, record.ops[:last], at, pre)
    here = path_tuple(path)
    forward_axes = labels_under(cx, here) - labels_under(cx, here + ("interventions",))
    for p, k, e in given:
        if writes(records[k]):
            forward_axes |= pre[f"{name}.{k}"]
    axes: dict[str, set[str]] = {}
    for p, k, e in given:
        site = labels_under(cx, ("sites", e.site)) if isinstance(e.site, str) else set()
        axes[f"{name}.{k}"] = labels_under(cx, path_tuple(p)) | site | operand_axes(cx, records[k].ops, at, axes) | \
            forward_axes
        records[k].axes = in_axis_order(cx, axes[f"{name}.{k}"])
    for p, k, e in given:               # a gradient depends on what it differentiates too
        if (of := gradient_of(records[k])) is not None:
            axes[f"{name}.{k}"] |= axes[f"{name}.{of}"]
            records[k].axes = in_axis_order(cx, axes[f"{name}.{k}"])
    fields = dict(spec.model_dump(exclude_defaults=True, exclude={"interventions"}), interventions=records,
                  dataset=dataset, at=at, schedule={"axes": in_axis_order(cx, forward_axes)})
    if spec.kind == "generate":
        return Generate(**fields, axes=in_axis_order(cx, forward_axes))       # the ids are the forward's own value
    return Forward(**fields)


def check_generate_option(path: str, key: str, value: Any) -> None:
    """A generate step's option, refused by name at compile where no model is needed to decide it: one of
    `generate`'s own parameters (`generation_config`, `logits_processor`, ...), which take what a document cannot
    write (the type is named), where a step's options are a generation config's fields; and a field's value of the
    wrong type (`max_new_tokens: "3"`), which transformers' `GenerationConfig` refuses with a TypeError. What else
    it rejects is the engine's `generate`'s to say when the step runs."""
    import inspect

    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    own = registry.signature(getattr(GenerationMixin, "generate")).parameters      # `self` is a protocol to pyright
    if key in own and own[key].kind is not inspect.Parameter.VAR_KEYWORD:
        takes = re.sub(r"\b[\w.]+\.(\w+)", r"\1", str(own[key].annotation).replace("typing.Optional[ForwardRef('", "")
                       .replace("')]", "")).replace(" | None", "")
        fail(path, f"{key!r} is a parameter of generate itself, taking {takes}, not a field of a generation config; "
                   "a step's options are the config's fields (max_new_tokens, do_sample, ...)")
    try:
        GenerationConfig(**{key: value}).validate()
    except TypeError as e:
        fail(path, f"{value!r} is not a value generate takes: {e}")
    except ValueError:
        pass


def compile_op(cx: Compiling, path: str, name: str, spec: OpStep, plan: Plan, at: dict[str, int]) -> OpStep:
    """Compiles an op step."""
    if spec.op == "decode":
        fail(path, "decode closes an encode of its own entry, so it runs inside the entry that encodes, after the "
             "encode and the edits of its coordinates; as a step there is none to close (detokenize turns a "
             "generation's ids into text)")
    of, args = value_record(cx, spec.of, f"{path}.of", at), arguments(spec)
    text = isinstance(of, OpStep) and of.op == "detokenize" and not arguments(of).get("tokens")
    if spec.op == "contains" and not text:
        fail(f"{path}.of", f"contains reads text, a detokenize's value (without tokens); {spec.of!r} is not one")
    kind = featurizer_kind(cx, plan, path, spec.op, args)
    if "featurizer" in args:        # a read's own value, at its site and in its layout: an entry's, or a load's whose
        # record is that of the entry it carries (`source`), as a load of a saved entry's value is, and a load of such
        # a load; not an op's over one
        carried = next(iter(entries(of.source).values())) if isinstance(of, Load) and of.source is not None else None
        if not (isinstance(of, Intervention) or carried is not None
                and all(getattr(of, f) == getattr(carried, f) for f in Labelled.model_fields)):
            fail(f"{path}.of", f"{spec.of!r} is an op's value, which no one site made, and a featurizer's {spec.op} "
                 f"acts on a value at its site: {spec.op} a read (an entry's value), or {spec.op} inside the entry "
                 "that reads")
    assert of.dims is not None
    operands = references(spec.op, args, resolver(cx, path, "classes" in of.dims, at))
    for key, (written, theirs) in operands.items():
        why = mismatch(of, theirs, spec.op, key, written, repr(spec.of), False, False, adapts(plan, spec))
        if why:
            fail(f"{path}.{key}", why)
    if spec.op == "readout" and of.read is not None:        # the one stream check, an entry's readout's too
        site = cast(Site, entry_of(cx.steps, of.read, at)[1].site)
        if why := ops.transforms.unstreamed(site.component):
            fail(f"{path}.of", f"{spec.of!r} is read at {site.component}: {why}")
    if spec.op == "readout" and "hidden_size" in of.kept:
        fail(path, narrowed_readout_refusal(repr(spec.of)))
    dims = layout(path, spec.op, args, of.dims, "step", kind)
    # An encode's value is in its featurizer's coordinates, as an entry's is that leaves its encode open.
    space = of.space + space_of([spec]) if spec.op == "encode" else of.space if dims[-1:] == of.dims[-1:] else []
    layers, kept, width = kept_labels(of, dims)
    dataset = of.dataset if of.dataset is not None else \
        next((t.dataset for _, t in operands.values() if "rows" in (t.dims or []) and t.dataset is not None), None)
    axes = labels_under(cx, path_tuple(path)) | named_axes(cx, spec.of, at) | operand_axes(cx, [spec], at, {})
    return OpStep(**spec.model_dump(exclude_defaults=True), at=at, dataset=dataset, dims=dims, layers=layers,
                  read=of.read, kept=kept, width=width, space=space, axes=in_axis_order(cx, axes))


def compile_load(cx: Compiling, path: str, name: str, spec: Load, plan: Plan, at: dict[str, int]) -> Load:
    """A saved value at one point of its axes (those of its run's sweeps it depends on, which `expand` declared or
    found declared alike), with the record its run's manifest holds for it: refused unless that run ran this
    document's model and the value's dataset has the same rows here (`Load`). Its read, where it holds positions,
    comes with it as the forward that made it with that entry alone, named after this step (`<step>.<entry>`, the
    value's `read`). The value itself the compiled plan reads from the file, as it reads a bundle's
    (`plan.loaded_value`)."""
    src, d = saved_value_record(cx, path, spec.from_, spec.value)
    ran, here = src["plan"], f"{path}.value"
    theirs, ours = ran.model.model_dump(), plan.model.model_dump()
    differ = [k for k in dict.fromkeys([*theirs, *ours]) if theirs.get(k) != ours.get(k)]
    if differ:
        fail(f"{path}.from", f"{spec.from_} ran another model than this document's: " + ", ".join(
            f"{k} {theirs.get(k)!r} where this document's is {ours.get(k)!r}" for k in differ))
    there = {label: at[label] for label in d["axes"]}         # the point here is the point there: same labels, values
    record = record_of(ran.steps, spec.value, there)
    made = agreeing_point(runnable(ran.steps)[made_by(ran, spec.value)], there)      # an eval's member's: its eval step
    producer, entry = entry_of(ran.steps, record.read, there) if record.read is not None else (None, None)
    if made.kind == "fit" and spec.value.partition(".")[2] in made.train or \
            made.kind == "op" and ops.DECLARED[made.op]["bundle"]:
        fail(here, f"{spec.value!r} is a bundle, which a featurizer of its kind loads, with 'file_path': "
             f"{os.path.join(spec.from_, d['file'])!r}")
    if made.kind == "generate" or producer is not None and producer.kind == "generate":
        fail(here, f"{spec.value!r} is a generation's value or holds its positions, which count the tokens that run "
             "generated; a load brings a forward's values and what op steps and fits made of them")
    if record.space:
        fail(here, f"{spec.value!r} is in the coordinates of {', '.join(repr(s['featurizer']) for s in record.space)}, "
             f"a featurizer of {spec.from_}'s; load the value it encodes and encode it here")
    if record.dataset is not None:
        ds, theirs_data = record.dataset, ran.data.get(record.dataset)
        if ds not in cx.rows or theirs_data is None:
            fail(here, f"{spec.value!r} was made over the rows of dataset {ds!r}, which this document does not "
                 f"declare; declare it as {spec.from_} did")
        if len(cx.rows[ds]) != theirs_data.n_rows or data.digest(cx.rows[ds]) != theirs_data.sha256:
            fail(here, f"{spec.value!r} was made over the rows of dataset {ds!r}, and this document's {ds!r} is not "
                 f"the rows {spec.from_} read, in their order: " + (
                     f"{len(cx.rows[ds])} rows where it read {theirs_data.n_rows}" if len(cx.rows[ds]) !=
                     theirs_data.n_rows else "as many rows, whose digest differs"))
        if plan.data[ds].template != theirs_data.template:      # a chat's tokens, which the digest does not cover
            fail(here, f"{spec.value!r} was made over the rows of dataset {ds!r}, whose chats {spec.from_} rendered "
                 f"with template {theirs_data.template}, and this document's {ds!r} has {plan.data[ds].template}")
    read, carried, axes = None, None, in_axis_order(cx, set(d["axes"]))
    if record.read is not None and producer is not None and entry is not None:
        read = f"{name}.{record.read.rpartition('.')[2]}"
        if isinstance(producer.input, str) and not isinstance(data.field(cx.rows[str(record.dataset)][0],
                                                                         producer.input), str):
            cx.chats.add(str(record.dataset))       # its tokens render a chat, as the run's forward did
        carried = Forward(kind="forward", input=producer.input, dataset=record.dataset, interventions={
            read.partition(".")[2]: entry.model_copy(update={
                "read": read, "positions": {str(record.dataset): entry.positions[str(record.dataset)]}, "axes": axes})})
    if not os.path.exists(os.path.join(src["out"], d["file"])):       # its value the compiled plan reads
        fail(here, f"{os.path.join(spec.from_, d['file'])} is missing, which its manifest says holds {spec.value!r}")
    return Load(**spec.model_dump(exclude_defaults=True), at=at, dims=record.dims, dataset=record.dataset,
                layers=record.layers, read=read, width=record.width, kept=record.kept, axes=axes, source=carried)


def loaded_run(cx: Compiling, path: str, out: str) -> dict:
    """The run a load reads, its output directory `out` under the corpus root, read once per directory
    (`Compiling.sources`): the plan its manifest records, without its featurizers (a load needs its records, not the
    bundles they loaded), and each saved value's record by name (`described`). A directory with no manifest, or one
    this clab cannot read, is refused by name."""
    if out not in cx.sources:
        from clab.output import manifest          # the output directory's one reader, a later stage's
        where = os.path.join(cx.root, out)
        try:
            held, described = manifest(where)
        except FileNotFoundError:
            fail(f"{path}.from", f"no run at {out} under the corpus root (no manifest.json); run the document that "
                 f"saves the value there first, with --out {out}")
        except ValueError as e:                 # a manifest an older clab wrote, named as the document names it
            fail(f"{path}.from", str(e).replace(where, out))
        try:
            ran = Plan(**{**held, "featurizers": {}})
        except PlanError as e:
            fail(f"{path}.from", f"{out}/manifest.json is not a run this clab reads: {e}")
        cx.sources[out] = {"out": where, "plan": ran, "described": described}
    return cx.sources[out]


def saved_value_record(cx: Compiling, path: str, out: str, value: str) -> tuple[dict, dict]:
    """The run at `out` a load reads (`loaded_run`) and the record its manifest holds for saved value `value`; refused
    by name where the run saved no such value."""
    src = loaded_run(cx, path, out)
    if value not in src["described"]:
        fail(f"{path}.value", f"{out} saved no {value!r}; it saved {', '.join(src['described'])}")
    return src, src["described"][value]


# -- references --------------------------------------------------------------------------------------


def mark_gathered(steps: dict, used: set[str]) -> None:
    """Marks each compiled entry the run takes off the model (`gathered`), a fit's body's and its evals' too: one
    `used` names (a value something references, a save, or the entries a fit's objective scalar is over), and the
    entry a gathered `grad` entry differentiates. Any other entry only writes, and an entry that neither writes nor
    is gathered does nothing."""
    for name, entry in runnable(steps).items():
        for point in points(entry):
            if point.kind == "fit":
                mark_gathered(point.steps, used)
            for k, e in entries(point).items():
                e.gathered = f"{name}.{k}" in used
            for e in list(entries(point).values()):
                if e.gathered and (of := gradient_of(e)) is not None:
                    entries(point)[of].gathered = True


def compile_featurizer(cx: Compiling, plan: Plan, name: str, at: dict[str, int]) -> Featurizer:
    """A declared featurizer at one point, with the bundle its `file_path` names, checked key by key against
    the document."""
    f, path = plan.featurizers[name], f"featurizers.{name}"
    assert not isinstance(f, list)
    fields, classes = f.model_extra or {}, None
    if fields.get("classes") is not None:
        check_column_ref(cx, fields["classes"], f"{path}.classes", None)
        try:
            classes = labels(cx.rows, fields["classes"])
        except ValueError as e:
            fail(f"{path}.classes", str(e))
    missing = [k for k in KINDS[f.kind].drawn if fields.get(k) is None]
    if fields.get("file_path") is None and missing:
        fail(path, f"a {f.kind!r} featurizer drawn without a bundle needs {' and '.join(map(repr, missing))}; or "
             "load one with 'file_path'")
    record = {**f.model_dump(exclude_defaults=True), "at": at}
    if isinstance(fields.get("site"), str):         # the site it was trained at, inlined as an entry's is
        if fields["site"] not in plan.sites:
            fail(f"{path}.site", f"no site {fields['site']!r}; declared: {', '.join(plan.sites) or 'none'}")
        named = plan.sites[fields["site"]]
        if not isinstance(named.layers, int) or named.select:
            fail(f"{path}.site", f"site {fields['site']!r} is {named.model_dump(exclude_defaults=True)}; the site a "
                 "featurizer was trained at is a component at one layer, {'component', 'layers': <int>}")
        record["site"] = fields["site"] = {"component": named.component, "layers": named.layers}
    if fields.get("file_path") is None:
        return Featurizer(**record)
    try:
        tensors, header = bundles.load(cx.root, f.kind, fields, plan.model, classes)
    except ValueError as e:
        fail(f"{path}.file_path", str(e))
    return Featurizer(**record, bundle=tensors, header=header)


def compile_fit(cx: Compiling, path: str, name: str, spec: Fit, plan: Plan, at: dict[str, int]) -> Fit:
    """A fit's body runs over minibatches of its datasets' rows, the same rows of each: so its datasets have
    as many rows as each other, a value made before the fit that the body uses with a row per example is of a
    dataset a forward of the body reads (with no forward, of one dataset), and no eval reads such a value's dataset
    as another. It trains featurizers that have something to train, its objective depends on one of them, and it
    holds no fit. The body compiles with each trained featurizer's bare name written `<fit>.<f>`, in a scope that
    holds the fit; then each eval compiles it again, as ordinary steps after the fit (`Fit.evals`)."""
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
    fit = Fit(**spec.model_dump(exclude_defaults=True, exclude={"steps", "evals"}), steps={}, at=at, evals={})
    own = {f: f"{name}.{f}" for f in spec.train}            # a trained featurizer's bare name, in the body
    within, view = replace(cx, steps={**cx.steps, name: fit}), rename_bundles(plan, own, {})
    body: dict[str, Any] = {}
    harvests: dict[str, str] = {}   # each value made before the fit, with rows, that the body uses, by its first user
    trained = set(spec.train) | set(own.values())   # bare in an objective's scalar, `<fit>.<f>` in the body
    for n, s in spec.steps.items():
        if isinstance(s, Fit):
            fail(f"{path}.steps.{n}", "a fit's body holds no fit")
        if isinstance(s, Load):
            fail(f"{path}.steps.{n}", "a fit's body holds no load: load the value before the fit, and the body takes "
                 "its rows by id, as it does a harvest's")
        if n in plan.data or n in plan.featurizers or "." in n:
            fail(f"{path}.steps.{n}", f"{n!r} is also a dataset or a featurizer, or has a dot; a body's names are "
                 "a root's")
        assert not isinstance(s, list), "a point holds one step under each name"
        for p, k, e in entries_with_paths(plan, f"{path}.steps.{n}", n, s) if isinstance(s, Forward) else []:
            for i in [i for i, o in enumerate(e.ops) if o.op == "grad"][:1]:
                fail(f"{p}.ops[{i}]" + (f" in {path}.steps.{n}" if p.startswith("interventions.") else ""),
                     "grad inside a fit's body is not served: the fit differentiates its objective for the "
                     "featurizers it trains; take the gradient in a forward before or after the fit")
        s = rename_spec(s, own, {})
        body[n] = within.steps[n] = compile_step(within, f"{path}.steps.{n}", n, s, view, at)
        refs = referenced_names(cx, s.model_dump(exclude_defaults=True))
        for r in refs:
            thing = r.partition(".")[0]
            if thing in cx.steps and thing not in body and "rows" in (record_of(cx.steps, r, at).dims or []):
                harvests.setdefault(r, n)
        if any(r in trained or r.partition(".")[0] in trained for r in refs):
            trained.add(n)                  # a step whose value depends on what the fit trains
    body_ops = [n for n, s in body.items() if s.kind == "op"]
    for term in spec.objective:
        thing, _, scalar = term.partition(".")
        if term not in body_ops and not (thing in spec.train and scalar in featurizers.KINDS[
                kind_of(plan, thing)].scalars):
            fail(f"{path}.objective.{term}", f"not a body op or a trained featurizer's scalar; body ops: "
                 f"{', '.join(body_ops)}")
        if term not in body_ops:            # a scalar is over what the body's reads of its coordinates saw
            reads = open_encodes(body, own[thing])
            cx.used.update(r for r in reads if "." in r)        # the entries among them, which the run gathers
            if scalar in featurizers.KINDS[kind_of(plan, thing)].over_encodes and not reads:
                fail(f"{path}.objective.{term}", f"{scalar!r} is over the coordinates the body reads of {thing!r}, an "
                     f"encode no decode closes (an entry whose ops end in {{'op': 'encode', 'featurizer': {thing!r}}}, "
                     "or an encode step), and this body has none: read its coordinates in the body, or make the term "
                     "an op over what it decodes")
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
                if record_of(cx.steps, ref, at).dataset == a:
                    fail(where, f"the body uses {ref!r}, made before the fit with a row per example of {a!r}, and "
                         f"this eval reads {a!r} as {b!r}, over whose rows no step made it; evaluate on {b!r} in "
                         "steps after the fit, which reach what it trained as "
                         f"{', '.join(repr(f'{name}.{f}') for f in spec.train)}")
            if a in cx.chats and plan.data[a].template != plan.data[b].template:
                fail(where, f"dataset {a!r} is read as {b!r}, and their templates differ ({a!r}: "
                     f"{plan.data[a].template}, {b!r}: {plan.data[b].template}); a chat renders with its own "
                     "dataset's template, so give the two the same one")
    counts = {d: len(cx.rows[d]) for d in {s.dataset for s in body.values()} - {None}}
    if len(set(counts.values())) > 1:
        fail(f"{path}.steps", f"the body's datasets have {counts} rows; it runs over minibatches of the same rows of "
             "each, paired row by row, so they have as many rows: repeat the smaller dataset's rows (or cut the "
             "larger's) to the same count, or fit on one")
    # The rows a minibatch's ids are: of the datasets the body's forwards read, or, with none, its first harvest's.
    rows = {s.dataset for s in body.values() if s.kind != "op"} or \
        {record_of(cx.steps, r, at).dataset for r in list(harvests)[:1]}
    for ref, n in harvests.items():
        d = record_of(cx.steps, ref, at).dataset
        if d not in rows:
            theirs = " and ".join(map(repr, sorted(map(str, rows))))
            fail(f"{path}.steps.{n}", f"the body uses {ref!r}, made before the fit with a row per example of {d!r}, "
                 f"which no forward of the body reads; the body runs over the rows of {theirs}, so a minibatch would "
                 f"take rows of {d!r} at the ids of its rows of {theirs}, pairing the two row by row: read {d!r} in a "
                 f"forward of the body, or make {ref!r} over {theirs}")
    fit.steps = body
    for label, swap in spec.eval.items():
        scope, names, fit.evals[label] = replace(cx, steps={**cx.steps, name: fit}), dict(own), {}
        for n, s in spec.steps.items():
            names[n] = f"{name}.{label}.{n}"    # it and the body's steps before it: a later one's name is the root's
            pulls = isinstance(s, (Forward, Generate)) and bundles_of(s)       # a bundle, whose entries it renames
            view = rename_bundles(plan, names, swap) if pulls else plan
            try:
                made = compile_step(scope, f"{path}.steps.{n}", names[n], rename_spec(s, names, swap), view, at)
            except PlanError as e:
                raise PlanError(f"{path}.eval.{label}: {e}") from None
            for record in [made, *entries(made).values()]:
                if isinstance(record, Labelled):
                    record.axes = list(at)
            if isinstance(made, Forward):
                made.schedule, made.batch_size = {"axes": list(at)}, cast(int, spec.batch_size)
            fit.evals[label][n] = made
    return fit


def rename_spec(spec: Any, names: dict[str, str], datasets: dict[str, str]) -> Any:
    """A body step's spec (or a bundle's entries, as a dict) with each name `names` maps renamed where it is the
    thing a reference names (a step, or a featurizer: `v_cf.x` under `{"v_cf": "fit.held_out.v_cf"}`), and each
    dataset `datasets` maps read as the one it maps to (in an input, a column an op or a scope names); a `grad`'s
    `of`, an entry of its own forward, keeps its name."""
    def first(ref: Any, to: dict[str, str]) -> Any:      # a reference (or a column's, or a list of them) renamed
        if isinstance(ref, list):
            return [first(r, to) for r in ref]
        if isinstance(ref, dict) and "column" in ref:
            return {**ref, "column": first(ref["column"], to)}
        thing, dot, rest = ref.partition(".") if isinstance(ref, str) else ("", "", "")
        return f"{to[thing]}{dot}{rest}" if thing in to else ref

    def rename_op(o: dict) -> dict:
        params = ops.parameters(o["op"])
        kinds = {k: params[k].annotation if k in params else None for k in o}
        return {k: first(v, names) if kinds[k] in (ops.Value, ops.Featurizer) or k == "of" and o["op"] != "grad" else
                first(v, datasets) if kinds[k] in (ops.Tokens, ops.Text) else v for k, v in o.items()}

    def rename_entry(e: dict) -> dict:
        pos, out = e.get("pos"), {**e, "ops": [rename_op(o) for o in e.get("ops", [])]}
        if isinstance(pos, dict) and "column" in (pos.get("scope") or {}):
            out["pos"] = {**pos, "scope": {**pos["scope"], "column": first(pos["scope"]["column"], datasets)}}
        return out

    def listed(value: Any) -> Any:      # a step's `interventions`: a bundle's name, entries, or a list of either
        return [listed(v) for v in value] if isinstance(value, list) else value if isinstance(value, str) else \
            {k: rename_entry(e) for k, e in value.items()}

    if not isinstance(spec, dict):
        return type(spec).model_validate(rename_spec(spec.model_dump(exclude_defaults=True), names, datasets))
    if spec.get("kind") == "op":
        return rename_op(spec)
    if "kind" in spec:
        return {**spec, "input": first(spec["input"], datasets), "interventions": listed(spec.get("interventions", {}))}
    return listed(spec)


def rename_bundles(plan: Plan, names: dict[str, str], datasets: dict[str, str]) -> Plan:
    """The plan a body step compiles in under `rename_spec`'s `names` and `datasets`: each bundle of entries it may pull
    in renamed so too, a name of the bundle's own entries left bare, which `rescope_entry` names in the step."""
    if not plan.interventions:
        return plan
    bundles = {b: rename_spec({k: e.model_dump(exclude_defaults=True) for k, e in bundle.items()},
                              {n: v for n, v in names.items() if n not in bundle}, datasets)
               for b, bundle in plan.interventions.items()}
    return plan.model_copy(update={"interventions": {b: {k: Intervention.model_validate(e) for k, e in bundle.items()}
                                                     for b, bundle in bundles.items()}})
