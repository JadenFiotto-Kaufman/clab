"""Everything that needs the model, checked before any forward: the records completed with what the model
decides, what the run needs found, and what it cannot run refused by name."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import answer, check_classes, check_tokens
from clab.compile.arguments import carried, mismatch, references
from clab.compile.interventions import check_operands, refuse, resolved
from clab.compile.positions import indices
from clab.compile.schedule import scheduled
from clab.compile.state import Found
from clab.compile.uses import widths
from clab.format import address, data, selectors
from clab.format.selectors import Selector, picked, selected
from clab.plan import (
    Forward,
    Labelled,
    Plan,
    Site,
    adapts,
    arguments,
    body_view,
    entries,
    entry_of,
    gathers,
    held_rows,
    kind_of,
    points,
    record_of,
    runnable,
    serving,
    writes,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def preflight(engine: Engine, plan: Plan) -> Found:
    """Everything that needs the model, checked before any forward, and the plan's records completed with it:
    each read's and write's `layers` and `kept` resolved among the engine's sizes, each prefill record's
    `positions` per row and the `width` they confirm, every op step's labels from its producer's, each forward's
    and generation's `schedule` (`schedule.scheduled`: the sites it visits in the engine's forward order, with the
    records done at each, and where it stops), and each
    featurizer's `width` at the sites it is used at (`uses.widths`). Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs beside the records: each dataset input's encoded rows and padded width
    (`inputs`, a fit body's among them) and room for what each generate step's rows said (`said`).

    Refuses by name (`<step>.<entry>`): a component the engine does not serve; a layer or a `select` outside the
    model's sizes (on `key`, outside the input's shortest row), or a `select` that keeps none of them; a `swap` of a
    component of ids with a number that is no id under the model's size (`address.COMPONENTS`' `ids`); a position
    that does not fit some row's scope, or whose scope is not in a row a write or a read of static width needs; a
    token column that is not one token on some row; an operand that breaks Ops' one rule once the model's sizes
    and the rows' positions are known (`arguments.mismatch`: a layer the op acts at that the operand does not hold,
    heads in another order, one head for several, another width); a write whose operand its own forward reads no
    earlier, a write to a component with a `key` axis (the attention pattern, the scores) whose operand was read on
    another input that does not tokenize to the same length on every row, which would misalign the keys, and a
    window of several positions that reaches past the tokens two such inputs share (`check_operands`, `windowed`);
    a `pca` over no more vectors than its `k`; a featurizer used at two widths (at a site, the site's; inside an
    `encode`, that one's coordinates'), at a width its bundle's header does not say (or, with no header, whose
    tensors are not as wide), with a field that counts more directions than that width (`counts`), at another
    layer than the one it says it was trained at, reading its input elsewhere than its bundle was made reading it
    (`read_with`), or encoded with a `select` outside its features or keeping none of them; and a trained
    featurizer used at no site. A write whose operand its own forward reads at a site the engine reaches no earlier
    than the write's is refused where the schedule is written. A refusal inside a fit's body names the fit, and one
    in an eval its eval step (`<fit>.<eval>.<step>`).
    """
    refuse(engine, plan)
    rows = {n: held_rows(plan, n) for n in plan.data}
    classes = {name: featurizers.labels(rows, (spec.model_extra or {})["classes"])
               for name, entry in plan.featurizers.items() for spec in points(entry)[:1] if spec.kind == "linear"}
    found = Found()
    prepare(engine, plan, plan.steps, classes, found)
    widths(engine, plan)
    return found


def prepare(engine: Engine, plan: Plan, steps: dict, classes: dict[str, list], found: Found) -> None:
    """`preflight`'s checks and findings for `steps`, in the order they run (`plan.runnable`: a fit's eval steps
    after it), over the rows `plan` holds, into `found`; a fit's body is prepared at the fit's place, over its
    training rows, in the plan as the body sees it (`plan.body_view`), a refusal there named after the fit.
    `classes` is each probe's class list. Once every point of a step has its records, each serving point's schedule
    is written (`scheduled`)."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    alike: dict[tuple, list] = {}           # each distinct prefill resolution, which records alike share
    for name, entry in runnable(steps).items():
        for step in points(entry):
            if step.kind == "fit":
                try:
                    prepare(engine, body_view(plan, name, step.at), step.steps, classes, found)
                except ValueError as e:
                    raise ValueError(f"{name}.{e}") from None
                continue
            if step.kind == "op":
                args, of = arguments(step), record_of(plan.steps, step.of, step.at)
                assert step.dims is not None and of.dims is not None
                step.layers, step.kept, step.width = carried(of, step.dims)     # completed from its producer's
                for key, (written, theirs) in references(step.op, args, resolving(plan, step.at)).items():
                    why = mismatch(of, theirs, step.op, key, written, repr(step.of), False, False, adapts(plan, step))
                    if why:
                        raise ValueError(f"{name}: operand {why}")
                if "classes" in of.dims:
                    check_classes(plan, name, step.op, args, of.space[-1]["featurizer"], classes)
                else:
                    check_tokens(engine, plan, found, name, step.op, args, of.dims, *placed(plan, of, step.at, found))
                bundle = ops.DECLARED[step.op]["bundle"]
                for field in featurizers.KINDS[bundle].counts if bundle else ():
                    width = len(of.kept["hidden_size"]) if "hidden_size" in of.kept else sizes["hidden_size"]
                    if args[field] > width:
                        raise ValueError(f"{name}: {step.op} {field} {args[field]} counts directions of a "
                                         f"value {width} wide, which has no more than {width}")
                if step.op == "pca" and of.read is not None:
                    got = resolved(plan.steps, of.read, step.at)
                    n = sum(len(p) for p in got if isinstance(p, list)) if got is not None else None
                    if n is not None and n <= args["k"]:
                        raise ValueError(f"{name}: pca of {n} vectors has no rank-{args['k']} basis; "
                                         f"it needs more than {args['k']}")
                continue
            if step.kind == "load":         # the input its read was made on, whose tokens its positions are of
                if step.source is not None:
                    tokenized(engine, plan, found, name, step.source)
                continue
            rows, encoded = held_rows(plan, step.dataset), tokenized(engine, plan, found, name, step)
            for key, record in entries(step).items():
                where, site = f"{name}.{key}", cast(Site, record.site)
                layers, select = cast("Selector | None", site.layers), cast("dict[str, Selector] | None", site.select)
                record.layers = None if layers is None else picked(where, layers, sizes["num_layers"])
                for axis, index in (select or {}).items():
                    if selectors.offset(index) is not None:
                        continue                    # relative to each query: past a row's keys it is missing
                    shortest = min(len(e["ids"]) for e in encoded)
                    size, what = (shortest, f"the {shortest} tokens of the input's shortest row") \
                        if axis == "key" else (sizes[axis], f"the model's {sizes[axis]}")
                    selected(where, axis, index, size, what)
                record.kept = selectors.kept(select, sizes)
                if writes(record) and (axis := address.COMPONENTS[site.component].get("ids")):
                    for o in record.ops:
                        given, n = arguments(o).get("with") if o.op == "swap" else None, sizes[axis]
                        if isinstance(given, (int, float)) and not (0 <= given < n and given == int(given)):
                            raise ValueError(f"{where}: swap with {given!r}: {site.component} holds ids, and an id is "
                                             f"a whole number from 0 to {n - 1} (the model's {axis}, {n})")
                pos = positions.parse(record.pos)
                resolved_ = None
                if positions.prefill(pos) or step.kind == "forward":    # a forward's one call is its prefill
                    resolved_ = indices(where, record, step, rows, encoded, tokenizer, alike)
                    record.positions = {**record.positions, step.dataset: resolved_}
                    counts = {len(p) if isinstance(p, list) else 0 for p in resolved_}     # a missing row holds none
                    record.width = counts.pop() if len(counts) == 1 else None   # what the rows confirm
                probe = None                        # past a probe's encode, a column names its classes
                for o in record.ops:
                    if probe is None:
                        check_tokens(engine, plan, found, where, o.op, arguments(o), None, resolved_,
                                     [e["ids"] for e in encoded], bool((pos["scope"] or {}).get("before")))
                    else:
                        check_classes(plan, where, o.op, arguments(o), probe, classes)
                    if o.op == "encode" and kind_of(plan, arguments(o)["featurizer"]) == "linear":
                        probe = arguments(o)["featurizer"]
                # A metric of an answer of several tokens lines up with the read's positions, which a generation knows
                # once it ends: it and the ops after it wait for that (`lined_up`).
                first = len(record.ops) - len(gathers(record))
                record.per_call = next((i - first for i in range(first, first + record.per_call)
                                        if lined_up(engine, plan, found, record.ops[i], where)), record.per_call)
            # What `check_operands` finds of a write's operand depends on the write's component, positions and
            # selection and on the op's arguments, never its layers: each such write and op is checked once.
            checked: set[tuple] = set()
            for key, record in entries(step).items():   # the operands, once every entry of the step has its labels
                where, edits = f"{name}.{key}", len(record.ops) - len(gathers(record))
                dims = address.dims(cast(Site, record.site).component)
                for i, o in enumerate(record.ops):
                    # What the op meets: the op before it left it, as its record says (its labels the entry's).
                    ours = Labelled(dims=dims, space=record.ops[i - 1].space if i else [], dataset=record.dataset,
                                    layers=record.layers, read=record.read, width=record.width if "pos" in dims else None,
                                    kept={a: v for a, v in record.kept.items() if a in dims})
                    for k, (written, theirs) in references(o.op, arguments(o), resolving(plan, step.at)).items():
                        forward = written.partition(".")[0] == name
                        why = mismatch(ours, theirs, o.op, k, written, repr(step.input), i < edits, forward,
                                       adapts(plan, o))
                        if why:
                            raise ValueError(f"{where}: operand {why}")
                    write = (cast(Site, record.site).component, repr(record.pos), repr(record.kept), o.op,
                             repr(arguments(o)))
                    if i < edits and write not in checked:
                        checked.add(write)
                        check_operands(engine, plan, where, o, step, record, found)
                    dims = o.dims or dims
        if points(entry)[0].kind in ("forward", "generate"):
            for step, served in cast("list[tuple[Forward, list[Forward]]]", serving(plan, name)):
                scheduled(engine, plan, name, step, served)


def tokenized(engine: Engine, plan: Plan, found: Found, name: str, step: Forward) -> list[dict]:
    """The rows of `step`'s input as the tokenizer encodes them (`data.encode`), encoded once per input
    (`found.inputs`, with the padded width) and recorded as token strings in `plan.run.tokens` under the input's
    name; a chat of a template that reads the clock without its date pinned, and a row that does not encode, are
    refused by name."""
    assert plan.run is not None and step.dataset is not None
    source, tokenizer = data.input_name(step.input), engine.tokenizer
    if source not in found.inputs:
        rows, template = held_rows(plan, step.dataset), plan.data[step.dataset].template
        if any(isinstance(data.field(r, step.input), list) for r in rows) and data.clocked(tokenizer, template):
            raise ValueError(f"{name}: {source}: the checkpoint's chat template calls strftime_now, which writes "
                             "today's date into the prompt, so these rows render to other tokens on another day and "
                             f"nothing in the run record says so; pin it in data.{step.dataset}.template, "
                             "{\"date_string\": \"26 Jul 2024\"}")
        encoded = []
        for i, r in enumerate(rows):
            try:
                encoded.append(data.encode(tokenizer, data.field(r, step.input), template))
            except ValueError as e:
                raise ValueError(f"{name}: {source} row {i}: {e}") from None
        found.inputs[source] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
        plan.run.tokens[source] = [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
    return found.inputs[source]["rows"]


def lined_up(engine: Engine, plan: Plan, found: Found, o: Any, where: str) -> bool:
    """Whether op `o` scores a column with an answer of several tokens (`Answer.aligned`), whose token i scores the
    read's position i."""
    params = ops.parameters(o.op)
    return any(answer(engine, plan, found, v, where).aligned for k, v in arguments(o).items()
               if k in params and params[k].annotation == ops.Tokens)


def resolving(plan: Plan, at: dict[str, int]) -> Callable[[str, Any, inspect.Parameter], Labelled | None]:
    """How preflight resolves an op's named arguments (`arguments.references`), every name known to resolve: a
    value to its record at `at`, a column to a row per example of its dataset, a literal to a row per example of any
    (`arguments.resolver`), a featurizer to nothing."""
    def resolve(key: str, value: Any, p: inspect.Parameter) -> Labelled | None:
        if p.annotation == ops.Value:
            return record_of(plan.steps, value, at)
        if p.annotation == ops.Featurizer:
            return None
        if data.literal_of(value) is not None:
            return Labelled(dims=["rows"])      # the same text on every row of the value it goes with
        return Labelled(dims=["rows"], dataset=(data.column_of(value) or (str(value),))[0].partition(".")[0])
    return resolve


def placed(plan: Plan, of: Labelled, at: dict, found: Found) -> tuple[list | None, list[list[int]] | None, bool]:
    """The positions per row a value holds (its read's, `resolved`: a list, or the marker of a row it does not
    reach), where they are known before the run, with each row's input ids and whether that read is a scope that
    is `before` (whose next tokens an answer scores); else None, None and False."""
    if of.read is None:
        return None, None, False
    where, read = resolved(plan.steps, of.read, at), of.read
    if where is None:
        return None, None, False
    step, record = entry_of(plan.steps, read, at)
    scope = positions.parse(record.pos)["scope"] or {}
    encoded = found.inputs.get(data.input_name(step.input))
    return where, None if encoded is None else [e["ids"] for e in encoded["rows"]], bool(scope.get("before"))
