"""Everything that needs the model, checked before any forward: the records completed with what the model
decides, what the run needs found, and what it cannot run refused by name."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import answer, check_classes, check_tokens
from clab.compile.arguments import k_past, kept_labels, mismatch, references, select_kept, selected_layers
from clab.compile.interventions import check_operands, read_positions, refuse_unserved
from clab.compile.positions import prefill_positions
from clab.compile.schedule import write_schedule
from clab.compile.state import Findings
from clab.compile.uses import check_named_widths, write_widths
from clab.format import address, data, selectors
from clab.format.selectors import Selector, picked, selected
from clab.format.sweeps import coordinate_text
from clab.plan import (
    Forward,
    Generate,
    Labelled,
    Plan,
    PlanError,
    Site,
    adapts,
    arguments,
    body_view,
    declared,
    entries,
    entry_of,
    gathers,
    held_rows,
    horizon,
    kind_of,
    points,
    record_of,
    runnable,
    serving,
    writes,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def preflight(engine: Engine, plan: Plan) -> Findings:
    """Everything that needs the model, checked before any forward, and the plan's records completed with it:
    each read's and write's `layers` and `kept` resolved among the engine's sizes, each prefill record's
    `positions` per row and the `width` they confirm, every op step's labels from its producer's, each forward's
    and generation's `schedule` (`schedule.write_schedule`: the sites it visits in the engine's forward order, with the
    records done at each, and where it stops), and each
    featurizer's `width` at the sites it is used at (`uses.write_widths`). Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs beside the records (`state.Findings`).

    Refuses by name (`<step>.<entry>`, and ` at <label>=<value>` at a point of a sweep): a component the engine does
    not serve; a layer or a `select` outside the model's sizes (on `key`, outside the input's shortest row), or a
    `select` that keeps none of them; a `swap` of a component of ids with a number that is no id under the model's
    size (`address.COMPONENTS`' `ids`); a position that does not fit some row's scope, or whose scope is not in a row
    a write or a read of static width needs; a token column that is not one token on some row; an operand that breaks
    Ops' one rule once the model's sizes and the rows' positions are known (`arguments.mismatch`: a layer the op acts
    at that the operand does not hold, heads in another order, one head for several, another width, a whole scope from
    another forward whose rows hold different numbers of tokens, each count's rows named); a write to a component with
    a `key` axis (the attention pattern, the scores) whose operand was read on another input that does not tokenize to
    the same length on every row, which would misalign the keys, and a window of several positions that reaches past
    the tokens two such inputs share (`check_operands`, `check_window`); a `pca` over no more vectors than its `k`; a
    featurizer used at two widths (at a site, the site's; inside an `encode`, that one's coordinates'), at a width its
    bundle's header does not say (or, with no header, whose tensors are not as wide), with a field that counts more
    directions than that width (`counts`), at another layer than the one it says it was trained at, reading its input
    elsewhere than its bundle was made reading it (`check_input_site`), or encoded with a `select` outside its
    features or keeping none of them; a featurizer drawn from a value (a constant's `init`) that keeps other indices
    of its axis than a use does (`check_named_widths`); and a trained featurizer used at no site. A write whose
    operand its own forward reads at a site the engine reaches no earlier than the write's is refused where the
    schedule is written. A refusal inside a fit's body names the fit, and one in an eval its eval step
    (`<fit>.<eval>.<step>`).
    """
    refuse_unserved(engine, plan)
    rows = {n: held_rows(plan, n) for n in plan.data}
    classes = {name: featurizers.labels(rows, (spec.model_extra or {})["classes"])
               for name, entry in plan.featurizers.items() for spec in points(entry)[:1] if spec.kind == "linear"}
    findings = Findings()
    prepare(engine, plan, plan.steps, classes, findings)
    write_widths(engine, plan)
    check_named_widths(engine, plan)
    return findings


def prepare(engine: Engine, plan: Plan, steps: dict, classes: dict[str, list], findings: Findings) -> None:
    """`preflight`'s checks and findings for `steps`, in the order they run (`plan.runnable`: a fit's eval steps
    after it), over the rows `plan` holds, into `findings`; a fit's body is prepared at the fit's place, over its
    training rows, in the plan as the body sees it (`plan.body_view`), a refusal there named after the fit.
    `classes` is each probe's class list. Once every point of a step has its records, each serving point's schedule
    is written (`write_schedule`)."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    alike: dict[tuple, list] = {}           # each distinct prefill resolution, which records alike share
    for name, entry in runnable(steps).items():
        for step in points(entry):
            with naming_point(plan, step.at):
                if step.kind == "fit":
                    try:
                        prepare(engine, body_view(plan, name, step.at), step.steps, classes, findings)
                    except ValueError as e:
                        raise ValueError(f"{name}.{e}") from None
                    continue
                if step.kind == "op":
                    args, of = arguments(step), record_of(plan.steps, step.of, step.at, plan.featurizers)
                    assert step.dims is not None and of.dims is not None
                    compiled = dict(step.kept)
                    step.layers, step.kept, step.width = kept_labels(of, step.dims)     # completed from its producer's
                    if step.op == "select":         # its own, held to the model's sizes and resolved among them
                        step.layers, step.kept = selected_layers(of, step.kept | select_kept(
                            at_point(plan, name, step.at), of, args, sizes, probe_of(plan, of, classes), None,
                            compiled))
                    if why := k_past(step.op, args, of, sizes, probe_of(plan, of, classes)):
                        raise ValueError(f"{at_point(plan, name, step.at)}: {why}")
                    for key, (written, theirs) in references(step.op, args, preflight_resolver(plan, step.at)).items():
                        why = mismatch(of, theirs, step.op, key, written, repr(step.of), False, False,
                                       adapts(plan, step), lambda read: entry_of(plan.steps, read, step.at)[1])
                        if why:
                            raise ValueError(f"{name}: operand {why}")
                    if "classes" in of.dims:
                        check_classes(plan, name, step.op, args, of.space[-1]["featurizer"], classes)
                    else:
                        check_tokens(engine, plan, findings, name, step.op, args, of.dims,
                                     *known_positions(plan, of, step.at, findings))
                    bundle = ops.DECLARED[step.op]["bundle"]
                    for field in featurizers.KINDS[bundle].counts if bundle else ():
                        width = len(of.kept["hidden_size"]) if "hidden_size" in of.kept else sizes["hidden_size"]
                        if args[field] > width:
                            raise ValueError(f"{name}: {step.op} {field} {args[field]} counts directions of a "
                                             f"value {width} wide, which has no more than {width}")
                    if step.op == "pca" and of.read is not None:
                        row_positions = read_positions(plan.steps, of.read, step.at)
                        n = sum(len(p) for p in row_positions if isinstance(p, list)) \
                            if row_positions is not None else None
                        if n is not None and n <= args["k"]:
                            raise ValueError(f"{name}: pca of {n} vectors has no rank-{args['k']} basis; "
                                             f"it needs more than {args['k']}")
                    continue
                if step.kind == "load":         # the input its read was made on, whose tokens its positions are of
                    if step.source is not None:
                        encode_input(engine, plan, findings, name, step.source)
                    continue
                rows, encoded = held_rows(plan, step.dataset), encode_input(engine, plan, findings, name, step)
                check_longest(engine, name, step, findings)
                for key, record in entries(step).items():
                    where, site = f"{name}.{key}", cast(Site, record.site)
                    layers = cast("Selector | None", site.layers)
                    select = cast("dict[str, Selector] | None", site.select)
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
                                raise ValueError(f"{where}: swap with {given!r}: {site.component} holds ids, and an "
                                                 f"id is a whole number from 0 to {n - 1} (the model's {axis}, {n})")
                    pos = positions.parse(record.pos)
                    row_positions = None
                    if positions.prefill_only(pos) or step.kind == "forward":    # a forward's one call is its prefill
                        row_positions = prefill_positions(where, record, step, rows, encoded, tokenizer, alike)
                        record.positions = {**record.positions, step.dataset: row_positions}
                        # a missing row holds none
                        counts = {len(p) if isinstance(p, list) else 0 for p in row_positions}
                        record.width = counts.pop() if len(counts) == 1 else None   # what the rows confirm
                    probe = None                        # past a probe's encode, a column names its classes
                    for o in record.ops:
                        if probe is None:
                            check_tokens(engine, plan, findings, where, o.op, arguments(o), None, row_positions,
                                         [e["ids"] for e in encoded], bool((pos["scope"] or {}).get("before")))
                        else:
                            check_classes(plan, where, o.op, arguments(o), probe, classes)
                        if o.op == "encode" and kind_of(plan, arguments(o)["featurizer"]) == "linear":
                            probe = arguments(o)["featurizer"]
                    # A metric of an answer of several tokens lines up with the read's positions, which a generation
                    # knows once it ends: it and the ops after it wait for that (`scores_aligned_answer`).
                    first = len(record.ops) - len(gathers(record))
                    record.per_call = next((i - first for i in range(first, first + record.per_call)
                                            if scores_aligned_answer(engine, plan, findings, record.ops[i], where)),
                                           record.per_call)
                # What `check_operands` finds of a write's operand depends on the write's component, positions and
                # selection and on the op's arguments, never its layers: each such write and op is checked once.
                checked: set[tuple] = set()
                for key, record in entries(step).items():   # the operands, once every entry of the step has its labels
                    where, edits = f"{name}.{key}", len(record.ops) - len(gathers(record))
                    dims = address.dims(cast(Site, record.site).component)
                    for i, o in enumerate(record.ops):
                        # What the op meets: the op before it left it, as its record says (its labels the entry's).
                        ours = Labelled(dims=dims, space=record.ops[i - 1].space if i else [], dataset=record.dataset,
                                        layers=record.layers, read=record.read,
                                        width=record.width if "pos" in dims else None,
                                        kept={a: v for a, v in record.kept.items() if a in dims})
                        if why := k_past(o.op, arguments(o), ours, sizes, probe_of(plan, ours, classes)):
                            raise ValueError(f"{at_point(plan, where, step.at)}: {why}")
                        for k, (written, theirs) in references(o.op, arguments(o),
                                                               preflight_resolver(plan, step.at)).items():
                            forward = written.rpartition(".")[0] == name
                            why = mismatch(ours, theirs, o.op, k, written, repr(step.input), i < edits, forward,
                                           adapts(plan, o), lambda read: entry_of(plan.steps, read, step.at)[1])
                            if why:
                                raise ValueError(f"{where}: operand {why}")
                        write = (cast(Site, record.site).component, repr(record.pos), repr(record.kept), o.op,
                                 repr(arguments(o)))
                        if i < edits and write not in checked:
                            checked.add(write)
                            check_operands(engine, plan, where, o, step, record, findings)
                        dims = o.dims or dims
        if points(entry)[0].kind in ("forward", "generate"):
            for step, served in cast("list[tuple[Forward, list[Forward]]]", serving(plan, name)):
                with naming_point(plan, step.at):
                    write_schedule(engine, plan, name, step, served)
                    check_last_only(engine, name, step, findings)


@contextmanager
def naming_point(plan: Plan, at: dict[str, int]) -> Iterator[None]:
    """A refusal at one point of a sweep names the point after its path (`at_point`), as compile's do; one whose
    path names it already (a fit body's step at the fit's point) is left as it is."""
    try:
        yield
    except ValueError as e:
        path, sep, why = str(e).partition(": ")
        if not at or not sep or " at " in path or type(e) not in (ValueError, PlanError):
            raise
        raise type(e)(f"{at_point(plan, path, at)}: {why}") from None


def at_point(plan: Plan, where: str, at: dict[str, int]) -> str:
    """A refusal's `where` at one point of a sweep, named by its coordinate (`<where> at max_new_tokens=2`), as the run
    names a step's out-of-memory refusal; `where` alone at a point on no axis."""
    return where + (f" at {coordinate_text(plan, at)}" if at else "")


def probe_of(plan: Plan, record: Labelled, classes: dict[str, list]) -> dict[str, tuple[int, str]]:
    """How many classes the probe a value's `classes` axis is of holds (the last encode left open around it), among
    `classes` by declared featurizer, as `arguments.k_past` takes it; none for a value with no `classes` axis."""
    if "classes" not in (record.dims or []) or not record.space:
        return {}
    held = classes.get(declared(plan.featurizers, record.space[-1]["featurizer"]))
    return {} if held is None else {"classes": (len(held), "the probe's")}


def check_last_only(engine: Engine, name: str, step: Forward, findings: Findings) -> None:
    """Refuses by name a record the schedule does at a component the engine serves at the prompt's last position
    alone (`engine.last_only`, the vllm engine's head; an engine without it serves every position) that meets any
    other prefill position on some row: the prefill records at those sites (a decode call's one position is the
    last), each at the positions preflight resolved for it, against the row's tokens. Read once the schedule is
    written, so that a read of `logits` the run's readout `at-reads` moved to the last block's output is not held
    to it, and one it left at the head is."""
    last_only, encoded = getattr(engine, "last_only", {}), findings.inputs[data.input_name(step.input)]["rows"]
    for site in step.schedule.get("sites", []):
        why = last_only.get(site["component"])
        for r in site["records"] if why else []:
            if "calls" in r:
                continue
            key = r["write"] if "write" in r else r["read"]
            for row, e in zip(entries(step)[key].positions.get(cast(str, step.dataset)) or [], encoded):
                if isinstance(row, list) and row != [len(e["ids"]) - 1]:
                    raise ValueError(f"{name}.{key}: the {engine.name} engine serves {site['component']} at the "
                                     f"prompt's last position alone ({why}); this entry meets "
                                     f"position{'s' if len(row) != 1 else ''} {row} on a row of {len(e['ids'])} "
                                     "tokens: name pos -1, or read layer_output at the last block and apply the "
                                     "readout op")


def check_longest(engine: Engine, name: str, step: Forward, findings: Findings) -> None:
    """Refuses by name a step whose longest row, plus what a generation adds (`horizon`, its `max_new_tokens`), is
    longer than the longest sequence the engine serves (`engine.longest`; an engine without it is unbounded): the
    vllm engine's `max_len`, past which vLLM would refuse the prompt or cut the generation short with no EOS."""
    longest = getattr(engine, "longest", None)
    if longest is None:
        return
    source = data.input_name(step.input)
    width = findings.inputs[source]["width"]
    more = (horizon(cast(Generate, step)) or 0) if step.kind == "generate" else 0
    if width + more > longest:
        raise ValueError(f"{name}: {source}'s longest row has {width} tokens"
                         + (f", and the step generates up to {more} more" if more else "")
                         + f", past the {longest} the {engine.name} engine serves (its max_len; raise it with -o "
                         "max_len, up to the checkpoint's context)")


def encode_input(engine: Engine, plan: Plan, findings: Findings, name: str, step: Forward) -> list[dict]:
    """The rows of `step`'s input as the tokenizer encodes them (`data.encode`), encoded once per input
    (`findings.inputs`, with the padded width) and recorded as token strings in `plan.run.tokens` under the input's
    name; a chat of a template that reads the clock without its date pinned, and a row that does not encode, are
    refused by name."""
    assert plan.run is not None and step.dataset is not None
    source, tokenizer = data.input_name(step.input), engine.tokenizer
    if source not in findings.inputs:
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
        findings.inputs[source] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
        plan.run.tokens[source] = [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
    return findings.inputs[source]["rows"]


def scores_aligned_answer(engine: Engine, plan: Plan, findings: Findings, o: Any, where: str) -> bool:
    """Whether op `o` scores a column with an answer of several tokens (`Answer.aligned`), whose token i scores the
    read's position i."""
    params = ops.parameters(o.op)
    return any(answer(engine, plan, findings, v, where).aligned for k, v in arguments(o).items()
               if k in params and params[k].annotation == ops.Tokens)


def preflight_resolver(plan: Plan, at: dict[str, int]) -> Callable[[str, Any, inspect.Parameter], Labelled | None]:
    """How preflight resolves an op's named arguments (`arguments.references`), every name known to resolve: a
    value to its record at `at`, a column to a row per example of its dataset, a literal to a row per example of any
    (`arguments.resolver`), a featurizer to nothing."""
    def resolve(key: str, value: Any, p: inspect.Parameter) -> Labelled | None:
        if p.annotation == ops.Value:
            return record_of(plan.steps, value, at, plan.featurizers)
        if p.annotation == ops.Featurizer:
            return None
        if data.literal_of(value) is not None:
            return Labelled(dims=["rows"])      # the same text on every row of the value it goes with
        return Labelled(dims=["rows"], dataset=(data.column_of(value) or (str(value),))[0].partition(".")[0])
    return resolve


def known_positions(plan: Plan, of: Labelled, at: dict,
                    findings: Findings) -> tuple[list | None, list[list[int]] | None, bool]:
    """The positions per row a value holds (its read's, `read_positions`: a list, or the marker of a row it does not
    reach), where they are known before the run, with each row's input ids and whether that read is a scope that
    is `before` (whose next tokens an answer scores); else None, None and False."""
    if of.read is None:
        return None, None, False
    row_positions, read = read_positions(plan.steps, of.read, at), of.read
    if row_positions is None:
        return None, None, False
    step, record = entry_of(plan.steps, read, at)
    scope = positions.parse(record.pos)["scope"] or {}
    encoded = findings.inputs.get(data.input_name(step.input))
    return row_positions, None if encoded is None else [e["ids"] for e in encoded["rows"]], bool(scope.get("before"))
