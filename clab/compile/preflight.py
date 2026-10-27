"""Everything that needs the model, checked before any forward: the records completed with what the model
decides, what the run needs found, and what it cannot run refused by name."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import check_classes, check_tokens
from clab.compile.arguments import carried, mismatch
from clab.compile.interventions import check_operands, met, operands, refuse, resolved, width_of
from clab.compile.positions import indices
from clab.compile.state import Found
from clab.compile.uses import widths
from clab.format import data, selectors
from clab.format.selectors import Selector, picked, selected
from clab.format.sweeps import put
from clab.plan import (
    Fit,
    Labelled,
    Plan,
    Site,
    arguments,
    body_view,
    entries,
    entry_of,
    gathers,
    held_rows,
    kind_of,
    points,
    record_of,
)

if TYPE_CHECKING:
    from clab.engines import Engine


def preflight(engine: Engine, plan: Plan) -> Found:
    """Everything that needs the model, checked before any forward, and the plan's records completed with it:
    each read's and write's `layers` and `kept` resolved among the engine's sizes, each prefill record's
    `positions` per row, and every op step's labels from its producer's. Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs beside the records: each dataset input's encoded rows and padded width
    (`inputs`), room for what each generate step's rows said (`said`), each featurizer's width (`widths`, the
    root's alone), and for each fit point the same for its body (`bodies`): over its training rows (`train`) and
    under each of its named `eval` substitutions (`evals`), each beside the plan the body runs in, which holds the
    body's steps and each substituted dataset's rows under the body's name for it.

    Refuses by name (`<step>.<entry>`): a component the engine does not serve; a layer or a `select` outside the
    model's sizes (on `key`, outside the input's shortest row), or a `select` that keeps none of them; a position
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
    featurizer used at no site. A refusal inside a fit's body names the fit, and under an eval the eval and the
    datasets it reads.
    """
    refuse(engine, plan)
    rows = {n: held_rows(plan, n) for n in plan.data}
    classes = {name: featurizers.labels(rows, (spec.model_extra or {})["classes"])
               for name, entry in plan.featurizers.items() for spec in points(entry)[:1] if spec.kind == "linear"}
    found = prepare(engine, plan, plan.steps, {}, classes, {})
    found.widths = widths(engine, plan)
    return found


def body(engine: Engine, plan: Plan, fit: str, step: Fit, label: str | None, swap: dict[str, str],
         classes: dict[str, list]) -> tuple[Plan, Found]:
    """A fit body's plan (under eval `label`, with each dataset `swap` maps read under the body's name for it) and
    its `prepare`, which completes the body's records (their positions under the dataset each run reads); a
    refusal names the fit and, under an eval, which one and the datasets it reads in place of the body's, whose
    rows a row number counts."""
    view = body_view(plan, fit, step.at, swap)
    try:
        return view, prepare(engine, view, step.steps, swap, classes, {f: f"{fit}.{f}" for f in step.train})
    except ValueError as e:
        if label is None:
            raise ValueError(f"{fit}.{e}") from None
        read = ", ".join(f"dataset {a!r} read as {b!r}" for a, b in swap.items())
        raise ValueError(f"{fit}: eval {label!r}, with {read} (a row is the eval's dataset's): {e}") from None


def prepare(engine: Engine, plan: Plan, steps: dict, swap: dict[str, str], classes: dict[str, list],
            aliases: dict[str, str]) -> Found:
    """`preflight`'s checks and findings for `steps`, in order (a fit's body prepared at the fit's place, over its
    training rows and under each eval), over the rows `plan` holds: the root's, or a body's. `swap` names each
    dataset as the run reads it (an eval substitution's `{"train": "test"}` records `train.input` as `test.input`
    and the records' positions under `test`); `classes` is each probe's class list; `aliases` a body's names for
    the featurizers its fit trains."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    found = Found(swap=swap)
    for name, entry in steps.items():
        for step in points(entry):
            if step.kind == "fit":
                train = body(engine, plan, name, step, None, {}, classes)
                put(found.bodies, name, step.at, {"train": train, "evals": {
                    label: body(engine, plan, name, step, label, other, classes) if other else train
                    for label, other in step.eval.items()}})
                continue
            if step.kind == "op":
                args, of = arguments(step), record_of(plan.steps, step.of, step.at)
                assert step.dims is not None and of.dims is not None
                step.layers, step.kept = carried(of, step.dims)         # completed from its producer's
                for key, (written, theirs) in operands(plan, step.op, args, step.at).items():
                    widths_ = (width_of(plan.steps, of, step.at, swap), width_of(plan.steps, theirs, step.at, swap))
                    why = mismatch(of, theirs, step.op, key, written, repr(step.of), False, False, widths_)
                    if why:
                        raise ValueError(f"{name}: operand {why}")
                if "classes" in of.dims:
                    check_classes(plan, name, step.op, args, of.space[-1]["featurizer"], classes, swap)
                else:
                    check_tokens(engine, plan, found, name, step.op, args, of.dims, *placed(plan, of, step.at, found))
                bundle = ops.DECLARED[step.op]["bundle"]
                for field in featurizers.KINDS[bundle].counts if bundle else ():
                    width = len(of.kept["hidden_size"]) if "hidden_size" in of.kept else sizes["hidden_size"]
                    if args[field] > width:
                        raise ValueError(f"{name}: {step.op} {field} {args[field]} counts directions of a "
                                         f"value {width} wide, which has no more than {width}")
                if step.op == "pca" and of.read is not None:
                    got = resolved(plan.steps, of.read, step.at, swap)
                    n = sum(len(p) for p in got if isinstance(p, list)) if got is not None else None
                    if n is not None and n <= args["k"]:
                        raise ValueError(f"{name}: pca of {n} vectors has no rank-{args['k']} basis; "
                                         f"it needs more than {args['k']}")
                continue
            rows = held_rows(plan, step.dataset)
            source = data.input_name(step.input)
            if source not in found.inputs:
                encoded = []
                for i, r in enumerate(rows):
                    try:
                        encoded.append(data.encode(tokenizer, data.field(r, step.input),
                                                   plan.data[step.dataset].template))
                    except ValueError as e:
                        raise ValueError(f"{name}: {source} row {i}: {e}") from None
                found.inputs[source] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
                plan.run.tokens[data.input_name(step.input, swap)] = \
                    [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
            encoded = found.inputs[source]["rows"]
            for key, record in entries(step).items():
                where, site = f"{name}.{key}", cast(Site, record.site)
                layers, select = cast("Selector | None", site.layers), cast("dict[str, Selector] | None", site.select)
                record.layers = None if layers is None else picked(where, layers, sizes["num_layers"])
                for axis, index in (select or {}).items():
                    shortest = min(len(e["ids"]) for e in encoded)
                    size, what = (shortest, f"the {shortest} tokens of the input's shortest row") \
                        if axis == "key" else (sizes[axis], f"the model's {sizes[axis]}")
                    selected(where, axis, index, size, what)
                record.kept = selectors.kept(select, sizes)
                pos = positions.parse(record.pos)
                resolved_ = None
                if positions.prefill(pos) or step.kind == "forward":    # a forward's one call is its prefill
                    resolved_ = indices(where, record, step, rows, encoded, tokenizer)
                    record.positions = {**record.positions, swap.get(step.dataset, step.dataset): resolved_}
                probe = None                        # past a probe's encode, a column names its classes
                for o in record.ops:
                    if probe is None:
                        forced = [e["ids"] for e in encoded] if (pos["scope"] or {}).get("before") else None
                        check_tokens(engine, plan, found, where, o.op, arguments(o), None, resolved_, forced)
                    else:
                        check_classes(plan, where, o.op, arguments(o), probe, classes, swap)
                    if o.op == "encode" and kind_of(plan, arguments(o)["featurizer"]) == "linear":
                        probe = arguments(o)["featurizer"]
            for key, record in entries(step).items():   # the operands, once every entry of the step has its labels
                where, edits = f"{name}.{key}", len(record.ops) - len(gathers(record))
                for i, o in enumerate(record.ops):
                    ours = met(record, i, aliases)
                    for k, (written, theirs) in operands(plan, o.op, arguments(o), step.at).items():
                        forward = written.partition(".")[0] == name
                        widths_ = (width_of(plan.steps, ours, step.at, swap), width_of(plan.steps, theirs, step.at, swap))
                        why = mismatch(ours, theirs, o.op, k, written, repr(step.input), i < edits, forward, widths_)
                        if why:
                            raise ValueError(f"{where}: operand {why}")
                    if i < edits:
                        check_operands(engine, plan, where, o, step, record, found)
    return found


def placed(plan: Plan, of: Labelled, at: dict, found: Found) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    """The positions per row a value holds (its read's, `resolved`), where they are known before the run, else
    None; and each row's input ids where that read is a scope that is `before` (whose next tokens an answer
    scores), else None."""
    if of.read is None:
        return None, None
    where, read = resolved(plan.steps, of.read, at, found.swap), of.read
    if where is None:
        return None, None
    step, record = entry_of(plan.steps, read, at)
    scope = positions.parse(record.pos)["scope"] or {}
    return where, [e["ids"] for e in found.inputs[data.input_name(step.input)]["rows"]] \
        if scope.get("before") else None
