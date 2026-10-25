"""Everything that needs the model, checked before any forward: what the run needs found, and what it
cannot run refused by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import check_classes, check_tokens, placed, probe_of
from clab.compile.interventions import check_labels, check_layers, check_operands, held, refuse
from clab.compile.positions import indices
from clab.compile.state import Found
from clab.compile.uses import widths
from clab.format import address, data, selectors
from clab.format.selectors import made_at, picked, selected
from clab.format.sweeps import lookup, put
from clab.plan import Fit, Plan, arguments, body_view, dims_of, entries, gathers, kind_of, points

if TYPE_CHECKING:
    from clab.engines import Engine


def preflight(engine: Engine, plan: Plan) -> Found:
    """Everything that needs the model, checked before any forward. Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs: each dataset input's encoded rows and padded width (`inputs`),
    every prefill record's positions per row (`positions`, which the run extends with the other records'),
    the indices every record's `select` keeps on each axis (`selects`, resolved here once, among the engine's
    sizes), room for what each generate step's rows said (`said`), each featurizer's width (`widths`, the root's
    alone), and for each fit point the same for its body (`bodies`): over its training rows (`train`) and under
    each of its named `eval` substitutions (`evals`), each beside the plan the body runs in, which holds the
    body's steps and each substituted dataset's rows under the body's name for it.

    Refuses by name (`<step>.<entry>`): a component the engine does not serve; a layer or a `select` outside the
    model's sizes (on `key`, outside the input's shortest row), or a `select` that keeps none of them; a position that does not fit some
    row's scope, or whose scope is not in a row a write or a read of static width needs; a token column that is not
    one token on some row; an op whose operand has a `layers` axis without every layer its entry is made at
    (`check_layers`); a write whose operand was selected at a different width or narrowing, or holds other
    indices of an axis the write's site narrows (`check_labels`: heads in another order, one head for several); a
    write to a component with a `key` axis (the attention pattern, the scores) whose operand was read on another
    input that does not tokenize to the same length on every row, which would misalign the keys, and a window of
    several positions that reaches past the tokens two such inputs share (`windowed`); a `pca` over no more
    vectors than its `k`; a featurizer used at two widths (at a site, the site's; inside an `encode`, that one's
    coordinates'), at a width its bundle's header does not say (or, with no header, whose tensors are not as
    wide), with a field that counts more directions than that width (`counts`), at another layer than the one
    it says it was trained at, reading its input elsewhere than its bundle was made reading it (`read_with`),
    or encoded with a `select` outside its features or keeping none of them; and a trained featurizer used at
    no site. A refusal inside a fit's body names the fit, and under an eval the eval and the datasets it reads.
    """
    refuse(engine, plan)
    rows = {n: d.rows or [] for n, d in plan.data.items()}
    classes = {name: featurizers.labels(rows, (spec.model_extra or {})["classes"])
               for name, entry in plan.featurizers.items() for spec in points(entry)[:1] if spec.kind == "linear"}
    found = prepare(engine, plan, plan.steps, {}, classes)
    found.widths = widths(engine, plan)
    for name, entry in plan.steps.items():
        for step in points(entry):
            if step.kind == "fit":
                train = body(engine, plan, name, step, None, {}, classes)
                put(found.bodies, name, step.at, {"train": train, "evals": {
                    label: body(engine, plan, name, step, label, swap, classes) if swap else train
                    for label, swap in step.eval.items()}})
    return found


def body(engine: Engine, plan: Plan, fit: str, step: Fit, label: str | None, swap: dict[str, str],
         classes: dict[str, list]) -> tuple[Plan, Found]:
    """A fit body's plan (under eval `label`, with each dataset `swap` maps read under the body's name for it) and
    its `prepare`; a refusal names the fit and, under an eval, which one and the datasets it reads in place of
    the body's, whose rows a row number counts."""
    view = body_view(plan, fit, step.at, swap)
    try:
        return view, prepare(engine, view, step.steps, swap, classes)
    except ValueError as e:
        if label is None:
            raise ValueError(f"{fit}.{e}") from None
        read = ", ".join(f"dataset {a!r} read as {b!r}" for a, b in swap.items())
        raise ValueError(f"{fit}: eval {label!r}, with {read} (a row is the eval's dataset's): {e}") from None


def prepare(engine: Engine, plan: Plan, steps: dict, swap: dict[str, str], classes: dict[str, list]) -> Found:
    """`preflight`'s checks and findings for `steps` (a fit is prepared apart), over the rows `plan` holds:
    the root's, or a body's. `swap` names each dataset as the run record knows it (an eval substitution's
    `{"train": "test"}` records `train.input` as `test.input`); `classes` is each probe's class list."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    found = Found()
    shapes: dict = {}
    for name, entry in steps.items():
        for step in points(entry):
            if step.kind == "fit":
                continue
            if step.kind == "op":
                args, ours, dims = arguments(step), held(plan, step.of, step.at, sizes), dims_of(plan, step.of)
                check_labels(plan, name, step, step.at, ours, dims, sizes, False)
                if "classes" in dims:
                    check_classes(plan, name, step.op, args, probe_of(plan, step.of), classes, swap)
                else:
                    check_tokens(engine, plan, found, name, step.op, args, dims, *placed(plan, found, step.of, step.at))
                bundle = ops.DECLARED[step.op]["bundle"]
                for field in featurizers.KINDS[bundle].counts if bundle else ():
                    width = (ours or {}).get("hidden_size", range(sizes["hidden_size"]))
                    if args[field] > len(width):
                        raise ValueError(f"{name}: {step.op} {field} {args[field]} counts directions of a "
                                         f"value {len(width)} wide, which has no more than {len(width)}")
                if step.op == "pca" and step.of in found.positions:
                    n = sum(map(len, lookup(found.positions, step.of, step.at)))
                    if n <= args["k"]:
                        raise ValueError(f"{name}: pca of {n} vectors has no rank-{args['k']} basis; "
                                         f"it needs more than {args['k']}")
                continue
            rows = plan.data[step.dataset].rows or []
            source = data.input_name(step.input)
            if source not in found.inputs:
                encoded = []
                for i, r in enumerate(rows):
                    try:
                        encoded.append(data.encode(tokenizer, data.field(r, step.input)))
                    except ValueError as e:
                        raise ValueError(f"{name}: {source} row {i}: {e}") from None
                found.inputs[source] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
                plan.run.tokens[data.input_name(step.input, swap)] = \
                    [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
            encoded = found.inputs[source]["rows"]
            later = []                      # a write's operands, checked once the step's entries have their shapes
            for key, record in entries(step).items():
                where, site = f"{name}.{key}", record.site
                if site.layers is not None:
                    picked(where, site.layers, sizes["num_layers"])
                for axis, index in (site.select or {}).items():
                    shortest = min(len(e["ids"]) for e in encoded)
                    size, what = (shortest, f"the {shortest} tokens of the input's shortest row") \
                        if axis == "key" else (sizes[axis], f"the model's {sizes[axis]}")
                    selected(where, axis, index, size, what)
                kept = selectors.kept(site.select, sizes)
                put(found.selects, where, step.at, kept)
                pos, select = positions.parse(record.pos), {a: len(v) for a, v in kept.items()}
                resolved = None
                if positions.prefill(pos) or step.kind == "forward":    # a forward's one call is its prefill
                    resolved = indices(where, record, step, rows, encoded, tokenizer)
                    put(found.positions, where, step.at, resolved)
                    widths = [len(r) for r in resolved]
                    shape = (widths[0] if len(set(widths)) == 1 else tuple(widths), select)
                else:               # resolved in the run, from what each row generates; one position a call
                    shape = (positions.width(pos) or 1, select)
                probe = None                        # past a probe's encode, a column names its classes
                for o in record.ops:
                    if probe is None:
                        forced = [e["ids"] for e in encoded] if (pos["scope"] or {}).get("before") else None
                        check_tokens(engine, plan, found, where, o.op, arguments(o), None, resolved, forced)
                    else:
                        check_classes(plan, where, o.op, arguments(o), probe, classes, swap)
                    if o.op == "encode" and kind_of(plan, arguments(o)["featurizer"]) == "linear":
                        probe = arguments(o)["featurizer"]
                put(shapes, where, step.at, shape)
                for o in record.ops:
                    check_layers(plan, where, o, step.at, made_at(site, sizes["num_layers"]), sizes["num_layers"])
                later += [(where, o, shape, site) for o in record.ops[:len(record.ops) - len(gathers(record))]]
            for where, o, shape, site in later:
                check_operands(engine, plan, where, o, shape, step, site, found, shapes)
            for key, record in entries(step).items():
                where, edits = f"{name}.{key}", len(record.ops) - len(gathers(record))
                for i, o in enumerate(record.ops):
                    check_labels(plan, where, o, step.at, held(plan, where, step.at, sizes),
                                 address.dims(record.site.component), sizes, i < edits)
    return found
