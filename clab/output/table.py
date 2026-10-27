"""Saved values as tables: a row per example and position, labelled with the tokens the run recorded."""

import itertools
import json
from typing import Any, cast

from torch import Tensor

from clab import ops
from clab.compile import materialized, positions
from clab.format import data
from clab.format.sweeps import agreeing, by_point, labelled
from clab.plan import Intervention, Labelled, Plan, arguments, body_view, entries, entry_of, points, record_of, walk

# The column a value's example and token axes are tabled under.
COLUMNS = {"rows": "row", "layers": "layer", "pos": "pos", "key": "key"}


def table(plan: Plan, names: tuple[str, ...], columns: list[str] | None) -> list[dict[str, Any]]:
    """The saved values `names` as rows, joined wide on the keys they share, one column per name.

    A value with a rows axis gives a row per example and token: `at` (its point's coordinates; a value whose step is
    not swept has no `at`), `row`, `example_id` when the dataset has one, `layer` on a `layers` axis (the model's
    layer), the token axes with their tokens (`pos`/`pos_token`, the position a read resolved to, as a document's
    `pos` counts them: the index into the row's prompt tokens, BOS first, then in a generation its generated ones,
    with the forward `call` holding it for a read in a generation; `key`/`key_token` on an attention pattern,
    numbered the same way, the row's own keys only), the dataset columns the step used
    (`columns=` to choose), and `value` last: the element, or the remaining axes as nested lists. A row a read
    did not reach (a call beyond the prefill it never made, a scope it does not say, an index that picks none of
    its tokens) gives one row, `alignment_missing` true, with no value; a row's positions past its own are left
    out. A value with no rows axis gives a row per element with its axis indices.

    Several values are joined on the keys both carry (`keyed`): the sweep axes they share, each by its coordinate,
    and the columns they share (`row`, `layer`, `pos`, `key`; a value with no rows axis its own axes, so two
    histories over `epochs` join epoch by epoch), each row of the first beside every row of the next that agrees
    with it on them, so a row of a value with fewer keys (an unswept value beside a swept one, one over `src`
    beside one over `src` and `dst`, one with no `layers` axis beside one with) repeats beside every row it
    agrees with, and the joined `at` holds both points' coordinates. A value with no keys, one number, sits beside
    every row. Two values that share no key are refused by name: a value with no rows axis (a fit's history, over
    `epochs`) tables alone beside one with, and so do two over different axes; and two that share keys but agree
    on no row (two reads at different positions share `pos`) are refused rather than joined to nothing, as are two
    with a token axis read on different inputs, whose `pos` or `key` count different prompts."""
    for name in names:
        if name not in plan.saves:
            raise ValueError(f"{name!r} was not saved; add it to saves")
    found = {d: name for name in names if (d := dataset_of(plan, name)) is not None}
    if len(found) > 1:
        (a, x), (b, y) = list(found.items())[:2]
        raise ValueError(f"{x!r} has a row per example of {a!r} and {y!r} of {b!r}; a table joins the values of "
                         "one dataset")
    if len(names) == 1:
        return rows_of(plan, names[0], columns)
    joined: list[dict[str, Any]] = []
    keys: tuple[list[str], list[str], str | None] = ([], [], None)
    for i, name in enumerate(names):
        rows = [{(name if k == "value" else k): v for k, v in row.items()} for row in rows_of(plan, name, columns)]
        mine = keyed(plan, name)
        if i:
            joined = join(joined, keys, rows, mine, " and ".join(map(repr, names[:i])), name)
        else:
            joined = rows
        keys = (keys[0] + [k for k in mine[0] if k not in keys[0]], keys[1] + [k for k in mine[1] if k not in keys[1]],
                keys[2] or mine[2])
    return joined


def keyed(plan: Plan, name: str) -> tuple[list[str], list[str], str | None]:
    """What a saved value's rows are keyed by, as its record says: the labels of its step's sweep axes (its `at`),
    and its axis columns: for a value with a rows axis `row`, and `layer`, `pos` and `key` where it has those axes
    (`COLUMNS`); for one without, each of its axes. With a token axis, also the input its read was made on, whose
    prompts its `pos` and `key` count."""
    first = points(plan.steps[name.partition(".")[0]])[0]
    record = record_of(plan.steps, name, first.at)
    dims = record.dims or []
    axes = [COLUMNS[a] for a in dims if a in COLUMNS] if "rows" in dims else list(dims)
    source = data.input_name(entry_of(plan.steps, record.read, first.at)[0].input) \
        if record.read is not None and ("pos" in dims or "key" in dims) else None
    return list(first.at), axes, source


def join(left: list[dict[str, Any]], mine: tuple[list[str], list[str], str | None], right: list[dict[str, Any]],
         theirs: tuple[list[str], list[str], str | None], x: str, y: str) -> list[dict[str, Any]]:
    """`left` (the rows of the values `x`, keyed by `mine`: sweep labels, axis columns and the input a token axis
    counts, `keyed`) beside `right` (the rows of `y`, keyed by `theirs`), joined on the keys both carry, as `table`
    says; `x` and `y` name them in a refusal."""
    labels, axes = [k for k in mine[0] if k in theirs[0]], [k for k in mine[1] if k in theirs[1]]
    if not (labels or axes) and any(mine[:2]) and any(theirs[:2]):
        alone = ("a value with no rows axis tables alone" if ("row" in mine[1]) != ("row" in theirs[1])
                 else "table them one at a time")
        raise ValueError(f"{x} is keyed by {spelled(mine)} and {y!r} by {spelled(theirs)}, which share no axis "
                         f"to join on; {alone}")
    tokens = [a for a in axes if a in ("pos", "key")]
    if tokens and None not in (mine[2], theirs[2]) and mine[2] != theirs[2]:
        raise ValueError(f"{x} is read on {mine[2]} and {y!r} on {theirs[2]}, whose rows tokenize to positions of their "
                         f"own, so the two {' and '.join(tokens)} columns count different prompts; table them one at a "
                         "time")

    def key(row: dict[str, Any]) -> str:
        return json.dumps([row.get("at", {}).get(k) for k in labels] + [row.get(k) for k in axes])
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in right:
        by_key.setdefault(key(row), []).append(row)
    out = []
    for row in left:
        for other in by_key.get(key(row), []):
            merged = {**row, **other}
            if "at" in merged:              # the two points' coordinates together, first
                merged = {"at": {**row.get("at", {}), **other.get("at", {})},
                          **{k: v for k, v in merged.items() if k != "at"}}
            out.append(merged)
    if not out and left and right:
        raise ValueError(f"{x} and {y!r} share {', '.join(labels + axes)} and agree on no row of it (two reads at "
                         "different positions share pos but never the same one); table them one at a time")
    return out


def spelled(keys: tuple[list[str], list[str], str | None]) -> str:
    """A value's keys as a message names them: `sweep axes src, dst and columns row, pos`."""
    labels, axes, _ = keys
    parts = [f"sweep axes {', '.join(labels)}"] if labels else []
    return " and ".join(parts + ([f"columns {', '.join(axes)}"] if axes else []))


def rows_of(plan: Plan, name: str, columns: list[str] | None) -> list[dict[str, Any]]:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if step.kind == "fit":
        if member.partition(".")[0] in ("history", *step.train):
            dims = record_of(plan.steps, name, {}).dims or []
            if member in step.train:
                raise ValueError(f"{name!r} is a trained featurizer's bundle, which has no rows to table; read it "
                                 "from plan.results")
            return [{"at": at, **row} if at else row for at, value in by_point(plan, name)
                    for row in elements(value, dims, None)]
        return rows_of(evaluated(plan, thing, name), member.partition(".")[2], columns)
    out = []
    for p, (at, value) in zip(points(plan.steps[thing]), by_point(plan, name)):
        record = record_of(plan.steps, name, p.at)
        dims = record.dims or []
        if isinstance(value, dict) and "rows" not in dims:
            what = (f"a bundle (a {step.op}'s value)" if ops.DECLARED[step.op]["bundle"] else
                    f"laid out {dims}, a {step.op} over a value with no rows axis, its ids and probabilities one list each"
                    ) if step.kind == "op" else f"laid out {dims}"
            raise ValueError(f"{name!r} has no rows to table: it is {what}; read it from plan.results")
        examples = materialized(plan, record.dataset) if record.dataset else []
        place = located(plan, record, p.at) if record.read else None
        used = columns_used(plan, name, p.at)
        for row in elements(value, dims, place):
            if "row" in row and examples:
                example = examples[row["row"]]
                row = {**({"example_id": example["example_id"]} if "example_id" in example else {}), **row,
                       **{c: example[c] for c in (columns if columns is not None else used)}}
            last = {"value": row.pop("value")} if "value" in row else {}
            out.append({**({"at": at} if at else {}), **row, **last})       # the value last, as in a joined table
    return out


def dataset_of(plan: Plan, name: str) -> str | None:
    """The dataset a saved value has a row per example of, if any, as its record says (a member of a fit's named
    eval: the one its body's step read under that eval)."""
    record = record_of(plan.steps, name, points(plan.steps[name.partition(".")[0]])[0].at)
    return record.dataset if "rows" in (record.dims or []) else None


def evaluated(plan: Plan, fit: str, name: str) -> Plan:
    """The plan as the fit's named eval saw it, for tabling its member `name` (`<fit>.<eval>.<member>`): its
    body's view under that eval's substitution, the run record of the body's inputs under the names the body
    gives them, and the body's records with their positions under those names too (the eval's dataset read as
    the body's)."""
    label, _, member = name.partition(".")[2].partition(".")
    swap = points(plan.steps[fit])[0].eval[label]
    assert plan.run is not None
    tokens = dict(plan.run.tokens)
    for _, step in walk(points(plan.steps[fit])[0].steps):
        if step.kind in ("forward", "generate") and data.input_name(step.input, swap) in plan.run.tokens:
            tokens[data.input_name(step.input)] = plan.run.tokens[data.input_name(step.input, swap)]
    view = body_view(plan, fit, swap=swap)

    def under(step: Any) -> Any:
        if step.kind not in ("forward", "generate"):
            return step
        read = swap.get(step.dataset, step.dataset)
        return step.model_copy(update={"interventions": {
            k: e.model_copy(update={"positions": {step.dataset: e.positions[read]} if read in e.positions else {}})
            for k, e in entries(step).items()}})
    body = {n: [under(p) for p in s] if isinstance(s, list) else under(s)
            for n, s in view.steps.items() if n in points(plan.steps[fit])[0].steps}
    return view.model_copy(update={"steps": {**view.steps, **body}, "run": plan.run.model_copy(update={"tokens": tokens}),
                                   "results": {member: plan.results[name]}, "saves": [member]})


def elements(value: Tensor | list[Any] | dict[str, Any], dims: list[str], place: dict[str, Any] | None
             ) -> list[dict[str, Any]]:
    """A value's rows: one per index on its example and token axes, the rest of it as the row's value; a row the
    read did not reach (`place` says so) is one row, `alignment_missing`, whatever its token axes hold, none
    included (a read no row reached is `[rows, 0]`)."""
    assert not isinstance(value, dict) or "rows" in dims, "a bundle has no rows to table (rows_of refuses it)"
    first = next(iter(value.values())) if isinstance(value, dict) else value
    shape = list(first.shape) if isinstance(first, Tensor) else [len(first)]
    # The axes a row is per: the example axes, then the token axes within each; every axis of a value with no rows.
    outer = [i for i, a in enumerate(dims) if a in ("layers", "rows") or "rows" not in dims]
    inner = [i for i, a in enumerate(dims) if a in ("pos", "key") and "rows" in dims]
    out = []
    for index in itertools.product(*(range(shape[i]) for i in outer)):
        whole = {COLUMNS.get(dims[i], dims[i]): layer(place, k) if dims[i] == "layers" else k
                 for i, k in zip(outer, index)}
        if place and "row" in whole and place["positions"][whole["row"]] == positions.MISSING:
            out.append({**whole, positions.MISSING: True})      # one row says so, in place of its tokens
            continue
        for within in itertools.product(*(range(shape[i]) for i in inner)):
            at = {**dict(zip(outer, index)), **dict(zip(inner, within))}
            row: dict[str, Any] = dict(whole)
            for i, k in zip(inner, within):
                row.update(token(place, at[dims.index("rows")], dims[i], k) if place else {dims[i]: k})
            if row.get("key", 0) is None or row.get("pos", 0) is None:
                continue                                    # past the row's own keys or positions
            pick = tuple(at.get(i, slice(None)) for i in range(len(dims)))
            row["value"] = {k: part(v, pick) for k, v in value.items()} if isinstance(value, dict) else part(value, pick)
            out.append(row)
    return out


def layer(place: dict[str, Any] | None, k: int) -> int:
    """Index `k` on a `layers` axis as the model's layer it is, by the value's own `layers` label."""
    return place["layers"][k] if place and place["layers"] is not None else k


def token(place: dict[str, Any], row: int, axis: str, k: int) -> dict[str, Any]:
    """Index `k` on a token axis as the token it is: on `pos` through the read's resolved positions, on `key`
    through its `select` or, unselected, the row's own key `k`, each a position of the row as `pos` counts them;
    None past the row's own positions or keys."""
    if axis == "pos":
        return label(place, row, place["positions"][row][k], axis) if k < len(place["positions"][row]) else {axis: None}
    index = place["keys"][k] if place["keys"] is not None else k
    return label(place, row, index, axis) if index < len(place["prompt"][row]) else {axis: None}


def label(place: dict[str, Any], row: int, index: int, axis: str) -> dict[str, Any]:
    """A row's position `index` as a table labels it: `pos` and `pos_token` (or `key`, `key_token`), the token
    that sits there, and for a read in a generation the forward `call` that holds it."""
    call = {"call": positions.call_of(index, len(place["prompt"][row]))} if place["beyond"] else {}
    return {axis: index, f"{axis}_token": place["tokens"][row][index], **call}


def part(value: Tensor | list[Any], pick: tuple[int | slice, ...]) -> Any:
    if isinstance(value, Tensor):
        got = value[pick]
        return got.item() if got.dim() == 0 else got.tolist()
    for i in pick:
        if isinstance(i, int):
            value = value[i]
    return value


def columns(plan: Plan, args: dict[str, Any]) -> list[str]:
    """The dataset columns among an op's arguments (a column reference, as the string or the object), by name."""
    refs = [data.column_of(v) for v in args.values()]
    return [r[0].partition(".")[2] for r in refs if r is not None and r[0].partition(".")[0] in plan.data]


def columns_used(plan: Plan, name: str, at: dict[str, int]) -> list[str]:
    """The dataset columns a value's chain of steps used, at the point agreeing with `at`: those its op steps'
    arguments name, back to the read, then the read's input columns first and the columns its entry's ops name."""
    used: list[str] = []
    steps = plan.steps
    while True:
        thing, _, member = name.partition(".")
        step = agreeing(steps[thing], at)
        if step.kind == "fit":
            steps, name, at = step.steps, member.partition(".")[2], step.at
            continue
        if step.kind == "op":
            used += columns(plan, arguments(step))
            name = step.of
            continue
        used[:0] = [data.parse_input(c)[1] for c in data.input_columns(step.input)]
        for o in entries(step)[member].ops if member else []:
            used += columns(plan, arguments(o))
        return list(dict.fromkeys(used))


def located(plan: Plan, record: Labelled, at: dict[str, int]) -> dict[str, Any]:
    """For a value's read (`record.read`) at the point agreeing with `at`: the rows' prompt tokens and their
    positions' tokens (the prompt's, then in a generation what each row generated), whether it was read in a
    generation (whose positions are labelled with their call), the positions each row resolved to (the read's
    record's, under the value's dataset), the keys it selects, and the model layers the value's `layers` axis is
    labelled with, where it has one."""
    assert plan.run is not None and record.read is not None
    producer, _ = entry_of(plan.steps, record.read, at)
    read = cast(Intervention, record_of(plan.steps, record.read, at))
    prompt = plan.run.tokens[data.input_name(producer.input)]
    said = plan.run.tokens.get(record.read.partition(".")[0]) if producer.kind == "generate" else None
    if said is not None and producer.at:
        said = said[labelled(plan, producer.at)]
    return {"prompt": prompt, "tokens": [p + s for p, s in zip(prompt, said)] if said else prompt,
            "beyond": producer.kind == "generate",
            "keys": read.kept.get("key"),
            "positions": read.positions[cast(str, read.dataset)],
            "layers": record.layers if "layers" in (record.dims or []) else None}
