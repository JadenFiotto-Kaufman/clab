"""Saved values as tables: a row per example and position, labelled with the tokens the run recorded."""

import itertools
import json
from typing import Any, cast

from torch import Tensor

from clab import ops
from clab.compile import dataset_rows, positions
from clab.format import data
from clab.format.sweeps import agreeing_point, coordinate_text, projection
from clab.plan import (Intervention, Labelled, OpStep, Plan, arguments, by_point, entries, entry_of, made_by, points,
                        projections, record_of, runnable)

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

    Several values join on the keys they share, the sweep axes by coordinate and the axis columns (`row`, `layer`,
    `pos`, `key`; a value with no rows axis its own axes, under the same names), each row of the one with more keys
    beside every row of the other that agrees with it on them, an axis one value lacks broadcast over as an operand
    broadcasts over an axis it leaves out (an unswept number sits beside every row, a swept one beside every row at
    each of its points, a history over `epochs` beside every row at each epoch), the first value's columns first
    and the joined `at` both points' coordinates; a shared `pos` or `key` that does not mean one thing for both,
    counting the positions of different inputs or a plain index beside a token position, is refused by name.
    """
    from clab.output import run_record

    run = run_record(plan)
    by_dataset: dict[str, str] = {}             # the dataset each value has a row per example of, as its record says
    for name in names:
        if name not in plan.saves:
            raise ValueError(f"{name!r} was not saved; add it to saves")
        if name not in plan.results:
            raise ValueError(f"{name!r} has no result: the run stopped before its step; {run.stopped}")
        record = record_of(plan.steps, name, points(plan.steps[name.partition(".")[0]])[0].at)
        if "rows" in (record.dims or []) and record.dataset is not None:
            by_dataset[record.dataset] = name
    if len(by_dataset) > 1:
        (a, x), (b, y) = list(by_dataset.items())[:2]
        raise ValueError(f"{x!r} has a row per example of {a!r} and {y!r} of {b!r}; a table joins the values of "
                         "one dataset")
    if len(names) == 1:
        return rows_of(plan, names[0], columns)
    joined: list[dict[str, Any]] = []
    keys: tuple[list[str], list[str], str | None] = ([], [], None)
    for i, name in enumerate(names):
        rows = [{(name if k == "value" else k): v for k, v in row.items()} for row in rows_of(plan, name, columns)]
        mine = row_keys(plan, name)
        if i:
            joined = join(joined, keys, rows, mine, " and ".join(map(repr, names[:i])), name)
        else:
            joined = rows
        keys = (keys[0] + [k for k in mine[0] if k not in keys[0]], keys[1] + [k for k in mine[1] if k not in keys[1]],
                keys[2] or mine[2])
    return joined


def row_keys(plan: Plan, name: str) -> tuple[list[str], list[str], str | None]:
    """What a saved value's rows are keyed by, as its record says: the labels of the sweep axes it depends on (its
    `axes`), and its axis columns (`COLUMNS`): for a value with a rows axis `row`, and `layer`, `pos` and `key` where it has
    those axes; for one without, each of its axes under the same names. With a token axis, also what its `pos` and
    `key` count: the positions of the input its read was made on, or, with no rows axis, a plain index along it."""
    first = points(plan.steps[name.partition(".")[0]])[0]
    record = record_of(plan.steps, name, first.at)
    dims = record.dims or []
    axes = [COLUMNS.get(a, a) for a in dims if "rows" not in dims or a in COLUMNS]
    counts = None
    if "pos" in dims or "key" in dims:
        counts = "a plain index along it (a value with no rows axis)" if "rows" not in dims else \
            f"positions of {data.input_name(entry_of(plan.steps, cast(str, record.read), first.at)[0].input)}"
    return list(record.axes), axes, counts


def join(left: list[dict[str, Any]], mine: tuple[list[str], list[str], str | None], right: list[dict[str, Any]],
         theirs: tuple[list[str], list[str], str | None], x: str, y: str) -> list[dict[str, Any]]:
    """`left` (the rows of the values `x`, keyed by `mine`: sweep labels, axis columns and what a token axis counts,
    `row_keys`) beside `right` (the rows of `y`, keyed by `theirs`), joined on the keys both carry, as `table` says;
    `x` and `y` name them in the refusal."""
    labels, axes = [k for k in mine[0] if k in theirs[0]], [k for k in mine[1] if k in theirs[1]]
    tokens = [a for a in axes if a in ("pos", "key")]
    if tokens and mine[2] != theirs[2]:
        raise ValueError(f"{x} has a {' and '.join(tokens)} column counting {mine[2]}, and {y!r} one counting "
                         f"{theirs[2]}: the column does not mean one thing for both, so they do not join; table them "
                         "one at a time")

    def key(row: dict[str, Any]) -> str:
        return json.dumps([row.get("at", {}).get(k) for k in labels] + [row.get(k) for k in axes])
    # The value with more keys (a swept one beside an unswept one) gives the rows their order; the left's columns
    # come first either way.
    many, few = (left, right) if len(mine[0]) + len(mine[1]) >= len(theirs[0]) + len(theirs[1]) else (right, left)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in few:
        by_key.setdefault(key(row), []).append(row)
    out = []
    for row in many:
        for other in by_key.get(key(row), []):
            a, b = (row, other) if many is left else (other, row)
            merged = {**a, **b}
            if "at" in merged:              # the two points' coordinates together, first
                merged = {"at": {**a.get("at", {}), **b.get("at", {})},
                          **{k: v for k, v in merged.items() if k != "at"}}
            out.append(merged)
    return out


def rows_of(plan: Plan, name: str, columns: list[str] | None) -> list[dict[str, Any]]:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if step.kind == "fit" and member.partition(".")[0] in ("history", *step.train):
        dims = record_of(plan.steps, name, {}).dims or []
        if member in step.train:
            raise ValueError(f"{name!r} is a trained featurizer's bundle, which has no rows to table; read it "
                             "from plan.results")
        return [{"at": at, **row} if at else row for at, value in by_point(plan, name)
                for row in value_rows(value, dims, None)]
    out = []
    for (_, p), (at, value) in zip(projections(plan, name), by_point(plan, name)):
        record = record_of(plan.steps, name, p.at)
        dims = record.dims or []
        if isinstance(value, dict) and "rows" not in dims:
            what = (f"a bundle (a {record.op}'s value)" if ops.DECLARED[record.op]["bundle"] else
                    f"laid out {dims}, a {record.op} over a value with no rows axis, its ids and probabilities one "
                    "list each") if isinstance(record, OpStep) else f"laid out {dims}"
            raise ValueError(f"{name!r} has no rows to table: it is {what}; read it from plan.results")
        examples = dataset_rows(plan, record.dataset) if record.dataset else []
        place = read_labels(plan, record, p.at) if record.read else None
        used = columns_used(plan, name, p.at)
        for row in value_rows(value, dims, place):
            if "row" in row and examples:
                example = examples[row["row"]]
                row = {**({"example_id": example["example_id"]} if "example_id" in example else {}), **row,
                       **{c: example[c] for c in (columns if columns is not None else used)}}
            last = {"value": row.pop("value")} if "value" in row else {}
            out.append({**({"at": at} if at else {}), **row, **last})       # the value last, as in a joined table
    return out


def value_rows(value: Tensor | list[Any] | dict[str, Any], dims: list[str], place: dict[str, Any] | None
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
    layers = place["layers"] if place else None             # the model's layer at each index of a layers axis
    out = []
    for index in itertools.product(*(range(shape[i]) for i in outer)):
        whole = {COLUMNS.get(dims[i], dims[i]): layers[k] if dims[i] == "layers" and layers is not None else k
                 for i, k in zip(outer, index)}
        if place and "row" in whole and place["positions"][whole["row"]] == positions.MISSING:
            out.append({**whole, positions.MISSING: True})      # one row says so, in place of its tokens
            continue
        for within in itertools.product(*(range(shape[i]) for i in inner)):
            at = {**dict(zip(outer, index)), **dict(zip(inner, within))}
            row: dict[str, Any] = dict(whole)
            for i, k in zip(inner, within):
                row.update(token(place, at[dims.index("rows")], dims[i], k, row.get("pos")) if place else {dims[i]: k})
            if row.get("key", 0) is None or row.get("pos", 0) is None:
                continue                                    # past the row's own keys or positions
            pick = tuple(at.get(i, slice(None)) for i in range(len(dims)))
            row["value"] = {k: part(v, pick) for k, v in value.items()} if isinstance(value, dict) else part(value, pick)
            out.append(row)
    return out


def token(place: dict[str, Any], row: int, axis: str, k: int, at: int | None = None) -> dict[str, Any]:
    """Index `k` on a token axis as the token it is: on `pos` through the read's resolved positions, on `key`
    through its `select` or, unselected, the row's own key `k`, each a position of the row as `pos` counts them, and
    a key relative to the query at its offset from the position `at` the row is labelled with (its `offset` alone
    where the value has no `pos` axis left); None past the row's own positions or keys."""
    if axis == "pos":
        return position_label(place, row, place["positions"][row][k], axis) if k < len(place["positions"][row]) \
            else {axis: None}
    if isinstance(place["keys"], dict):
        if at is None:
            return {"offset": place["keys"]["offset"][k]}
        index = at + place["keys"]["offset"][k]
        return position_label(place, row, index, axis) if 0 <= index < len(place["prompt"][row]) else {axis: None}
    index = place["keys"][k] if place["keys"] is not None else k
    return position_label(place, row, index, axis) if index < len(place["prompt"][row]) else {axis: None}


def position_label(place: dict[str, Any], row: int, index: int, axis: str) -> dict[str, Any]:
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


def argument_columns(plan: Plan, args: dict[str, Any]) -> list[str]:
    """The dataset columns among an op's arguments (a column reference, as the string or the object), by name."""
    refs = [data.column_of(v) for v in args.values()]
    return [r[0].partition(".")[2] for r in refs if r is not None and r[0].partition(".")[0] in plan.data]


def columns_used(plan: Plan, name: str, at: dict[str, int]) -> list[str]:
    """The dataset columns a value's chain of steps used, at the point agreeing with `at`: those its op steps'
    arguments name, back to the read (a load's, the one it carries; none where it carries none), then the read's
    input columns first and the columns its entry's ops name."""
    used: list[str] = []
    steps = runnable(plan.steps)
    while True:
        made = made_by(plan, name)          # an eval's member's: its eval step
        step, member = agreeing_point(steps[made], at), name[len(made) + 1:]
        if step.kind == "op":
            used += argument_columns(plan, arguments(step))
            name = step.of
            continue
        if step.kind == "load":             # the read it carries, as the run that saved it made it
            if step.source is None:
                return list(dict.fromkeys(used))
            step, member = step.source, cast(str, step.read).partition(".")[2]
        used[:0] = [data.parse_input(c)[1] for c in data.input_columns(step.input)]
        for o in entries(step)[member].ops if member else []:
            used += argument_columns(plan, arguments(o))
        return list(dict.fromkeys(used))


def read_labels(plan: Plan, record: Labelled, at: dict[str, int]) -> dict[str, Any]:
    """For a value's read (`record.read`) at the point agreeing with `at`: the rows' prompt tokens and their
    positions' tokens (the prompt's, then in a generation what each row generated), whether it was read in a
    generation (whose positions are labelled with their call), the positions each row resolved to (the read's
    record's, under the value's dataset), the keys it selects, and the model layers the value's `layers` axis is
    labelled with, where it has one."""
    assert plan.run is not None and record.read is not None
    producer, _ = entry_of(plan.steps, record.read, at)
    read = cast(Intervention, record_of(plan.steps, record.read, at))
    prompt = plan.run.tokens[data.input_name(producer.input)]
    said = plan.run.tokens.get(record.read.rpartition(".")[0]) if producer.kind == "generate" else None
    if said is not None and projection(producer.at, cast(Labelled, producer).axes):
        said = said[coordinate_text(plan, projection(producer.at, cast(Labelled, producer).axes))]
    return {"prompt": prompt, "tokens": [p + s for p, s in zip(prompt, said)] if said else prompt,
            "beyond": producer.kind == "generate",
            "keys": read.kept.get("key"),
            "positions": read.positions[cast(str, read.dataset)],
            "layers": record.layers if "layers" in (record.dims or []) else None}
