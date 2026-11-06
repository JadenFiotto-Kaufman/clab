"""Saved values as tables: a row per example and position, labelled with the tokens the run recorded."""

import itertools
import json
import math
from typing import Any, cast

import torch
from torch import Tensor

from clab import ops
from clab.compile import dataset_rows, positions
from clab.format import data
from clab.format.sweeps import agreeing_point, coordinate, coordinate_text, projection
from clab.output import run_record, shape_or_dtype
from clab.plan import (Intervention, Labelled, OpStep, Plan, arguments, entries, entry_of, made_by, points, producer,
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
    numbered the same way, the row's own keys only; a `permute`'s value, whose rows hold other rows' positions, by
    plain index), the dataset columns the step used
    (`columns=` to choose), and `value` last: the element, or the remaining axes as nested lists. A row a read
    did not reach (a call beyond the prefill it never made, a scope it does not say, an index that picks none of
    its tokens) gives one row, `alignment_missing` true, with no value; a row's positions past its own are left
    out. A value with no rows axis gives a row per element with its axis indices, a `layers` axis by the model's
    layer and an axis a `select` keeps by the index kept; a `top_k` over `rows` (alone, or
    among a list of axes) a row per pick, its other axes' indices and `k` first, then the ranked value's row at the
    pick as its own table labels it, its `value` the ranked value there.

    Several values join on the keys they share, the sweep axes by coordinate and the axis columns (`row`, `layer`,
    `pos`, `key`; a value with no rows axis its own axes, under the same names), each row of the one with more keys
    beside every row of the other that agrees with it on them, an axis one value lacks broadcast over as an operand
    broadcasts over an axis it leaves out (an unswept number sits beside every row, a swept one beside every row at
    each of its points, a history over `epochs` beside every row at each epoch), the first value's columns first
    and the joined `at` both points' coordinates; a shared `pos` or `key` that does not mean one thing for both,
    counting the positions of different inputs or a plain index beside a token position, is refused by name.
    """
    run = run_record(plan)
    by_dataset: dict[str, str] = {}             # the dataset each value has a row per example of, as its record says
    for name in names:
        if name not in plan.saves:
            raise ValueError(f"{name!r} was not saved; add it to saves")
        if name not in plan.results:
            raise ValueError(f"{name!r} has no result: the run stopped before its step; {run.stopped}")
        record = record_of(plan.steps, name, points(producer(plan, name))[0].at, plan.featurizers)
        if len(names) > 1 and ranks_rows(record):
            raise ValueError(f"{name!r} is a top_k over rows, whose rows are its picks, and a top_k over rows tables "
                             "alone; table it by itself")
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
    `key` count: the positions of the input its read was made on, or, with no rows axis or no read (a permuted value's
    rows hold other rows' positions), a plain index along it."""
    first = points(producer(plan, name))[0]
    record = record_of(plan.steps, name, first.at, plan.featurizers)
    dims = record.dims or []
    axes = [COLUMNS.get(a, a) for a in dims if "rows" not in dims or a in COLUMNS]
    counts = None
    if "pos" in dims or "key" in dims:
        counts = "a plain index along it (a value with no rows axis)" if "rows" not in dims else \
            "a plain index along it (a permuted value, whose rows hold other rows' positions)" if record.read is None \
            else f"positions of {data.input_name(entry_of(plan.steps, record.read, first.at)[0].input)}"
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
            if "at" in merged:
                merged = {"at": {**a.get("at", {}), **b.get("at", {})},
                          **{k: v for k, v in merged.items() if k != "at"}}
            out.append(merged)
    return out


def by_point(plan: Plan, name: str) -> list[tuple[dict[str, Any], Any]]:
    """A saved value per point of its own axes: each point's coordinates (none for a value on no axis) and its
    value."""
    return [({label: plan.sweeps[label].values[i] for label, i in at.items()},
             plan.results[name][coordinate(plan, at)] if at else plan.results[name]) for at, _ in projections(plan, name)]


def rows_of(plan: Plan, name: str, columns: list[str] | None) -> list[dict[str, Any]]:
    thing, _, member = name.partition(".")
    step = points(producer(plan, name))[0]
    fitted = step.kind == "fit" and member.partition(".")[0] in ("history", *step.train)
    if fitted and member in step.train:
        raise ValueError(f"{name!r} is a trained featurizer's bundle, which has no rows to table; read it "
                         "from plan.results")
    out = []
    for (_, p), (at, value) in zip(projections(plan, name), by_point(plan, name)):
        record = record_of(plan.steps, name, p.at, plan.featurizers)
        dims = record.dims or []
        if fitted:              # a fit's history, over each point's own epochs
            out += value_rows(value, dims, None, at, [], [])
            continue
        if isinstance(value, dict) and "rows" not in dims and not ranks_rows(record):
            raise ValueError(f"{name!r} has no rows to table: it is {rowless(record, dims, list(value))}")
        used = columns_used(plan, name, p.at)
        if isinstance(value, dict) and "rows" not in dims:
            out += picks(plan, cast(OpStep, record), value, at, p, columns if columns is not None else used)
            continue
        examples = dataset_rows(plan, record.dataset) if record.dataset else []
        place = read_labels(plan, record, p.at) if record.read else None if "rows" in dims else {"indices": {
            **{a: v for a, v in record.kept.items() if isinstance(v, list) and v},       # a select's, as a read's
            **({"layers": record.layers} if "layers" in dims else {})}}
        out += value_rows(value, dims, place, at, examples, columns if columns is not None else used)
    return out


def ranks_rows(record: Labelled) -> bool:
    """Whether a value is a `top_k` over `rows`, alone or among a list of axes: one whose picks are rows."""
    over = arguments(record).get("over") if isinstance(record, OpStep) and record.op == "top_k" else None
    return "rows" in (over if isinstance(over, list) else [over])


def picks(plan: Plan, record: OpStep, value: dict[str, Any], at: dict[str, Any], point: Any, columns: list[str]
          ) -> list[dict[str, Any]]:
    """A `top_k` over `rows` (alone, or among a list of axes) as rows: one per pick that picks an entry, holding
    the value's other axes as index columns, as a value with no rows axis has them, and `k`, but those the ranked
    value's own table labels its rows by (`layers`, `pos`, `key`); then the ranked value's row at the pick, as its
    table labels it (`example_id`, `row`, `layer`, `pos` and `key` with their tokens through its read, and its
    `call` in a generation; a pick past the row's own positions or keys left out, as there), another ranked axis's
    index under its name, and the dataset `columns` of that row; and `value` last, the ranked value there, beside
    its `tokens` where the op decoded them."""
    over = arguments(record)["over"]
    axes = over if isinstance(over, list) else [over]
    ids = value["ids"] if isinstance(over, list) else {over: value["ids"]}
    ranked = record_of(plan.steps, record.of, point.at, plan.featurizers)
    examples = dataset_rows(plan, ranked.dataset) if ranked.dataset else []
    place = read_labels(plan, ranked, point.at) if ranked.read else None
    indices = place["indices"] if place else {}
    second = "probs" if "probs" in value else "values"
    shape, dims = list(value[second].shape), record.dims or []
    labelled = [a for a in ranked.dims or [] if a in ("rows", "layers", "pos", "key")]     # its table's row labels
    picked = {a: t.reshape(-1).tolist() for a, t in ids.items()}
    scores = value[second].reshape(-1).tolist()
    tokens = value.get("tokens")
    for _ in range(len(shape) - 1) if tokens is not None else ():
        tokens = [t for each in cast(list, tokens) for t in each]
    out = []
    for e, score in enumerate(scores):
        if picked[axes[0]][e] == -1:
            continue                        # a pick past the entries there
        own = {a: e // math.prod(shape[j + 1:]) % shape[j] for j, a in enumerate(dims)}
        where = {**own, **{a: picked[a][e] for a in axes}}         # the pick's index on each axis of the ranked value
        row: dict[str, Any] = {"at": at} if at else {}
        row.update((COLUMNS.get(a, a), indices[a][i] if a in indices else i) for a, i in own.items()
                   if a not in labelled)
        r = where["rows"]
        example = examples[r] if examples else None
        if example is not None and "example_id" in example:
            row["example_id"] = example["example_id"]
        for a in ranked.dims or []:
            i = where[a]
            if a in ("pos", "key") and place is not None:
                ks, got = along(place, r, a, i + 1, row.get("pos")) \
                    if place["positions"][r] != positions.MISSING else ([], {})
                if i not in ks:
                    break                   # past the row's own positions or keys
                row.update((c, column[ks.index(i)]) for c, column in got.items())
            elif a in labelled or a in axes:
                row[COLUMNS.get(a, a)] = indices[a][i] if a in indices else i
        else:
            if example is not None:
                row.update((c, example[c]) for c in columns)
            row.pop("value", None)
            row["value"] = {second: score, "tokens": tokens[e]} if tokens is not None else score
            out.append(row)
    return out


def rowless(record: Labelled, dims: list[str], held: list[str]) -> str:
    """What a saved value of several tensors (`held`) with no rows axis is, and where to read it, as `rows_of`'s
    refusal says: a bundle; another op's value over a value with no rows axis. A `top_k` over `rows` tables
    (`picks`)."""
    names = [{"probs": "probabilities"}.get(k, k) for k in held]
    said = " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)
    if not isinstance(record, OpStep):
        return f"laid out {dims}; read it from plan.results"
    if ops.DECLARED[record.op]["bundle"]:
        return f"a bundle (a {record.op}'s value); read it from plan.results"
    return f"laid out {dims}, a {record.op} over a value with no rows axis, its {said} one list each; read it from " \
        "plan.results"


def value_rows(value: Tensor | list[Any] | dict[str, Any], dims: list[str], place: dict[str, Any] | None,
               at: dict[str, Any], examples: list[dict], columns: list[str]) -> list[dict[str, Any]]:
    """A value's rows: one per index on its example and token axes, the rest of it as the row's value, after its
    point's coordinates (`at`, where it has any), its row's `example_id` and, after its labels, the dataset `columns`
    of its row (`examples`, the dataset's rows); a row the read did not reach (`place` says so) is one row,
    `alignment_missing`, whatever its token axes hold, none included (a read no row reached is `[rows, 0]`).

    Built column by column: the index columns from the shape, the token labels from one lookup per row and index
    through `place` (`along`), and the values from one `tolist` of the elements kept, the value laid out as its
    example axes, its token axes, then the rest; the dicts last, a key holding the last of the values given it in
    that order and `value` moved last."""
    first = value
    while isinstance(first, dict):          # a dict's first member, a top_k's `ids` over several axes its first axis's
        first = next(iter(first.values()))
    # a nested list's (a top_k's tokens) as far as its lists are of one length, up to the value's axes
    shape = list(first.shape) if isinstance(first, Tensor) else cast(list, shape_or_dtype(first, "shape"))[:len(dims)]
    # The axes a row is per: the example axes, then the token axes within each; every axis of a value with no rows.
    outer = [i for i, a in enumerate(dims) if a in ("layers", "rows") or "rows" not in dims]
    inner = [i for i, a in enumerate(dims) if a in ("pos", "key") and "rows" in dims]
    within = math.prod(shape[i] for i in inner)             # an outer index's elements, kept or not
    strides = [math.prod(shape[i] for i in outer[j + 1:]) for j in range(len(outer))]
    rows = outer.index(dims.index("rows")) if "rows" in dims else None
    kept: list[int] = []                    # the elements kept, by flat index over [outer..., inner...]
    labels: dict[str, list] = {}            # their token labels, a column each
    held: list[tuple[int, int | None]] = []     # each outer index with how many elements it keeps; None: missing
    for o in range(math.prod(shape[i] for i in outer)):
        r = o // strides[rows] % shape[outer[rows]] if rows is not None else 0
        if place and rows is not None and place["positions"][r] == positions.MISSING:
            held.append((o, None))          # one row says so, in place of its tokens
            continue
        ws, got = [0], {}
        for i in inner:                     # each token axis in turn, within each index the axes before it kept
            ws_, got_ = [], {c: [] for c in got}
            for j, w in enumerate(ws):
                ks, labelled = along(place, r, dims[i], shape[i], got["pos"][j] if place and "pos" in got else None)
                ws_ += map((w * shape[i]).__add__, ks)
                for c, column in got.items():
                    if c not in labelled:
                        got_[c] += [column[j]] * len(ks)
                for c, column in labelled.items():
                    got_.setdefault(c, []).extend(column)
            ws, got = ws_, got_
        kept += map((o * within).__add__, ws)
        for c, column in got.items() if ws else ():
            labels.setdefault(c, []).extend(column)
        held.append((o, len(ws)))
    index = torch.tensor(kept, dtype=torch.long)

    def cells(v: Any) -> list:              # the value at each element kept: a dict's per member
        if isinstance(v, Tensor):
            return laid(v, index, outer + inner, len(dims))
        if isinstance(v, dict):
            return [dict(zip(v, each)) for each in zip(*map(cells, v.values()))]
        return [part(v, pick_of(g, shape, outer, inner, strides, within, len(dims))) for g in kept]
    values = cells(value)
    indices = place["indices"] if place else {}      # the model's layer, or the index kept, at each index of an axis
    named = [COLUMNS.get(dims[i], dims[i]) for i in outer]
    out: list[dict[str, Any]] = []
    start = 0                               # where a run's elements begin among those kept
    for (missing, eid), run in itertools.groupby(held, key=lambda h: (
            h[1] is None, rows is not None and bool(examples) and
            "example_id" in examples[h[0] // strides[rows] % shape[outer[rows]]])):
        run = list(run)
        counts = [1 if n is None else n for _, n in run]
        given: dict[str, list] = {}         # each key's column, in the order a row writes its keys
        if at:
            given["at"] = [at] * sum(counts)
        example = [examples[o // strides[rows] % shape[outer[rows]]] for o, _ in run] \
            if rows is not None and examples else []
        if eid:
            given["example_id"] = spread([e["example_id"] for e in example], counts)
        for c, (j, i) in zip(named, enumerate(outer)):     # an outer index's own, by arithmetic on the shape
            own = [o // strides[j] % shape[i] for o, _ in run]
            given[c] = spread([indices[dims[i]][k] for k in own] if dims[i] in indices else own, counts)
        if missing:
            given[positions.MISSING] = [True] * len(run)
        else:
            n = sum(counts)
            given.update((c, column[start:start + n]) for c, column in labels.items())
            given["value"] = values[start:start + n]
            start += n
        for c in columns if example else ():
            given[c] = spread([e[c] for e in example], counts)
        if "value" in given:
            given["value"] = given.pop("value")
        out += [dict(zip(given, row)) for row in zip(*given.values())]
    return out


def spread(values: list, counts: list[int]) -> list:
    """Each of `values` as many times as `counts` says, in order."""
    out: list = []
    for v, n in zip(values, counts):
        out += [v] * n
    return out


def along(place: dict[str, Any] | None, row: int, axis: str, size: int, at: int | None
          ) -> tuple[list[int], dict[str, list]]:
    """The indices along token axis `axis` (`size` long) of row `row` that are its own, and their labels, a column
    each: on `pos` the positions the read resolved to and the tokens there, and for a read in a generation the
    forward `call` that holds each; on `key` each key position (through the read's `select`, or unselected the
    row's own key at that index) and its token, the row's own keys only, a key relative to the query at its offset
    from the position `at` the row is labelled with (its `offset` alone where the value has no `pos` axis left).
    Without a read, each index itself."""
    if place is None:
        return list(range(size)), {axis: list(range(size))}
    prompt = len(place["prompt"][row])
    if axis == "pos":
        index = place["positions"][row][:size]
        ks = list(range(len(index)))
    elif isinstance(place["keys"], dict):
        offsets = [place["keys"]["offset"][k] for k in range(size)]
        if at is None:
            return list(range(size)), {"offset": offsets}
        ks = [k for k in range(size) if 0 <= at + offsets[k] < prompt]
        index = [at + offsets[k] for k in ks]
    else:
        keys = place["keys"]
        ks = [k for k in range(size) if (keys[k] if keys is not None else k) < prompt]
        index = [keys[k] if keys is not None else k for k in ks]
    tokens = place["tokens"][row]
    got = {axis: index, f"{axis}_token": [tokens[i] for i in index]}
    if place["beyond"]:
        got["call"] = [positions.call_of(i, prompt) for i in index]
    return ks, got


def laid(value: Tensor, index: Tensor, order: list[int], n: int) -> list:
    """The elements `index` picks of `value` laid out with the axes `order` names first, flattened, then the rest
    (of its `n`), as plain values: each an element or the rest as nested lists."""
    rest = [i for i in range(n) if i not in order]
    flat = value.permute(order + rest).reshape(math.prod(value.shape[i] for i in order), *[value.shape[i] for i in rest])
    return flat[index.to(value.device)].tolist()


def pick_of(g: int, shape: list[int], outer: list[int], inner: list[int], strides: list[int], within: int,
            n: int) -> tuple[int | slice, ...]:
    """The index of element `g` (flat over the outer axes, then the inner) along each of the `n` axes, the rest
    whole, as `part` takes it."""
    o, w = divmod(g, within)
    at = {i: o // strides[j] % shape[i] for j, i in enumerate(outer)}
    for i in reversed(inner):
        w, at[i] = divmod(w, shape[i])
    return tuple(at.get(i, slice(None)) for i in range(n))


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
    input columns first and the columns its entry's ops name; back to a `permute` alone, whose rows hold other rows'
    values, so the read's columns are not the row's."""
    used: list[str] = []
    steps = runnable(plan.steps)
    while True:
        made = made_by(plan, name)          # an eval's member's: its eval step
        if made not in steps or points(steps[made])[0].kind == "fit":     # a featurizer's tensor, a fit's member
            return list(dict.fromkeys(used))
        step, member = agreeing_point(steps[made], at), name[len(made) + 1:]
        if step.kind == "op":
            used += argument_columns(plan, arguments(step))
            if step.op == "permute":        # the read's columns are other rows' from here on
                return list(dict.fromkeys(used))
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
    record's, under the value's dataset), the keys it selects, and what labels an index on each axis that has
    labels (`indices`): the model's layers on a `layers` axis, the indices kept on an axis a `select` keeps."""
    assert plan.run is not None and record.read is not None
    producer, _ = entry_of(plan.steps, record.read, at)
    read = entry_of(plan.steps, record.read, at)[1]
    prompt = plan.run.tokens[data.input_name(producer.input)]
    said = plan.run.tokens.get(record.read.rpartition(".")[0]) if producer.kind == "generate" else None
    if said is not None and projection(producer.at, cast(Labelled, producer).axes):
        said = said[coordinate_text(plan, projection(producer.at, cast(Labelled, producer).axes))]
    return {"prompt": prompt, "tokens": [p + s for p, s in zip(prompt, said)] if said else prompt,
            "beyond": producer.kind == "generate",
            "keys": read.kept.get("key"),
            "positions": read.positions[cast(str, read.dataset)],
            "indices": {**{a: v for a, v in record.kept.items() if isinstance(v, list)},
                        **({"layers": record.layers} if "layers" in (record.dims or []) else {})}}
