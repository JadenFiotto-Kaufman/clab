"""The run's state, each value under its name by the coordinates of the point that produced it, and an op's
arguments resolved by name against it."""

from typing import Any, cast

import torch
from torch import Tensor

from ...plan.model import Plan, dims_of, parameters
from .. import data, ops


def bound(engine, plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None,
          of: Any = None, vocabulary: list | None = None, live: dict | None = None,
          dims: list[str] | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up (a featurizer to its instance; a read of the same
    forward, in `live`, to `{"live": <the engine's name for it>}`), columns resolved against the value's own
    vocabulary (the tokenizer's, `answered`, or a probe's class list) or read as text, sliced to `rows`. On a
    ragged value `of`, an argument with a row per example is taken at each entry's row."""
    params, out = parameters(plan, name, args), {}
    for key, value in args.items():
        annotation, per_row = params[key].annotation, False
        if annotation == ops.Value and isinstance(value, str) and value in (live or {}):
            out[key] = {"live": (live or {})[value]}
            continue
        if annotation == ops.Value and isinstance(value, str):
            given, value = value, lookup(state, value, at)
            per_row = dims_of(plan, given)[:1] == ["rows"]
            value = value[rows] if rows is not None and per_row else value
        elif annotation == ops.Tokens:
            if vocabulary is None:
                out[key] = answered(*tokens(engine, plan, value, rows), of, dims or [], rows)
                continue
            value, per_row = indexed(plan, value, vocabulary, rows)[:, None], True
        elif annotation == ops.Text:
            column, strip, lower = data.column_of(value) or ("", False, False)
            out[key] = [(v if isinstance(v, list) else [v], strip, lower) for v in column_texts(plan, column, rows)]
            continue
        elif annotation == ops.Featurizer:
            value = lookup(state, value, at)
        if per_row and isinstance(of, dict) and "tok" in of:
            value = value[(of["row"] - (rows.start if rows is not None else 0)).to(cast(Tensor, value).device)]
        out[key] = value
    return out


def indexed(plan: Plan, column: str, vocabulary: list, rows: slice | None) -> Tensor:
    """A column's values as their indices in a probe's class list."""
    values = column_texts(plan, column, None)       # preflight checked each is a class
    return torch.tensor([vocabulary.index(v) for v in (values[rows] if rows is not None else values)])


def column_texts(plan: Plan, column: str, rows: slice | None) -> list:
    """A column's value on each of its dataset's rows (a string, or a list of them), sliced to `rows`."""
    name, _, path = column.partition(".")
    values = [data.lookup(r, path) for r in plan.data[name].rows or []]
    return values[rows] if rows is not None else values


def tokens(engine, plan: Plan, ref, rows: slice | None) -> tuple[list, list, str]:
    """A column reference's answer on each row, sliced to `rows`: its value, and per token the ids it may be
    (`data.answers`), and the column's name."""
    column, strip, lower = data.column_of(ref) or ("", False, False)
    values = column_texts(plan, column, rows)
    return values, data.answers(engine.tokenizer, values, column, strip, lower), column


def within_row(row: Tensor) -> Tensor:
    """Each of a ragged value's entries' index among its own row's, the entries in row order."""
    return torch.arange(len(row)) - torch.searchsorted(row, row)


def answered(values: list, sets: list, column: str, of: Any, dims: list[str], rows: slice | None) -> Tensor:
    """Each row's answer laid against the value `of` a metric scores, as the ids each position may be, padded
    with -1 (`metrics._aligned`): a one-token answer at every position of its row (`[rows, m]`, or
    `[entries, m]` on a ragged value), token i of a longer one at the row's position i (`[rows, pos, m]`, or
    `[entries, m]`), which needs the row to have as many positions as the answer has tokens."""
    start = rows.start if rows is not None else 0
    most = max(len(ids) for row in sets for ids in row)
    sets = [[ids + [-1] * (most - len(ids)) for ids in row] for row in sets]
    ragged = isinstance(of, dict) and "tok" in of
    if all(len(row) == 1 for row in sets):
        ids = torch.tensor([row[0] for row in sets])
        return ids[of["row"] - start] if ragged else ids
    if ragged:
        row, j = of["row"] - start, within_row(of["row"])
        lined(values, sets, torch.bincount(row, minlength=len(sets)).tolist(), column, start, True)
        return torch.tensor([sets[r][int(i) if len(sets[r]) > 1 else 0] for r, i in zip(row.tolist(), j)],
                            dtype=torch.long).reshape(-1, most)
    if "pos" not in dims:
        raise ValueError(f"{column}: an answer of several tokens scores a position each, and the value has no pos "
                         f"axis ({dims})")
    width = cast(Tensor, of).shape[dims.index("pos")]
    lined(values, sets, [width] * len(sets), column, start, False)
    return torch.tensor([[row[i] if len(row) > 1 else row[0] for i in range(width)] for row in sets])


def lined(values: list, sets: list, widths: list[int], column: str, start: int, ragged: bool) -> None:
    """Refuses a row whose answer of several tokens does not have one of the read's positions per token; a
    ragged row with none is missing, not refused."""
    for r, (value, k, n) in enumerate(zip(values, (len(s) for s in sets), widths)):
        if k > 1 and n != k and not (ragged and n == 0):
            raise ValueError(unaligned(column, start + r, value, k, n))


def unaligned(column: str, row: int, value, k: int, n: int) -> str:
    """Why a row's answer does not line up with the read a metric scores."""
    return (f"{column} row {row}: {value!r} is {k} tokens; the read has {n} position{'s' * (n != 1)} on that row; "
            "token i of an answer scores the read's position i, so the read carries one per token (a scope with "
            "\"before\": true reads the positions that predict the answer's tokens)")


def put(state: dict, name: str, at: dict, value) -> None:
    # A new dict each time: a fit's body namespace is a copy of the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")
