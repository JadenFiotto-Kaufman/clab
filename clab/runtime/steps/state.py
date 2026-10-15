"""The run's state, each value under its name by the coordinates of the point that produced it, and an op's
arguments resolved by name against it."""

from typing import Any

import torch
from torch import Tensor

from ...plan.model import Plan, dims_of, parameters
from .. import data, ops


def bound(engine, plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None,
          of: Any = None, vocabulary: list | None = None, live: dict | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up (a featurizer to its instance; a read of the same
    forward, in `live`, to `{"live": <the engine's name for it>}`), columns resolved against the value's own
    vocabulary (the tokenizer's, or a probe's class list), sliced to `rows`. On a ragged value `of`, an
    argument with a row per example is taken at each entry's row."""
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
            value, per_row = (tokens(engine, plan, value, args.get("token_form", "spaced"), rows) if vocabulary is None
                              else indexed(plan, value, vocabulary, rows)), True
        elif annotation == ops.Featurizer:
            value = lookup(state, value, at)
        if per_row and isinstance(of, dict) and "tok" in of:
            value = value[of["row"] - (rows.start if rows is not None else 0)]
        out[key] = value
    return out


def indexed(plan: Plan, column: str, vocabulary: list, rows: slice | None) -> Tensor:
    """A column's values as their indices in a probe's class list."""
    name, _, col = column.partition(".")
    values = [r[col] for r in plan.data[name].rows or []]       # preflight checked each is a class
    return torch.tensor([vocabulary.index(v) for v in (values[rows] if rows is not None else values)])


def tokens(engine, plan: Plan, column: str, form: str, rows: slice | None) -> Tensor:
    name, _, col = column.partition(".")
    texts = [r[col] for r in plan.data[name].rows or []]
    return data.token_ids(engine.tokenizer, texts[rows] if rows is not None else texts, form, column)


def put(state: dict, name: str, at: dict, value) -> None:
    # A new dict each time: a fit's body namespace is a copy of the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")
