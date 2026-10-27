"""Sweeps: a value standing for several, the points it makes, a point's coordinates, and the run's state by
point."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clab.plan import Plan


UNSWEPT = ("kind", "op", "site", "component", "input", "interventions", "train", "featurizer", "path", "rows", "hf",
           "revision", "config", "split", "file_path", "of", "saves")


def find_sweeps(document: dict) -> dict[tuple, str]:
    """Every `{"sweep": [...], "label"?: "<name>"}` standing for a value, by document path, each with its label:
    the `label` it gives, letters, digits and underscores (a coordinate is `<label>=<value>`, joined by commas),
    else the name of the field it stands for (the last key of its path). A sweep's values hold no sweep.

    A sweep may stand for any value except under `description` and `model`, on a name, a kind, a dataset's
    source, or a `file_path`. One label is one axis: sweeps with one label, by default or given, and the same
    values are one axis, so their points pair up (zip), as one sweep reached from two places does (a swept site
    two steps use); sweeps with one label and different values are refused, and sweeps with different labels
    cross. So a read's `layers` swept `{"sweep": [0, 1, 2], "label": "src"}` and a write's `{"sweep": [0, 1, 2],
    "label": "dst"}` are a 3×3 grid in one step, each point keyed `src=0,dst=2`; swept `{"sweep": [0, 1, 2]}`
    each, both are labelled `layers`, so they pair up, the diagonal, layer L with layer L, and the grid gives one
    of them a `label`. A step that uses a swept thing, or references a value that varies, becomes a list of
    points, one per combination of its axes, each a whole step whose `at` gives its index on each axis; a step no
    sweep touches runs once. A saved value of a swept step is a dict keyed by coordinate: the axis's value, or a
    tuple of them in axis order.
    """
    found: dict[tuple, str] = {}
    first: dict[str, tuple[str, list]] = {}         # each label's first sweep: where it stands and its values

    def walk(value: Any, path: tuple) -> None:
        if isinstance(value, dict) and "sweep" in value and set(value) <= {"sweep", "label"}:
            where = ".".join(map(str, path))
            field = next(k for k in reversed(path) if isinstance(k, str))
            if path[0] in ("description", "model") or field in UNSWEPT:
                raise ValueError(f"{where}: a sweep cannot stand for {field!r}")
            given = value.get("label")
            label = field if given is None else given
            if given is not None and not (isinstance(label, str) and re.fullmatch(r"\w+", label)):
                raise ValueError(f"{where}.label: {label!r}: a sweep's label is letters, digits and underscores, "
                                 "since its points are keyed `<label>=<value>`, joined by commas")
            if holds_sweep(value["sweep"]):
                raise ValueError(f"{where}: a sweep's values hold no sweep")
            if label in first and json.dumps(first[label][1]) != json.dumps(value["sweep"]):
                other, values = first[label]
                raise ValueError(f"{where}: label {label!r} is also the sweep at {other}, over {values}; one label "
                                 f"is one axis, whose sweeps pair their points: give one of them its own label to "
                                 f"cross them ({{'sweep': [...], 'label': '<name>'}}), or the same values to pair "
                                 "them")
            first.setdefault(label, (where, value["sweep"]))
            found[path] = label
        elif isinstance(value, dict):
            for k, v in value.items():
                if path or k not in ("results", "run", "sweeps"):        # a plan's own derived fields hold none
                    walk(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + (i,))

    walk(document, ())
    return found


def holds_sweep(value: Any) -> bool:
    """Whether `value` is, or holds anywhere inside it, a `{"sweep": ...}`."""
    if isinstance(value, dict):
        return "sweep" in value or any(map(holds_sweep, value.values()))
    return isinstance(value, list) and any(map(holds_sweep, value))


def points(entry: Any) -> list:
    """A plan entry as its list of points; `[entry]` when it is not swept."""
    return entry if isinstance(entry, list) else [entry]


def agreeing(entry: Any, at: dict[str, int]) -> Any:
    """The point of `entry` that agrees with the point `at` on the entry's own axes."""
    return next(p for p in points(entry) if all(at.get(k) == i for k, i in p.at.items()))


def coordinate(plan: Plan, at: dict[str, int]) -> Any:
    """A point's key in a swept result: its axis's value (a list as a tuple), or the tuple of its axes' values
    in axis order."""
    values = [tuple(v) if isinstance(v := plan.sweeps[label].values[i], list) else v for label, i in at.items()]
    return values[0] if len(values) == 1 else tuple(values)


def labelled(plan: Plan, at: dict[str, int]) -> str:
    """A point's coordinate as text: `pos=-1`, `layers=0,pos=-1`."""
    return ",".join(f"{label}={json.dumps(plan.sweeps[label].values[i], separators=(',', ':'))}"
                    for label, i in at.items())


def put(state: MutableMapping, name: str, at: dict[str, int], value: Any) -> None:
    # A new dict each time: a fit's body namespace falls back to the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: Mapping, name: str, at: dict[str, int]) -> Any:
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def by_point(plan: Plan, name: str) -> list[tuple[dict[str, Any], Any]]:
    """A saved value per point: each point's coordinates (none when its step is not swept) and its value."""
    entry = plan.steps[name.partition(".")[0]]
    if not isinstance(entry, list):
        return [({}, plan.results[name])]
    return [({label: plan.sweeps[label].values[i] for label, i in p.at.items()}, plan.results[name][coordinate(plan, p.at)])
            for p in entry]
