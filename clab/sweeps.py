"""Sweeps: a value standing for several, the points it makes, a point's coordinates, and the run's state by
point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plan import Plan


UNSWEPT = ("kind", "op", "site", "component", "input", "interventions", "train", "featurizer", "path", "rows",
           "split", "file_path", "of", "hf", "revision", "saves")


def find_sweeps(document: dict) -> dict[tuple, str]:
    """Every `{"sweep": [...]}` standing for a value, by document path, labelled by the last key of its path.

    A sweep may stand for any value except under `description` and `model`, on a name, a kind, a dataset's
    source, or a `file_path`. Two sweeps with one label are refused; one sweep reached from two places (a swept site
    two steps use) is one axis, so its points zip rather than cross. A step that uses a swept thing, or
    references a value that varies, becomes a list of points, one per combination of its axes, each a whole
    step whose `at` gives its index on each axis; a step no sweep touches runs once. A saved value of a
    swept step is a dict keyed by coordinate: the axis's value, or a tuple of them in axis order.
    """
    found: dict[tuple, str] = {}

    def walk(value: Any, path: tuple) -> None:
        if isinstance(value, dict) and set(value) == {"sweep"}:
            where = ".".join(map(str, path))
            label = next(k for k in reversed(path) if isinstance(k, str))
            if path[0] in ("description", "model") or label in UNSWEPT:
                raise ValueError(f"{where}: a sweep cannot stand for {label!r}")
            for other, other_label in found.items():
                if other_label == label:
                    raise ValueError(f"{where}: label {label!r} is also the sweep at {'.'.join(map(str, other))}; a "
                                     "sweep's label is the name of the field it stands for, so two sweeps of one "
                                     "field name cannot cross: sweep one of them and put each value of the other in "
                                     "a step of its own (or have both use one swept site or step, which zips them)")
            found[path] = label
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in ("results", "run", "sweeps"):
                    walk(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + (i,))

    walk(document, ())
    return found


def points(entry: Any) -> list:
    """A plan entry as its list of points; `[entry]` when it is not swept."""
    return entry if isinstance(entry, list) else [entry]


def agreeing(entry: Any, at: dict) -> Any:
    """The point of `entry` that agrees with the point `at` on the entry's own axes."""
    return next(p for p in points(entry) if all(at.get(k) == i for k, i in p.at.items()))


def coordinate(plan: Plan, at: dict) -> Any:
    """A point's key in a swept result: its axis's value (a list as a tuple), or the tuple of its axes' values
    in axis order."""
    values = [tuple(v) if isinstance(v := plan.sweeps[label].values[i], list) else v for label, i in at.items()]
    return values[0] if len(values) == 1 else tuple(values)


def labelled(plan: Plan, at: dict, labels: bool = True) -> str:
    """A point's coordinate as text: `pos=-1`, `layers=0,pos=-1`; without its labels, `-1`, `0,-1`."""
    parts = [json.dumps(plan.sweeps[label].values[i], separators=(",", ":")) for label, i in at.items()]
    return ",".join(f"{label}={part}" for label, part in zip(at, parts)) if labels else ",".join(parts)


def put(state: dict, name: str, at: dict, value) -> None:
    # A new dict each time: a fit's body namespace is a copy of the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def by_point(plan: Plan, name: str) -> list[tuple[dict, Any]]:
    """A saved value per point: each point's coordinates (none when its step is not swept) and its value."""
    entry = plan.steps[name.partition(".")[0]]
    if not isinstance(entry, list):
        return [({}, plan.results[name])]
    return [({label: plan.sweeps[label].values[i] for label, i in p.at.items()}, plan.results[name][coordinate(plan, p.at)])
            for p in entry]
