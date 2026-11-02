"""Sweeps: a value standing for several, the points it makes, a point's coordinates, and the run's state by
point."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clab.plan import Plan


UNSWEPT = ("kind", "op", "site", "component", "input", "interventions", "train", "featurizer", "file_path", "of",
           "saves")


def find_sweeps(document: dict) -> dict[tuple, str]:
    """Every `{"sweep": [...], "label"?: "<name>"}` standing for a value, by document path, each with its label:
    the `label` it gives, letters, digits and underscores (a coordinate is `<label>=<value>`, joined by commas),
    else the name of the field it stands for (the last key of its path). A sweep's values are numbers, strings,
    booleans or lists of them, each point keyed by its value, and hold no sweep: a `pos` sweeps its frame, its index
    or a scope's occurrence, not the object.

    A sweep may stand for any value except under `description`, `model` and `data` (a dataset's rows are compiled
    once, and a step reads them by name), on a name, a kind, or a `file_path`. One label is one axis: sweeps with
    one label, by default or given, and the same
    values are one axis, so their points pair up (zip), as one sweep reached from two places does (a swept site
    two steps use); sweeps with one label and different values are refused, and sweeps with different labels
    cross. So a read's `layers` swept `{"sweep": [0, 1, 2], "label": "src"}` and a write's `{"sweep": [0, 1, 2],
    "label": "dst"}` are a 3×3 grid in one step, each point keyed `src=0,dst=2`; swept `{"sweep": [0, 1, 2]}`
    each, both are labelled `layers`, so they pair up, the diagonal, layer L with layer L, and the grid gives one
    of them a `label`. A step that uses a swept thing, or references a value that varies, becomes a list of
    points, one per combination of its axes, each a whole step whose `at` gives its index on each axis; a step no
    sweep touches runs once.

    A value of a swept step depends on some of the step's axes, which its record says (`axes`): the sweeps on its
    own entry (its site, its positions, an op's literal argument), those of every value or featurizer its ops
    name, and, for an entry of a forward, the forward's own, which are the step's options' and its writes' (each
    write's site, positions, ops up to its last edit and their operands; an operand from an earlier entry of the
    forward counts that entry's); an op step's value depends on its `of`'s and its operands' axes beside its own
    sweeps, and a fit's members on the fit's. So a read of a forward whose writes do not vary depends on its own
    sweeps alone, and a generation's ids on the forward's. A value is made once per distinct point on its axes,
    and a saved value is a dict keyed by coordinate on those axes (the axis's value, or a tuple of them in axis
    order), the value itself where it depends on none; so a read swept over layers beside writes that do not vary
    is a dict by layer, and a sampled generation whose points differ in reads alone is one sample. Two points of a
    forward or generate step run one forward exactly when they agree on the forward's axes (`schedule`): a sweep
    on what a step reads runs one forward, a sweep on what it writes one per point.
    """
    found: dict[tuple, str] = {}
    first: dict[str, tuple[str, list]] = {}         # each label's first sweep: where it stands and its values

    def walk(value: Any, path: tuple) -> None:
        if isinstance(value, dict) and "sweep" in value and set(value) <= {"sweep", "label"}:
            where = ".".join(map(str, path))
            field = next(k for k in reversed(path) if isinstance(k, str))
            if path[0] in ("description", "model", "data") or field in UNSWEPT:
                raise ValueError(f"{where}: a sweep cannot stand for {field!r}"
                                 + ("; a dataset's rows are compiled once, and a step reads them by name: write one "
                                    "dataset per value" if path[0] == "data" else
                                    f"; a run loads one model: write one document per {field}" if path[0] == "model"
                                    else ""))
            given = value.get("label")
            label = field if given is None else given
            if given is not None and not (isinstance(label, str) and re.fullmatch(r"\w+", label)):
                raise ValueError(f"{where}.label: {label!r}: a sweep's label is letters, digits and underscores, "
                                 "since its points are keyed `<label>=<value>`, joined by commas")
            if holds_sweep(value["sweep"]):
                raise ValueError(f"{where}: a sweep's values hold no sweep")
            scalars(where, field, value["sweep"])
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


def scalars(where: str, field: str, values: Any) -> None:
    """A sweep's values are numbers, strings, booleans or lists of them, since each point is keyed by its value: an
    object is refused with an example that sweeps a scalar inside it, its first scalar leaf, or, standing for a
    `pos`, one of the fields a pos takes a sweep on (its frame, its index, a scope's occurrence)."""
    for value in values if isinstance(values, list) else []:
        said = "a sweep's values are numbers, strings, booleans or lists of them, each point keyed by its value"
        if isinstance(value, dict) and field == "pos":
            raise ValueError(f"{where}: {said}; {value!r} is an object: sweep a scalar inside the pos, which takes one "
                             "on frame, index and a scope's occurrence ({'frame': 0, 'index': {'sweep': [-1, -2]}}, or "
                             "{'scope': {'column': ..., 'occurrence': {'sweep': [0, 1]}}})")
        if isinstance(value, dict):
            path, leaf = [], value            # the example sweeps the object's first scalar leaf
            while isinstance(leaf, dict) and leaf:
                key, leaf = next(iter(leaf.items()))
                path.append(key)
            example = f"{{'sweep': [{leaf!r}, ...]}}"
            for key in reversed(path or ["<field>"]):
                example = f"{{{key!r}: {example}}}"
            raise ValueError(f"{where}: {said}; {value!r} is an object: sweep a scalar field inside it, {example}")
        if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
            raise ValueError(f"{where}: {said}; {value!r} holds a list or an object")


def holds_sweep(value: Any) -> bool:
    """Whether `value` is, or holds anywhere inside it, a `{"sweep": ...}`."""
    if isinstance(value, dict):
        return "sweep" in value or any(map(holds_sweep, value.values()))
    return isinstance(value, list) and any(map(holds_sweep, value))


def points(entry: Any) -> list:
    """A plan entry as its list of points; `[entry]` when it is not swept."""
    return entry if isinstance(entry, list) else [entry]


def agreeing(entry: Any, at: dict[str, int]) -> Any:
    """The point of `entry` that agrees with the point `at` on the entry's own axes. A swept entry's points are
    the product of its axes in axis order, the last fastest (`compile.expand`), so that point is the one at `at`'s
    index on each axis, each axis as long as the last point's index on it plus one: found without a scan of the
    points, which a list in another order falls back to."""
    listed, i = points(entry), 0
    for label, last in listed[-1].at.items():
        i = i * (last + 1) + at.get(label, 0)
    if i < len(listed) and all(at.get(k) == v for k, v in listed[i].at.items()):
        return listed[i]
    return next(p for p in listed if all(at.get(k) == v for k, v in p.at.items()))


def coordinate(plan: Plan, at: dict[str, int]) -> Any:
    """A point's key in a swept result: its axis's value (a list as a tuple), or the tuple of its axes' values
    in axis order."""
    values = [tuple(v) if isinstance(v := plan.sweeps[label].values[i], list) else v for label, i in at.items()]
    return values[0] if len(values) == 1 else tuple(values)


def labelled(plan: Plan, at: dict[str, int]) -> str:
    """A point's coordinate as text: `pos=-1`, `layers=0,pos=-1`."""
    return spelled({label: axis.values for label, axis in plan.sweeps.items()}, at)


def spelled(values: dict[str, list], at: dict[str, int]) -> str:
    """A point's coordinate as text, given each axis's `values`: the one spelling of a point, in the run record,
    a saved file's keys and a refusal at one point of a sweep."""
    return ",".join(f"{label}={json.dumps(values[label][i], separators=(',', ':'))}" for label, i in at.items())


def put(state: MutableMapping, name: str, at: dict[str, int], value: Any) -> None:
    # A new dict each time: a fit's body namespace falls back to the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: Mapping, name: str, at: dict[str, int]) -> Any:
    """The value `name` at the producer's point that agrees with `at` on the producer's axes: keyed by `at` on the
    axes of the value's first key, which its points share (`put`), found without a scan of its points; a value
    whose points differ in their axes (an op's, where a swept name picks operands on different axes; a featurizer a
    swept fit trains, whose live instance sits under the empty key beside the earlier points' keys inside the
    body) is scanned for the first that agrees."""
    by_point = state[name]
    key = tuple((k, at.get(k)) for k, _ in next(iter(by_point), ()))
    if key in by_point:
        return by_point[key]
    for coords, value in by_point.items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def projected(at: dict[str, int], axes: list[str]) -> dict[str, int]:
    """The point `at` projected onto `axes`: its index on each of them, in axis order."""
    return {label: at[label] for label in axes if label in at}


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}
