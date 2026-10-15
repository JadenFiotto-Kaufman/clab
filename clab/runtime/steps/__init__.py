"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp engine it runs inside
one session; under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
"""

import importlib.metadata
import platform
import time
from typing import Any

import torch
from torch import Tensor

from ...plan.model import Plan, Run, coordinate, labelled, points
from ...registry import registers
from .preflight import instance, preflight
from .state import lookup, put

KINDS: dict = {}
kind = registers(KINDS)


def begin(engine, plan: Plan, batch_size: int, remote: bool | str) -> float:
    """What every engine's `execute` checks first, and the run record it starts; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    plan.run = Run(engine=engine.name, remote=remote, batch_size=batch_size,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def run(engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, without gradients except where a fit trains; then each saved
    value into `plan.results` (a trained featurizer as its bundle's tensors)."""
    assert plan.run is not None
    plan.run.versions = versions()
    found = preflight(engine, plan)
    state = made(plan, found["widths"])
    with torch.no_grad():
        for name, entry in plan.steps.items():
            for step in points(entry):
                KINDS[step.kind](engine, plan, name, step, state, batch_size, found)
    for name in plan.saves:
        entry = plan.steps[name.partition(".")[0]]
        plan.results[name] = ({coordinate(plan, p.at): finished(saved(plan, state, name, p.at)) for p in entry}
                              if isinstance(entry, list) else finished(saved(plan, state, name, {})))
    positions = dict(found["positions"])
    for fit, by_point in found["bodies"].items():        # a body's records, as each eval run resolved them
        for sources in by_point.values():
            for label, (_, body) in sources["evals"].items():
                for where, resolved in body["positions"].items():
                    key = f"{fit}.{label}.{where}"
                    positions[key] = {**positions.get(key, {}), **resolved}
    plan.run.positions = {where: by_coordinate(plan, by_point) for where, by_point in positions.items()}
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found["said"].items()})


def saved(plan: Plan, state: dict, name: str, at: dict) -> Any:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    value = lookup(state, name, at)
    return value.tensors() if step.kind == "fit" and member in step.train else value


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def made(plan: Plan, widths: dict) -> dict:
    """The run's state at its start: each declared featurizer used at a site, at each of its points, made
    where the model is."""
    state: dict = {}
    for name, entry in plan.featurizers.items():
        for spec in points(entry) if name in widths else []:
            put(state, name, spec.at, instance(plan, spec, widths[name], None))
    return state


def finished(value):
    """Detached CPU clones all the way down; a result never holds a view of a larger buffer."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: finished(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finished(v) for v in value]
    return value


def versions() -> dict[str, str]:
    """The installed versions where the model is."""
    found = {p: importlib.metadata.version(p) for p in ("clab", "nnsight", "nnterp", "transformers", "torch")}
    return {**found, "python": platform.python_version()}


# The kinds register in this order.
from . import forward, generate, op, fit  # noqa: E402,F401
