"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
"""

import functools
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ... import registry
from ...plan.model import Plan, Run, coordinate, labelled, points
from ...registry import registers
from .preflight import instance, preflight
from .state import lookup, put

if TYPE_CHECKING:
    from ...engines import Parameterized

KINDS: dict = {}
kind = registers(KINDS)


def begin(engine, plan: Plan, batch_size: int, options: dict) -> float:
    """What every engine's `execute` checks first, and the run record it starts: the engine, the batch size,
    the device, the options it loaded with and those it runs with (`options`, each declared by its `execute`'s
    signature) after their defaults, and what is installed on the client (`versions`); returns the start time."""
    runs = registry.spelled(type(engine).execute, ("self", "plan", "batch_size"))
    for key in options:
        if key not in runs:
            raise ValueError(f"the {engine.name} engine runs with no {key!r}; it runs with: "
                             f"{', '.join(runs) or 'nothing'}")
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    given = {**engine.options, **{k: options.get(k, p.default) for k, p in runs.items()}}
    plan.run = Run(engine=engine.name, batch_size=batch_size, device=engine.device, options=given,
                   client=versions(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def run(engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, without gradients except where a fit trains; then each saved
    value into `plan.results` (a trained featurizer as its bundle's tensors), and what served the run: the
    dtypes and devices of the model's parameters (`placed`)."""
    assert plan.run is not None
    plan.run.versions, plan.run.num_layers = versions(), engine.sizes()["num_layers"]
    found = preflight(engine, plan)
    state = made(plan, found["widths"], home(engine))
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
    plan.run.dtype, plan.run.devices = placed(engine.model)


def placed(model: "Parameterized") -> tuple[list[str], list[str]]:
    """The distinct dtypes and devices of `model`'s parameters, each in the order first met. `run` reads them
    inside the session after the steps, so they are the server's under the ndif engine, and those of a model
    that loads its weights at its first forward once it has."""
    dtypes, devices = {}, {}
    for p in model.parameters():
        dtypes[str(p.dtype).removeprefix("torch.")] = devices[str(p.device)] = None
    return list(dtypes), list(devices)


def saved(plan: Plan, state: dict, name: str, at: dict) -> Any:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    value = lookup(state, name, at)
    return value.tensors() if step.kind == "fit" and member in step.train else value


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def home(engine) -> str:
    """Where the model is, which is where the run makes featurizers and an engine sends a batch's inputs: where
    its first parameter is (on an NDIF server, or wherever the engine's `device` put it: the first of several
    under a device map that spreads it), else, while the model is a meta shell that loads its weights at its
    first forward, the engine's device."""
    device = str(next(engine.model.parameters()).device)
    return (engine.device or "cpu") if device == "meta" else device


def made(plan: Plan, widths: dict, device: str) -> dict:
    """The run's state at its start: each declared featurizer used at a site, at each of its points, made
    where the model is (`device`)."""
    state: dict = {}
    for name, entry in plan.featurizers.items():
        for spec in points(entry) if name in widths else []:
            put(state, name, spec.at, instance(plan, spec, widths[name], None, device))
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


@functools.cache
def versions() -> dict[str, str]:
    """Every distribution installed here by name, sorted, clab's with its commit (`version`), and `python`. `run`
    calls it inside the session, so under the ndif engine they are the server's; `begin` calls it on the client.
    Read once per process: what is installed does not change while it runs."""
    found = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    found["clab"] = version()
    return {**dict(sorted(found.items(), key=lambda item: item[0].lower())), "python": platform.python_version()}


def version() -> str:
    """clab's version here, with the commit it is when one is known: the one an install from git recorded
    (`direct_url.json`), else `git describe --always --dirty` of the checkout clab is imported from, else the
    static version alone."""
    dist = importlib.metadata.distribution("clab")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    if commit:
        return f"{dist.version}+{commit[:12]}"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if not os.path.exists(os.path.join(root, ".git")):         # not a checkout (never a git repo above one)
        return dist.version
    try:
        described = subprocess.run(["git", "-C", root, "describe", "--always", "--dirty"], capture_output=True,
                                   text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return dist.version
    return f"{dist.version}+{described}"


# The kinds register in this order.
from . import forward, generate, op, fit  # noqa: E402,F401
