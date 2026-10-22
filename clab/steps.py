"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .answers import probe_of
from . import ops
from .batches import batches, bind, bound, home, joined, layers, read, read_name, stacked
from .featurizers.uses import instance
from .plan import Plan, Run, arguments, dims_of, entries, points
from .preflight import Found, preflight
from .sweeps import by_coordinate, coordinate, lookup, put

if TYPE_CHECKING:
    from .engines import Parameterized

def begin(engine, plan: Plan, batch_size: int) -> float:
    """What every engine's `execute` checks first, and the run record it starts: the engine, the batch size,
    the device and the options it loaded with; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    plan.run = Run(engine=engine.name, batch_size=batch_size, device=engine.device, options=engine.options,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def run(engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, without gradients except where a fit trains; then each saved
    value into `plan.results` (a trained featurizer as its bundle's tensors), and what served the run: the
    dtypes and devices of the model's parameters (`placed`)."""
    assert plan.run is not None
    plan.run.versions, plan.run.num_layers = versions(), engine.sizes()["num_layers"]
    found = preflight(engine, plan)
    state = made(plan, found.widths, home(engine))
    with torch.no_grad():
        for name, entry in plan.steps.items():
            for step in points(entry):
                KINDS[step.kind](engine, plan, name, step, state, batch_size, found)
    for name in plan.saves:
        entry = plan.steps[name.partition(".")[0]]
        plan.results[name] = ({coordinate(plan, p.at): finished(saved(plan, state, name, p.at)) for p in entry}
                              if isinstance(entry, list) else finished(saved(plan, state, name, {})))
    positions = dict(found.positions)
    for fit, by_point in found.bodies.items():        # a body's records, as each eval run resolved them
        for sources in by_point.values():
            for label, (_, body) in sources["evals"].items():
                for where, resolved in body.positions.items():
                    key = f"{fit}.{label}.{where}"
                    positions[key] = {**positions.get(key, {}), **resolved}
    plan.run.positions = {where: by_coordinate(plan, by_point) for where, by_point in positions.items()}
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found.said.items()})
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


PACKAGES = ("nnsight", "nnterp", "torch", "transformers")


def versions() -> dict[str, str]:
    """What served the run, by name: clab's version with its commit (`version`), the packages it runs on, and
    `python`. `run` calls it inside the session, so under the ndif engine they are the server's."""
    return {"clab": version(), **{p: importlib.metadata.version(p) for p in PACKAGES},
            "python": platform.python_version()}


def version() -> str:
    """clab's version here, with the commit it is when one is known: the one an install from git recorded
    (`direct_url.json`), else `git describe --always --dirty` of the checkout clab is imported from, else the
    static version alone."""
    dist = importlib.metadata.distribution("clab")
    url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = url.get("vcs_info", {}).get("commit_id")
    if commit:
        return f"{dist.version}+{commit[:12]}"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, ".git")):         # not a checkout (never a git repo above one)
        return dist.version
    try:
        described = subprocess.run(["git", "-C", root, "describe", "--always", "--dirty"], capture_output=True,
                                   text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return dist.version
    return f"{dist.version}+{described}"




def forward(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: Found) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each gathered entry joined along
    rows.

    An entry whose `layers` are a slice or picks is one read per layer, the ops after its last edit run on
    each, stacked layer-first; one whose rows hold different numbers of positions is as wide as the widest,
    NaN past a row's own."""
    parts = []
    gathered = {k: r for k, r in entries(step).items() if r.gathered}
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        sites, records = bind(engine, plan, name, step, state, found, chunk, pads)
        got = engine.forward(batch, records)
        parts.append({key: stacked(record, [read(engine, plan, record, got[read_name(key, layer)], sites[key], chunk,
                                                 state, step.at, found.answers)
                                            for layer in layers(engine, record)])
                      for key, record in gathered.items()})
    joined(state, name, step, parts)


def op(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: Found) -> None:
    """The op over the whole value `of` names, given each of its rows' dataset row. `top_k`'s value also holds
    its tokens, decoded."""
    of = lookup(state, step.of, step.at)
    producer = points(plan.steps[step.of.partition(".")[0]])[0]
    vocabulary = lookup(state, probe_of(plan, step.of), step.at).classes if "classes" in dims_of(plan, step.of) \
        else None
    dims = dims_of(plan, step.of)
    rows = torch.arange(of.shape[dims.index("rows")]) if "rows" in dims and isinstance(of, Tensor) else None
    value = ops.call(step.op, of, bound(plan, step.op, arguments(step), state, step.at, None, of, vocabulary,
                                        dims=dims, answers=found.answers), engine=engine, dims=dims,
                     eos=generate.stops(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    if step.op == "top_k":
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)


def decoded(tokenizer, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are; None for a missing one (-1)."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    return [tokenizer.decode([i]) if i >= 0 else None for i in ids.tolist()]


from . import fit, generate  # noqa: E402

# The step kinds, each run as `(engine, plan, name, step, state, batch_size, found)`.
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit}
