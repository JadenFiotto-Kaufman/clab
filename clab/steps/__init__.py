"""Walk a compiled plan inside the run: its hub datasets' rows loaded, the preflight pass, every step and point in
order, then the saved values into `plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
A forward or generate step's points that share one forward (`Forward.shares`, which compile decides) run as one
group (`groups`), every point's entries bound to it under the point's own coordinates. Each step kind returns the
model calls its group made and the tokens they fed (`batches.fed`), which `run` records with the group's time and
memory (`measured`) at the point that runs the forward.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from clab import ops
from clab.compile import materialized
from clab.compile.answers import probe_of
from clab.compile.interventions import acts_at
from clab.compile.preflight import preflight
from clab.compile.state import Found
from clab.compile.uses import instance
from clab.format.sweeps import by_coordinate, coordinate, lookup, put
from clab.plan import Entry, Forward, OpStep, Plan, Run, Step, arguments, dims_of, points
from clab.steps import fit, generate
from clab.steps.batches import bind, bound, chunked, fed, home, joined, reads

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine, Parameterized


def begin(engine: Engine, plan: Plan, batch_size: int) -> float:
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


def run(engine: Engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, a group of points that share a forward together (`groups`), over
    the plan `held`, without gradients except where a fit trains, each group's cost `measured` into
    `plan.run.costs` at the point that runs the forward, and `SHARING` at each other; then each saved value into
    `plan.results` (a trained featurizer as its bundle's tensors), and what served the run: the dtypes and devices
    of the model's parameters (`placed`)."""
    plan = held(plan)
    assert plan.run is not None
    plan.run.versions, plan.run.num_layers = versions(), engine.sizes()["num_layers"]
    found = preflight(engine, plan)
    state = made(plan, found.widths, home(engine))
    costs: dict = {}
    with torch.no_grad():
        for name, entry in plan.steps.items():
            for group in groups(entry):
                cost = measured(engine, lambda: KINDS[group[0].kind](engine, plan, name, group, state, batch_size,
                                                                       found))
                for step in group:
                    put(costs, name, step.at, cost if step is group[0] else dict(SHARING))
    plan.run.costs = {name: by_coordinate(plan, by_point) for name, by_point in costs.items()}
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


def groups(entry: Entry) -> list[list[Step]]:
    """A step's points as they run, in order: each point that runs a forward, a generation, an op or a fit, with
    the points that share its forward after it (`Forward.shares`, its `at`)."""
    out: dict[tuple, list[Step]] = {}
    for step in points(entry):
        served = (step.shares if isinstance(step, Forward) else None) or step.at
        out.setdefault(tuple(served.items()), []).append(step)
    return list(out.values())


# What a point records whose forward another point of its group runs, which records the group's cost.
SHARING = {"seconds": 0.0, "memory": None, "forwards": 0, "tokens": 0}


def measured(engine: Engine, call: Callable[[], Counter]) -> dict[str, Any]:
    """What a step cost: `call`, which runs it, timed (`seconds`), with the model's CUDA devices (`gpus`)
    synchronized at both ends; the peak memory allocated on each of them while it ran, weights included, in bytes
    by device (`memory`, None where the model is not on CUDA); and the model calls it made and the tokens they
    fed, which `call` returns (`forwards`, `tokens`). A remote engine's run makes no CUDA call and its `memory`
    is None: on an NDIF server the process is the server's, whose own per-request accounting a reset of the
    peak would clobber, and its sandbox's runner would measure its own tensors."""
    devices = [] if engine.remote else gpus(engine)
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    count = call()
    for device in devices:
        torch.cuda.synchronize(device)
    seconds = round(time.perf_counter() - started, 3)
    memory = {d: torch.cuda.max_memory_allocated(d) for d in devices} or None
    return {"seconds": seconds, "memory": memory, "forwards": count["forwards"], "tokens": count["tokens"]}


def gpus(engine: Engine) -> list[str]:
    """The CUDA devices the model is on: where its parameters are (a meta shell's, which loads its weights at its
    first forward, where the engine places it: `home`)."""
    devices = dict.fromkeys(home(engine) if d == "meta" else d for d in placed(engine.model)[1])
    return [d for d in devices if d.startswith("cuda")]


def held(plan: Plan) -> Plan:
    """The plan as the run reads it: every dataset's rows in it, a hub dataset's loaded here, where the run is (on the
    server, under `ndif`), and refused unless their digest is the one compile recorded (`materialized`). It is a
    view: its `results` and `run` are the plan's own, which the run fills."""
    return plan.model_copy(update={"data": {n: d.model_copy(update={"rows": materialized(plan, n)})
                                            for n, d in plan.data.items()}})


def placed(model: Parameterized) -> tuple[list[str], list[str]]:
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


def made(plan: Plan, widths: dict[str, int], device: str) -> dict:
    """The run's state at its start: each declared featurizer used at a site, at each of its points, made
    where the model is (`device`)."""
    state: dict = {}
    for name, entry in plan.featurizers.items():
        for spec in points(entry) if name in widths else []:
            put(state, name, spec.at, instance(plan, spec, widths[name], None, device))
    return state


def finished(value: Any) -> Any:
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
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(root, ".git")):         # not a checkout (never a git repo above one)
        return dist.version
    try:
        described = subprocess.run(["git", "-C", root, "describe", "--always", "--dirty"], capture_output=True,
                                   text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return dist.version
    return f"{dist.version}+{described}"


def forward(engine: Engine, plan: Plan, name: str, group: list[Forward], state: dict, batch_size: int,
            found: Found) -> Counter:
    """The step's input in consecutive batches, each through `engine.forward` once for the points of `group`
    (those that share the forward; one point where none does); each point's gathered entries joined along rows.
    Returns the model calls it made and the tokens they fed (`fed`).

    An entry whose `layers` are a slice or picks is one write (when it writes) and one read per layer, the ops
    after its last edit run on each layer's read, stacked layer-first; one whose rows hold different numbers of
    positions is as wide as the widest, NaN past a row's own."""
    parts, count = [[] for _ in group], Counter()
    for chunk, batch, pads in chunked(engine, group[0], found, batch_size):
        sites, records = bind(engine, plan, name, group, state, found, chunk, pads)
        got = engine.forward(batch, records)
        count += fed(batch, pads)
        for i, step in enumerate(group):
            parts[i].append(reads(engine, plan, step, i, got, sites[i], chunk, state, found.answers))
    for step, part in zip(group, parts):
        joined(state, name, step, part)
    return count


def op(engine: Engine, plan: Plan, name: str, group: list[OpStep], state: dict, batch_size: int,
       found: Found) -> Counter:
    """The op over the whole value `of` names, given each of its rows' dataset row, at the layers `of` is at
    (`acts_at`), to which an operand's `layers` axis is aligned (`bound`). `top_k`'s value over a vocabulary's
    tokens also holds them, decoded. It calls no model; `group` is its one point."""
    (step,) = group
    of = lookup(state, step.of, step.at)
    producer = points(plan.steps[step.of.partition(".")[0]])[0]
    dims = dims_of(plan, step.of)
    vocabulary = lookup(state, probe_of(plan, step.of), step.at).classes if "classes" in dims else None
    rows = torch.arange(of.shape[dims.index("rows")]) if "rows" in dims and isinstance(of, Tensor) else None
    sizes = engine.sizes()
    value = ops.call(step.op, of, bound(plan, step.op, arguments(step), state, step.at, None, found.answers, of,
                                        vocabulary, dims=dims, layers=acts_at(plan, step.of, step.at, sizes),
                                        sizes=sizes), engine=engine, dims=dims,
                     eos=generate.stops(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    if step.op == "top_k" and arguments(step).get("over", ops.transforms.RANKED) == ops.transforms.RANKED:
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)
    return Counter()


def decoded(tokenizer: PreTrainedTokenizerBase, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are; None for a missing one (-1)."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    return [tokenizer.decode([i]) if i >= 0 else None for i in ids.tolist()]


# The step kinds, each run as `(engine, plan, name, group, state, batch_size, found)` over a group of the step's
# points (`groups`: those that share a forward, else one) and returning the model calls it made and the tokens they
# fed (`batches.fed`).
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit}
