"""Walk a compiled plan inside the run: its hub datasets' rows loaded, the preflight pass, every step and point that
runs in order, then the saved values into `plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a `ChainMap` (a fit's body runs in
a child of it, carrying the minibatch's row ids: `batches.ROWS`) that never leaves the run: each value under its
name, by its coordinates on its own axes (`Labelled.axes`).
The steps run in the order `plan.runnable` lists them, a fit's eval steps after it. A forward or generate step
runs at the points no other point's forward serves (`plan.serving`), executing the point's schedule batch by batch
(`batches.bind_schedule`); an op step once per distinct point on its value's axes, and a load as one, putting a saved
value where the model is; each adds the model calls it made and the tokens they fed to its `cost`, which `run` records
with its time and memory (`measure_cost`).
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import ChainMap
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
from torch import Tensor

from clab import ops
from clab.compile import materialized
from clab.compile.preflight import preflight
from clab.compile.schedule import gathered
from clab.compile.state import Found
from clab.compile.uses import instance
from clab.format import data
from clab.format.sweeps import agreeing, by_coordinate, coordinate, labelled, lookup, projected, put
from clab.plan import (Forward, Load, OpStep, Plan, Run, arguments, made_by, points, projections, record_of, runnable,
                        serving)
from clab.steps import fit, generate
from clab.steps.batches import (bind_arguments, bind_schedule, input_batches, join_batches, layers_made_at,
                                model_device, scope_ids, stack_layers, stop_ids, value_at, value_in_prefill)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine, Parameterized


def start_run(engine: Engine, plan: Plan, batch_size: int, padding: str = "dataset") -> float:
    """What every engine's `execute` checks first (a compiled plan, for this engine's checkpoint, at a batch size of
    one row or more and a padding `data.PADDINGS` names), and the run record it starts: the engine, the batch size
    and padding, the device and the options it loaded with; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    check_run_options(batch_size, padding)
    ours, theirs = plan.model.model_dump(), engine.spec.model_dump()
    differ = [k for k in dict.fromkeys([*theirs, *ours]) if theirs.get(k) != ours.get(k)]
    if differ:                  # a loader key written null is one left out
        raise ValueError("the plan's model is not the one this engine loaded: " + ", ".join(
            f"{k} {ours.get(k)!r} where the engine's is {theirs.get(k)!r}" for k in differ))
    plan.run = Run(engine=engine.name, batch_size=batch_size, padding=padding, device=engine.device,
                   options=engine.options, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def check_run_options(batch_size: int, padding: str = "dataset") -> int:
    """`batch_size` and `padding` as every run takes them: one row or more per batch, and a padding
    `data.PADDINGS` names, each refused by name otherwise (`clab.run` asks before the model loads, `start_run` for an
    engine's `execute` called on its own)."""
    if batch_size < 1:
        raise ValueError(f"batch_size {batch_size}: a batch holds one row or more (--batch-size N, N at least 1)")
    if padding not in data.PADDINGS:
        raise ValueError(f"padding {padding!r}: one of {', '.join(map(repr, data.PADDINGS))} (--padding)")
    return batch_size


def run(engine: Engine, plan: Plan, batch_size: int) -> None:
    """Every step in document order, a fit's eval steps after it (`runnable`), at each point of it that runs
    (`serving`: a forward's or generation's serving points, an op's one per distinct point on its value's axes, a
    fit's every one), over the plan with its rows (`plan_with_rows`), without gradients except where a fit trains, a
    forward's batches of `batch_size` rows unless its record fixes another (a fit's eval step's: the fit's), each
    run's cost measured into `plan.run.costs` (`measure_cost`) with the points it served; then (`record_results`)
    each saved value into `plan.results` (a trained featurizer as its bundle's tensors), and what served the run: the
    model's sizes, and the dtypes and devices of the model's parameters (`dtypes_and_devices`). Where each read and
    write resolved, and each featurizer's width, is on its record, which preflight and the steps complete in place. A
    step that runs out of CUDA memory stops the run: the refusal (`out_of_memory_refusal`) names the step, the batch
    size and every value the run holds, is recorded as `plan.run.stopped`, and the steps that finished are recorded
    and saved as if the run had ended there."""
    plan = plan_with_rows(plan)
    assert plan.run is not None
    plan.run.versions, plan.run.sizes = versions(), engine.sizes()
    found = preflight(engine, plan)
    state = state_at_start(plan, model_device(engine))
    devices, costs = engine.cuda, {}
    with torch.no_grad():
        for name in runnable(plan.steps):
            for step, served in serving(plan, name):
                cost: dict[str, Any] = {"seconds": 0.0, "memory": None, "forwards": 0, "tokens": 0, "real": 0,
                                        **({"serves": [labelled(plan, p.at) for p in served]} if step.at else {})}
                rows = step.batch_size if isinstance(step, Forward) and step.batch_size is not None else batch_size
                try:
                    measure_cost(devices, cost, lambda: KINDS[step.kind](
                        engine, plan, name, step, state, rows, found, cost,
                        *([served] if isinstance(step, Forward) else [])), step.kind not in NO_MODEL)
                    put(costs, name, step.at, cost)
                except torch.OutOfMemoryError as e:
                    plan.run.stopped = out_of_memory_refusal(plan, name, step.at, rows, state, e)
                    record_results(engine, plan, state, costs, found)
                    raise ValueError(plan.run.stopped) from e
    record_results(engine, plan, state, costs, found)


def record_results(engine: Engine, plan: Plan, state: ChainMap, costs: dict, found: Found) -> None:
    """What `run` records once its steps have run, or stopped: each step's cost, each saved value of a step whose
    every run finished (`made_by`; a trained featurizer as its bundle's tensors; a swept value keyed by its
    coordinates on its own axes), how each input tokenized, and the dtypes and devices of the model's parameters."""
    assert plan.run is not None
    plan.run.costs = {name: by_coordinate(plan, by_point) for name, by_point in costs.items()}
    done = {name for name in runnable(plan.steps) if len(costs.get(name, {})) == len(serving(plan, name))}
    for name in (n for n in plan.saves if made_by(plan, n) in done):
        ats = [at for at, _ in projections(plan, name)]
        plan.results[name] = ({coordinate(plan, at): detached_cpu_clone(saved(plan, state, name, at)) for at in ats}
                              if ats != [{}] else detached_cpu_clone(saved(plan, state, name, {})))
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found.said.items()})
    plan.run.dtype, plan.run.devices = dtypes_and_devices(engine.model)


def out_of_memory_refusal(plan: Plan, name: str, at: dict, batch_size: int, state: ChainMap,
                          error: BaseException) -> str:
    """The refusal for a step that ran out of CUDA memory: the step and its point, the batch size (a fit's, and its
    eval steps', the fit's `batch_size`), what torch said (its first sentences), and every value the run holds
    (`state`: each bound value stays on the model's device until the run ends) with its shape, dtype and device,
    then what to do."""
    where = f"steps.{name}" + (f" at {labelled(plan, at)}" if at else "")
    head = name.partition(".")[0]
    owner = agreeing(plan.steps[head], at)             # the step itself, or the fit whose eval step it is
    if owner.kind == "fit":
        batch_size = cast(int, owner.batch_size)
    message = ". ".join(str(error).strip().split(". ")[:3]).rstrip(".")
    listed = [f"{n}{f'[{labelled(plan, dict(c))}]' if c else ''} {s}" for n, by_point in state.items()
              if isinstance(by_point, dict) for c, v in by_point.items() if (s := value_text(v))]
    holds = (f"holds {', '.join(listed)}, each on the model's device until the run ends" if listed else
             "holds no value yet")
    smaller = f"give steps.{head} a smaller batch_size" if owner.kind == "fit" else "run with a smaller --batch-size"
    return (f"{where}: out of CUDA memory at batch size {batch_size} ({message}); the run {holds}; {smaller}, or name "
            "and save fewer values")


def value_text(value: Any) -> str | None:
    """A value's shape, dtype and device as a refusal lists it: a tensor's, a dict's per tensor-valued key
    (`top_k`'s `ids` and `probs`, not its token strings); None for anything else (a featurizer instance)."""
    if isinstance(value, Tensor):
        return f"{list(value.shape)} {str(value.dtype).removeprefix('torch.')} on {value.device}"
    if isinstance(value, dict) and any(isinstance(v, Tensor) for v in value.values()):
        return "{" + ", ".join(f"{k} {value_text(v)}" for k, v in value.items() if isinstance(v, Tensor)) + "}"
    return None


def measure_cost(devices: list[str], cost: dict[str, Any], call: Callable[[], None], synced: bool = True) -> None:
    """What a run of a step cost, written into `cost` beside the model calls it made and the tokens they fed,
    which `call`, running it, adds (`forwards`, `tokens`): its time (`seconds`), with the model's CUDA `devices`
    synchronized at both ends where it calls the model (`synced`), and the peak memory allocated on each of them
    while it ran, weights included, in bytes by device (`memory`, None where the model is not on CUDA), which the
    allocator counts as the host asks for memory, so reading it waits for nothing. A step that calls no model (an op
    step, a load: `NO_MODEL`) is timed as the host runs it, without draining the devices' queues: its kernels finish
    under the next step that calls the model, before that one's timer starts. The engine says which devices it
    measures (`engine.cuda`, read once per run by `run`, after preflight, where the model is placed: a walk of its
    parameters): a remote engine's run makes no CUDA call and its `memory` is None."""
    for device in devices:
        if synced:
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    call()
    for device in devices if synced else ():
        torch.cuda.synchronize(device)
    cost["seconds"] = round(time.perf_counter() - started, 3)
    cost["memory"] = {d: torch.cuda.max_memory_allocated(d) for d in devices} or None


def cuda_devices(engine: Engine) -> list[str]:
    """The CUDA devices the model is on: where its parameters are (a meta shell's, which loads its weights at its
    first forward, where the engine places it: `model_device`). What an engine that runs here answers for `cuda`."""
    devices = dict.fromkeys(model_device(engine) if d == "meta" else d for d in dtypes_and_devices(engine.model)[1])
    return [d for d in devices if d.startswith("cuda")]


def plan_with_rows(plan: Plan) -> Plan:
    """The plan as the run reads it: every dataset's rows in it, a hub dataset's loaded here, where the run is (on the
    server, under `ndif`), and refused unless their digest is the one compile recorded (`materialized`). It is a
    view: its `results` and `run` are the plan's own, which the run fills."""
    return plan.model_copy(update={"data": {n: d.model_copy(update={"rows": materialized(plan, n)})
                                            for n, d in plan.data.items()}})


def dtypes_and_devices(model: Parameterized) -> tuple[list[str], list[str]]:
    """The distinct dtypes and devices of `model`'s parameters, each in the order first met. `run` reads them
    inside the session after the steps, so they are the server's under the ndif engine, and those of a model
    that loads its weights at its first forward once it has."""
    dtypes, devices = {}, {}
    for parameter in model.parameters():
        dtypes[str(parameter.dtype).removeprefix("torch.")] = devices[str(parameter.device)] = None
    return list(dtypes), list(devices)


def saved(plan: Plan, state: ChainMap, name: str, at: dict) -> Any:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    value = lookup(state, name, at)
    return value.tensors() if step.kind == "fit" and member in step.train else value


def state_at_start(plan: Plan, device: str) -> ChainMap:
    """The run's state at its start: each declared featurizer used at a site (its record says its `width` there),
    at each of its points, made where the model is (`device`)."""
    state: ChainMap = ChainMap()
    for name, entry in plan.featurizers.items():
        for spec in points(entry):
            if spec.width is not None:
                put(state, name, spec.at, instance(plan, spec, spec.width, None, device))
    return state


def detached_cpu_clone(value: Any) -> Any:
    """Detached CPU clones all the way down; a result never holds a view of a larger buffer."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: detached_cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detached_cpu_clone(v) for v in value]
    return value


PACKAGES = ("nnsight", "nnterp", "torch", "transformers")


def versions() -> dict[str, str]:
    """What served the run, by name: clab's version with its commit (`version`), the packages it runs on, and
    `python`. `run` calls it inside the session, so under the ndif engine they are the server's."""
    return {"clab": version(), **{p: importlib.metadata.version(p) for p in PACKAGES},
            "python": platform.python_version()}


@functools.cache
def version() -> str:
    """clab's version here, with the commit it is when one is known: the one an install from git recorded
    (`direct_url.json`), else `git describe --always --dirty` of the checkout clab is imported from, else the
    static version alone. Worked out once per process: what is imported does not change under it, and a `git
    describe` on every run costs more than some runs do."""
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


def forward(engine: Engine, plan: Plan, name: str, step: Forward, state: ChainMap, batch_size: int, found: Found,
            cost: dict[str, Any], served: list[Forward], values_read: dict | None = None) -> None:
    """The step's input in batches of dataset row ids padded as the run says (`input_batches`, `plan.run.padding`), each
    through `engine.forward` with the point's schedule bound to it (`bind_schedule`); each value the forward reads
    (`values_read`, `gathered` over the points its run serves, `served`: each gathered entry at each distinct point on
    its axes among them, which a fit gives once for its every run of the body) joined along rows from its own reads,
    in the run's row order. Adds to `cost` the model calls it made, one per batch, the prompt tokens they fed,
    padding included, and the rows' own of them (`real`).

    An entry whose `layers` are a slice or picks is one write (when it writes) and one read per layer, the ops
    after its last edit run on each layer's read, stacked layer-first; one whose rows hold different numbers of
    positions is as wide as the widest, NaN past a row's own."""
    assert plan.run is not None
    values_read = gathered(served) if values_read is None else values_read
    parts: dict[tuple[str, tuple], list] = {k: [] for k in values_read}
    batched_ids: list[int] = []
    for ids, batch, pads in input_batches(engine, step, state, found, batch_size, plan.run.padding):
        batched_ids += ids
        bound_schedule, binds = bind_schedule(engine, plan, name, step, state, found, ids, pads, values_read)
        got = engine.forward(batch, bound_schedule)
        missing = [r["name"] for s in bound_schedule["sites"] for r in s["records"]
                   if "name" in r and r["name"] not in got]
        if missing:
            raise RuntimeError(f"{name}: the {engine.name} engine ended the forward at {bound_schedule['stop']} "
                               f"without reading {', '.join(missing)}; the schedule's stop site comes after its last "
                               "read")
        cost["forwards"] += 1
        cost["tokens"] += batch["input_ids"].numel()
        cost["real"] += batch["input_ids"].numel() - sum(pads)
        for k, (record, own, point) in values_read.items():
            parts[k].append(stack_layers(record, [
                value_in_prefill(engine, plan, record, got[binds[k + (layer,)][0]["name"]], binds[k + (layer,)][0], ids,
                                 state, point.at, found, layer)
                for layer in layers_made_at(engine, record)]))
    for k, (record, own, _) in values_read.items():
        join_batches(state, f"{name}.{k[0]}", record, own, parts[k], batched_ids)


def op(engine: Engine, plan: Plan, name: str, step: OpStep, state: ChainMap, batch_size: int, found: Found,
       cost: dict[str, Any]) -> None:
    """The op over the value `of` names at the rows the run is over (`scope_ids`, else every row of its dataset),
    given each one's dataset row, at the layers `of`'s record says it is at (its `layers` axis's labels, or the one
    layer it was read at), to which an operand's `layers` axis is aligned (`bind_arguments`); a `classes` axis's
    vocabulary is the probe its coordinates are in. `top_k`'s value over a vocabulary's tokens also holds them, decoded.
    It calls no model, and its value is put at the point's coordinates on its own axes."""
    of, record = value_at(plan, state, step.of, step.at, scope_ids(state)), record_of(plan.steps, step.of, step.at)
    dims = record.dims or []
    producer = points(runnable(plan.steps)[made_by(plan, step.of)])[0]
    ids = (scope_ids(state) if scope_ids(state) is not None else list(range(plan.data[record.dataset].n_rows or 0))) \
        if "rows" in dims and record.dataset is not None else None
    rows = None if ids is None else torch.tensor(ids)
    vocabulary = lookup(state, record.space[-1]["featurizer"], step.at).classes if "classes" in dims else None
    acts = record.layers if "layers" in dims else record.layers[0] if record.layers else None
    value = ops.call(step.op, of, bind_arguments(plan, step.op, arguments(step), state, step.at, scope_ids(state),
                                                 found, of, vocabulary, dims=dims, layers=acts),
                     engine=engine, dims=dims, layers=acts,
                     eos=stop_ids(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    if step.op == "top_k" and arguments(step).get("over", ops.transforms.RANKED) == ops.transforms.RANKED:
        value["tokens"] = token_strings(engine.tokenizer, value["ids"])
    put(state, name, projected(step.at, step.axes), value)


def load(engine: Engine, plan: Plan, name: str, step: Load, state: ChainMap, batch_size: int, found: Found,
         cost: dict[str, Any]) -> None:
    """The saved value the plan carries (`Load.loaded`, which the compiled plan read from its run's file and which
    travels with it, in a request too), put where the model is (`model_device`), every tensor of it, at the point's
    coordinates on the value's own axes. It calls no model and reads no file."""
    assert step.loaded is not None, f"{name}: a compiled plan holds each load's value until it has run"
    put(state, name, projected(step.at, step.axes), move_to_device(step.loaded, model_device(engine)))


def move_to_device(value: Any, device: str) -> Any:
    """A value with every tensor in it on `device`, all the way down."""
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def token_strings(tokenizer: PreTrainedTokenizerBase, ids: Tensor) -> list:
    """Each token id as its token string, nested as the ids are; None for a missing one (-1). The tokenizer's
    `convert_ids_to_tokens` is the one spelling of a token wherever a run shows one (`pos_token`, `run.tokens`,
    `detokenize` with `tokens`), so a word-start marker (`▁Fed`, `ĠFed`) stays, where `decode` would drop it."""
    if ids.dim() > 1:
        return [token_strings(tokenizer, x) for x in ids]
    texts = iter(tokenizer.convert_ids_to_tokens([i for i in ids.tolist() if i >= 0]))
    return [next(texts) if i >= 0 else None for i in ids.tolist()]


# The step kinds, each run as `(engine, plan, name, step, state, batch_size, found, cost)`, adding the model calls
# it made and the tokens they fed (`tokens`, and the rows' own, `real`) to `cost` (a forward and a generation also take
# the points their run serves, `served`, and what they read, `values_read`).
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit, "load": load}

# The step kinds that call no model, whose runs `measure_cost` times without synchronizing the model's devices.
NO_MODEL = ("op", "load")
