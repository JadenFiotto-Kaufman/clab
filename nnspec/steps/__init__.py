"""Walk a compiled plan inside the run: its hub datasets' rows loaded, the preflight pass, every step and point that
runs in order, then the saved values into `plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks and vllm engines, in plain Python. The cross-step state is a `ChainMap` (a fit's body runs in
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

from nnspec import ops
from nnspec.compile import dataset_rows
from nnspec.compile.arguments import select_kept, selected_layers
from nnspec.compile.preflight import at_point, preflight
from nnspec.compile.schedule import READOUTS, values_read_by
from nnspec.compile.state import Findings
from nnspec.compile.uses import instance, named_values
from nnspec.format import data
from nnspec.format.sweeps import agreeing_point, coordinate, coordinate_text, lookup, projection, put
from nnspec.plan import (Forward, Load, OpStep, Plan, Run, arguments, entries, gradient_of, made_by, points,
                         projections, record_of, runnable, saved_value_axes, serving, tensors_named, top_k_members,
                         walk)
from nnspec.steps import fit, generate
from nnspec.steps.batches import (argument_binding, bind_arguments, bind_schedule, input_batches, join_batches,
                                  layers_made_at, model_device, schedule_binding, scope_ids, stack_layers, stop_ids,
                                  value_at, value_in_prefill)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from nnspec.engines import Engine, Parameterized


def start_run(engine: Engine, plan: Plan, batch_size: int, padding: str = "dataset", readout: str = "full") -> float:
    """What every engine's `execute` checks first, on the client before any session opens, from the plan and the
    engine's declarations alone (a compiled plan, for this engine's checkpoint, at a batch size of one row or more, a
    padding `data.PADDINGS` names and a readout `schedule.READOUTS` names; no `grad` entry where the engine refuses
    one, `engine.gradless` saying why, as the refusal does after the entry's name; an engine without it, or with
    None, takes them; and where the engine's session is `remote`, which sends the plan elsewhere with its loads'
    values inside, no more than `LOADED` bytes of tensors over the document's loads, refused before anything is sent),
    and the run record it starts: the engine, the batch size, padding and readout, the device and the options it
    loaded with; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    check_run_options(batch_size, padding, readout)
    ours, theirs = plan.model.model_dump(), engine.spec.model_dump()
    differ = [k for k in dict.fromkeys([*theirs, *ours]) if theirs.get(k) != ours.get(k)]
    if differ:                  # a loader key written null is one left out
        raise ValueError("the plan's model is not the one this engine loaded: " + ", ".join(
            f"{k} {ours.get(k)!r} where the engine's is {theirs.get(k)!r}" for k in differ))
    gradless = getattr(engine, "gradless", None)
    for name, step in walk(plan.steps) if gradless else ():
        for key in [k for k, e in entries(step).items() if gradient_of(e) is not None][:1]:
            raise ValueError(f"{name}.{key}: {gradless}")
    carried: dict[str, int] = {}
    for name, step in walk(plan.steps) if engine.remote else ():
        if step.kind == "load":
            carried[name] = carried.get(name, 0) + nbytes(step.loaded)
    if sum(carried.values()) > LOADED:
        raise ValueError(f"{', '.join(carried)}: the values this document loads hold "
                         f"{sum(carried.values()) / 2**20:.1f} MiB of tensors (" + ", ".join(
                             f"{n} {b / 2**20:.1f} MiB" for n, b in carried.items()) + f"), and a request carries "
                         f"at most {LOADED // 2**20} MiB inside its plan; run it where the file is, on the nnterp "
                         "or hooks engine")
    plan.run = Run(engine=engine.name, batch_size=batch_size, padding=padding, readout=readout, device=engine.device,
                   options=engine.options, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


# The bytes of loaded tensors a request to a session elsewhere carries inside its plan, over every `load` of the
# document (`Load.loaded`): a cached operand for a large model fits, and a harvest of many rows runs where its file is.
LOADED = 512 * 2**20


def nbytes(value: Any) -> int:
    """The bytes of every tensor in a value, all the way down."""
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    return sum(nbytes(v) for v in value.values()) if isinstance(value, dict) else 0


def check_run_options(batch_size: int, padding: str, readout: str = "full") -> None:
    """`batch_size`, `padding` and `readout` as every run takes them: one row or more per batch, a padding
    `data.PADDINGS` names and a readout `schedule.READOUTS` names, each refused by name otherwise (`nnspec.run` asks
    before the model loads, `start_run` for an engine's `execute` called on its own)."""
    if batch_size < 1:
        raise ValueError(f"batch_size {batch_size}: a batch holds one row or more (--batch-size N, N at least 1)")
    if padding not in data.PADDINGS:
        raise ValueError(f"padding {padding!r}: one of {', '.join(map(repr, data.PADDINGS))} (--padding)")
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r}: one of {', '.join(map(repr, READOUTS))} (--readout)")


def run(engine: Engine, plan: Plan, batch_size: int) -> None:
    """Every step in document order, a fit's eval steps after it (`runnable`), at each point of it that runs
    (`serving`: a forward's or generation's serving points, an op's one per distinct point on its value's axes, a
    fit's every one), over the plan with its rows (`plan_with_rows`), without gradients except where a fit trains, a
    forward's batches of `batch_size` rows unless its record fixes another (a fit's eval step's: the fit's), each
    run's cost measured into `plan.run.costs` (`measure_cost`) with the points it served; then what `record_results`
    records, each saved value into `plan.results` among it, and the model's sizes and torch's thread count where the
    model is, at the start (`Run.threads`). Where each read and write
    resolved, and each featurizer's width, is on its record, which preflight and the steps complete in place. A step
    that runs out of CUDA memory stops the run: its refusal (`out_of_memory_refusal`) is recorded as
    `plan.run.stopped`, and the steps that finished are recorded and saved as if the run had ended there."""
    plan = plan_with_rows(plan)
    assert plan.run is not None
    plan.run.versions, plan.run.sizes, plan.run.threads = versions(), engine.sizes(), torch.get_num_threads()
    findings = preflight(engine, plan)
    state: ChainMap = ChainMap()
    device, devices, costs = model_device(engine), engine.cuda, {}
    with torch.no_grad():
        for name in runnable(plan.steps):
            make_featurizers(plan, state, device)
            for step, served in serving(plan, name):
                cost: dict[str, Any] = {"seconds": 0.0, "memory": None, "forwards": 0, "tokens": 0, "real": 0,
                                        **({"serves": [coordinate_text(plan, p.at) for p in served]}
                                           if step.at else {})}
                rows = step.batch_size if isinstance(step, Forward) and step.batch_size is not None else batch_size
                try:
                    measure_cost(devices, cost, lambda: KINDS[step.kind](
                        engine, plan, name, step, state, rows, findings, cost,
                        *([served] if isinstance(step, Forward) else [])), step.kind not in NO_MODEL)
                    put(costs, name, step.at, cost)
                except torch.OutOfMemoryError as e:
                    plan.run.stopped = out_of_memory_refusal(plan, name, step.at, rows, state, e)
                    record_results(engine, plan, state, costs, findings)
                    raise ValueError(plan.run.stopped) from e
    record_results(engine, plan, state, costs, findings)


def keyed_by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {coordinate_text(plan, dict(c)): v for c, v in by_point.items()}


def record_results(engine: Engine, plan: Plan, state: ChainMap, costs: dict, findings: Findings) -> None:
    """What `run` records once its steps have run, or stopped: each step's cost, each saved value of a step whose
    every run finished (`made_by`; a trained featurizer as its bundle's tensors; a featurizer's tensor once the run
    made the featurizer; a swept value keyed by its coordinates on its own axes), how each input tokenized, and the dtypes and devices of the model's parameters."""
    assert plan.run is not None
    plan.run.costs = {name: keyed_by_coordinate(plan, by_point) for name, by_point in costs.items()}
    done = {name for name in runnable(plan.steps) if len(costs.get(name, {})) == len(serving(plan, name))}
    done |= {f for f in plan.featurizers if f in state}         # made, its named tensors beside it
    for name in (n for n in plan.saves if made_by(plan, n) in done):
        thing, _, member = name.partition(".")
        step = points(plan.steps[thing])[0] if thing in plan.steps else None
        ats = [at for at, _ in projections(plan, name)]
        values = [lookup(state, name, at) for at in ats]
        if step is not None and step.kind == "fit" and member in step.train:
            values = [v.tensors() for v in values]
        plan.results[name] = ({coordinate(plan, at): detached_cpu_clone(v) for at, v in zip(ats, values)}
                              if saved_value_axes(plan, name) else detached_cpu_clone(values[0]))
    plan.run.tokens.update({name: keyed_by_coordinate(plan, by_point) for name, by_point in findings.said.items()})
    plan.run.dtype, plan.run.devices = dtypes_and_devices(engine.model)


def out_of_memory_refusal(plan: Plan, name: str, at: dict, batch_size: int, state: ChainMap,
                          error: BaseException) -> str:
    """The refusal for a step that ran out of CUDA memory: the step and its point, the batch size (a fit's, and its
    eval steps', the fit's `batch_size`), what torch said (its first sentences), and every value the run holds
    (`state`: each bound value stays on the model's device until the run ends) with its shape, dtype and device,
    then what to do."""
    where = f"steps.{name}" + (f" at {coordinate_text(plan, at)}" if at else "")
    head = name.partition(".")[0]
    owner = agreeing_point(plan.steps[head], at)             # the step itself, or the fit whose eval step it is
    if owner.kind == "fit":
        batch_size = cast(int, owner.batch_size)
    message = ". ".join(str(error).strip().split(". ")[:3]).rstrip(".")
    listed = [f"{n}{f'[{coordinate_text(plan, dict(c))}]' if c else ''} {s}" for n, by_point in state.items()
              if isinstance(by_point, dict) for c, v in by_point.items() if (s := value_text(v))]
    holds = (f"holds {', '.join(listed)}, each on the model's device until the run ends" if listed else
             "holds no value yet")
    smaller = f"give steps.{head} a smaller batch_size" if owner.kind == "fit" else "run with a smaller --batch-size"
    return (f"{where}: out of CUDA memory at batch size {batch_size} ({message}); the run {holds}; {smaller}, or name "
            "and save fewer values")


def value_text(value: Any) -> str | None:
    """A value's shape, dtype and device as a refusal lists it: a tensor's, a dict's per key that holds tensors
    (`top_k`'s `ids` and `probs`, not its token strings); None for anything else (a featurizer instance)."""
    if isinstance(value, Tensor):
        return f"{list(value.shape)} {str(value.dtype).removeprefix('torch.')} on {value.device}"
    held = {k: t for k, v in value.items() if (t := value_text(v))} if isinstance(value, dict) else {}
    return "{" + ", ".join(f"{k} {t}" for k, t in held.items()) + "}" if held else None


def measure_cost(devices: list[str], cost: dict[str, Any], call: Callable[[], None], synced: bool) -> None:
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
    server, under `ndif`), and refused unless their digest is the one compile recorded (`dataset_rows`). It is a
    view: its `results` and `run` are the plan's own, which the run fills."""
    return plan.model_copy(update={"data": {n: d.model_copy(update={"rows": dataset_rows(plan, n)})
                                            for n, d in plan.data.items()}})


def dtypes_and_devices(model: Parameterized) -> tuple[list[str], list[str]]:
    """The distinct dtypes and devices of `model`'s parameters, each in the order first met. `run` reads them
    inside the session after the steps, so they are the server's under the ndif engine, and those of a model
    that loads its weights at its first forward once it has."""
    dtypes, devices = {}, {}
    for parameter in model.parameters():
        dtypes[str(parameter.dtype).removeprefix("torch.")] = devices[str(parameter.device)] = None
    return list(dtypes), list(devices)


def make_featurizers(plan: Plan, state: ChainMap, device: str) -> None:
    """Each declared featurizer used at a site (its record says its `width` there), at each of its points, made where
    the model is (`device`) into the run's `state` once what it is drawn from is there: before the run's first step,
    or, with a field that names a value (a constant's `init`), before the first step after the one that made it,
    which compile puts before the featurizer's first use (`uses.check_named_values`), or once the featurizer whose
    tensor it names is made. Each of its tensors the plan names (`plan.tensors_named`) goes beside it, a copy
    (`Kind.value`). `run` calls it before each step; one made already stays as it is."""
    named, making = tensors_named(plan), True
    while making:               # again, for one drawn from a tensor of a featurizer made in this pass
        making = False
        for name, entry in plan.featurizers.items():
            for spec in points(entry):
                made = tuple(spec.at.items()) in state.get(name, {})
                if spec.width is None or made or any(ref not in state for ref in named_values(spec).values()):
                    continue
                featurizer = instance(plan, spec, spec.width, None, device, state)
                put(state, name, spec.at, featurizer)
                for tensor in (t for t in featurizer.laid if f"{name}.{t}" in named):
                    put(state, f"{name}.{tensor}", spec.at, featurizer.value(tensor))
                making = True


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
    """What served the run, by name: nnspec's version with its commit (`version`), the packages it runs on, and
    `python`. `run` calls it inside the session, so under the ndif engine they are the server's."""
    return {"nnspec": version(), **{p: importlib.metadata.version(p) for p in PACKAGES},
            "python": platform.python_version()}


@functools.cache
def version() -> str:
    """nnspec's version here, with the commit it is when one is known: the one an install from git recorded
    (`direct_url.json`), else `git describe --always --dirty` of the checkout nnspec is imported from, else the
    static version alone. Worked out once per process: what is imported does not change under it, and a `git
    describe` on every run costs more than some runs do."""
    dist = importlib.metadata.distribution("nnspec")
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


def forward(engine: Engine, plan: Plan, name: str, step: Forward, state: ChainMap, batch_size: int, findings: Findings,
            cost: dict[str, Any], served: list[Forward], values_read: dict | None = None,
            binding: list[dict] | None = None) -> None:
    """The step's input in batches of dataset row ids padded as the run says (`input_batches`, `plan.run.padding`), each
    through `engine.forward` with the point's schedule bound to it (`bind_schedule`, from the half of its binding that
    depends on the step alone, `binding`: `schedule_binding`'s, made once for every batch); each value the forward
    reads (`values_read`, `values_read_by` over the points its run serves, `served`: each gathered entry at each
    distinct point on its axes among them) joined along rows from its own reads, in the run's row order, but one the
    schedule frees (`freed`: read only for the forward's own writes, which take each batch's read live), of which no
    batch is kept. A fit gives both once for its every run of the body. Adds to `cost` the model calls it made, one
    per batch, the prompt tokens they fed, padding included, and the rows' own of them (`real`).

    An entry whose `layers` are a slice or picks is one write (when it writes) and one read per layer, the ops
    after its last edit run on each layer's read, stacked layer-first; one whose rows hold different numbers of
    positions is as wide as the widest, NaN past a row's own."""
    assert plan.run is not None
    values_read = values_read_by(served) if values_read is None else values_read
    binding = schedule_binding(engine, plan, name, step, values_read) if binding is None else binding
    freed = step.schedule.get("freed", [])
    parts: dict[tuple[str, tuple], list] = {k: [] for k in values_read if k[0] not in freed}
    batched_ids: list[int] = []
    for ids, batch, pads in input_batches(engine, step, state, findings, batch_size, plan.run.padding):
        batched_ids += ids
        bound_schedule, binds = bind_schedule(engine, plan, step, binding, state, findings, ids, pads)
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
        for k in parts:
            record, own, point = values_read[k]
            parts[k].append(stack_layers(record, [
                value_in_prefill(engine, plan, record, got[binds[k + (layer,)][0]["name"]], binds[k + (layer,)][0], ids,
                                 state, point.at, findings, layer)
                for layer in layers_made_at(record)]))
    for k, part in parts.items():
        record, own, _ = values_read[k]
        join_batches(state, f"{name}.{k[0]}", record, own, part, batched_ids)


def op(engine: Engine, plan: Plan, name: str, step: OpStep, state: ChainMap, batch_size: int, findings: Findings,
       cost: dict[str, Any]) -> None:
    """The op over the value `of` names at the rows the run is over (`scope_ids`, else every row of its dataset),
    given each one's dataset row, at the layers `of`'s record says it is at (its `layers` axis's labels, or the one
    layer it was read at), to which an operand's `layers` axis is aligned (`argument_binding`); a `classes` axis's
    vocabulary is the probe its coordinates are in, and the ids a generation stops on are those of the generation that
    made `of`, through any `permute` of it. `top_k`'s value over a vocabulary's tokens also holds them, decoded,
    and a `permute`'s record the permutation it drew (`OpStep.permutation`), and a `select`'s the indices it kept along
    each axis, checked against the value's own entries (an axis whose number only the run knows: `arguments.select_kept`,
    refused naming the step). It calls no model, and its value is put at the point's coordinates on its own axes."""
    of = value_at(plan, state, step.of, step.at, scope_ids(state))
    record = record_of(plan.steps, step.of, step.at, plan.featurizers)
    dims = record.dims or []
    makers = {**plan.featurizers, **runnable(plan.steps)}       # a featurizer's tensor is the featurizer's
    producer = points(makers[made_by(plan, step.of)])[0]
    while isinstance(producer, OpStep) and producer.op == "permute":     # what the value is: a generation's ids too
        producer = points(makers[made_by(plan, producer.of)])[0]
    ids = (scope_ids(state) if scope_ids(state) is not None else list(range(plan.data[record.dataset].n_rows or 0))) \
        if "rows" in dims and record.dataset is not None else None
    rows = None if ids is None else torch.tensor(ids)
    vocabulary = lookup(state, record.space[-1]["featurizer"], step.at).classes if "classes" in dims else None
    acts = record.layers if "layers" in dims else record.layers[0] if record.layers else None
    binding = argument_binding(plan, step.op, arguments(step), step.at, dims=dims, layers=acts)
    if step.op == "select":     # its picks among the value's own entries, which the run alone holds along some axes
        met = record.model_copy(update={"kept": {a: v for a, v in record.kept.items() if v != []}})
        step.layers, step.kept = selected_layers(record, step.kept | select_kept(
            at_point(plan, name, step.at), met, arguments(step), None,
            {a: (of.shape[dims.index(a)], "the value's") for a in arguments(step)["select"]}))
    value = ops.call(step.op, of, bind_arguments(plan, binding, state, scope_ids(state), findings, of, vocabulary,
                                                 dims=dims),
                     engine=engine, dims=dims, layers=acts,
                     eos=stop_ids(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    over = arguments(step).get("over", ops.transforms.RANKED) if step.op == "top_k" else None
    if over == ops.transforms.RANKED or isinstance(over, list) and ops.transforms.RANKED in over:
        value["tokens"] = token_strings(engine.tokenizer, value["ids"] if over == ops.transforms.RANKED else
                                        value["ids"][ops.transforms.RANKED])
    if step.op == "permute":        # what it drew, on its record: the dataset row each row's value is
        order = ops.transforms.permutation(arguments(step)["seed"], of.shape[dims.index("rows")],
                                           arguments(step).get("derangement", False))
        step.permutation = order.tolist() if rows is None else rows[order].tolist()
    put(state, name, projection(step.at, step.axes), value)
    for member in top_k_members(step) if step.op == "top_k" else []:      # each by its own name, as a fit's are
        part = value
        for key in member.split("."):
            part = part[key]
        put(state, f"{name}.{member}", projection(step.at, step.axes), part)


def load(engine: Engine, plan: Plan, name: str, step: Load, state: ChainMap, batch_size: int, findings: Findings,
         cost: dict[str, Any]) -> None:
    """The saved value the plan carries (`Load.loaded`, which the compiled plan read from its run's file and which
    travels with it, in a request too), put where the model is (`model_device`), every tensor of it, at the point's
    coordinates on the value's own axes. It calls no model and reads no file."""
    assert step.loaded is not None, f"{name}: a compiled plan holds each load's value until it has run"
    put(state, name, projection(step.at, step.axes), move_to_device(step.loaded, model_device(engine)))


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


# The step kinds, each run as `(engine, plan, name, step, state, batch_size, findings, cost)`, adding the model calls
# it made and the tokens they fed (`tokens`, and the rows' own, `real`) to `cost` (a forward and a generation also take
# the points their run serves, `served`, and what they read, `values_read`).
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit, "load": load}

# The step kinds that call no model, whose runs `measure_cost` times without synchronizing the model's devices.
NO_MODEL = ("op", "load")
