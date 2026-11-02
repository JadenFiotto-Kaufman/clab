"""Walk a compiled plan inside the run: its hub datasets' rows loaded, the preflight pass, every step and point that
runs in order, then the saved values into `plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a `ChainMap` (a fit's body runs in
a child of it, carrying the minibatch's row ids: `batches.ROWS`) that never leaves the run: each value under its
name, by its coordinates on its own axes (`Labelled.axes`).
A forward or generate step runs at the points no other point's forward serves (`plan.serving`), executing the
point's schedule batch by batch (`batches.bind`); an op step once per distinct point on its value's axes; each adds
the model calls it made and the tokens they fed to its `cost`, which `run` records with its time and memory
(`measured`).
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
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from clab import ops
from clab.compile import materialized
from clab.compile.preflight import preflight
from clab.compile.schedule import gathered
from clab.compile.state import Found
from clab.compile.uses import instance
from clab.format import data
from clab.format.sweeps import by_coordinate, coordinate, labelled, lookup, projected, put
from clab.plan import Forward, OpStep, Plan, Run, arguments, points, projections, record_of, serving
from clab.steps import fit, generate
from clab.steps.batches import bind, bound, chunked, home, joined, layers, over, read, stacked, stops, value_at

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine, Parameterized


def begin(engine: Engine, plan: Plan, batch_size: int, padding: str = "dataset") -> float:
    """What every engine's `execute` checks first (a compiled plan, for this engine's checkpoint, at a batch size of
    one row or more and a padding `data.PADDINGS` names), and the run record it starts: the engine, the batch size
    and padding, the device and the options it loaded with; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    batched(batch_size, padding)
    ours, theirs = plan.model.model_dump(), engine.spec.model_dump()
    differ = [k for k in dict.fromkeys([*theirs, *ours]) if theirs.get(k) != ours.get(k)]
    if differ:                  # a loader key written null is one left out
        raise ValueError("the plan's model is not the one this engine loaded: " + ", ".join(
            f"{k} {ours.get(k)!r} where the engine's is {theirs.get(k)!r}" for k in differ))
    plan.run = Run(engine=engine.name, batch_size=batch_size, padding=padding, device=engine.device,
                   options=engine.options, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def batched(batch_size: int, padding: str = "dataset") -> int:
    """`batch_size` and `padding` as every run takes them: one row or more per batch, and a padding
    `data.PADDINGS` names, each refused by name otherwise (`clab.run` asks before the model loads, `begin` for an
    engine's `execute` called on its own)."""
    if batch_size < 1:
        raise ValueError(f"batch_size {batch_size}: a batch holds one row or more (--batch-size N, N at least 1)")
    if padding not in data.PADDINGS:
        raise ValueError(f"padding {padding!r}: one of {', '.join(map(repr, data.PADDINGS))} (--padding)")
    return batch_size


def run(engine: Engine, plan: Plan, batch_size: int) -> None:
    """Every step in document order, at each point of it that runs (`serving`: a forward's or generation's serving
    points, an op's one per distinct point on its value's axes, a fit's every one), over the plan `held`, without
    gradients except where a fit trains, each run's cost `measured` into `plan.run.costs` with the points it
    served; then (`ended`) each saved value into `plan.results` (a trained featurizer as its bundle's tensors), and
    what served the run: the model's sizes, and the dtypes and devices of the model's parameters (`placed`). Where
    each read and write resolved, and each featurizer's width, is on its record, which preflight and the steps
    complete in place. A step that runs out of CUDA memory stops the run: the refusal (`exhausted`) names the step,
    the batch size and every value the run holds, is recorded as `plan.run.stopped`, and the steps that finished
    are recorded and saved as if the run had ended there."""
    plan = held(plan)
    assert plan.run is not None
    plan.run.versions, plan.run.sizes = versions(), engine.sizes()
    found = preflight(engine, plan)
    state = made(plan, home(engine))
    devices, costs = engine.cuda, {}
    with torch.no_grad():
        for name in plan.steps:
            for step, served in serving(plan, name):
                cost: dict[str, Any] = {"seconds": 0.0, "memory": None, "forwards": 0, "tokens": 0,
                                        **({"serves": [labelled(plan, p.at) for p in served]} if step.at else {})}
                try:
                    measured(devices, cost, lambda: KINDS[step.kind](
                        engine, plan, name, step, state, batch_size, found, cost,
                        *([served] if isinstance(step, Forward) else [])))
                    put(costs, name, step.at, cost)
                except torch.OutOfMemoryError as e:
                    plan.run.stopped = exhausted(plan, name, step.at, batch_size, state, e)
                    ended(engine, plan, state, costs, found)
                    raise ValueError(plan.run.stopped) from e
    ended(engine, plan, state, costs, found)


def ended(engine: Engine, plan: Plan, state: ChainMap, costs: dict, found: Found) -> None:
    """What `run` records once its steps have run, or stopped: each step's cost, each saved value of a step whose
    every run finished (a trained featurizer as its bundle's tensors; a swept value keyed by its coordinates on its
    own axes), how each input tokenized, and the dtypes and devices of the model's parameters."""
    assert plan.run is not None
    plan.run.costs = {name: by_coordinate(plan, by_point) for name, by_point in costs.items()}
    done = {name for name in plan.steps if len(costs.get(name, {})) == len(serving(plan, name))}
    for name in (n for n in plan.saves if n.partition(".")[0] in done):
        ats = [at for at, _ in projections(plan, name)]
        plan.results[name] = ({coordinate(plan, at): finished(saved(plan, state, name, at)) for at in ats}
                              if ats != [{}] else finished(saved(plan, state, name, {})))
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found.said.items()})
    plan.run.dtype, plan.run.devices = placed(engine.model)


def exhausted(plan: Plan, name: str, at: dict, batch_size: int, state: ChainMap, error: BaseException) -> str:
    """The refusal for a step that ran out of CUDA memory: the step and its point, the batch size, what torch said
    (its first sentences), and every value the run holds (`state`: each bound value stays on the model's device
    until the run ends) with its shape, dtype and device, then what to do."""
    where = f"steps.{name}" + (f" at {labelled(plan, at)}" if at else "")
    said = ". ".join(str(error).strip().split(". ")[:3]).rstrip(".")
    held = [f"{n}{f'[{labelled(plan, dict(c))}]' if c else ''} {s}" for n, by_point in state.items()
            if isinstance(by_point, dict) for c, v in by_point.items() if (s := sized(v))]
    holds = (f"holds {', '.join(held)}, each on the model's device until the run ends" if held else
             "holds no value yet")
    return (f"{where}: out of CUDA memory at batch size {batch_size} ({said}); the run {holds}; run with a smaller "
            "--batch-size, or name and save fewer values")


def sized(value: Any) -> str | None:
    """A value's shape, dtype and device as a refusal lists it: a tensor's, a dict's per tensor-valued key
    (`top_k`'s `ids` and `probs`, not its token strings); None for anything else (a featurizer instance)."""
    if isinstance(value, Tensor):
        return f"{list(value.shape)} {str(value.dtype).removeprefix('torch.')} on {value.device}"
    if isinstance(value, dict) and any(isinstance(v, Tensor) for v in value.values()):
        return "{" + ", ".join(f"{k} {sized(v)}" for k, v in value.items() if isinstance(v, Tensor)) + "}"
    return None


def measured(devices: list[str], cost: dict[str, Any], call: Callable[[], None]) -> None:
    """What a run of a step cost, written into `cost` beside the model calls it made and the tokens they fed,
    which `call`, running it, adds (`forwards`, `tokens`): its time (`seconds`), with the model's CUDA `devices`
    synchronized at both ends, and the peak memory allocated on each of them while it ran, weights included, in
    bytes by device (`memory`, None where the model is not on CUDA). The engine says which devices it measures
    (`engine.cuda`, read once per run by `run`, after preflight, where the model is placed: a walk of its
    parameters): a remote engine's run makes no CUDA call and its `memory` is None."""
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    call()
    for device in devices:
        torch.cuda.synchronize(device)
    cost["seconds"] = round(time.perf_counter() - started, 3)
    cost["memory"] = {d: torch.cuda.max_memory_allocated(d) for d in devices} or None


def gpus(engine: Engine) -> list[str]:
    """The CUDA devices the model is on: where its parameters are (a meta shell's, which loads its weights at its
    first forward, where the engine places it: `home`). What an engine that runs here answers for `cuda`."""
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


def saved(plan: Plan, state: ChainMap, name: str, at: dict) -> Any:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    value = lookup(state, name, at)
    return value.tensors() if step.kind == "fit" and member in step.train else value


def made(plan: Plan, device: str) -> ChainMap:
    """The run's state at its start: each declared featurizer used at a site (its record says its `width` there),
    at each of its points, made where the model is (`device`)."""
    state: ChainMap = ChainMap()
    for name, entry in plan.featurizers.items():
        for spec in points(entry):
            if spec.width is not None:
                put(state, name, spec.at, instance(plan, spec, spec.width, None, device))
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
            cost: dict[str, Any], served: list[Forward], reading: dict | None = None) -> None:
    """The step's input in batches of dataset row ids padded as the run says (`chunked`, `plan.run.padding`), each
    through `engine.forward` with the point's schedule bound to it (`bind`); each value the forward reads
    (`reading`, `gathered` over the points its run serves, `served`: each gathered entry at each distinct point on
    its axes among them, which a fit gives once for its every run of the body) joined along rows from its own
    reads. Adds the model calls it made, one per batch, and the prompt tokens they fed to `cost`.

    An entry whose `layers` are a slice or picks is one write (when it writes) and one read per layer, the ops
    after its last edit run on each layer's read, stacked layer-first; one whose rows hold different numbers of
    positions is as wide as the widest, NaN past a row's own."""
    assert plan.run is not None
    reading = gathered(served) if reading is None else reading
    parts: dict[tuple[str, tuple], list] = {k: [] for k in reading}
    for ids, batch, pads in chunked(engine, step, state, found, batch_size, plan.run.padding):
        bound, binds = bind(engine, plan, step, state, found, ids, pads, reading)
        got = engine.forward(batch, bound)
        missing = [r["name"] for s in bound["sites"] for r in s["records"] if "name" in r and r["name"] not in got]
        if missing:
            raise RuntimeError(f"{name}: the {engine.name} engine ended the forward at {bound['stop']} without "
                               f"reading {', '.join(missing)}; the schedule's stop site comes after its last read")
        cost["forwards"] += 1
        cost["tokens"] += batch["input_ids"].shape[1] * len(pads) - sum(pads)
        for k, (record, own, p) in reading.items():
            parts[k].append(stacked(record, [read(engine, plan, record, got[binds[k + (layer,)][0]["name"]],
                                                  binds[k + (layer,)][0], ids, state, p.at, found, layer)
                                             for layer in layers(engine, record)]))
    for k, (record, own, _) in reading.items():
        joined(state, f"{name}.{k[0]}", record, own, parts[k])


def op(engine: Engine, plan: Plan, name: str, step: OpStep, state: ChainMap, batch_size: int, found: Found,
       cost: dict[str, Any]) -> None:
    """The op over the value `of` names at the rows the run is over (`over`, else every row of its dataset),
    given each one's dataset row, at the layers `of`'s record says it is at (its `layers` axis's labels, or the one
    layer it was read at), to which an operand's `layers` axis is aligned (`bound`); a `classes` axis's vocabulary
    is the probe its coordinates are in. `top_k`'s value over a vocabulary's tokens also holds them, decoded. It
    calls no model, and its value is put at the point's coordinates on its own axes."""
    of, record = value_at(plan, state, step.of, step.at, over(state)), record_of(plan.steps, step.of, step.at)
    dims = record.dims or []
    producer = points(plan.steps[step.of.partition(".")[0]])[0]
    ids = (over(state) if over(state) is not None else list(range(plan.data[record.dataset].n_rows or 0))) \
        if "rows" in dims and record.dataset is not None else None
    rows = None if ids is None else torch.tensor(ids)
    vocabulary = lookup(state, record.space[-1]["featurizer"], step.at).classes if "classes" in dims else None
    acts = record.layers if "layers" in dims else record.layers[0] if record.layers else None
    value = ops.call(step.op, of, bound(plan, step.op, arguments(step), state, step.at, over(state), found, of,
                                        vocabulary, dims=dims, layers=acts), engine=engine, dims=dims,
                     eos=stops(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    if step.op == "top_k" and arguments(step).get("over", ops.transforms.RANKED) == ops.transforms.RANKED:
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, projected(step.at, step.axes), value)


def decoded(tokenizer: PreTrainedTokenizerBase, ids: Tensor) -> list:
    """Each token id as its token string, nested as the ids are; None for a missing one (-1). The tokenizer's
    `convert_ids_to_tokens` is the one spelling of a token wherever a run shows one (`pos_token`, `run.tokens`,
    `detokenize` with `tokens`), so a word-start marker (`▁Fed`, `ĠFed`) stays, where `decode` would drop it."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    texts = iter(tokenizer.convert_ids_to_tokens([i for i in ids.tolist() if i >= 0]))
    return [next(texts) if i >= 0 else None for i in ids.tolist()]


# The step kinds, each run as `(engine, plan, name, step, state, batch_size, found, cost)`, adding the model calls
# it made and the tokens they fed to `cost` (a forward and a generation also take the points their run serves,
# `served`, and what they read, `reading`).
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit}
