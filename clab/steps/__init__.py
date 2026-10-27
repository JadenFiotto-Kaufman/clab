"""Walk a compiled plan inside the run: its hub datasets' rows loaded, the preflight pass, every step and point in
order, then the saved values into `plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp and ndif engines it runs inside one session (on the server,
under ndif); under the hooks engine, in plain Python. The cross-step state is a `ChainMap` (a fit's body runs in
a child of it, carrying the minibatch's row ids: `batches.ROWS`) that never leaves the run: each value under its
name, by the coordinates of the point it is at.
Each step kind returns the model calls it made and the tokens they fed (`batches.fed`), which `run` records
with the step's time and memory (`measured`).
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import ChainMap, Counter
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from clab import ops
from clab.compile import materialized
from clab.compile.preflight import preflight
from clab.compile.state import Found
from clab.compile.uses import instance
from clab.format.sweeps import by_coordinate, coordinate, labelled, lookup, put
from clab.plan import Forward, OpStep, Plan, Run, arguments, entries, points, record_of
from clab.steps import fit, generate
from clab.steps.batches import bind, bound, chunked, fed, home, joined, layers, over, read, read_name, stacked, value_at

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine, Parameterized


def begin(engine: Engine, plan: Plan, batch_size: int) -> float:
    """What every engine's `execute` checks first (a compiled plan, for this engine's checkpoint, at a batch size of
    one row or more), and the run record it starts: the engine, the batch size, the device and the options it
    loaded with; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    batched(batch_size)
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    plan.run = Run(engine=engine.name, batch_size=batch_size, device=engine.device, options=engine.options,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def batched(batch_size: int) -> int:
    """`batch_size` as every run takes it: one row or more per batch, refused by name below that (`clab.run` asks
    before the model loads, `begin` for an engine's `execute` called on its own)."""
    if batch_size < 1:
        raise ValueError(f"batch_size {batch_size}: a batch holds one row or more (--batch-size N, N at least 1)")
    return batch_size


def run(engine: Engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, over the plan `held`, without gradients except where a fit trains,
    each one's cost `measured` into `plan.run.costs`; then (`ended`) each saved value into `plan.results` (a
    trained featurizer as its bundle's tensors), and what served the run: the model's sizes, and the dtypes and
    devices of the model's parameters (`placed`). Where each read and write resolved, and each featurizer's width,
    is on its record, which preflight and the steps complete in place. A step that runs out of CUDA memory stops
    the run: the refusal (`exhausted`) names the step, the batch size and every value the run holds, is recorded
    as `plan.run.stopped`, and the steps that finished are recorded and saved as if the run had ended there."""
    plan = held(plan)
    assert plan.run is not None
    plan.run.versions, plan.run.sizes = versions(), engine.sizes()
    found = preflight(engine, plan)
    state = made(plan, home(engine))
    costs: dict = {}
    with torch.no_grad():
        for name, entry in plan.steps.items():
            for step in points(entry):
                try:
                    put(costs, name, step.at, measured(engine, lambda: KINDS[step.kind](
                        engine, plan, name, step, state, batch_size, found)))
                except torch.OutOfMemoryError as e:
                    plan.run.stopped = exhausted(plan, name, step.at, batch_size, state, e)
                    ended(engine, plan, state, costs, found)
                    raise ValueError(plan.run.stopped) from e
    ended(engine, plan, state, costs, found)


def ended(engine: Engine, plan: Plan, state: ChainMap, costs: dict, found: Found) -> None:
    """What `run` records once its steps have run, or stopped: each step's cost, each saved value of a step whose
    every point finished (a trained featurizer as its bundle's tensors), how each input tokenized, and the dtypes
    and devices of the model's parameters."""
    assert plan.run is not None
    plan.run.costs = {name: by_coordinate(plan, by_point) for name, by_point in costs.items()}
    done = {name for name, entry in plan.steps.items() if len(costs.get(name, {})) == len(points(entry))}
    for name in (n for n in plan.saves if n.partition(".")[0] in done):
        entry = plan.steps[name.partition(".")[0]]
        plan.results[name] = ({coordinate(plan, p.at): finished(saved(plan, state, name, p.at)) for p in entry}
                              if isinstance(entry, list) else finished(saved(plan, state, name, {})))
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


def measured(engine: Engine, call: Callable[[], Counter]) -> dict[str, Any]:
    """What a step cost: `call`, which runs it, timed (`seconds`), with the model's CUDA devices (`gpus`)
    synchronized at both ends; the peak memory allocated on each of them while it ran, weights included, in bytes
    by device (`memory`, None where the model is not on CUDA); and the model calls it made and the tokens they
    fed, which `call` returns (`forwards`, `tokens`). The engine says which devices it measures (`engine.cuda`): a
    remote engine's run makes no CUDA call and its `memory` is None."""
    devices = engine.cuda
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


def forward(engine: Engine, plan: Plan, name: str, step: Forward, state: ChainMap, batch_size: int,
            found: Found) -> Counter:
    """The step's input in batches of dataset row ids (`chunked`), each through `engine.forward`; each gathered
    entry joined along rows. Returns the model calls it made and the tokens they fed (`fed`).

    An entry whose `layers` are a slice or picks is one write (when it writes) and one read per layer, the ops
    after its last edit run on each layer's read, stacked layer-first; one whose rows hold different numbers of
    positions is as wide as the widest, NaN past a row's own."""
    parts, count = [], Counter()
    gathered = {k: r for k, r in entries(step).items() if r.gathered}
    for ids, batch, pads in chunked(engine, step, state, found, batch_size):
        sites, records = bind(engine, plan, name, step, state, found, ids, pads)
        got = engine.forward(batch, records)
        count += fed(batch, pads)
        parts.append({key: stacked(record, [read(engine, plan, record, got[read_name(key, layer)], sites[key], ids,
                                                 state, step.at, found, layer)
                                            for layer in layers(engine, record)])
                      for key, record in gathered.items()})
    joined(state, name, step, parts)
    return count


def op(engine: Engine, plan: Plan, name: str, step: OpStep, state: ChainMap, batch_size: int, found: Found) -> Counter:
    """The op over the value `of` names at the rows the run is over (`over`, else every row of its dataset),
    given each one's dataset row, at the layers `of`'s record says it is at (its `layers` axis's labels, or the one
    layer it was read at), to which an operand's `layers` axis is aligned (`bound`); a `classes` axis's vocabulary
    is the probe its coordinates are in. `top_k`'s value over a vocabulary's tokens also holds them, decoded. It
    calls no model."""
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
                     eos=generate.stops(engine, producer) if producer.kind == "generate" else engine.eos, rows=rows)
    if step.op == "top_k" and arguments(step).get("over", ops.transforms.RANKED) == ops.transforms.RANKED:
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)
    return Counter()


def decoded(tokenizer: PreTrainedTokenizerBase, ids: Tensor) -> list:
    """Each token id as its token string, nested as the ids are; None for a missing one (-1). The tokenizer's
    `convert_ids_to_tokens` is the one spelling of a token wherever a run shows one (`pos_token`, `run.tokens`,
    `detokenize` with `tokens`), so a word-start marker (`▁Fed`, `ĠFed`) stays, where `decode` would drop it."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    texts = iter(tokenizer.convert_ids_to_tokens([i for i in ids.tolist() if i >= 0]))
    return [next(texts) if i >= 0 else None for i in ids.tolist()]


# The step kinds, each run as `(engine, plan, name, step, state, batch_size, found)` and returning the model calls
# it made and the tokens they fed (`batches.fed`).
KINDS = {"forward": forward, "generate": generate.generate, "op": op, "fit": fit.fit}
