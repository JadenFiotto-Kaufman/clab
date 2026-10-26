"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

`device` is one torch device, which nnsight's loader takes as its `device`, or a device map transformers
computes (`auto`, `balanced`, ...), which it takes as its `device_map` and which may spread the model over
several GPUs; the batch's inputs go where the model's first parameter is, and transformers moves them on
from there.

`tp` above 1 shards the model by transformers' tensor parallelism instead (`DistributedConfig(tp_size=tp)`, as
nnsight's `distributed_config`), over `tp` processes that `torchrun --nproc-per-node=<tp>` starts: `clab run DOC
--device cuda -o tp=N`, each rank on its own GPU, or `--device cpu` where none is visible. Every rank runs the whole
plan, which branches on nothing a rank computes, and samples from rank 0's seed; only rank 0 writes the output
directory and prints, and its run record lists every rank's device. A value the ranks hold in pieces is refused by
name (`split`): the attention's interior but its output projection's input (`attention_premix`), the MLP's
activation, and the output of an embedding the plan shards by vocabulary (`token_embeddings` on a tied-embedding
checkpoint); the residual stream, the attention's and the MLP's outputs, the head and the logits are whole on every
rank. Results under tensor parallelism are not bit-equal to unsharded ones, since a sharded matmul sums its parts
in another order: the ranks agree to the bit, and agree with `tp` 1 to that rounding.

The engine holds its model and nothing else: inside the session the engine and its model are the only
client-side objects, and nnsight ships every name a block closes over to wherever the block runs.
"""

import os
import time
import warnings
from typing import Any

import nnsight
import torch
from nnterp import StandardizedTransformer
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from clab import steps
from clab.engines import engine
from clab.engines.standard import COMPUTED, SIZES, build, by_site, envoy, inputs, loading, nnterp_name, stops
from clab.format.address import COMPONENTS
from clab.plan import Model, Plan
from clab.steps import batches
from clab.steps.batches import landed, now


@engine("nnterp")
class NnterpEngine:
    name: str
    remote: bool | str = False      # what the session's `remote=` is: the ndif engine's option

    def __init__(self, model: Model, *, device: str = "cpu", tp: int = 1) -> None:
        """The checkpoint `model` pins, in its dtype (`loading`), placed by `device` (the loader's `device_map`
        when transformers computes it, else its `device`) or sharded over `tp` ranks (`sharded`); the block's
        other keys go to the loader as they are.

        device: one torch device ("cpu", "cuda:0") as the loader's device, or "auto", "balanced", ... as its device_map; under tp, "cuda" or "cpu"
        tp: how many ranks tensor parallelism shards the model over, one process each under `torchrun --nproc-per-node=<tp>`; 1 does not shard it
        """
        loaded, placed = loading(model, "nnterp", device), sharded(tp, device)
        self.spec, self.device, self.options = model, device, {"tp": tp}
        self.model = build(model, "nnterp", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=True, **loaded, **placed, **extra))

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan:
        """The whole plan in one session, opened with `remote` (False here; the ndif engine's option there);
        the results and the run record are the one thing that comes home, into the client's own plan. Under `tp`
        every rank samples from rank 0's seed, so that a sampled generation picks the same tokens on every rank,
        and the run record lists every rank's device."""
        started = steps.begin(self, plan, batch_size)
        ranks = self.options.get("tp", 1)
        if ranks > 1:
            seed = [torch.initial_seed()]
            torch.distributed.broadcast_object_list(seed, src=0)
            torch.manual_seed(seed[0])
        with self.model.session(remote=self.remote):
            steps.run(self, plan, batch_size)
            home = nnsight.save({"results": plan.results, "run": plan.run})
        plan.results, plan.run = home["results"], home["run"]
        assert plan.run is not None
        if ranks > 1:
            every: list[Any] = [None] * ranks
            torch.distributed.all_gather_object(every, plan.run.devices)
            plan.run.devices = list(dict.fromkeys(d for devices in every for d in devices))
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.model.tokenizer)

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """Why nnterp does not serve a component on this checkpoint, from its `support()`, and under `tp` why the
        ranks hold it in pieces (`split`); None where it is served."""
        support, tp = self.model.support(), self.options.get("tp", 1)
        reasons = {c: support.get(nnterp_name(c)) for c in COMPONENTS}
        for c in COMPONENTS if tp > 1 else []:
            layers = range(self.model.num_layers) if COMPONENTS[c]["per_layer"] else [None]
            pieces = {layer: f"under tp {tp} {why}" for layer in layers if (why := split(self.model, c, layer))}
            if pieces and not reasons[c]:
                reasons[c] = pieces[None] if None in pieces else pieces
        return reasons

    def rank(self, component: str, layer: int | None) -> tuple[int, int]:
        """Where the component falls in this checkpoint's forward: nnterp's measured order."""
        return self.model.rank(nnterp_name(component), layer)

    def sizes(self) -> dict[str, int]:
        return {k: getattr(self.model, k) for k in SIZES}

    def forward(self, batch: dict[str, Tensor], records: list[dict]) -> dict[str, Tensor]:
        """One trace over the batch, its one forward call done as `visited` says."""
        sites = by_site(records)
        order = sorted(sites, key=lambda s: self.rank(*s))
        got: dict[str, Tensor] = {}
        with self.model.trace(inputs(batch, ("input_ids", "attention_mask", "position_ids"), batches.home(self))):
            self.visited(order, sites, 0, [], got)
        return got

    def generate(self, batch: dict[str, Tensor], records: list[dict],
                 options: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """`model.generate` over the batch, with `options` as its keyword arguments: the new ids, and each read.

        One loop over the forward calls does each call as `visited` says (from the inputs seen so far: each decode
        call's `input_ids`, read first, where a write has an online scope). It is bounded by the steps the reads ask
        for, or open when a read asks for every step;
        either way it ends where the generation does, which nnsight warns of (that one warning is silenced; a
        read out of forward order shows up instead as a missing step, which `generate.generate` refuses), so the
        ids are read in an invoke of their own, which the loop's end does not cut short."""
        model = self.model
        sites = by_site(records)
        order = sorted(sites, key=lambda s: self.rank(*s))
        asked = [r["calls"] for r in records if "calls" in r]
        loop = None if None in asked else sorted({0, *(s for a in asked for s in a)})
        got: dict[str, Any] = {r["name"]: [] for r in records if "calls" in r and "ops" not in r}
        online = any("online" in r for r in records)
        result: dict[str, Tensor] = {}
        seen: list[Tensor] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".* was never reached: the loop asked for a step the run did "
                                                      r"not make")
            with model.generate(**options) as tracer:
                with tracer.invoke(inputs(batch, ("input_ids", "attention_mask"), batches.home(self))):
                    for step in tracer.all() if loop is None else tracer.iter[loop]:
                        if online and step > 0:
                            seen.append(model.input_ids)
                        self.visited(order, sites, step, seen, got)
                with tracer.invoke():
                    result["ids"] = tracer.result
        return result["ids"][:, batch["input_ids"].shape[1]:], got

    def visited(self, order: list[tuple[str, int | None]], sites: dict[tuple[str, int | None], list[dict]], call: int,
                seen: list[Tensor], got: dict[str, Any]) -> None:
        """Forward call `call` (0 is a trace's one call, or a generation's prefill): each site in forward `order`,
        doing there what `now` says and, at a decode call, `landed`, from the inputs `seen` so far."""
        for component, layer in order:
            here = now(sites[(component, layer)], call)
            here = landed(here, seen, self.tokenizer) if call > 0 else here
            if here:
                host = envoy(self.model, component, layer)
                value, wrote = batches.visit(getattr(host, component), component, here, got)
                if wrote:
                    setattr(host, component, value)

    def readout(self, hidden: Tensor) -> Tensor:
        """nnterp's `project_on_vocab`: the final norm, the head, and whatever the model does after the head."""
        return self.model.project_on_vocab(hidden)


def sharded(tp: int, device: str) -> dict[str, Any]:
    """Where the loader puts the model: at `tp` 1 where `device` says (its `device_map` when transformers computes
    it, else its `device`), and above 1 sharded by transformers' tensor parallelism, `DistributedConfig(tp_size=tp)`
    as nnsight's `distributed_config`, each rank's shard on cuda:<LOCAL_RANK> where a GPU is visible (`device`
    "cuda"), else on the CPU (`device` "cpu"). Refused by name: a `tp` below 1, a `tp` above 1 in a process that is
    not one of `tp` torchrun started (WORLD_SIZE), a WORLD_SIZE other than `tp` (each process would run the whole
    plan), and under `tp` any other `device`, or more ranks on a node than it has GPUs visible."""
    from transformers import DistributedConfig

    world = os.environ.get("WORLD_SIZE")
    if tp < 1:
        raise ValueError(f"tp {tp}: tensor parallelism shards the model over at least one rank")
    if tp > 1 and world is None:
        raise ValueError(f"tp {tp} shards the model over {tp} processes, one per rank, that torchrun starts: "
                         f"`torchrun --nproc-per-node={tp} --no-python clab run <document> -o tp={tp}`; this process "
                         "is not one of them (WORLD_SIZE is unset)")
    if world is not None and int(world) != tp:
        raise ValueError(f"tp {tp}: torchrun started {world} processes (WORLD_SIZE), each of which runs the whole "
                         f"plan, and tp shards the model over {tp}; give both the one number: `torchrun "
                         f"--nproc-per-node={world} --no-python clab run <document> -o tp={world}`")
    if tp == 1:
        return {"device_map": device} if device in COMPUTED else {"device": device}
    kind = "cuda" if torch.cuda.is_available() else "cpu"
    if device != kind:
        raise ValueError(f"device {device!r}: under tp {tp} transformers puts each rank's shard "
                         + ("on its own GPU, cuda:<LOCAL_RANK>, so the device is 'cuda'" if kind == "cuda" else
                            "on the CPU, no GPU being visible, so the device is 'cpu'"))
    here = int(os.environ.get("LOCAL_WORLD_SIZE", tp))
    if kind == "cuda" and here > torch.cuda.device_count():
        raise ValueError(f"tp {tp}: each rank holds its shard on its own GPU, and torchrun started {here} ranks on "
                         f"this node, which shows {torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES)")
    return {"distributed_config": DistributedConfig(tp_size=tp)}


# The tensor-parallel styles that leave a module's output split across ranks past the module, so that the forward
# of the module holding it runs on pieces; every other style hands its output on whole to every rank.
SPLITS = ("colwise", "packed_colwise", "sequence_parallel", "ep_router", "grouped_gemm")


def split(model: StandardizedTransformer, component: str, layer: int | None) -> str | None:
    """Why the ranks hold the component in pieces when the model is sharded by its tensor-parallel plan (a style
    per module, transformers' `tp_plan`), or None where every rank holds it whole: the one place that decides it,
    from the plan and the module nnterp reaches the component at. nnsight makes whole the input and the output of
    a module the plan names (it gathers a shard and reduces a partial), all but a vocab-parallel embedding's
    output, which it cannot reassemble. Any other value is whole unless a forward that runs on pieces computes it:
    that of a module the plan names, of a module holding a child whose style leaves its output split (`SPLITS`:
    the attention holds its `colwise` projections, the MLP its gate and up), or of any module inside one."""
    from transformers.distributed.tensor_parallel import _get_parameter_plan

    hf: Any = model._module
    plan = getattr(hf, "tp_plan", None) or {}

    def style(name: str) -> str | None:
        return _get_parameter_plan(name, plan, is_weight=False) if name else None

    host: Any = envoy(model, component, layer)
    key = getattr(type(host), component).path(host)
    while key.startswith("../"):
        host, key = host.parent, key[3:]
    parts = key.split(".")
    inside = "source" in parts
    children, side = (parts[:parts.index("source")], "inside") if inside else (parts[:-1], parts[-1])
    for child in children:
        host = getattr(host, child)
    name = host.path.partition(".")[2]           # the module as transformers names it, "" for the root
    if not inside and style(name):
        return (None if (style(name), side) != ("embedding_rowwise", "output") else
                f"{name} is a vocab-parallel embedding, whose output nnsight cannot reassemble")
    forward = name if inside else name.rpartition(".")[0] if name else None
    while forward is not None:
        if style(forward):
            return f"it is computed inside {forward}, which is sharded {style(forward)!r}"
        for child, _ in hf.get_submodule(forward).named_children():
            held = f"{forward}.{child}" if forward else child
            if style(held) in SPLITS:
                return (f"it is computed inside {forward or 'the model'}, whose forward runs on each rank's piece "
                        f"({held} is {style(held)!r})")
        forward = forward.rpartition(".")[0] if forward else None
    return None
