"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

`device` is one torch device, which nnsight's loader takes as its `device` (and a quantized model's as its
`device_map` too, where it is quantized), or a device map transformers computes (`auto`, `balanced`, ...), which it
takes as its `device_map` and which may spread the model over several GPUs; the batch's inputs go where the model's
first parameter is, and transformers moves them on from there.

`tp` above 1 shards the model by transformers' tensor parallelism instead (`DistributedConfig(tp_size=tp)`, as
nnsight's `distributed_config`), over the `tp` processes torchrun starts, each rank on its own GPU: `torchrun
--nproc-per-node=N --no-python clab run DOC --device cuda -o tp=N` (`--device cpu` where no GPU is visible). Every
rank runs the whole plan, which branches on nothing a rank computes, and samples from a seed rank 0 draws; only
rank 0 writes the output directory and prints, and its run record lists every rank's device. A value the ranks hold
in pieces is refused by name (`split`): the attention's interior but its output projection's input
(`attention_premix`), the MLP's activation, and the output of an embedding the plan shards by vocabulary
(`token_embeddings`, where the checkpoint's plan says `embedding_rowwise`); the residual stream, the attention's
and the MLP's outputs, the head and the logits are whole on every rank. Results under tensor parallelism are not
bit-equal to unsharded ones, since a sharded matmul sums its parts in another order: the ranks agree to the bit, and
agree with `tp` 1 to that rounding, a fit's trained values further: Adam amplifies the rounding (in bfloat16 it can
flip a greedy generation's near-tie token too).

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
from clab.engines import engine, standard
from clab.engines.standard import SIZES, build, by_site, envoy, inputs, loading, nnterp_name, placing, served, stops
from clab.plan import QUANTIZED, Model, Plan
from clab.steps import batches
from clab.steps.batches import landed, now


@engine("nnterp")
class NnterpEngine:
    name: str
    remote: bool | str = False      # what the session's `remote=` is: the ndif engine's option
    order: dict[tuple[str, int | None], tuple[int, int]] | None = None     # under tp, measured on a meta shell

    def __init__(self, model: Model, *, device: str = "cpu", tp: int = 1) -> None:
        """The checkpoint `model` pins, in its dtype (`loading`), placed by `device` (`placing`) or sharded over
        `tp` ranks (`sharded`); the block's other keys go to the loader as they are.

        device: one torch device ("cpu", "cuda:0") as the loader's device, or "auto", "balanced", ... as its device_map; under tp, "cuda" or "cpu"
        tp: how many ranks tensor parallelism shards the model over, one process each under `torchrun --nproc-per-node=<tp>`; 1 does not shard it
        """
        placed, loaded = sharded(model, tp, device), loading(model, "nnterp", device)
        self.spec, self.device, self.options = model, device, {"tp": tp}
        if tp > 1:          # nnterp's probe of the sharded model would read every value, the embedding's among them
            self.order = measured(build(model, "nnterp", lambda extra: StandardizedTransformer(
                model.key, revision=model.revision, dispatch=False, device_map=None, **extra)))
        self.model = build(model, "nnterp", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=True, **loaded, **placed, **extra))

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan:
        """The whole plan in one session, opened with `remote` (False here; the ndif engine's option there), the
        model `ready` first; the results, the run record and the step records as the run completed them (each
        value's labels, and where each read and write resolved) are what comes home, into the client's own plan.
        Under `tp` every rank samples from a seed rank 0 draws afresh for the run, so that a sampled generation
        picks the same tokens on every rank, and the run record lists every rank's device."""
        started = steps.begin(self, plan, batch_size)
        ranks = self.options.get("tp", 1)
        if ranks > 1:
            seed = [int(torch.randint(2**62, ()))]
            torch.distributed.broadcast_object_list(seed, src=0)
            torch.manual_seed(seed[0])
        with self.model.session(remote=self.remote):
            self.ready()
            steps.run(self, plan, batch_size)
            home = nnsight.save({"results": plan.results, "run": plan.run, "steps": plan.steps})
        plan.results, plan.run, plan.steps = home["results"], home["run"], home["steps"]
        assert plan.run is not None
        if ranks > 1:
            every: list[Any] = [None] * ranks
            torch.distributed.all_gather_object(every, plan.run.devices)
            plan.run.devices = list(dict.fromkeys(d for devices in every for d in devices))
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    def ready(self) -> None:
        """What the model does once, done before the run's first step so that no step's cost holds it: a meta shell
        (ndif under `remote="local"`) loads its weights, which it would at its first trace, and nnterp measures its
        forward order (`rank`), which it would at the first site a step ranks (under `tp` the engine measured it at
        load, on a meta shell); each is a no-op once done."""
        self.model.dispatch()
        self.rank("layer_output", 0)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    @property
    def cuda(self) -> list[str]:
        """The CUDA devices the model is on, whose peak memory a step's cost measures; none under a remote session:
        on an NDIF server the process is the server's, whose own per-request accounting a reset of the peak would
        clobber, and its sandbox's runner would measure its own tensors."""
        return [] if self.remote else steps.gpus(self)

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.model.tokenizer)

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """nnterp's answer (`standard.unserved`), and under `tp` why the ranks hold a value in pieces (`split`)."""
        reasons, tp, n = standard.unserved(self.model), self.options.get("tp", 1), self.model.num_layers
        for c, reason in reasons.items() if tp > 1 else []:
            pieces: dict[Any, str] = {layer: f"under tp {tp} {why}" for layer in served(c, reason, n)
                                      if (why := split(self.model, c, layer))}
            if pieces and not isinstance(reason, str):
                reasons[c] = pieces[None] if None in pieces else {**pieces, **(reason or {})}
        return reasons

    def rank(self, component: str, layer: int | None) -> tuple[int, int]:
        """Where the component falls in this checkpoint's forward: nnterp's measured order (under `tp` measured on
        a meta shell at load, since a probe of the sharded model reads values the ranks cannot reassemble)."""
        if self.order is not None:
            return self.order[component, None if layer is None else range(self.model.num_layers)[layer]]
        return self.model.rank(nnterp_name(component), layer)

    def sizes(self) -> dict[str, int]:
        return {k: getattr(self.model, k) for k in SIZES}

    def forward(self, batch: dict[str, Tensor], records: list[dict]) -> dict[str, Tensor]:
        """One trace over the batch, its one forward call done as `visited` says and stopped right after its last
        site, the deepest (nnsight's `tracer.stop()`), unless the engine is remote (the ndif engine says why).

        nnsight leaves the stop in reference cycles, which the collector can leave for many forwards: its two
        tracebacks (`__traceback__`, and the `__intervention_tb__` nnsight stashes) hold the stopped forward's frames
        and every activation in them, and its `__context__` (the exception nnsight skips this block's body with) holds
        this frame and its callers'. The engine drops all three once the trace is done."""
        sites = by_site(records)
        order = sorted(sites, key=lambda s: self.rank(*s))
        got: dict[str, Tensor] = {}
        stopped: list[BaseException] = []
        with self.model.trace(inputs(batch, ("input_ids", "attention_mask", "position_ids"),
                                     batches.home(self))) as tracer:
            self.visited(order, sites, 0, [], got)
            if not self.remote:
                try:
                    tracer.stop()
                except BaseException as stop:
                    stopped.append(stop)
                    raise
        for stop in stopped:
            stop.__traceback__ = stop.__context__ = stop.__intervention_tb__ = None  # type: ignore[attr-defined]
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
            here = landed(here, seen, self.tokenizer, call) if call > 0 else here
            if here:
                host = envoy(self.model, component, layer)
                value, wrote = batches.visit(getattr(host, component), component, here, got)
                if wrote:
                    setattr(host, component, value)

    def readout(self, hidden: Tensor) -> Tensor:
        """nnterp's `project_on_vocab`: the final norm, the head, and whatever the model does after the head."""
        return self.model.project_on_vocab(hidden)


def measured(shell: StandardizedTransformer) -> dict[tuple[str, int | None], tuple[int, int]]:
    """nnterp's rank of every component `shell` serves, at each layer it serves it, by `(component, layer)`."""
    return {(c, layer): shell.rank(nnterp_name(c), layer) for c, refused in standard.unserved(shell).items()
            for layer in served(c, refused, shell.num_layers)}


def sharded(model: Model, tp: int, device: str) -> dict[str, Any]:
    """Where the loader puts the model: at `tp` 1 where `device` says (`placing`), and above 1 sharded by
    transformers' tensor parallelism, `DistributedConfig(tp_size=tp)` as nnsight's `distributed_config`, each rank's
    shard on cuda:<LOCAL_RANK> where a GPU is visible (`device` "cuda"), else on the CPU (`device` "cpu"). Refused by
    name before anything loads: a `tp` below 1, a `tp` above 1 in a process that is not one of `tp` torchrun started
    (WORLD_SIZE), a WORLD_SIZE other than `tp` (each process would run the whole plan), and under `tp` a bitsandbytes
    `dtype` (its packed weights do not shard), a checkpoint whose plan nnsight cannot split `tp` ways (`max_tp_size`),
    any other `device`, and more ranks on a node than it has GPUs visible."""
    from nnsight.modeling.tp import max_tp_size
    from transformers import AutoConfig, DistributedConfig

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
        return placing(model, device)
    if model.dtype in QUANTIZED and model.dtype != "fp8":
        raise ValueError(f"model.dtype {model.dtype!r}: bitsandbytes' packed weights do not shard across ranks, so it "
                         f"runs at tp 1 alone, not tp {tp}")
    extra = model.model_extra or {}
    most = max_tp_size(AutoConfig.from_pretrained(model.key, revision=model.revision,
                                                  trust_remote_code=extra.get("trust_remote_code", False)))
    if most is None or most % tp:
        raise ValueError(f"tp {tp}: {model.key} cannot be sharded {tp} ways: "
                         + ("its config gives no tensor-parallel plan nnsight can gather, or its sizes divide no "
                            "degree above 1; run it at tp 1, on one GPU or spread over several with --device auto"
                            if most is None else
                            f"its sizes divide only {', '.join(str(d) for d in range(2, most + 1) if most % d == 0)}"))
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
# of the module holding it runs on pieces; every other style a `base_model_tp_plan` uses hands its output on whole.
SPLITS = ("colwise", "packed_colwise")


def split(model: StandardizedTransformer, component: str, layer: int | None) -> str | None:
    """Why the ranks hold the component in pieces when the model is sharded by its tensor-parallel plan (a style
    per module, transformers' `tp_plan`), or None where every rank holds it whole: the one place that decides it,
    from the plan and the module nnterp reaches the component at. nnsight makes whole the input and the output of
    a module the plan names (it gathers a shard and reduces a partial), all but a vocab-parallel embedding's
    output, which it cannot reassemble. Any other value is whole unless a forward that runs on pieces computes it:
    that of a module the plan names, of a module holding a child whose style leaves its output split (`SPLITS`:
    the attention holds its `colwise` projections, the MLP its gate and up), or of any module inside one. A module
    the plan names is taken as whole on both sides, so a `replicated_with_grad_allreduce` one inside a forward that
    runs on pieces (Qwen3's `q_norm` and `k_norm`) would be handed over in pieces; no component sits on such a
    boundary."""
    from transformers.distributed.tensor_parallel import _get_parameter_plan

    hf: Any = model._module
    plan = getattr(hf, "tp_plan", None) or {}

    def style(name: str) -> str | None:
        return _get_parameter_plan(name, plan, is_weight=False) if name else None

    host: Any = envoy(model, component, layer)
    value = getattr(type(host), component)
    key, inside = value.path(host), standard.inside(value, host)
    while key.startswith("../"):
        host, key = host.parent, key[3:]
    parts = key.split(".")
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
