"""The second engine: a plain HuggingFace model driven by forward hooks, in this process.

It executes only `AutoModelForCausalLM` (its forward and its `generate`) with `register_forward_hook` and
`register_forward_pre_hook`. nnterp is used once, at load, as a meta shell: it says which native module and boundary
each component is (the module whose output or input nnterp's key for it names), the sizes, the final norm and head
its `readout` runs in the standard form (not the model's own, as nnterp's is), and what the checkpoint supports,
layer by layer, which this engine refuses by name before anything runs, as well as a value nnterp reads inside a
module's forward (`.source`) or among a module's call arguments (`.inputs`) wherever the module is. A value nnterp
holds flat over tokens at its module (a mixture of experts' routing and routed sum, `[batch * seq, ...]`) is served
as nnterp serves it, `[batch, seq, ...]`. `device` is one torch device, which the loaded model moves to
(a quantized one, which cannot move, loads there), or a device map transformers computes (`auto`, `balanced`, ...),
which `from_pretrained` takes as its `device_map` and may spread over several GPUs; the batch's inputs go where the
model's first parameter is, and transformers moves them on from there.
"""

import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator

import nnterp  # noqa: F401  (before any transformers.models import, which otherwise segfaults)
import torch
from nnterp import StandardizedTransformer
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM

from clab import steps
from clab.engines import engine, standard
from clab.engines.standard import COMPUTED, build, envoy, inputs, loading, stops
from clab.format.address import COMPONENTS
from clab.plan import Model, Plan
from clab.steps import batches
from clab.steps.batches import acting_at_call, stamped_at_call


@engine("hooks")
class HooksEngine:
    name: str
    remote: bool | str = False          # it runs in this process
    stops: bool = True                  # a forward ends right after the schedule's stop site (`_Stopped`)
    unwritable: dict[str, str] = {}     # every component it serves, it serves whole (`Engine`)
    last_only: dict[str, str] = {}

    def __init__(self, model: Model, *, device: str = "cpu") -> None:
        """The checkpoint `model` pins, as `AutoModelForCausalLM` in its dtype (`loading`) placed by `device`
        (`from_pretrained`'s `device_map` when transformers computes it or the model is quantized, else moved there
        once loaded), with what the meta shell says of it: each component's native module and boundary, and the
        sizes.

        device: one torch device ("cpu", "cuda:0") the model moves to, or "auto", "balanced", ... as its device_map
        """
        loaded = loading(model, "hooks", device)
        # A model is quantized where `device_map` says (the first GPU when it says nothing) and cannot move after.
        mapped = device in COMPUTED or "quantization_config" in loaded
        shell = build(model, "hooks", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device_map=None, **extra))
        sites = {c: reach(shell, c, refused) for c, refused in standard.unserved(shell).items()}
        self.spec, self.device, self.options = model, device, {}
        self.tokenizer = shell.tokenizer
        self.config = {"sizes": standard.sizes(shell), "sites": sites}
        self.model: Any = build(model, "hooks", lambda extra: AutoModelForCausalLM.from_pretrained(
            model.key, revision=model.revision, **loaded, device_map=device if mapped else None, **extra).eval())
        self.model.requires_grad_(False)        # no gradient reaches a weight, so no graph is kept for one
        if not mapped:
            self.model.to(device)
        self.config["order"] = order(self.model, sites, batches.model_device(self))

    def execute(self, plan: Plan, *, batch_size: int = 8, padding: str = "dataset",
                readout: str = "full") -> Plan:
        """The whole plan in this process."""
        started = steps.start_run(self, plan, batch_size, padding, readout)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.tokenizer)

    @property
    def cuda(self) -> list[str]:
        """The CUDA devices the model is on, whose peak memory a step's cost measures."""
        return steps.cuda_devices(self)

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """nnterp's answer, layer by layer, but at every layer with the module a value nnterp reads inside a forward
        or among a call's arguments, said as clab says it (`standard.own`)."""
        return standard.own({c: site["refused"] for c, site in self.config["sites"].items()})

    def rank(self, component: str, layer: int | None) -> int:
        """Where the component falls in this model's forward: when its hook fires, as measured at load."""
        return self.config["order"][component, layer]

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def forward(self, batch: dict[str, Tensor], bound: dict[str, Any]) -> dict[str, Tensor]:
        """One call of the model, under the bound schedule's hooks (`hooked`), ended right after its stop site,
        whose hook raises `_Stopped`, then its gradients taken from the values its hooks captured, which keep their
        graph (`batches.differentiate`, under `torch.enable_grad()` with the call where the bound schedule has
        any)."""
        got: dict[str, Tensor] = {}
        with torch.enable_grad() if bound["grad"] else nullcontext():
            with self.hooked(bound["sites"], got, bound["stop"] if self.stops else None):
                try:
                    self.model(**inputs(batch, ("input_ids", "attention_mask", "position_ids"),
                                        batches.model_device(self)))
                except _Stopped:
                    pass
            batches.differentiate(got, bound["grad"])
        return got

    def generate(self, batch: dict[str, Tensor], bound: dict[str, Any],
                 options: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """`generate` over the batch, with `options` as its keyword arguments, under the bound schedule's hooks
        (`hooked`): the new ids, and each read."""
        got: dict[str, Any] = {}
        with self.hooked(bound["sites"], got, None):
            ids = self.model.generate(**inputs(batch, ("input_ids", "attention_mask"), batches.model_device(self)),
                                      **options)
        return ids[:, batch["input_ids"].shape[1]:], got

    @contextmanager
    def hooked(self, sites: list[dict[str, Any]], got: dict[str, Any], stop: list | None) -> Iterator[None]:
        """A hook at every site of the bound schedule, in its order, removed after. A pre-hook on the model keeps
        each forward call's `input_ids` (call 0 is a forward's one call, or a generation's prompt), and each site's
        hook does what `acting_at_call` (and, at a decode call, `stamped_at_call`) says there; the hook of the `stop`
        site (the last, hooked last so that the sites sharing its module boundary are done first) then raises
        `_Stopped`."""
        calls: list[Tensor] = []
        handles = [self.model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs["input_ids"]),
                                                        with_kwargs=True)]
        handles += [hook(self, s["component"], s["layer"], s["records"], got, calls,
                         [s["component"], s["layer"]] == stop) for s in sites]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def readout(self, hidden: Tensor) -> Tensor:
        """The standard form, not the model's own: the final norm and head nnterp's names reach, then `cap *
        tanh(logits / cap)` where the text config sets `final_logit_softcapping`. It agrees with nnterp's on the
        corpus and may differ from, or fail on, a family whose model does more after the head (Cohere's `logit_scale`,
        Granite's `logits_scaling`, Falcon-H1's multiplier): what nnterp adds."""
        norm, head = (self.model.get_submodule(self.config["sites"][c]["paths"][None])
                      for c in ("final_norm_output", "lm_head_output"))
        cap, logits = getattr(self.model.config.get_text_config(), "final_logit_softcapping", None), head(norm(hidden))
        return cap * torch.tanh(logits / cap) if cap else logits


class _Stopped(Exception):
    """What a forward's deepest hook raises once its site's records are done, ending the forward there."""


def hook(engine: HooksEngine, component: str, layer: int | None, here: list[dict], got: dict[str, Any],
         calls: list[Tensor], ends: bool) -> RemovableHandle:
    """A hook at the component's native module that does the site's records `here` in turn, in nnterp's layout:
    those `acting_at_call` says on the current forward call (`calls` holds each call's `input_ids` so far); then, where
    it `ends` the forward, raises `_Stopped`."""
    reach = engine.config["sites"][component]
    module = engine.model.get_submodule(reach["paths"][layer])
    shape = engine.config["sizes"]["num_heads"], engine.config["sizes"]["head_dim"]

    def visit(value: Tensor) -> tuple[Tensor, bool]:
        records = acting_at_call(here, len(calls) - 1)
        records = stamped_at_call(records, calls[1:], engine.tokenizer, len(calls) - 1) if len(calls) > 1 else records
        # A value nnterp holds flat over tokens, [batch * seq, ...], is this call's [batch, seq, ...] rows; else a flat
        # [batch, seq, hidden] module boundary serves a (heads, head_dim) layout here.
        tokens = reach["tokens"] and value.dim() == len(COMPONENTS[component]["dims"]) - 1
        flat = not tokens and len(COMPONENTS[component]["dims"]) == 4 and value.dim() == 3
        value = (value.unflatten(0, (calls[-1].shape[0], -1)) if tokens else
                 value.unflatten(-1, shape) if flat else value)
        value, wrote = batches.visit_site(value, component, records, got)
        if ends:
            raise _Stopped
        return (value.flatten(0, 1) if tokens else value.flatten(-2) if flat else value), wrote

    if reach["boundary"] == "input":
        def pre(module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            value, wrote = visit(args[0])
            return (value, *args[1:]) if wrote else None
        return module.register_forward_pre_hook(pre)

    def post(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        if component == "logits":
            value, wrote = visit(output.logits)
            if wrote:
                output.logits = value
            return output
        first = output[0] if isinstance(output, tuple) else output
        value, wrote = visit(first)
        if not wrote:
            return None
        return (value, *output[1:]) if isinstance(output, tuple) else value
    return module.register_forward_hook(post)


def order(model: torch.nn.Module, sites: dict[str, dict[str, Any]], device: str) -> dict[tuple[str, int | None], int]:
    """When each served component's hook fires in a forward, by `(component, layer)`, measured on a two-token
    input on `device`, where the model takes its input: components at one module boundary fire together and
    share a rank."""
    fired: list[tuple[str, str]] = []

    def noted(key: tuple[str, str]) -> Callable[..., None]:
        def note(*_: object) -> None:
            if key not in fired:
                fired.append(key)
        return note
    handles = []
    for path, boundary in {(p, s["boundary"]) for s in sites.values() for p in s["paths"].values()}:
        module = model.get_submodule(path)
        handles.append(module.register_forward_pre_hook(noted((path, boundary))) if boundary == "input" else
                       module.register_forward_hook(noted((path, boundary))))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2]], device=device))
    finally:
        for handle in handles:
            handle.remove()
    return {(c, layer): fired.index((path, s["boundary"])) for c, s in sites.items()
            for layer, path in s["paths"].items()}


def reach(shell: StandardizedTransformer, component: str, refused: str | dict[Any, str] | None) -> dict[str, Any]:
    """Each layer's module nnterp's key reads, or why not: the value is inside a forward there, or among a module's
    call arguments, or nnterp's reason; and whether nnterp holds it flat over tokens there (`tokens`, a
    `TokenEProperty`)."""
    from nnterp.components.tokens import TokenEProperty

    per_layer, paths, reasons, boundary = COMPONENTS[component]["per_layer"], {}, {}, None
    tokens = False
    for layer in range(shell.num_layers) if per_layer else [None]:
        why = refused.get(layer) if isinstance(refused, dict) else refused
        try:
            host: Any = envoy(shell, component, layer)
        except AttributeError:          # no such module at this layer: nnterp says why
            if not why:
                raise
            reasons[layer] = why
            continue
        value = getattr(type(host), component, None)
        if value is None:               # no such value on this layer's module (a dense MLP's): nnterp says why
            assert why, f"nnterp serves {component} at layer {layer} on a module that has no such value"
            reasons[layer] = why
            continue
        key, inside = value.path(host), standard.inside(value, host)
        argument, tokens = key.rpartition(".")[2] == "inputs", isinstance(value, TokenEProperty)
        if inside or argument or why:
            reasons[layer] = (f"nnterp reads it inside a module's forward ({key}), not at a module boundary" if inside
                              else f"nnterp reads it among a module's call arguments ({key}), not at a module boundary"
                              if argument else why)
            continue
        host, boundary = standard.reached(host, key)
        paths[layer] = host.path.removeprefix("model").removeprefix(".")
    return {"paths": paths, "boundary": boundary, "tokens": tokens,
            "refused": (reasons or None) if per_layer else reasons.get(None)}
