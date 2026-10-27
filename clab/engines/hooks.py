"""The second engine: a plain HuggingFace model driven by forward hooks, in this process.

It executes only `AutoModelForCausalLM` (its forward and its `generate`) with
`register_forward_hook` and `register_forward_pre_hook`. nnterp is used once, at load, as a meta shell
(no weights): it says which native module and boundary each component is
(the module whose output or input nnterp's own key for it names), the sizes,
and what the checkpoint supports. A component nnterp reaches inside a
module's forward (`.source`) is not a module boundary, and is refused by name
before anything runs. `device` is one torch device, which the loaded model moves to, or a device map
transformers computes (`auto`, `balanced`, ...), which `from_pretrained` takes as its `device_map` and which
may spread the model over several GPUs; the batch's inputs go where the model's first parameter is, and
transformers moves them on from there.
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import nnterp  # noqa: F401  (before any transformers.models import, which otherwise segfaults)
import torch
from nnterp import StandardizedTransformer
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from clab import steps
from clab.engines import engine
from clab.engines.standard import COMPUTED, SIZES, build, by_site, envoy, inputs, mrope, nnterp_name, stops, vision
from clab.format.address import COMPONENTS
from clab.plan import Model, Plan
from clab.steps import batches
from clab.steps.batches import landed, now


@engine("hooks")
class HooksEngine:
    name: str

    def __init__(self, model: Model, *, device: str = "cpu") -> None:
        """The checkpoint `model` pins, as `AutoModelForCausalLM` placed by `device` (`from_pretrained`'s
        `device_map` when transformers computes it, else moved there once loaded), with what the meta shell says
        of it: each component's native module and boundary, and the sizes.

        device: one torch device ("cpu", "cuda:0") the model moves to, or "auto", "balanced", ... as its device_map
        """
        shell = build(model, "hooks", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device_map=None, **extra))
        support = shell.support()
        sites = {c: reach(shell, c, support) for c in COMPONENTS}
        self.spec, self.device, self.options = model, device, {}
        self.tokenizer = shell.tokenizer
        self.vision = vision(getattr(shell, "processor", None), shell.config)
        self.config = {"sizes": {k: getattr(shell, k) for k in SIZES}, "sites": sites}
        # The wrapper of a vision-language checkpoint under its task (the shell's), the text model otherwise.
        auto = AutoModelForImageTextToText if shell.task == "image-text-to-text" else AutoModelForCausalLM
        self.model: Any = build(model, "hooks", lambda extra: auto.from_pretrained(
            model.key, revision=model.revision, dtype=getattr(torch, model.dtype),
            device_map=device if device in COMPUTED else None, **{k: v for k, v in extra.items() if k != "task"}).eval())
        if device not in COMPUTED:
            self.model.to(device)
        self.mrope = mrope(self.model)
        self.config["order"] = order(self.model, sites, batches.home(self))

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan:
        """The whole plan in this process."""
        started = steps.begin(self, plan, batch_size)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.tokenizer)

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """Why this engine cannot reach a component: nnterp's own reason, or that nnterp reads it inside a
        module's forward rather than at a module boundary; None where it can."""
        return {c: site["refused"] for c, site in self.config["sites"].items()}

    def rank(self, component: str, layer: int | None) -> int:
        """Where the component falls in this model's forward: when its hook fires, as measured at load."""
        return self.config["order"][component, layer]

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def forward(self, batch: dict[str, Tensor], records: list[dict]) -> dict[str, Tensor]:
        """One call of the model, under the records' hooks (`hooked`)."""
        got: dict[str, Tensor] = {}
        with self.hooked(records, got):
            self.model(**inputs(batch, batches.home(self), self.mrope))
        return got

    def generate(self, batch: dict[str, Tensor], records: list[dict],
                 options: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """`generate` over the batch, with `options` as its keyword arguments, under the records' hooks
        (`hooked`): the new ids, and each read."""
        got: dict[str, Any] = {r["name"]: [] for r in records if "calls" in r and "ops" not in r}
        with self.hooked(records, got):
            ids = self.model.generate(**inputs(batch, batches.home(self), True), **options)
        return ids[:, batch["input_ids"].shape[1]:], got

    @contextmanager
    def hooked(self, records: list[dict], got: dict[str, Any]) -> Iterator[None]:
        """A hook at every site the records name, removed after. A pre-hook on the model keeps each forward call's
        `input_ids` (call 0 is a forward's one call, or a generation's prompt), and each site's hook does what `now`
        (and, at a decode call, `landed`) says there."""
        calls: list[Tensor] = []
        handles = [self.model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs["input_ids"]),
                                                        with_kwargs=True)]
        handles += [hook(self, component, layer, here, got, calls)
                    for (component, layer), here in by_site(records).items()]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def readout(self, hidden: Tensor) -> Tensor:
        """The final norm, then the head: the modules nnterp's names for their outputs reach."""
        norm, head = (self.model.get_submodule(self.config["sites"][c]["paths"][None])
                      for c in ("final_norm_output", "lm_head_output"))
        return head(norm(hidden))


def hook(engine: HooksEngine, component: str, layer: int | None, here: list[dict], got: dict[str, Any],
         calls: list[Tensor]) -> RemovableHandle:
    """A hook at the component's native module that does the site's records `here` in turn, in nnterp's layout:
    those `now` says on the current forward call (`calls` holds each call's `input_ids` so far)."""
    reach = engine.config["sites"][component]
    module = engine.model.get_submodule(reach["paths"][layer])
    shape = engine.config["sizes"]["num_heads"], engine.config["sizes"]["head_dim"]

    def visit(value: Tensor) -> tuple[Tensor, bool]:
        records = now(here, len(calls) - 1)
        records = landed(records, calls[1:], engine.tokenizer) if len(calls) > 1 else records
        # A flat [batch, seq, hidden] module boundary serves a (heads, head_dim) layout here.
        flat = len(COMPONENTS[component]["dims"]) == 4 and value.dim() == 3
        value = value.unflatten(-1, shape) if flat else value
        value, wrote = batches.visit(value, component, records, got)
        return (value.flatten(-2) if flat else value), wrote

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


def reach(shell: StandardizedTransformer, component: str, support: dict[str, Any]) -> dict[str, Any]:
    """Where the component is on the native model: per layer, the module whose output or input nnterp's key names."""
    per_layer = COMPONENTS[component]["per_layer"]
    reason_for = support.get(nnterp_name(component))
    paths, boundary = {}, None
    for layer in (range(shell.num_layers) if per_layer else [None]):
        reason = reason_for.get(layer) if isinstance(reason_for, dict) else reason_for
        if reason:
            return {"paths": {}, "boundary": None, "refused": reason}
        host: Any = envoy(shell, component, layer)
        key = getattr(type(host), component).path(host)
        if "source" in key.split("."):
            return {"paths": {}, "boundary": None,
                    "refused": f"nnterp reads it inside a module's forward ({key}), not at a module boundary"}
        while key.startswith("../"):
            host, key = host.parent, key[3:]
        *children, boundary = key.split(".")
        for child in children:
            host = getattr(host, child)
        paths[layer] = host.path.removeprefix("model").removeprefix(".")
    return {"paths": paths, "boundary": boundary, "refused": None}

