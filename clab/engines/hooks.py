"""The second engine: a plain HuggingFace model driven by forward hooks, in this process.

It executes only `AutoModelForCausalLM` (its forward and its `generate`) with
`register_forward_hook` and `register_forward_pre_hook`. nnterp is used once, at load, as a meta shell
(no weights): it says which native module and boundary each component is
(the module whose output or input nnterp's own key for it names), the sizes,
and what the checkpoint supports. A component nnterp reaches inside a
module's forward (`.source`) is not a module boundary, and is refused by name
before anything runs, as is `remote`.
"""

import time
from typing import Any

import nnterp  # noqa: F401  (before any transformers.models import, which otherwise segfaults)
import torch
from nnterp import StandardizedTransformer
from torch import Tensor
from transformers import AutoModelForCausalLM

from ..address import COMPONENTS
from ..plan import Model, Plan
from ..runtime import ops, steps
from . import engine
from .standard import SIZES, by_site, envoy, keep, landed, load, nnterp_name, now, stops


@engine("hooks")
class HooksEngine:
    name: str

    def __init__(self, model: Model):
        """The checkpoint `model` pins, as `AutoModelForCausalLM` on the CPU, with what the meta shell says of it:
        each component's native module and boundary, the sizes, and the modules the readout runs."""
        shell = load(model, "hooks", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device_map=None, **extra))
        support = shell.support()
        sites = {c: reach(shell, c, support) for c in COMPONENTS}
        self.spec = model
        self.tokenizer = shell.tokenizer
        self.config = {"sizes": {k: getattr(shell, k) for k in SIZES}, "sites": sites,
                       "readout": [sites[c]["paths"]["None"] for c in ("final_norm_output", "lm_head_output")]}
        self.model: Any = load(model, "hooks", lambda extra: AutoModelForCausalLM.from_pretrained(
            model.key, revision=model.revision, dtype=getattr(torch, model.dtype), **extra).eval())
        self.config["order"] = order(self.model, sites)

    def execute(self, plan: Plan, *, batch_size: int = 8, remote: bool | str = False) -> Plan:
        if remote:
            raise ValueError("the hooks engine runs in this process; remote runs need the nnterp engine")
        started = steps.begin(self, plan, batch_size, remote)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.tokenizer)

    def unserved(self) -> dict:
        """Why this engine cannot reach a component: nnterp's own reason, or that nnterp reads it inside a
        module's forward rather than at a module boundary; None where it can."""
        return {c: site["refused"] for c, site in self.config["sites"].items()}

    def rank(self, component: str, layer: int | None) -> int:
        """Where the component falls in this model's forward: when its hook fires, as measured at load."""
        return self.config["order"][f"{component}@{layer}"]

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]:
        """One call of the model with a hook at every site the records name; the hooks are removed after."""
        sites = by_site(writes, reads)
        got: dict[str, Tensor] = {}
        handles = [hook(self.model, self.config, component, layer, here, there, got, None)
                   for (component, layer), (here, there) in sites.items()]
        try:
            self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"])
        finally:
            for handle in handles:
                handle.remove()
        return got

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Any]]:
        """`generate` over the batch, with `options` as its keyword arguments: the new ids, and each read. A
        pre-hook on the model counts its forward calls (0 is the prompt) and keeps each decode call's
        `input_ids`, and each site's hook does what `now` (and, for an online scope, `landed`) says there."""
        sites = by_site(writes, reads)
        got: dict[str, Any] = {r["name"]: [] for r in reads if "calls" in r}
        call: dict = {"n": -1, "seen": [], "tokenizer": self.tokenizer}

        def count(module, args, kwargs):
            call["n"] += 1
            if call["n"] > 0:
                call["seen"].append(kwargs["input_ids"])
        handles = [self.model.register_forward_pre_hook(count, with_kwargs=True)]
        handles += [hook(self.model, self.config, component, layer, here, there, got, call)
                    for (component, layer), (here, there) in sites.items()]
        try:
            ids = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **options)
        finally:
            for handle in handles:
                handle.remove()
        return ids[:, batch["input_ids"].shape[1]:], got

    def readout(self, hidden: Tensor) -> Tensor:
        """The final norm, then the head: the modules nnterp's names for their outputs reach."""
        norm, head = (self.model.get_submodule(path) for path in self.config["readout"])
        return head(norm(hidden))


def hook(model, config: dict, component: str, layer: int | None, here: list, there: list, got: dict,
         call: dict | None):
    """A hook at the component's native module that applies `here` and gathers `there`, in nnterp's layout;
    under a generation (`call` counts its forward calls), what `now` says on the current call."""
    reach = config["sites"][component]
    module = model.get_submodule(reach["paths"][str(layer)])
    shape = config["sizes"]["num_heads"], config["sizes"]["head_dim"]

    def visit(value: Tensor) -> tuple[Tensor, bool]:
        writes, reads = (here, there) if call is None else now(here, there, call["n"])
        if call is not None and call["n"] > 0:
            writes = landed(writes, call["seen"], call["tokenizer"])
        # A flat [batch, seq, hidden] module boundary serves a (heads, head_dim) layout here.
        flat = len(COMPONENTS[component]["dims"]) == 4 and value.dim() == 3
        value = value.unflatten(-1, shape) if flat else value
        if writes:
            value = ops.write(value, component, writes, got)
        for record in reads:
            keep(got, record, ops.select(value, component, record))
        return (value.flatten(-2) if flat else value), bool(writes)

    if reach["boundary"] == "input":
        def pre(module, args):
            value, wrote = visit(args[0])
            return (value, *args[1:]) if wrote else None
        return module.register_forward_pre_hook(pre)

    def post(module, args, output):
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


def order(model, sites: dict) -> dict[str, int]:
    """When each served component's hook fires in a forward (`<component>@<layer>`), measured on a two-token
    input: components at one module boundary fire together and share a rank."""
    fired: list = []

    def noted(key: tuple):
        def note(*_):
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
            model(input_ids=torch.tensor([[1, 2]]))
    finally:
        for handle in handles:
            handle.remove()
    return {f"{c}@{layer}": fired.index((path, s["boundary"])) for c, s in sites.items()
            for layer, path in s["paths"].items()}


def reach(shell: StandardizedTransformer, component: str, support: dict) -> dict:
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
        paths[str(layer)] = host.path.removeprefix("model").removeprefix(".")
    return {"paths": paths, "boundary": boundary, "refused": None}

