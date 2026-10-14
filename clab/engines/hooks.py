"""The second engine: a plain HuggingFace model driven by forward hooks, in this process.

It executes only `AutoModelForCausalLM` with `register_forward_hook` and
`register_forward_pre_hook`. nnterp is used once, at load, as a meta shell
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

from .. import ops, steps
from ..address import COMPONENTS
from ..plan import Model, Plan
from . import DTYPES, engine
from .standard import SIZES, by_site, envoy, nnterp_name


@engine("hooks")
class HooksEngine:
    name: str

    def __init__(self, model: Model):
        """The checkpoint `model` pins, as `AutoModelForCausalLM` on the CPU, with what the meta shell says of it:
        each component's native module and boundary, the sizes, and the modules the readout runs."""
        extra: dict[str, Any] = {"attn_implementation": model.attn_implementation} if model.attn_implementation else {}
        shell = StandardizedTransformer(model.key, revision=model.revision, dispatch=False, device_map=None, **extra)
        support = shell.support()
        sites = {c: reach(shell, c, support) for c in COMPONENTS}
        self.spec = model
        self.tokenizer = shell.tokenizer
        self.config = {"sizes": {k: getattr(shell, k) for k in SIZES}, "sites": sites,
                       "readout": [sites[c]["paths"]["None"] for c in ("final_norm_output", "lm_head_output")]}
        self.model = AutoModelForCausalLM.from_pretrained(model.key, revision=model.revision,
                                                          dtype=getattr(torch, DTYPES[model.dtype]), **extra).eval()

    def execute(self, plan: Plan, *, batch_size: int = 8, remote: bool | str = False) -> Plan:
        if remote:
            raise ValueError("the hooks engine runs in this process; remote runs need the nnterp engine")
        started = steps.begin(self, plan, batch_size, remote)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    def unserved(self) -> dict:
        """Why this engine cannot reach a component: nnterp's own reason, or that nnterp reads it inside a
        module's forward rather than at a module boundary; None where it can."""
        return {c: site["refused"] for c, site in self.config["sites"].items()}

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]:
        """One call of the model with a hook at every site the records name; the hooks are removed after."""
        sites = by_site(writes, reads)
        got: dict[str, Tensor] = {}
        handles = [hook(self.model, self.config, component, layer, here, there, got)
                   for (component, layer), (here, there) in sites.items()]
        try:
            self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"])
        finally:
            for handle in handles:
                handle.remove()
        return got

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("M3")

    def readout(self, hidden: Tensor) -> Tensor:
        """The final norm, then the head: the modules nnterp's names for their outputs reach."""
        norm, head = (self.model.get_submodule(path) for path in self.config["readout"])
        with torch.no_grad():
            return head(norm(hidden))


def hook(model, config: dict, component: str, layer: int | None, here: list, there: list, got: dict):
    """A hook at the component's native module that applies `here` and gathers `there`, in nnterp's layout."""
    reach = config["sites"][component]
    module = model.get_submodule(reach["paths"][str(layer)])
    shape = config["sizes"]["num_heads"], config["sizes"]["head_dim"]

    def visit(value: Tensor) -> Tensor:
        # A flat [batch, seq, hidden] module boundary serves a (heads, head_dim) layout here.
        flat = len(COMPONENTS[component]["dims"]) == 4 and value.dim() == 3
        value = value.unflatten(-1, shape) if flat else value
        if here:
            value = ops.write(value, component, here)
        for record in there:
            got[record["name"]] = ops.select(value, component, record)
        return value.flatten(-2) if flat else value

    if reach["boundary"] == "input":
        def pre(module, args):
            value = visit(args[0])
            return (value, *args[1:]) if here else None
        return module.register_forward_pre_hook(pre)

    def post(module, args, output):
        if component == "logits":
            value = visit(output.logits)
            if here:
                output.logits = value
            return output
        first = output[0] if isinstance(output, tuple) else output
        value = visit(first)
        if not here:
            return None
        return (value, *output[1:]) if isinstance(output, tuple) else value
    return module.register_forward_hook(post)


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

