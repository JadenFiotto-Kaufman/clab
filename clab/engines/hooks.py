"""The second engine: a plain HuggingFace model driven by forward hooks, in this process.

It executes only `AutoModelForCausalLM` with `register_forward_hook` and
`register_forward_pre_hook`. nnterp is used once, at load, as a meta shell
(no weights): it says which native module and boundary each component is
(the module whose output or input nnterp's own key for it names), the sizes,
and what the checkpoint supports. A component nnterp reaches inside a
module's forward (`.source`) is not a module boundary, and is refused by name
before anything runs, as is `remote`.
"""

from typing import Any

import nnterp  # noqa: F401  (before any transformers.models import, which otherwise segfaults)
import torch
from nnterp import StandardizedTransformer
from torch import Tensor
from transformers import AutoModelForCausalLM

from .. import ops, steps
from ..address import COMPONENTS, nnterp_name
from ..compile import points
from .nnterp import SIZES, envoy


class HooksEngine:
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def execute(self, plan: dict, *, batch_size: int = 8, remote: bool | str = False) -> dict:
        if remote:
            raise ValueError("the hooks engine runs in this process; remote runs need the nnterp engine")
        refuse(plan, self.config["sites"])
        steps.run(self, plan, batch_size)
        return plan

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]:
        """One call of the model with a hook at every site the records name; the hooks are removed after."""
        sites: dict[tuple, tuple[list, list]] = {}
        for record in writes:
            sites.setdefault((record["component"], record["layer"]), ([], []))[0].append(record)
        for record in reads:
            sites.setdefault((record["component"], record["layer"]), ([], []))[1].append(record)
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
        raise NotImplementedError("M2")


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


def refuse(plan: dict, sites: dict) -> None:
    """Every component the plan names that this engine cannot reach, by name, before anything runs."""
    for name, entry in plan["steps"].items():
        for step in points(entry):
            for record in step.get("writes", []) + step.get("reads", []):
                reason = sites[record["component"]]["refused"]
                if reason:
                    raise ValueError(f"step {name!r}, {record['name']!r}: the hooks engine does not serve "
                                     f"{record['component']}: {reason}")


def reach(shell: StandardizedTransformer, component: str, support: dict) -> dict:
    """Where the component is on the native model: per layer, the module whose output or input nnterp's key names."""
    per_layer = COMPONENTS[component]["host"] != "model"
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


def load(model: dict) -> HooksEngine:
    """The checkpoint the plan pins, as `AutoModelForCausalLM` on the CPU, with what the meta shell says of it."""
    extra = {"attn_implementation": model["attn_implementation"]} if model["attn_implementation"] else {}
    shell = StandardizedTransformer(model["key"], revision=model["revision"], dispatch=False, device_map=None, **extra)
    support = shell.support()
    config = {"model": model, "sizes": {k: getattr(shell, k) for k in SIZES},
              "sites": {c: reach(shell, c, support) for c in COMPONENTS}}
    loaded = AutoModelForCausalLM.from_pretrained(model["key"], revision=model["revision"],
                                                  dtype=getattr(torch, model["dtype"]), **extra)
    return HooksEngine(loaded.eval(), shell.tokenizer, config)
