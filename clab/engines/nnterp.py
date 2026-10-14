"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

The engine holds its model and nothing else: inside the session the engine
and its model are the only client-side objects, and nnsight ships every name a
block closes over to wherever the block runs.
"""

import nnsight
import torch
from nnterp import StandardizedTransformer
from torch import Tensor

from .. import ops, steps
from .standard import SIZES, by_site, envoy, nnterp_name


class NnterpEngine:
    def __init__(self, model: StandardizedTransformer):
        self.model = model

    def execute(self, plan: dict, *, batch_size: int = 8, remote: bool | str = False) -> dict:
        """The whole document in one session; the filled plan is the one thing that comes home."""
        with self.model.session(remote=remote):
            steps.run(self, plan, batch_size)
            nnsight.save(plan)
        return plan

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def sizes(self) -> dict[str, int]:
        return {k: getattr(self.model, k) for k in SIZES}

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]:
        """One trace over the batch, visiting each site in forward order: writes first, then the reads there."""
        model = self.model
        sites = by_site(writes, reads)
        order = sorted(sites, key=lambda s: model.rank(nnterp_name(s[0]), s[1]))
        got: dict[str, Tensor] = {}
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids")}
        with model.trace(inputs):
            for component, layer in order:
                here, there = sites[(component, layer)]
                host = envoy(model, component, layer)
                value = getattr(host, component)
                if here:
                    value = ops.write(value, component, here)
                    setattr(host, component, value)
                for record in there:
                    got[record["name"]] = ops.select(value, component, record)
        return got

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("M3")

    def readout(self, hidden: Tensor) -> Tensor:
        raise NotImplementedError("M2")


def load(model: dict) -> NnterpEngine:
    """The checkpoint the plan pins, on the CPU."""
    extra = {"attn_implementation": model["attn_implementation"]} if model["attn_implementation"] else {}
    loaded = StandardizedTransformer(model["key"], revision=model["revision"], dtype=getattr(torch, model["dtype"]),
                                     device="cpu", dispatch=True, **extra)
    return NnterpEngine(loaded)
