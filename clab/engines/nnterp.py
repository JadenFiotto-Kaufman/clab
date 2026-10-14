"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

The engine holds its model and nothing else: inside the session the engine and its model are the only
client-side objects, and nnsight ships every name a block closes over to wherever the block runs.
"""

import time

import nnsight
import torch
from nnterp import StandardizedTransformer
from torch import Tensor

from .. import ops, steps
from ..address import COMPONENTS
from ..plan import Model, Plan
from . import engine
from .standard import SIZES, by_site, envoy, loader, nnterp_name


@engine("nnterp")
class NnterpEngine:
    name: str

    def __init__(self, model: Model):
        """The checkpoint `model` pins, on the CPU; the block's other keys go to the loader as they are."""
        extra = loader(model, "nnterp", ("device", "dispatch"))
        self.spec = model
        self.model = StandardizedTransformer(model.key, revision=model.revision, device="cpu", dispatch=True,
                                             dtype=getattr(torch, model.dtype), **extra)

    def execute(self, plan: Plan, *, batch_size: int = 8, remote: bool | str = False) -> Plan:
        """The whole plan in one session; the filled plan is the one thing that comes home."""
        started = steps.begin(self, plan, batch_size, remote)
        with self.model.session(remote=remote):
            steps.run(self, plan, batch_size)
            nnsight.save(plan)
        assert plan.run is not None
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def unserved(self) -> dict:
        """Why nnterp does not serve a component on this checkpoint, from its `support()`; None where it does."""
        support = self.model.support()
        return {c: support.get(nnterp_name(c)) for c in COMPONENTS}

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
        """nnterp's `project_on_vocab`: the final norm, the head, and whatever the model does after the head."""
        return self.model.project_on_vocab(hidden)
