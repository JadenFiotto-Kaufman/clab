"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

The engine holds its model and nothing else: inside the session the engine and its model are the only
client-side objects, and nnsight ships every name a block closes over to wherever the block runs.
"""

import time
import warnings
from typing import Any

import nnsight
import torch
from nnterp import StandardizedTransformer
from torch import Tensor

from .. import ops, steps
from ..address import COMPONENTS
from ..plan import Model, Plan
from . import engine
from .standard import SIZES, by_site, envoy, keep, loader, nnterp_name, now, stops


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

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.model.tokenizer)

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
                 options: dict) -> tuple[Tensor, dict[str, Any]]:
        """`model.generate` over the batch, with `options` as its keyword arguments: the new ids, and each read.

        One loop over the forward calls visits, at each call, every site in forward order and does there what
        `now` says. It is bounded by the steps the reads ask for, or open when a read asks for every step;
        either way it ends where the generation does, which nnsight warns of and which is expected here, so
        the ids are read in an invoke of their own, which the loop's end does not cut short."""
        model = self.model
        sites = by_site(writes, reads)
        order = sorted(sites, key=lambda s: model.rank(nnterp_name(s[0]), s[1]))
        asked = [r["steps"] for r in reads if "steps" in r]
        steps = None if None in asked else sorted({0, *(s for a in asked for s in a)})
        got: dict[str, Any] = {r["name"]: [] for r in reads if "steps" in r}
        result: dict = {}
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask")}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*was never reached")
            with model.generate(**options) as tracer:
                with tracer.invoke(inputs):
                    for step in tracer.all() if steps is None else tracer.iter[steps]:
                        for component, layer in order:
                            here, there = now(*sites[(component, layer)], step)
                            if not here and not there:
                                continue
                            host = envoy(model, component, layer)
                            value = getattr(host, component)
                            if here:
                                value = ops.write(value, component, here)
                                setattr(host, component, value)
                            for record in there:
                                keep(got, record, ops.select(value, component, record))
                with tracer.invoke():
                    result["ids"] = tracer.result
        return result["ids"][:, batch["input_ids"].shape[1]:], got

    def readout(self, hidden: Tensor) -> Tensor:
        """nnterp's `project_on_vocab`: the final norm, the head, and whatever the model does after the head."""
        return self.model.project_on_vocab(hidden)
