"""The primary engine: nnterp's StandardizedTransformer, everything inside one nnsight session.

`device` is one torch device, which nnsight's loader takes as its `device`, or a device map transformers
computes (`auto`, `balanced`, ...), which it takes as its `device_map` and which may spread the model over
several GPUs; the batch's inputs go where the model's first parameter is, and transformers moves them on
from there.

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

from clab import steps
from clab.engines import engine
from clab.engines.standard import COMPUTED, SIZES, build, by_site, envoy, inputs, nnterp_name, stops
from clab.format.address import COMPONENTS
from clab.plan import Model, Plan
from clab.steps import batches
from clab.steps.batches import landed, now


@engine("nnterp")
class NnterpEngine:
    name: str
    remote: bool | str = False      # what the session's `remote=` is: the ndif engine's option

    def __init__(self, model: Model, *, device: str = "cpu"):
        """The checkpoint `model` pins, placed by `device`: the loader's `device_map` when transformers computes
        it, else its `device`; the block's other keys go to the loader as they are.

        device: one torch device ("cpu", "cuda:0") as the loader's device, or "auto", "balanced", ... as its device_map
        """
        self.spec, self.device, self.options = model, device, {}
        placed = {"device_map": device} if device in COMPUTED else {"device": device}
        self.model = build(model, "nnterp", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=True, dtype=getattr(torch, model.dtype), **placed, **extra))

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan:
        """The whole plan in one session, opened with `remote` (False here; the ndif engine's option there);
        the results and the run record are the one thing that comes home, into the client's own plan."""
        started = steps.begin(self, plan, batch_size)
        with self.model.session(remote=self.remote):
            steps.run(self, plan, batch_size)
            home = nnsight.save({"results": plan.results, "run": plan.run})
        plan.results, plan.run = home["results"], home["run"]
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

    def rank(self, component: str, layer: int | None) -> tuple:
        """Where the component falls in this checkpoint's forward: nnterp's measured order."""
        return self.model.rank(nnterp_name(component), layer)

    def sizes(self) -> dict[str, int]:
        return {k: getattr(self.model, k) for k in SIZES}

    def forward(self, batch: dict, records: list[dict]) -> dict[str, Tensor]:
        """One trace over the batch, visiting each site in forward order and doing its records there in turn."""
        model = self.model
        sites = by_site(records)
        order = sorted(sites, key=lambda s: model.rank(nnterp_name(s[0]), s[1]))
        got: dict[str, Tensor] = {}
        with model.trace(inputs(batch, ("input_ids", "attention_mask", "position_ids"), batches.home(self))):
            for component, layer in order:
                host = envoy(model, component, layer)
                value, wrote = batches.visit(getattr(host, component), component, sites[(component, layer)], got)
                if wrote:
                    setattr(host, component, value)
        return got

    def generate(self, batch: dict, records: list[dict], options: dict) -> tuple[Tensor, dict[str, Any]]:
        """`model.generate` over the batch, with `options` as its keyword arguments: the new ids, and each read.

        One loop over the forward calls visits, at each call, every site in forward order and does there what
        `now` says (and, for a write with an online scope, `landed`, from the inputs seen so far: each decode
        call's `input_ids`, read first). It is bounded by the steps the reads ask for, or open when a read asks
        for every step;
        either way it ends where the generation does, which nnsight warns of (that one warning is silenced; a
        read out of forward order shows up instead as a missing step, which `generate.generate` refuses), so the
        ids are read in an invoke of their own, which the loop's end does not cut short."""
        model = self.model
        sites = by_site(records)
        order = sorted(sites, key=lambda s: model.rank(nnterp_name(s[0]), s[1]))
        asked = [r["calls"] for r in records if "calls" in r]
        loop = None if None in asked else sorted({0, *(s for a in asked for s in a)})
        got: dict[str, Any] = {r["name"]: [] for r in records if "calls" in r and "ops" not in r}
        online = any("online" in r for r in records)
        result: dict = {}
        seen: list = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".* was never reached: the loop asked for a step the run did "
                                                      r"not make")
            with model.generate(**options) as tracer:
                with tracer.invoke(inputs(batch, ("input_ids", "attention_mask"), batches.home(self))):
                    for step in tracer.all() if loop is None else tracer.iter[loop]:
                        if online and step > 0:
                            seen.append(model.input_ids)
                        for component, layer in order:
                            here = now(sites[(component, layer)], step)
                            here = landed(here, seen, model.tokenizer) if step > 0 else here
                            if not here:
                                continue
                            host = envoy(model, component, layer)
                            value, wrote = batches.visit(getattr(host, component), component, here, got)
                            if wrote:
                                setattr(host, component, value)
                with tracer.invoke():
                    result["ids"] = tracer.result
        return result["ids"][:, batch["input_ids"].shape[1]:], got

    def readout(self, hidden: Tensor) -> Tensor:
        """nnterp's `project_on_vocab`: the final norm, the head, and whatever the model does after the head."""
        return self.model.project_on_vocab(hidden)
