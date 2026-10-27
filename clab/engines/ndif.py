"""The nnterp engine on an NDIF server: the same engine, with a weightless model and a remote session.

It builds nnterp's StandardizedTransformer without dispatching it (a meta shell: the tree, the config and
the tokenizer, no weights; nothing downloads but those) and opens the run's one session with `remote`. The
session body is the nnterp engine's: what ships is the engine (by reference: clab is installed on the server),
the plan (pure data) and the batch size; everything that needs the model, preflight included, runs on the
server, and the results and the run record come home. The server's address and API key are nnsight's own
(`nnsight.CONFIG`, or its environment variables). `remote="local"` runs nnsight's in-process simulation of
that trip, which dispatches the shell here, on `device`, at its first trace.
"""

from typing import Literal

import torch
from nnterp import StandardizedTransformer

from clab.engines import engine
from clab.engines.nnterp import NnterpEngine
from clab.engines.standard import build, mrope, vision
from clab.format import data
from clab.plan import Model, Plan, walk


@engine("ndif")
class NdifEngine(NnterpEngine):
    def __init__(self, model: Model, *, device: str = "cpu", remote: Literal[True, "local"] = True) -> None:
        """`model`'s meta shell; `remote` is True for the NDIF server nnsight is configured for, or "local"
        for its in-process simulation, which places the model on `device`. On a server the server places it,
        so `device` stays at its default and the engine's is None (`plan.run.devices` says where it ran).

        device: one torch device ("cpu", "cuda:0") where remote "local" places the model; under true, "cpu" alone
        remote: true for the NDIF server nnsight is configured for, "local" for nnsight's in-process simulation of it
        """
        if remote is True and device != "cpu":
            raise ValueError(f"device: the ndif engine cannot honor {device!r}: the server places the model")
        try:
            torch.device(device)
        except RuntimeError:
            raise ValueError(f"device: the ndif engine places a local model on one torch device ('cpu', "
                             f"'cuda:0'), not {device!r}") from None
        self.spec, self.options, self.remote = model, {"remote": remote}, remote
        self.device: str | None = None if remote is True else device
        self.model = build(model, "ndif", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device=device, dtype=getattr(torch, model.dtype),
            **extra))
        self.vision = vision(getattr(self.model, "processor", None), self.model.config)
        self.mrope = mrope(self.model._module)

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan:
        """The nnterp engine's `execute`, after refusing by name a chat with an image part: the image is a file
        beside the document, which the server has no way to read."""
        for name, step in walk(plan.steps):
            for i, row in enumerate((plan.data[step.dataset].rows if step.kind in ("forward", "generate") and
                                     step.dataset else None) or []):
                if data.image_parts(data.field(row, step.input)):
                    raise ValueError(f"{name}: {data.input_name(step.input)} row {i} has an image part, a file "
                                     "under the corpus root's data/, which the ndif engine's server cannot read; "
                                     "the nnterp and hooks engines run it here")
        return super().execute(plan, batch_size=batch_size)
