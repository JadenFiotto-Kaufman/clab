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

from ..plan import Model
from . import engine
from .nnterp import NnterpEngine
from .standard import build


@engine("ndif")
class NdifEngine(NnterpEngine):
    def __init__(self, model: Model, *, device: str = "cpu", remote: Literal[True, "local"] = True):
        """`model`'s meta shell; `remote` is True for the NDIF server nnsight is configured for, or "local"
        for its in-process simulation, which places the model on `device`. On a server the server places it,
        so `device` stays at its default and the engine's is None (`plan.run.devices` says where it ran).

        device: one torch device ("cpu", "cuda:0") where remote "local" places the model; under true, "cpu" alone
        remote: true for the NDIF server nnsight is configured for, "local" for nnsight's in-process simulation of it
        """
        if remote is True and device != "cpu":
            raise ValueError(f"device: the ndif engine cannot honor {device!r}: the server places the model")
        self.spec, self.options, self.remote = model, {"remote": remote}, remote
        self.device: str | None = None if remote is True else device
        self.model = build(model, "ndif", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device=device, dtype=getattr(torch, model.dtype),
            **extra))
