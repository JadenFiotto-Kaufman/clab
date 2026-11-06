"""The nnterp engine on an NDIF server: the same engine, with a weightless model and a remote session.

It builds nnterp's StandardizedTransformer without dispatching it (a meta shell: the tree, the config and
the tokenizer, no weights; nothing downloads but those) and opens the run's one session with `remote`. The
session body is the nnterp engine's: what ships is the engine (by reference: nnspec is installed on the server),
the plan (pure data) and the batch size; everything that needs the model, preflight included, runs on the
server, and the results and the run record come home. The server holds the weights and shards the model as its
deployment does, so on a server a quantized `dtype` is refused by name, and the engine takes no `tp`. The server's
address and API key are nnsight's own (`nnsight.CONFIG`, or its environment variables). `remote="local"` runs
nnsight's in-process simulation of that trip. The shell then loads its weights here, on `device`, before the first
step of a run that calls the model (`NnterpEngine.ready`, which `execute` calls before the session opens). Its
forwards stop where the schedule says, under either `remote`, as the nnterp engine's do (on a server, as the nnspec
installed there says), since the nnsight nnspec requires leaves no frame of a stopped forward behind. A `load`'s value
travels in the request, inside the plan, as a bundle's tensors do, under either `remote`, so `remote="local"` runs
what a server runs and a server reads no file of the client's; a document whose loads hold more than 512 MiB of
tensors is refused before anything is sent.
"""

from typing import Literal

import torch
from nnterp import StandardizedTransformer

from nnspec.engines import engine, standard
from nnspec.engines.nnterp import NnterpEngine
from nnspec.engines.standard import build, loading, placing
from nnspec.plan import QUANTIZED, Model


@engine("ndif")
class NdifEngine(NnterpEngine):
    def __init__(self, model: Model, *, device: str = "cpu", remote: Literal[True, "local"] = True) -> None:
        """`model`'s meta shell; `remote` is True for the NDIF server nnsight is configured for, or "local"
        for its in-process simulation, which places the model on `device`. On a server the server places it,
        so `device` stays at its default and the engine's is None (`plan.run.devices` says where it ran), and it
        holds the weights as its deployment does, so a quantized `dtype` is refused by name there.

        device: one torch device ("cpu", "cuda:0") where remote "local" places the model; under true, "cpu" alone
        remote: true for the NDIF server nnsight is configured for, "local" for nnsight's in-process simulation of it
        """
        if remote is True and device != "cpu":
            raise ValueError(f"device: the ndif engine cannot honor {device!r}: the server places the model")
        if remote is True and model.dtype in QUANTIZED:
            raise ValueError(f"model.dtype {model.dtype!r}: the ndif engine cannot honor it: the server decides how a "
                             "model's weights are held")
        try:
            torch.device(device)
        except RuntimeError:
            raise ValueError(f"device: the ndif engine places a local model on one torch device ('cpu', "
                             f"'cuda:0'), not {device!r}") from None
        self.spec, self.options, self.remote = model, {"remote": remote}, remote
        self.device: str | None = None if remote is True else device
        loaded = loading(model, "ndif", device)
        self.model = build(model, "ndif", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, **placing(model, device), **loaded, **extra))
        self.config = {"sizes": standard.sizes(self.model)}         # the shell's, the checkpoint's
