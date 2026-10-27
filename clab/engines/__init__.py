"""The engine protocol, and the engines clab ships: `nnterp`, `hooks` and `ndif`. The nnterp engine's `readout` is the
model's own; the hooks engine's is the standard form (final norm, head, the config's softcap), which agrees on the
corpus (the agreement matrix pins it, tiny Gemma-2's softcap included) and is not promised beyond it.

An engine registers under its name with `@engine("<name>")` and loads its checkpoint in `__init__(model, *,
device="cpu", **options)` from a plan's `model` block: `device` is where it places the model (its one placement
argument, which each engine reads as it likes; `cpu` by default) and `options` are its own keyword parameters,
each with its default and a `<name>: <what it takes>` line in the constructor's docstring, which `clab describe
engines` and FORMAT.md print. `load` refuses an option the engine does not declare, naming the ones it does,
and under torchrun (WORLD_SIZE above 1) an engine without `tp`, each of whose processes would run the whole plan,
before anything loads; an option the engine cannot honor it refuses by name. It keeps `device` as asked (None
where it does not place the model: an NDIF server does) and `options` as given, after their defaults, which
`execute` records in `plan.run`.

`execute(plan, batch_size=...)` is the single entry point and the only method an engine that ignores
`clab.steps` must have; it returns the filled plan. A hub dataset's rows are not in the plan: an engine that
ignores `clab.steps` loads them where it runs with `compile.materialized`, as `steps.run` does (`steps.held`). The
rest is what `steps.run` asks of an engine: one batch through the model with its records done (`forward`), a
generation (`generate`), the model's own unembedding (`readout`), what it cannot serve, layer by layer
(`unserved`), where a component falls in its forward (`rank`, any sortable value), its sizes, its tokenizer, and
the ids its generation stops on (`eos`).

The records are one list in the order a step's entries run: a write (it has `ops`) or a read (it has none),
each at a site. A forward visits the sites in its own forward order and does each site's records there in
turn, a write on the value as the records before it left it and a read taking that value, with
`batches.visit(value, component, records, got)`; it gathers nothing a record does not ask for, and computes
nothing past its deepest site, by `rank`: it stops right after it (an engine that runs on returns the same
values, later); a generation runs whole. A write's op argument may be `{"live": <read name>}`: the value of a
read the same call has already made (`got[<read name>]`), which `batches.visit` resolves. Preflight has
checked, by `rank`, that the read's site comes first; an engine that runs the plan without `clab.steps` must
resolve such an argument the same way. An engine holds its model and
its configuration, nothing else: `model` is the model (whatever it runs, with `.parameters()`), `spec` the model
block it loaded, `name` the name it registered under, `device` and `options` how it loaded, `remote` what its
session's `remote=` is (False for a run in this process), and `cuda` the CUDA devices a step's cost measures the
peak memory of: the model's, or none where the run is remote (the server's process is its own to account).
"""

import inspect
import os
from typing import Any, Callable, Iterator, Protocol

from torch import Tensor

from clab.format import registry
from clab.plan import Model, Plan

# What every engine's constructor takes beside its own options.
SUPPLIED = ("self", "model", "device")


class Parameterized(Protocol):
    """A model as the run reads it: its parameters, whose dtypes and devices `plan.run` records."""

    def parameters(self) -> Iterator[Tensor]: ...


class Engine(Protocol):
    name: str
    spec: Model
    model: Parameterized
    options: dict[str, Any]
    remote: bool | str

    @property
    def device(self) -> str | None: ...

    @property
    def cuda(self) -> list[str]: ...

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan: ...

    def forward(self, batch: dict[str, Tensor], records: list[dict]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict[str, Tensor], records: list[dict],
                 options: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]: ...

    def readout(self, hidden: Tensor) -> Tensor: ...

    def unserved(self) -> dict[str, str | dict[int, str] | None]: ...

    def rank(self, component: str, layer: int | None) -> Any: ...

    def sizes(self) -> dict[str, int]: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def eos(self) -> list[int]: ...


ENGINES: dict[str, Callable[..., Engine]] = {}


def engine(name: str) -> Callable:
    """A class decorator that registers an engine under `name`."""
    def register(cls: type) -> type:
        cls.name = name
        ENGINES[name] = cls
        return cls
    return register


def options(name: str) -> dict[str, inspect.Parameter]:
    """The options engine `name` loads with: its constructor's keyword parameters beside `device`."""
    return registry.spelled(ENGINES[name].__init__, SUPPLIED)


def load(name: str, model: Model, device: str = "cpu", **given: Any) -> Engine:
    """Engine `name` loaded for `model`, on `device`, with its options `given`: an unknown engine, an option it
    does not declare, one of another type, and torchrun's processes for an engine without `tp` are refused before
    anything loads."""
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; valid: {', '.join(ENGINES)}")
    for key in given:
        if key not in options(name):
            raise ValueError(f"the {name} engine takes no {key!r}; its options: {', '.join(['device', *options(name)])}")
    world = os.environ.get("WORLD_SIZE", "1")
    if int(world) > 1 and "tp" not in options(name):
        raise ValueError(f"WORLD_SIZE {world}: torchrun started {world} processes, and the {name} engine takes no "
                         "`tp` to shard the model over them, so each would run the whole plan; run it without torchrun")
    registry.check(ENGINES[name].__init__, given, SUPPLIED, f"the {name} engine")
    return ENGINES[name](model, device=device, **given)


from clab.engines import nnterp, hooks, ndif  # noqa: E402,F401,I001  (the order the tables list them)
