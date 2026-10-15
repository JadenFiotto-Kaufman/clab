"""The engine protocol, and the engines clab ships: `nnterp`, `hooks` and `ndif`.

An engine registers under its name with `@engine("<name>")` and loads its checkpoint in `__init__(model, *,
device="cpu", **options)` from a plan's `model` block. The run options that say how and where it runs are
`device` (where it places the model and the inputs; `cpu` by default) and the options its constructor's
keyword parameters declare, each with its default: `options(name)` reads them, and `load` refuses an unknown
key, naming the ones the engine takes, before anything loads; an option the engine cannot honor it refuses by
name. It keeps `device` and every option after defaults (`options`), which `execute` records in `plan.run`.

`execute(plan, batch_size=...)` is the single entry point and the only method an engine that ignores
`clab.runtime.steps` must have; it returns the filled plan. The rest is what `steps.run` asks of an engine:
one batch through the model with writes applied and reads gathered (`forward`), a generation (`generate`),
the model's own unembedding (`readout`), what it cannot serve (`unserved`), where a component falls in its
forward (`rank`, any sortable value), its sizes, its tokenizer, and the ids its generation stops on (`eos`).

A forward visits the sites in its own forward order, applying each site's writes before gathering its reads,
and a write's op argument may be `{"live": <read name>}`: the value of a read the same call has already made
(`got[<read name>]`), which `ops.write(value, component, writes, got)` resolves. Preflight has checked, by
`rank`, that the read's site comes first; an engine that runs the plan without `clab.runtime.steps` must resolve such
an argument the same way. An
engine holds its model and its configuration, nothing else: `spec` is the model block it loaded, `name`
the name it registered under, `device` and `options` how it runs.
"""

import inspect
from typing import Any, Callable, Protocol

from torch import Tensor

from .. import registry
from ..plan import Model, Plan


class Engine(Protocol):
    name: str
    spec: Model
    device: str
    options: dict[str, Any]

    def execute(self, plan: Plan, *, batch_size: int = 8) -> Plan: ...

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Any]]: ...

    def readout(self, hidden: Tensor) -> Tensor: ...

    def unserved(self) -> dict: ...

    def rank(self, component: str, layer: int | None) -> Any: ...

    def sizes(self) -> dict[str, int]: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def eos(self) -> list[int]: ...


ENGINES: dict[str, Callable[..., Engine]] = {}
# What every engine's constructor takes beside its own options.
SUPPLIED = ("self", "model", "device")


def engine(name: str) -> Callable:
    """A class decorator that registers an engine under `name`."""
    def register(cls):
        cls.name = name
        ENGINES[name] = cls
        return cls
    return register


def options(name: str) -> dict[str, inspect.Parameter]:
    """The options engine `name` takes beside `device`, with their defaults: its constructor's keyword
    parameters."""
    return registry.spelled(ENGINES[name].__init__, SUPPLIED)


def load(name: str, model: Model, device: str = "cpu", **given: Any) -> Engine:
    """Engine `name` loaded for `model`, on `device`, with the options `given`: an unknown engine, or an
    option it does not declare or of another type, is refused before anything loads."""
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; valid: {', '.join(ENGINES)}")
    registry.check(ENGINES[name].__init__, given, SUPPLIED, f"the {name} engine")
    return ENGINES[name](model, device=device, **given)


from . import nnterp, hooks, ndif  # noqa: E402,F401  (the order the tables list them)
