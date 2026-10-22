"""The engine protocol, and the engines clab ships: `nnterp`, `hooks` and `ndif`.

An engine registers under its name with `@engine("<name>")` and loads its checkpoint in `__init__(model, *,
device="cpu", **options)` from a plan's `model` block. The run options that say how and where it runs are
`device` (where it places the model, its one placement argument, which each engine reads as it likes; `cpu` by
default), `batch_size`, and the options an engine declares, each with its default: those it loads with are its
constructor's keyword parameters, those it runs with its `execute`'s, and no option is both (`engine` refuses
one that is). What each option takes is a `<name>: <text>` line in the docstring of the method that declares
it (`batch_size`'s in the protocol's `execute`), which `clab describe engines` and FORMAT.md print.
`options(name)` reads them; `route` sends each option given to the one that declares it and refuses one
neither does, naming both lists, before anything loads; an option the engine cannot honor it refuses by name.
It keeps `device` as asked (None where it does not place the model: an NDIF server does) and the options it
loaded with (`options`), which `execute` records in `plan.run` beside the ones it runs with, after their
defaults.

`execute(plan, batch_size=..., **options)` is the single entry point and the only method an engine that
ignores `clab.runtime.steps` must have; it returns the filled plan. The rest is what `steps.run` asks of an engine:
one batch through the model with its records done (`forward`), a generation (`generate`),
the model's own unembedding (`readout`), what it cannot serve (`unserved`), where a component falls in its
forward (`rank`, any sortable value), its sizes, its tokenizer, and the ids its generation stops on (`eos`).

The records are one list in the order a step's entries run: a write (it has `ops`) or a read (it has none),
each at a site. A forward visits the sites in its own forward order and does each site's records there in
turn, a write on the value as the records before it left it and a read taking that value, with
`ops.visit(value, component, records, got)`; it gathers nothing a record does not ask for. A write's op argument
may be `{"live": <read name>}`: the value of a read the same call has already made (`got[<read name>]`), which
`ops.visit` resolves. Preflight has checked, by `rank`, that the read's site comes first; an engine that runs
the plan without `clab.runtime.steps` must resolve such an argument the same way. An
engine holds its model and its configuration, nothing else: `model` is the model (whatever it runs, with
`.parameters()`), `spec` the model block it loaded, `name` the name it registered under, `device` and
`options` how it runs.
"""

import inspect
from typing import Any, Callable, Iterator, Protocol

from torch import Tensor

from .. import registry
from ..plan import Model, Plan


class Parameterized(Protocol):
    """A model as the run reads it: its parameters, whose dtypes and devices `plan.run` records."""

    def parameters(self) -> Iterator[Tensor]: ...


class Engine(Protocol):
    name: str
    spec: Model
    model: Parameterized
    device: str | None
    options: dict[str, Any]

    def execute(self, plan: Plan, *, batch_size: int = 8, **options: Any) -> Plan:
        """The filled plan.

        batch_size: how many of a step's rows go through the model in one forward or generation
        """
        ...

    def forward(self, batch: dict, records: list[dict]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict, records: list[dict], options: dict) -> tuple[Tensor, dict[str, Any]]: ...

    def readout(self, hidden: Tensor) -> Tensor: ...

    def unserved(self) -> dict: ...

    def rank(self, component: str, layer: int | None) -> Any: ...

    def sizes(self) -> dict[str, int]: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def eos(self) -> list[int]: ...


ENGINES: dict[str, Callable[..., Engine]] = {}
# What every engine's constructor and `execute` take beside their own options.
SUPPLIED = {"loads": ("self", "model", "device"), "runs": ("self", "plan", "batch_size")}


def engine(name: str) -> Callable:
    """A class decorator that registers an engine under `name`; an option it declares both to load and to run
    is refused here, since a run option routes to one of them by name."""
    def register(cls):
        both = set(declared(cls)["loads"]) & set(declared(cls)["runs"])
        if both:
            raise ValueError(f"engine {name!r}: {', '.join(sorted(both))} is both a load and a run option")
        cls.name = name
        ENGINES[name] = cls
        return cls
    return register


def declared(cls) -> dict[str, dict[str, inspect.Parameter]]:
    """The options an engine class takes, with their defaults: those it loads with (its constructor's keyword
    parameters beside `device`) and those it runs with (its `execute`'s beside `batch_size`)."""
    return {"loads": registry.spelled(cls.__init__, SUPPLIED["loads"]),
            "runs": registry.spelled(cls.execute, SUPPLIED["runs"])}


def options(name: str) -> dict[str, dict[str, inspect.Parameter]]:
    """The options engine `name` loads with and runs with (`declared`)."""
    return declared(ENGINES[name])


def route(cls, given: dict) -> tuple[dict, dict]:
    """Run options `given` split by name into those the engine loads with and those it runs with, each checked
    against its signature; a key it declares for neither is refused, naming both lists (`device` and
    `batch_size` first, which every engine takes)."""
    found = declared(cls)
    for key in given:
        if key not in found["loads"] and key not in found["runs"]:
            raise ValueError(f"the {cls.name} engine takes no {key!r}; it loads with: "
                             f"{', '.join(['device', *found['loads']])}; it runs with: "
                             f"{', '.join(['batch_size', *found['runs']])}")
    loads = {k: v for k, v in given.items() if k in found["loads"]}
    runs = {k: v for k, v in given.items() if k in found["runs"]}
    registry.check(cls.__init__, loads, SUPPLIED["loads"], f"the {cls.name} engine")
    registry.check(cls.execute, runs, SUPPLIED["runs"], f"the {cls.name} engine")
    return loads, runs


def load(name: str, model: Model, device: str = "cpu", **given: Any) -> Engine:
    """Engine `name` loaded for `model`, on `device`, with the load options `given`: an unknown engine, an
    option it does not declare or of another type, and a run option (which goes to `execute`) are refused
    before anything loads."""
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; valid: {', '.join(ENGINES)}")
    loads, runs = route(ENGINES[name], given)
    if runs:
        raise ValueError(f"{', '.join(runs)}: a run option of the {name} engine; it goes to execute, not load")
    return ENGINES[name](model, device=device, **loads)


from . import nnterp, hooks, ndif  # noqa: E402,F401  (the order the tables list them)
