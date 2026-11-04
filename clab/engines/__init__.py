"""The engine protocol, and the engines clab ships: `nnterp`, `hooks`, `ndif` and `vllm`. The nnterp engine's `readout` is the
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

`execute(plan, batch_size=..., padding=..., readout=...)` is the single entry point and the only method an engine
that ignores `clab.steps` must have; it returns the filled plan, `padding` (`data.PADDINGS`) and `readout`
(`schedule.READOUTS`) recorded in `plan.run` beside the batch size, by which `steps.run` pads every batch and preflight
writes each forward's schedule. A hub dataset's rows are not in the plan: an engine that ignores `clab.steps` loads
them where it runs with `compile.dataset_rows`, as `steps.run` does (`steps.plan_with_rows`).
The rest is what `steps.run` asks of an engine: one batch through the model with a bound schedule executed (`forward`),
a generation (`generate`), the model's own unembedding (`readout`), what it cannot serve, layer by layer (`unserved`),
where a component falls in its forward (`rank`, any sortable value, which preflight reads once per site to write the
schedule's order), its sizes, its tokenizer, and the ids its generation stops on (`eos`).

A bound schedule (`batches.bind_schedule`) is a forward's `schedule` bound to one batch: `sites`, each its `component`
and `layer` and its `records`, the sites in this engine's forward order and the records in the order they are done at
each, a write (it has `ops`) or a read (it has `name`); `stop`, the site a forward ends at, or None; and for a
generation `loop`, the forward calls to run the records at (None: every one), and `online`, whether a write is found
call by call from the inputs fed so far. An engine reaches each site's module in turn and does its records there with
`batches.visit_site(value, component, records, got)`, a write on the value as the records before it left it and a read
taking that value into `got` under its name, and decides nothing: it visits what the schedule lists, in its order,
gathers nothing a record does not ask for, and ends the forward right after `stop` where it declares that it `stops` (a
class attribute; an engine that runs on returns the same values, later); a generation runs whole, each call doing what
`batches.acting_at_call` and `batches.stamped_at_call` say of the records. A write's op argument may be `{"live": <read
name>}`: the value of a read the same call has already made (`got[<read name>]`), which `batches.visit_site` resolves;
the schedule has the read's site first. An engine holds its model and its configuration, nothing else: `model` is the
model (whatever it runs, with `.parameters()`), `spec` the model block it loaded, `name` the name it registered under,
`device` and `options` how it loaded, `remote` what its session's `remote=` is (False for a run in this process),
`stops` whether it honours the schedule's stop, and `cuda` the CUDA devices a step's cost measures the peak memory of:
the model's, or none where the run is remote (the server's process is its own to account). Two attributes an engine
may add, which preflight reads where present: `last_only`, the components it serves at the prompt's last position
alone, each mapped to the reason (the vllm engine's head), any other prefill position refused by name; and `longest`,
the longest sequence it serves, prompt and new tokens (the vllm engine's `max_len`), a step whose rows do not fit
refused by name. And one `steps.start_run` reads where present, on the client before any session opens: `gradless`,
why the engine refuses a `grad` entry (the vllm engine's, the nnterp engine's under `tp`; None where it takes them),
a step's `grad` entry refused by name with it. It does not cover a fit, which the vllm engine refuses in its
`execute` and the nnterp engine trains under `tp`.
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
    stops: bool

    @property
    def device(self) -> str | None: ...

    @property
    def cuda(self) -> list[str]: ...

    def execute(self, plan: Plan, *, batch_size: int = 8, padding: str = "dataset",
                readout: str = "full") -> Plan: ...

    def forward(self, batch: dict[str, Tensor], bound: dict[str, Any]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict[str, Tensor], bound: dict[str, Any],
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


def load(name: str, model: Model, device: str | None = None, **given: Any) -> Engine:
    """Engine `name` loaded for `model`, on `device` (None: the engine's own default, its constructor's, `cpu` but
    for the vllm engine's `cuda:0`), with its options `given`: an unknown engine, an option it does not declare, one
    of another type, and torchrun's processes for an engine without `tp` are refused before anything loads (a
    `device` no engine can place a model on, every constructor refuses: `standard.loading`)."""
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; valid: {', '.join(ENGINES)}")
    if device is None:
        device = registry.signature(ENGINES[name].__init__).parameters["device"].default
    for key in given:
        if key not in options(name):
            raise ValueError(f"the {name} engine takes no {key!r}; its options: {', '.join(options(name)) or 'none'} "
                             "(where the model runs is `device`, given as --device)")
    world = os.environ.get("WORLD_SIZE", "1")
    if int(world) > 1 and "tp" not in options(name):
        raise ValueError(f"WORLD_SIZE {world}: torchrun started {world} processes, and the {name} engine takes no "
                         "`tp` to shard the model over them, so each would run the whole plan; run it without torchrun")
    registry.check(ENGINES[name].__init__, given, SUPPLIED, f"the {name} engine")
    return ENGINES[name](model, device=device, **given)


from clab.engines import nnterp, hooks, ndif, vllm  # noqa: E402,F401,I001  (the order the tables list them)
