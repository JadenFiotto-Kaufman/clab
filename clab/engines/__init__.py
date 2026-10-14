"""The engine protocol, and the engines clab ships.

An engine loads its checkpoint in `__init__` from a plan's `model` block and registers under its name with
`@engine("<name>")`. `execute(plan, ...)` is the single entry point and the only method an engine that
ignores `steps.py` must have; it returns the filled plan. The rest is what `steps.run` asks of an engine:
one batch through the model with writes applied and reads gathered (`forward`), a generation (`generate`),
the model's own unembedding (`readout`), what it cannot serve (`unserved`), its sizes, its tokenizer, the
ids its generation stops on (`eos`) and the options its `generate` takes (`generation_keys`). An
engine holds its model and its configuration, nothing else: `spec` is the model block it loaded, and `name`
the name it registered under.
"""

from typing import Any, Callable, Protocol

from torch import Tensor

from ..plan import Model, Plan


class Engine(Protocol):
    name: str
    spec: Model

    def execute(self, plan: Plan, *, batch_size: int = 8, remote: bool | str = False) -> Plan: ...

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Any]]: ...

    def readout(self, hidden: Tensor) -> Tensor: ...

    def unserved(self) -> dict: ...

    def sizes(self) -> dict[str, int]: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def eos(self) -> list[int]: ...

    def generation_keys(self) -> set[str]: ...


ENGINES: dict[str, Callable[[Model], Engine]] = {}


def engine(name: str) -> Callable:
    """A class decorator that registers an engine under `name`."""
    def register(cls):
        cls.name = name
        ENGINES[name] = cls
        return cls
    return register


from . import hooks, nnterp  # noqa: E402,F401
