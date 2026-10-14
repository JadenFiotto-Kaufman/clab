"""The engine protocol, and the engines clab ships.

`execute(plan, ...)` is the single entry point and the only method an engine
that ignores `steps.py` must have; it returns the filled plan. The rest is
what `steps.run` asks of an engine: one batch through the model with writes
applied and reads gathered (`forward`), a generation (`generate`), the
model's own unembedding (`readout`), its sizes and its tokenizer.
"""

from typing import Any, Callable, Protocol

from torch import Tensor


class Engine(Protocol):
    def execute(self, plan: dict, *, batch_size: int = 8, remote: bool | str = False) -> dict: ...

    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]: ...

    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Tensor]]: ...

    def readout(self, hidden: Tensor) -> Tensor: ...

    def sizes(self) -> dict[str, int]: ...

    @property
    def tokenizer(self) -> Any: ...


from . import hooks, nnterp  # noqa: E402

ENGINES: dict[str, Callable[[dict], Engine]] = {"nnterp": nnterp.load, "hooks": hooks.load}
