"""clab runs mechanistic-interpretability experiments from declarative JSON documents.

`Plan` is a document as a model: compile it, run it on an engine, dump it to a directory, load it back,
table its values. `run` does the whole of that in one call; `describe` prints the format's tables.
"""

import os

from .cli import describe
from .plan import Plan, name

__all__ = ["Plan", "run", "describe"]


def run(plan: "Plan | str", engine="nnterp", out: str | None = None, batch_size: int = 8,
        remote: bool | str = False) -> Plan:
    """`Plan(path)` -> `compile` -> `execute` on `engine` (a registered name, or a loaded engine, so one
    model serves many documents) -> `dump` into `out` (default `out/<document name>`). Returns the filled plan."""
    from .engines import ENGINES

    compiled = (plan if isinstance(plan, Plan) else Plan(plan)).compile()
    loaded = ENGINES[engine](compiled.model) if isinstance(engine, str) else engine
    filled = loaded.execute(compiled, batch_size=batch_size, remote=remote)
    filled.dump(out or os.path.join("out", name(filled)))
    return filled
