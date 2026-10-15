"""clab runs mechanistic-interpretability experiments from declarative JSON documents.

`Plan` is a document as a model: compile it, run it on an engine, dump it to a directory, load it back,
table its values. `run` does the whole of that in one call; `describe` prints the format's tables.
"""

import os
from typing import TYPE_CHECKING

from .cli import describe
from .plan.model import Plan, name

if TYPE_CHECKING:
    from .engines import Engine

__all__ = ["Plan", "run", "describe"]


def run(plan: "Plan | str", engine: "str | Engine" = "nnterp", out: str | None = None, batch_size: int = 8,
        device: str | None = None, **options) -> Plan:
    """`Plan(path)` -> `compile` -> `execute` on `engine` -> `dump` into `out` (default `out/<document name>`).
    Returns the filled plan. `engine` is a registered name, loaded on `device` (default `cpu`; for nnterp and
    hooks one torch device or a device map transformers computes, `"auto"` among them), or a loaded engine,
    so one model serves many documents. Each of `options` goes where the engine declares it: to its
    constructor if it loads with it (`remote="local"` for ndif), to `execute` if it runs with it. A loaded
    engine was placed and configured when it loaded: it takes run options only, and refuses a `device` other
    than its own."""
    from .engines import ENGINES, load, route

    compiled = (plan if isinstance(plan, Plan) else Plan(plan)).compile()
    if isinstance(engine, str):
        loads, runs = route(ENGINES[engine], options) if engine in ENGINES else ({}, {})
        loaded = load(engine, compiled.model, device or "cpu", **loads)          # refuses an unknown engine
    else:
        loads, runs = route(type(engine), options)
        if (device is not None and device != engine.device) or loads:
            raise ValueError(f"the loaded {engine.name} engine runs on {engine.device} with {engine.options}; device "
                             "and the options it loads with are given when an engine loads")
        loaded = engine
    filled = loaded.execute(compiled, batch_size=batch_size, **runs)
    filled.dump(out or os.path.join("out", name(filled)))
    return filled
