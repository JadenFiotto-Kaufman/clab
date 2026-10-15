"""clab runs mechanistic-interpretability experiments from declarative JSON documents.

`Plan` is a document as a model: compile it, run it on an engine, dump it to a directory, load it back,
table its values. `run` does the whole of that in one call; `describe` prints the format's tables.
"""

import os

from .cli import describe
from .plan.model import Plan, name

__all__ = ["Plan", "run", "describe"]


def run(plan: "Plan | str", engine="nnterp", out: str | None = None, batch_size: int = 8, device: str | None = None,
        **options) -> Plan:
    """`Plan(path)` -> `compile` -> `execute` on `engine` -> `dump` into `out` (default `out/<document name>`).
    Returns the filled plan. `engine` is a registered name, loaded on `device` (default `cpu`) with the engine's
    own `options` (`device_map="auto"` for nnterp or hooks, `remote="local"` for ndif), or a loaded engine, so
    one model serves many documents; a loaded engine was placed and configured when it loaded, and is refused
    a `device` other than its own and any option."""
    from .engines import load

    compiled = (plan if isinstance(plan, Plan) else Plan(plan)).compile()
    if not isinstance(engine, str) and ((device is not None and device != engine.device) or options):
        raise ValueError(f"the loaded {engine.name} engine runs on {engine.device} with {engine.options}; device "
                         "and options are given when an engine loads")
    loaded = load(engine, compiled.model, device or "cpu", **options) if isinstance(engine, str) else engine
    filled = loaded.execute(compiled, batch_size=batch_size)
    filled.dump(out or os.path.join("out", name(filled)))
    return filled
