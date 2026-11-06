"""nnspec runs mechanistic-interpretability experiments from declarative JSON documents.

`Plan` is a document as a model: compile it, run it on an engine, dump it to a directory, load it back,
table its values. `run` does the whole of that in one call; `describe` prints the format's tables.
"""

import os
from typing import TYPE_CHECKING, Any

from nnspec.docs import describe
from nnspec.plan import Plan, name

if TYPE_CHECKING:
    from nnspec.engines import Engine

__all__ = ["Plan", "run", "describe"]


def run(plan: "Plan | str", engine: "str | Engine" = "nnterp", out: str | None = None, batch_size: int = 8,
        device: str | None = None, padding: str = "dataset", readout: str = "full", **options: Any) -> Plan:
    """`Plan(path)` -> `compile` -> `execute` on `engine` -> `dump` into `out` (default `out/<document name>`).
    Returns the filled plan. `padding` is how each batch is padded (`format.data.PADDINGS`: `dataset`, the
    default, `batch` or `sorted`), and `readout` how a forward computes `logits` (`compile.schedule.READOUTS`: `full`,
    the default, or `at-reads`). `engine` is a registered name, loaded on `device` (default the engine's own,
    `cpu`, or `cuda:0` for vllm; for nnterp and hooks one torch device or a device map transformers computes,
    `"auto"` among them) with the engine's own
    `options` (`remote="local"` for ndif), or a loaded engine, so one model serves many documents. A loaded
    engine was configured, its device among it, when it loaded, so it takes no `device` or option here. Under torchrun
    (the nnterp engine's `tp`) every rank runs and returns the filled plan, and rank 0 alone writes `out`. A run
    that stops at a step (out of CUDA memory, `plan.run.stopped`) is written to `out` as far as it got, in this
    process, before its refusal is raised, which then says where."""
    from nnspec.engines import load
    from nnspec.steps import check_run_options

    check_run_options(batch_size, padding, readout)
    compiled = (plan if isinstance(plan, Plan) else Plan(plan)).compile()
    if isinstance(engine, str):
        engine = load(engine, compiled.model, device, **options)
    elif options or (device is not None and device != engine.device):
        raise ValueError(f"the loaded {engine.name} engine runs on {engine.device} with {engine.options}; device "
                         "and options are given when an engine loads")
    first = os.environ.get("RANK", "0") == "0"
    where = out or os.path.join("out", name(compiled))
    try:
        filled = engine.execute(compiled, batch_size=batch_size, padding=padding, readout=readout)
    except ValueError as e:
        if first and compiled.run is not None and compiled.run.stopped:
            compiled.dump(where)
            raise ValueError(f"{e}; the steps before it are saved in {where}") from e
        raise
    if first:
        filled.dump(where)
    return filled
