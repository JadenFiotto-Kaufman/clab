"""The `forward` step kind."""

from ...plan.model import Plan
from . import kind
from .batches import batches, bind, joined, layers, read, stacked


@kind
def forward(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each read joined along rows.

    A read whose `layers` are a slice or picks is one read per layer, its ops run on each, stacked
    layer-first; a ragged read is gathered at the batch's widest row and kept as `{values, row, tok}`: one
    entry per token, its row and its token index."""
    parts = []
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        sites, writes, reads = bind(engine, plan, name, step, step.reads, state, found, chunk, pads)
        got = engine.forward(batch, writes, reads)
        parts.append({key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk,
                                                 pads, state, step.at, found["answers"])
                                            for layer in layers(engine, record)])
                      for key, record in step.reads.items()})
    joined(state, name, step, parts)
