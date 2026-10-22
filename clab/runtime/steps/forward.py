"""The `forward` step kind."""

from ...plan.model import Plan, entries
from . import kind
from .batches import batches, bind, joined, layers, read, stacked


@kind
def forward(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each gathered entry joined along
    rows.

    An entry whose `layers` are a slice or picks is one read per layer, the ops after its last edit run on
    each, stacked layer-first; a ragged one is gathered at the batch's widest row and kept as
    `{values, row, tok}`: one entry per token, its row and its token index."""
    parts = []
    gathered = {k: r for k, r in entries(step).items() if r.gathered}
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        sites, records = bind(engine, plan, name, step, state, found, chunk, pads)
        got = engine.forward(batch, records)
        parts.append({key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk,
                                                 pads, state, step.at, found["answers"])
                                            for layer in layers(engine, record)])
                      for key, record in gathered.items()})
    joined(state, name, step, parts)
