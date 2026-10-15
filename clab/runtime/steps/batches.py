"""A step's input in batches, and a record bound to one batch: its site in the padded batch, its writes'
ops, its reads' values with their ops applied."""

from typing import Any, cast

import torch
from torch import Tensor

from ... import address
from ...plan.model import Plan, arguments, encodes, space
from .. import data, featurizers, ops, positions
from .preflight import head
from .state import bound, lookup, put


def site(record, rows: list[list[int]], pads: list[int], last_only: bool = False) -> dict:
    """What the engine needs to reach a record's selection in this batch: its component, its token indices
    into the padded batch (`positions`, `[rows, width]`; a ragged row repeats its last index to the batch's
    widest, `keep` marking its own, `tok` the unpadded indices), and the part of the feature axes it keeps
    (`select`). Where the value
    holds the last position alone (`last_only`: a generation's head in the prefill) every index is -1."""
    width = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (width - len(row))
             for row, pad in zip(rows, pads)]
    tok = [[i - pad for i in row] for row, pad in zip(index, pads)]
    index = [[-1] * width for _ in rows] if last_only else index
    return {"component": record.site.component, "positions": torch.tensor(index), "select": selection(record, pads),
            "keep": torch.tensor([[j < len(row) for j in range(width)] for row in rows]), "tok": torch.tensor(tok)}


def decoding(record, n: int, pads: list[int]) -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last."""
    return {"component": record.site.component, "positions": torch.full((n, 1), -1), "select": selection(record, pads)}


def selection(record, pads: list[int]) -> dict:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps. On
    the `key` axis an index counts the row's own tokens, so each row's is shifted by its padding (`[rows, n]`),
    as a position is."""
    layout = address.dims(record.site.component, None, False)
    return {layout.index(axis): torch.tensor([[i + pad for i in v] for pad in pads] if axis == "key" else v)
            for axis, v in (record.site.select or {}).items()}


def edit(engine, plan: Plan, record, o, state: dict, at: dict, rows: slice, live: dict | None = None) -> dict:
    """A write's op bound to one batch, as `ops.write` runs it: in a featurizer's space, with its instance and
    the features its `select` keeps; an operand that names a read of the same forward (`live`: by name, the
    engine's name for it) is bound to that name, for the forward to fill."""
    dims = address.dims(record.site.component, None, False)
    within = space(o)
    return {"op": o.op, "args": bound(engine, plan, o.op, arguments(o), state, at, rows, live=live),
            "featurizer": lookup(state, within, at) if within is not None else None,
            "features": features(o), "dims": dims[:-1] + ["features"] if within is not None else dims}


def features(o) -> Tensor | None:
    """The features an op in a featurizer's space keeps, if it narrows them."""
    return torch.tensor(o.select["features"]) if o.select else None


def batches(engine, step, found: dict, batch_size: int):
    """The step's input in consecutive batches: each one's rows (a slice), its ids padded on the left to the
    input's longest row, and each row's padding."""
    encoded = found["inputs"][step.input]
    n = len(encoded["rows"])
    for lo in range(0, n, batch_size):
        chunk = slice(lo, min(lo + batch_size, n))
        ids = [e["ids"] for e in encoded["rows"][chunk]]
        yield chunk, data.pad(engine.tokenizer, ids, encoded["width"]), [encoded["width"] - len(x) for x in ids]


def bind(engine, plan: Plan, name: str, step, reads: dict, state: dict, found: dict, rows: slice,
         pads: list[int], beyond: frozenset | set = frozenset()) -> tuple[dict, list[dict], list[dict]]:
    """A step's prefill writes (all but `beyond`) and the prefill `reads` bound to one batch: each record's
    site, the writes with their ops, and the reads, one per layer (`<read>@<layer>`)."""
    writes_here = {k: w for k, w in step.writes.items() if k not in beyond}
    sites = {key: site(record, lookup(found["positions"], f"{name}.{key}", step.at)[rows], pads, head(step, record))
             for key, record in {**writes_here, **reads}.items()}
    live = {f"{name}.{key}": f"{key}@{record.site.layers}" for key, record in reads.items()}
    writes = [{**sites[key], "name": key, "layer": record.site.layers,
               "ops": [edit(engine, plan, record, o, state, step.at, rows, live) for o in record.ops]}
              for key, record in writes_here.items()]
    return sites, writes, [{**sites[key], "name": f"{key}@{layer}", "layer": layer}
                           for key, record in reads.items() for layer in layers(engine, record)]


def layers(engine, record) -> list:
    """The layers a read is made at: every one under `layers: "all"`, else its own."""
    return list(range(engine.sizes()["num_layers"])) if record.site.layers == "all" else [record.site.layers]


def stacked(record, values: list) -> Any:
    """A read's per-layer values as its value: the one, or under `layers: "all"` stacked layer-first."""
    if record.site.layers != "all":
        return values[0]
    if isinstance(values[0], Tensor):
        return torch.stack(values)
    return {**values[0], "values": torch.stack([v["values"] for v in values])}


def joined(state: dict, name: str, step, parts: list[dict]) -> None:
    """Each read's per-batch values joined along rows (a ragged read's entries, after the layer axis)."""
    for key, record in step.reads.items():
        axis = 1 if record.site.layers == "all" else 0
        values = [p[key] for p in parts]
        put(state, f"{name}.{key}", step.at, torch.cat(values, axis) if isinstance(values[0], Tensor) else
            {k: torch.cat([v[k] for v in values], axis if k == "values" else 0) for k in values[0]})


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, pads: list[int], state: dict,
         at: dict) -> Any:
    """One layer of a prefill read in one batch: its entries when it is ragged, then its ops."""
    if positions.width(positions.parse(record.pos)) is None:
        keep = site["keep"]
        value = {"values": value[keep], "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep],
                 "tok": site["tok"][keep]}
    return with_ops(engine, plan, record, value, state, at, rows)


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice) -> Any:
    """A read's value with its ops applied in listed order: one in a featurizer's space on the value's
    coordinates there, the change written back. Past a probe's `encode` a column names its classes."""
    dims = address.dims(record.site.component, None, positions.width(positions.parse(record.pos)) is None)
    vocabulary = None
    for o in record.ops:
        within = space(o)
        f = lookup(state, within, at) if within is not None else None
        args = bound(engine, plan, o.op, arguments(o), state, at, rows, value, vocabulary)
        value = apply(engine, o.op, value, args, dims[:-1] + ["features"] if f else dims, featurizer=f,
                      features=features(o))
        if f is None:
            dims = ops.out_dims(o.op, dims, arguments(o), ops.kind(o.op, args))
        if encodes(o):
            vocabulary = ops.featurizer_of(o.op, args).get("classes")
    return value


def apply(engine, name: str, value: Any, args: dict, dims: list[str], eos: list[int] | None = None,
          featurizer: dict | None = None, features: Tensor | None = None) -> Any:
    """Op `name` on a value, or in `featurizer`'s space on its coordinates there (the `features` it keeps),
    the change written back; on a ragged value, on its entries, which stay ragged while the op keeps them."""
    if isinstance(value, dict) and "tok" in value:
        got = apply(engine, name, value["values"], args, dims, eos, featurizer, features)
        kind = ops.kind(name, args)
        return {**value, "values": got} if featurizer is not None or "entries" in ops.out_dims(name, dims, args, kind) \
            else got

    def fn(x: Any) -> Any:
        return ops.call(name, x, args, engine=engine, dims=dims, eos=eos)
    return fn(value) if featurizer is None else featurizers.within(featurizer, cast(Tensor, value), fn, features)
