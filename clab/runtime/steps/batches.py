"""A step's input in batches, and a record bound to one batch: its site in the padded batch, its writes'
ops, its reads' values with their ops applied."""

from typing import Any, cast

import torch
from torch import Tensor

from ... import address
from ...plan.model import Plan, arguments, encodes, space
from .. import data, featurizers, ops, positions
from .preflight import head, made_at
from .state import bound, lookup, put, within_row


def site(record, picks: dict, rows: list[list[int]], pads: list[int], last_only: bool = False) -> dict:
    """What the engine needs to reach a record's selection in this batch: its component, its token indices
    into the padded batch (`positions`, `[rows, width]`; a ragged row repeats its last index to the batch's
    widest, `keep` marking its own, `tok` the unpadded indices), and the part of the feature axes it keeps
    (`select`: `picks`, the indices preflight resolved its `select` to). Where the value
    holds the last position alone (`last_only`: a generation's head in the prefill) every index is -1."""
    width = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (width - len(row))
             for row, pad in zip(rows, pads)]
    tok = [[i - pad for i in row] for row, pad in zip(index, pads)]
    index = [[-1] * width for _ in rows] if last_only else index
    return {"component": record.site.component, "positions": torch.tensor(index),
            "select": selection(record, picks, pads),
            "keep": torch.tensor([[j < len(row) for j in range(width)] for row in rows]), "tok": torch.tensor(tok)}


def decoding(record, picks: dict, prompts: list[int], pads: list[int]) -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last; and, in
    `tok`, the position before decode call 1's input in each row of `prompts` tokens, which with the call
    number is the input's."""
    return {"component": record.site.component, "positions": torch.full((len(prompts), 1), -1),
            "select": selection(record, picks, pads), "tok": torch.tensor([[p - 1] for p in prompts])}


def selection(record, picks: dict, pads: list[int]) -> dict:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps
    (`picks`, by axis). On the `key` axis an index counts the row's own tokens, so each row's is shifted by its
    padding (`[rows, n]`), as a position is."""
    layout = address.dims(record.site.component, None, False)
    return {layout.index(axis): torch.tensor([[i + pad for i in v] for pad in pads] if axis == "key" else v)
            for axis, v in picks.items()}


def edit(engine, plan: Plan, record, o, state: dict, at: dict, rows: slice, live: dict | None = None) -> dict:
    """A write's op bound to one batch, as `ops.write` runs it: in a featurizer's space, with its instance and
    the features its `select` keeps; an operand that names a read of the same forward (`live`: by name, the
    engine's name for it) is bound to that name, for the forward to fill."""
    dims = address.dims(record.site.component, None, False)
    within = space(o)
    f = lookup(state, within, at) if within is not None else None
    return {"op": o.op, "args": bound(engine, plan, o.op, arguments(o), state, at, rows, live=live),
            "rows": torch.arange(rows.start, rows.stop), "featurizer": f,
            "features": features(o, f), "dims": dims[:-1] + ["features"] if within is not None else dims}


def features(o, f) -> Tensor | None:
    """The features an op in featurizer `f`'s space keeps, if it narrows them: its `select` among `f`'s."""
    return torch.tensor(positions.select(o.select["features"], f.size())) if o.select else None


def batches(engine, step, found: dict, batch_size: int):
    """The step's input in consecutive batches: each one's rows (a slice), its ids padded on the left to the
    input's longest row, and each row's padding."""
    encoded = found["inputs"][data.input_name(step.input)]
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
    sites = {key: site(record, lookup(found["selects"], f"{name}.{key}", step.at),
                       lookup(found["positions"], f"{name}.{key}", step.at)[rows], pads, head(step, record))
             for key, record in {**writes_here, **reads}.items()}
    live = {f"{name}.{key}": f"{key}@{layers(engine, record)[0]}" for key, record in reads.items()}
    writes = [{**sites[key], "name": key, "layer": layers(engine, record)[0],
               "ops": [edit(engine, plan, record, o, state, step.at, rows, live) for o in record.ops]}
              for key, record in writes_here.items()]
    return sites, writes, [{**sites[key], "name": f"{key}@{layer}", "layer": layer}
                           for key, record in reads.items() for layer in layers(engine, record)]


def layers(engine, record) -> list:
    """The layers a record is made at: those its site's `layers` select among the model's (an int's one)."""
    return made_at(record.site, engine.sizes()["num_layers"])


def stacked(record, values: list) -> Any:
    """A read's per-layer values as its value: the one, or under `layers` that stack, stacked layer-first."""
    if not address.stacks(record.site.layers):
        return values[0]
    # Under a device that spreads the model ("auto") the layers' values can live on different GPUs; they stack on
    # the first's. (Not tested: this host has one GPU.)
    if isinstance(values[0], Tensor):
        return torch.stack([v.to(values[0].device) for v in values])
    return {**values[0], "values": torch.stack([v["values"].to(values[0]["values"].device) for v in values])}


def joined(state: dict, name: str, step, parts: list[dict]) -> None:
    """Each read's per-batch values joined along rows (a ragged read's entries, after the layer axis)."""
    for key, record in step.reads.items():
        axis = 1 if address.stacks(record.site.layers) else 0
        values = [p[key] for p in parts]
        put(state, f"{name}.{key}", step.at, torch.cat(values, axis) if isinstance(values[0], Tensor) else
            {k: torch.cat([v[k] for v in values], axis if k == "values" else 0) for k in values[0]})


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, pads: list[int], state: dict,
         at: dict) -> Any:
    """One layer of a prefill read in one batch: its entries when it is ragged, then its ops."""
    if positions.width(positions.parse(record.pos)) is None:
        keep = site["keep"]
        value = {"values": value[keep.to(value.device)],
                 "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep], "tok": site["tok"][keep]}
    return with_ops(engine, plan, record, value, state, at, rows, site["tok"])


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice,
             places: Tensor | None) -> Any:
    """A read's value with its ops applied in listed order: one in a featurizer's space on the value's
    coordinates there, the change written back. Past a probe's `encode` a column names its classes. An op is
    given each row's dataset row and each position's token position (`places`, `[rows, width]`; a ragged
    value's own `row` and `tok`)."""
    dims = address.dims(record.site.component, None, positions.width(positions.parse(record.pos)) is None)
    vocabulary = None
    for o in record.ops:
        within = space(o)
        f = lookup(state, within, at) if within is not None else None
        args = bound(engine, plan, o.op, arguments(o), state, at, rows, value, vocabulary, dims=dims)
        value = apply(engine, o.op, value, args, dims[:-1] + ["features"] if f else dims, featurizer=f,
                      features=features(o, f), rows=torch.arange(rows.start, rows.stop), places=places)
        if f is None:
            dims = ops.out_dims(o.op, dims, arguments(o), ops.kind(o.op, args))
        if encodes(o):
            vocabulary = getattr(ops.featurizer_of(o.op, args), "classes", None)
    return value


def dense(value: dict, dims: list[str], rows: Tensor) -> tuple[Tensor, list[str]]:
    """A ragged value over the dataset rows `rows` as a rectangular one, `entries` become `rows, pos`: each
    row's entries in order, NaN past its last (a row with none is missing all the way)."""
    axis = dims.index("entries")
    values = value["values"].movedim(axis, 0)
    row = value["row"] - rows[0]
    j = within_row(value["row"])
    width = int(j.max()) + 1 if len(j) else 1
    out = torch.full((len(rows), width, *values.shape[1:]), float("nan"), dtype=values.dtype, device=values.device)
    out[row.to(values.device), j.to(values.device)] = values
    return out.movedim((0, 1), (axis, axis + 1)), dims[:axis] + ["rows", "pos"] + dims[axis + 1:]


def apply(engine, name: str, value: Any, args: dict, dims: list[str], eos: list[int] | None = None,
          featurizer: featurizers.Kind | None = None, features: Tensor | None = None, rows: Tensor | None = None,
          places: Tensor | None = None) -> Any:
    """Op `name` on a value, or in `featurizer`'s space on its coordinates there (the `features` it keeps),
    the change written back; on a ragged value, on its entries, which stay ragged while the op keeps them.
    `rows` and `places` say which dataset row and token position each of the value's rows and positions
    is, where the run knows them; a ragged value's are its `row` and `tok`."""
    over = args.get("over") if ops.DECLARED[name]["group"] == "reductions" else None
    if isinstance(value, dict) and "tok" in value and over is not None and "pos" in over:
        if "rows" not in over:          # each row's entries are its positions: one value per row
            assert rows is not None
            value, dims = dense(value, dims, rows)
            return apply(engine, name, value, args, dims, eos, rows=rows)
        args = {**args, "over": [a for a in over if a not in ("rows", "pos")] + ["entries"]}
    if isinstance(value, dict) and "tok" in value:
        got = apply(engine, name, value["values"], args, dims, eos, featurizer, features, value["row"], value["tok"])
        kind = ops.kind(name, args)
        return {**value, "values": got} if featurizer is not None or "entries" in ops.out_dims(name, dims, args, kind) \
            else got

    def fn(x: Any) -> Any:
        return ops.call(name, x, args, engine=engine, dims=dims, eos=eos, rows=rows, positions=places)
    return fn(value) if featurizer is None else featurizer.within(cast(Tensor, value), fn, features)
