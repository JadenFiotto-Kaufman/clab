"""A step's input in batches, and an entry bound to one batch: its site in the padded batch, the ops it writes
with, its value with the ops after its last edit applied."""

from typing import Any

import torch
from torch import Tensor

from ... import address
from ...plan.model import Plan, arguments, brackets, entries, gathers, writes
from .. import data, ops, positions
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


def edit(engine, plan: Plan, record, state: dict, at: dict, rows: slice, live: dict | None = None,
         answers: dict | None = None) -> list[dict]:
    """An entry's ops up to its last edit, bound to one batch, as `ops.visit` runs them: each with the layout of
    the value it meets (`dims`), the dataset rows of the batch, and its arguments bound, an operand that names an
    earlier entry of the same forward (`live`: by name, the engine's name for it) to that name, for the forward
    to fill. An `encode` carries how many ops lie between it and the `decode` that closes it (`inner`) and the
    features its `select` keeps."""
    dims = address.dims(record.site.component, None, False)
    closes, outer, out = brackets(record.ops), [], []
    for i, o in enumerate(record.ops[:len(record.ops) - len(gathers(record))]):
        args = bound(engine, plan, o.op, arguments(o), state, at, rows, live=live, answers=answers)
        out.append({"op": o.op, "args": args, "rows": torch.arange(rows.start, rows.stop), "dims": dims})
        if o.op == "encode":
            f, select = args["featurizer"], arguments(o).get("select")
            out[-1]["inner"] = closes[i]
            if select is not None:
                out[-1]["features"] = torch.tensor(positions.select(select["features"], f.size()))
            outer.append(dims)
            dims = dims[:-1] + [f.axis]
        elif o.op == "decode" and args.get("featurizer") is not None:
            dims = outer.pop()
    return out


def batches(engine, step, found: dict, batch_size: int):
    """The step's input in consecutive batches: each one's rows (a slice), its ids padded on the left to the
    input's longest row, and each row's padding."""
    encoded = found["inputs"][data.input_name(step.input)]
    n = len(encoded["rows"])
    for lo in range(0, n, batch_size):
        chunk = slice(lo, min(lo + batch_size, n))
        ids = [e["ids"] for e in encoded["rows"][chunk]]
        yield chunk, data.pad(engine.tokenizer, ids, encoded["width"]), [encoded["width"] - len(x) for x in ids]


def bind(engine, plan: Plan, name: str, step, state: dict, found: dict, rows: slice,
         pads: list[int]) -> tuple[dict, list[dict]]:
    """A forward's entries bound to one batch: each one's site, and the records the engine does in their order:
    an entry that writes as a write with its ops, then, when it is gathered, its reads, one per layer
    (`<entry>@<layer>`). An entry that does neither is left out."""
    chosen = {k: r for k, r in entries(step).items() if writes(r) or r.gathered}
    sites = {key: site(record, lookup(found["selects"], f"{name}.{key}", step.at),
                       lookup(found["positions"], f"{name}.{key}", step.at)[rows], pads, head(step, record))
             for key, record in chosen.items()}
    live = {f"{name}.{key}": f"{key}@{layers(engine, record)[0]}" for key, record in chosen.items() if record.gathered}
    records = []
    for key, record in chosen.items():
        if writes(record):
            records.append({**sites[key], "name": key, "layer": layers(engine, record)[0],
                            "ops": edit(engine, plan, record, state, step.at, rows, live, found["answers"])})
        if record.gathered:
            records += [{**sites[key], "name": f"{key}@{layer}", "layer": layer} for layer in layers(engine, record)]
    return sites, records


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
    """Each gathered entry's per-batch values joined along rows (a ragged one's entries, after the layer
    axis)."""
    for key, record in entries(step).items():
        if not record.gathered:
            continue
        axis = 1 if address.stacks(record.site.layers) else 0
        values = [p[key] for p in parts]
        put(state, f"{name}.{key}", step.at, torch.cat(values, axis) if isinstance(values[0], Tensor) else
            {k: torch.cat([v[k] for v in values], axis if k == "values" else 0) for k in values[0]})


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, pads: list[int], state: dict,
         at: dict, answers: dict) -> Any:
    """One layer of an entry's value taken off the model in the prefill, in one batch: its entries when it is
    ragged, then the ops after its last edit (`answers` as preflight resolved them)."""
    if positions.width(positions.parse(record.pos)) is None:
        keep = site["keep"]
        value = {"values": value[keep.to(value.device)],
                 "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep], "tok": site["tok"][keep]}
    return with_ops(engine, plan, record, value, state, at, rows, site["tok"], answers)


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice,
             places: Tensor | None, answers: dict) -> Any:
    """An entry's value taken off the model with the ops after its last edit applied in turn. Past a probe's
    `encode` a column names its classes. An op is given each row's dataset row and each position's token
    position (`places`, `[rows, width]`; a ragged value's own `row` and `tok`)."""
    dims = address.dims(record.site.component, None, positions.width(positions.parse(record.pos)) is None)
    vocabulary = None
    for o in gathers(record):
        args = bound(engine, plan, o.op, arguments(o), state, at, rows, value, vocabulary, dims=dims, answers=answers)
        value = apply(engine, o.op, value, args, dims, rows=torch.arange(rows.start, rows.stop), places=places)
        dims = ops.out_dims(o.op, dims, arguments(o), ops.kind(o.op, args))
        if o.op == "encode":
            vocabulary = getattr(args["featurizer"], "classes", None)
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
          rows: Tensor | None = None, places: Tensor | None = None) -> Any:
    """Op `name` on a value; on a ragged value, on its entries, which stay ragged while the op keeps them.
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
        got = apply(engine, name, value["values"], args, dims, eos, value["row"], value["tok"])
        kind = ops.kind(name, args)
        return {**value, "values": got} if "entries" in ops.out_dims(name, dims, args, kind) else got
    return ops.call(name, value, args, engine=engine, dims=dims, eos=eos, rows=rows, positions=places)
