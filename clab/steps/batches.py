"""A step's input in batches, and an entry bound to one batch: its site in the padded batch, the ops it writes
with, its value with the ops after its last edit applied."""

from typing import Any

import torch
from torch import Tensor

from clab import ops
from clab.compile import positions
from clab.compile.answers import answered, column_texts, indexed, refkey
from clab.compile.positions import head
from clab.compile.state import Found
from clab.format import address, data, selectors
from clab.format.address import pos_axis
from clab.format.selectors import made_at
from clab.format.sweeps import lookup, put
from clab.plan import Plan, arguments, bracketed, dims_of, entries, gathers, writes


def site(record, picks: dict, rows: list[list[int]], pads: list[int], width: int, last_only: bool) -> dict:
    """What the engine needs to reach a record's selection in this batch, `width` tokens wide: its component,
    its token indices into the padded batch (`positions`, `[rows, n]`; a row with fewer positions than the
    batch's widest repeats its last index, `keep` marking its own, `tok` the unpadded indices, -1 past its
    own), the part of the feature axes it keeps
    (`select`: `picks`, the indices preflight resolved its `select` to), and on a `key` axis each row's own keys
    (`keys`). Where the value holds the last position alone (`last_only`: a generation's head in the prefill)
    every index is -1."""
    n = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (n - len(row))
             for row, pad in zip(rows, pads)]
    tok = [row + [-1] * (n - len(row)) for row in rows]
    index = [[-1] * n for _ in rows] if last_only else index
    return {"component": record.site.component, "positions": torch.tensor(index),
            "select": selection(record, picks), **keys(record, pads, width),
            "keep": torch.tensor([[j < len(row) for j in range(n)] for row in rows]), "tok": torch.tensor(tok)}


def decoding(record, picks: dict, prompts: list[int]) -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last; and, in
    `tok`, the position before decode call 1's input in each row of `prompts` tokens, which with the call
    number is the input's."""
    return {"component": record.site.component, "positions": torch.full((len(prompts), 1), -1),
            "select": selection(record, picks), "tok": torch.tensor([[p - 1] for p in prompts])}


def keys(record, pads: list[int], width: int) -> dict:
    """A `key` axis as each row's own keys, its positions (`positions.prompted`), which a key selector counts:
    `dim` is the axis's dimension in the selection's layout, `index` (`[rows, width]`) each row's keys in the
    padded batch, in order, then its padding's, and `own` which of those are the row's (`owned`). Nothing
    on a component with no key axis."""
    layout = address.dims(record.site.component)
    if "key" not in layout:
        return {}
    index = [[p + pad for p in positions.prompted(width - pad)] + list(range(pad)) for pad in pads]
    own = [[j < width - pad for j in range(width)] for pad in pads]
    return {"keys": {"dim": layout.index("key"), "index": torch.tensor(index), "own": torch.tensor(own)}}


def selection(record, picks: dict) -> dict:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps
    (`picks`, by axis): on `key`, among the row's own keys, which `keys` puts first, in order."""
    layout = address.dims(record.site.component)
    return {layout.index(axis): torch.tensor(v) for axis, v in picks.items()}


def edit(plan: Plan, record, state: dict, at: dict, rows: slice, answers: dict, live: dict | None = None) -> list[dict]:
    """An entry's ops up to its last edit, bound to one batch, as `visit` runs them: each with the layout of
    the value it meets (`dims`), the dataset rows of the batch, and its arguments bound, an operand that names an
    earlier entry of the same forward (`live`: by name, the engine's name for it) to that name, for the forward
    to fill. An `encode` holds the ops its `decode` closes over, bound the same way (`inner`), and the features
    its `select` keeps."""
    last = len(record.ops) - len(gathers(record))

    def bind(nodes: list[dict], dims: list[str]) -> list[dict]:
        out = []
        for node in nodes:
            o = node["op"]
            args = bound(plan, o.op, arguments(o), state, at, rows, answers, live=live)
            out.append({"op": o.op, "args": args, "rows": torch.arange(rows.start, rows.stop), "dims": dims})
            if o.op == "encode":
                f, select = args["featurizer"], arguments(o).get("select")
                if select is not None:
                    out[-1]["features"] = torch.tensor(selectors.select(select["features"], f.size()))
                out[-1]["inner"] = bind(node["inner"], dims[:-1] + [f.axis])
        return out
    return bind([n for n in bracketed(record.ops) if n["at"] < last], address.dims(record.site.component))


def chunked(engine, step, found: Found, batch_size: int):
    """The step's input in consecutive batches: each one's rows (a slice), its ids padded on the left to the
    input's longest row, and each row's padding."""
    encoded = found.inputs[data.input_name(step.input)]
    n = len(encoded["rows"])
    for lo in range(0, n, batch_size):
        chunk = slice(lo, min(lo + batch_size, n))
        ids = [e["ids"] for e in encoded["rows"][chunk]]
        yield chunk, data.pad(engine.tokenizer, ids, encoded["width"]), [encoded["width"] - len(x) for x in ids]


def read_name(key: str, layer, prefill: bool = False) -> str:
    """The engine's name for a read of entry `key` made at `layer` (None for a model's own value): `<key>@<layer>`,
    with `#0` after it for the prefill part of an entry that also reads at decode calls."""
    return f"{key}@{layer}" + ("#0" if prefill else "")


def write_name(key: str, prefill: bool = False) -> str:
    """The engine's name for a write of entry `key`: the key, with `#0` after it for the prefill part of a write
    that also lands at decode calls."""
    return key + ("#0" if prefill else "")


def bind(engine, plan: Plan, name: str, step, state: dict, found: Found, rows: slice, pads: list[int],
         among: set | None = None) -> tuple[dict, list[dict]]:
    """The step's prefill entries (those `among`, default every one) bound to one batch: each one's site, and
    the records the engine does in their order: an entry that writes as a write with its ops, then, when it is
    gathered, its reads, one per layer. An entry that does neither is left out."""
    chosen = {k: r for k, r in entries(step).items() if (writes(r) or r.gathered) and (among is None or k in among)}
    width = found.inputs[data.input_name(step.input)]["width"]
    sites = {key: site(record, lookup(found.selects, f"{name}.{key}", step.at),
                       lookup(found.positions, f"{name}.{key}", step.at)[rows], pads, width, head(step, record))
             for key, record in chosen.items()}
    live = {f"{name}.{key}": read_name(key, layers(engine, record)[0]) for key, record in chosen.items()
            if record.gathered}
    records = []
    for key, record in chosen.items():
        if writes(record):
            records.append({**sites[key], "name": write_name(key), "layer": layers(engine, record)[0],
                            "ops": edit(plan, record, state, step.at, rows, found.answers, live)})
        if record.gathered:
            records += [{**sites[key], "name": read_name(key, layer), "layer": layer} for layer in layers(engine, record)]
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
    return {k: torch.stack([v[k].to(values[0][k].device) for v in values]) for k in values[0]}


def joined(state: dict, name: str, step, parts: list[dict]) -> None:
    """Each gathered entry's per-batch values joined along rows, on the first batch's device, as `stacked`
    joins layers (a batch whose rows all stopped before the calls a read names made its blank on the CPU, and
    under a device map batches can come back apart); a batch narrower than the widest along `pos` is padded
    to it first (`padded`)."""
    for key, record in entries(step).items():
        if not record.gathered:
            continue
        dims = record.dims or []
        values = [p[key] for p in parts]

        def cat(parts: list[Tensor]) -> Tensor:
            if "pos" in dims:
                parts = [padded(v, dims.index("pos"), max(v.shape[dims.index("pos")] for v in parts)) for v in parts]
            return torch.cat([v.to(parts[0].device) for v in parts], dims.index("rows"))
        put(state, f"{name}.{key}", step.at, cat(values) if isinstance(values[0], Tensor) else
            {k: cat([v[k] for v in values]) for k in values[0]})


def padded(value: Tensor, dim: int, n: int) -> Tensor:
    """`value` padded along `dim` to `n` with what marks a missing position: NaN, or -1 in ids."""
    shape = list(value.shape)
    shape[dim] = n - shape[dim]
    missing = float("nan") if value.is_floating_point() else -1
    return value if shape[dim] == 0 else torch.cat([value, torch.full(shape, missing, dtype=value.dtype,
                                                                       device=value.device)], dim)


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, state: dict, at: dict,
         answers: dict) -> Any:
    """One layer of an entry's value taken off the model in the prefill, in one batch, NaN past a row's own
    positions (`keep`), then the ops after its last edit (`answers` as preflight resolved them)."""
    keep = site["keep"].to(value.device)
    if not keep.all():
        value = value.masked_fill(~keep.reshape(*keep.shape, *[1] * (value.dim() - 2)), float("nan"))
    return with_ops(engine, plan, record, value, state, at, rows, site["tok"], answers)


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice,
             places: Tensor | None, answers: dict) -> Any:
    """An entry's value taken off the model with the ops after its last edit applied in turn. Past a probe's
    `encode` a column names its classes. An op is given each row's dataset row and each position's token
    position (`places`, `[rows, width]`, -1 past a row's own)."""
    dims = address.dims(record.site.component)
    vocabulary = None
    for o in gathers(record):
        args = bound(plan, o.op, arguments(o), state, at, rows, answers, value, vocabulary, dims=dims)
        value = ops.call(o.op, value, args, engine=engine, dims=dims, rows=torch.arange(rows.start, rows.stop),
                         positions=places)
        dims = ops.out_dims(o.op, dims, arguments(o), getattr(args.get("featurizer"), "kind", None))
        if o.op == "encode":
            vocabulary = getattr(args["featurizer"], "classes", None)
    return value


def bound(plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None, answers: dict,
          of: Any = None, vocabulary: list | None = None, live: dict | None = None,
          dims: list[str] | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up (a featurizer to its instance; an entry of the same
    forward, in `live`, to `{"live": <the engine's name for its read>}`), columns resolved against the value's own
    vocabulary (the tokenizer's, as preflight resolved them into `answers` and `answered` lays them against
    the value `of` laid out `dims`; or a probe's class list) or read as text, sliced to `rows`."""
    params, out = ops.parameters(name), {}
    for key, value in args.items():
        annotation = params[key].annotation
        if annotation == ops.Value and isinstance(value, str) and live and value in live:
            out[key] = {"live": live[value]}
            continue
        if annotation == ops.Value and isinstance(value, str):
            given, value = value, lookup(state, value, at)
            value = value[rows] if rows is not None and dims_of(plan, given)[:1] == ["rows"] else value
        elif annotation == ops.Tokens:
            if vocabulary is None:
                out[key] = answered(answers[refkey(value)], of, dims or [], rows)
                continue
            value = indexed(plan, value, vocabulary, rows)[:, None]
        elif annotation == ops.Text:
            column, strip, lower = data.column_of(value) or ("", False, False)
            out[key] = [(v if isinstance(v, list) else [v], strip, lower) for v in column_texts(plan, column, rows)]
            continue
        elif annotation == ops.Featurizer:
            value = lookup(state, value, at)
        out[key] = value
    return out


def taken(x: Tensor, record: dict) -> Tensor:
    """The record's positions and narrowing of `x`, a site's value with its token axis at 1.

    `record["positions"]` is `[rows, width]` token indices; a `key` axis is each row's own keys (`owned`);
    `record["select"]` maps a dimension of that layout to the indices kept along it, the same for every row. The
    indices, built where the plan runs, move to the value's device.
    """
    x = owned(x[torch.arange(x.shape[0], device=x.device)[:, None], record["positions"].to(x.device)], record)
    for dim, index in record["select"].items():
        x = take(x, dim, index)
    return x


def owned(x: Tensor, record: dict) -> Tensor:
    """A selection's `key` axis as each row's own keys, its positions in order, NaN past them, where the record has
    one (`record["keys"]`: each row's keys in the padded batch, then its padding's, and which are its own); else
    `x` as it is."""
    if "keys" not in record:
        return x
    dim, keys = record["keys"]["dim"], record["keys"]
    x = x.gather(dim, per_row(x, dim, keys["index"].to(x.device)))
    return x.masked_fill(~per_row(x, dim, keys["own"].to(x.device)), float("nan"))


def unowned(current: Tensor, new: Tensor, record: dict) -> Tensor:
    """`new`, a selection in each row's own key order (`owned`), back in the batch's order over `current`: the
    row's own keys from `new`, its padding's as `current` has them."""
    dim, keys = record["keys"]["dim"], record["keys"]
    index, own = (per_row(current, dim, keys[k].to(current.device)) for k in ("index", "own"))
    return current.scatter(dim, index, torch.where(own, new, current.gather(dim, index)))


def take(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` at `index` along `dim`, the same indices on every row."""
    return x.index_select(dim, index.to(x.device))


def per_row(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Each row's `[rows, n]` indices expanded to `x`'s shape with `n` along `dim`, for gather and scatter."""
    shape = [1] * x.dim()
    shape[0], shape[dim] = index.shape
    size = list(x.shape)
    size[dim] = index.shape[1]
    return index.reshape(shape).expand(size)


def visit(value: Tensor, component: str, records: list[dict], got: dict) -> tuple[Tensor, bool]:
    """`value` with a site's records done in listed order, and whether any wrote: a write's ops run on its
    selection of the current value, out of place, and the result lands there; a read takes its selection of
    the current value into `got` under its name (at a decode call, appended: one per call it reads).

    `original` is a write's selection as it was before any of these writes. An argument bound to an entry of
    the same forward (`{"live": <its read's name>}`) is that read's value in `got`, where this forward put it.
    An op is given the dataset row of each of the selection's rows (the op's `rows`) and the token position of
    each of its positions (the write's `tok`, plus the decode `call` it lands at, which the engine stamps). A
    write narrows at most one dimension of its selection (compile refuses more). A ragged write's rows are
    padded to one width; only the positions its `keep` marks are written. A `key` axis is each row's own keys in
    order, NaN past them (`owned`), to a read and to a write's ops alike, and a write writes those alone
    (`unowned`). Every index, mask and result moves to the value's device, and a result to its dtype, where it
    meets the value.
    """
    axis = pos_axis(component)
    before = x = value.movedim(axis, 1)
    rows = torch.arange(x.shape[0], device=x.device)[:, None]
    wrote = False
    for w in records:
        if "ops" not in w:
            if "calls" in w:
                got[w["name"]].append(taken(x, w))
            else:
                got[w["name"]] = taken(x, w)
            continue
        if not wrote:
            x, wrote = before.clone(), True
        pos = w["positions"].to(x.device)
        current = x[rows, pos]
        new, original = owned(current, w), owned(before[rows, pos], w)
        whole = new
        for dim, index in w["select"].items():
            new, original = take(new, dim, index), take(original, dim, index)
        at = w["tok"] + w.get("call", 0) if "tok" in w else None        # each position, at a decode call its own
        new = edited(new, original, w["ops"], got, at)
        new = new.to(device=current.device, dtype=current.dtype)
        for dim, index in w["select"].items():
            new = whole.index_copy(dim, index.to(current.device), new)
        new = unowned(current, new, w) if "keys" in w else new
        keep = w["keep"].to(x.device) if "keep" in w else torch.ones(pos.shape, dtype=torch.bool, device=x.device)
        x[rows.expand_as(pos)[keep], pos[keep]] = new.to(device=x.device, dtype=x.dtype)[keep]
    return x.movedim(1, axis), wrote


def edited(new: Tensor, original: Tensor, bound: list[dict], got: dict, places: Tensor | None) -> Tensor:
    """`new` after a write's bound ops, each on the current value in turn. An `encode` runs the ops its `decode`
    closes over (`inner`) on the value's coordinates, on the features its `select` keeps, and writes the change
    back (`Kind.within`); `original` goes in with them, encoded the same way."""
    for o in bound:
        args = {k: live(got, v) if isinstance(v, dict) and "live" in v else v for k, v in o["args"].items()}
        if o["op"] != "encode":
            new = ops.call(o["op"], new, args, original=original, dims=o["dims"], rows=o.get("rows"), positions=places)
            continue
        f: Any = args["featurizer"]
        features = o.get("features")
        start = f.encode(original)
        start = start if features is None else start.index_select(-1, features.to(start.device))
        new = f.within(new, lambda z, start=start, inner=o["inner"]: edited(z, start, inner, got, places), features)
    return new


def live(got: dict, operand: dict) -> Tensor:
    """A write's operand that names a read of its own forward: that read's value, which the engine made before
    it reached the write."""
    if operand["live"] not in got:
        raise ValueError(f"operand {operand['live']!r} is a read of this forward the engine had not made when it "
                         "reached the write; it reads it at a later site")
    return got[operand["live"]]


def home(engine) -> str:
    """Where the model is, which is where the run makes featurizers and an engine sends a batch's inputs: where
    its first parameter is (on an NDIF server, or wherever the engine's `device` put it: the first of several
    under a device map that spreads it), else, while the model is a meta shell that loads its weights at its
    first forward, the engine's device."""
    device = str(next(engine.model.parameters()).device)
    return (engine.device or "cpu") if device == "meta" else device


def now(records: list, call: int) -> list:
    """What a generation does at a site on forward call `call` (0 is the prefill), in order: a record with no
    `calls` acts on the prefill only, one with `calls` on those decode calls (None: every one)."""
    def wanted(record: dict) -> bool:
        if "calls" not in record:
            return call == 0
        return call > 0 and (record["calls"] is None or call in record["calls"])
    return [r for r in records if wanted(r)]


def landed(records: list[dict], seen: list, tokenizer) -> list[dict]:
    """The records at a decode call, each write stamped with the call (`call`, the number of inputs `seen` so
    far), a write with an `online` scope narrowed to the rows it lands on there (`keep`), as
    `positions.landing` decides from those inputs (one `[rows, 1]` per decode call); one that lands on no row
    is left out. A read is as it was."""
    out = []
    for w in records:
        if "ops" not in w:
            out.append(w)
            continue
        online, w = w.get("online"), {**w, "call": len(seen)}
        if online is None:
            out.append(w)
            continue
        rows = [positions.landing(online["pos"], texts, [int(t[r, 0]) for t in seen], set(online["eos"]), tokenizer)
                for r, texts in enumerate(online["texts"])]
        if any(rows):
            out.append({**w, "keep": torch.tensor(rows)[:, None]})
    return out
