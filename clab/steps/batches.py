"""A step's input in batches, and a forward's schedule bound to one batch: each record's site in the padded batch,
the ops a write runs there, and a read's value with the ops after its last edit applied."""

from __future__ import annotations

from collections import ChainMap
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
from torch import Tensor

from clab import ops
from clab.compile import positions
from clab.compile.answers import answered, column_texts, indexed, refkey
from clab.compile.positions import head, texts
from clab.compile.state import Found
from clab.format import address, data, selectors
from clab.format.address import pos_axis
from clab.format.sweeps import lookup, put
from clab.plan import Forward, Generate, Intervention, Plan, Site, arguments, bracketed, entries, gathers, held_rows, record_of

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def site(record: Intervention, picks: dict, rows: list, pads: list[int], width: int, last_only: bool,
         device: str = "cpu") -> dict:
    """What the engine needs to reach a record's selection in this batch, `width` tokens wide: its component,
    its token indices into the padded batch (`positions`, `[rows, n]`; a row with fewer positions than the
    batch's widest repeats its last index, `keep` marking its own, `tok` the unpadded indices, -1 past its
    own; a row the record does not reach, the marker in `rows`, holds none), the part of the feature axes it keeps
    (`select`: `picks`, the indices preflight resolved its `select` to), and on a `key` axis each row's own keys
    (`keys`). Where the value holds the last position alone (`last_only`: a generation's head in the prefill)
    every index is -1. Where every row holds `n` positions there is no `keep`, so that a read or a write of
    equal-width rows indexes by no mask, which the device would have to count before the host could go on. The
    indices and masks the value meets are made on `device`, where the model is (`placed`); `tok` stays on the host,
    where the ops that read it (`noise`) count with it."""
    rows = [row if isinstance(row, list) else [] for row in rows]
    n = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (n - len(row))
             for row, pad in zip(rows, pads)]
    tok = [row + [-1] * (n - len(row)) for row in rows]
    index = [[-1] * n for _ in rows] if last_only else index
    ragged = any(len(row) < n for row in rows)
    return {"component": cast(Site, record.site).component, "positions": placed(index, device),
            "select": selection(record, picks, device), **keys(record, pads, width, device),
            **({"keep": placed([[j < len(row) for j in range(n)] for row in rows], device)} if ragged else {}),
            "tok": torch.tensor(tok)}


def decoding(record: Intervention, picks: dict, prompts: list[int], device: str = "cpu") -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last, on `device`;
    and, in `tok`, the position before decode call 1's input in each row of `prompts` tokens, which with the call
    number is the input's."""
    return {"component": cast(Site, record.site).component,
            "positions": torch.full((len(prompts), 1), -1, device=device),
            "select": selection(record, picks, device), "tok": torch.tensor([[p - 1] for p in prompts])}


def keys(record: Intervention, pads: list[int], width: int, device: str = "cpu") -> dict:
    """A `key` axis as each row's own keys, its positions (`positions.prompted`), which a key selector counts:
    `dim` is the axis's dimension in the selection's layout, `index` (`[rows, width]`) each row's keys in the
    padded batch, in order, then its padding's, and `own` which of those are the row's (`owned`), both on
    `device`. Nothing on a component with no key axis."""
    layout = address.dims(cast(Site, record.site).component)
    if "key" not in layout:
        return {}
    index = [[p + pad for p in positions.prompted(width - pad)] + list(range(pad)) for pad in pads]
    own = [[j < width - pad for j in range(width)] for pad in pads]
    return {"keys": {"dim": layout.index("key"), "index": placed(index, device), "own": placed(own, device)}}


def selection(record: Intervention, picks: dict, device: str = "cpu") -> dict[int, Tensor]:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps
    (`picks`, by axis), on `device`: on `key`, among the row's own keys, which `keys` puts first, in order."""
    layout = address.dims(cast(Site, record.site).component)
    return {layout.index(axis): placed(v, device) for axis, v in picks.items()}


def placed(values: Any, device: str | torch.device) -> Tensor:
    """`values` (nested lists, or a tensor on the host) as a tensor on `device`: on a GPU copied from pinned
    memory without the host waiting for the copy (`non_blocking`), which runs in stream order before any kernel
    that reads it, so that binding a batch never drains the GPU's queue; anywhere else moved as torch moves it."""
    host = torch.as_tensor(values)
    return host.pin_memory().to(device, non_blocking=True) if torch.device(device).type == "cuda" else host.to(device)


def edit(plan: Plan, record: Intervention, state: ChainMap, at: dict, ids: list[int], found: Found,
         layer: int | None, resolved: dict[str, dict] | None = None, device: str = "cpu") -> list[dict]:
    """An entry's ops up to its last edit at one of the layers it is made at (`layer`), bound to one batch, as
    `visit` runs them: each with the layout of the value it meets there (`dims`: the op before it left it, as its
    record says; the site's selection before the first), the batch's dataset row ids (`ids`), and its arguments
    bound at that layer (`bound`), each named operand as the schedule `resolved` it (by name: a producer's point, or
    the name of a read of the same forward, for the forward to fill). An `encode` holds the ops its `decode` closes
    over, bound the same way (`inner`), and the features its `select` keeps, on `device`, where the model and its
    featurizers are."""
    last = len(record.ops) - len(gathers(record))
    laid = [address.dims(cast(Site, record.site).component), *(o.dims or [] for o in record.ops)]

    def bind(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            o, dims = node["op"], laid[node["at"]]
            args = bound(plan, o.op, arguments(o), state, at, ids, found, resolved=resolved, dims=dims, layers=layer)
            out.append({"op": o.op, "args": args, "rows": torch.tensor(ids), "dims": dims})
            if o.op == "encode":
                f, select = args["featurizer"], arguments(o).get("select")
                if select is not None:
                    out[-1]["features"] = placed(selectors.select(select["features"], f.size()), device)
                out[-1]["inner"] = bind(node["inner"])
        return out
    edits = bind([n for n in bracketed(record.ops) if n["at"] < last])
    del bind            # it refers to itself, a cycle that would hold `state` until the collector ran
    return edits


# The key under which a scope of the run's state (a `ChainMap` map) holds the dataset row ids its values hold: a
# fit body's minibatch; the root's scope holds none, every row of each value's dataset.
ROWS = ("rows",)


def over(state: ChainMap) -> list[int] | None:
    """The rows the run is over: the ids the scope it is in carries (a fit body's minibatch), or None, every row
    (the root, and a run of a body that scores)."""
    return state.maps[0].get(ROWS)


def chunked(engine: Engine, step: Forward, state: ChainMap, found: Found, batch_size: int
            ) -> Iterator[tuple[list[int], dict, list]]:
    """The step's input in batches of `batch_size` dataset row ids: of every row in order, or of the rows the run is
    over (`over`); each one's ids, its tokens padded on the left to the input's longest row, its padding."""
    encoded = found.inputs[data.input_name(step.input)]
    rows = over(state)
    rows = rows if rows is not None else list(range(len(encoded["rows"])))
    for ids in (rows[lo:lo + batch_size] for lo in range(0, len(rows), batch_size)):
        tokens = [e["ids"] for e in data.by_id(encoded["rows"], ids)]
        yield ids, data.pad(engine.tokenizer, tokens, encoded["width"]), [encoded["width"] - len(x) for x in tokens]


def stops(engine: Engine, step: Generate) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def bind(engine: Engine, plan: Plan, step: Forward, state: ChainMap, found: Found, ids: list[int], pads: list[int],
         reading: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]]
         ) -> tuple[dict[str, Any], dict[tuple[str, tuple, int | None], list[dict]]]:
    """The schedule of `step` (a serving point) bound to one batch of dataset row `ids`, as the engine executes
    it: its sites in order, each record with what reaches its selection in the padded batch (`site`; at a decode
    call `decoding`; a part over the prompt of an entry beyond the prefill at every prompt position), a write with
    its ops bound (`edit`), its decode `calls` and, found `online`, each row's texts to find it by, a read with its
    name; the `stop` site; and a generation's `loop` and whether any write is `online`. A read is bound at the
    record of the value it is (`reading`, `schedule.gathered`), and the bound reads are also returned by entry,
    the value's coordinates and layer, each in the order its parts are done (a forward's one record; beyond a
    generation's prefill the part over the prompt, where there is one, then the part at its decode calls)."""
    schedule, width = step.schedule, found.inputs[data.input_name(step.input)]["width"]
    device = home(engine)
    prompts = [len(e["ids"]) for e in data.by_id(found.inputs[data.input_name(step.input)]["rows"], ids)]
    rows = data.by_id(held_rows(plan, cast(str, step.dataset)), ids) if schedule.get("online") else []
    sites, binds = [], {}
    for s in schedule["sites"]:
        records = []
        for r in s["records"]:
            key = r["write"] if "write" in r else r["read"]
            record = entries(step)[key] if "write" in r else reading[key, tuple(r["at"].items())][0]
            head_only = head(step, record)
            if "calls" in r:
                bound = decoding(record, record.kept, prompts, device)
            elif r.get("prompt"):
                where = [positions.prompted(p, head_only and "write" in r) for p in prompts]
                bound = site(record, record.kept, where, pads, width, head_only, device)
            else:
                bound = site(record, record.kept, data.by_id(record.positions[read_over(step, found)], ids), pads,
                             width, head_only, device)
            bound["layer"] = s["layer"]
            if "write" in r:
                resolved = {v["name"]: v for o in r["ops"] for v in o.values() if isinstance(v, dict) and "name" in v}
                bound["ops"] = edit(plan, record, state, step.at, ids, found, s["layer"], resolved, device)
                if r.get("online"):
                    pos = positions.parse(record.pos)
                    bound["online"] = {"pos": pos, "texts": [texts(pos, row) for row in rows],
                                       "eos": sorted(stops(engine, cast(Generate, step)))}
            else:
                bound["name"] = r["name"]
                binds.setdefault((key, tuple(r["at"].items()), s["layer"]), []).append(bound)
            if "calls" in r:
                bound["calls"] = r["calls"]
            records.append(bound)
        sites.append({"component": s["component"], "layer": s["layer"], "records": records})
    return {"sites": sites, "stop": schedule["stop"], "loop": schedule.get("loop"),
            "online": schedule.get("online", False)}, binds


def read_over(step: Forward, found: Found) -> str:
    """The dataset a step's records are read over in this run, which their `positions` are under: the step's, or
    the one a fit's eval reads in its place (`found.swap`)."""
    return found.swap.get(cast(str, step.dataset), cast(str, step.dataset))


def layers(engine: Engine, record: Intervention) -> list[int] | list[None]:
    """The layers a record is made at, a read's and a write's alike, as its record says: those its site's
    `layers` pick among the model's, in order (an int's one; None for a model's own value)."""
    return cast("list[int] | list[None]", record.layers if record.layers is not None else [None])


def stacked(record: Intervention, values: list) -> Any:
    """An entry's per-layer values as its value: the one, or under `layers` that stack, stacked layer-first."""
    if not address.stacks(cast(Site, record.site).layers):
        return values[0]
    # Under a device that spreads the model ("auto") the layers' values can live on different GPUs; they stack on
    # the first's.
    if isinstance(values[0], Tensor):
        return torch.stack([v.to(values[0].device) for v in values])
    return {k: torch.stack([v[k].to(values[0][k].device) for v in values]) for k in values[0]}


def joined(state: ChainMap, name: str, record: Intervention, at: dict[str, int], values: list) -> None:
    """A gathered entry's per-batch `values` joined along rows, on the first batch's device, as `stacked`
    joins layers (a batch whose rows all stopped before the calls a read names made its blank on the CPU, and
    under a device map batches can come back apart); a batch narrower than the widest along `pos` is padded
    to it first (`padded`). The value is put under `name` (`<step>.<entry>`) at `at`, its coordinates on its own
    axes."""
    dims = record.dims or []

    def cat(parts: list[Tensor]) -> Tensor:
        if "pos" in dims:
            parts = [padded(v, dims.index("pos"), max(v.shape[dims.index("pos")] for v in parts)) for v in parts]
        return torch.cat([v.to(parts[0].device) for v in parts], dims.index("rows"))
    put(state, name, at, cat(values) if isinstance(values[0], Tensor) else
        {k: cat([v[k] for v in values]) for k in values[0]})


def padded(value: Tensor, dim: int, n: int) -> Tensor:
    """`value` padded along `dim` to `n` with what marks a missing position: NaN, or -1 in ids."""
    shape = list(value.shape)
    shape[dim] = n - shape[dim]
    missing = float("nan") if value.is_floating_point() else -1
    return value if shape[dim] == 0 else torch.cat([value, torch.full(shape, missing, dtype=value.dtype,
                                                                       device=value.device)], dim)


def read(engine: Engine, plan: Plan, record: Intervention, value: Tensor, site: dict, ids: list[int], state: ChainMap,
         at: dict, found: Found, layer: int | None) -> Any:
    """One layer of an entry's value (`layer`) taken off the model in the prefill, in one batch (its dataset row `ids`),
    NaN past a row's own positions (`keep`, which only a batch whose rows hold different numbers of them has), then
    the ops after its last edit."""
    if "keep" in site:
        keep = site["keep"].to(value.device)
        value = value.masked_fill(~keep.reshape(*keep.shape, *[1] * (value.dim() - 2)), float("nan"))
    return with_ops(engine, plan, record, value, state, at, ids, site["tok"], found, layer)


def with_ops(engine: Engine, plan: Plan, record: Intervention, value: Tensor, state: ChainMap, at: dict,
             ids: list[int], places: Tensor | None, found: Found, layer: int | None) -> Any:
    """An entry's value at one of its layers (`layer`) taken off the model with the ops after its last edit
    applied in turn, each on the layout its record says the op before it left, their operands bound at that
    layer (`bound`). Past a probe's `encode` a column names its classes. An op is given each row's dataset row
    (`ids`) and each position's token position (`places`, `[rows, width]`, -1 past a row's own)."""
    dims, last = address.dims(cast(Site, record.site).component), len(record.ops) - len(gathers(record))
    vocabulary = None
    for i, o in enumerate(record.ops):
        if i >= last:
            args = bound(plan, o.op, arguments(o), state, at, ids, found, value, vocabulary, dims=dims, layers=layer)
            value = ops.call(o.op, value, args, engine=engine, dims=dims, rows=torch.tensor(ids), positions=places)
            if o.op == "encode":
                vocabulary = getattr(args["featurizer"], "classes", None)
        dims = o.dims or dims
    return value


def bound(plan: Plan, name: str, args: dict, state: ChainMap, at: dict, ids: list[int] | None, found: Found,
          of: Any = None, vocabulary: list | None = None, resolved: dict[str, dict] | None = None,
          dims: list[str] | None = None, layers: int | list[int] | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up at the point the schedule `resolved` them to for a
    write's ops (by name: a value or a featurizer at its own point, `at`, or an entry of the same forward as
    `{"live": <the name of its read>}`), else at the consumer's point `at` (a featurizer to its instance), columns
    resolved against the value's own
    vocabulary (the tokenizer's, as preflight resolved them into `found.answers` and `answered` lays them against
    the value `of` laid out `dims`; or a probe's class list) or read as text, at the dataset row `ids` (`value_at`;
    None: every row). The op acts at `layers` (inside an entry one of its layers, in a step `of`'s labels or its one
    layer, None at no layer), and an operand with a `layers` axis is aligned to them by label, its record's: its
    slice labelled L where the op acts at layer L (an entry of the same forward, its read at L), its slices in `of`'s
    order where it acts at several; one without is the same at every layer. Then an operand broadcasts by axis
    name: it is reshaped to the value's layout `dims`, a singleton axis standing where the value has an axis the
    operand leaves out (Ops' one rule lets it leave out rows, pos and layers)."""
    params, out = ops.parameters(name), {}
    resolved = resolved or {}
    for key, value in args.items():
        annotation = params[key].annotation
        if annotation == ops.Value and isinstance(value, str):
            record = record_of(plan.steps, value, at)
            axes, stacked = list(record.dims or []), "layers" in (record.dims or [])
            if "live" in resolved.get(value, {}):
                value = {"live": resolved[value]["live"]}
            else:
                value = value_at(plan, state, value, at, ids, resolved[value]["at"] if value in resolved else None)
                if stacked and layers is not None:
                    labels = cast(list, record.layers)
                    if isinstance(layers, int):
                        value, axes = value[labels.index(layers)], axes[1:]
                    elif layers != labels:
                        value = value[[labels.index(layer) for layer in layers]]
                if dims is not None and axes and isinstance(value, Tensor) and not (name == "apply" and key == "with"):
                    value = value.reshape([value.shape[axes.index(a)] if a in axes else 1 for a in dims])
        elif annotation == ops.Tokens:
            value = (indexed(plan, value, vocabulary, ids)[:, None] if vocabulary is not None else
                     answered(found.answers[refkey(value)], of, dims or [], ids))
            value = placed(value, of.device) if isinstance(of, Tensor) else value
        elif annotation == ops.Text:
            column, strip, lower = data.column_of(value) or ("", False, False)
            value = [(v if isinstance(v, list) else [v], strip, lower) for v in column_texts(plan, column, ids)]
        elif annotation == ops.Featurizer:
            value = lookup(state, value, resolved[value]["at"] if value in resolved else at)
        out[key] = value
    return out


def value_at(plan: Plan, state: ChainMap, name: str, at: dict, ids: list[int] | None,
             own: dict[str, int] | None = None) -> Any:
    """Value `name` at the consumer's point `at` (or at `own`, its coordinates on the value's own axes, where the
    schedule resolved them; its record is the one at `at`, whose point names every axis of the producer's step), at
    the dataset row `ids` (every one: None) of those it holds, which the scope it was found in says (`ROWS`: a fit
    body's values hold its minibatch's rows, the root's every row of their dataset), taken by id (`data.by_id`)."""
    value, dims = lookup(state, name, at if own is None else own), record_of(plan.steps, name, at).dims or []
    scope = next(m for m in state.maps if name in m)
    return data.by_id(value, ids, scope.get(ROWS), dims.index("rows")) if "rows" in dims else value


def taken(x: Tensor, record: dict) -> Tensor:
    """The record's positions and narrowing of `x`, a site's value with its token axis at 1.

    `record["positions"]` is `[rows, width]` token indices; a `key` axis is each row's own keys (`owned`);
    `record["select"]` maps a dimension of that layout to the indices kept along it, the same for every row. The
    indices are on the model's device (`bind` made them there), and move only to a value elsewhere (a device map
    that spreads the model, a record built on the host).
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
    padded to one width; only the positions its `keep` marks are written (a write whose rows are equal has no
    `keep` and writes by index alone). A `key` axis is each row's own keys in order, NaN past them (`owned`), to a
    read and to a write's ops alike, and a write writes those alone (`unowned`). Every index, mask and result moves
    to the value's device, and a result to its dtype, where it meets the value: no move for the indices `bind`
    made, which are on the model's device already.
    """
    axis = pos_axis(component)
    before = x = value.movedim(axis, 1)
    rows = torch.arange(x.shape[0], device=x.device)[:, None]
    wrote = False
    for w in records:
        if "ops" not in w:
            if "calls" in w:
                got.setdefault(w["name"], []).append(taken(x, w))
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
        new = new.to(device=x.device, dtype=x.dtype)
        if "keep" in w:
            keep = w["keep"].to(x.device)
            x[rows.expand_as(pos)[keep], pos[keep]] = new[keep]
        else:
            x[rows.expand_as(pos), pos] = new
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


def home(engine: Engine) -> str:
    """Where the model is, which is where the run makes featurizers and an engine sends a batch's inputs: where
    its first parameter is (on an NDIF server, or wherever the engine's `device` put it: the first of several
    under a device map that spreads it), else, while the model is a meta shell that loads its weights at its
    first forward, the engine's device."""
    device = str(next(engine.model.parameters()).device)
    return (engine.device or "cpu") if device == "meta" else device


def now(records: list[dict], call: int) -> list[dict]:
    """What a generation does at a site on forward call `call` (0 is the prefill), in order: a record with no
    `calls` acts on the prefill only, one with `calls` on those decode calls (None: every one)."""
    def wanted(record: dict) -> bool:
        if "calls" not in record:
            return call == 0
        return call > 0 and (record["calls"] is None or call in record["calls"])
    return [r for r in records if wanted(r)]


def landed(records: list[dict], seen: list[Tensor], tokenizer: PreTrainedTokenizerBase, call: int) -> list[dict]:
    """The records at decode call `call`, each write stamped with it (`call`, which positions its ops at the
    call's own token), a write with an `online` scope narrowed to the rows it lands on there (`keep`), as
    `positions.landing` decides from the inputs `seen` so far (one `[rows, 1]` per decode call); one that lands
    on no row is left out. A read is as it was."""
    out = []
    for w in records:
        w = {**w, "call": call} if "ops" in w else w
        online = w.get("online")
        if online is None:
            out.append(w)
            continue
        rows = [positions.landing(online["pos"], texts, [int(t[r, 0]) for t in seen], set(online["eos"]), tokenizer)
                for r, texts in enumerate(online["texts"])]
        if any(rows):
            out.append({**w, "keep": torch.tensor(rows)[:, None]})
    return out
