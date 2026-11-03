"""A step's input in batches, and a forward's schedule bound to one batch: each record's site in the padded batch,
the ops a write runs there, and a read's value with the ops after its last edit applied."""

from __future__ import annotations

import functools
from collections import ChainMap
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

import torch
from torch import Tensor

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import answer_ids, class_indices, refkey, texts_of
from clab.compile.positions import reads_head, scope_texts
from clab.compile.state import Findings
from clab.format import address, data, selectors
from clab.format.address import pos_axis
from clab.format.sweeps import lookup, put
from clab.plan import (Forward, Generate, Intervention, Plan, Site, arguments, bracketed, entries, gathers, gradient_of,
                        held_rows, kind_of, record_of)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def selection_in_batch(record: Intervention, picks: dict, row_positions: list, pads: list[int], width: int,
                       last_only: bool, device: str = "cpu", span: int | None = None) -> dict:
    """What the engine needs to reach a record's selection in this batch, `width` tokens wide: its component,
    its token indices into the padded batch (`positions`, `[rows, n]`; a row with fewer positions than the
    batch's widest repeats its last index, `keep` marking its own, `tok` the unpadded indices, -1 past its
    own; a row the record does not reach, the marker in `row_positions`, holds none), the part of the feature axes it
    keeps (`select`: `picks`, the indices preflight resolved its `select` to), and on a `key` axis each row's own keys,
    as many as the input's longest row has (`keys`, `span`; None: `width`). Where the value holds the last position
    alone (`last_only`: a generation's head in the prefill) every index is -1. Where every row holds `n` positions there
    is no `keep`, so that a read or a write of equal-width rows indexes by no mask, which the device would have to count
    before the host could go on. The indices and masks the value meets are made on `device`, where the model is
    (`tensor_on_device`); `tok` stays on the host, where the ops that read it (`noise`) count with it."""
    row_positions = [row if isinstance(row, list) else [] for row in row_positions]
    n = max(1, *map(len, row_positions))
    indices = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (n - len(row))
             for row, pad in zip(row_positions, pads)]
    token_positions = [row + [-1] * (n - len(row)) for row in row_positions]
    indices = [[-1] * n for _ in row_positions] if last_only else indices
    ragged = any(len(row) < n for row in row_positions)
    query_positions = [row + [row[-1] if row else 0] * (n - len(row)) for row in row_positions]
    return {"component": cast(Site, record.site).component, "positions": tensor_on_device(indices, device),
            "select": select_indices(record, picks, device),
            **key_index(record, pads, width, device, span, query_positions),
            **({"keep": tensor_on_device([[j < len(row) for j in range(n)] for row in row_positions], device)}
               if ragged else {}),
            "tok": torch.tensor(token_positions)}


def selection_at_decode_call(record: Intervention, picks: dict, prompts: list[int], device: str = "cpu") -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last, on `device`;
    and, in `tok`, the position before decode call 1's input in each row of `prompts` tokens, which with the call
    number is the input's."""
    return {"component": cast(Site, record.site).component,
            "positions": torch.full((len(prompts), 1), -1, device=device),
            "select": select_indices(record, picks, device), "tok": torch.tensor([[p - 1] for p in prompts])}


def key_index(record: Intervention, pads: list[int], width: int, device: str = "cpu", span: int | None = None,
              query_positions: list[list[int]] | None = None) -> dict:
    """A `key` axis as each row's own keys, its positions (`positions.prompt_positions`), which a key selector counts:
    `dim` is the axis's dimension in the selection's layout, `index` (`[rows, span]`) each row's keys in the
    padded batch, `width` tokens wide, in order, then its padding's, then up to `span` (the input's longest row;
    None: `width`) the first key again, and `own` which of those are the row's (`own_keys`), both on `device`. So a
    key axis is as wide as the input's longest row whatever the batch's width. Keys relative to the query (`kept`'s
    `{"offset": [...]}`) are `[rows, n, offsets]` instead: at each row's position p among its `query_positions` (`[rows,
    n]`, as the selection's positions are), key p + k in the padded batch, `own` where that is one of the row's keys
    (its key 0 in the index where not). Nothing on a component with no key axis."""
    layout = address.dims(cast(Site, record.site).component)
    if "key" not in layout:
        return {}
    relative = record.kept.get("key")
    if isinstance(relative, dict):
        assert query_positions is not None, "a read's selection is bound with its positions"
        valid = [[[0 <= p + k < width - pad for k in relative["offset"]] for p in row]
                 for row, pad in zip(query_positions, pads)]
        index = [[[p + k + pad if ok else 0 for k, ok in zip(relative["offset"], oks)] for p, oks in zip(row, okay)]
                 for row, pad, okay in zip(query_positions, pads, valid)]
        return {"keys": {"dim": layout.index("key"), "index": tensor_on_device(index, device),
                         "own": tensor_on_device(valid, device)}}
    span = width if span is None else span
    index = [[p + pad for p in positions.prompt_positions(width - pad)] + list(range(pad)) + [0] * (span - width)
             for pad in pads]
    own = [[j < width - pad for j in range(span)] for pad in pads]
    return {"keys": {"dim": layout.index("key"), "index": tensor_on_device(index, device),
                     "own": tensor_on_device(own, device)}}


def select_indices(record: Intervention, picks: dict, device: str = "cpu") -> dict[int, Tensor]:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps
    (`picks`, by axis), on `device`: on `key`, among the row's own keys, which `key_index` puts first, in order; keys
    relative to the query `key_index` takes itself, so they are not among these."""
    layout = address.dims(cast(Site, record.site).component)
    return {layout.index(axis): tensor_on_device(v, device) for axis, v in picks.items() if not isinstance(v, dict)}


def tensor_on_device(values: Any, device: str | torch.device) -> Tensor:
    """`values` (nested lists, or a tensor on the host) as a tensor on `device`: on a GPU copied from pinned
    memory without the host waiting for the copy (`non_blocking`), which runs in stream order before any kernel
    that reads it, so that binding a batch never drains the GPU's queue; anywhere else moved as torch moves it."""
    host = torch.as_tensor(values)
    return host.pin_memory().to(device, non_blocking=True) if torch.device(device).type == "cuda" else host.to(device)


def make_once(shared: dict, key: tuple, make: Callable[[], Any]) -> Any:
    """What `make` makes, made once in `shared`, a batch's tensors by what each is made from (`bind_schedule`'s): what
    the batch's records share, they share."""
    if key not in shared:
        shared[key] = make()
    return shared[key]


def bind_ops(plan: Plan, record: Intervention, state: ChainMap, at: dict, ids: list[int], findings: Findings,
             layer: int | None, operands: dict[str, dict] | None = None, device: str = "cpu",
             span: tuple[int, int] | None = None, shared: dict | None = None) -> list[dict]:
    """An entry's ops up to its last edit (or those `span` names, by index: a read's `per_call` ops) at one of the
    layers it is made at (`layer`), bound to one batch, as `visit_site` runs them: each with the layout of the value it
    meets there (`dims`: the op before it left it, as its record says; the site's selection before the first), the
    batch's dataset row ids (`ids`, and as one host tensor every op shares, `rows`), the layer, and its arguments
    bound at that layer (`bind_arguments`), each named operand as the schedule resolved it (`operands`, by name: a
    producer's point, or the name of a read of the same forward, for the forward to fill), a column's ids on `device`,
    and so is a number an edit takes as a value (`add`'s `with`, `scale`'s `factor`, a gate's optional `with`), which
    the edit would otherwise copy to the device, waiting for the copy, at each call. An `encode` holds the ops its
    `decode` closes over, bound the same way (`inner`), and the features its `select` keeps, on `device`, where the
    model and its featurizers are. The rows, each operand and each number are made once in the batch (`shared`,
    `bind_schedule`'s), whose every write shares them."""
    first, last = span or (0, len(record.ops) - len(gathers(record)))
    laid = [address.dims(cast(Site, record.site).component), *(o.dims or [] for o in record.ops)]
    shared = {} if shared is None else shared
    rows = make_once(shared, ("rows",), lambda: torch.tensor(ids))

    def bind_nodes(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            op, dims = node["op"], laid[node["at"]]
            args = bind_arguments(plan, op.op, arguments(op), state, at, ids, findings, operands=operands, dims=dims,
                                  layers=layer, shared=shared)
            params, editing = ops.parameters(op.op), ops.DECLARED[op.op]["group"] == "edits"
            for k, v in args.items():
                if params[k].annotation == ops.Tokens:
                    args[k] = tensor_on_device(v, device)
                elif editing and params[k].annotation == ops.Value and isinstance(v, (int, float)):
                    args[k] = make_once(shared, ("number", repr(v)), lambda v=v: tensor_on_device(v, device))
            out.append({"op": op.op, "args": args, "rows": rows, "dims": dims, "layers": layer})
            if op.op == "encode":
                featurizer, select = args["featurizer"], arguments(op).get("select")
                if select is not None:
                    out[-1]["features"] = tensor_on_device(selectors.select(select["features"], featurizer.size()),
                                                           device)
                out[-1]["inner"] = bind_nodes(node["inner"])
        return out
    edits = bind_nodes([n for n in bracketed(record.ops) if first <= n["at"] < last])
    del bind_nodes            # it refers to itself, a cycle that would hold `state` until the collector ran
    return edits


# The key under which a scope of the run's state (a `ChainMap` map) holds the dataset row ids its values hold: a
# fit body's minibatch; the root's scope holds none, every row of each value's dataset.
ROWS = ("rows",)


def scope_ids(state: ChainMap) -> list[int] | None:
    """The rows the run is over: the ids the scope it is in carries (a fit body's minibatch), or None, every row
    (the root, and a run of a body that scores)."""
    return state.maps[0].get(ROWS)


def input_batches(engine: Engine, step: Forward, state: ChainMap, findings: Findings, batch_size: int,
                  padding: str = "dataset") -> Iterator[tuple[list[int], dict, list]]:
    """The step's input in batches of `batch_size` dataset row ids, of every row in order or of the rows the run is
    over (`scope_ids`), padded as `padding` says (`data.PADDINGS`): under `dataset` and `batch` in that order, under
    `sorted` longest first by the encoded rows' token counts (rows of one count in that order); each batch's ids, its
    tokens padded on the left to the input's longest row (`dataset`) or to the batch's own (`batch`, `sorted`), and
    its padding. A step that joins its batches puts their rows back in the run's order (`in_run_order`)."""
    encoded = findings.inputs[data.input_name(step.input)]
    run_ids = scope_ids(state)
    run_ids = run_ids if run_ids is not None else list(range(len(encoded["rows"])))
    if padding == "sorted":
        run_ids = sorted(run_ids, key=lambda i: len(encoded["rows"][i]["ids"]), reverse=True)     # stable, as written
    for ids in (run_ids[lo:lo + batch_size] for lo in range(0, len(run_ids), batch_size)):
        tokens = [e["ids"] for e in data.by_id(encoded["rows"], ids)]
        width = encoded["width"] if padding == "dataset" else max(map(len, tokens))
        yield ids, data.pad(engine.tokenizer, tokens, width), [width - len(x) for x in tokens]


def in_run_order(state: ChainMap, value: Any, ids: list[int], axis: int = 0) -> Any:
    """`value`, whose rows along `axis` are dataset row `ids` in the order a step's batches ran them (`input_batches`),
    in the order of the rows the run is over (`scope_ids`; every row of the dataset, in order): as it is where the
    batches ran in that order, else taken by id (`data.by_id`), which puts `sorted` padding's rows back."""
    run_ids = scope_ids(state)
    run_ids = run_ids if run_ids is not None else sorted(ids)
    return value if ids == run_ids else data.by_id(value, run_ids, ids, axis)


def stop_ids(engine: Engine, step: Generate) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def bind_schedule(engine: Engine, plan: Plan, name: str, step: Forward, state: ChainMap, findings: Findings,
                  ids: list[int], pads: list[int],
                  values_read: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]]
                  ) -> tuple[dict[str, Any], dict[tuple[str, tuple, int | None], list[dict]]]:
    """The schedule of `step` (a serving point of step `name`) bound to one batch of dataset row `ids`, as the engine
    executes it: its sites in order, each record with what reaches its selection in the padded batch
    (`selection_in_batch`; at a decode call `selection_at_decode_call`; a part over the prompt of an entry beyond the
    prefill at every prompt position, the head's at the last, the one the prefill computes), a write with its ops bound
    (`bind_ops`), on a component of ids the size they are under (`ids`: the entry, the axis and the model's size of it,
    which `visit_site` holds what lands to), its decode `calls` and, found `online`, each row's texts to find it by, a
    read with its name (a `grad` entry's marked `capture`) and the ops it runs at each call bound (`per_call`, by
    `bind_ops`, at the point the read is for); the `stop` site; what the forward differentiates (`grad`, one item per
    distinct value: `of`, each of its reads' names with the `value_in_prefill` that makes the value from it, and `wrt`,
    the bound `grad` reads it serves; empty where it takes no gradient); and a generation's `loop` and whether any write
    is `online`. A read is bound at the record of the value it is (`values_read`, `schedule.values_read_by`), and the
    bound reads are also returned by entry, the value's coordinates and layer, each in the order its parts are done (a
    forward's one record; beyond a generation's prefill the part over the prompt, where there is one, then the part at
    its decode calls). Records that reach the same selection in the batch (one component, selection and positions: an
    entry at each of its layers, entries alike but for their ops) share its tensors, made once."""
    schedule, span = step.schedule, findings.inputs[data.input_name(step.input)]["width"]
    device = model_device(engine)
    prompts = [len(e["ids"]) for e in data.by_id(findings.inputs[data.input_name(step.input)]["rows"], ids)]
    width = prompts[0] + pads[0]            # the batch's: each row's tokens and its padding
    rows = data.by_id(held_rows(plan, cast(str, step.dataset)), ids) if schedule.get("online") else []
    sites, binds, named = [], {}, {}
    # the batch's tensors (each selection, its rows, operands) by what each is made from
    shared: dict[tuple, Any] = {}
    for site in schedule["sites"]:
        records = []
        for site_record in site["records"]:
            key = site_record["write"] if "write" in site_record else site_record["read"]
            record = entries(step)[key] if "write" in site_record else \
                values_read[key, tuple(site_record["at"].items())][0]
            head_only = reads_head(step, record)
            # The rows' positions, which records alike in the prefill hold as one list (`positions.prefill_positions`).
            row_positions = None if "calls" in site_record or site_record.get("prompt") else \
                record.positions[cast(str, step.dataset)]
            reached = (cast(Site, record.site).component, repr(record.kept), head_only, "calls" in site_record,
                       site_record.get("prompt"), None if row_positions is None else id(row_positions))
            bound_record = dict(make_once(shared, reached, lambda: selection_at_decode_call(
                record, record.kept, prompts, device) if "calls" in site_record else selection_in_batch(
                    record, record.kept, [positions.prompt_positions(p, head_only) for p in prompts]
                    if row_positions is None else data.by_id(row_positions, ids), pads, width, head_only, device,
                    span)))
            bound_record["layer"] = site["layer"]
            if "write" in site_record:
                operands = {v["name"]: v for o in site_record["ops"] for v in o.values()
                            if isinstance(v, dict) and "name" in v}
                bound_record["ops"] = bind_ops(plan, record, state, step.at, ids, findings, site["layer"], operands,
                                               device, shared=shared)
                if axis := address.COMPONENTS[cast(Site, record.site).component].get("ids"):
                    bound_record["ids"] = {"entry": f"{name}.{key}", "axis": axis, "size": engine.sizes()[axis]}
                if site_record.get("online"):
                    pos = positions.parse(record.pos)
                    bound_record["online"] = {"pos": pos, "texts": [scope_texts(pos, row) for row in rows],
                                              "eos": sorted(stop_ids(engine, cast(Generate, step)))}
            else:
                bound_record["name"] = site_record["name"]
                if gradient_of(record) is not None:
                    bound_record["capture"] = True
                # bound at the point it is read for, as `apply_ops_after_edit` binds the rest
                if site_record.get("per_call"):
                    first = len(record.ops) - len(gathers(record))
                    bound_record["per_call"] = bind_ops(plan, record, state,
                                                        values_read[key, tuple(site_record["at"].items())][2].at, ids,
                                                        findings, site["layer"], device=device,
                                                        span=(first, first + site_record["per_call"]), shared=shared)
                binds.setdefault((key, tuple(site_record["at"].items()), site["layer"]), []).append(bound_record)
                named[site_record["name"]] = (bound_record, record,
                                              values_read[key, tuple(site_record["at"].items())][2])
            if "calls" in site_record:
                bound_record["calls"] = site_record["calls"]
            records.append(bound_record)
        sites.append({"component": site["component"], "layer": site["layer"], "records": records})
    grad = [{"of": [(n, functools.partial(value_in_prefill, engine, plan, named[n][1], bound_record=named[n][0],
                                          ids=ids, state=state, at=named[n][2].at, findings=findings,
                                          layer=named[n][0]["layer"]))
                    for n in g["of"]],
             "wrt": [named[n][0] for n in g["wrt"]]} for g in schedule.get("grad", [])]
    return {"sites": sites, "stop": schedule["stop"], "grad": grad, "loop": schedule.get("loop"),
            "online": schedule.get("online", False)}, binds


def layers_made_at(record: Intervention) -> list[int] | list[None]:
    """The layers a record is made at, a read's and a write's alike, as its record says: those its site's
    `layers` pick among the model's, in order (an int's one; None for a model's own value)."""
    return cast("list[int] | list[None]", record.layers if record.layers is not None else [None])


def stack_layers(record: Intervention, values: list) -> Any:
    """An entry's per-layer values as its value: the one, or under `layers` that stack, stacked layer-first."""
    if not address.stacks(cast(Site, record.site).layers):
        return values[0]
    # Under a device that spreads the model ("auto") the layers' values can live on different GPUs; they stack on
    # the first's.
    if isinstance(values[0], Tensor):
        return torch.stack([v.to(values[0].device) for v in values])
    return {k: torch.stack([v[k].to(values[0][k].device) for v in values]) for k in values[0]}


def join_batches(state: ChainMap, name: str, record: Intervention, at: dict[str, int], values: list,
                 ids: list[int] | None = None) -> None:
    """A gathered entry's per-batch `values` joined along rows, on the first batch's device, as `stack_layers`
    joins layers (a batch whose rows all stopped before the calls a read names made its blank on the CPU, and
    under a device map batches can come back apart); a batch narrower than the widest along `pos` is padded
    to it first (`pad_with_missing`). The batches' rows are dataset row `ids` in the order they ran, put back in the
    run's order (`in_run_order`). The value is put under `name` (`<step>.<entry>`) at `at`, its coordinates on its own
    axes."""
    dims = record.dims or []

    def cat(parts: list[Tensor]) -> Tensor:
        if "pos" in dims:
            parts = [pad_with_missing(v, dims.index("pos"), max(v.shape[dims.index("pos")] for v in parts))
                     for v in parts]
        return torch.cat([v.to(parts[0].device) for v in parts], dims.index("rows"))
    value = cat(values) if isinstance(values[0], Tensor) else {k: cat([v[k] for v in values]) for k in values[0]}
    put(state, name, at, value if ids is None else in_run_order(state, value, ids, dims.index("rows")))


def pad_with_missing(value: Tensor, dim: int, n: int) -> Tensor:
    """`value` padded along `dim` to `n` with what marks a missing position: NaN, or -1 in ids."""
    shape = list(value.shape)
    shape[dim] = n - shape[dim]
    missing = float("nan") if value.is_floating_point() else -1
    return value if shape[dim] == 0 else torch.cat([value, torch.full(shape, missing, dtype=value.dtype,
                                                                       device=value.device)], dim)


def value_in_prefill(engine: Engine, plan: Plan, record: Intervention, value: Tensor, bound_record: dict,
                     ids: list[int], state: ChainMap, at: dict, findings: Findings, layer: int | None) -> Any:
    """One layer of an entry's value (`layer`) taken off the model in the prefill, in one batch (its dataset row `ids`),
    NaN past a row's own positions, -1 in ids (`keep`, which only a batch whose rows hold different numbers of them
    has), then the ops after its last edit."""
    if "keep" in bound_record:
        keep = bound_record["keep"].to(value.device)
        value = value.masked_fill(~keep.reshape(*keep.shape, *[1] * (value.dim() - 2)),
                                  float("nan") if value.is_floating_point() else -1)
    return apply_ops_after_edit(engine, plan, record, value, state, at, ids, bound_record["tok"], findings, layer)


def apply_ops_after_edit(engine: Engine, plan: Plan, record: Intervention, value: Any, state: ChainMap, at: dict,
                         ids: list[int], token_positions: Tensor | None, findings: Findings, layer: int | None,
                         start: int | None = None) -> Any:
    """An entry's value at one of its layers (`layer`) taken off the model with the ops after its last edit
    applied in turn (from op `start` on, where a generation ran those before it at each call), each on the layout
    its record says the op before it left, their operands bound at that layer (`bind_arguments`). Past a probe's
    `encode` a column names its classes. An op is given each row's dataset row (`ids`) and each position's token
    position (`token_positions`, `[rows, width]`, -1 past a row's own)."""
    dims, last = address.dims(cast(Site, record.site).component), len(record.ops) - len(gathers(record))
    first = last if start is None else start
    vocabulary, rows = None, torch.tensor(ids) if first < len(record.ops) else None
    for i, op in enumerate(record.ops):
        if i >= first:
            args = bind_arguments(plan, op.op, arguments(op), state, at, ids, findings, value, vocabulary, dims=dims,
                                  layers=layer)
            value = ops.call(op.op, value, args, engine=engine, dims=dims, layers=layer, rows=rows,
                             positions=token_positions)
            if op.op == "encode":
                vocabulary = getattr(args["featurizer"], "classes", None)
        dims = op.dims or dims
    return value


def bind_arguments(plan: Plan, name: str, args: dict, state: ChainMap, at: dict, ids: list[int] | None,
                   findings: Findings, of: Any = None, vocabulary: list | None = None,
                   operands: dict[str, dict] | None = None, dims: list[str] | None = None,
                   layers: int | list[int] | None = None, shared: dict | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up at the point the schedule resolved them to for a
    write's ops (`operands`, by name: a value or a featurizer at its own point, `at`, or an entry of the same forward as
    `{"live": <the name of its read>}`), else at the consumer's point `at` (a featurizer to its instance), columns
    resolved against the value's own
    vocabulary (the tokenizer's, as preflight resolved them into `findings.answers` and `answer_ids` lays them against
    the value `of` laid out `dims`; or a probe's class list) or read as text, at the dataset row `ids` (`value_at`;
    None: every row). The op acts at `layers` (inside an entry one of its layers, in a step `of`'s labels or its one
    layer, None at no layer), and an operand with a `layers` axis is aligned to them by label, its record's: its
    slice labelled L where the op acts at layer L (an entry of the same forward, its read at L), its slices in `of`'s
    order where it acts at several; one without is the same at every layer. Then an operand broadcasts by axis
    name: it is reshaped to the value's layout `dims`, a singleton axis standing where the value has an axis the
    operand leaves out (Ops' one rule lets it leave out rows, pos and layers). Where a write's ops are bound to a
    batch, a value is taken at its rows once in it (`shared`, `value_at`). A number an edit takes as a value goes to
    the value's device, which the edit would otherwise copy it to, waiting for the copy."""
    params, out, editing = ops.parameters(name), {}, ops.DECLARED[name]["group"] == "edits"
    operands = operands or {}
    adapter = name == "apply" and featurizers.KINDS[kind_of(plan, args["featurizer"])].adapts      # an adapter's input
    for key, value in args.items():
        annotation = params[key].annotation
        if annotation == ops.Value and isinstance(value, str):
            record = record_of(plan.steps, value, at)
            axes, stacked = list(record.dims or []), "layers" in (record.dims or [])
            if "live" in operands.get(value, {}):
                value = {"live": operands[value]["live"]}
            else:
                value = value_at(plan, state, value, at, ids, operands[value]["at"] if value in operands else None,
                                 shared)
                if stacked and layers is not None:
                    labels = cast(list, record.layers)
                    if isinstance(layers, int):
                        value, axes = value[labels.index(layers)], axes[1:]
                    elif layers != labels:
                        value = value[[labels.index(layer) for layer in layers]]
                if dims is not None and axes and isinstance(value, Tensor) and not (key == "with" and adapter):
                    value = value.reshape([value.shape[axes.index(a)] if a in axes else 1 for a in dims])
        elif annotation == ops.Tokens:
            value = (class_indices(plan, value, vocabulary, ids, of, dims or [])[:, None]
                     if vocabulary is not None else answer_ids(findings.answers[refkey(value)], of, dims or [], ids))
            value = tensor_on_device(value, of.device) if isinstance(of, Tensor) else value
        elif annotation == ops.Text:
            strip, lower = data.flags_of(value)
            value = [(v if isinstance(v, list) else [v], strip, lower)
                     for v in texts_of(plan, value, ids, of, dims or [])]
        elif annotation == ops.Featurizer:
            value = lookup(state, value, operands[value]["at"] if value in operands else at)
        if editing and annotation == ops.Value and isinstance(value, (int, float)) and isinstance(of, Tensor):
            value = tensor_on_device(value, of.device)    # as `bind_ops` binds a write's number, on the value's device
        out[key] = value
    return out


def value_at(plan: Plan, state: ChainMap, name: str, at: dict, ids: list[int] | None,
             own: dict[str, int] | None = None, shared: dict | None = None) -> Any:
    """Value `name` at the consumer's point `at` (or at `own`, its coordinates on the value's own axes, where the
    schedule resolved them; its record is the one at `at`, whose point names every axis of the producer's step), at
    the dataset row `ids` (every one: None) of those it holds, which the scope it was found in says (`ROWS`: a fit
    body's values hold its minibatch's rows, the root's every row of their dataset), taken by id (`data.by_id`). In
    a batch (`shared`, `bind_schedule`'s) a tensor is taken at its rows once, by the index of those rows its scope holds
    them at, made once on its device."""
    value, dims = lookup(state, name, at if own is None else own), record_of(plan.steps, name, at).dims or []
    scope = next(m for m in state.maps if name in m)
    if "rows" not in dims:
        return value
    held_ids = scope.get(ROWS)
    if shared is None or ids is None or not isinstance(value, Tensor):
        return data.by_id(value, ids, held_ids, dims.index("rows"))
    index = make_once(shared, ("index", id(scope), str(value.device)), lambda: tensor_on_device(
        ids if held_ids is None else data.by_id(list(range(len(held_ids))), ids, held_ids), value.device))
    return make_once(shared, ("value", id(value)), lambda: value.index_select(dims.index("rows"), index))


def selection_of(value: Tensor, record: dict) -> Tensor:
    """The record's positions and narrowing of `value`, a site's value with its token axis at 1.

    `record["positions"]` is `[rows, width]` token indices; a `key` axis is each row's own keys (`own_keys`);
    `record["select"]` maps a dimension of that layout to the indices kept along it, the same for every row. The
    indices are on the model's device (`bind_schedule` made them there), and move only to a value elsewhere (a device
    map that spreads the model, a record built on the host).
    """
    value = own_keys(value[torch.arange(value.shape[0], device=value.device)[:, None],
                           record["positions"].to(value.device)], record)
    for dim, index in record["select"].items():
        value = index_along(value, dim, index)
    return value


def own_keys(value: Tensor, record: dict) -> Tensor:
    """A selection's `key` axis as each row's own keys, its positions in order, NaN past them, where the record has
    one (`record["keys"]`: each row's keys in the padded batch, then its padding's, and which are its own, as many
    as the input's longest row has keys, past the batch's width too); else `value` as it is."""
    if "keys" not in record:
        return value
    dim, keys = record["keys"]["dim"], record["keys"]
    value = value.gather(dim, expanded_row_indices(value, dim, keys["index"].to(value.device)))
    return value.masked_fill(~expanded_row_indices(value, dim, keys["own"].to(value.device)), float("nan"))


def back_in_batch_order(current: Tensor, new: Tensor, record: dict) -> Tensor:
    """`new`, a selection in each row's own key order (`own_keys`), back in the batch's order over `current`: the
    row's own keys from `new`, its padding's as `current` has them; the keys past the batch's width (`own_keys`'s,
    up to the input's longest row) are no row's own, and go nowhere."""
    dim, keys, n = record["keys"]["dim"], record["keys"], current.shape[record["keys"]["dim"]]
    index, own = (expanded_row_indices(current, dim, keys[k][:, :n].to(current.device)) for k in ("index", "own"))
    return current.scatter(dim, index, torch.where(own, new.narrow(dim, 0, n), current.gather(dim, index)))


def index_along(value: Tensor, dim: int, index: Tensor) -> Tensor:
    """`value` at `index` along `dim`, the same indices on every row."""
    return value.index_select(dim, index.to(value.device))


def expanded_row_indices(value: Tensor, dim: int, index: Tensor) -> Tensor:
    """Each row's `[rows, n]` indices expanded to `value`'s shape with `n` along `dim`, for gather and scatter; each
    row's and position's, `[rows, pos, n]` (keys relative to the query), with `pos` along `value`'s token axis, 1."""
    shape = [1] * value.dim()
    shape[0], shape[dim] = index.shape[0], index.shape[-1]
    if index.dim() == 3:
        shape[1] = index.shape[1]
    size = list(value.shape)
    size[dim] = index.shape[-1]
    return index.reshape(shape).expand(size)


def visit_site(value: Tensor, component: str, records: list[dict], got: dict) -> tuple[Tensor, bool]:
    """`value` with a site's records done in listed order, and whether the engine sets it back (a record wrote, or
    captured it): a write's ops run on its selection of the current value, out of place, and the result lands there; a
    read takes its selection of the current value into `got` under its name (at a decode call, appended: one per call it
    reads), with the ops it runs at each call applied to it (`per_call`, a generation's beyond its prefill, which the
    part over the prompt runs at the prefill), so that what it holds per call is their value. A `grad` read (`capture`)
    puts there the current value whole instead, the tensor the forward goes on with, made a leaf that requires grad
    where it requires none, so that the graph starts at the shallowest such site; `differentiate` takes its gradient
    after the forward, and a later write here works on a copy of it, so that its gradient passes through the write.

    `original` is a write's selection as it was before any of these writes. An argument bound to an entry of
    the same forward (`{"live": <its read's name>}`) is that read's value in `got`, where this forward put it.
    An op is given the dataset row of each of the selection's rows (the op's `rows`) and the token position of
    each of its positions (the write's `tok`, plus the decode `call` it lands at, which the engine stamps). A
    write narrows at most one dimension of its selection (compile refuses more). A ragged write's rows are
    padded to one width; only the positions its `keep` marks are written (a write whose rows are equal has no
    `keep` and writes by index alone). A `key` axis is each row's own keys in order, NaN past them (`own_keys`), to a
    read and to a write's ops alike, and a write writes those alone (`back_in_batch_order`). A write of ids (`ids`,
    `bind_schedule`'s) runs its ops in float32 and is refused by its entry where what would land is no id under the
    model's size (`check_written_ids`), before the model reads it. Every index, mask and result moves to the value's
    device, and a result to its dtype, where it meets the value: no move for the indices `bind_schedule` made, which are
    on the model's device already.
    """
    axis = pos_axis(component)
    before = site_value = value.movedim(axis, 1)
    row_indices = torch.arange(site_value.shape[0], device=site_value.device)[:, None]
    wrote = copied = False      # copied: `site_value` is this visit's own copy, which a write may change in place
    for record in records:
        if "ops" not in record:
            if record.get("capture"):
                if not site_value.requires_grad:
                    leaf = site_value.detach().requires_grad_()
                    before, site_value = (leaf if before is site_value else before), leaf
                got[record["name"]], wrote, copied = site_value, True, False
            else:
                read = selection_of(site_value, record)
                if "per_call" in record:
                    read = apply_bound_ops(read, read, record["per_call"], got, record["tok"] + record.get("call", 0))
                if "calls" in record:
                    got.setdefault(record["name"], []).append(read)
                else:
                    got[record["name"]] = read
            continue
        if not copied:
            site_value, wrote, copied = site_value.clone(), True, True
        indices = record["positions"].to(site_value.device)
        current = site_value[row_indices, indices]
        new, original = own_keys(current, record), own_keys(before[row_indices, indices], record)
        whole = new
        for dim, index in record["select"].items():
            new, original = index_along(new, dim, index), index_along(original, dim, index)
        # each position, at a decode call its own
        token_positions = record["tok"] + record.get("call", 0) if "tok" in record else None
        # A write of ids runs on them in float32, as the model would never see them, and is held to its size first.
        new = apply_bound_ops(new.float() if "ids" in record else new, original, record["ops"], got, token_positions)
        if "ids" in record:
            check_written_ids(new, component, record, token_positions)
        new = new.to(device=current.device, dtype=current.dtype)
        for dim, index in record["select"].items():
            new = whole.index_copy(dim, index.to(current.device), new)
        new = back_in_batch_order(current, new, record) if "keys" in record else new
        new = new.to(device=site_value.device, dtype=site_value.dtype)
        if "keep" in record:
            keep = record["keep"].to(site_value.device)
            site_value[row_indices.expand_as(indices)[keep], indices[keep]] = new[keep]
        else:
            site_value[row_indices.expand_as(indices), indices] = new
    return site_value.movedim(1, axis), wrote


def check_written_ids(new: Tensor, component: str, record: dict, token_positions: Tensor | None) -> None:
    """Refuses, by the write's entry, what a write of ids (`record["ids"]`, `bind_schedule`'s) would hand the model at a
    position it lands on (its `keep`) that is no id under the model's size: a hole, -1, where its operand holds no id (a
    generation's read on a row that stopped early, a position past a row's own); a missing entry, NaN; and a number that
    is not a whole one from 0 below the size, which the model would index with out of range or truncated. It looks at
    what the ops wrote (`new`, in float32) before it is cast to ids, and so before it lands; the refusal names the first
    such entry's dataset row and token position (`token_positions`)."""
    ids = record["ids"]
    bad = (new < 0) | (new >= ids["size"]) | (new != new.round()) | new.isnan()
    if "keep" in record:
        bad &= record["keep"].to(bad.device).reshape(*record["keep"].shape, *[1] * (bad.dim() - 2))
    if not bad.any():
        return
    first = bad.nonzero()[0].tolist()
    value, (i, j) = float(new[tuple(first)]), first[:2]
    rows = record["ops"][0]["rows"]
    where = f"row {int(rows[i])}" + (f", token {int(token_positions[i, j])}" if token_positions is not None else "")
    said, why = (("a hole, -1,", "where its operand holds no id (a row that stopped before the call it was read at, "
                  "or a position past a row's own)") if value == -1 else
                 ("a missing entry, NaN,", "where its operand is missing") if value != value else
                 (f"{value:g}", "which is no id"))
    raise ValueError(f"{ids['entry']}: writes {said} into {component} at {where}, {why}; an id is a whole number from "
                     f"0 to {ids['size'] - 1} (the model's {ids['axis']}, {ids['size']}), which the model indexes with")


def apply_bound_ops(new: Tensor, original: Tensor, bound_ops: list[dict], got: dict,
                    token_positions: Tensor | None) -> Tensor:
    """`new` after a write's bound ops, each on the current value in turn. An `encode` runs the ops its `decode`
    closes over (`inner`) on the value's coordinates, on the features its `select` keeps, and writes the change
    back (`Kind.within`); `original` goes in with them, encoded the same way."""
    for op in bound_ops:
        args = {k: live_operand(got, v) if isinstance(v, dict) and "live" in v else v for k, v in op["args"].items()}
        if op["op"] != "encode":
            new = ops.call(op["op"], new, args, original=original, dims=op["dims"], layers=op.get("layers"),
                           rows=op.get("rows"), positions=token_positions)
            continue
        featurizer: Any = args["featurizer"]
        features = op.get("features")
        start = featurizer.encode(original)
        start = start if features is None else start.index_select(-1, features.to(start.device))
        new = featurizer.within(new, lambda z, start=start, inner=op["inner"]: apply_bound_ops(z, start, inner, got,
                                                                                              token_positions),
                                features)
    return new


def differentiate(got: dict[str, Any], grad: list[dict]) -> None:
    """What a forward's `grad` reads are, taken once the forward has reached its stop site, into `got`: for each item
    of the bound schedule's `grad` (`bind_schedule`), the sum of its value over every axis, made by `value_in_prefill`
    from the reads `got` holds (a missing entry, NaN, adds nothing), and one `torch.autograd.grad` of it for the values
    `visit_site` captured for the `grad` reads it serves, each read's gradient (zero where the sum does not depend on
    the value) selected and positioned as its record says (`selection_of`) under its name. The value is a number a
    gradient passes through, since compile refuses an `of` that runs an op none does (`transforms.NO_GRADIENT`); one
    that depends on no captured value has no graph, and every gradient of it is zero (a layer's attention scores of its
    values, a parallel block's MLP output of its attention's). Every value `got` holds is then detached, so that no
    graph outlives the call. A forward that takes no gradient is left as it is."""
    if not grad:
        return
    captured = {r["name"]: got[r["name"]] for g in grad for r in g["wrt"]}
    with torch.enable_grad():
        for i, g in enumerate(grad):
            scalar = sum(made(got[name]).float().nansum() for name, made in g["of"])
            assert isinstance(scalar, Tensor), "every value differentiated is read at least once"
            values = list({id(v): v for v in (captured[r["name"]] for r in g["wrt"])}.values())
            computed = torch.autograd.grad(scalar, values, retain_graph=i + 1 < len(grad), allow_unused=True) \
                if scalar.requires_grad else [None] * len(values)
            of_value = {id(v): torch.zeros_like(v) if d is None else d for v, d in zip(values, computed)}
            for r in g["wrt"]:
                got[r["name"]] = selection_of(of_value[id(captured[r["name"]])], r)
    for name, value in got.items():
        got[name] = value.detach() if isinstance(value, Tensor) else value


def live_operand(got: dict, operand: dict) -> Tensor:
    """A write's operand that names a read of its own forward: that read's value, which the engine made before
    it reached the write."""
    if operand["live"] not in got:
        raise ValueError(f"operand {operand['live']!r} is a read of this forward the engine had not made when it "
                         "reached the write; it reads it at a later site")
    return got[operand["live"]]


def model_device(engine: Engine) -> str:
    """Where the model is, which is where the run makes featurizers and an engine sends a batch's inputs: where
    its first parameter is (on an NDIF server, or wherever the engine's `device` put it: the first of several
    under a device map that spreads it), else, while the model is a meta shell that loads its weights at its
    first forward, the engine's device."""
    device = str(next(engine.model.parameters()).device)
    return (engine.device or "cpu") if device == "meta" else device


def acting_at_call(records: list[dict], call: int) -> list[dict]:
    """What a generation does at a site on forward call `call` (0 is the prefill), in order: a record with no
    `calls` acts on the prefill only, one with `calls` on those decode calls (None: every one)."""
    def wanted(record: dict) -> bool:
        if "calls" not in record:
            return call == 0
        return call > 0 and (record["calls"] is None or call in record["calls"])
    return [r for r in records if wanted(r)]


def stamped_at_call(records: list[dict], seen: list[Tensor], tokenizer: PreTrainedTokenizerBase,
                    call: int) -> list[dict]:
    """The records at decode call `call`, each write and each read with `per_call` ops stamped with it (`call`,
    which positions their ops at the call's own token), a write with an `online` scope narrowed to the rows it lands
    on there (`keep`), as `positions.lands_at_call` decides from the inputs `seen` so far (one `[rows, 1]` per decode
    call); one that lands on no row is left out. Any other read is as it was."""
    out = []
    for record in records:
        record = {**record, "call": call} if "ops" in record or "per_call" in record else record
        online = record.get("online")
        if online is None:
            out.append(record)
            continue
        rows = [positions.lands_at_call(online["pos"], texts, [int(t[r, 0]) for t in seen], set(online["eos"]),
                                        tokenizer)
                for r, texts in enumerate(online["texts"])]
        if any(rows):
            out.append({**record, "keep": torch.tensor(rows)[:, None]})
    return out
