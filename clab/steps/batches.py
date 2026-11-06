"""A step's input in batches, and a forward's schedule bound to one batch: each record's site in the padded batch,
the ops a write runs there, and a read's value with the ops after its last edit applied."""

from __future__ import annotations

import functools
import json
from collections import ChainMap
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

import torch
from torch import Tensor

from clab import featurizers, ops
from clab.compile import positions
from clab.compile.answers import answer_ids, refkey, texts_of
from clab.compile.positions import reads_head, scope_texts
from clab.compile.preflight import at_point
from clab.compile.state import Findings
from clab.format import address, data, registry, selectors
from clab.format.sweeps import lookup, put
from clab.plan import (Forward, Generate, Intervention, Op, OpStep, Plan, Site, arguments, bracketed, entries,
                        gathers, gradient_of, held_rows, kind_of, made_by, record_of)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def selection_in_batch(record: Intervention, picks: dict, row_positions: list, pads: list[int], width: int,
                       last_only: bool, device: str = "cpu", span: int | None = None) -> dict:
    """What the engine needs to reach a record's selection in this batch, `width` tokens wide: its token indices
    into the padded batch (`positions`, `[rows, n]`; a row with fewer positions than the batch's widest repeats its
    last index, `keep` marking its own and `kept` listing them, `[2, k]` (row, position) pairs a write lands by, `tok`
    the unpadded indices, -1 past its own; a row the record does not reach, the marker in `row_positions`, holds none),
    the part of the feature axes it keeps (`select`: `picks`, the indices
    preflight resolved its `select` to), and on a `key` axis each row's own keys, as many as the input's longest row
    has (`keys`, `span`; None: `width`). Where the value holds the last position
    alone (`last_only`: a generation's head in the prefill) every index is -1. Where every row holds `n` positions there
    is no `keep`, so that a read or a write of equal-width rows indexes by no mask; a ragged write indexes by `kept`,
    which the host counts here, so that the device never counts a mask before the host can go on. The indices and
    masks the value meets are made on `device`, where the model is (`tensor_on_device`); `tok` stays on the host, where
    the ops that read it (`noise`) count with it."""
    row_positions = [row if isinstance(row, list) else [] for row in row_positions]
    n = max(map(len, row_positions))
    indices = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (n - len(row))
             for row, pad in zip(row_positions, pads)]
    token_positions = [row + [-1] * (n - len(row)) for row in row_positions]
    indices = [[-1] * n for _ in row_positions] if last_only else indices
    keep = torch.tensor([[j < len(row) for j in range(n)] for row in row_positions]) \
        if any(len(row) < n for row in row_positions) else None
    query_positions = [row + [row[-1] if row else 0] * (n - len(row)) for row in row_positions]
    return {"positions": tensor_on_device(torch.tensor(indices, dtype=torch.long), device),
            "select": select_indices(record, picks, device),
            **key_index(record, pads, width, device, span, query_positions),
            **({"keep": tensor_on_device(keep, device), "kept": tensor_on_device(keep.nonzero().T, device)}
               if keep is not None else {}),
            "tok": torch.tensor(token_positions, dtype=torch.long)}


def key_index(record: Intervention, pads: list[int], width: int, device: str, span: int | None,
              query_positions: list[list[int]]) -> dict:
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
        keys = torch.tensor(query_positions, dtype=torch.long)[..., None] + torch.tensor(relative["offset"])
        pad = torch.tensor(pads)[:, None, None]
        own = (keys >= 0) & (keys < width - pad)
        return {"keys": {"dim": layout.index("key"), "index": tensor_on_device(torch.where(own, keys + pad, 0), device),
                         "own": tensor_on_device(own, device)}}
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


def op_binding(plan: Plan, record: Intervention, at: dict, layer: int | None,
               operands: dict[str, dict] | None = None, span: tuple[int, int] | None = None) -> list[dict]:
    """The half of binding an entry's ops up to its last edit (or those `span` names, by index: a read's `per_call`
    ops) at one of the layers it is made at (`layer`) that depends on the step alone, made once per step run and
    completed for each batch by `bind_ops`: each op as its brackets nest (`bracketed`), with the layout of the value
    it meets there (`dims`: the op before it left it, as its record says; the site's selection before the first), the
    layer, its arguments' binding at that layer and the consumer's point `at` (`argument_binding`, each named operand
    as the schedule resolved it, `operands`, by name: a producer's point, or the name of a read of the same forward,
    for the forward to fill), which of them are a column (`tokens`) and which a value an edit takes (`numbers`), and an
    `encode`'s `select` and the ops its `decode` closes over (`inner`), bound the same way."""
    first, last = span or (0, len(record.ops) - len(gathers(record)))
    laid = [address.dims(cast(Site, record.site).component), *(o.dims or [] for o in record.ops)]

    def nodes_of(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            op, dims, args = node["op"], laid[node["at"]], arguments(node["op"])
            params, editing = ops.parameters(op.op), ops.DECLARED[op.op]["group"] == "edits"
            out.append({"op": op.op, "dims": dims, "layers": layer,
                        "args": argument_binding(plan, op.op, args, at, operands=operands, dims=dims, layers=layer),
                        "tokens": [k for k in args if params[k].annotation == ops.Tokens],
                        "numbers": [k for k in args if editing and params[k].annotation == ops.Value]})
            if op.op == "encode":
                out[-1].update(select=args.get("select"), inner=nodes_of(node["inner"]))
        return out
    binding = nodes_of([n for n in bracketed(record.ops) if first <= n["at"] < last])
    del nodes_of            # it refers to itself, a cycle the collector would have to find
    return binding


def supplied_to(nodes: list[dict]) -> set[str]:
    """The values the run supplies an op (`ops.SUPPLIED`) that an entry's ops take (`op_binding`'s nodes, or bound):
    those an op's function names (`renormalize` its `original`, `noise` its `positions`), and those the ops an
    `encode` holds take, which it hands them (`original` encoded)."""
    return {name for node in nodes for name in (*registry.signature(ops.OPS[node["op"]]).parameters,
                                                *supplied_to(node.get("inner", []))) if name in ops.SUPPLIED}


def bind_ops(plan: Plan, binding: list[dict], state: ChainMap, ids: list[int], findings: Findings,
             device: str = "cpu", shared: dict | None = None) -> list[dict]:
    """An entry's ops bound to one batch from their binding (`op_binding`, the step's), as `visit_site` runs them:
    each with the layout of the value it meets, the batch's dataset row ids (`ids`, and as one host tensor every op
    shares, `rows`), the layer, and its arguments bound (`bind_arguments`), a column's ids on `device`, and so is a
    number an edit takes as a value (`add`'s `with`, `scale`'s `factor`, a gate's optional `with`), which the edit
    would otherwise copy to the device, waiting for the copy, at each call. An `encode` holds the ops its `decode`
    closes over, bound the same way (`inner`), and the features its `select` keeps, on `device`, where the model and
    its featurizers are. The rows, each operand and each number are made once in the batch (`shared`,
    `bind_schedule`'s), whose every write shares them."""
    shared = {} if shared is None else shared
    rows = make_once(shared, ("rows",), lambda: torch.tensor(ids))

    def bind_nodes(nodes: list[dict]) -> list[dict]:
        out = []
        for node in nodes:
            args = bind_arguments(plan, node["args"], state, ids, findings, dims=node["dims"], shared=shared)
            for k in node["tokens"]:
                args[k] = tensor_on_device(args[k], device)
            for k in node["numbers"]:
                if isinstance(v := args[k], (int, float)):
                    args[k] = make_once(shared, ("number", repr(v)), lambda v=v: tensor_on_device(v, device))
            out.append({"op": node["op"], "args": args, "rows": rows, "dims": node["dims"], "layers": node["layers"],
                        "rowed": [k for k, how in node["args"].items()
                                  if "tokens" in how or "text" in how or "name" in how and "rows" in how["dims"]]})
            if node["op"] == "encode":
                if node["select"] is not None:
                    out[-1]["features"] = tensor_on_device(selectors.select(node["select"]["features"],
                                                                            args["featurizer"].size()), device)
                out[-1]["inner"] = bind_nodes(node["inner"])
        return out
    edits = bind_nodes(binding)
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
                  padding: str) -> Iterator[tuple[list[int], dict, list]]:
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


def in_run_order(state: ChainMap, value: Any, ids: list[int], axis: int = 0, dataset: str | None = None) -> Any:
    """`value`, whose rows along `axis` are dataset row `ids` of `dataset` in the order a step's batches ran them
    (`input_batches`), in the order of the rows the run is over (`scope_ids`; every row of the dataset, in order): as
    it is where the batches ran in that order, else taken by id (`data.by_id`), which puts `sorted` padding's rows
    back."""
    run_ids = scope_ids(state)
    run_ids = run_ids if run_ids is not None else sorted(ids)
    return value if ids == run_ids else data.by_id(value, run_ids, ids, axis, dataset)


def stop_ids(engine: Engine, step: Generate) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def unreached_rows(plan: Plan, name: str, at: dict) -> dict[int, tuple[str, dict, Intervention, int]] | None:
    """Each row of value `name` (at the point `at`) that holds no value because a read it is, or descends from row for
    row, holds no position there, as the run recorded the read's positions (`positions.MISSING`: a call the row's
    generation never made, or an index that picks none of its tokens): by row, the read, its point and record, and the
    read's row (a `permute`'s, the row it drew). A value descends from every named `Value` its op step binds (`of`,
    then each argument in the op's order: an `add`'s `with`, a `scale`'s `factor`) and every one an entry's ops bind,
    each row for row; where two miss one row, the first is named. None where no row misses: the value, or every
    operand that misses, has no `rows` axis, it comes through a `count` (which counts such a row 0, a number), or
    no read it descends from has positions the run recorded (a generation in a fit's body records none)."""
    record = record_of(plan.steps, name, at, plan.featurizers)
    if "rows" not in (record.dims or []):
        return None
    found: dict[int, tuple[str, dict, Intervention, int]] = {}
    if isinstance(record, OpStep):
        if record.op == "count":
            return None
        for operand in [record.of, *operands_named(record)]:
            for row, missing in (unreached_rows(plan, operand, record.at) or {}).items():
                found.setdefault(row, missing)
        if record.op == "permute":      # row i holds the row the permutation drew
            found = {row: found[source] for row, source in enumerate(record.permutation or []) if source in found}
        return found or None
    if not isinstance(record, Intervention):
        return None
    found = {row: (name, at, record, row) for row, p in enumerate(record.positions.get(cast(str, record.dataset)) or [])
             if p == positions.MISSING}
    for operand in (n for o in record.ops for n in operands_named(o)):
        for row, missing in (unreached_rows(plan, operand, at) or {}).items():
            found.setdefault(row, missing)
    return found or None


def operands_named(o: Op | OpStep) -> list[str]:
    """The values op `o` binds by name, in its arguments' order: each `Value` argument written as a name."""
    args, takes = arguments(o), ops.parameters(o.op)
    return [args[k] for k, p in takes.items() if p.annotation == ops.Value and isinstance(args.get(k), str)]


def refuse_unreached(plan: Plan, findings: Findings, bound: dict, ids: list[int], landed: list[int | None]) -> None:
    """A write takes an operand read on every row it lands on: refused by name where an operand from another step
    holds no value on a row the write lands on (`bound["unreached"]`, `schedule_binding`'s, from `unreached_rows`),
    which it would otherwise write NaN into. `landed` is, per row of the batch (dataset row `ids`), the call the write
    lands at there (0, the prefill), or None where it lands on none (a generation's row that stopped first). The
    refusal names the write, the operand, the row and the call its read names that the row's generation never made."""
    for operand, missing in bound["unreached"].items():
        row, call = next(((row, call) for row, call in zip(ids, landed) if call is not None and row in missing),
                         (None, None))
        if row is None:
            continue
        read, at, record, source = missing[row]
        pos = positions.parse(record.pos)
        producer = made_by(plan, read)
        # A miss is a generation's read: a prefill's is refused in preflight, and a loaded read's rows are not followed.
        n = len(lookup(findings.said, producer, at)[source])
        frame, shown = pos["frame"], json.dumps(record.pos if isinstance(record.pos, int) else
                                                record.pos.model_dump(mode="json", exclude_defaults=True))
        made = f"{producer!r} made {n} call{'s' * (n != 1)} on row {source} ({'0' if n == 1 else f'0 to {n - 1}'})"
        why = f"{made}, and {read!r} reads " + (f"call {frame} ({shown})" if isinstance(frame, int) and frame >= n
                                                else f"{shown}, which they do not hold")
        given = f"operand {operand!r}" + (f", which descends from {read!r}," if operand != read else "")
        held = "" if source == row else f", which holds row {source} of {read!r}"
        where = f" at call {call}" if call else ""
        raise ValueError(f"{bound['write']}: {given} holds no value on row {row}{held} ({positions.MISSING}): {why}; a "
                         f"write takes an operand read on every row it lands on, and this one lands on row {row}"
                         f"{where}: read {read!r} at a call every row of {producer!r} makes ({{\"frame\": -1}} is "
                         "each row's last), or give it a min_new_tokens and a max_new_tokens that reach the call")


def schedule_binding(engine: Engine, plan: Plan, name: str, step: Forward,
                     values_read: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]]) -> list[dict]:
    """The half of binding the schedule of `step` (a serving point of step `name`) that depends on the step alone, made
    once per step run (a fit's body forward once per fit) and completed for each batch by `bind_schedule`: its sites
    in order, each with its records, each its schedule record (`scheduled`) and entry (`record`; a read at the record
    of the value it is, `values_read`, `schedule.values_read_by`, with the point it is read for, `point`, and the key
    it is returned by, `binds`), whether it reads the head alone (`head_only`), what records alike share its selection
    by (`reached`, but for the rows' positions), the fields of its bound record no batch changes (`fields`: its layer,
    its decode `calls`, a write's ops' supplied values (`takes`, `supplied_to`) and on a component of ids the size they
    are under, a read's name and, a `grad` entry's, `capture`),
    a write's ops (`op_binding`, at the operand points the schedule resolved) and, found `online`, its scope and the
    ids its rows stop on, and, where an operand from another step holds no value on some rows (`unreached_rows`), its
    name and those rows (`write`, `unreached`), which `refuse_unreached` holds the rows it lands on to; and a read's ops
    at each call (`per_call`, `op_binding` at the point it is read for)."""
    sites = []
    for site in step.schedule["sites"]:
        records = []
        for site_record in site["records"]:
            key = site_record["write"] if "write" in site_record else site_record["read"]
            read = None if "write" in site_record else (key, tuple(site_record["at"].items()))     # its value's
            record = entries(step)[key] if read is None else values_read[read][0]
            head_only = reads_head(step, record)
            layer = None if site_record.get("readout") else site["layer"]       # logits are at no layer
            bound: dict[str, Any] = {
                "scheduled": site_record, "record": record, "head_only": head_only,
                "reached": (cast(Site, record.site).component, repr(record.kept), head_only, "calls" in site_record,
                            site_record.get("prompt")), "fields": {"layer": layer}}
            if read is None:
                operands = {v["name"]: v for o in site_record["ops"] for v in o.values()
                            if isinstance(v, dict) and "name" in v}
                bound["ops"] = op_binding(plan, record, step.at, site["layer"], operands)
                bound["fields"]["takes"] = supplied_to(bound["ops"])
                unreached = {o: found for o, v in operands.items() if "live" not in v and o not in plan.featurizers
                             and (found := unreached_rows(plan, o, step.at)) is not None}
                if unreached:
                    bound.update(write=at_point(plan, f"{name}.{key}", step.at), unreached=unreached)
                if axis := address.COMPONENTS[cast(Site, record.site).component].get("ids"):
                    bound["fields"]["ids"] = {"entry": f"{name}.{key}", "axis": axis, "size": engine.sizes()[axis]}
                if site_record.get("online"):
                    bound["online"] = {"pos": positions.parse(record.pos),
                                       "eos": sorted(stop_ids(engine, cast(Generate, step)))}
            else:
                point = values_read[read][2]
                bound.update(point=point, binds=(*read, layer))
                bound["fields"]["name"] = site_record["name"]
                if gradient_of(record) is not None:
                    bound["fields"]["capture"] = True
                # bound at the point it is read for, as `apply_ops_after_edit` binds the rest
                if site_record.get("per_call"):
                    first = len(record.ops) - len(gathers(record))
                    bound["per_call"] = op_binding(plan, record, point.at, site["layer"],
                                                   span=(first, first + site_record["per_call"]))
                    bound["fields"]["takes"] = supplied_to(bound["per_call"])
            if "calls" in site_record:
                bound["fields"]["calls"] = site_record["calls"]
            records.append(bound)
        sites.append({"component": site["component"], "layer": site["layer"], "records": records})
    return sites


def bind_schedule(engine: Engine, plan: Plan, step: Forward, binding: list[dict], state: ChainMap, findings: Findings,
                  ids: list[int], pads: list[int]
                  ) -> tuple[dict[str, Any], dict[tuple[str, tuple, int | None], list[dict]]]:
    """The schedule of `step` bound to one batch of dataset row `ids`, as the engine executes it, from the half of its
    binding that depends on the step alone (`binding`, `schedule_binding`'s): its sites in order, each record with what
    reaches its selection in the padded batch (`selection_in_batch`; at a decode call the call's one position, the
    last, its `tok` the position before decode call 1's input, which with the call number is the input's; a part over
    the prompt of an entry beyond the prefill at every prompt position, the head's at the last, the one the prefill
    computes) and its fields, a write with its ops bound (`bind_ops`), on a component of ids the size they are under
    (`ids`: the entry, the axis and the model's size of it, which `visit_site` holds what lands to), its decode `calls`
    and, found `online`, each row's texts to find it by, a read with its name (a `grad` entry's marked `capture`) and
    the ops it runs at each call bound (`per_call`, by `bind_ops`, at the point the read is for); the `stop` site; what
    the forward differentiates (`grad`, one item per distinct value: `of`, each of its reads' names with the
    `value_in_prefill` that makes the value from it, and `wrt`, the bound `grad` reads it serves; empty where it takes
    no gradient); and a generation's `loop` and whether any write is `online`. A read is bound at the record of the
    value it is (`schedule.values_read_by`), and the bound reads are also returned by entry, the value's coordinates and
    layer, each in the order its parts are done (a forward's one record; beyond a generation's prefill the part over
    the prompt, where there is one, then the part at its decode calls). Records that reach the same selection in the
    batch (one component, selection and positions: an entry at each of its layers, entries alike but for their ops)
    share its tensors, made once. A write in the prefill lands on every row of the batch, so one whose operand holds no
    value on one of them is refused here, before the batch's forward (`refuse_unreached`)."""
    schedule, span = step.schedule, findings.inputs[data.input_name(step.input)]["width"]
    device = model_device(engine)
    prompts = [len(e["ids"]) for e in data.by_id(findings.inputs[data.input_name(step.input)]["rows"], ids)]
    width = prompts[0] + pads[0]            # the batch's: each row's tokens and its padding
    rows = data.by_id(held_rows(plan, cast(str, step.dataset)), ids) if schedule.get("online") else []
    sites, binds, named = [], {}, {}
    # the batch's tensors (each selection, its rows, operands) by what each is made from
    shared: dict[tuple, Any] = {}
    for site in binding:
        records = []
        for bound in site["records"]:
            site_record, record, head_only = bound["scheduled"], bound["record"], bound["head_only"]
            if "unreached" in bound and "calls" not in site_record:       # a write in the prefill lands on every row
                refuse_unreached(plan, findings, bound, ids, [0] * len(ids))
            # The rows' positions, which records alike in the prefill hold as one list (`positions.prefill_positions`).
            row_positions = None if "calls" in site_record or site_record.get("prompt") else \
                record.positions[cast(str, step.dataset)]
            reached = (*bound["reached"], None if row_positions is None else id(row_positions))
            bound_record = dict(make_once(shared, reached, lambda: {
                "positions": torch.full((len(prompts), 1), -1, device=device),
                "select": select_indices(record, record.kept, device),
                "tok": torch.tensor([[p - 1] for p in prompts])} if "calls" in site_record else selection_in_batch(
                    record, record.kept, [positions.prompt_positions(p, head_only) for p in prompts]
                    if row_positions is None else data.by_id(row_positions, ids), pads, width, head_only, device,
                    span)))
            bound_record.update(bound["fields"])
            if site_record.get("readout"):          # narrowed once read out, off the model (`value_in_prefill`)
                bound_record.update(readout=bound_record["select"], select={})
            if "ops" in bound:
                bound_record["ops"] = bind_ops(plan, bound["ops"], state, ids, findings, device, shared)
                if "online" in bound:
                    pos = bound["online"]["pos"]
                    bound_record["online"] = {"pos": pos, "texts": [scope_texts(pos, row) for row in rows],
                                              "eos": bound["online"]["eos"]}
            else:
                if "per_call" in bound:
                    bound_record["per_call"] = bind_ops(plan, bound["per_call"], state, ids, findings, device, shared)
                binds.setdefault(bound["binds"], []).append(bound_record)
                named[bound_record["name"]] = (bound_record, record, bound["point"])
            records.append(bound_record)
        sites.append({"component": site["component"], "layer": site["layer"], "records": records})
    grad = [{"of": [(n, functools.partial(value_in_prefill, engine, plan, named[n][1], bound_record=named[n][0],
                                          ids=ids, state=state, at=named[n][2].at, findings=findings,
                                          layer=named[n][0]["layer"]))
                    for n in g["of"]],
             "wrt": [named[n][0] for n in g["wrt"]]} for g in schedule.get("grad", [])]
    return {"sites": sites, "stop": schedule["stop"], "grad": grad, "loop": schedule.get("loop"),
            "online": schedule.get("online", False)}, binds


def schedule_for_row(bound: dict[str, Any], r: int) -> dict[str, Any]:
    """A bound schedule's half that is row `r`'s (`r` the row's index in the batch), as an engine that runs each row
    as a request of its own hands it (the vllm engine): each record's indices, masks, keys and texts at that row
    alone, `[1, ...]`, and a write's operands with a rows axis taken at it (`record_for_row`); the rest shared."""
    return {**bound, "sites": [{**site, "records": [record_for_row(record, r) for record in site["records"]]}
                               for site in bound["sites"]]}


def record_for_row(record: dict[str, Any], r: int) -> dict[str, Any]:
    """A bound record at row `r` alone: its `positions` and `tok` `[1, n]`, its `keep` and the `kept` pairs counted
    from it, its `keys` index and own, an online write's texts, and its ops' and `per_call` ops' arguments that hold
    a rows axis (`rowed`, `bind_ops`'s) with the op's `rows` (`op_for_row`); every other field as it is. Each slice
    is a copy (`row_of`): a view travels with its whole storage, the batch's, where the row is pickled into a
    request."""
    out = dict(record)
    for key in ("positions", "tok"):
        if key in record:
            out[key] = row_of(record[key], r)
    if "keep" in record:
        keep = row_of(record["keep"], r)
        out["keep"], out["kept"] = keep, keep.nonzero().T
    if "keys" in record:
        out["keys"] = {**record["keys"], "index": row_of(record["keys"]["index"], r),
                       "own": row_of(record["keys"]["own"], r)}
    if "online" in record:
        out["online"] = {**record["online"], "texts": record["online"]["texts"][r:r + 1]}
    for key in ("ops", "per_call"):
        if key in record:
            out[key] = [op_for_row(op, r) for op in record[key]]
    return out


def op_for_row(op: dict[str, Any], r: int) -> dict[str, Any]:
    """A bound op at row `r` alone: its `rowed` arguments (a tensor or a list per row) and its `rows` taken at
    `r`, an `encode`'s inner ops the same way."""
    out = {**op, "args": {k: row_of(v, r) if k in op["rowed"] else v for k, v in op["args"].items()},
           "rows": row_of(op["rows"], r)}
    if "inner" in op:
        out["inner"] = [op_for_row(inner, r) for inner in op["inner"]]
    return out


def row_of(value: Any, r: int) -> Any:
    """Row `r` of a per-row value as a `[1, ...]` tensor of its own, or a list of the one element: a tensor's slice is
    cloned, since a view pickles with its whole storage."""
    return value[r:r + 1].clone() if isinstance(value, Tensor) else value[r:r + 1]


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
    to it first with what marks a missing position, NaN, or -1 in ids. The batches' rows are dataset row `ids` in the
    order they ran, put back in the run's order (`in_run_order`). The value is put under `name` (`<step>.<entry>`) at
    `at`, its coordinates on its own axes."""
    dims = record.dims or []

    def cat(parts: list[Tensor]) -> Tensor:
        if "pos" in dims:
            pos = dims.index("pos")
            n = max(v.shape[pos] for v in parts)
            parts = [v if v.shape[pos] == n else torch.cat([v, torch.full(
                [*v.shape[:pos], n - v.shape[pos], *v.shape[pos + 1:]], float("nan") if v.is_floating_point() else -1,
                dtype=v.dtype, device=v.device)], pos) for v in parts]
        return torch.cat([v.to(parts[0].device) for v in parts], dims.index("rows"))
    value = cat(values) if isinstance(values[0], Tensor) else {k: cat([v[k] for v in values]) for k in values[0]}
    put(state, name, at, value if ids is None else in_run_order(state, value, ids, dims.index("rows"), record.dataset))


def value_in_prefill(engine: Engine, plan: Plan, record: Intervention, value: Tensor, bound_record: dict,
                     ids: list[int], state: ChainMap, at: dict, findings: Findings, layer: int | None) -> Any:
    """One layer of an entry's value (`layer`) taken off the model in the prefill, in one batch (its dataset row `ids`),
    NaN past a row's own positions, -1 in ids (`keep`, which only a batch whose rows hold different numbers of them
    has), then the ops after its last edit. A read of `logits` the schedule does at the last block's output
    (`readout`) is first read out there (`engine.readout`) and narrowed as its `select` says."""
    if "readout" in bound_record:
        value = engine.readout(value)
        for dim, index in bound_record["readout"].items():
            value = index_along(value, dim, index)
    if "keep" in bound_record:
        keep = bound_record["keep"].to(value.device)
        value = value.masked_fill(~keep.reshape(*keep.shape, *[1] * (value.dim() - 2)),
                                  float("nan") if value.is_floating_point() else -1)
    return apply_ops_after_edit(engine, plan, record, value, state, at, ids, bound_record["tok"], findings, layer)


def apply_ops_after_edit(engine: Engine, plan: Plan, record: Intervention, value: Any, state: ChainMap, at: dict,
                         ids: list[int], token_positions: Tensor, findings: Findings, layer: int | None,
                         start: int | None = None) -> Any:
    """An entry's value at one of its layers (`layer`) taken off the model with the ops after its last edit
    applied in turn (from op `start` on, where a generation ran those before it at each call), each on the layout
    its record says the op before it left, their operands bound at that layer (`argument_binding`,
    `bind_arguments`). Past a probe's `encode` a column names its classes. An op is given each row's dataset row
    (`ids`) and each position's token position (`token_positions`, `[rows, width]`, -1 past a row's own)."""
    dims, last = address.dims(cast(Site, record.site).component), len(record.ops) - len(gathers(record))
    first = last if start is None else start
    vocabulary, rows = None, torch.tensor(ids) if first < len(record.ops) else None
    for i, op in enumerate(record.ops):
        if i >= first:
            binding = argument_binding(plan, op.op, arguments(op), at, dims=dims, layers=layer)
            args = bind_arguments(plan, binding, state, ids, findings, value, vocabulary, dims=dims)
            value = ops.call(op.op, value, args, engine=engine, dims=dims, layers=layer, rows=rows,
                             positions=token_positions)
            if op.op == "encode":
                vocabulary = getattr(args["featurizer"], "classes", None)
        dims = op.dims or dims
    return value


def argument_binding(plan: Plan, name: str, args: dict, at: dict, *, operands: dict[str, dict] | None = None,
                     dims: list[str], layers: int | list[int] | None = None) -> dict[str, dict]:
    """The half of binding op `name`'s arguments (`args`) that depends on the op and its point alone, which
    `bind_arguments` completes with what a batch holds: by argument, a value as written (`value`: a literal, a number,
    an entry of the same forward as `{"live": <the name of its read>}`, from `operands`), whether an edit takes it as a
    value (`number`), or what to look up and where: a value's `name` at the point the schedule resolved it to for a
    write's ops (`operands`, by name: its own coordinates), else at the consumer's point `at`, with its layout and
    dataset (`dims`, `dataset`, its record's), the slice of its `layers` axis the op takes (`layer`) and the shape it
    broadcasts by (`shape`); a column, through the value's vocabulary (`tokens`) or as text with its flags (`text`,
    `flags`); a featurizer at its point (`featurizer`, `at`).

    The op acts at `layers` (inside an entry one of its layers, in a step `of`'s labels or its one layer, None at no
    layer), and an operand with a `layers` axis is aligned to them by label, its record's: its slice labelled L where
    the op acts at layer L (`layer` its index; an entry of the same forward, its read at L), its slices in `of`'s order
    where it acts at several (`layer` their indices); one without is the same at every layer (`layer` None). Then an
    operand broadcasts by axis name: it is reshaped to the value's layout `dims`, a singleton axis standing where the
    value has an axis the operand leaves out (`shape`, for each axis of `dims` the operand's axis there or None; Ops'
    one rule lets it leave out rows, pos and layers), except an adapter's input, which keeps its own."""
    params, out, editing = ops.parameters(name), {}, ops.DECLARED[name]["group"] == "edits"
    operands = operands or {}
    adapter = name == "apply" and featurizers.KINDS[kind_of(plan, args["featurizer"])].adapts      # an adapter's input
    for key, value in args.items():
        annotation = params[key].annotation
        number = editing and annotation == ops.Value
        if annotation == ops.Value and isinstance(value, str):
            record = record_of(plan.steps, value, at, plan.featurizers)
            axes = list(record.dims or [])
            if "live" in operands.get(value, {}):
                out[key] = {"value": {"live": operands[value]["live"]}, "number": number}
                continue
            layer = None
            if "layers" in axes and layers is not None:
                labels = cast(list, record.layers)
                if isinstance(layers, int):
                    layer, axes = labels.index(layers), axes[1:]
                elif layers != labels:
                    layer = [labels.index(each) for each in layers]
            out[key] = {"name": value, "at": operands[value]["at"] if value in operands else at,
                        "dims": record.dims or [], "dataset": record.dataset, "layer": layer, "number": number,
                        "shape": [axes.index(a) if a in axes else None for a in dims]
                        if axes and not (key == "with" and adapter) else None}
        elif annotation == ops.Tokens:
            out[key] = {"tokens": value}
        elif annotation == ops.Text:
            out[key] = {"text": value, "flags": data.flags_of(value)}
        elif annotation == ops.Featurizer:
            out[key] = {"featurizer": value, "at": operands[value]["at"] if value in operands else at}
        else:
            out[key] = {"value": value, "number": number}
    return out


def bind_arguments(plan: Plan, binding: dict[str, dict], state: ChainMap, ids: list[int] | None, findings: Findings,
                   of: Any = None, vocabulary: list | None = None, *, dims: list[str],
                   shared: dict | None = None) -> dict:
    """An op's arguments bound to tensors from their binding (`argument_binding`): a value looked up where it says, at
    the dataset row `ids` (`value_at`; None: every row), its slice of `layers` taken and broadcast to the value's
    layout `dims`; a column resolved against the value's own vocabulary (the tokenizer's, as preflight resolved it into
    `findings.answers` and `answer_ids` lays it against the value `of` laid out `dims`; or a probe's class list) or
    read as text, at those rows; a featurizer to its instance. Where a write's ops are bound to a batch, a value is
    looked up and taken at its rows once in it (`shared`, `value_at`). A number an edit takes as a value goes to the
    value's device, which the edit would otherwise copy it to, waiting for the copy."""
    out = {}
    for key, how in binding.items():
        if "name" in how:
            value = value_at(plan, state, how["name"], how["at"], ids, dims=how["dims"], dataset=how["dataset"]) \
                if shared is None else make_once(shared, ("operand", how["name"], tuple(how["at"].items())),
                                                 lambda how=how: value_at(plan, state, how["name"], how["at"], ids,
                                                                          shared=shared, dims=how["dims"],
                                                                          dataset=how["dataset"]))
            if how["layer"] is not None:
                value = value[how["layer"]]
            if how["shape"] is not None and isinstance(value, Tensor):
                value = value.reshape([1 if i is None else value.shape[i] for i in how["shape"]])
        elif "tokens" in how:
            column = how["tokens"]
            # under a probe, each row's class index (preflight checked each is one)
            value = (torch.tensor([vocabulary.index(v) for v in texts_of(plan, column, ids, of, dims)])[:, None]
                     if vocabulary is not None else answer_ids(findings.answers[refkey(column)], of, dims, ids))
            value = tensor_on_device(value, of.device) if isinstance(of, Tensor) else value
        elif "text" in how:
            strip, lower = how["flags"]
            value = [(v if isinstance(v, list) else [v], strip, lower)
                     for v in texts_of(plan, how["text"], ids, of, dims)]
        elif "featurizer" in how:
            value = lookup(state, how["featurizer"], how["at"])
        else:
            value = how["value"]
        if how.get("number") and isinstance(value, (int, float)) and isinstance(of, Tensor):
            value = tensor_on_device(value, of.device)
        out[key] = value
    return out


def value_at(plan: Plan, state: ChainMap, name: str, at: dict, ids: list[int] | None,
             own: dict[str, int] | None = None, shared: dict | None = None, dims: list[str] | None = None,
             dataset: str | None = None) -> Any:
    """Value `name` at the consumer's point `at` (or at `own`, its coordinates on the value's own axes, where the
    schedule resolved them; its record is the one at `at`, whose point names every axis of the producer's step, and its
    layout `dims` and `dataset` where given), at the dataset row `ids` (every one: None) of those it holds, which the
    scope it was found in says (`ROWS`: a fit body's values hold its minibatch's rows, the root's every row of their
    dataset), taken by id (`data.by_id`, which refuses an id it does not hold naming the dataset). In a batch
    (`shared`, `bind_schedule`'s) a tensor is taken at its rows once, by the index of those rows its scope holds them
    at, made once on its device."""
    value = lookup(state, name, at if own is None else own)
    if dims is None:
        record = record_of(plan.steps, name, at, plan.featurizers)
        dims, dataset = record.dims or [], record.dataset
    scope = next(m for m in state.maps if name in m)
    if "rows" not in dims:
        return value
    held_ids = scope.get(ROWS)
    if shared is None or ids is None or not isinstance(value, Tensor):
        return data.by_id(value, ids, held_ids, dims.index("rows"), dataset)
    index = make_once(shared, ("index", id(scope), str(value.device)), lambda: tensor_on_device(
        ids if held_ids is None else data.by_id(list(range(len(held_ids))), ids, held_ids, dataset=dataset),
        value.device))
    return make_once(shared, ("value", id(value)), lambda: value.index_select(dims.index("rows"), index))


def selection_of(value: Tensor, record: dict) -> Tensor:
    """The record's positions and narrowing of `value`, a site's value with its token axis at 1.

    `record["positions"]` is `[rows, width]` token indices; a `key` axis is each row's own keys (`own_keys`);
    `record["select"]` maps a dimension of that layout to the indices kept along it, the same for every row. The
    indices are on the model's device (`bind_schedule` made them there), and move only to a value elsewhere (a device
    map that spreads the model, a record built on the host).
    """
    value = own_keys(value[torch.arange(value.shape[0], device=value.device)[:, None],
                           moved_to(record["positions"], value.device)], record)
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
    value = value.gather(dim, expanded_row_indices(value, dim, moved_to(keys["index"], value.device)))
    return value.masked_fill(~expanded_row_indices(value, dim, moved_to(keys["own"], value.device)), float("nan"))


def index_along(value: Tensor, dim: int, index: Tensor) -> Tensor:
    """`value` at `index` along `dim`, the same indices on every row."""
    return value.index_select(dim, moved_to(index, value.device))


def moved_to(index: Tensor, device: torch.device) -> Tensor:
    """`index` on `device`: itself where it is there already, as the indices `bind_schedule` made are, which the check
    tells for less than a `.to` that moves nothing costs; else moved (a record built on the host, a device map that
    spreads the model)."""
    return index if index.device == device else index.to(device)


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

    `original` is a write's selection as it was before any of these writes, taken where its ops take it (`takes`,
    `bind_schedule`'s: `supplied_to`; `renormalize`, an `encode` around one). An argument bound to an entry of
    the same forward (`{"live": <its read's name>}`) is that read's value in `got`, where this forward put it.
    An op is given the dataset row of each of the selection's rows (the op's `rows`) and, where one takes it, the token
    position of each of its positions (`token_positions_of`: the write's `tok`, plus the decode `call` it lands at,
    which the engine stamps). A write narrows at most one dimension of its selection (compile refuses more). A
    ragged write's rows are padded to one width; only the positions its `keep` marks are written, by the indices of
    those (`kept`), which the host counted when it bound the batch (a write whose rows are equal has no `keep` and
    writes by index alone). A `key` axis is each row's own keys in order, NaN past them (`own_keys`), to a read and to
    a write's ops alike, and a write writes those alone, its padding's keys as they were and the keys past the batch's
    width (up to the input's longest row) nowhere, being no row's own. A write of ids (`ids`,
    `bind_schedule`'s) runs its ops in float32 and is refused by its entry where what would land is no id under the
    model's size (`check_written_ids`), before the model reads it. Every index, mask and result moves to the value's
    device, and a result to its dtype, where it meets the value and only where it is elsewhere (`moved_to`): the
    indices `bind_schedule` made are on the model's device already.
    """
    layout = address.COMPONENTS[component]["dims"]         # the token axis in its own layout: a pattern's query
    axis = layout.index("query") if "query" in layout else layout.index("seq")
    before = site_value = value.movedim(axis, 1)
    device = site_value.device
    row_indices = torch.arange(site_value.shape[0], device=device)[:, None]
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
                    read = apply_bound_ops(read, read, record["per_call"], got, token_positions_of(record))
                if "calls" in record:
                    got.setdefault(record["name"], []).append(read)
                else:
                    got[record["name"]] = read
            continue
        if not copied:
            site_value, wrote, copied = site_value.clone(), True, True
        indices = moved_to(record["positions"], device)
        current = site_value[row_indices, indices]
        new = whole = own_keys(current, record)
        original = own_keys(before[row_indices, indices], record) if "original" in record.get("takes", ops.SUPPLIED) \
            else None
        for dim, index in record["select"].items():
            new = index_along(new, dim, index)
            original = None if original is None else index_along(original, dim, index)
        new = apply_bound_ops(new.float() if "ids" in record else new, original, record["ops"], got,
                              token_positions_of(record))
        if "ids" in record:
            check_written_ids(new, component, record)
        if new.device != device or new.dtype != site_value.dtype:       # `current`'s, which it lands in
            new = new.to(device=device, dtype=site_value.dtype)
        for dim, index in record["select"].items():
            new = whole.index_copy(dim, moved_to(index, device), new)
        if "keys" in record:        # back in the batch's order over `current`
            dim, keys, n = record["keys"]["dim"], record["keys"], current.shape[record["keys"]["dim"]]
            index, own = (expanded_row_indices(current, dim, moved_to(keys[k][:, :n], device))
                          for k in ("index", "own"))
            new = current.scatter(dim, index, torch.where(own, new.narrow(dim, 0, n), current.gather(dim, index)))
        if "kept" in record:        # the (row, position) pairs its `keep` marks, by index: the host counted them
            i, j = moved_to(record["kept"], device)
            site_value[i, indices[i, j]] = new[i, j]
        else:
            site_value[row_indices.expand_as(indices), indices] = new
    return site_value.movedim(1, axis), wrote


def token_positions_of(record: dict) -> Tensor | None:
    """The token position of each of a record's positions (its `tok`, plus the decode `call` it lands at, which the
    engine stamps), where an op of its takes them (`takes`: `noise`'s `positions`; a record without `takes` is
    given every supplied value it holds); else None, which nothing reads."""
    return record["tok"] + record.get("call", 0) if "tok" in record and "positions" in record.get(
        "takes", ops.SUPPLIED) else None


def check_written_ids(new: Tensor, component: str, record: dict) -> None:
    """Refuses, by the write's entry, what a write of ids (`record["ids"]`, `bind_schedule`'s) would hand the model at a
    position it lands on (its `keep`) that is no id under the model's size: a hole, -1, where its operand holds no id (a
    generation's read on a row that stopped early, a position past a row's own); a missing entry, NaN; and a number that
    is not a whole one from 0 below the size, which the model would index with out of range or truncated. It looks at
    what the ops wrote (`new`, in float32) before it is cast to ids, and so before it lands; the refusal names the first
    such entry's dataset row and token position (its `tok`, plus the decode `call` it lands at)."""
    ids = record["ids"]
    bad = (new < 0) | (new >= ids["size"]) | (new != new.round()) | new.isnan()
    if "keep" in record:
        bad &= record["keep"].to(bad.device).reshape(*record["keep"].shape, *[1] * (bad.dim() - 2))
    if not bad.any():
        return
    first = bad.nonzero()[0].tolist()
    value, (i, j) = float(new[tuple(first)]), first[:2]
    rows = record["ops"][0]["rows"]
    where = f"row {int(rows[i])}" + (f", token {int(record['tok'][i, j]) + record.get('call', 0)}" if "tok" in record
                                     else "")
    said, why = (("a hole, -1,", "where its operand holds no id (a row that stopped before the call it was read at, "
                  "or a position past a row's own)") if value == -1 else
                 ("a missing entry, NaN,", "where its operand is missing") if value != value else
                 (f"{value:g}", "which is no id"))
    raise ValueError(f"{ids['entry']}: writes {said} into {component} at {where}, {why}; an id is a whole number from "
                     f"0 to {ids['size'] - 1} (the model's {ids['axis']}, {ids['size']}), which the model indexes with")


def apply_bound_ops(new: Tensor, original: Tensor | None, bound_ops: list[dict], got: dict,
                    token_positions: Tensor | None) -> Tensor:
    """`new` after a write's bound ops, each on the current value in turn. An `encode` runs the ops its `decode`
    closes over (`inner`) on the value's coordinates, on the features its `select` keeps, and writes the change
    back (`Kind.within`); `original` goes in with them, encoded the same way (None, where no op takes it)."""
    for op in bound_ops:
        args = {k: live_operand(got, v) if isinstance(v, dict) and "live" in v else v for k, v in op["args"].items()}
        if op["op"] != "encode":
            new = ops.call(op["op"], new, args, original=original, dims=op["dims"], layers=op.get("layers"),
                           rows=op.get("rows"), positions=token_positions)
            continue
        featurizer: Any = args["featurizer"]
        features = op.get("features")
        start = None if original is None else featurizer.encode(original)
        start = start if features is None or start is None else start.index_select(-1, features.to(start.device))
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
        raise RuntimeError(f"operand {operand['live']!r} is a read of this forward the engine had not made when it "
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
    on there (`keep`, and `kept`, those rows by index, as `selection_in_batch` lists them), as
    `positions.lands_at_call` decides from the inputs `seen` so far (one `[rows, 1]` per decode call); one that lands on
    no row is left out. Any other read is as it was."""
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
            keep = torch.tensor(rows)[:, None]
            out.append({**record, "keep": keep, "kept": keep.nonzero().T})
    return out
