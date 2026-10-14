"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp engine it runs inside
one session; under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
"""

import importlib.metadata
import platform
import time
from typing import Any

import torch
from torch import Tensor

from . import address, data, ops, positions
from .plan import Plan, Run, arguments, coordinate, dims_of, labelled, points, space
from .registry import registers

# The step kinds and ops later milestones serve, refused before anything runs.
LATER = {"fit": "M4", "encode": "M4"}

KINDS: dict = {}
kind = registers(KINDS)


def begin(engine, plan: Plan, batch_size: int, remote: bool | str) -> float:
    """What every engine's `execute` checks first, and the run record it starts; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    plan.run = Run(engine=engine.name, remote=remote, batch_size=batch_size,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def run(engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order; then each saved value into `plan.results`."""
    assert plan.run is not None
    plan.run.versions = versions()
    found = preflight(engine, plan)
    state: dict = {}
    for name, entry in plan.steps.items():
        for step in points(entry):
            KINDS[step.kind](engine, plan, name, step, state, batch_size, found)
    for name in plan.saves:
        entry = plan.steps[name.partition(".")[0]]
        plan.results[name] = ({coordinate(plan, p.at): finished(lookup(state, name, p.at)) for p in entry}
                              if isinstance(entry, list) else finished(lookup(state, name, {})))
    plan.run.positions = {where: by_coordinate(plan, by_point) for where, by_point in found["positions"].items()}
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found["said"].items()})


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def preflight(engine, plan: Plan) -> dict:
    """Everything that needs the model, checked before any forward. Fills `plan.run.tokens` with each input's
    tokens, and returns each dataset input's encoded rows and padded width (`inputs`), every prefill
    record's positions per row (`positions`, which the run extends with the other records'), and
    room for what each generate step's rows said (`said`).

    Refuses by name (`<step>.<record>`): a step kind or op a later milestone serves; a component the engine
    does not serve; a layer or a `select` outside the model's sizes (on `key`, outside the input's shortest
    row); a generate option its `generate` does not take, or one in `FIXED` at another value; a
    position that does not fit some row's scope, or whose scope is not in a row a write or a rectangular
    read needs; a token column that is not one token on some row; a write whose operand was selected at a
    different width or narrowing; a write to a component with a `key` axis (the attention pattern, the
    scores) whose operand was read on another input that does not tokenize to the same length on every row,
    which would misalign the keys; and a `pca` over no more vectors than its `k`.
    """
    for name, step in walk(plan.steps):
        if step.kind not in KINDS:
            raise NotImplementedError(f"{name}: {step.kind} steps land in {LATER[step.kind]}")
    refuse(engine, plan)
    assert plan.run is not None
    sizes, tokenizer, special = engine.sizes(), engine.tokenizer, set(engine.tokenizer.all_special_ids)
    found: dict = {"inputs": {}, "positions": {}, "said": {}}
    shapes: dict = {}
    for name, step in walk(plan.steps):
        if step.kind == "op":
            refuse_later(name, step.op, None)
            check_tokens(engine, plan, name, step.op, arguments(step))
            if step.op == "pca" and step.of in found["positions"]:
                n = sum(map(len, lookup(found["positions"], step.of, step.at)))
                if n <= arguments(step)["k"]:
                    raise ValueError(f"{name}: pca of {n} vectors has no rank-{arguments(step)['k']} basis; "
                                     f"it needs more than {arguments(step)['k']}")
            continue
        if step.kind == "generate":
            generation_options(name, step.model_extra or {})
        rows = plan.data[step.dataset].rows or []
        if step.input not in found["inputs"]:
            encoded = []
            for i, r in enumerate(rows):
                try:
                    encoded.append(data.encode(tokenizer, data.field(r, step.input)))
                except ValueError as e:
                    raise ValueError(f"{name}: {step.input} row {i}: {e}") from None
            found["inputs"][step.input] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
            plan.run.tokens[step.input] = [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
        encoded = found["inputs"][step.input]["rows"]
        for side in ("writes", "reads"):
            for key, record in getattr(step, side).items():
                where, site = f"{name}.{key}", record.site
                if isinstance(site.layers, int) and not 0 <= site.layers < sizes["num_layers"]:
                    raise ValueError(f"{where}: layer {site.layers}; the model has {sizes['num_layers']} layers")
                for axis, index in (site.select or {}).items():
                    shortest = min(len(e["ids"]) for e in encoded)
                    size, what = (shortest, f"the {shortest} tokens of the input's shortest row") if axis == "key" \
                        else (sizes[axis], f"the model's {sizes[axis]}")
                    if any(not 0 <= i < size for i in index):
                        raise ValueError(f"{where}: select {axis} {index} outside {what}")
                for o in record.ops:
                    refuse_later(where, o.op, space(o))
                    check_tokens(engine, plan, where, o.op, arguments(o))
                if not positions.prefill(positions.parse(record.pos)):
                    continue        # resolved from what each row generates, in the run
                resolved = indices(where, side == "writes", record, step, rows, encoded, tokenizer,
                                   head(step, record))
                put(found["positions"], where, step.at, resolved)
                shape = (len(resolved[0]), {axis: len(v) for axis, v in (site.select or {}).items()})
                put(shapes, where, step.at, shape)
                for o in record.ops if side == "writes" else []:
                    check_operands(plan, where, o, shape, step, site.component, found["inputs"], shapes)
    return found


# The generate options a step cannot forward as they are, the one value each may have, and why.
FIXED = {
    "num_return_sequences": (1, "more sequences than rows would break the step's one row of ids per input row"),
    "num_beams": (1, "beams change the rows each forward call sees, so a read's row is no longer the input's"),
    "return_dict_in_generate": (False, "the step's value is the ids"),
    "use_cache": (True, "without the cache a decode step recomputes the prompt, and the prompt-call writes "
                        "would not reach it"),
}


def generation_options(name: str, options: dict) -> None:
    """Refuses an option in `FIXED` at another value. Every other option is the engine's `generate`'s to
    accept or reject, when the step runs (`generated`)."""
    for key, value in options.items():
        if key in FIXED and value != FIXED[key][0]:
            raise ValueError(f"{name}: {key} {value!r}; it stays {FIXED[key][0]!r}: {FIXED[key][1]}")


def stops(engine, step) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def indices(where: str, write: bool, record, step, rows: list[dict], encoded: list[dict], tokenizer,
            head_only: bool) -> list[list[int]]:
    """A prefill record's positions in each row, which the data alone decides: a scope text that is absent or
    doubled, or a rectangular index that does not fit, is refused by name, and so is a write whose positions
    hold no token of a row."""
    pos, special = positions.parse(record.pos), set(tokenizer.all_special_ids)
    found = []
    for i, (row, enc) in enumerate(zip(rows, encoded)):
        try:
            got = positions.resolve(pos, enc, None, set(), special, tokenizer, text(pos, row), head_only)
        except ValueError as e:
            raise ValueError(f"{where}: row {i}: {e}") from None
        if write and not got:
            raise ValueError(f"{where}: row {i}: the write's positions hold no token of this row; a write needs "
                             "one to write")
        found.append(list(got))
    return found


def text(pos: dict, row: dict) -> str | None:
    """The text a record's scope names in one row: its column's (`<dataset>.<column path>`), or the literal."""
    scope = pos["scope"] or {}
    if "column" in scope:
        return str(data.lookup(row, scope["column"].partition(".")[2]))
    return scope.get("text")


def head(step, record) -> bool:
    """Whether the record reads the head in a generation, whose prefill computes it at its last position only."""
    return step.kind == "generate" and record.site.component in ("lm_head_output", "logits")


def refuse(engine, plan: Plan) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name. `engine.unserved()`
    maps a component to None (served), why not, or why not per layer (nnterp's `support()` shape)."""
    reasons = engine.unserved()
    for name, step in walk(plan.steps):
        for key, record in {**getattr(step, "writes", {}), **getattr(step, "reads", {})}.items():
            reason, layers = reasons.get(record.site.component), record.site.layers
            if isinstance(reason, dict):
                reason = next((r for layer, r in reason.items() if r and layers in (layer, "all")), None)
            if reason:
                raise ValueError(f"{name}.{key}: the {engine.name} engine does not serve {record.site.component}: "
                                 f"{reason}")


def walk(steps: dict, prefix: str = ""):
    """Every step and point, a fit's body steps after the fit, each with its dotted name."""
    for name, entry in steps.items():
        for step in points(entry):
            yield prefix + name, step
            if step.kind == "fit":
                yield from walk(step.steps, f"{prefix}{name}.")


def refuse_later(where: str, op: str, featurizer: str | None) -> None:
    if op in LATER:
        raise NotImplementedError(f"{where}: op {op!r} lands in {LATER[op]}")
    if featurizer is not None:
        raise NotImplementedError(f"{where}: op {op!r} in featurizer {featurizer!r}'s space lands in M4")


def check_tokens(engine, plan: Plan, where: str, name: str, args: dict) -> None:
    for key, p in ops.parameters(name).items():
        if p.annotation == ops.Tokens and key in args:
            try:
                tokens(engine, plan, args[key], args.get("token_form", "spaced"), None)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None


def check_operands(plan: Plan, where: str, o, shape: tuple, step, component: str, inputs: dict, shapes: dict) -> None:
    """A write's operands that another record selected: the same width and narrowing, and, on a `key` axis,
    inputs that line up token for token."""
    for key, p in ops.parameters(o.op).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        if theirs != shape:
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer = points(plan.steps[value.partition(".")[0]])[0].input
        if "key" in address.COMPONENTS[component]["dims"] and producer != step.input:
            for row, (a, b) in enumerate(zip(inputs[producer]["rows"], inputs[step.input]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {step.input} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")


# -- the kinds ------------------------------------------------------------------------------------------


@kind
def forward(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each read joined along rows.

    A `layers: "all"` read is one read per layer, its ops run on each, stacked layer-first; a ragged read is
    gathered at the batch's widest row and kept as `{values, row, tok}`: one entry per token, its row and its
    token index."""
    parts = []
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        sites, writes, reads = bind(engine, plan, name, step, step.reads, state, found, chunk, pads)
        with torch.no_grad():
            got = engine.forward(batch, writes, reads)
        parts.append({key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk,
                                                 pads, state, step.at) for layer in layers(engine, record)])
                      for key, record in step.reads.items()})
    joined(state, name, step, parts)


@kind
def generate(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.generate` with the step's own keys as
    its options; each read joined along rows, and the new ids joined into the step's value.

    A record in the prefill (`frame: 0`) is a forward's. One beyond it goes to the engine in two parts: its
    prefill positions (for a read, the whole prefill, since which of its positions it means can depend on
    what the row says), and its decode calls, each at that call's one position (`calls`: the frame's own,
    or None for every decode call). After the generation each row resolves from what it generated
    (`positions.resolve`); a row that stopped before a call a write selects is not written there (the call
    computes nothing of that row's) and is recorded as missing. The resolution is recorded per row in
    `plan.run.positions` and each row's continuation, as token strings, in `plan.run.tokens[<step>]`. A
    batch whose rows all stopped early makes fewer calls; its ids are filled after their EOS to the longest
    batch's."""
    eos, rows, special = stops(engine, step), plan.data[step.dataset].rows or [], set(engine.tokenizer.all_special_ids)
    encoded = found["inputs"][step.input]["rows"]
    first = {k: r for k, r in step.reads.items() if positions.prefill(positions.parse(r.pos))}
    later = {k: r for k, r in step.reads.items() if k not in first}
    beyond = {k: w for k, w in step.writes.items() if not positions.prefill(positions.parse(w.pos))}
    parts, new, resolved = [], [], {f"{name}.{key}": [] for key in {**later, **beyond}}
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        n = chunk.stop - chunk.start
        sites, writes, reads = bind(engine, plan, name, step, first, state, found, chunk, pads, set(beyond))
        for key, record in beyond.items():
            pos, edits = positions.parse(record.pos), [edit(engine, plan, record, o, state, step.at, chunk)
                                                        for o in record.ops]
            if positions.with_prefill(pos):
                where = [positions.held([0], e, [], special, head(step, record)) for e in encoded[chunk]]
                writes.append({**site(record, where, pads, head(step, record)), "name": f"{key}#0",
                               "layer": record.site.layers, "ops": edits})
            writes.append({**decoding(record, n, pads), "name": key, "layer": record.site.layers, "ops": edits,
                           "calls": positions.calls(pos)})
        for key, record in later.items():
            pos = positions.parse(record.pos)
            for layer in layers(engine, record):
                if positions.with_prefill(pos):
                    whole = [list(range(len(e["ids"]))) for e in encoded[chunk]]
                    reads.append({**site(record, whole, pads, head(step, record)), "name": f"{key}@{layer}#0",
                                  "layer": layer})
                reads.append({**decoding(record, n, pads), "name": f"{key}@{layer}", "layer": layer,
                              "calls": positions.calls(pos)})
        with torch.no_grad():
            ids, got = generated(engine, name, batch, writes, reads, dict(step.model_extra or {}))
        for record in reads:
            if "calls" in record:
                made = range(1, ids.shape[1]) if record["calls"] is None else \
                    [k for k in record["calls"] if k < ids.shape[1]]
                if len(got[record["name"]]) != len(made):
                    raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got[record['name']])} of the "
                                       f"{len(made)} decode calls it made; a read out of forward order shifts calls")
        out = {key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk, pads,
                                          state, step.at) for layer in layers(engine, record)])
               for key, record in first.items()}
        for key, record in {**later, **beyond}.items():
            pos = positions.parse(record.pos)
            where = [positions.resolve(pos, e, ids[i].tolist(), set(eos), special, engine.tokenizer, text(pos, row),
                                       head(step, record))
                     for i, (row, e) in enumerate(zip(rows[chunk], encoded[chunk]))]
            resolved[f"{name}.{key}"] += where
            if key in later:
                out[key] = stacked(record, [at_calls(engine, plan, record, got.get(f"{key}@{layer}#0"),
                                                     got[f"{key}@{layer}"], where, encoded[chunk], pads, chunk,
                                                     state, step.at, head(step, record))
                                            for layer in layers(engine, record)])
        parts.append(out)
        new.append(ids)
    joined(state, name, step, parts)
    width = max(x.shape[1] for x in new)
    ids = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=eos[0]) for x in new])
    put(state, name, step.at, ids)
    for where, per_row in resolved.items():
        put(found["positions"], where, step.at, per_row)
    put(found["said"], name, step.at,
        [engine.tokenizer.convert_ids_to_tokens(positions.continuation(x, set(eos))) for x in ids.tolist()])


@kind
def op(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The op over the whole value `of` names. `top_k`'s value also holds its tokens, decoded."""
    of = lookup(state, step.of, step.at)
    producer = points(plan.steps[step.of.partition(".")[0]])[0]
    value = apply(engine, step.op, of, bound(engine, plan, step.op, arguments(step), state, step.at, None, of),
                  dims_of(plan, step.of), stops(engine, producer) if producer.kind == "generate" else engine.eos)
    if step.op == "top_k":
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)


# -- binding a record to one batch ------------------------------------------------------------------------


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


def edit(engine, plan: Plan, record, o, state: dict, at: dict, rows: slice) -> dict:
    """A write's op bound to one batch, as `ops.write` runs it."""
    return {"op": o.op, "featurizer": space(o), "args": bound(engine, plan, o.op, arguments(o), state, at, rows),
            "dims": address.dims(record.site.component, None, False)}


def generated(engine, name: str, batch: dict, writes: list[dict], reads: list[dict], options: dict):
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`."""
    try:
        return engine.generate(batch, writes, reads, options)
    except (TypeError, ValueError) as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None


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
    writes = [{**sites[key], "name": key, "layer": record.site.layers,
               "ops": [edit(engine, plan, record, o, state, step.at, rows) for o in record.ops]}
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
    if positions.ragged(positions.parse(record.pos)):
        keep = site["keep"]
        value = {"values": value[keep], "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep],
                 "tok": site["tok"][keep]}
    return with_ops(engine, plan, record, value, state, at, rows)


def at_calls(engine, plan: Plan, record, whole: Tensor | None, values: list[Tensor], where: list, encoded: list[dict],
             pads: list[int], rows: slice, state: dict, at: dict, head_only: bool) -> Any:
    """One layer of a read beyond the prefill in one batch, arranged by the positions each row resolved to
    (`where`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (`whole`, `[rows, width, ...]`, column `p` the row's token `p`; its one column under
    `head_only`), a generated input's from
    its decode call (`values`, one `[rows, 1, ...]` per call the record asked for that the batch made); then
    its ops. A rectangular read is `[rows, width, ...]`, NaN on a row it does not reach; a ragged one
    `{values, row, tok}`, `tok` the position, with no entries for such a row."""
    pos = positions.parse(record.pos)
    asked = positions.calls(pos)
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    sizes, select = engine.sizes(), record.site.select or {}
    rest = [len(select[a]) if a in select else sizes[a] for a in address.dims(record.site.component, None, False)[2:]]
    some = whole if whole is not None else values[0] if values else None
    dtype = some.dtype if some is not None else getattr(torch, plan.model.dtype)

    def at_position(i: int, p: int) -> Tensor:
        prompt = len(encoded[i]["ids"])
        if p < prompt:
            assert whole is not None
            return whole[i, 0 if head_only else p]
        return by_call[p - prompt + 1][i, 0]

    if not positions.ragged(pos):
        blank = torch.full((positions.width(pos), *rest), float("nan"), dtype=dtype)
        value = torch.stack([blank if isinstance(got, str) else torch.stack([at_position(i, p) for p in got])
                             for i, got in enumerate(where)])
    else:
        kept = [(i, p) for i, got in enumerate(where) if not isinstance(got, str) for p in got]
        value = {"values": torch.stack([at_position(i, p) for i, p in kept]) if kept else
                 torch.zeros((0, *rest), dtype=dtype),
                 "row": torch.tensor([rows.start + i for i, _ in kept], dtype=torch.long),
                 "tok": torch.tensor([p for _, p in kept], dtype=torch.long)}
    return with_ops(engine, plan, record, value, state, at, rows)


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice) -> Any:
    """A read's value with its ops applied in listed order."""
    dims = address.dims(record.site.component, None, positions.ragged(positions.parse(record.pos)))
    for o in record.ops:
        value = apply(engine, o.op, value, bound(engine, plan, o.op, arguments(o), state, at, rows, value), dims)
        dims = ops.out_dims(o.op, dims, arguments(o))
    return value


def bound(engine, plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None,
          of: Any = None) -> dict:
    """An op's arguments bound to tensors: names looked up, columns tokenized, sliced to `rows`. On a ragged
    value `of`, an argument with a row per example is taken at each entry's row."""
    params, out = ops.parameters(name), {}
    for key, value in args.items():
        annotation, per_row = params[key].annotation, False
        if annotation == ops.Value and isinstance(value, str):
            given, value = value, lookup(state, value, at)
            per_row = dims_of(plan, given)[:1] == ["rows"]
            value = value[rows] if rows is not None and per_row else value
        elif annotation == ops.Tokens:
            value, per_row = tokens(engine, plan, value, args.get("token_form", "spaced"), rows), True
        elif annotation == ops.Featurizer:
            raise NotImplementedError(f"M4: featurizer {value!r}")
        if per_row and isinstance(of, dict) and "tok" in of:
            value = value[of["row"] - (rows.start if rows is not None else 0)]
        out[key] = value
    return out


def apply(engine, name: str, value, args: dict, dims: list[str], eos: list[int] | None = None):
    """Op `name` on a value; on a ragged value, on its entries, which stay ragged while the op keeps them."""
    if isinstance(value, dict) and "tok" in value:
        got = ops.call(name, value["values"], args, engine=engine, dims=dims, eos=eos)
        return {**value, "values": got} if "entries" in ops.out_dims(name, dims, args) else got
    return ops.call(name, value, args, engine=engine, dims=dims, eos=eos)


def tokens(engine, plan: Plan, column: str, form: str, rows: slice | None) -> Tensor:
    name, _, col = column.partition(".")
    texts = [r[col] for r in plan.data[name].rows or []]
    return data.token_ids(engine.tokenizer, texts[rows] if rows is not None else texts, form, column)


def decoded(tokenizer, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are; None for a missing one (-1)."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    return [tokenizer.decode([i]) if i >= 0 else None for i in ids.tolist()]


# -- state ------------------------------------------------------------------------------------------------


def put(state: dict, name: str, at: dict, value) -> None:
    state.setdefault(name, {})[tuple(at.items())] = value


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def finished(value):
    """Detached CPU clones all the way down; a result never holds a view of a larger buffer."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: finished(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finished(v) for v in value]
    return value


def versions() -> dict[str, str]:
    """The installed versions where the model is."""
    found = {p: importlib.metadata.version(p) for p in ("clab", "nnsight", "nnterp", "transformers", "torch")}
    return {**found, "python": platform.python_version()}
