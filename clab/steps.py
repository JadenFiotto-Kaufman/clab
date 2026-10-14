"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards, its tokenizer, its
sizes and its readout, and does everything else itself, in plain torch. Under the nnterp engine it runs inside
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
LATER = {"generate": "M3", "fit": "M4", "encode": "M4"}

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


def preflight(engine, plan: Plan) -> dict:
    """Everything that needs the model, checked before any forward. Fills `plan.run.tokens` and
    `plan.run.positions`, and returns each dataset input's encoded rows and padded width (`inputs`) and every
    record's token indices per row (`positions`).

    Refuses by name (`<step>.<record>`): a step kind or op a later milestone serves; a component the engine
    does not serve; a layer or a `select`
    outside the model's sizes, or a `select` on the padded batch's `key` axis; a position that does not fit some row's scope, or whose scope is not in a row
    a write or a rectangular read needs; a token column that is not one token on some row; a write whose
    operand was selected at a different width or narrowing; a write to a component with a `key` axis (the
    attention pattern, the scores) whose operand was read on another input that does not tokenize to the
    same length on every row, which would misalign the keys; and a `pca` over no more vectors than its `k`.
    """
    for name, step in walk(plan.steps):
        if step.kind not in KINDS:
            raise NotImplementedError(f"{name}: {step.kind} steps land in {LATER[step.kind]}")
    refuse(engine, plan)
    assert plan.run is not None
    sizes, tokenizer, special = engine.sizes(), engine.tokenizer, set(engine.tokenizer.all_special_ids)
    found: dict = {"inputs": {}, "positions": {}}
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
                    if axis == "key":
                        raise NotImplementedError(f"{where}: select on 'key' indexes the padded batch's keys, "
                                                  "which differ per row; it lands when positions can name keys")
                    if any(not 0 <= i < sizes[axis] for i in index):
                        raise ValueError(f"{where}: select {axis} {index} outside the model's {sizes[axis]}")
                resolved = indices(where, side == "writes", record, step.input, rows, encoded, special)
                put(found["positions"], where, step.at, resolved)
                shape = (len(resolved[0]), {axis: len(v) for axis, v in (site.select or {}).items()})
                put(shapes, where, step.at, shape)
                for o in record.ops:
                    refuse_later(where, o.op, space(o))
                    check_tokens(engine, plan, where, o.op, arguments(o))
                    if side == "writes":
                        check_operands(plan, where, o, shape, step, site.component, found["inputs"], shapes)
    for where, by_point in found["positions"].items():
        plan.run.positions[where] = by_point[()] if () in by_point else \
            {labelled(plan, dict(coords)): rows for coords, rows in by_point.items()}
    return found


def indices(where: str, write: bool, record, input: str, rows: list[dict], encoded: list[dict],
            special: set) -> list[list[int]]:
    """A record's token indices in each row. A row whose scope is not in its input, or whose ragged window is
    empty, has none on a ragged read and is refused on a write, which needs a token to write (and, for a
    missing scope, on a rectangular read)."""
    pos = positions.parse(record.pos)
    found = []
    for i, (row, enc) in enumerate(zip(rows, encoded)):
        scope = pos["scope"] or {}
        text = data.variable(row, input, scope["variable"]) if "variable" in scope else None
        try:
            got = positions.resolve(pos, enc, special, text)
        except ValueError as e:
            raise ValueError(f"{where}: row {i}: {e}") from None
        if got is None and (write or not positions.ragged(pos)):
            what = f"segment {scope['segment']!r}" if "segment" in scope else \
                f"variable {scope['variable']!r} ({text!r})"
            raise ValueError(f"{where}: row {i}: {what} is not in its input")
        if write and got == []:
            raise ValueError(f"{where}: row {i}: the write's positions hold no token of this row; a write needs "
                             "one to write")
        found.append(got or [])
    return found


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
    encoded, at = found["inputs"][step.input], step.at
    n, layers = len(encoded["rows"]), engine.sizes()["num_layers"]
    every = {key: range(layers) if r.site.layers == "all" else [r.site.layers] for key, r in step.reads.items()}
    parts = []
    for lo in range(0, n, batch_size):
        chunk = slice(lo, min(lo + batch_size, n))
        ids = [e["ids"] for e in encoded["rows"][chunk]]
        batch = data.pad(engine.tokenizer, ids, encoded["width"])
        pads = [encoded["width"] - len(x) for x in ids]
        sites = {key: site(record, lookup(found["positions"], f"{name}.{key}", at)[chunk], pads)
                 for key, record in {**step.writes, **step.reads}.items()}
        writes = [{**sites[key], "name": key, "layer": record.site.layers,
                   "ops": [edit(engine, plan, record, o, state, at, chunk) for o in record.ops]}
                  for key, record in step.writes.items()]
        reads = [{**sites[key], "name": f"{key}@{layer}", "layer": layer}
                 for key in step.reads for layer in every[key]]
        with torch.no_grad():
            got = engine.forward(batch, writes, reads)
        out = {}
        for key, record in step.reads.items():
            values = [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk, pads, state, at)
                      for layer in every[key]]
            if record.site.layers != "all":
                out[key] = values[0]
            elif isinstance(values[0], Tensor):
                out[key] = torch.stack(values)
            else:
                out[key] = {**values[0], "values": torch.stack([v["values"] for v in values])}
        parts.append(out)
    for key, record in step.reads.items():
        axis = 1 if record.site.layers == "all" else 0
        values = [p[key] for p in parts]
        put(state, f"{name}.{key}", at, torch.cat(values, axis) if isinstance(values[0], Tensor) else
            {k: torch.cat([v[k] for v in values], axis if k == "values" else 0) for k in values[0]})


@kind
def op(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The op over the whole value `of` names. `top_k`'s value also holds its tokens, decoded."""
    args = bound(engine, plan, step.op, arguments(step), state, step.at, None)
    value = apply(engine, step.op, lookup(state, step.of, step.at), args, dims_of(plan, step.of))
    if step.op == "top_k":
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)


# -- binding a record to one batch ------------------------------------------------------------------------


def site(record, rows: list[list[int]], pads: list[int]) -> dict:
    """What the engine needs to reach a record's selection in this batch: its component, its token indices
    into the padded batch (`positions`, `[rows, width]`; a ragged row repeats its last index to the batch's
    widest, `keep` marking its own), and the part of the feature axes it keeps (`select`)."""
    width = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (width - len(row))
             for row, pad in zip(rows, pads)]
    layout = address.dims(record.site.component, None, False)
    select = {layout.index(axis): torch.tensor(v) for axis, v in (record.site.select or {}).items()}
    return {"component": record.site.component, "positions": torch.tensor(index), "select": select,
            "keep": torch.tensor([[j < len(row) for j in range(width)] for row in rows])}


def edit(engine, plan: Plan, record, o, state: dict, at: dict, rows: slice) -> dict:
    """A write's op bound to one batch, as `ops.write` runs it."""
    return {"op": o.op, "featurizer": space(o), "args": bound(engine, plan, o.op, arguments(o), state, at, rows),
            "dims": address.dims(record.site.component, None, False)}


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, pads: list[int], state: dict,
         at: dict) -> Any:
    """One layer of a read in one batch: its entries when it is ragged, then its ops."""
    ragged = positions.ragged(positions.parse(record.pos))
    if ragged:
        keep = site["keep"]
        value = {"values": value[keep], "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep],
                 "tok": (site["positions"] - torch.tensor(pads)[:, None])[keep]}
    dims = address.dims(record.site.component, None, ragged)
    for o in record.ops:
        value = apply(engine, o.op, value, bound(engine, plan, o.op, arguments(o), state, at, rows), dims)
        dims = ops.out_dims(o.op, dims, arguments(o))
    return value


def bound(engine, plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None) -> dict:
    """An op's arguments bound to tensors: names looked up, columns tokenized, sliced to `rows`."""
    params, out = ops.parameters(name), {}
    for key, value in args.items():
        annotation = params[key].annotation
        if annotation == ops.Value and isinstance(value, str):
            given, value = value, lookup(state, value, at)
            if rows is not None and dims_of(plan, given)[:1] == ["rows"]:
                value = value[rows]
        elif annotation == ops.Tokens:
            value = tokens(engine, plan, value, args.get("token_form", "spaced"), rows)
        elif annotation == ops.Featurizer:
            raise NotImplementedError(f"M4: featurizer {value!r}")
        out[key] = value
    return out


def apply(engine, name: str, value, args: dict, dims: list[str]):
    """Op `name` on a value; on a ragged value, on its entries, which stay ragged while the op keeps them."""
    if isinstance(value, dict) and "tok" in value:
        got = ops.call(name, value["values"], args, engine=engine, dims=dims)
        return {**value, "values": got} if "entries" in ops.out_dims(name, dims, args) else got
    return ops.call(name, value, args, engine=engine, dims=dims)


def tokens(engine, plan: Plan, column: str, form: str, rows: slice | None) -> Tensor:
    name, _, col = column.partition(".")
    texts = [r[col] for r in plan.data[name].rows or []]
    return data.token_ids(engine.tokenizer, texts[rows] if rows is not None else texts, form, column)


def decoded(tokenizer, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are."""
    return [tokenizer.decode([i]) for i in ids.tolist()] if ids.dim() == 1 else [decoded(tokenizer, x) for x in ids]


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
