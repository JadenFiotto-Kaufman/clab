"""Walk a plan inside the run: preflight, then every step and point in order, filling result slots.

This assumes only the engine protocol (`engines.Engine`): it asks the engine
for forwards, its tokenizer and its readout, and does everything else itself,
in plain torch. Under the nnterp engine it runs inside one session; under the
hooks engine, in plain Python. The cross-step state is a dict that lives in
`run`'s frame and never leaves the run: each value under its name, by the
coordinates of the point that produced it.
"""

import torch

from . import address, data, ops, positions
from .compiler import dims_of, points
from .registry import registers

# The step kinds and ops later milestones serve, refused before anything runs.
LATER = {"generate": "M3", "fit": "M4", "readout": "M2", "pca": "M2", "top_k": "M2", "encode": "M4"}

KINDS: dict = {}
kind = registers(KINDS)


def run(engine, plan: dict, batch_size: int) -> None:
    """Every step and point in document order; each result slot filled when its step finishes."""
    widths = preflight(engine, plan)
    state: dict = {}
    for name, entry in plan["steps"].items():
        for step in points(entry):
            KINDS[step["kind"]](engine, plan, name, step, state, batch_size, widths)
            for key, slot in step["result"].items():
                slot["value"] = finished(lookup(state, key, step["at"]))


def preflight(engine, plan: dict) -> dict[str, int]:
    """Everything that needs the model, checked before any forward. Returns each dataset input's padded width.

    Refuses by name (`<step>.<record>`): a step kind or op a later milestone serves; a layer or narrowing
    outside the model's sizes; a position that does not fit some row's tokens; a token column that is not
    one token on some row; a write whose operand was selected at a different width or narrowing; and a write
    to a component with a `key` axis (the attention pattern, the scores) whose operand was read on another
    input that does not tokenize to the same length on every row, which would misalign the keys.
    """
    for name, step in walk(plan["steps"]):
        if step["kind"] not in KINDS:
            raise NotImplementedError(f"{name}: {step['kind']} steps land in {LATER[step['kind']]}")
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    for name, step in walk(plan["steps"]):
        if step["kind"] == "op" and step["op"] in LATER:
            raise NotImplementedError(f"{name}: op {step['op']!r} lands in {LATER[step['op']]}")
        for record in step.get("writes", []) + step.get("reads", []):
            where = f"{name}.{record['name']}"
            if record["layer"] == "all":
                raise NotImplementedError(f"{where}: layers 'all' lands in M2")
            if record["layer"] is not None and not 0 <= record["layer"] < sizes["num_layers"]:
                raise ValueError(f"{where}: layer {record['layer']}; the model has {sizes['num_layers']} layers")
            for key, index in record["select"].items():
                axis = address.COMPONENTS[record["component"]]["narrow"][key]
                if any(not 0 <= i < sizes[axis] for i in index):
                    raise ValueError(f"{where}: {key} {index} outside the model's {sizes[axis]} ({axis})")
            for o in record["ops"]:
                if o["op"] in LATER:
                    raise NotImplementedError(f"{where}: op {o['op']!r} lands in {LATER[o['op']]}")
                if o["featurizer"] is not None:
                    raise NotImplementedError(f"{where}: op {o['op']!r} in featurizer {o['featurizer']!r}'s space "
                                              "lands in M4")

    widths: dict[str, int] = {}
    lengths: dict[str, list[int]] = {}
    shapes: dict = {}
    for name, step in walk(plan["steps"]):
        if step["kind"] == "op":
            check_tokens(engine, plan, name, step["op"], step["args"])
            continue
        inp, at = step["input"], step["at"]
        texts = [data.field(r, inp) for r in plan["data"][inp["data"]]["rows"]]
        try:
            lengths[data.key(inp)] = data.lengths(tokenizer, texts)
        except NotImplementedError as e:
            raise NotImplementedError(f"{name}: {e}") from None
        widths[data.key(inp)] = max(lengths[data.key(inp)])
        batch = data.encode(tokenizer, texts, widths[data.key(inp)])
        for side in ("reads", "writes"):
            for record in step[side]:
                where = f"{name}.{record['name']}"
                try:
                    shape = (positions.resolve(record["pos"], batch).shape[1], [len(v) for v in record["select"].values()])
                except (ValueError, NotImplementedError) as e:
                    raise type(e)(f"{where}: {e}") from None
                put(shapes, where, at, shape)
                for o in record["ops"]:
                    check_tokens(engine, plan, where, o["op"], o["args"])
                    if side == "writes":
                        check_operands(plan, where, o, shape, inp, lengths, shapes, at, record["component"])
    return widths


def walk(steps: dict, prefix: str = ""):
    """Every step and point, a fit's body steps after the fit, each with its dotted name."""
    for name, entry in steps.items():
        for step in points(entry):
            yield prefix + name, step
            if step["kind"] == "fit":
                yield from walk(step["steps"], f"{prefix}{name}.")


def check_tokens(engine, plan: dict, where: str, name: str, args: dict) -> None:
    params = ops.parameters(name)
    for key, value in args.items():
        if key in params and params[key].annotation == ops.Tokens:
            try:
                tokens(engine, plan, value, args.get("token_form", "spaced"), None)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None


def check_operands(plan: dict, where: str, o: dict, shape: tuple, inp: dict, lengths: dict, shapes: dict, at: dict,
                   component: str) -> None:
    """A write's operands that another record selected: the same width and narrowing, and, on a `key` axis,
    inputs that line up token for token."""
    params = ops.parameters(o["op"])
    for key, value in o["args"].items():
        if params[key].annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, at)
        if theirs != shape:
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer = points(plan["steps"][value.partition(".")[0]])[0]["input"]
        if "key" in address.COMPONENTS[component]["dims"] and data.key(producer) != data.key(inp):
            for row, (a, b) in enumerate(zip(lengths[data.key(producer)], lengths[data.key(inp)])):
                if a != b:
                    raise ValueError(f"{where}: operand {value!r} was read on {data.key(producer)}, row {row} of "
                                     f"which is {a} tokens; {data.key(inp)} has {b} there, so their key axes "
                                     "do not line up")


# -- the kinds ------------------------------------------------------------------


@kind
def forward(engine, plan: dict, name: str, step: dict, state: dict, batch_size: int, widths: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each read joined along rows."""
    inp, at = step["input"], step["at"]
    rows = plan["data"][inp["data"]]["rows"]
    parts = []
    for lo in range(0, len(rows), batch_size):
        chunk = slice(lo, lo + batch_size)
        batch = data.encode(engine.tokenizer, [data.field(r, inp) for r in rows[chunk]], widths[data.key(inp)])
        writes = [{**site(r, batch), "ops": [bound(engine, plan, o, state, at, chunk) for o in r["ops"]]}
                  for r in step["writes"]]
        reads = [site(r, batch) for r in step["reads"]]
        with torch.no_grad():
            got = engine.forward(batch, writes, reads)
        for read in step["reads"]:
            for o in read["ops"]:
                args = bound(engine, plan, o, state, at, chunk)["args"]
                got[read["name"]] = ops.call(o["op"], got[read["name"]], args, engine=engine, dims=o["dims"])
        parts.append(got)
    for read in step["reads"]:
        put(state, f"{name}.{read['name']}", at, torch.cat([p[read["name"]] for p in parts]))


@kind
def op(engine, plan: dict, name: str, step: dict, state: dict, batch_size: int, widths: dict) -> None:
    """The op over the whole value `of` names."""
    args = dict(step["args"])
    of = args.pop("of")
    record = bound(engine, plan, {"op": step["op"], "args": args, "featurizer": None}, state, step["at"], None)
    put(state, name, step["at"], ops.call(step["op"], lookup(state, of, step["at"]), record["args"], engine=engine,
                                          dims=dims_of(plan, of)))



# -- binding a record to one batch ----------------------------------------------


def site(record: dict, batch: dict) -> dict:
    """What the engine needs to reach a write's or read's selection in this batch: where, which tokens, which part."""
    layout = address.dims(record["component"], None, False)
    narrow = address.COMPONENTS[record["component"]]["narrow"]
    return {"name": record["name"], "component": record["component"], "layer": record["layer"],
            "positions": positions.resolve(record["pos"], batch),
            "select": {layout.index(narrow[k]): torch.tensor(v) for k, v in record["select"].items()}}


def bound(engine, plan: dict, record: dict, state: dict, at: dict, rows: slice | None) -> dict:
    """An op record with its arguments bound to tensors: names looked up, columns tokenized, sliced to `rows`."""
    params = ops.parameters(record["op"])
    args = {}
    for key, value in record["args"].items():
        annotation = params[key].annotation
        if annotation == ops.Value and isinstance(value, str):
            value = lookup(state, value, at)
            if rows is not None and dims_of(plan, record["args"][key])[:1] == ["rows"]:
                value = value[rows]
        elif annotation == ops.Tokens:
            value = tokens(engine, plan, value, record["args"].get("token_form", "spaced"), rows)
        elif annotation == ops.Featurizer:
            raise NotImplementedError(f"M4: featurizer {value!r}")
        args[key] = value
    return {**record, "args": args}


def tokens(engine, plan: dict, column: str, form: str, rows: slice | None) -> torch.Tensor:
    name, _, col = column.partition(".")
    texts = [r[col] for r in plan["data"][name]["rows"]]
    return data.token_ids(engine.tokenizer, texts[rows] if rows is not None else texts, form, column)


# -- state ----------------------------------------------------------------------


def put(state: dict, name: str, at: dict, value) -> None:
    state.setdefault(name, {})[tuple(sorted(at.items()))] = value


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def finished(value):
    """Detached CPU clones all the way down; a slot never holds a view of a larger buffer."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: finished(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finished(v) for v in value]
    return value
