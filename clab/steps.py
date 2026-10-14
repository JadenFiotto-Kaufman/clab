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
from .compile import dims_of, points

# The step kinds later milestones serve, refused before anything runs.
LATER = {"generate": "M3", "fit": "M4"}


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
    """The checks that need the model, before any forward. Returns each dataset input's padded width.

    Refuses by name: a step kind a later milestone serves; a position that does
    not fit some row's tokens; a token column that is not one token on some row.
    """
    for name, entry in plan["steps"].items():
        kind = points(entry)[0]["kind"]
        if kind not in KINDS:
            raise NotImplementedError(f"step {name!r} is a {kind} step; {kind} steps are served from {LATER[kind]}")
    tokenizer, widths = engine.tokenizer, {}
    for name, entry in plan["steps"].items():
        for step in points(entry):
            if step["kind"] == "op":
                params = ops.parameters(step["op"])
                for key, value in step["args"].items():
                    if key in params and params[key].annotation == ops.Tokens:
                        tokens(engine, plan, value, step["args"].get("token_form", "spaced"), None)
                continue
            inp = step["input"]
            texts = [data.field(r, inp) for r in plan["data"][inp["data"]]["rows"]]
            try:
                width = widths.setdefault(data.key(inp), max(data.lengths(tokenizer, texts)))
                batch = data.encode(tokenizer, texts, width)
            except NotImplementedError as e:
                raise NotImplementedError(f"step {name!r}: {e}") from None
            for record in step["writes"] + step["reads"]:
                try:
                    if record["layer"] == "all":
                        raise NotImplementedError("M2: layers 'all'")
                    positions.resolve(record["pos"], batch)
                except (ValueError, NotImplementedError) as e:
                    raise type(e)(f"step {name!r}, {record['name']!r}: {e}") from None
    return widths


# -- the kinds ------------------------------------------------------------------


def forward(engine, plan: dict, name: str, step: dict, state: dict, batch_size: int, widths: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each read joined along rows."""
    inp, at = step["input"], step["at"]
    rows = plan["data"][inp["data"]]["rows"]
    parts = []
    for lo in range(0, len(rows), batch_size):
        chunk = slice(lo, lo + batch_size)
        batch = data.encode(engine.tokenizer, [data.field(r, inp) for r in rows[chunk]], widths[data.key(inp)])
        writes = [{**site(r, batch), "ops": [bound(engine, plan, o, state, at, chunk) for o in r["ops"]],
                   "dims": r["dims"]} for r in step["writes"]]
        reads = [site(r, batch) for r in step["reads"]]
        with torch.no_grad():
            got = engine.forward(batch, writes, reads)
        for read in step["reads"]:
            dims = address.dims(read["component"], read["layer"], positions.ragged(read["pos"]))
            for o in read["ops"]:
                args = bound(engine, plan, o, state, at, chunk)["args"]
                got[read["name"]] = ops.call(o["op"], got[read["name"]], args, engine=engine, dims=dims)
                dims = ops.out_dims(o["op"], dims, o["args"])
        parts.append(got)
    for read in step["reads"]:
        put(state, f"{name}.{read['name']}", at, torch.cat([p[read["name"]] for p in parts]))


def op(engine, plan: dict, name: str, step: dict, state: dict, batch_size: int, widths: dict) -> None:
    """The op over the whole value `of` names."""
    args = dict(step["args"])
    of = args.pop("of")
    record = bound(engine, plan, {"op": step["op"], "args": args, "featurizer": None}, state, step["at"], None)
    put(state, name, step["at"], ops.call(step["op"], lookup(state, of, step["at"]), record["args"], engine=engine,
                                          dims=dims_of(plan, of)))


KINDS = {"forward": forward, "op": op}


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
