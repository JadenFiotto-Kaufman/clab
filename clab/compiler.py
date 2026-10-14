"""document -> plan: pure data, every name checked, every sweep expanded.

Compile knows the document and the files; the run knows the model. Anything
decided from the document, the data files and bundle headers is checked here;
anything that needs the tokenizer or a size of the model is checked by
`steps.preflight`, inside the run.

A plan is nested dicts, lists, strings and numbers (plus the tensors of a
bundle a document names by `file_path`):

- `format`, `document` (`name`, `path`, `sha256`), `description`, `model`;
- `data`: each dataset's `path`, `split`, the `columns` the document uses and
  its `rows`, pruned to those columns, because nothing else reaches the run;
- `featurizers`: each declared featurizer's fields, with its `bundle`;
- `sweeps`: each axis by label, `{"path", "values"}`;
- `steps`: each step by name, in document order: a step record, or, when the
  step varies over a sweep, the list of its points.

Every step record carries `at` (its index on each axis it varies over, `{}`
when it is not a point) and `result`: the slots `saves` makes, each
`{"save": <file>, "value": None}`, filled when the step finishes.
"""

import hashlib
import inspect
import itertools
import json
import os
import re
import types
from typing import Any, Literal, NoReturn, Union, get_args, get_origin

from safetensors import safe_open

from . import address, data, featurizers, ops, positions
from .registry import registers

FORMAT = "clab/1"


class CompileError(ValueError):
    pass


def fail(path: str, message: str) -> NoReturn:
    raise CompileError(f"{path}: {message}")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def points(entry: dict | list) -> list[dict]:
    """A plan entry as its list of points; `[entry]` when it is not swept."""
    return entry if isinstance(entry, list) else [entry]


def dims_of(plan: dict, name: str) -> list[str]:
    """The axes of the value `name` (a step's value, or a step's read)."""
    thing, _, member = name.partition(".")
    step = points(plan["steps"][thing])[0]
    if member:
        return next(r["dims"] for r in step["reads"] if r["name"] == member)
    return step["dims"]


def place(plan: dict, at: dict, file: str) -> str:
    """A saved value's path in the output: one `<label>=<value>` directory per axis of its point."""
    dirs = []
    for label in plan["sweeps"]:
        if label in at:
            value = plan["sweeps"][label]["values"][at[label]]
            dirs.append(f"{label}={','.join(map(str, value)) if isinstance(value, list) else value}")
    return "/".join(dirs + [file])


# -- the blocks of a document: each function's signature is its contract -------

ROOT = {
    "description": "what the experiment is, in prose",
    "model": "the checkpoint, and how to load it",
    "data": "each dataset by name",
    "sites": "named locations, referenced by `site`",
    "interventions": "named bundles of writes and reads, pulled into a step by `interventions`",
    "featurizers": "named spaces a read or write happens in",
    "steps": "what runs, in order; its last entry `saves` names what comes home",
}
DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def model_block(key: str, revision: str, dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
          attn_implementation: Literal["eager", "sdpa"] | None = None) -> dict:
    """The pinned checkpoint. The attention interior needs `attn_implementation: "eager"`."""
    return {"key": key, "revision": revision, "dtype": DTYPES[dtype], "attn_implementation": attn_implementation}


def data_block(path: str | None = None, rows: list | None = None) -> dict:
    """`path` is `data/<path>.json` under the corpus root, `#<split>` keeping the rows of that split; or inline `rows`."""
    return {"path": path, "rows": rows}


def site_block(component: str, layers: int | Literal["all"] | None = None, heads: list[int] | None = None,
         units: list[int] | None = None) -> dict:
    """A component, narrowed: `layers` (an int, or `"all"`: one value, layer axis first) on a block's
    component; at most one of `heads` and `units`, as the component allows."""
    return {"component": component, "layers": layers, "heads": heads, "units": units}


def access_block(site: str | dict, pos: int | dict, ops: list | None = None) -> dict:
    """A write or a read: the `site` it lands on (a declared name, or one inline), the positions `pos` names
    there, and the `ops` it runs on that selection."""
    return {"site": site, "pos": pos, "ops": ops}


def optimizer_block(name: Literal["adamw", "adam", "sgd"], lr: float, weight_decay: float = 0.0) -> dict:
    """The fit's optimizer, over the trained featurizers' parameters only."""
    return {"name": name, "lr": lr, "weight_decay": weight_decay}


def early_stop_block(metric: str, mode: Literal["max", "min"], patience: int) -> dict:
    """End the fit after `patience` epochs without improvement of a body op's mean, scored on the training data."""
    return {"metric": metric, "mode": mode, "patience": patience}


def eval_block(data: dict) -> dict:
    """After the fit, run the body once with each dataset `data` maps read as the other (`{"train": "test"}`)."""
    return {"data": data}


def spell(annotation: Any) -> str:
    if get_origin(annotation) is Literal:
        return "one of " + ", ".join(map(repr, get_args(annotation)))
    if get_origin(annotation) in (Union, types.UnionType):
        return " or ".join(spell(a) for a in get_args(annotation))
    if get_origin(annotation) is list:
        return f"list of {spell(get_args(annotation)[0])}"
    return "null" if annotation is type(None) else getattr(annotation, "__name__", str(annotation))


def fits(value: Any, annotation: Any) -> bool:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is inspect.Parameter.empty or annotation is Any:
        return True
    if origin is Literal:
        return value in args and not isinstance(value, bool)
    if origin in (Union, types.UnionType):
        return any(fits(value, a) for a in args)
    if annotation is type(None):
        return value is None
    if origin is list or annotation is list:
        return isinstance(value, list) and all(fits(v, args[0]) for v in value) if args else isinstance(value, list)
    if origin is dict or annotation is dict:
        return isinstance(value, dict)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, annotation)


def fields(fn, given: Any, path: str, skip: int = 0) -> dict:
    """`given` checked against `fn`'s signature (from parameter `skip` on): every key known, every required one
    present, every value of its annotated type. Returns `given` as a dict."""
    if not isinstance(given, dict):
        fail(path, f"{given!r} is not an object")
    params = list(inspect.signature(fn).parameters.values())[skip:]
    named = {p.name: p for p in params if p.kind is not p.VAR_KEYWORD and p.name not in ops.SUPPLIED
             and p.name not in featurizers.SUPPLIED}
    open_ = any(p.kind is p.VAR_KEYWORD for p in params)
    for key, value in given.items():
        if key in named:
            if not fits(value, named[key].annotation):
                fail(f"{path}.{key}", f"{value!r} is not {spell(named[key].annotation)}")
        elif not open_:
            fail(f"{path}.{key}", f"unknown field; valid: {', '.join(named)}")
    for key, p in named.items():
        if p.default is p.empty and key not in given:
            fail(path, f"missing {key!r}; required: {', '.join(k for k, q in named.items() if q.default is q.empty)}")
    return given


# -- step kinds ---------------------------------------------------------------

KINDS: dict = {}
kind = registers(KINDS)


@kind
def forward(cx: dict, path: str, name: str, *, input: str, writes: dict | None = None, reads: dict | None = None,
            interventions: str | list[str] | None = None) -> dict:
    """One forward over every row of a dataset input. `input` is `<dataset>.<column>` or
    `<dataset>.<column>[i]`; `writes` land where they name before anything later reads them; each read binds
    `<step>.<read>`; `interventions` pulls named bundles of writes and reads in, in listed order, before the
    step's own.

    Rows run in consecutive batches of the run's batch size, each padded on the left to the input's longest
    row, with position ids that count real tokens. A row's numbers then do not depend on which rows share its
    batch, but they do depend on the batch's shape: a trailing partial batch is a smaller matrix, and can
    differ in the last bits. Results reproduce to the bit at a fixed batch size."""
    inp = parse_input(cx, input, f"{path}.input")
    w, r = records(cx, path, inp, writes, reads, interventions, generated=False)
    for read in r:
        cx["names"][f"{name}.{read['name']}"] = {"kind": "read", "dims": read["dims"], "data": inp["data"]}
    return {"kind": "forward", "at": {}, "input": inp, "writes": w, "reads": r, "result": {}}


@kind
def generate(cx: dict, path: str, name: str, *, input: str, max_new_tokens: int, min_new_tokens: int = 0,
             do_sample: bool = False, writes: dict | None = None, reads: dict | None = None,
             interventions: str | list[str] | None = None) -> dict:
    """Generation from every row of a dataset input. Writes land on the prompt; reads in the `generated` frame
    read decode steps. The step's own value is the new token ids, `[rows, new]`."""
    inp = parse_input(cx, input, f"{path}.input")
    w, r = records(cx, path, inp, writes, reads, interventions, generated=True)
    for read in r:
        cx["names"][f"{name}.{read['name']}"] = {"kind": "read", "dims": read["dims"], "data": inp["data"]}
    cx["names"][name] = {"kind": "ids", "dims": ["rows", "new"], "data": inp["data"]}
    options = {"max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens, "do_sample": do_sample}
    return {"kind": "generate", "at": {}, "input": inp, "options": options, "writes": w, "reads": r, "result": {}}


@kind
def op(cx: dict, path: str, name: str, *, op: str, of: str, **arguments) -> dict:
    """An op over a value earlier steps produced: `of` names it, and the op's arguments sit beside it (see Ops).
    The step's value is the op's result."""
    info = value_ref(cx, of, f"{path}.of", data=None, number=False)
    record, dims = op_record(cx, path, {"op": op, **arguments}, info["dims"], "step", info["data"])
    returns = inspect.signature(ops.OPS[op]).return_annotation
    cx["names"][name] = {"kind": "op" if returns is not dict else "bundle", "dims": dims, "data": info["data"]}
    return {"kind": "op", "at": {}, "op": op, "args": {"of": of, **record["args"]}, "data": info["data"],
            "dims": dims, "result": {}}


@kind
def fit(cx: dict, path: str, name: str, *, train: list[str], objective: dict, epochs: int, batch_size: int,
        seed: int, optimizer: dict, steps: dict, early_stop: dict | None = None, eval: dict | None = None) -> dict:
    """Train the featurizers `train` names. Each epoch runs the body `steps` over minibatches of `batch_size`
    rows and minimizes the weighted sum `objective` (`{term: weight}`; a term is a body op, averaged, or a
    trained featurizer's scalar, `<featurizer>.<scalar>`). Inside the body a trained featurizer's name and
    `<fit>.<featurizer>` are the one being trained. Afterwards `<fit>.<featurizer>` is the trained one,
    `<fit>.<op>` a body op under `eval`, `<fit>.<step>.<read>` a body read, and `<fit>.history.<op>` the
    body op's mean per epoch."""
    declared = cx["doc"].get("featurizers", {})
    for i, f in enumerate(train):
        if f not in declared:
            fail(f"{path}.train[{i}]", f"no featurizer {f!r}; declared: {', '.join(declared) or 'none'}")
    outer = cx["names"]
    cx["names"] = dict(outer)
    for f in train:
        cx["names"][f"{name}.{f}"] = {"kind": "featurizer", "dims": [], "data": None}
    body = {}
    check_names(steps, {}, f"{path}.steps")
    for step_name, spec in steps.items():
        if step_name == "saves":
            fail(f"{path}.steps.saves", "saves is declared only at the root of steps; name an inner value as "
                 f"{name}.<member>")
        body[step_name] = compile_step(cx, f"{path}.steps.{step_name}", step_name, spec)
    body_ops = [s for s, b in body.items() if b["kind"] == "op"]
    for term, weight in objective.items():
        thing, _, scalar = term.partition(".")
        valid = term in body_ops or (thing in train and scalar in featurizers.SCALARS.get(declared[thing]["kind"], ()))
        if not valid:
            scalars = [f"{f}.{s}" for f in train for s in featurizers.SCALARS.get(declared[f]["kind"], ())]
            fail(f"{path}.objective.{term}", f"not a body op or a trained featurizer's scalar; valid: "
                 f"{', '.join(body_ops + scalars)}")
        if not fits(weight, float):
            fail(f"{path}.objective.{term}", f"{weight!r} is not a number")
    stop = None
    if early_stop is not None:
        stop = early_stop_block(**fields(early_stop_block, early_stop, f"{path}.early_stop"))
        if stop["metric"] not in body_ops:
            fail(f"{path}.early_stop.metric", f"{stop['metric']!r} is not a body op; body ops: {', '.join(body_ops)}")
    swap = None
    if eval is not None:
        swap = eval_block(**fields(eval_block, eval, f"{path}.eval"))
        for a, b in swap["data"].items():
            for d in (a, b):
                if d not in cx["rows"]:
                    fail(f"{path}.eval.data", f"no dataset {d!r}; datasets: {', '.join(cx['rows'])}")
    names = cx["names"]
    cx["names"] = outer
    for f in train:
        cx["names"][f"{name}.{f}"] = {"kind": "featurizer", "dims": [], "data": None}
    for step_name, b in body.items():
        if b["kind"] == "op":
            cx["names"][f"{name}.{step_name}"] = names[step_name]
            cx["names"][f"{name}.history.{step_name}"] = {"kind": "op", "dims": ["epochs"], "data": None}
        for read in b.get("reads", []):
            cx["names"][f"{name}.{step_name}.{read['name']}"] = names[f"{step_name}.{read['name']}"]
    opt = optimizer_block(**fields(optimizer_block, optimizer, f"{path}.optimizer"))
    return {"kind": "fit", "at": {}, "train": train, "objective": objective, "epochs": epochs,
            "batch_size": batch_size, "seed": seed, "optimizer": opt, "early_stop": stop, "eval": swap,
            "steps": body, "result": {}}


# -- the pieces of a step -------------------------------------------------------

INPUT = re.compile(r"^(\w+)\.(\w+)(?:\[(\d+)\])?$")


def parse_input(cx: dict, text: str, path: str) -> dict:
    match = INPUT.match(text)
    if not match:
        fail(path, f"{text!r} is not <dataset>.<column> or <dataset>.<column>[i]")
    name, column, index = match.group(1), match.group(2), match.group(3)
    if name not in cx["rows"]:
        fail(path, f"no dataset {name!r}; datasets: {', '.join(cx['rows'])}")
    rows = cx["rows"][name]
    if any(column not in r for r in rows):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(rows[0])}")
    if index is not None and any(not isinstance(r[column], list) or int(index) >= len(r[column]) for r in rows):
        fail(path, f"column {column!r} has no entry {index} on every row")
    cx["columns"][name].add(column)
    return {"data": name, "column": column, "index": None if index is None else int(index), "variables": None}


def records(cx: dict, path: str, inp: dict, writes, reads, interventions, generated: bool) -> tuple[list, list]:
    """A step's writes and reads, named bundles first in listed order, as plan records."""
    given: dict[str, list] = {"writes": [], "reads": []}
    bundles = cx["doc"].get("interventions", {})
    for n in [interventions] if isinstance(interventions, str) else interventions or []:
        if n not in bundles:
            fail(f"{path}.interventions", f"no intervention {n!r}; declared: {', '.join(bundles) or 'none'}")
        for key in bundles[n]:
            if key not in given:
                fail(f"interventions.{n}.{key}", "unknown field; valid: writes, reads")
        for side in given:
            given[side] += [(f"interventions.{n}.{side}.{k}", k, v) for k, v in bundles[n].get(side, {}).items()]
    for side, spec in (("writes", writes), ("reads", reads)):
        given[side] += [(f"{path}.{side}.{k}", k, v) for k, v in (spec or {}).items()]
    for side, entries in given.items():
        seen = [k for _, k, _ in entries]
        for k in seen:
            if seen.count(k) > 1:
                fail(f"{path}.{side}", f"two {side[:-1]}s named {k!r}")
    w = [site_record(cx, p, k, v, inp, generated, write=True) for p, k, v in given["writes"]]
    r = [site_record(cx, p, k, v, inp, generated, write=False) for p, k, v in given["reads"]]
    if any(rec["pos"]["scope"] and "variable" in rec["pos"]["scope"] for rec in w + r):
        if inp["index"] is None:
            inp["variables"] = {"column": None, "index": None}
        else:
            inp["variables"] = {"column": f"{inp['column']}_variables", "index": inp["index"]}
            cx["columns"][inp["data"]].add(f"{inp['column']}_variables")
    return w, r


def site_record(cx: dict, path: str, name: str, spec: Any, inp: dict, generated: bool, write: bool) -> dict:
    """One write or read: where it lands (`site`, `pos`) and the ops it runs there."""
    access = access_block(**fields(access_block, spec, path))
    where = access["site"]
    if isinstance(where, str):
        declared = cx["doc"].get("sites", {})
        if where not in declared:
            fail(f"{path}.site", f"no site {where!r}; declared: {', '.join(declared) or 'none'}")
        where, site_path = declared[where], f"sites.{access['site']}"
    else:
        site_path = f"{path}.site"
    s = site_block(**fields(site_block, where, site_path))
    component = s["component"]
    if component not in address.COMPONENTS:
        fail(f"{site_path}.component", f"unknown component {component!r}; valid: {', '.join(address.COMPONENTS)}")
    meaning = address.COMPONENTS[component]
    if meaning["per_layer"] and s["layers"] is None:
        fail(site_path, f"{component} is a block's value and needs 'layers'")
    if not meaning["per_layer"] and s["layers"] is not None:
        fail(f"{site_path}.layers", f"{component} is the model's own value and takes no layers")
    narrow = {k: list(s[k]) for k in ("heads", "units") if s[k] is not None}
    if len(narrow) > 1:
        fail(site_path, "a site narrows one axis: heads or units, not both")
    for k in narrow:
        if k not in meaning["narrow"]:
            fail(f"{site_path}.{k}", f"{component} is narrowed by {', '.join(meaning['narrow'])}, not {k}")
    try:
        pos = positions.parse(access["pos"])
    except ValueError as e:
        fail(f"{path}.pos", str(e))
    if pos["frame"] == "generated" and not generated:
        fail(f"{path}.pos", "the generated frame is a generate step's; a forward has only its prompt")
    variable = (pos["scope"] or {}).get("variable")
    if variable is not None and pos["frame"] == "prompt" and inp["index"] is not None:
        column, i = f"{inp['column']}_variables", inp["index"]
        entries = [r.get(column) for r in cx["rows"][inp["data"]]]
        if any(not isinstance(e, list) or len(e) <= i or variable not in e[i] for e in entries):
            fail(f"{path}.pos.scope.variable", f"no {variable!r} in {column}[{i}] on every row")
    elif variable is not None:
        column_ref(cx, f"{inp['data']}.{variable}", f"{path}.pos.scope.variable", inp["data"])
    dims = address.dims(component, s["layers"], positions.ragged(pos))
    if pos["frame"] == "generated" and pos["scope"] is not None and not positions.ragged(pos):
        dims = ["entries" if a == "rows" else a for a in dims]
    record = {"name": name, "component": component, "layer": s["layers"], "select": narrow, "pos": pos}
    selection, chain = dims, []
    for i, o in enumerate(access["ops"] or []):
        if isinstance(o, dict) and o.get("op") == "readout" and not meaning["stream"]:
            fail(f"{path}.ops[{i}]", f"readout projects a residual-stream value; {component} is not one")
        rec, out = op_record(cx, f"{path}.ops[{i}]", o, dims, "write" if write else "read", inp["data"])
        if write and out != selection:
            fail(f"{path}.ops[{i}]", f"{o['op']!r} turns {dims} into {out}; a write's ops keep the value's layout")
        chain.append(rec)
        dims = out
    return {**record, "ops": chain, "dims": dims}


def op_record(cx: dict, path: str, spec: Any, dims: list[str], where: str, data: str | None) -> tuple[dict, list]:
    """One op with its arguments bound and checked against its signature, and the layout of what it returns.

    `where` is `"step"`, `"read"` or `"write"`; inside a read or write the record also carries `dims`, the
    layout it acts on, and the op's `featurizer` and `features`."""
    if not isinstance(spec, dict) or spec.get("op") not in ops.OPS:
        fail(f"{path}.op", f"unknown op {spec.get('op') if isinstance(spec, dict) else spec!r}; "
             f"valid: {', '.join(ops.OPS)}")
    name = spec["op"]
    params = ops.parameters(name)
    given = {k: v for k, v in spec.items() if k != "op"}
    site = where != "step"
    space = {"featurizer": None, "features": None}
    if site:
        for key in space:
            if key in given and key not in params:
                space[key] = given.pop(key)
        if space["featurizer"] is not None:
            featurizer_ref(cx, space["featurizer"], f"{path}.featurizer")
        if space["features"] is not None:
            if space["featurizer"] is None:
                fail(f"{path}.features", "features are a featurizer's latents; name the 'featurizer' they belong to")
            if not fits(space["features"], list[int]):
                fail(f"{path}.features", f"{space['features']!r} is not a list of ints")
    if where != "write" and "original" in inspect.signature(ops.OPS[name]).parameters:
        fail(f"{path}.op", f"{name!r} restores the value a site had before its writes; it runs only inside a write")
    for key in given:
        if key not in params:
            fail(f"{path}.{key}", f"{name!r} takes no {key!r}; its arguments: {', '.join(params) or 'none'}")
    # Inside a write an operand stands for the selection: its own layout, or one vector of its features.
    layout = dims[:-1] + ["features"] if space["featurizer"] is not None else dims
    operand = [layout, [a for a in layout if a not in address.EXAMPLE_AXES]]
    args = {}
    for key, p in params.items():
        if key not in given:
            if p.default is p.empty:
                fail(path, f"{name!r} needs {key!r}")
            args[key] = p.default
            continue
        value, at = given[key], f"{path}.{key}"
        if p.annotation == ops.Value:
            info = value_ref(cx, value, at, data=data, number=True)
            if where == "write" and info["kind"] != "number" and info["dims"] not in operand:
                fail(at, f"{value!r} is laid out {info['dims']}; this write's selection is {layout}, so an operand "
                     f"is {operand[0]} or {operand[1]}")
        elif p.annotation == ops.Tokens:
            column_ref(cx, value, at, data)
        elif p.annotation == ops.Featurizer:
            featurizer_ref(cx, value, at)
        elif not fits(value, p.annotation):
            fail(at, f"{value!r} is not {spell(p.annotation)}")
        args[key] = value
    try:
        out = ops.out_dims(name, dims, args)
    except ValueError as e:
        fail(f"{path}.op", str(e))
    gone = [a for a in dims if a not in out]
    if site and {"rows", "entries"} & set(gone):
        fail(f"{path}.op", f"{name!r} aggregates over rows, and inside a read or write it sees one batch; "
             f"make it a step after the forward")
    record = {"op": name, "args": args, **({**space, "dims": dims} if site else {})}
    return record, out


def value_ref(cx: dict, value: Any, path: str, data: str | None, number: bool) -> dict:
    """A named value produced before this point, or (where `number`) a number."""
    if number and fits(value, float):
        return {"kind": "number", "dims": [], "data": None}
    info = cx["names"].get(value) if isinstance(value, str) else None
    if info is None or info["kind"] in ("featurizer", "scalar"):
        known = sorted(k for k, v in cx["names"].items() if v["kind"] not in ("featurizer", "scalar"))
        fail(path, f"{value!r} names no value produced before this step; known: {', '.join(known) or 'none'}")
    if data is not None and info["data"] is not None and info["dims"][:1] in (["rows"], ["entries"]) \
            and info["data"] != data:
        fail(path, f"{value!r} has a row per example of {info['data']!r}, and this runs over {data!r}")
    cx["refs"].append(value)
    return info


def column_ref(cx: dict, value: Any, path: str, data: str | None) -> None:
    """`<dataset>.<column>`, a column present on every row, of the dataset the value's rows come from."""
    name, _, column = value.partition(".") if isinstance(value, str) else ("", "", "")
    if name not in cx["rows"]:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx['rows'])}")
    rows = cx["rows"][name]
    if any(column not in r for r in rows):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(rows[0])}")
    if data is not None and name != data:
        fail(path, f"{value!r} is a column of {name!r}, and the value it goes with has a row per example of {data!r}")
    cx["columns"][name].add(column)


def featurizer_ref(cx: dict, value: Any, path: str) -> None:
    if not (isinstance(value, str) and cx["names"].get(value, {}).get("kind") == "featurizer"):
        known = sorted(k for k, v in cx["names"].items() if v["kind"] == "featurizer")
        fail(path, f"{value!r} names no featurizer here; featurizers: {', '.join(known) or 'none'}")
    cx["refs"].append(value)


def compile_step(cx: dict, path: str, name: str, spec: Any) -> dict:
    if not isinstance(spec, dict) or spec.get("kind") not in KINDS:
        got = spec.get("kind") if isinstance(spec, dict) else spec
        fail(f"{path}.kind", f"unknown step kind {got!r}; valid: {', '.join(KINDS)}")
    given = {k: v for k, v in spec.items() if k != "kind"}
    fn = KINDS[spec["kind"]]
    fields(fn, given, path, skip=3)
    return fn(cx, path, name, **given)


def check_names(steps: dict, others: dict, path: str) -> None:
    """Names are one flat namespace per scope: no dots, and no name both a step and a dataset or featurizer.

    A reference is "thing, then member", split at the first dot: `counterfactual.v_cf` is step
    `counterfactual`'s read `v_cf`; `mean` is an op step's value; `pairs.cf_answer` is a dataset's column;
    `rot` is a declared featurizer and `mask.mask` its scalar; `fit.rot` is a fit's member. Values are
    produced in document order and referenced only after they exist. A fit's body is a scope of its own: it
    sees every value produced before the fit, and nothing leaves it except through `<fit>.<member>`. A named
    intervention is a template, so the names inside it resolve where a step pulls it in. Under a sweep, a
    name means the producer's point that agrees with the consumer's coordinates on the producer's axes.
    """
    for name in steps:
        if "." in name:
            fail(f"{path}.{name}", "a name has no dots: a dot separates a thing from its member")
        if name in others:
            fail(f"{path}.{name}", f"{name!r} is also a {others[name]}")


# -- sweeps -------------------------------------------------------------------

UNSWEPT = ("kind", "op", "site", "component", "input", "interventions", "train", "featurizer", "path",
           "file_path", "of")


def find_sweeps(doc: dict) -> dict[tuple, str]:
    """Every `{"sweep": [...]}` standing for a value, by document path, labelled by the last key of its path.

    A sweep may stand for any value except under `description` and `model`, on a name, a kind, a data `path`
    or a `file_path`. Two sweeps with one label are refused; one sweep reached from two places (a swept site
    two steps use) is one axis, so its points zip rather than cross. A step that uses a swept thing, or
    references a value that varies, becomes a list of points, one per combination of its axes, each a whole
    step whose `at` gives its index on each axis; a step no sweep touches runs once. A saved value of a point
    lands under one `<label>=<value>/` directory per axis.
    """
    found: dict[tuple, str] = {}

    def walk(value, path):
        if isinstance(value, dict) and set(value) == {"sweep"}:
            where = ".".join(map(str, path))
            label = next(k for k in reversed(path) if isinstance(k, str))
            if path[0] in ("description", "model") or label in UNSWEPT:
                fail(where, f"a sweep cannot stand for {label!r}")
            if not isinstance(value["sweep"], list) or not value["sweep"]:
                fail(where, "a sweep is a non-empty list of values")
            if "sweep" in json.dumps(value["sweep"]):
                fail(where, "a sweep's values hold no sweep")
            for other, other_label in found.items():
                if other_label == label:
                    fail(where, f"label {label!r} is also the sweep at {'.'.join(map(str, other))}; rename one")
            found[path] = label
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + (i,))

    walk(doc, ())
    return found


def substitute(value: Any, at: dict, sweeps: dict, path: tuple = ()) -> Any:
    """`value` with every sweep whose label `at` names replaced by its value at that index."""
    if path in sweeps and sweeps[path] in at:
        return value["sweep"][at[sweeps[path]]]
    if isinstance(value, dict):
        return {k: substitute(v, at, sweeps, path + (k,)) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, at, sweeps, path + (i,)) for i, v in enumerate(value)]
    return value


def footprint(doc: dict, spec: Any, path: tuple) -> list[tuple]:
    """The document paths a step reads: its own, and the sites, interventions and featurizers it names."""
    found = [path]

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                named = v if isinstance(v, list) else [v]
                if k == "site" and isinstance(v, str):
                    found.append(("sites", v))
                if k == "interventions" and not isinstance(v, dict):
                    for n in named:
                        found.append(("interventions", n))
                        walk(doc.get("interventions", {}).get(n))
                if k in ("featurizer", "train"):
                    found.extend(("featurizers", n) for n in named if n in doc.get("featurizers", {}))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(spec)
    return found


def expand(cx: dict, sweeps: dict, own: list[str], build) -> tuple[dict | list, list[str]]:
    """An entry compiled at every point of its axes: its own, and those of every value it references."""
    labels = list(dict.fromkeys(sweeps.values()))
    doc = cx["doc"]
    cx["refs"] = []
    cx["doc"] = substitute(doc, {label: 0 for label in own}, sweeps)
    build()
    axes = set(own) | {a for ref in cx["refs"] for a in cx["axes"].get(ref.partition(".")[0], [])}
    axes = [label for label in labels if label in axes]
    entries = []
    for index in itertools.product(*(range(len(cx["sweep_values"][a])) for a in axes)):
        at = dict(zip(axes, index))
        cx["doc"] = substitute(doc, at, sweeps)
        entry = build()
        entry["at"] = at
        for body in entry.get("steps", {}).values():
            body["at"] = at
        entries.append(entry)
    cx["doc"] = doc
    return (entries if axes else entries[0]), axes


# -- the whole document ---------------------------------------------------------


def compile(document: dict, path: str) -> dict:
    """The plan of `document`, read from `path`. Data and bundle files resolve against the corpus root, the
    parent of the document's directory."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    if not isinstance(document, dict):
        fail(path, "a document is a JSON object")
    for key in document:
        if key not in ROOT:
            fail(key, f"unknown root key; valid: {', '.join(ROOT)}")
    for key in ("model", "data", "steps"):
        if key not in document:
            fail(path, f"missing {key!r}")
    sweeps = find_sweeps(document)
    sweep_values = {}
    for where, label in sweeps.items():
        value = document
        for k in where:
            value = value[k]
        sweep_values[label] = {"path": ".".join(map(str, where)), "values": value["sweep"]}

    datasets = {}
    if not isinstance(document["data"], dict):
        fail("data", "data is an object of datasets by name")
    for name, spec in document["data"].items():
        d = data_block(**fields(data_block, spec, f"data.{name}"))
        if (d["path"] is None) == (d["rows"] is None):
            fail(f"data.{name}", "a dataset has a 'path' or inline 'rows', not both")
        if d["path"] is not None and not os.path.exists(os.path.join(root, "data", d["path"].partition("#")[0] + ".json")):
            fail(f"data.{name}.path", f"no file data/{d['path'].partition('#')[0]}.json under the corpus root")
        file = d["path"]
        rows = data.load_rows(root, file) if file is not None else d["rows"]
        if not rows:
            fail(f"data.{name}", "the dataset has no rows")
        datasets[name] = {"path": d["path"], "rows": rows}

    declared = document.get("featurizers", {})
    things = {**{n: "dataset" for n in datasets}, **{n: "featurizer" for n in declared}}
    for name in declared:
        if name in datasets:
            fail(f"featurizers.{name}", f"{name!r} is also a dataset")
    if not isinstance(document["steps"], dict):
        fail("steps", "steps is an object of steps by name")
    steps = {k: v for k, v in document["steps"].items() if k != "saves"}
    check_names(steps, things, "steps")

    cx: dict = {"doc": document, "root": root, "rows": {n: d["rows"] for n, d in datasets.items()},
                "columns": {n: set() for n in datasets}, "names": {}, "refs": [], "axes": {},
                "sweep_values": {label: s["values"] for label, s in sweep_values.items()}}
    plan: dict = {"format": FORMAT, "document": document_block(document, path, root),
                  "description": document.get("description", ""),
                  "model": model_block(**fields(model_block, document["model"], "model")),
                  "data": {}, "featurizers": {}, "sweeps": sweep_values, "steps": {}}

    for name, spec in declared.items():
        own = [label for where, label in sweeps.items() if where[:2] == ("featurizers", name)]
        plan["featurizers"][name], cx["axes"][name] = expand(
            cx, sweeps, own, lambda name=name: featurizer_record(cx, f"featurizers.{name}", cx["doc"]["featurizers"][name]))
        cx["names"][name] = {"kind": "featurizer", "dims": [], "data": None}
        for scalar in featurizers.SCALARS.get(spec.get("kind"), ()):
            cx["names"][f"{name}.{scalar}"] = {"kind": "scalar", "dims": [], "data": None}

    for name, spec in steps.items():
        reads = footprint(document, spec, ("steps", name))
        own = [label for where, label in sweeps.items() if any(where[:len(p)] == p for p in reads)]
        plan["steps"][name], cx["axes"][name] = expand(
            cx, sweeps, own, lambda name=name: compile_step(cx, f"steps.{name}", name, cx["doc"]["steps"][name]))

    save_slots(cx, plan, document["steps"].get("saves", {}))
    for name, d in datasets.items():
        columns = sorted(cx["columns"][name] | ({"example_id"} if all("example_id" in r for r in d["rows"]) else set()))
        path_, _, split = (d["path"] or "").partition("#")
        plan["data"][name] = {"path": path_ or None, "split": split or None, "columns": columns,
                              "rows": [{c: r[c] for c in columns} for r in d["rows"]]}
    return plan


def document_block(document: dict, path: str, root: str) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(path):
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    else:
        digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    return {"name": name, "path": os.path.relpath(os.path.abspath(path), root), "sha256": digest}


def featurizer_record(cx: dict, path: str, spec: Any) -> dict:
    if not isinstance(spec, dict) or spec.get("kind") not in featurizers.KINDS:
        got = spec.get("kind") if isinstance(spec, dict) else spec
        fail(f"{path}.kind", f"unknown featurizer kind {got!r}; valid: {', '.join(featurizers.KINDS)}")
    fn = featurizers.KINDS[spec["kind"]]
    given = fields(fn, {k: v for k, v in spec.items() if k != "kind"}, path)
    params = {k: p.default for k, p in inspect.signature(fn).parameters.items() if k not in featurizers.SUPPLIED}
    record = {"kind": spec["kind"], **params, **given}
    if record.get("classes") is not None:
        column_ref(cx, record["classes"], f"{path}.classes", None)
    bundle = None
    if record.get("file_path") is not None:
        bundle = load_bundle(cx, record, path)
    return {**record, "bundle": bundle}


def load_bundle(cx: dict, record: dict, path: str) -> dict:
    """A featurizer's saved tensors and header, checked against the document key by key."""
    file = os.path.join(cx["root"], record["file_path"])
    if not os.path.exists(file):
        fail(f"{path}.file_path", f"{record['file_path']} does not exist under the corpus root; "
             "run the document that writes it first")
    with safe_open(file, "pt") as f:
        header = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    expected = {"format": FORMAT, "kind": record["kind"], "model.key": cx["doc"]["model"]["key"],
                "model.revision": cx["doc"]["model"]["revision"]}
    if record.get("k") is not None:
        expected["k"] = str(record["k"])
    for key, value in expected.items():
        if header.get(key) != value:
            fail(f"{path}.file_path", f"{record['file_path']}: header {key!r} is {header.get(key)!r}; "
                 f"this document expects {value!r}")
    return {"header": header, "tensors": tensors}


def save_slots(cx: dict, plan: dict, spec: Any) -> None:
    """`saves` names the values that come home, `{name: file}` or `[names]`, each at the path its file gives
    under the output directory (`.json` a table, `.safetensors` tensors); a list entry is saved as
    `<name>.json` when it is an op's value, else `<name>.safetensors`. Every other value is gone when the
    run ends. Saves are declared only at the root of `steps`."""
    given = spec.items() if isinstance(spec, dict) else [(n, None) for n in spec]
    placed: dict[str, str] = {}
    for name, file in given:
        info = cx["names"].get(name)
        if info is None:
            fail(f"steps.saves.{name}", f"names no value; values: {', '.join(sorted(cx['names']))}")
        tables = info["kind"] == "op"
        file = file or name + (".json" if tables else ".safetensors")
        extension = os.path.splitext(file)[1]
        if extension not in (".json", ".safetensors") or (extension == ".json" and not tables):
            fail(f"steps.saves.{name}", f"{file!r}: an op's value saves as .json or .safetensors, anything else "
                 "as .safetensors")
        if file == "manifest.json":
            fail(f"steps.saves.{name}", "manifest.json is the run's own")
        for point in points(plan["steps"][name.partition(".")[0]]):
            point["result"][name] = {"save": file, "value": None}
            where = place(plan, point["at"], file)
            if where in placed:
                fail(f"steps.saves.{name}", f"{where} is also where {placed[where]!r} is saved")
            placed[where] = name
