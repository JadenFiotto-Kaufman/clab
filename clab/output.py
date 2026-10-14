"""A run's output directory and back, and saved values as tables.

One file per save, named after it: `<name>.safetensors` when the value is a tensor or a dict of tensors (a
bundle; a ragged read's `{values, row, tok}`), `<name>.json` otherwise, tensors in it as nested lists. A swept
save holds one entry per point, keyed by its coordinate: `pos=-1` in safetensors, `"-1"` in JSON
(`layers=0,pos=-1` and `"0,-1"` over two axes), a dict's own keys after a `/`. A file is the value and
nothing else, except that a bundle a featurizer loads (the value of an op that declares one, `pca`) carries
the header that featurizer checks.

`manifest.json` is the compiled plan without the rows it read from files and without bundle tensors, its
`results` each replaced by `{file, step, dims, axes, shape, dtype}`: that is how a result links back to the
step that made it. It records the document's path relative to the corpus root, and each dataset's `n_rows`. `shape` and `dtype` follow the value's structure (per key, per point).
The plan's `run` holds how each input tokenized, what each generate step's rows generated, and which token
(in the generated frame, which decode step) every positioned read and write resolved to per row, which
`table` uses to label positions. Values are never in the manifest.
"""

import itertools
import json
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from . import data, featurizers, ops, positions
from .plan import Plan, arguments, coordinate, corpus_root, dims_of, labelled, points


def dump(plan: Plan, out: str) -> None:
    if plan.run is None:
        raise ValueError("the plan has not run, so it has no results; execute it first")
    os.makedirs(out, exist_ok=True)
    described = {}
    for name in plan.saves:
        entry = plan.steps[name.partition(".")[0]]
        value = plan.results[name]
        if isinstance(entry, list):
            value = {labelled(plan, p.at, tensors(value)): value[coordinate(plan, p.at)] for p in entry}
        file = f"{name}.safetensors" if tensors(value) else f"{name}.json"
        if tensors(value):
            save_file(flat({name: value} if isinstance(value, Tensor) else value), os.path.join(out, file),
                      metadata=header(plan, name))
        else:
            with open(os.path.join(out, file), "w") as f:
                json.dump(value, f, default=lambda t: t.tolist(), ensure_ascii=False)
        described[name] = {"file": file, "step": name.partition(".")[0], "dims": dims_of(plan, name),
                           "axes": list(points(entry)[0].at), "shape": describe(value, "shape"),
                           "dtype": describe(value, "dtype")}
    manifest = plan.model_dump(mode="json", exclude_defaults=True)
    manifest["model"]["dtype"] = plan.model.dtype                 # recorded even where it is the default
    for d in manifest["data"].values():
        if "path" in d or "hf" in d:
            del d["rows"]
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({**manifest, "results": described}, f, indent=1, ensure_ascii=False)


def load(out: str) -> Plan:
    with open(os.path.join(out, "manifest.json")) as f:
        manifest = json.load(f)
    described = manifest.pop("results")
    plan = Plan(**manifest)
    for name, d in described.items():
        file = os.path.join(out, d["file"])
        if file.endswith(".safetensors"):
            value = nested(load_file(file))
            value = value[name] if isinstance(d["dtype"], str) else value
        else:
            with open(file) as f:
                value = json.load(f)
        value = restored(value, d["dtype"])
        entry = plan.steps[d["step"]]
        if isinstance(entry, list):
            value = {coordinate(plan, p.at): value[labelled(plan, p.at, file.endswith(".safetensors"))] for p in entry}
        plan.results[name] = value
    return plan


def tensors(value: Any) -> bool:
    return isinstance(value, Tensor) or isinstance(value, dict) and all(map(tensors, value.values()))


def flat(value: dict, prefix: str = "") -> dict[str, Tensor]:
    out = {}
    for k, v in value.items():
        out.update(flat(v, f"{prefix}{k}/") if isinstance(v, dict) else {prefix + k: v.contiguous()})
    return out


def nested(value: dict[str, Tensor]) -> dict:
    out: dict = {}
    for key, v in value.items():
        *path, last = key.split("/")
        here = out
        for k in path:
            here = here.setdefault(k, {})
        here[last] = v
    return out


def describe(value: Any, what: str) -> Any:
    """A value's `shape` or `dtype`, following its structure: a tensor's own, a dict's per key, a list's
    length (and no dtype)."""
    if isinstance(value, Tensor):
        return list(value.shape) if what == "shape" else str(value.dtype).removeprefix("torch.")
    if isinstance(value, dict):
        return {k: describe(v, what) for k, v in value.items()}
    return [len(value)] if isinstance(value, list) and what == "shape" else None


def restored(value: Any, dtype: Any) -> Any:
    """A loaded value with every entry its `dtype` says was a tensor a tensor again."""
    if isinstance(dtype, str):
        return value if isinstance(value, Tensor) else torch.tensor(value, dtype=getattr(torch, dtype))
    if isinstance(dtype, dict):
        return {k: restored(value[k], dtype[k]) for k in value}
    return value


def header(plan: Plan, name: str) -> dict[str, str] | None:
    """A bundle's header, when the value is one: the value of an op step whose op declares the featurizer kind
    its bundle is for."""
    step = points(plan.steps[name.partition(".")[0]])[0]
    kind = ops.DECLARED[step.op]["bundle"] if "." not in name and step.kind == "op" else None
    return featurizers.header(kind, arguments(step), plan.model) if kind else None


# -- tables ---------------------------------------------------------------------------------------------

JOIN = ("at", "row", "layer", "pos", "step", "key")


def table(plan: Plan, names: tuple[str, ...], columns: list[str] | None) -> list[dict]:
    """The saved values `names` as rows, joined wide on their shared axes, one column per name.

    A value with a rows axis gives a row per example and token: `at` (its point's coordinates; a value whose
    step is not swept has no `at`), `row`, `example_id` when the dataset has one, `layer` under `layers:
    "all"`, the token axes with their tokens (`pos`/`pos_token`, the index into the row's tokens a position
    resolved to; `step`/`step_token` in the generated frame, the decode step and the token it chose;
    `key`/`key_token` on an attention pattern), the dataset columns the step used (`columns=` to choose),
    and `value`: the element, or the remaining axes as nested lists. A ragged value gives a row per entry. A
    row a generated-frame read did not reach gives one row, `alignment_missing` (or `alignment_ambiguous`)
    true, with no value. A value with no rows axis gives a row per element with its axis indices."""
    for name in names:
        if name not in plan.saves:
            raise ValueError(f"{name!r} was not saved; add it to saves")
    if len(names) == 1:
        return rows_of(plan, names[0], columns)
    joined: list[dict] = []
    for name in names:
        rows = [{**{k: v for k, v in row.items() if k != "value"}, **({name: row["value"]} if "value" in row else {})}
                for row in rows_of(plan, name, columns)]
        joined = join(joined, rows) if joined else rows
    return joined


def join(left: list[dict], right: list[dict]) -> list[dict]:
    """Two tables side by side on the axes both carry; a row of the side with fewer axes (an unswept value
    beside a swept one, a value with no layer axis beside one with) repeats beside every row it agrees with."""
    def axes(rows: list[dict]) -> list[str]:
        return [k for k in JOIN if any(k in row for row in rows)]
    many, few = (left, right) if len(axes(left)) >= len(axes(right)) else (right, left)
    shared = [k for k in axes(many) if k in axes(few)]
    by_key: dict = {}
    for row in few:
        by_key.setdefault(json.dumps([row.get(k) for k in shared]), []).append(row)
    pairs = [(row, other) for row in many for other in by_key.get(json.dumps([row.get(k) for k in shared]), [])]
    merged = [{**a, **b} if many is left else {**b, **a} for a, b in pairs]
    return [{"at": m.pop("at"), **m} if "at" in m else m for m in merged]


def rows_of(plan: Plan, name: str, columns: list[str] | None) -> list[dict]:
    entry, dims = plan.steps[name.partition(".")[0]], dims_of(plan, name)
    source, used = provenance(plan, name)
    dataset = points(entry)[0].dataset
    examples = materialized(plan, dataset) if dataset else []
    out = []
    for p in points(entry):
        value = plan.results[name][coordinate(plan, p.at)] if isinstance(entry, list) else plan.results[name]
        at = {"at": {label: plan.sweeps[label].values[i] for label, i in p.at.items()}} if p.at else {}
        place = located(plan, source, p.at) if source else None
        for row in elements(value, dims, place):
            if "row" in row and examples:
                example = examples[row["row"]]
                row = {**({"example_id": example["example_id"]} if "example_id" in example else {}), **row,
                       **{c: example[c] for c in (columns if columns is not None else used)}}
            value = {"value": row.pop("value")} if "value" in row else {}
            out.append({**at, **row, **value})                      # the value last, as in a joined table
    return out


def elements(value: Any, dims: list[str], place: dict | None) -> list[dict]:
    """A value's rows: one per index on its example and token axes, the rest of it as the row's value."""
    if isinstance(value, dict) and "tok" in value:
        return entries(value, dims, place)
    if isinstance(value, dict) and "rows" not in dims:
        raise ValueError("a bundle has no rows to table; read it from plan.results")
    first = next(iter(value.values())) if isinstance(value, dict) else value
    shape = list(first.shape) if isinstance(first, Tensor) else [len(first)]
    # The axes a row is per: the example and token axes, or every axis of a value with no rows.
    axes = [i for i, a in enumerate(dims) if a in ("layers", "rows", "pos", "key") or "rows" not in dims]
    out, marked = [], set()
    for index in itertools.product(*(range(shape[i]) for i in axes)):
        at = dict(zip(axes, index))
        whole = {{"rows": "row", "layers": "layer"}.get(dims[i], dims[i]): k for i, k in at.items()
                 if dims[i] not in ("pos", "key")}
        if mark := unreached(place, whole.get("row")):
            if tuple(whole.items()) not in marked:      # one row says so, in place of its tokens
                marked.add(tuple(whole.items()))
                out.append({**whole, **mark})
            continue
        row: dict = dict(whole)
        for i, k in at.items():
            if dims[i] in ("pos", "key"):
                row.update(token(place, at[dims.index("rows")], dims[i], k))
        if row.get("key", 0) is None:
            continue                                    # a padding key
        pick = tuple(at.get(i, slice(None)) for i in range(len(dims)))
        row["value"] = {k: part(v, pick) for k, v in value.items()} if isinstance(value, dict) else part(value, pick)
        out.append(row)
    return out


def entries(value: dict, dims: list[str], place: dict | None) -> list[dict]:
    """A ragged value's rows: one per entry (per layer under `layers: "all"`), labelled with its row and the
    token it is (`tok`: a token index, or a decode step in the generated frame); then one row for each row a
    generated-frame read did not reach."""
    stack = value["values"] if "layers" in dims else value["values"][None]
    out = []
    for layer in range(stack.shape[0]):
        for i, (r, t) in enumerate(zip(value["row"].tolist(), value["tok"].tolist())):
            row = {"layer": layer} if "layers" in dims else {}
            out.append({**row, "row": r, **label(place, r, t, "pos"), "value": stack[layer, i].tolist()})
    marks = [{"row": r, **unreached(place, r)} for r in range(len(place["positions"]))
             if unreached(place, r)] if place else []
    return sorted(out + marks, key=lambda row: row["row"]) if marks else out


def unreached(place: dict | None, row: int | None) -> dict:
    """`{"alignment_missing": True}` (or ambiguous) for a row a generated-frame read did not reach, else {}."""
    resolved = place["positions"][row] if place and row is not None else None
    return {resolved: True} if isinstance(resolved, str) else {}


def token(place: dict | None, row: int, axis: str, k: int) -> dict:
    """Index `k` on a token axis as the token it is: on `pos` through the read's resolved positions, on `key`
    through its `select` (the row's own token indices) or, unselected, the padded batch's keys less the row's
    padding; None on a padding key."""
    if place is None:
        return {axis: k}
    if axis == "pos":
        return label(place, row, place["positions"][row][k], axis)
    index = place["keys"][k] if place["keys"] is not None else k - (place["width"] - len(place["tokens"][row]))
    return {axis: None} if index < 0 else label(place, row, index, axis)


def label(place: dict | None, row: int, index: int, axis: str) -> dict:
    """A row's token `index` as a table labels it: `pos` and `pos_token` (or `key`, `key_token`) in the
    prompt, `step` and `step_token`, the token that step chose, in the generated frame."""
    if place is None:
        return {axis: index}
    name = "step" if place["frame"] == "generated" else axis
    return {name: index, f"{name}_token": place["tokens"][row][index]}


def part(value: Any, pick: tuple) -> Any:
    if isinstance(value, Tensor):
        got = value[pick]
        return got.item() if got.dim() == 0 else got.tolist()
    for i in pick:
        if isinstance(i, int):
            value = value[i]
    return value


def provenance(plan: Plan, name: str) -> tuple[str | None, list[str]]:
    """The read a value's positions come from (following `of`), and the dataset columns its chain uses."""
    used = []
    while True:
        thing, _, member = name.partition(".")
        step = points(plan.steps[thing])[0]
        args = arguments(step) if step.kind == "op" else {}
        used += [v.partition(".")[2] for v in args.values() if isinstance(v, str) and v.partition(".")[0] in plan.data]
        if step.kind != "op":
            used.insert(0, data.parse_input(step.input)[1])
            for o in step.reads[member].ops if member else []:
                used += [v.partition(".")[2] for v in arguments(o).values()
                         if isinstance(v, str) and v.partition(".")[0] in plan.data]
            return (name if member else None), list(dict.fromkeys(used))
        name = step.of


def located(plan: Plan, source: str, at: dict) -> dict:
    """For the read `source` at the point agreeing with `at`: its frame, the rows' tokens (the prompt's, or
    what a generate step's rows generated), the positions each row resolved to, and the keys it selects."""
    assert plan.run is not None
    thing, _, member = source.partition(".")
    producer = next(p for p in points(plan.steps[thing]) if all(at.get(k) == i for k, i in p.at.items()))
    frame = positions.parse(producer.reads[member].pos)["frame"]
    tokens = plan.run.tokens[thing if frame == "generated" else producer.input]
    if frame == "generated" and producer.at:
        tokens = tokens[labelled(plan, producer.at)]
    resolved = plan.run.positions[source]
    keys = (producer.reads[member].site.select or {}).get("key")
    return {"frame": frame, "tokens": tokens, "width": max(map(len, tokens)), "keys": keys,
            "positions": resolved[labelled(plan, producer.at)] if producer.at else resolved}


def materialized(plan: Plan, name: str) -> list[dict]:
    """A dataset's rows: the plan's, or, after `Plan.load`, read again from its file and checked against the
    digest the run recorded."""
    d = plan.data[name]
    if d.rows is not None:
        return d.rows
    rows = data.rows(corpus_root(plan), d.path or "", d.split)
    if data.digest(rows) != d.sha256:
        raise ValueError(f"data.{name}: data/{d.path}.json is not what the run read (its rows' digest differs)")
    return rows
