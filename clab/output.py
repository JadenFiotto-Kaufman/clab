"""A filled plan to a directory and back: each saved value at the path its document gives, and a manifest.

`.json` files are tables, a row per element of the value; `.safetensors`
files hold the tensor under its name, with `dims` and `at` in the metadata.
`manifest.json` mirrors the plan: its steps as compiled, every slot's value
replaced by where it landed. `load` is the inverse of `write`.
"""

import importlib.metadata
import itertools
import json
import os
import platform

import torch
from safetensors.torch import load_file, save_file

from .compiler import dims_of, place, points

AXIS = {"rows": "row", "layers": "layer"}


def write(plan: dict, out: str, run: dict) -> dict:
    """Every filled slot to its file and `manifest.json`; returns the manifest. `run` says how it ran."""
    os.makedirs(out, exist_ok=True)
    saves: list[dict] = []
    steps = {}
    for name, entry in plan["steps"].items():
        described = [written(plan, name, point, out, saves) for point in points(entry)]
        steps[name] = described if isinstance(entry, list) else described[0]
    manifest = {
        "format": plan["format"], "document": plan["document"], "description": plan["description"],
        "model": plan["model"], **run, "versions": versions(),
        "sweeps": [{"label": label, **s} for label, s in plan["sweeps"].items()],
        "data": {n: {k: v for k, v in d.items() if k != "rows"} | {"n_rows": len(d["rows"])}
                 for n, d in plan["data"].items()},
        "featurizers": {n: [without_tensors(p) for p in points(f)] if isinstance(f, list) else without_tensors(f)
                        for n, f in plan["featurizers"].items()},
        "steps": steps, "saves": saves,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def written(plan: dict, name: str, point: dict, out: str, saves: list) -> dict:
    result = {}
    for key, slot in point["result"].items():
        file, value, dims = place(plan, point["at"], slot["save"]), slot["value"], dims_of(plan, key)
        os.makedirs(os.path.dirname(os.path.join(out, file)) or out, exist_ok=True)
        at = coordinates(plan, point["at"])
        if file.endswith(".json"):
            table = {"name": key, "step": name, "op": point.get("op"), "of": point.get("args", {}).get("of"),
                     "at": at, "dims": dims, "mean": value.float().mean().item(), "rows": flatten(plan, point, value, dims)}
            with open(os.path.join(out, file), "w") as f:
                json.dump(table, f, indent=1)
        else:
            save_file({key: value.contiguous()}, os.path.join(out, file),
                      metadata={"dims": json.dumps(dims), "at": json.dumps(at)})
        described = {"file": file, "kind": "table" if file.endswith(".json") else "tensor", "dims": dims,
                     "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        result[key] = {"save": slot["save"], "value": described}
        saves.append({"name": key, "step": name, "at": at, "file": file})
    return {**point, "result": result}


def load(out: str) -> dict:
    """The filled plan an output directory holds, its tensors loaded; datasets carry `n_rows` in place of rows."""
    with open(os.path.join(out, "manifest.json")) as f:
        manifest = json.load(f)
    plan = {k: manifest[k] for k in ("format", "document", "description", "model", "data", "featurizers")}
    plan["sweeps"] = {s["label"]: {"path": s["path"], "values": s["values"]} for s in manifest["sweeps"]}
    plan["steps"] = manifest["steps"]
    for entry in plan["steps"].values():
        for point in points(entry):
            for key, slot in point["result"].items():
                slot["value"] = value_at(out, key, slot["value"])
    return plan


def value_at(out: str, key: str, described: dict) -> torch.Tensor:
    path = os.path.join(out, described["file"])
    if path.endswith(".safetensors"):
        return load_file(path)[key]
    with open(path) as f:
        rows = json.load(f)["rows"]
    dtype = getattr(torch, described["dtype"])
    return torch.tensor([r["value"] for r in rows], dtype=dtype).reshape(described["shape"])


def table(plan: dict, name: str) -> list[dict]:
    """One flat table over every point of a saved value: a row per element, its point's coordinates under `at`
    (kept apart from the row's own axes, which a sweep label may share: `pos_sweep.json` sweeps `pos`)."""
    rows = []
    for point in points(plan["steps"][name.partition(".")[0]]):
        value = point["result"][name]["value"]
        at = coordinates(plan, point["at"])
        rows += [{"at": at, **row} for row in flatten(plan, point, value, dims_of(plan, name))]
    return rows


def flatten(plan: dict, point: dict, value: torch.Tensor, dims: list[str]) -> list[dict]:
    """A row per element: its index on each axis (`rows` as `row`, `layers` as `layer`), then its value."""
    dataset = point.get("data") or point.get("input", {}).get("data")
    ids = [r.get("example_id") for r in plan["data"][dataset].get("rows", [])] if dataset else []
    values = value.tolist()
    rows = []
    for index in itertools.product(*map(range, value.shape)):
        row = {AXIS.get(axis, axis): i for axis, i in zip(dims, index)}
        if "rows" in dims and any(ids):
            row["example_id"] = ids[index[dims.index("rows")]]
        element = values
        for i in index:
            element = element[i]
        rows.append({**row, "value": element})
    return rows


def coordinates(plan: dict, at: dict) -> dict:
    """A point's coordinates as each axis's value, by label."""
    return {label: plan["sweeps"][label]["values"][i] for label, i in at.items()}


def without_tensors(featurizer: dict) -> dict:
    bundle = featurizer.get("bundle")
    return {**featurizer, "bundle": None if bundle is None else {"header": bundle["header"]}}


def versions() -> dict:
    found = {p: importlib.metadata.version(p) for p in ("clab", "nnsight", "nnterp", "transformers", "torch")}
    return {**found, "python": platform.python_version()}
