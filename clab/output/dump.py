"""A filled plan into its output directory: one file per save, and the manifest."""

import json
import os
from typing import Any

from safetensors.torch import save_file
from torch import Tensor

from ..plan.model import Plan, arguments, coordinate, dims_of, kind_of, labelled, name as named, points, uses
from ..runtime import featurizers, ops
from .table import materialized, provenance


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
                      metadata=header(plan, name, plan.results[name] if not isinstance(entry, list) else
                                      next(iter(plan.results[name].values()))))
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


def tensors(value: Any) -> bool:
    return isinstance(value, Tensor) or isinstance(value, dict) and all(map(tensors, value.values()))


def flat(value: dict, prefix: str = "") -> dict[str, Tensor]:
    out = {}
    for k, v in value.items():
        out.update(flat(v, f"{prefix}{k}/") if isinstance(v, dict) else {prefix + k: v.contiguous()})
    return out


def describe(value: Any, what: str) -> Any:
    """A value's `shape` or `dtype`, following its structure: a tensor's own, a dict's per key, a list's
    length (and no dtype)."""
    if isinstance(value, Tensor):
        return list(value.shape) if what == "shape" else str(value.dtype).removeprefix("torch.")
    if isinstance(value, dict):
        return {k: describe(v, what) for k, v in value.items()}
    return [len(value)] if isinstance(value, list) and what == "shape" else None


def header(plan: Plan, name: str, value: Any) -> dict[str, str] | None:
    """A bundle's header, when the value is one (`value`, one point's): an op step's whose op declares the
    featurizer kind its bundle is for, made at the site of the read it was taken over, or a fit's trained
    featurizer, made at the first site it is used at."""
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if step.kind == "op" and not member and ops.DECLARED[step.op]["bundle"]:
        kind, fields = ops.DECLARED[step.op]["bundle"], arguments(step)
        source = provenance(plan, name)[0] or ""
        site = points(plan.steps[source.partition(".")[0]])[0].reads[source.partition(".")[2]].site
    elif step.kind == "fit" and member in step.train:
        kind, fields = kind_of(plan, member), points(plan.featurizers[member])[0].model_extra or {}
        site = uses(plan)[member][0][1]
    else:
        return None
    classes = featurizers.labels({n: materialized(plan, n) for n in plan.data}, fields["classes"]) \
        if "classes" in fields else None
    return featurizers.header(kind, fields, plan.model, classes, site, featurizers.width(kind, value), named(plan))
