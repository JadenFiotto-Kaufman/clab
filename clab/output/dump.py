"""A filled plan into its output directory: one file per save, and the manifest."""

import json
import os
from typing import Any

from safetensors.torch import save
from torch import Tensor

from ..plan.model import (Plan, agreeing, arguments, coordinate, dims_of, entries, kind_of, labelled, name as named,
                          operand_site, points, uses)
from ..runtime import featurizers, ops, positions
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
            write(os.path.join(out, file), flat({name: value} if isinstance(value, Tensor) else value),
                  headed(plan, name, entry))
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


def write(path: str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    """A safetensors file whose header's keys are sorted: safetensors orders them as it likes, so the same
    tensors and metadata are the same bytes on every run. It replaces any file at `path` with a new one rather
    than writing over it, since tensors loaded from the old one may still be mapped from its bytes."""
    blob = save(tensors, metadata=metadata)
    n = int.from_bytes(blob[:8], "little")
    text = json.dumps(json.loads(blob[8:8 + n]), sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(f"{path}.partial", "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text + blob[8 + n:])
    os.replace(f"{path}.partial", path)


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


def headed(plan: Plan, name: str, entry: Any) -> dict[str, str] | None:
    """The header of a saved bundle: its one point's, or a swept one's every point's, each key after the point's
    coordinate and a `/` as its tensors are (`k=4/k`), so the file describes each bundle it holds."""
    if not isinstance(entry, list):
        return header(plan, name, plan.results[name], entry)
    found: dict[str, str] = {}
    for p in entry:
        point = header(plan, name, plan.results[name][coordinate(plan, p.at)], p)
        if point is None:
            return None
        found["format"] = point.pop("format")
        found.update({f"{labelled(plan, p.at)}/{k}": v for k, v in point.items()})
    return found


def header(plan: Plan, name: str, value: Any, step: Any) -> dict[str, str] | None:
    """A bundle's header, when the value is one (`value`, at `step`, the point that made it): an op step's whose op
    declares the featurizer kind its bundle is for, made at the site of the read it was taken over, or a fit's
    trained featurizer, made at the first site it is used at."""
    member = name.partition(".")[2]
    if step.kind == "op" and not member and ops.DECLARED[step.op]["bundle"]:
        kind, fields = ops.DECLARED[step.op]["bundle"], arguments(step)
        source = provenance(plan, name)[0] or ""
        site = entries(agreeing(plan.steps[source.partition(".")[0]], step.at))[source.partition(".")[2]].site
        reads = None
    elif step.kind == "fit" and member in step.train:
        kind, fields = kind_of(plan, member), agreeing(plan.featurizers[member], step.at).model_extra or {}
        used = uses(plan)[member]
        where, site, _, o, _ = next((u for u in used if all(step.at.get(k) == i for k, i in u[2].items())), used[0])
        reads = operand_site(plan, where, arguments(o)["with"]) if arguments(o).get("with") else None
    else:
        return None
    classes = featurizers.labels({n: materialized(plan, n) for n in plan.data}, fields["classes"]) \
        if "classes" in fields else None
    site, reads = (counted(plan, s) for s in (site, reads))
    return featurizers.bundles.header(kind, fields, plan.model, classes, site, featurizers.KINDS[kind].width(value),
                                      named(plan), reads)


def counted(plan: Plan, site: Any) -> Any:
    """A site whose one layer is counted from the first, as a header records it."""
    if site is not None and isinstance(site.layers, int) and plan.run is not None and plan.run.num_layers is not None:
        return site.model_copy(update={"layers": positions.select(site.layers, plan.run.num_layers)[0]})
    return site
