"""A run's output directory and back.

One file per save, named after it: `<name>.safetensors` when the value is a tensor or a dict of tensors (a
bundle, `top_k`'s ids and probabilities), `<name>.json` otherwise, tensors in it as nested lists. A swept
save holds one entry per point, keyed by its coordinate: `pos=-1` in safetensors, `"-1"` in JSON
(`layers=0,pos=-1` and `"0,-1"` over two axes), a dict's own keys after a `/`. A file is the value and
nothing else, except that a bundle a featurizer loads (the value of an op that declares one, `pca`, and a
fit's trained featurizer) carries the header that featurizer checks; a swept one holds each point's header keys
after its coordinate and a `/`, as its tensors are (`rank=2/rank`), beside one `format`.

`manifest.json` is the compiled plan without the rows it read from files and without bundle tensors, its
`results` each replaced by `{file, step, dims, axes, shape, dtype}`: that is how a result links back to the
step that made it. It records the document's path relative to the corpus root, and each dataset's `n_rows`.
`shape` and `dtype` follow the value's structure (per key, per point). The plan's `run` holds how each input
tokenized, what each generate step's rows generated, and which position (as `pos` counts them: an index into the
row's prompt tokens, BOS first, then its generated ones) every read and write resolved to per row, which
`table` uses to label positions. Values are never in the manifest.
"""

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file, save
from torch import Tensor

from .featurizers.uses import headed
from .plan import Plan, dims_of, points
from .sweeps import coordinate, labelled


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
        if "path" in d:
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


def nested(value: dict[str, Tensor]) -> dict:
    out: dict = {}
    for key, v in value.items():
        *path, last = key.split("/")
        here = out
        for k in path:
            here = here.setdefault(k, {})
        here[last] = v
    return out


def restored(value: Any, dtype: Any) -> Any:
    """A loaded value with every entry its `dtype` says was a tensor a tensor again."""
    if isinstance(dtype, str):
        return value if isinstance(value, Tensor) else torch.tensor(value, dtype=getattr(torch, dtype))
    if isinstance(dtype, dict):
        return {k: restored(value[k], dtype[k]) for k in value}
    return value
