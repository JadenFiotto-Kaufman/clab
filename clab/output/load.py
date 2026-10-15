"""An output directory back into the filled plan: the inverse of `dump`."""

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from ..plan.model import Plan, coordinate, labelled


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
