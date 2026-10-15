"""Bundles: a featurizer's tensors in a safetensors file, and the header that says what made them."""

import inspect
import json
import os
from typing import Any

from safetensors import safe_open
from torch import Tensor

from . import FORMAT, KINDS


def header(kind: str, fields: dict, model: Any, labels: list | None = None, site: Any = None,
           width: int | None = None, document: str | None = None) -> dict[str, str]:
    """A bundle's header: the kind and its header fields (with their defaults; a probe's classes as its class
    list), the model, and, when written, the site it was used at, its width and the document's name."""
    params = inspect.signature(KINDS[kind]).parameters
    given = {**{k: p.default for k, p in params.items()}, **fields, "classes": labels}
    # A field a loading document leaves out is the header's to say, so it is not checked; 8 and 8.0 are one value.
    own = {k: json.dumps(v) if isinstance(v, (list, bool)) else str(int(v)) if isinstance(v, float) and v.is_integer()
           else str(v) for k in KINDS[kind].header if (v := given[k]) is not None}
    found = {"format": FORMAT, "kind": kind, **own, "model.key": model.key, "model.revision": model.revision,
             "model.dtype": model.dtype}
    if site is not None:
        found.update({"component": site.component, "layer": str(site.layers), "width": str(width),
                      "document": str(document)})
    return found


def load(root: str, kind: str, fields: dict, model: Any, labels: list | None = None
         ) -> tuple[dict[str, Tensor], dict[str, str] | None]:
    """The tensors and header of the bundle `fields["file_path"]` names under `root`: exactly the tensors the
    kind's bundle holds, and the header checked key by key against the featurizer that loads it and the
    plan's `model`; a header field the file leaves out is that field's default. A file with no clab header
    (no `format`) is the kind's to read (`Kind.headerless`), and has no header."""
    file = os.path.join(root, fields["file_path"])
    if not os.path.exists(file):
        raise ValueError(f"{fields['file_path']} does not exist under the corpus root; run the document that "
                         "writes it first")
    with safe_open(file, "pt") as f:
        found = dict(sorted((f.metadata() or {}).items()))
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    points = sorted({k.rpartition("/")[0] for k in tensors if "/" in k})
    if points:
        raise ValueError(f"{fields['file_path']} holds one bundle per point of a sweep ({', '.join(points)}); a "
                         "featurizer loads one bundle, so run the document that writes it unswept")
    if "format" not in found:
        return KINDS[kind].headerless(tensors, fields), None
    if set(tensors) != set(KINDS[kind].saves):
        raise ValueError(f"{fields['file_path']} holds {', '.join(sorted(tensors))}; a {kind} bundle holds "
                         f"{', '.join(KINDS[kind].saves)}")
    implied = {k: v for k, v in header(kind, {}, model, labels).items() if k in KINDS[kind].header}
    for key, value in header(kind, fields, model, labels).items():
        if found.get(key, implied.get(key)) != value:
            raise ValueError(f"{fields['file_path']}: header {key!r} is {found.get(key)!r}; this document "
                             f"expects {value!r}")
    return tensors, found
