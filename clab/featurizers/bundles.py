"""Bundles: a featurizer's tensors in a safetensors file, and the header that says what made them."""

import inspect
import json
import os
from typing import Any

from safetensors import safe_open
from torch import Tensor

from . import FORMAT, KINDS

# The keys every clab header holds beside `format` and its kind's own fields (`header`, below, writes them).
OURS = ("kind", "model.key", "model.revision", "model.dtype", "component", "layer", "width", "document",
        "with.component", "with.layer")


def header(kind: str, fields: dict, model: Any, labels: list | None = None, site: Any = None,
           width: int | None = None, document: str | None = None, reads: Any = None) -> dict[str, str]:
    """A bundle's header: the kind and its header fields (with their defaults; a probe's classes as its class
    list), the model, and, when written, the site it was used at, its width and the document's name, and the
    site it read its input at when that was another (`reads`, a cross-site adapter's `with`: `with.component`,
    `with.layer`)."""
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
    if reads is not None:
        found.update({"with.component": reads.component, "with.layer": str(reads.layers)})
    return found


def load(root: str, kind: str, fields: dict, model: Any, labels: list | None = None
         ) -> tuple[dict[str, Tensor], dict[str, str] | None]:
    """The tensors and header of the bundle `fields["file_path"]` names under `root`: exactly the tensors the
    kind's bundle holds, and the header checked key by key against the featurizer that loads it and the
    plan's `model`. A file whose `format` is not clab's is a published one, the kind's to read
    (`Kind.published`), and has no header; one that carries clab's header keys under another format is
    refused."""
    file = os.path.join(root, fields["file_path"])
    if not os.path.exists(file):
        raise ValueError(f"{fields['file_path']} does not exist under the corpus root; run the document that "
                         "writes it first")
    try:
        with safe_open(file, "pt") as f:
            found = dict(sorted((f.metadata() or {}).items()))
            tensors = {k: f.get_tensor(k) for k in f.keys()}
    except Exception:           # safetensors' own error is not a ValueError, and says nothing of what to do
        raise ValueError(f"{fields['file_path']} is not a safetensors file, which a file_path is read as; convert "
                         "it first (Gemma Scope's params.npz: numpy.load, then safetensors.torch.save_file)") from None
    points = sorted({k.rpartition("/")[0] for k in tensors if "/" in k})
    if points:
        raise ValueError(f"{fields['file_path']} holds one bundle per point of a sweep ({', '.join(points)}); a "
                         "featurizer loads one bundle, so run the document that writes it unswept")
    if not found.get("format", "").startswith("clab/"):
        ours = [k for k in OURS if k in found]
        if ours:
            raise ValueError(f"{fields['file_path']} carries clab header keys ({', '.join(ours)}) but its format is "
                             f"{found.get('format')!r}, not {FORMAT!r}; a clab bundle's header is whole, a published "
                             "file carries none of them")
        return KINDS[kind].published(tensors, fields), None
    if set(tensors) != set(KINDS[kind].saves):
        raise ValueError(f"{fields['file_path']} holds {', '.join(sorted(tensors))}; a {kind} bundle holds "
                         f"{', '.join(KINDS[kind].saves)}")
    for key, value in header(kind, fields, model, labels).items():
        if found.get(key) != value:
            raise ValueError(f"{fields['file_path']}: header {key!r} is {found.get(key)!r}; this document "
                             f"expects {value!r}")
    return tensors, found
