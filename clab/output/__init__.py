"""A run's output directory and back.

One file per save, named after it: `<name>.safetensors` when the value is a tensor or a dict of tensors (a
bundle, `top_k`'s ids and probabilities), `<name>.json` otherwise, tensors in it as nested lists. A swept
save holds one entry per point, keyed by its coordinate, `pos=-1` (`layers=0,pos=-1` over two axes), in
safetensors and JSON alike, a dict's own keys after a `/`. A file is the value and
nothing else, except that a bundle a featurizer loads (the value of an op that declares one, `pca`, and a
fit's trained featurizer) carries the header that featurizer checks; a swept one holds each point's header keys
after its coordinate and a `/`, as its tensors are (`rank=2/rank`), beside one `format`.

`manifest.json` is the compiled plan as the run completed it, without the rows it read from files (a hub dataset's
never were there; `table` loads either again, from its file or the hub, and checks it against its `sha256`, as
`dump` does a probe's classes dataset for the header of the bundle it saves) and without bundle tensors, its
`results` each replaced by `{file, step, dims, <its labels>, axes, shape, dtype}`: the value's layout and labels as
its record holds them (`dims`, and those it has of `dataset`, `layers`, `read`, `width`, `kept`, `space`), once
where every point agrees, else per point by coordinate, as `shape` and `dtype` are, so that a result links back to
the step that made it and says what each point of it is. It records the document's path relative to the corpus
root, and each dataset's `n_rows`. The plan's `steps` hold each record with its labels and, on every read and
write, the positions it resolved to per row, which `table` uses to label positions; its `featurizers` each one's
width; its `run` holds how each input tokenized, what each generate step's rows generated, the model's sizes, and
what each step cost (`costs`: its seconds, its peak memory on each CUDA device, its model calls and the tokens
they fed). Values are never in the manifest.
"""

import json
import os
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save
from torch import Tensor

from clab import ops
from clab.compile import materialized
from clab.compile.uses import uses
from clab.featurizers import bundles, labels
from clab.format.sweeps import coordinate, labelled
from clab.plan import Entry, Intervention, Labelled, Plan, Run, Site, Step, agreeing, arguments, entry_of, kind_of, points, record_of, scope
from clab.plan import name as named

# The labels a result's manifest record carries beside `dims`, each where the value has one.
LABELS = ("dataset", "layers", "read", "width", "kept", "space")


def ran(plan: Plan) -> Run:
    """The run record of a plan that has run; one that has not is refused: it has no results to write or table."""
    if plan.run is None:
        raise ValueError("the plan has not run, so it has no results; execute it first")
    return plan.run


def dump(plan: Plan, out: str) -> None:
    ran(plan)
    os.makedirs(out, exist_ok=True)
    described = {}
    for name in (n for n in plan.saves if n in plan.results):        # every save, unless the run stopped
        entry = plan.steps[name.partition(".")[0]]
        value = plan.results[name]
        if isinstance(entry, list):
            value = {labelled(plan, p.at): value[coordinate(plan, p.at)] for p in entry}
        file = f"{name}.safetensors" if tensors(value) else f"{name}.json"
        if tensors(value):
            write(os.path.join(out, file), flat({name: value} if isinstance(value, Tensor) else value),
                  headed(plan, name, entry))
        else:
            with open(os.path.join(out, file), "w") as f:
                json.dump(value, f, default=lambda t: t.tolist(), ensure_ascii=False)
        described[name] = {"file": file, "step": name.partition(".")[0], **layouts(plan, name),
                           "axes": list(points(entry)[0].at), "shape": describe(value, "shape"),
                           "dtype": describe(value, "dtype")}
    manifest = plan.model_dump(mode="json", exclude_defaults=True)
    manifest["model"]["dtype"] = plan.model.dtype                 # recorded even where it is the default
    for d in manifest["data"].values():
        if "path" in d:
            d.pop("rows", None)             # a loaded plan's are not there
    with open(os.path.join(out, "manifest.json"), "w") as f:
        f.write(pretty({**manifest, "results": described}) + "\n")


def pretty(value: Any, level: int = 0) -> str:
    """JSON indented one space per level, as `json.dump(indent=1)` writes it, except that a list holding no object
    (positions per row, a shape, a `layers` label) stays on one line, so the manifest reads as rows."""
    pad = " " * (level + 1)
    if isinstance(value, dict) and value:
        inner = ",\n".join(f"{pad}{json.dumps(k, ensure_ascii=False)}: {pretty(v, level + 1)}" for k, v in value.items())
        return "{\n" + inner + "\n" + " " * level + "}"
    if isinstance(value, list) and any(isinstance(v, dict) for v in value):
        return "[\n" + ",\n".join(f"{pad}{pretty(v, level + 1)}" for v in value) + "\n" + " " * level + "]"
    return json.dumps(value, ensure_ascii=False)


def layouts(plan: Plan, name: str) -> dict[str, Any]:
    """A saved value's `dims` and labels (`LABELS`, those it has), as its record holds them: once where its step is
    not swept or every point agrees, else per point by coordinate, as the manifest records a result's `shape`."""
    entry = plan.steps[name.partition(".")[0]]
    per = {labelled(plan, p.at): layout(record_of(plan.steps, name, p.at)) for p in points(entry)}
    if not isinstance(entry, list):
        return per[""]
    out = {}
    for key in ("dims", *LABELS):
        each = {c: v.get(key) for c, v in per.items()}
        if any(v is not None for v in each.values()):
            out[key] = next(iter(each.values())) if len({json.dumps(v) for v in each.values()}) == 1 else each
    return out


def layout(record: Labelled) -> dict[str, Any]:
    """A value's `dims` and the labels it has, as JSON (`dims` always: a value with no axes has `[]`)."""
    return {"dims": record.dims, **{k: v for k in LABELS if (v := getattr(record, k)) not in (None, {}, [])}}


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


def flat(value: dict[str, Any], prefix: str = "") -> dict[str, Tensor]:
    out = {}
    for k, v in value.items():
        out.update(flat(v, f"{prefix}{k}/") if isinstance(v, dict) else {prefix + k: v.contiguous()})
    return out


def describe(value: Any, what: Literal["shape", "dtype"]) -> list[int] | str | dict[str, Any] | None:
    """A value's `shape` or `dtype`, following its structure: a tensor's own, a dict's per key, a nested list's
    as far as its lists are of one length (`top_k`'s tokens, `[rows, pos, k]`; and no dtype)."""
    if isinstance(value, Tensor):
        return list(value.shape) if what == "shape" else str(value.dtype).removeprefix("torch.")
    if isinstance(value, dict):
        return {k: describe(v, what) for k, v in value.items()}
    if not isinstance(value, list) or what != "shape":
        return None
    shape, level = [len(value)], value
    while level and all(isinstance(v, list) for v in level) and len({len(v) for v in level}) == 1:
        shape.append(len(level[0]))
        level = [x for v in level for x in v]
    return shape


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
                value = restored(json.load(f), d["dtype"])
        entry = plan.steps[d["step"]]
        if isinstance(entry, list):
            value = {coordinate(plan, p.at): value[labelled(plan, p.at)] for p in entry}
        plan.results[name] = value
    return plan


def nested(value: dict[str, Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, v in value.items():
        *path, last = key.split("/")
        here = out
        for k in path:
            here = here.setdefault(k, {})
        here[last] = v
    return out


def restored(value: Any, dtype: str | dict[str, Any] | None) -> Any:
    """A value read from JSON with every entry its `dtype` says was a tensor a tensor again."""
    if isinstance(dtype, str):
        return torch.tensor(value, dtype=getattr(torch, dtype))
    if isinstance(dtype, dict):
        return {k: restored(value[k], dtype[k]) for k in value}
    return value


def headed(plan: Plan, name: str, entry: Entry) -> dict[str, str] | None:
    """The header of a saved bundle: its one point's, or a swept one's every point's, each key after the point's
    coordinate and a `/` as its tensors are (`k=4/k`), so the file describes each bundle it holds."""
    if not isinstance(entry, list):
        return header(plan, name, entry)
    found: dict[str, str] = {}
    for p in entry:
        point = header(plan, name, p)
        if point is None:
            return None
        found["format"] = point.pop("format")
        found.update({f"{labelled(plan, p.at)}/{k}": v for k, v in point.items()})
    return found


def header(plan: Plan, name: str, step: Step) -> dict[str, str] | None:
    """A bundle's header, when the value is one (at `step`, the point that made it), written from the records: an
    op step's whose op declares the featurizer kind its bundle is for, made at the site of the read it was taken
    over, as wide as that read's feature axis; or a fit's trained featurizer, made at the first site it is used at,
    as wide as its record says (`Featurizer.width`)."""
    assert plan.run is not None
    member = name.partition(".")[2]
    if step.kind == "op" and ops.DECLARED[step.op]["bundle"]:
        kind, fields = ops.DECLARED[step.op]["bundle"], arguments(step)
        value = record_of(plan.steps, name, step.at)
        read = record_of(plan.steps, value.read or "", step.at)
        site = at_layers(read)
        width = len(value.kept["hidden_size"]) if "hidden_size" in value.kept else plan.run.sizes["hidden_size"]
        reads = None
    elif step.kind == "fit" and member in step.train:
        trained = agreeing(plan.featurizers[member], step.at)
        kind, fields = kind_of(plan, member), trained.model_extra or {}
        used = uses(plan)[member]
        where, read, _, o, _ = next((u for u in used if all(step.at.get(k) == i for k, i in u[2].items())), used[0])
        given = arguments(o).get("with")
        site, width = at_layers(read), trained.width
        reads = at_layers(entry_of(scope(plan, where), given, step.at)[1]) if given else None
    else:
        return None
    dataset = fields["classes"].partition(".")[0] if "classes" in fields else None
    classes = labels({dataset: materialized(plan, dataset)}, fields["classes"]) if dataset else None
    return bundles.header(kind, fields, plan.model, classes, site, width, named(plan), reads)


def at_layers(read: Labelled) -> Site:
    """A read's site with the layers its record says it was made at, counted from the first, as a header records
    them: one int for one layer, a list for several."""
    assert isinstance(read, Intervention)
    site, made = read.site, read.layers
    assert isinstance(site, Site)
    return site if made is None else site.model_copy(update={"layers": made[0] if len(made) == 1 else made})
