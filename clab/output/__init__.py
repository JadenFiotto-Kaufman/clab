"""A run's output directory and back.

One file per save, named after it: `<name>.safetensors` when the value is a tensor or a dict of tensors (a
bundle, `top_k`'s ids and probabilities), `<name>.json` otherwise, tensors in it as nested lists. A swept
save holds one entry per point of the value's own axes (`axes`: those of its step's it depends on), keyed by its
coordinate, `pos=-1` (`layers=0,pos=-1` over two axes), in safetensors and JSON alike, a dict's own keys after a
`/`. A file is the value and
nothing else, except that a bundle a featurizer loads (the value of an op that declares one, `pca`, and a
fit's trained featurizer) carries the header that featurizer checks; a swept one holds each point's header keys
after its coordinate and a `/`, as its tensors are (`rank=2/rank`), beside one `format`.

`manifest.json` is the compiled plan as the run completed it, without the rows it read from files (a hub dataset's
never were there; `table` loads either again, from its file or the hub, and checks it against its `sha256`, as
`dump` does a probe's classes dataset for the header of the bundle it saves) and without bundle tensors, its
`results` each replaced by `{file, step, dims, <its labels>, axes, shape, dtype}`: the value's layout and labels as
its record holds them (`dims`, and those it has of `dataset`, `layers`, `read`, `width`, `kept`, `space`), once
where every point agrees, else per point by coordinate, as `shape` and `dtype` are, and `axes`, the sweep axes the
value depends on, so that a result links back to the step that made it and says what each point of it is. It
records the document's path relative to the corpus root, and each dataset's `n_rows`. The plan's `steps` hold
each record with its labels and, on every read and write, the positions it resolved to per row, which `table` uses
to label positions, and each forward's and generation's `schedule`, which forward served each point and what it
did. The manifest holds each distinct list of positions per row once, in its top-level `positions`, and a record's
`positions` names its list there by index under its dataset (`{"train": 0}`), so records that resolved alike share
one; a record holding its list itself reads as well. The plan's
`featurizers` hold each one's width; its `run` holds the batch size, `padding` and `readout` it ran at (the
padding and the readout even where they are the defaults), the thread count torch computed with (`threads`), the
model's sizes, and what each run of each step cost (`costs`: its seconds, its peak memory on each CUDA device, its
model calls, the tokens they fed, padding included, and of those the rows' own, `real`, and the points it served).
Values are never in the manifest. `tokens.json` beside it is the run's `tokens`, how each input tokenized and what
each generate step's rows generated, a token string per position of every row, which `table` labels positions with:
kept apart, since at many rows they are nearly all of what the manifest would hold, and `load` reads it back
into `run.tokens`.
"""

import json
import os
from typing import Any, Iterator, Literal

import torch
from safetensors.torch import load_file, save
from torch import Tensor

from clab import featurizers, ops
from clab.compile import dataset_rows
from clab.compile.uses import uses
from clab.featurizers import bundles, labels
from clab.format.sweeps import agreeing_point, coordinate, coordinate_text, points
from clab.plan import (Intervention, Labelled, OpStep, Plan, Run, Site, Step, arguments, entries, entry_of, kind_of,
                       made_by, projections, record_of, saved_value_axes, scope, walk)
from clab.plan import name as named

# The labels a result's manifest record carries beside `dims`, each where the value has one.
LABELS = ("dataset", "layers", "read", "width", "kept", "space")


def run_record(plan: Plan) -> Run:
    """The run record of a plan that has run; one that has not is refused: it has no results to write or table."""
    if plan.run is None:
        raise ValueError("the plan has not run, so it has no results; execute it first")
    return plan.run


def dump(plan: Plan, out: str) -> None:
    run = run_record(plan)
    os.makedirs(out, exist_ok=True)
    described = {}
    for name in (n for n in plan.saves if n in plan.results):        # every save, unless the run stopped
        value, axes = plan.results[name], saved_value_axes(plan, name)
        if axes:
            value = {coordinate_text(plan, at): value[coordinate(plan, at)] for at, _ in projections(plan, name)}
        file = f"{name}.safetensors" if tensors(value) else f"{name}.json"
        if tensors(value):
            write_safetensors(os.path.join(out, file), flat({name: value} if isinstance(value, Tensor) else value),
                              bundle_header(plan, name))
        else:
            with open(os.path.join(out, file), "w") as f:
                json.dump(value, f, default=lambda t: t.tolist(), ensure_ascii=False)
        described[name] = {"file": file, "step": made_by(plan, name), **saved_value_layout(plan, name),
                           "axes": axes, "shape": shape_or_dtype(value, "shape"),
                           "dtype": shape_or_dtype(value, "dtype")}
    manifest = plan.model_dump(mode="json", exclude_defaults=True, exclude={"run": {"tokens"}})
    manifest["model"]["dtype"] = plan.model.dtype                 # recorded even where it is the default
    manifest["run"]["padding"] = run.padding                       # so is the padding
    manifest["run"]["readout"] = run.readout                       # and the readout
    for d in manifest["data"].values():
        if "path" in d:
            d.pop("rows", None)             # a loaded plan's are not there
    table: list[list] = []                  # each distinct list of positions per row, once
    for entry in held_entries(manifest["steps"]):
        for dataset, given in entry.get("positions", {}).items():
            index = next((i for i, p in enumerate(table) if p == given), len(table))
            table += [given] if index == len(table) else []
            entry["positions"][dataset] = index
    text = pretty_json({**manifest, "results": described})
    if table:                               # the shared positions last, each list of them on a line of its own
        lines = ",\n".join(f"  {json.dumps(p, ensure_ascii=False)}" for p in table)
        text = text.removesuffix("\n}") + ',\n "positions": [\n' + lines + "\n ]\n}"
    with open(os.path.join(out, "manifest.json"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "tokens.json"), "w") as f:
        f.write(pretty_json(run.tokens) + "\n")


def held_entries(steps: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every read and write a manifest's `steps` hold, as JSON: each point's entries, a `load`'s read's, a fit's
    body's and its evals'."""
    for entry in steps.values():
        for step in points(entry):
            yield from step.get("interventions", {}).values()
            yield from step.get("source", {}).get("interventions", {}).values()
            yield from held_entries(step.get("steps", {}))
            for evaluated in step.get("evals", {}).values():
                yield from held_entries(evaluated)


def pretty_json(value: Any, level: int = 0) -> str:
    """JSON indented one space per level, as `json.dump(indent=1)` writes it, except that a list holding no object
    (positions per row, a shape, a `layers` label) stays on one line, so the manifest reads as rows."""
    pad = " " * (level + 1)
    if isinstance(value, dict) and value:
        inner = ",\n".join(f"{pad}{json.dumps(k, ensure_ascii=False)}: {pretty_json(v, level + 1)}"
                           for k, v in value.items())
        return "{\n" + inner + "\n" + " " * level + "}"
    if isinstance(value, list) and any(isinstance(v, dict) for v in value):
        return "[\n" + ",\n".join(f"{pad}{pretty_json(v, level + 1)}" for v in value) + "\n" + " " * level + "]"
    return json.dumps(value, ensure_ascii=False)


def saved_value_layout(plan: Plan, name: str) -> dict[str, Any]:
    """A saved value's `dims` (always: a value with no axes has `[]`) and labels (`LABELS`, those it has), as its
    record holds them: once where the value depends on no axis or every point of it agrees, else per point by
    coordinate, as the manifest records a result's `shape`."""
    per = {}
    for at, p in projections(plan, name):
        record = record_of(plan.steps, name, p.at, plan.featurizers)
        per[coordinate_text(plan, at)] = {"dims": record.dims, **{k: v for k in LABELS
                                                                  if (v := getattr(record, k)) not in (None, {}, [])}}
    if list(per) == [""]:
        return per[""]
    out = {}
    for key in ("dims", *LABELS):
        each = {c: v.get(key) for c, v in per.items()}
        if any(v is not None for v in each.values()):
            out[key] = next(iter(each.values())) if len({json.dumps(v) for v in each.values()}) == 1 else each
    return out


def write_safetensors(path: str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
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


def shape_or_dtype(value: Any, what: Literal["shape", "dtype"]) -> list[int] | str | dict[str, Any] | None:
    """A value's `shape` or `dtype`, following its structure: a tensor's own, a dict's per key, a nested list's
    as far as its lists are of one length (`top_k`'s tokens, `[rows, pos, k]`; and no dtype)."""
    if isinstance(value, Tensor):
        return list(value.shape) if what == "shape" else str(value.dtype).removeprefix("torch.")
    if isinstance(value, dict):
        return {k: shape_or_dtype(v, what) for k, v in value.items()}
    if not isinstance(value, list) or what != "shape":
        return None
    shape, level = [len(value)], value
    while level and all(isinstance(v, list) for v in level) and len({len(v) for v in level}) == 1:
        shape.append(len(level[0]))
        level = [x for v in level for x in v]
    return shape


def load(out: str) -> Plan:
    held, described = manifest(out)
    plan = Plan(**held)
    shared: list[list] = []                 # records whose positions are equal hold one list, as the run's alike did
    for record in recorded(plan.steps):
        for dataset, given in record.positions.items():
            record.positions[dataset] = next((p for p in shared if p == given), given)
            shared += [given] if record.positions[dataset] is given else []
    tokens = os.path.join(out, "tokens.json")
    if plan.run is not None and os.path.exists(tokens):
        with open(tokens) as f:
            plan.run.tokens = json.load(f)
    elif plan.run is not None and not plan.run.tokens:       # neither beside the manifest nor, as before, in it
        raise FileNotFoundError(f"{out}/tokens.json is missing, and the manifest holds no tokens: rerun the "
                                "document that wrote it")
    for name, d in described.items():
        value = read_saved_value(out, name, d)
        if d["axes"]:
            value = {coordinate(plan, at): value[coordinate_text(plan, at)] for at, _ in projections(plan, name)}
        for at, p in projections(plan, name):   # a top_k's ids over a list of axes in its order, which the file sorts
            record = record_of(plan.steps, name, p.at, plan.featurizers)
            over = arguments(record).get("over") if isinstance(record, OpStep) and record.op == "top_k" else None
            if isinstance(over, list):
                point = value[coordinate(plan, at)] if d["axes"] else value
                point["ids"] = {a: point["ids"][a] for a in over}
        plan.results[name] = value
    return plan


def recorded(steps: dict[str, Any]) -> Iterator[Intervention]:
    """Every read and write of a plan's `steps`, the records `held_entries` finds in its manifest: each point's
    entries, a `load`'s read's, a fit's evals' and its body's."""
    for _, step in walk(steps):
        hosts = [step, getattr(step, "source", None), *(s for e in getattr(step, "evals", {}).values()
                                                         for s in e.values())]
        for host in hosts:
            yield from entries(host).values() if host is not None else ()


def manifest(out: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The manifest of the run at `out`, as JSON, each record's `positions` its lists (an index into the
    manifest's `positions` the list there, records of one index holding one list), and apart from it its `results`:
    each saved value's record by name. A manifest whose value records have no `axes` is refused by name: it cannot
    say which sweeps a value depends on, and taken as unswept a swept value's points would load as one. So is one
    whose fit has an `eval` and no `evals`, which holds no record of an eval's values."""
    try:
        with open(os.path.join(out, "manifest.json")) as f:
            held = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"no run at {out!r} (no manifest.json); run the document first") from None
    if any("axes" not in d for d in held.get("results", {}).values()):
        raise ValueError(f"{out}/manifest.json: a saved value's record holds no 'axes', so it does not say which "
                         "sweeps its values depend on; rerun the document that wrote it")
    for name, entry in held.get("steps", {}).items():
        if any(p.get("kind") == "fit" and p.get("eval") and "evals" not in p for p in points(entry)):
            raise ValueError(f"{out}/manifest.json: fit {name!r} has an 'eval' and no 'evals', so it holds no "
                             "record of its evals' values; rerun the document that wrote it")
    table = held.pop("positions", [])
    for entry in held_entries(held.get("steps", {})):
        for dataset, given in entry.get("positions", {}).items():
            entry["positions"][dataset] = table[given] if isinstance(given, int) else given
    return held, held.pop("results")


def read_saved_value(out: str, name: str, d: dict[str, Any]) -> Any:
    """Saved value `name` as its file under `out` holds it, `d` its record in the manifest's `results`: a tensor or
    a dict of them again, a swept one keyed by each point's coordinate as text (`pos=-1`)."""
    file = os.path.join(out, d["file"])
    if file.endswith(".safetensors"):
        value: dict[str, Any] = {}
        for key, tensor in load_file(file).items():         # each `/` a level of the dict `flat` wrote
            *path, last = key.split("/")
            here = value
            for k in path:
                here = here.setdefault(k, {})
            here[last] = tensor
        return value[name] if isinstance(d["dtype"], str) else value
    with open(file) as f:
        return restore_tensors(json.load(f), d["dtype"])


def restore_tensors(value: Any, dtype: str | dict[str, Any] | None) -> Any:
    """A value read from JSON with every entry its `dtype` says was a tensor a tensor again."""
    if isinstance(dtype, str):
        return torch.tensor(value, dtype=getattr(torch, dtype))
    if isinstance(dtype, dict):
        return {k: restore_tensors(value[k], dtype[k]) for k in value}
    return value


def bundle_header(plan: Plan, name: str) -> dict[str, str] | None:
    """The header of a saved bundle: its one point's, or a swept one's every point's, each key after the point's
    coordinate and a `/` as its tensors are (`k=4/k`), so the file describes each bundle it holds."""
    ats = projections(plan, name)
    if not saved_value_axes(plan, name):
        return point_header(plan, name, ats[0][1])
    header: dict[str, str] = {}
    for at, p in ats:
        point = point_header(plan, name, p)
        if point is None:
            return None
        header["format"] = point.pop("format")
        header.update({f"{coordinate_text(plan, at)}/{k}": v for k, v in point.items()})
    return header


def point_header(plan: Plan, name: str, step: Step) -> dict[str, str] | None:
    """A bundle's header, when the value is one (at `step`, the point that made it), written from the records: an
    op step's whose op declares the featurizer kind its bundle is for, made at the site of the read it was taken
    over, as wide as that read's feature axis; or a fit's trained featurizer, made at the first site it is used at,
    as wide as its record says (`Featurizer.width`), its layers, for one over them, those its record keeps. Either way
    the read's selection is the one its record keeps (`kept`)."""
    assert plan.run is not None
    member = name.partition(".")[2]
    if step.kind == "op" and ops.DECLARED[step.op]["bundle"]:
        kind, fields = ops.DECLARED[step.op]["bundle"], arguments(step)
        value = record_of(plan.steps, name, step.at, plan.featurizers)
        read = entry_of(plan.steps, value.read or "", step.at)[1]
        site = site_as_recorded(read)
        width = len(value.kept["hidden_size"]) if "hidden_size" in value.kept else plan.run.sizes["hidden_size"]
        reads, kept = None, None
    elif step.kind == "fit" and member in step.train:
        trained = agreeing_point(plan.featurizers[member], step.at)
        kind, fields = kind_of(plan, member), trained.model_extra or {}
        used = uses(plan)[member]
        where, read, _, o, _ = next((u for u in used if all(step.at.get(k) == i for k, i in u[2].items())), used[0])
        given = arguments(o).get("with") if featurizers.KINDS[kind].adapts else None    # an adapter's input site
        site, width, kept = site_as_recorded(read, trained.kept.get("layers")), trained.width, trained.kept
        reads = site_as_recorded(entry_of(scope(plan, where), given, step.at)[1]) if given else None
    else:
        return None
    dataset = fields["classes"].partition(".")[0] if "classes" in fields else None
    classes = labels({dataset: dataset_rows(plan, dataset)}, fields["classes"]) if dataset else None
    return bundles.header(kind, fields, plan.model, classes, site, width, named(plan), reads, kept, read.kept)


def site_as_recorded(read: Labelled, layers: list[int] | None = None) -> Site:
    """A read's site with the layers its record says it was made at (or `layers`, a featurizer over them's own, in the
    order its tensors lay them out), counted from the first, as a header records them: one int for one layer, a list
    for several."""
    assert isinstance(read, Intervention)
    site, made = read.site, read.layers if layers is None else layers
    assert isinstance(site, Site)
    return site if made is None else site.model_copy(update={"layers": made[0] if len(made) == 1 else made})
