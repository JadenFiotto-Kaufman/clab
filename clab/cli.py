"""`clab run`, `clab show`, `clab table`, `clab describe` and `clab schema`: a thin command line over
`clab.run` and a run's output directory, and the format read off the code.

`show` prints what a run's directory holds (each result with its step, dims, axes and shape, and how the
run ran); `table` prints `plan.table(...)` over its saved values, aligned or as CSV. Both are `Plan.load`
and one call.

`describe` prints the format's tables by introspection: the records and their fields (from the plan's
models), the ops and their signatures, the components, the featurizer kinds and the position grammar.
`describe --markdown` renders all of it as `FORMAT.md`; `schema` prints the plan's JSON Schema.
"""

import argparse
import csv
import inspect
import io
import json
import os
import sys
from typing import Any, get_args

from pydantic import BaseModel, TypeAdapter

from . import address, compiler, featurizers, ops, output, plan
from .registry import spell as spell_type

WHAT = ("records", "steps", "ops", "components", "featurizers", "pos")
RECORDS = {"`model`": plan.Model, "`data.<name>`": plan.Data, "`sites.<name>`, or a `site` inline": plan.Site,
           "A write or a read": plan.Access, "An op in a write or a read": plan.Op, "`pos`": plan.Pos,
           "`interventions.<name>`": plan.Interventions}
STEPS = {name: model for name, model in plan.KINDS.items()}
FIT = {"optimizer": plan.Optimizer, "early_stop": plan.EarlyStop}


NAMES = {plan.Sweep: "a sweep", plan.Site: "site", plan.Pos: "pos", plan.Scope: "scope", plan.Op: "op",
         plan.Access: "write or read", plan.Optimizer: "optimizer", plan.EarlyStop: "early_stop",
         plan.Entry: "step"}


def spell(annotation: Any) -> str:
    return spell_type(annotation, NAMES)


def fields(model: type[BaseModel]) -> list[str]:
    """Each field a document writes: `name: type = default`."""
    out = []
    for name in plan.documented(model):
        if name == "kind":
            continue
        field = model.model_fields[name]
        text = f"{name}: {spell(field.annotation)}"
        out.append(text if field.is_required() else f"{text} = {field.default!r}")
    return out


def derived(model: type[BaseModel]) -> list[str]:
    return [f"`{name}` ({' '.join((f.description or '').split())})" for name, f in model.model_fields.items()
            if name not in plan.documented(model)]


def doc(thing) -> str:
    return inspect.getdoc(thing) or ""


def summary(thing) -> str:
    return " ".join(doc(thing).split())


def axes(name: str) -> str:
    if name == "featurizer":
        return "its mode's (see Featurizer kinds)"
    a = ops.DECLARED[name]

    def listed(verb: str) -> str:
        if not isinstance(a[verb], tuple):
            return f"the axes `{a[verb]}` names"
        if verb == "removes" and a[verb] == ops.VOCABULARY:
            return "its vocabulary axis"
        return ", ".join(" or ".join(x) if isinstance(x, tuple) else x for x in a[verb])
    parts = [f"{verb} {listed(verb)}" for verb in ("needs", "removes", "adds") if a[verb]]
    return "; ".join(parts) or "keeps the layout"


def arguments(name: str) -> list[str]:
    """An op's arguments as a document writes them; `...` where its mode's own follow (the featurizer op)."""
    out = []
    for key, p in ops.parameters(name).items():
        text = f"{key}: {spell_type(p.annotation)}"
        out.append(text if p.default is p.empty else f"{text} = {p.default!r}")
    takes_more = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(ops.OPS[name]).parameters.values())
    return out + ["..."] * takes_more


def modes(kind: str) -> list[str]:
    """Each mode a featurizer kind has, with what it produces and the mode's own arguments."""
    out = []
    for m in featurizers.modes(kind):
        fn = featurizers.mode(kind, m)
        own = [f"{k}: {spell_type(p.annotation)}" + ("" if p.default is p.empty else f" = {p.default!r}")
               for k, p in featurizers.registry.spelled(fn, ("f", "h")).items()]
        produces = featurizers.MODES[m]["produces"].format(axis=featurizers.DECLARED[kind]["axis"])
        out.append(f"`{m}`" + (f"({', '.join(own)})" if own else "") + f" gives {produces}"
                   + (f": {summary(fn)}" if doc(fn) else ""))
    return out


def schema() -> dict:
    """The document's JSON Schema: the plan's, without the fields compile and the run fill, with an op's
    arguments spelled out per op."""
    found = plan.Plan.model_json_schema()

    def strip(node: Any) -> None:
        if isinstance(node, dict):
            for key in [k for k, v in node.get("properties", {}).items() if isinstance(v, dict) and v.get("derived")]:
                del node["properties"][key]
                node["required"] = [k for k in node.get("required", []) if k != key]
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(found)
    for key in ("Op", "OpStep"):
        found["$defs"][key] = per_op(found["$defs"][key])
    return found


def per_op(base: dict) -> dict:
    """An op record's schema as one branch per op: its own fields beside the op's arguments, nothing else."""
    sweep = {"$ref": "#/$defs/Sweep"}
    kinds = {ops.Value: {"type": ["number", "string"]}, ops.Tokens: {"type": "string"},
             ops.Featurizer: {"type": "string"}}
    branches = []
    for name in ops.OPS:
        params = {k: p for k, p in ops.parameters(name).items() if k not in base["properties"]}
        if name == "featurizer":            # every mode's own arguments, which compile binds to the kind's mode
            params.update({k: p for table in featurizers.MODES.values() for fn in table["table"].values()
                           for k, p in featurizers.registry.spelled(fn, ("f", "h")).items()})
        # An optional thing (`Value | None`) is written as the thing, or left out.
        named = {k: next((a for a in get_args(p.annotation) if a in kinds), p.annotation) for k, p in params.items()}
        args = {k: {"anyOf": [kinds.get(named[k]) or TypeAdapter(p.annotation).json_schema(), sweep]}
                for k, p in params.items()}
        if name == "featurizer":
            args["mode"] = {"enum": list(featurizers.MODES)}
        branches.append({"type": "object", "properties": {**base["properties"], "op": {"const": name}, **args},
                         "required": base.get("required", []) + [k for k, p in ops.parameters(name).items()
                                                                 if p.default is p.empty],
                         "additionalProperties": False})
    return {"title": base["title"], "description": base.get("description", ""), "oneOf": branches}


def describe(what: str | None = None) -> str:
    """The format's tables, read off the code: one of `records`, `steps`, `ops`, `components`, `featurizers`,
    `pos`, or all of them."""
    sections = []
    if what in (None, "records"):
        sections.append("\n".join(["records:"] + [f"  {name.replace('`', '')}({', '.join(fields(model))})"
                                                  for name, model in RECORDS.items()]))
    if what in (None, "steps"):
        lines = ["step kinds:"]
        for name, model in {**STEPS, **FIT}.items():
            lines += [f"  {name}({', '.join(fields(model))})", f"      {summary(model)}"]
        sections.append("\n".join(lines))
    if what in (None, "ops"):
        lines = ["ops (`of` is the value acted on; inside a read or write, the selection there):"]
        for group in ops.GROUPS:
            lines.append(f"  {group}:")
            for name, fn in ops.OPS.items():
                if ops.DECLARED[name]["group"] == group:
                    lines += [f"    {name}({', '.join(arguments(name))})  [{axes(name)}]", f"        {summary(fn)}"]
        sections.append("\n".join(lines))
    if what in (None, "components"):
        lines = ["components:"]
        for name, c in address.COMPONENTS.items():
            narrow = ", ".join(["layers"] * c["per_layer"] + address.selectable(name))
            lines.append(f"  {name:24} [{', '.join(c['dims'])}]  narrowed by {narrow}  — {c['meaning']}")
        sections.append("\n".join(lines))
    if what in (None, "featurizers"):
        lines = ["featurizer kinds:"]
        for name, fn in featurizers.KINDS.items():
            params = [f"{k}: {spell_type(p.annotation)}" + ("" if p.default is p.empty else f" = {p.default!r}")
                      for k, p in featurizers.registry.spelled(fn, featurizers.SUPPLIED).items()]
            lines += [f"  {name}({', '.join(params)})", f"      {summary(fn)}",
                      f"      modes: {'; '.join(m.replace('`', '') for m in modes(name))}"]
        sections.append("\n".join(lines))
    if what in (None, "pos"):
        sections.append("pos:\n" + doc(plan.Pos))
    return "\n\n".join(sections)


def markdown() -> str:
    """FORMAT.md: the one-page reference, generated from the code (`clab describe --markdown`)."""

    def block(model) -> str:
        return "\n".join(f"- `{f}`" for f in fields(model)) + "\n\n" + doc(model) + "\n"

    def table(rows) -> str:
        return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "documents", "patching.json")) as f:
        example = f.read()
    parts = [
        "# The clab format\n",
        "Generated by `clab describe --markdown` from the code; a test fails when this file and the code "
        "disagree. A document is one JSON object that declares a complete experiment; `clab run DOC` runs it. "
        "Its JSON Schema is `schema.json` (`clab schema`).\n",
        "## Root keys\n",
        table([["key", "holds"], ["---", "---"]] + [[f"`{k}`", plan.Plan.model_fields[k].description or ""]
                                                    for k in plan.documented(plan.Plan)]),
    ]
    for name, model in RECORDS.items():
        parts += [f"### {name}\n", block(model)]
    parts += ["## Step kinds\n", "Each entry of `steps` is `{\"kind\": <kind>, ...fields}`.\n"]
    for name, model in STEPS.items():
        parts += [f"### `{name}`\n", block(model)]
    parts += ["### `fit` blocks\n"] + [part for name, model in FIT.items() for part in (f"`{name}`:\n", block(model))]
    parts += ["## Derived fields\n",
              "Fields a document leaves out, which compile (or, for `run`, the engine) fills:\n",
              "\n".join(f"- `{model.__name__}`: {', '.join(derived(model))}"
                        for model in (plan.Plan, plan.Data, plan.Access, plan.Forward, plan.Generate, plan.OpStep,
                                      plan.Featurizer)
                        if derived(model)) + "\n"]
    parts += ["## Ops\n", ops.ABOUT]
    for group in ops.GROUPS:
        parts += [f"### {group.capitalize()}\n",
                  table([["op", "arguments", "axes", "what"], ["---", "---", "---", "---"]]
                        + [[f"`{n}`", ", ".join(f"`{a}`" for a in arguments(n)) or "—", axes(n), summary(fn)]
                           for n, fn in ops.OPS.items() if ops.DECLARED[n]["group"] == group])]
    parts += ["## Components\n", doc(address) + "\n",
              table([["component", "dims", "narrowed by", "meaning"], ["---", "---", "---", "---"]]
                    + [[f"`{n}`", ", ".join(c["dims"]), ", ".join(["layers"] * c["per_layer"] + address.selectable(n)),
                        c["meaning"]] for n, c in address.COMPONENTS.items()])]
    parts += ["## Featurizer kinds\n", doc(featurizers) + "\n"]
    for name, fn in featurizers.KINDS.items():
        params = [f"- `{k}: {spell_type(p.annotation)}" + ("`" if p.default is p.empty else f" = {p.default!r}`")
                  for k, p in featurizers.registry.spelled(fn, featurizers.SUPPLIED).items()]
        parts += [f"### `{name}`\n", "\n".join(params) + "\n\n" + doc(fn) + "\n\n" + "Modes: "
                  + "; ".join(modes(name)) + ".\n"]
    parts += ["## Names and scoping\n", doc(compiler.step) + "\n",
              "## Sweeps\n", doc(plan.find_sweeps) + "\n",
              "## Saves and the output\n", doc(output) + "\n", doc(output.table) + "\n",
              "## A worked document\n", "`documents/patching.json`:\n", "```json\n" + example + "```\n"]
    return "\n".join(parts)


def show(filled: "plan.Plan") -> str:
    """A filled plan as text: how it ran, then each result with the step that made it, its dims, its sweep
    axes and its shape (per coordinate when swept)."""
    run = filled.run
    assert run is not None
    lines = [f"{plan.name(filled)}: run on {run.engine}, remote {run.remote}, batch size {run.batch_size}, "
             f"started {run.started}, {run.seconds} s",
             "  versions: " + ", ".join(f"{k} {v}" for k, v in run.versions.items())]
    rows = [["result", "step", "dims", "axes", "shape"]]
    for name in filled.saves:
        entry = filled.steps[name.partition(".")[0]]
        value = filled.results[name]
        if isinstance(entry, list):
            value = {plan.labelled(filled, p.at): value[plan.coordinate(filled, p.at)] for p in entry}
        rows.append([name, name.partition(".")[0], ", ".join(plan.dims_of(filled, name)),
                     ", ".join(plan.points(entry)[0].at) or "—", cell(output.describe(value, "shape"))])
    return "\n".join(lines + ["  " + line for line in aligned(rows)])


def cell(value: Any) -> str:
    """A table cell as text: a string as it is, anything else (`at` included) as compact JSON."""
    if isinstance(value, str):
        return value
    return "" if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def aligned(rows: list[list[str]], width: int | None = None) -> list[str]:
    """Rows of cells padded into columns, newlines inside a cell written `\\n`; a cell longer than `width` is
    cut short with an ellipsis."""
    rows = [[c.replace("\n", "\\n") for c in row] for row in rows]
    rows = [[c if width is None or len(c) <= width else c[:width - 1] + "…" for c in row] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]


def tabulated(rows: list[dict], as_csv: bool) -> str:
    """`plan.table` rows as text: aligned columns, or CSV with every cell whole."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    cells = [columns] + [[cell(row.get(c)) for c in columns] for row in rows]
    if not as_csv:
        return "\n".join(aligned(cells, 48))
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(cells)
    return buffer.getvalue().rstrip("\n")


def main(argv: list[str] | None = None) -> int:
    from .engines import ENGINES

    parser = argparse.ArgumentParser(prog="clab")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a document end to end")
    run.add_argument("document")
    run.add_argument("--engine", default="nnterp", choices=list(ENGINES))
    run.add_argument("--out", help="output directory (default out/<document name>)")
    run.add_argument("--remote", default="no", choices=("no", "local", "yes"))
    run.add_argument("--batch-size", type=int, default=8)
    look = commands.add_parser("show", help="print what a run's output directory holds")
    look.add_argument("out")
    rows = commands.add_parser("table", help="print saved values of a run's output directory as one table")
    rows.add_argument("out")
    rows.add_argument("names", nargs="+")
    rows.add_argument("--csv", action="store_true", help="CSV on stdout, every cell whole")
    tell = commands.add_parser("describe", help="print the format's tables, read off the code")
    tell.add_argument("what", nargs="?", choices=WHAT)
    tell.add_argument("--markdown", action="store_true", help="write FORMAT.md")
    commands.add_parser("schema", help="print the plan's JSON Schema")
    args = parser.parse_args(argv)

    if args.command == "describe":
        if args.markdown:
            with open("FORMAT.md", "w") as f:
                f.write(markdown())
            print("wrote FORMAT.md")
        else:
            print(describe(args.what))
        return 0
    if args.command == "schema":
        print(json.dumps(schema(), indent=1, ensure_ascii=False))
        return 0
    if args.command in ("show", "table"):
        try:
            filled = plan.Plan.load(args.out)
            print(show(filled) if args.command == "show" else tabulated(filled.table(*args.names), args.csv))
        except (ValueError, OSError) as e:
            print(f"clab: {e}", file=sys.stderr)
            return 1
        return 0

    import clab

    remote = {"no": False, "local": "local", "yes": True}[args.remote]
    name = os.path.splitext(os.path.basename(args.document))[0]
    out = args.out or os.path.join("out", name)
    try:
        clab.run(args.document, engine=args.engine, out=out, batch_size=args.batch_size, remote=remote)
    except (ValueError, NotImplementedError) as e:
        print(f"clab: {e}", file=sys.stderr)
        return 1
    print(f"{name} on {args.engine} (batch size {args.batch_size}, remote {args.remote}) -> {out}")
    with open(os.path.join(out, "manifest.json")) as f:
        described = json.load(f)["results"]
    for key, d in described.items():
        print(f"  {d['file']:28} {d['step']:16} {json.dumps(d['shape'])}")
    print(f"  {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
