"""`clab run`, `clab describe` and `clab schema`: a thin command line over `clab.run`, and the format read off
the code.

`describe` prints the format's tables by introspection: the records and their fields (from the plan's
models), the ops and their signatures, the components, the featurizer kinds and the position grammar.
`describe --markdown` renders all of it as `FORMAT.md`; `schema` prints the plan's JSON Schema.
"""

import argparse
import inspect
import json
import os
import sys
from typing import Any

from pydantic import BaseModel, TypeAdapter

from . import address, compiler, featurizers, ops, output, plan
from .registry import spell as spell_type

WHAT = ("records", "steps", "ops", "components", "featurizers", "pos")
RECORDS = {"`model`": plan.Model, "`data.<name>`": plan.Data, "`sites.<name>`, or a `site` inline": plan.Site,
           "A write or a read": plan.Access, "An op in a write or a read": plan.Op, "`pos`": plan.Pos,
           "`interventions.<name>`": plan.Interventions}
STEPS = {name: model for name, model in plan.KINDS.items()}
FIT = {"optimizer": plan.Optimizer, "early_stop": plan.EarlyStop, "eval": plan.Eval}


NAMES = {plan.Sweep: "a sweep", plan.Site: "site", plan.Pos: "pos", plan.Scope: "scope", plan.Op: "op",
         plan.Access: "write or read", plan.Optimizer: "optimizer", plan.EarlyStop: "early_stop", plan.Eval: "eval",
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
    a = ops.DECLARED[name]
    parts = [f"{verb} {', '.join(a[verb]) if isinstance(a[verb], tuple) else 'the axes `' + a[verb] + '` names'}"
             for verb in ("needs", "removes", "adds") if a[verb]]
    return "; ".join(parts) or "keeps the layout"


def arguments(name: str) -> list[str]:
    out = []
    for key, p in ops.parameters(name).items():
        text = f"{key}: {spell_type(p.annotation)}"
        out.append(text if p.default is p.empty else f"{text} = {p.default!r}")
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
        args = {k: {"anyOf": [kinds.get(p.annotation) or TypeAdapter(p.annotation).json_schema(), sweep]}
                for k, p in params.items()}
        branches.append({"type": "object", "properties": {**base["properties"], "op": {"const": name}, **args},
                         "required": base.get("required", []) + [k for k, p in params.items() if p.default is p.empty],
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
            lines += [f"  {name}({', '.join(params)})", f"      {summary(fn)}"]
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
                        for model in (plan.Plan, plan.Data, plan.Access, plan.Forward, plan.OpStep, plan.Featurizer)
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
        parts += [f"### `{name}`\n", "\n".join(params) + "\n\n" + doc(fn) + "\n"]
    parts += ["## Names and scoping\n", doc(compiler.step) + "\n",
              "## Sweeps\n", doc(plan.find_sweeps) + "\n",
              "## Saves and the output\n", doc(output) + "\n", doc(output.table) + "\n",
              "## A worked document\n", "`documents/patching.json`:\n", "```json\n" + example + "```\n"]
    return "\n".join(parts)


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
