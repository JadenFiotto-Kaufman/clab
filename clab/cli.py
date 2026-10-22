"""`clab run`, `clab show`, `clab table` and `clab describe`: a thin command line over
`clab.run` and a run's output directory, and the format read off the code.

`show` prints what a run's directory holds (each result with its step, dims, axes and shape, and how the
run ran); `table` prints `plan.table(...)` over its saved values, aligned or as CSV. Both are `Plan.load`
and one call.

`describe` prints the format's tables by introspection: the records and their fields (from the plan's
models), the ops and their signatures, the components, the featurizer kinds and the position grammar.
`describe --markdown` renders all of it as `FORMAT.md`.
"""

import argparse
import csv
import inspect
import io
import json
import os
import sys
from typing import Any, get_args

from pydantic import BaseModel

from . import address, output
from .plan import compile as compiler
from .plan import model as plan
from . import registry
from .runtime import featurizers, ops

WHAT = ("records", "steps", "ops", "components", "featurizers", "pos", "engines")
RECORDS = {"`model`": plan.Model, "`data.<name>`": plan.Data, "`sites.<name>`, or a `site` inline": plan.Site,
           "A column read as text": plan.Column, "A literal in a list `input`": plan.TextPart,
           "An entry of `interventions`": plan.Intervention, "An op in an entry": plan.Op, "`pos`": plan.Pos}
STEPS = {name: model for name, model in plan.KINDS.items()}
FIT = {"optimizer": plan.Optimizer, "early_stop": plan.EarlyStop}


NAMES = {plan.Sweep: "a sweep", plan.Site: "site", plan.Pos: "pos", plan.Scope: "scope", plan.Op: "op",
         plan.Intervention: "entry", plan.Optimizer: "optimizer", plan.EarlyStop: "early_stop",
         plan.Entry: "step", plan.Column: "column", plan.TextPart: "literal",
         get_args(plan.Interventions)[0]: "a bundle's name, an object of entry, or a list of them"}


def as_json(value: Any) -> str:
    """A value as a document writes it: JSON (`true`, `null`, `"local"`), or as Python spells it where JSON
    cannot."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_defaults=True)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def spell_type(annotation: Any, names: dict | None = None) -> str:
    """A type as the tables spell it, a literal's values as JSON."""
    return registry.spell(annotation, names, as_json)


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
        out.append(text if field.is_required() else f"{text} = {as_json(field.default)}")
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
    if a["layout"] is not None:
        return summary(a["layout"])

    def listed(verb: str) -> str:
        if not isinstance(a[verb], tuple):
            return f"the axes `{a[verb]}` names"
        if verb == "removes" and a[verb] == ops.VOCABULARY:
            return "its vocabulary axis"
        return ", ".join(" or ".join(x) if isinstance(x, tuple) else x for x in a[verb])
    parts = [f"{verb} {listed(verb)}" for verb in ("needs", "removes", "adds") if a[verb]]
    return "; ".join(parts) or "keeps the layout"


def arguments(name: str) -> list[str]:
    """An op's arguments as a document writes them; `...` where its featurizer kind's own follow (`apply`)."""
    out = []
    for key, p in ops.parameters(name).items():
        text = f"{key}: {spell_type(p.annotation)}"
        out.append(text if p.default is p.empty else f"{text} = {as_json(p.default)}")
    takes_more = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(ops.OPS[name]).parameters.values())
    return out + ["..."] * takes_more


def served(kind: str) -> list[str]:
    """Each op a featurizer kind serves, with what it does and the arguments of the kind's own it takes."""
    out = []
    cls = featurizers.KINDS[kind]
    for name in cls.ops():
        fn = cls.method(name)
        own = [f"{k}: {spell_type(p.annotation)}" + ("" if p.default is p.empty else f" = {as_json(p.default)}")
               for k, p in featurizers.registry.spelled(fn, ("self", "h", "z", "new")).items()]
        produces = featurizers.OPS[name]["produces"].format(axis=cls.axis)
        out.append(f"`{name}`" + (f"({', '.join(own)})" if own else "") + f" {produces}"
                   + (f": {summary(fn)}" if doc(fn) else ""))
    return out


def run_options(name: str) -> list[tuple[str, str]]:
    """Engine `name`'s options, each `key: type = default` beside what its constructor's docstring says it
    takes: `device`, then its own."""
    from .engines import ENGINES, SUPPLIED, options

    cls: Any = ENGINES[name]
    params = {"device": inspect.signature(cls.__init__).parameters["device"], **options(name)}
    said = registry.described(cls.__init__)
    return [(f"{k}: {spell_type(p.annotation)} = {as_json(p.default)}", said.get(k, "")) for k, p in params.items()]


def describe(what: str | None = None) -> str:
    """The format's tables, read off the code: one of `records`, `steps`, `ops`, `components`, `featurizers`,
    `pos`, `engines`, or all of them."""
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
        lines = ["ops (`of` is the value acted on; inside an entry, the value it meets there):"]
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
        for name, cls in featurizers.KINDS.items():
            params = [f"{k}: {spell_type(p.annotation)}" + ("" if p.default is p.empty else f" = {as_json(p.default)}")
                      for k, p in featurizers.registry.spelled(cls, featurizers.SUPPLIED).items()]
            lines += [f"  {name}({', '.join(params)})", f"      {summary(cls)}",
                      f"      ops: {'; '.join(m.replace('`', '') for m in served(name))}"]
        sections.append("\n".join(lines))
    if what in (None, "pos"):
        sections.append("pos:\n" + doc(plan.Pos))
    if what in (None, "engines"):
        from . import engines
        lines = ["engines (`clab run --engine NAME --device D --batch-size N -o KEY=VALUE`):"]
        for name, cls in engines.ENGINES.items():
            lines.append(f"  {name}: " + " ".join(doc(inspect.getmodule(cls)).split("\n\n")[0].split()))
            lines += [f"      {o}" + (f" — {text}" if text else "") for o, text in run_options(name)]
        sections.append("\n".join(lines))
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
        "disagree. A document is one JSON object that declares a complete experiment; `clab run DOC` runs it.\n",
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
                        for model in (plan.Plan, plan.Data, plan.Intervention, plan.Forward, plan.Generate, plan.OpStep,
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
    for name, cls in featurizers.KINDS.items():
        params = [f"- `{k}: {spell_type(p.annotation)}" + ("`" if p.default is p.empty else f" = {as_json(p.default)}`")
                  for k, p in featurizers.registry.spelled(cls, featurizers.SUPPLIED).items()]
        listed = "; ".join(served(name))
        parts += [f"### `{name}`\n", "\n".join(params) + "\n\n" + doc(cls) + "\n\n" + "Ops: " + listed
                  + ("" if listed.endswith(".") else ".") + "\n"]
    from . import engines
    parts += ["## Engines and run options\n", "\n\n".join(doc(engines).split("\n\n")[:2]) + "\n",
              "Every engine runs with `batch_size`, how many of a step's rows go through the model in one forward "
              "or generation (default 8); the options below are each engine's own.\n"]
    for name, cls in engines.ENGINES.items():
        listed = "".join(f"- `{o}`" + (f": {text}" if text else "") + "\n" for o, text in run_options(name))
        parts += [f"### `{name}`\n", listed + "\n" + doc(inspect.getmodule(cls)) + "\n"]
    parts += ["## Names and scoping\n", doc(compiler.step) + "\n",
              "## Sweeps\n", doc(plan.find_sweeps) + "\n",
              "## Saves and the output\n", doc(output) + "\n", doc(output.table) + "\n",
              "## A worked document\n", "`documents/patching.json`:\n", "```json\n" + example + "```\n"]
    return "\n".join(parts)


def how(run: "plan.Run") -> str:
    """Where a run ran and with which engine options: `device cpu`, or `device placed by the server, remote
    true`."""
    return f"device {run.device or 'placed by the server'}" + "".join(f", {k} {as_json(v)}"
                                                                      for k, v in run.options.items())


def show(filled: "plan.Plan") -> str:
    """A filled plan as text: how it ran (of what was installed, clab's and python's versions), then each result
    with the step that made it, its dims, its sweep axes and its shape (per coordinate when swept)."""
    run = filled.run
    assert run is not None
    lines = [f"{plan.name(filled)}: run on {run.engine}, {how(run)}, batch size {run.batch_size}, "
             f"started {run.started}, {run.seconds} s",
             f"  served: {', '.join(run.dtype) or '—'} on {', '.join(run.devices) or '—'}",
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


def option(text: str) -> tuple[str, Any]:
    """A `-o key=value`: the value as JSON (`true`, `2`, `{"a": 1}`), else as the text it is (`auto`)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise ValueError(f"-o {text!r}: an option is KEY=VALUE")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main(argv: list[str] | None = None) -> int:
    from .engines import ENGINES

    parser = argparse.ArgumentParser(prog="clab")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a document end to end")
    run.add_argument("document")
    run.add_argument("--engine", default="nnterp", choices=list(ENGINES))
    run.add_argument("--out", help="output directory (default out/<document name>)")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--device", help="where the engine places the model (default cpu)")
    run.add_argument("-o", dest="options", action="append", default=[], metavar="KEY=VALUE",
                     help="an option of the engine's (`clab describe engines`), the value read as JSON, else as text")
    look = commands.add_parser("show", help="print what a run's output directory holds")
    look.add_argument("out")
    rows = commands.add_parser("table", help="print saved values of a run's output directory as one table")
    rows.add_argument("out")
    rows.add_argument("names", nargs="+")
    rows.add_argument("--csv", action="store_true", help="CSV on stdout, every cell whole")
    tell = commands.add_parser("describe", help="print the format's tables, read off the code")
    tell.add_argument("what", nargs="?", choices=WHAT)
    tell.add_argument("--markdown", action="store_true", help="write FORMAT.md")
    args = parser.parse_args(argv)

    if args.command == "describe":
        if args.markdown:
            with open("FORMAT.md", "w") as f:
                f.write(markdown())
            print("wrote FORMAT.md")
        else:
            print(describe(args.what))
        return 0
    if args.command in ("show", "table"):
        try:
            filled = plan.Plan.load(args.out)
            print(show(filled) if args.command == "show" else tabulated(filled.table(*args.names), args.csv))
        except (ValueError, OSError) as e:
            print(f"clab: {e}", file=sys.stderr)
            return 1
        return 0

    from nnsight.intervention.backends.remote import RemoteError

    import clab

    name = os.path.splitext(os.path.basename(args.document))[0]
    out = args.out or os.path.join("out", name)
    try:
        options = dict(option(text) for text in args.options)
        for key in {"device", "batch_size"} & set(options):
            raise ValueError(f"-o {key}: it is given as --{key.replace('_', '-')}")
        filled = clab.run(args.document, engine=args.engine, out=out, batch_size=args.batch_size,
                          device=args.device, **options)
    except (ValueError, NotImplementedError) as e:
        print(f"clab: {e}", file=sys.stderr)
        return 1
    except RemoteError as e:          # a refusal on the server comes home as nnsight's error, its last line the reason
        print(f"clab: {(str(e).strip().splitlines() or [''])[-1]}", file=sys.stderr)
        return 1
    assert filled.run is not None
    print(f"{name} on {args.engine} (batch size {args.batch_size}, {how(filled.run)}) -> {out}")
    with open(os.path.join(out, "manifest.json")) as f:
        described = json.load(f)["results"]
    for key, d in described.items():
        print(f"  {d['file']:28} {d['step']:16} {json.dumps(d['shape'])}")
    print(f"  {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
