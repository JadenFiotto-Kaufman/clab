"""`clab describe`: the format's tables by introspection (the records and their fields, the ops and their
signatures, the components, the featurizer kinds, the position grammar, the engines), and `describe --markdown`,
which renders all of it as FORMAT.md."""

import inspect
import json
import os
from typing import Any, get_args

from pydantic import BaseModel

from clab import compile as compiler
from clab import featurizers, ops, output, plan
from clab.format import address, registry, sweeps
from clab.output import table

WHAT = ("records", "steps", "ops", "components", "featurizers", "pos", "engines")

# The repo root: the directory the package sits in, where FORMAT.md and the corpus are.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RECORDS = {"`model`": plan.Model, "`data.<name>`": plan.Data, "`sites.<name>`, or a `site` inline": plan.Site,
           "A column read as text": plan.Column, "A literal in a list `input`": plan.TextPart,
           "An entry of `interventions`": plan.Intervention, "An op in an entry": plan.Op, "`pos`": plan.Pos}


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


def spell(annotation: Any) -> str:
    """A type as the tables spell it, a literal's values as JSON."""
    return registry.spell(annotation, NAMES, as_json)


def signature(params: dict[str, inspect.Parameter]) -> list[str]:
    """Parameters as a document writes them: `key: type`, and `= default` where there is one."""
    return [f"{k}: {spell(p.annotation)}" + ("" if p.default is p.empty else f" = {as_json(p.default)}")
            for k, p in params.items()]


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


def doc(thing: object) -> str:
    return inspect.getdoc(thing) or ""


def summary(thing: object) -> str:
    return " ".join(doc(thing).split())


def axes(name: str) -> str:
    a = ops.DECLARED[name]
    if a["layout"] is not None:
        return summary(a["layout"])

    def listed(verb: str) -> str:
        if verb == "removes" and a[verb] == ops.VOCABULARY:
            return "its vocabulary axis"
        return ", ".join(" or ".join(x) if isinstance(x, tuple) else x for x in a[verb])
    parts = [f"{verb} {listed(verb)}" for verb in ("needs", "removes", "adds") if a[verb]]
    return "; ".join(parts) or "keeps the layout"


def served(kind: str) -> list[str]:
    """Each op a featurizer kind serves, with what it does and the arguments of the kind's own it takes."""
    out = []
    cls = featurizers.KINDS[kind]
    for name in cls.ops():
        fn = cls.method(name)
        own = signature(registry.spelled(fn, ("self", "h", "z", "new")))
        produces = featurizers.OPS[name]["produces"].format(axis=cls.axis)
        out.append(f"`{name}`" + (f"({', '.join(own)})" if own else "") + f" {produces}"
                   + (f": {summary(fn)}" if doc(fn) else ""))
    return out


def run_options(name: str) -> list[tuple[str, str]]:
    """Engine `name`'s options, each `key: type = default` beside what its constructor's docstring says it
    takes: `device`, then its own."""
    from clab.engines import ENGINES, options

    cls: Any = ENGINES[name]
    params = {"device": inspect.signature(cls.__init__).parameters["device"], **options(name)}
    said = registry.described(cls.__init__)
    return [(text, said.get(k, "")) for k, text in zip(params, signature(params))]


def describe(what: str | None = None) -> str:
    """The format's tables, read off the code: one of `records`, `steps`, `ops`, `components`, `featurizers`,
    `pos`, `engines`, or all of them; any other name is refused, naming those."""
    if what is not None and what not in WHAT:
        raise ValueError(f"describe {what!r}: nothing is described under that name; the registries are "
                         + ", ".join(WHAT))
    sections = []
    if what in (None, "records"):
        sections.append("\n".join(["records:"] + [f"  {name.replace('`', '')}({', '.join(fields(model))})"
                                                  for name, model in RECORDS.items()]))
    if what in (None, "steps"):
        lines = ["step kinds:"]
        for name, model in {**plan.KINDS, **FIT}.items():
            lines += [f"  {name}({', '.join(fields(model))})", f"      {summary(model)}"]
        sections.append("\n".join(lines))
    if what in (None, "ops"):
        lines = ["ops (`of` is the value acted on; inside an entry, the value it meets there):"]
        for group in ops.GROUPS:
            lines.append(f"  {group}:")
            for name, fn in ops.OPS.items():
                if ops.DECLARED[name]["group"] == group:
                    lines += [f"    {name}({', '.join(signature(ops.parameters(name)))})  [{axes(name)}]",
                              f"        {summary(fn)}"]
        sections.append("\n".join(lines))
    if what in (None, "components"):
        lines = ["components:"]
        for name, c in address.COMPONENTS.items():
            narrow = ", ".join(["layers"] * c["per_layer"] + address.selectable(name))
            lines.append(f"  {name:24} [{', '.join(address.dims(name))}]  narrowed by {narrow}"
                         + ("  stream" if c["stream"] else "") + f"  — {c['meaning']}")
        sections.append("\n".join(lines))
    if what in (None, "featurizers"):
        lines = ["featurizer kinds:"]
        for name, cls in featurizers.KINDS.items():
            lines += [f"  {name}({', '.join(signature(registry.spelled(cls, featurizers.SUPPLIED)))})",
                      f"      {summary(cls)}",
                      f"      ops: {'; '.join(m.replace('`', '') for m in served(name))}"]
        sections.append("\n".join(lines))
    if what in (None, "pos"):
        sections.append("pos:\n" + doc(plan.Pos))
    if what in (None, "engines"):
        from clab import engines
        lines = ["engines (`clab run --engine NAME --device D --batch-size N -o KEY=VALUE`):"]
        for name, cls in engines.ENGINES.items():
            lines.append(f"  {name}: " + " ".join(doc(inspect.getmodule(cls)).split("\n\n")[0].split()))
            lines += [f"      {o}" + (f" — {text}" if text else "") for o, text in run_options(name)]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def markdown() -> str:
    """FORMAT.md: the one-page reference, generated from the code (`clab describe --markdown`)."""

    def block(model: type[BaseModel]) -> str:
        return "\n".join(f"- `{f}`" for f in fields(model)) + "\n\n" + doc(model) + "\n"

    def grid(rows: list[list[str]]) -> str:
        return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"

    with open(os.path.join(ROOT, "documents", "patching.json")) as f:
        example = f.read()
    parts = [
        "# The clab format\n",
        "Generated by `clab describe --markdown` from the code; a test fails when this file and the code "
        "disagree. A document is one JSON object that declares a complete experiment; `clab run DOC` runs it.\n",
        "## Root keys\n",
        grid([["key", "holds"], ["---", "---"]] + [[f"`{k}`", plan.Plan.model_fields[k].description or ""]
                                                    for k in plan.documented(plan.Plan)]),
    ]
    for name, model in RECORDS.items():
        parts += [f"### {name}\n", block(model)]
    parts += ["## Step kinds\n", "Each entry of `steps` is `{\"kind\": <kind>, ...fields}`.\n"]
    for name, model in plan.KINDS.items():
        parts += [f"### `{name}`\n", block(model)]
    parts += ["### `fit` blocks\n"] + [part for name, model in FIT.items() for part in (f"`{name}`:\n", block(model))]
    labels = derived(plan.Labelled)

    def own(model: type[BaseModel]) -> list[str]:
        """A model's derived fields beside a value's labels, which it has as a `Labelled`."""
        return [f for f in derived(model) if not (issubclass(model, plan.Labelled) and f in labels)]
    parts += ["## Derived fields\n",
              "Fields a document leaves out, which compile (or, for `run`, the engine) fills. Compile writes what the "
              "document decides; preflight, with the model in hand, completes the same records in place (a `layers` or "
              "`kept` a slice or a negative selector leaves to the model, every read's and write's `positions`), so the "
              "manifest holds the records as the run completed them.\n",
              f"- A value's record (`Intervention`, `OpStep`, and a `Generate`'s own value): {', '.join(labels)}\n"
              + "\n".join(f"- `{model.__name__}`: {', '.join(own(model))}"
                          for model in (plan.Plan, plan.Data, plan.Intervention, plan.Op, plan.Forward, plan.Generate,
                                        plan.OpStep, plan.Featurizer) if own(model)) + "\n"]
    parts += ["## Ops\n", ops.ABOUT]
    for group in ops.GROUPS:
        parts += [f"### {group.capitalize()}\n",
                  grid([["op", "arguments", "axes", "what"], ["---", "---", "---", "---"]]
                        + [[f"`{n}`", ", ".join(f"`{a}`" for a in signature(ops.parameters(n))) or "—", axes(n),
                            summary(fn)]
                           for n, fn in ops.OPS.items() if ops.DECLARED[n]["group"] == group])]
    parts += ["## Components\n", doc(address) + "\n",
              grid([["component", "dims", "narrowed by", "stream", "meaning"], ["---", "---", "---", "---", "---"]]
                    + [[f"`{n}`", ", ".join(address.dims(n)),
                        ", ".join(["layers"] * c["per_layer"] + address.selectable(n)),
                        "yes" if c["stream"] else "no", c["meaning"]] for n, c in address.COMPONENTS.items()])]
    parts += ["## Featurizer kinds\n", doc(featurizers) + "\n"]
    for name, cls in featurizers.KINDS.items():
        params = [f"- `{p}`" for p in signature(registry.spelled(cls, featurizers.SUPPLIED))]
        listed = "; ".join(served(name))
        parts += [f"### `{name}`\n", "\n".join(params) + "\n\n" + doc(cls) + "\n\n" + "Ops: " + listed
                  + ("" if listed.endswith(".") else ".") + "\n"]
    from clab import engines
    parts += ["## Engines and run options\n", "\n\n".join(doc(engines).split("\n\n")[:2]) + "\n",
              "Every engine runs with `batch_size`, how many of a step's rows go through the model in one forward "
              "or generation (default 8); the options below are each engine's own.\n"]
    for name, cls in engines.ENGINES.items():
        listed = "".join(f"- `{o}`" + (f": {text}" if text else "") + "\n" for o, text in run_options(name))
        parts += [f"### `{name}`\n", listed + "\n" + doc(inspect.getmodule(cls)) + "\n"]
    parts += ["## Names and scoping\n", doc(compiler.step) + "\n",
              "## Sweeps\n", doc(sweeps.find_sweeps) + "\n",
              "## Saves and the output\n", doc(output) + "\n", doc(table.table) + "\n",
              "## A worked document\n", "`documents/patching.json`:\n", "```json\n" + example + "```\n"]
    return "\n".join(parts)


def write_markdown() -> str:
    """`clab describe --markdown`: `markdown()` written as FORMAT.md beside the package, at the repo root, from
    whatever directory the command runs in; returns the path written."""
    path = os.path.join(ROOT, "FORMAT.md")
    with open(path, "w") as f:
        f.write(markdown())
    return path
