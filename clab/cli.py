"""`clab run`, `clab show`, `clab table` and `clab describe`: a thin command line over
`clab.run` and a run's output directory, and the format read off the code.

`show` prints what a run's directory holds (how the run ran and what each step cost, then each result with
its step, dims, axes and shape); `table` prints `plan.table(...)` over its saved values, aligned or as CSV.
Both are `Plan.load` and one call.

`describe` prints the format's tables by introspection: the records and their fields (from the plan's
models), the ops and their signatures, the components, the featurizer kinds and the position grammar.
`describe --markdown` renders all of it as `FORMAT.md`.
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import Any

from clab import output
from clab.docs import WHAT, as_json, describe, markdown
from clab.format.sweeps import coordinate, labelled
from clab.plan import Plan, Run, dims_of, name, points


def how(run: Run) -> str:
    """Where a run ran and with which engine options: `device cpu`, or `device placed by the server, remote
    true`."""
    return f"device {run.device or 'placed by the server'}" + "".join(f", {k} {as_json(v)}"
                                                                      for k, v in run.options.items())


def show(filled: Plan) -> str:
    """A filled plan as text: how it ran (of what was installed, clab's and python's versions), what each step
    cost (`costs`), then each result with the step that made it, its dims, its sweep axes and its shape (per
    coordinate when swept)."""
    run = filled.run
    assert run is not None
    lines = [f"{name(filled)}: run on {run.engine}, {how(run)}, batch size {run.batch_size}, "
             f"started {run.started}, {run.seconds} s",
             f"  served: {', '.join(run.dtype) or '—'} on {', '.join(run.devices) or '—'}",
             "  versions: " + ", ".join(f"{k} {v}" for k, v in run.versions.items())]
    lines += ["  " + line for line in aligned(costs(filled))] if run.costs else []        # a run before costs
    rows = [["result", "step", "dims", "axes", "shape"]]
    for saved in filled.saves:
        entry = filled.steps[saved.partition(".")[0]]
        value = filled.results[saved]
        if isinstance(entry, list):
            value = {labelled(filled, p.at): value[coordinate(filled, p.at)] for p in entry}
        rows.append([saved, saved.partition(".")[0], ", ".join(dims_of(filled, saved)),
                     ", ".join(points(entry)[0].at) or "—", cell(output.describe(value, "shape"))])
    return "\n".join(lines + ["  " + line for line in aligned(rows)])


def costs(filled: Plan) -> list[list[str]]:
    """What each step cost (`plan.run.costs`) as table rows, one per step and point: its seconds, its peak memory
    on each device in MiB (— where none was measured), its model calls and the tokens they fed."""
    run = filled.run
    assert run is not None
    rows = [["step", "at", "seconds", "memory", "forwards", "tokens"]]
    for step, entry in filled.steps.items():
        for p in points(entry) if step in run.costs else []:
            cost = run.costs[step][labelled(filled, p.at)] if isinstance(entry, list) else run.costs[step]
            memory = ", ".join(f"{d} {b / 2 ** 20:.1f} MiB" for d, b in (cost["memory"] or {}).items()) or "—"
            rows.append([step, labelled(filled, p.at) or "—", f"{cost['seconds']:.3f}", memory,
                         str(cost["forwards"]), str(cost["tokens"])])
    return rows


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
    from clab.engines import ENGINES

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
            filled = Plan.load(args.out)
            print(show(filled) if args.command == "show" else tabulated(filled.table(*args.names), args.csv))
        except (ValueError, OSError) as e:
            print(f"clab: {e}", file=sys.stderr)
            return 1
        return 0

    from nnsight.intervention.backends.remote import RemoteError

    import clab

    stem = os.path.splitext(os.path.basename(args.document))[0]
    out = args.out or os.path.join("out", stem)
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
    print(f"{stem} on {args.engine} (batch size {args.batch_size}, {how(filled.run)}) -> {out}")
    with open(os.path.join(out, "manifest.json")) as f:
        described = json.load(f)["results"]
    for d in described.values():
        print(f"  {d['file']:28} {d['step']:16} {json.dumps(d['shape'])}")
    print(f"  {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
