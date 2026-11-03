"""`clab run`, `clab show`, `clab table` and `clab describe`: a thin command line over `clab.run`, a run's output
directory and the format read off the code (`clab.docs`).

`show` and `table` are each `Plan.load` and one call: what the run's directory holds as `show` prints it, or
`plan.table(...)` over its saved values, aligned or as CSV. `describe` prints the format's tables, and `describe
--markdown` writes them as `FORMAT.md` beside the package, at the repo root.

A refusal is one `clab: <reason>` line on stderr and exit status 1. A reader that goes away (`clab show OUT |
head -1`) ends the command quietly.
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import Any

from clab import docs, output
from clab.compile import positions
from clab.docs import WHAT, as_json, describe
from clab.format.data import PADDINGS
from clab.format.sweeps import coordinate, coordinate_text
from clab.plan import Plan, Run, entries, made_by, name, points, projections, runnable, saved_value_axes, serving


def how(run: Run) -> str:
    """Where a run ran and with which engine options: `device cpu`, or `device placed by the server, remote
    true`."""
    return f"device {run.device or 'placed by the server'}" + "".join(f", {k} {as_json(v)}"
                                                                      for k, v in run.options.items())


def show(filled: Plan) -> str:
    """A filled plan as text: how it ran (of what was installed, clab's and python's versions; where it stopped,
    if it did), what each step cost (`costs`), where each entry resolved (`resolved`), then each result with the
    step that made it, its dims (per coordinate when swept and its points differ), its sweep axes and its shape
    (`shaped`: counted over its points when swept). A compiled plan that has not run: its name, and which points
    each run of a step will serve (`served`)."""
    run = filled.run
    if run is None:
        rows = [[step, coordinate_text(filled, p.at) or "—", served(filled, entry, points_)]
                for step, entry in runnable(filled.steps).items() for p, points_ in serving(filled, step)]
        return "\n".join([f"{name(filled)}: compiled, not run"]
                         + ["  " + line for line in aligned([["step", "at", "serves"]] + rows)])
    lines = [f"{name(filled)}: run on {run.engine}, {how(run)}, batch size {run.batch_size}, padding {run.padding}, "
             f"started {run.started}, {run.seconds} s",
             f"  served: {', '.join(run.dtype) or '—'} on {', '.join(run.devices) or '—'}",
             "  versions: " + ", ".join(f"{k} {v}" for k, v in run.versions.items())]
    lines += [f"  stopped: {run.stopped}"] if run.stopped else []
    lines += ["  " + line for line in aligned(costs(filled))] if run.costs else []        # a run before costs
    lines += ["  " + line for line in aligned(resolved(filled))]
    rows = [["result", "step", "dims", "axes", "shape"]]
    for saved in (s for s in filled.saves if s in filled.results):          # every one, unless the run stopped
        value, axes = filled.results[saved], saved_value_axes(filled, saved)
        if axes:
            value = {coordinate_text(filled, at): value[coordinate(filled, at)] for at, _ in projections(filled, saved)}
        dims = output.saved_value_layout(filled, saved)["dims"]
        rows.append([saved, made_by(filled, saved), ", ".join(dims) if isinstance(dims, list) else cell(dims),
                     ", ".join(axes) or "—", shaped(output.shape_or_dtype(value, "shape"), bool(axes))])
    return "\n".join(lines + ["  " + line for line in aligned(rows)])


def costs(filled: Plan) -> list[list[str]]:
    """What each step cost (`plan.run.costs`) as table rows, one per run of a step (`runnable`: a fit's eval steps
    after it; `serving`: a forward's or generation's per forward, an op's per distinct value, a fit's per point),
    at the point that ran, with the points it served beside it (`served`): its seconds, its peak memory on each
    device in MiB (— where none was measured), its model calls, the tokens they fed, padding included, and the
    rows' own of them (`real`)."""
    run = filled.run
    assert run is not None
    rows = [["step", "at", "serves", "seconds", "memory", "forwards", "tokens", "real"]]
    for step, entry in runnable(filled.steps).items():
        for p, points_ in serving(filled, step) if step in run.costs else []:
            cost = run.costs[step].get(coordinate_text(filled, p.at)) if isinstance(entry, list) else run.costs[step]
            serves = served(filled, entry, points_)
            if cost is None:                    # a point the run stopped before
                rows.append([step, coordinate_text(filled, p.at) or "—", serves, "—", "—", "—", "—", "—"])
                continue
            memory = ", ".join(f"{d} {b / 2 ** 20:.1f} MiB" for d, b in (cost["memory"] or {}).items()) or "—"
            rows.append([step, coordinate_text(filled, p.at) or "—", serves, f"{cost['seconds']:.3f}", memory,
                         str(cost["forwards"]), str(cost["tokens"]), str(cost.get("real", "—"))])
    return rows


def served(filled: Plan, entry: Any, points_: list) -> str:
    """The points one run of a step serves, as a cell: `—` for itself alone, `all N points`, else each one."""
    if len(points_) == 1:
        return "—"
    if len(points_) == len(points(entry)):
        return f"all {len(points_)} points"
    return ", ".join(coordinate_text(filled, p.at) for p in points_)


def resolved(filled: Plan) -> list[list[str]]:
    """Where each read and write resolved, as table rows: one per entry and distinct `positions` per row
    (printed by `compact`) under its step's dataset, with the point it is at, or how many of the step's points
    share it (a grid's positions vary along one axis, or none); a fit's eval steps' entries as
    `<fit>.<eval>.<step>.<entry>`."""
    rows = [["entry", "at", "positions"]]
    for step, entry in runnable(filled.steps).items():
        listed = points(entry)
        # Each entry's distinct positions per row, by entry and their JSON, with the points that share them.
        found: dict[tuple[str, str], tuple[list | None, list[str]]] = {}
        for p in listed:
            for key, per_row in placed(step, p):
                found.setdefault((key, json.dumps(per_row)), (per_row, []))[1].append(coordinate_text(filled, p.at))
        for (key, _), (per_row, ats) in found.items():
            shared = ats[0] or "—" if len(ats) == 1 else f"all {len(ats)} points" if len(ats) == len(listed) else \
                f"{len(ats)} of {len(listed)} points"
            rows.append([key, shared, compact(per_row)])
    return rows


def placed(step: str, p: Any) -> list[tuple[str, list | None]]:
    """Each entry of step `step`'s point `p` by name with its positions per row under the step's dataset (a load's,
    the read it carries); a fit's none, its eval steps' their own."""
    host = p.source if p.kind == "load" else p
    return [(f"{step}.{k}", e.positions.get(p.dataset)) for k, e in entries(host).items()]


# The most rows, and the most positions in a row, that `show` prints one by one; past it, the first three and the
# last, so that one entry's cell stays within a line or two however many rows the dataset has.
SHOWN = 12


def compact(per_row: list | None) -> str:
    """An entry's positions per row as one cell: the one list when every row resolved to the same (`[10] on all
    4 rows`), else each row's in row order, `—` for a row the entry did not reach (`alignment_missing`). Past
    `SHOWN` rows, the first three rows', `…`, the last row's and the count (`50,000 rows; clab table shows each`);
    a row of more than `SHOWN` positions is cut the same way (`[0,1,2,…,511]`)."""
    if not per_row:
        return "—"
    if len(per_row) > 1 and all(r == per_row[0] for r in per_row):
        return f"{row_cell(per_row[0])} on all {len(per_row):,} rows"
    if len(per_row) <= SHOWN:
        return ", ".join(map(row_cell, per_row))
    return (", ".join(map(row_cell, per_row[:3])) + f", …, {row_cell(per_row[-1])} "
            f"({len(per_row):,} rows; clab table shows each)")


def row_cell(positions_: Any) -> str:
    """One row's positions as `compact` prints them."""
    if positions_ == positions.MISSING:
        return "—"
    if isinstance(positions_, list) and len(positions_) > SHOWN:
        return "[" + ",".join(map(str, positions_[:3])) + f",…,{positions_[-1]}]"
    return cell(positions_)


def shaped(shape: Any, swept: bool) -> str:
    """A saved value's shape as `run` and `show` print it: an unswept value's as it is (a dict value's per key); a
    swept value's counted over its points, `3 points, each [4, 1]` where they share one shape, the distinct shapes
    listed where there are at most three, else counted (the manifest holds each point's)."""
    if not swept:
        return cell(shape)
    distinct = list(dict.fromkeys(cell(s) for s in shape.values()))
    n = len(shape)
    if n == 1:
        return f"1 point, {distinct[0]}"
    if len(distinct) == 1:
        return f"{n} points, each {distinct[0]}"
    if len(distinct) <= 3:
        return f"{n} points, each {', '.join(distinct[:-1])} or {distinct[-1]}"
    return f"{n} points of {len(distinct)} shapes"


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
    run.add_argument("--padding", default="dataset", metavar="|".join(PADDINGS),
                     help="how each batch is padded: dataset (the default, exact), batch or sorted (fewer tokens fed, "
                          "numbers moved in the last bits)")
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
    tell.add_argument("--markdown", action="store_true", help="write FORMAT.md at the repo root")
    tell.add_argument("--verbose", action="store_true", help="say where --markdown wrote")
    args = parser.parse_args(argv)
    try:
        code = command(args)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader went away (`| head -1`): the interpreter's own flush of stdout at exit goes nowhere too.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    return code


def command(args: argparse.Namespace) -> int:
    """The parsed command, run: its exit status, a refusal said in one `clab:` line on stderr."""
    if args.command == "describe":
        if args.markdown:                   # at the repo root, from whatever directory the command runs in
            path = os.path.join(docs.ROOT, "FORMAT.md")
            with open(path, "w") as f:
                f.write(docs.markdown())
            if args.verbose:
                print(f"wrote {path}")
        else:
            print(describe(args.what))
        return 0
    if args.command in ("show", "table"):
        try:
            filled = Plan.load(args.out)
            print(show(filled) if args.command == "show" else tabulated(filled.table(*args.names), args.csv))
        except BrokenPipeError:
            raise                           # an OSError, and no refusal: the reader went away, which `main` ends
        except (ValueError, OSError) as e:
            print(f"clab: {e}", file=sys.stderr)
            return 1
        return 0

    from nnsight.intervention.backends.remote import RemoteError

    import clab

    stem = os.path.splitext(os.path.basename(args.document))[0]
    out = args.out or os.path.join("out", stem)
    first = os.environ.get("RANK", "0") == "0"          # under torchrun, rank 0 alone writes `out` and prints
    try:
        options = dict(option(text) for text in args.options)
        for key in {"device", "batch_size", "padding"} & set(options):
            raise ValueError(f"-o {key}: it is given as --{key.replace('_', '-')}")
        filled = clab.run(args.document, engine=args.engine, out=out, batch_size=args.batch_size,
                          device=args.device, padding=args.padding, **options)
    except (ValueError, NotImplementedError, OSError, RemoteError) as e:
        # A refusal on the server comes home as nnsight's error, its last line the reason; a missing document is
        # the OSError that names it.
        reason = (str(e).strip().splitlines() or [""])[-1] if isinstance(e, RemoteError) else str(e)
        if first:
            print(f"clab: {reason}", file=sys.stderr)
        return 1
    assert filled.run is not None
    if not first:
        return 0
    print(f"{stem} on {args.engine} (batch size {args.batch_size}, {how(filled.run)}) -> {out}")
    with open(os.path.join(out, "manifest.json")) as f:
        described = json.load(f)["results"]
    rows = [[d["file"], d["step"], shaped(d["shape"], bool(d["axes"]))] for d in described.values()]
    for line in aligned(rows) if rows else []:        # each column as wide as its longest cell: an eval's are long
        print("  " + line)
    print(f"  {os.path.join(out, 'manifest.json')}")
    return 0
