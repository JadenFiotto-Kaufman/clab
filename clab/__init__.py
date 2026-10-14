"""clab runs mechanistic-interpretability experiments from declarative JSON documents.

`run` compiles a document, runs it on an engine and writes its output
directory; `load` reads an output directory back into the filled plan;
`table` flattens a saved value over every point; `describe` prints the
format's tables; `compile` turns a document into its plan without running it.
"""

import os
import time

from .cli import describe
from .compile import compile, load as load_document
from .output import load, table, write

__all__ = ["run", "load", "table", "describe", "compile"]


def run(path: str, engine: str = "nnterp", out: str | None = None, batch_size: int = 8,
        remote: bool | str = False) -> dict:
    """Run the document at `path` on `engine` and write its output to `out` (default `out/<document name>`).
    Returns the filled plan."""
    from .engines import ENGINES

    plan = compile(load_document(path), path)
    loaded = ENGINES[engine](plan["model"])
    started = time.time()
    filled = loaded.execute(plan, batch_size=batch_size, remote=remote)
    out = out or os.path.join("out", plan["document"]["name"])
    write(filled, out, {"engine": engine, "remote": remote, "batch_size": batch_size,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                        "seconds": round(time.time() - started, 3)})
    return filled
