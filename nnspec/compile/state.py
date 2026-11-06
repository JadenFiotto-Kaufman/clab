"""What compile knows while it works (`Compiling`: the context every function that resolves a name is handed,
whose compiled records are its symbol table), and what preflight found with the model in hand (`Findings`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnspec.compile.answers import Answer
    from nnspec.plan import Entry, FeaturizerEntry, Plan


@dataclass
class Compiling:
    """What compiling an entry needs beside the entry, passed to every function that resolves a name.

    Read only: the document as a dict (`doc`), the corpus root, each dataset's rows, each sweep's label by
    document path (`sweeps`) and its values by label (`values`, to which a load adds a sweep its value depends on
    and the document does not declare, in axis order after the document's own). Filled as entries compile, in
    document order:
    the records compiled so far (`featurizers`, `steps`: a point each, or a list of points), which is where every
    name a later entry references resolves (`plan.record_of`, at the consumer's point), each entry's sweep axes
    (`axes_of`), every value any entry references (`used`) but as a live operand (`live`: an earlier entry of the
    forward an entry of it writes with), which together decide what the run gathers, and `used` what it keeps. A fit's
    body compiles in a copy whose `steps` is its own scope (the root's records and the fit being compiled, then the
    body's own as they compile), and each of its evals in a copy whose `steps` are the root's and that fit; each
    shares `axes_of`, `used` and `live`. `points` holds each point of the document built so far, by its
    coordinates (`document_at_point`), `chats` each dataset a step reads a chat column of (shared with a body), which a
    `template` needs, and `sources` each run a load reads, by its directory, read once (`compile.loaded_run`)."""
    doc: dict
    root: str
    rows: dict[str, list[dict]]
    sweeps: dict[tuple, str]
    values: dict[str, list]
    featurizers: dict[str, FeaturizerEntry]
    steps: dict[str, Entry]
    axes_of: dict[str, list[str]]
    used: set[str]
    live: set[str]
    points: dict[tuple, Plan] = field(default_factory=dict)
    chats: set[str] = field(default_factory=set)
    sources: dict[str, dict] = field(default_factory=dict)


@dataclass
class Findings:
    """What preflight found, which the run needs beside the records it completed: each dataset input's encoded
    rows and padded width (`inputs`, a fit body's forwards' among them), room for what each generate step's rows
    said (`said`) and each column reference's answers (`answers`, see `answers.answer`). Each table with a row per
    example holds every row of its dataset, by id; which rows a run is over is its scope's to say
    (`batches.scope_ids`)."""
    inputs: dict[str, dict] = field(default_factory=dict)
    said: dict = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)
