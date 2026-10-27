"""What compile knows while it works (`Compiling`: the context every function that resolves a name is handed,
whose compiled records are its symbol table), and what preflight found with the model in hand (`Found`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clab.compile.answers import Answer
    from clab.plan import Entry, FeaturizerEntry, Plan


@dataclass
class Compiling:
    """What compiling an entry needs beside the entry, passed to every function that resolves a name.

    Read only: the document as a dict (`doc`), the corpus root, each dataset's rows, each sweep's label by
    document path (`sweeps`) and its values by label (`values`). Filled as entries compile, in document order:
    the records compiled so far (`featurizers`, `steps`: a point each, or a list of points), which is where every
    name a later entry references resolves (`plan.record_of`, at the consumer's point), each entry's sweep axes
    (`axes_of`), and every value any entry references (`used`), which decides what the run gathers. A fit's body
    compiles in a copy whose `steps` is its own scope (the root's records, then its own as they compile) and maps
    each featurizer it trains to the name it resolves to there (`aliases`: `rot` is `<fit>.rot`, the instance being
    trained); it shares `axes_of` and `used`. `points` holds each point of the document built so far, by its
    coordinates (`point`), and `chats` each dataset a step reads a chat column of (shared with a body), which a
    `template` needs."""
    doc: dict
    root: str
    rows: dict[str, list[dict]]
    sweeps: dict[tuple, str]
    values: dict[str, list]
    featurizers: dict[str, FeaturizerEntry]
    steps: dict[str, Entry]
    axes_of: dict[str, list[str]]
    used: set[str]
    aliases: dict[str, str] = field(default_factory=dict)
    points: dict[tuple, Plan] = field(default_factory=dict)
    chats: set[str] = field(default_factory=set)


@dataclass
class Found:
    """What preflight found, which the run needs beside the records it completed: each dataset input's encoded
    rows and padded width (`inputs`), room for what each generate step's rows said (`said`), each column
    reference's answers (`answers`, see `answers.answer`), each featurizer's width (`widths`, the root's alone),
    and for each fit point the same for its body (`bodies`): over its training rows (`train`) and under each of
    its named `eval` substitutions (`evals`), each beside the plan the body runs in. `swap` is the substitution
    the body runs under (`{<body dataset>: <dataset read>}`; the root's is none), which names the dataset a
    record's positions are under. Each table with a row per example holds every row of its dataset, by id; which
    rows a run is over is its scope's to say (`batches.over`)."""
    swap: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, dict] = field(default_factory=dict)
    said: dict = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)
    widths: dict[str, int] = field(default_factory=dict)
    bodies: dict = field(default_factory=dict)
