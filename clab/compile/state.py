"""What compile knows while it works: each name produced so far (`Known`), the context every function that
resolves a name is handed (`Compiling`), and what preflight found with the model in hand (`Found`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clab.compile.answers import Answer
    from clab.plan import Plan


@dataclass
class Known:
    """What compile knows of a name: its `kind` (a `value`, a `column`, a `featurizer`, a featurizer's `scalar`)
    and, for a value, its layout (`dims`), the dataset its rows are aligned to (`data`), the coordinates it is
    in (`space`: the encodes left open around it, outermost first, each its featurizer and `select`; none for a
    site's own layout), its site's `select` on the axes it still has, how many positions it holds on every row
    (`width`, None when rows differ) and whether it is text (a `detokenize`). `name` is how an op's argument
    referenced it, where `references` resolved one."""
    dims: list[str] = field(default_factory=list)
    data: str | None = None
    kind: str = "value"
    space: list[dict] = field(default_factory=list)
    select: dict = field(default_factory=dict)
    width: int | None = None
    text: bool = False
    name: str = ""


@dataclass
class Compiling:
    """What compiling an entry needs beside the entry, passed to every function that resolves a name.

    Read only: the document as a dict (`doc`), the corpus root, each dataset's rows, each sweep's label by
    document path (`sweeps`) and its values by label (`values`). Filled as entries compile, in document
    order: each name produced so far (`names`, each a `Known`), each entry's sweep axes (`axes_of`), the names the entry being compiled references (`refs`),
    which decide the axes it inherits, and every value any entry references (`used`), which decides what the run
    gathers. A fit's body compiles in a copy whose
    `names` is its own scope; it shares `refs`, since what the body references is what the fit references,
    `axes_of` and `used`, and maps each featurizer it trains to the name it resolves to there (`aliases`:
    `rot` is `<fit>.rot`, the instance being trained). `points` holds each point of the document built so far,
    by its coordinates (`point`)."""
    doc: dict
    root: str
    rows: dict[str, list[dict]]
    sweeps: dict[tuple, str]
    values: dict[str, list]
    names: dict[str, Known]
    axes_of: dict[str, list[str]]
    refs: list[str]
    used: set[str]
    aliases: dict[str, str] = field(default_factory=dict)
    points: dict[tuple, Plan] = field(default_factory=dict)


@dataclass
class Found:
    """What preflight found, which the run needs: each dataset input's encoded rows and padded width
    (`inputs`), every prefill record's positions per row by point (`positions`, which the run extends with the
    other records'), the indices every record's `select` keeps on each axis (`selects`), room for what each
    generate step's rows said (`said`), each column reference's answers (`answers`, see `answers.answer`),
    each featurizer's width (`widths`, the root's alone), and for each fit point the same for its body
    (`bodies`): over its training rows (`train`) and under each of its named `eval` substitutions (`evals`),
    each beside the plan the body runs in."""
    inputs: dict[str, dict] = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    said: dict = field(default_factory=dict)
    selects: dict = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)
    widths: dict[str, int] = field(default_factory=dict)
    bodies: dict = field(default_factory=dict)
