"""What compile knows while it works: each name produced so far (`Known`) and the context every function that
resolves a name is handed (`Compiling`)."""

from dataclasses import dataclass, field


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
    rows: dict
    sweeps: dict
    values: dict
    names: dict[str, "Known"]
    axes_of: dict
    refs: list
    used: set
    aliases: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
