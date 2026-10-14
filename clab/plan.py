"""The plan: one model that is a document as written and, compiled, the document resolved for running.

`Plan(path)` reads a document; `plan.compile()` returns it compiled (rows loaded, sites and named
interventions inlined, bundles loaded, sweeps expanded into points, names resolved); an engine's
`execute` fills `plan.results` and `plan.run`; `plan.dump(out)` writes a run's output directory and
`Plan.load(out)` reads one back. The records (sites, reads and writes, ops, positions, steps, datasets,
featurizers) are the same models in both states; a *derived* field is one compile or the run fills.

The plan's methods are views and serialization, never execution: it pickles, prints and means the same
thing wherever it lands, which is what lets it travel into a remote session and back.
"""

import json
import os
from typing import Annotated, Any, Literal, NoReturn, TypeVar, Union, get_args, get_origin

from pydantic import (BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError, field_serializer,
                      field_validator, model_validator)
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import core_schema
from torch import Tensor

from . import address, data, featurizers, ops


class PlanError(ValueError):
    """A document that cannot mean anything; the message starts with the document path of what is wrong."""


def fail(path: str, message: str) -> NoReturn:
    raise PlanError(f"{path}: {message}" if path else message)


def derived(default: Any, description: str, **more: Any) -> Any:
    """A field compile or the run fills, which a document leaves out."""
    return Field(default, description=description, json_schema_extra={"derived": True}, **more)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, use_attribute_docstrings=True)

    @model_validator(mode="before")
    @classmethod
    def _known(cls, given: Any) -> Any:
        if isinstance(given, dict) and cls.model_config.get("extra") == "forbid":
            for key in given:
                if key not in cls.model_fields:
                    raise ValueError(f"unknown field {key!r}; valid: {', '.join(documented(cls))}")
        return given


def documented(cls: type[BaseModel]) -> list[str]:
    """The fields a document writes: every field but the derived ones."""
    return [k for k, f in cls.model_fields.items()
            if not (isinstance(f.json_schema_extra, dict) and f.json_schema_extra.get("derived"))]


class Sweep(Record):
    """A value standing for several. Whatever uses it runs once per value, a point each; one sweep reached
    from two places is one axis."""
    sweep: list[Any] = Field(min_length=1)

    @field_validator("sweep")
    @classmethod
    def _flat(cls, values: list) -> list:
        if '"sweep"' in json.dumps(values):
            raise ValueError("a sweep's values hold no sweep")
        return values


class ByShape:
    """A union validated by the one member the input's shape picks (a sweep, a dict, a string, a number, a
    list, null), so a wrong value gets one error, from the member it was meant for."""

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        members = get_args(source)

        def pick(value: Any) -> str:
            return f"~{next((i for i, m in enumerate(members) if shaped(m, value)), 0)}"
        return core_schema.tagged_union_schema({f"~{i}": handler.generate_schema(m) for i, m in enumerate(members)},
                                               discriminator=pick)


def shaped(member: Any, value: Any) -> bool:
    """Whether `value` has the shape of union member `member`."""
    if member is Sweep:
        return isinstance(value, Sweep) or isinstance(value, dict) and "sweep" in value
    if isinstance(member, type) and issubclass(member, BaseModel):
        return isinstance(value, member) or isinstance(value, dict) and "sweep" not in value
    if get_origin(member) is Literal:
        return isinstance(value, tuple(type(a) for a in get_args(member)))
    if get_origin(member) in (list, tuple):
        return isinstance(value, (list, tuple))
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return {int: number and isinstance(value, int), float: number, str: isinstance(value, str),
            bool: isinstance(value, bool)}.get(member, False)


T = TypeVar("T")
Swept = Annotated[Union[T, Sweep], ByShape()]
AT = "the point's index on each sweep axis it varies over"
DATASET = "the dataset whose rows the step's values are aligned to"


class Model(Record):
    """The pinned checkpoint. The attention interior needs `attn_implementation: "eager"`."""
    key: str
    revision: str
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    attn_implementation: Literal["eager", "sdpa"] | None = None


class Data(Record):
    """A dataset: `path` is `data/<path>.json` under the corpus root, or `rows` are inline, or `hf` names a
    pinned hub dataset (`revision`, `take` rows of `split`). `split` keeps the rows of that split."""
    path: str | None = None
    rows: list[dict] | None = None
    hf: str | None = None
    revision: str | None = None
    split: str | None = None
    take: int | None = None
    sha256: str | None = derived(None, "the digest of the rows, which a re-materialization must match")
    n_rows: int | None = derived(None, "how many rows the dataset has")

    @model_validator(mode="after")
    def _one_source(self):
        sources = [s for s in (self.path, self.hf) if s is not None]
        if len(sources) > 1 or not sources and self.rows is None:
            raise ValueError("a dataset is a 'path', inline 'rows' or a hub dataset 'hf'")
        if self.hf is not None and self.revision is None:
            raise ValueError("a hub dataset is pinned: give its 'revision'")
        if self.hf is None and (self.revision is not None or self.take is not None):
            raise ValueError("'revision' and 'take' belong to a hub dataset ('hf')")
        if not sources and self.split is not None:
            raise ValueError("'split' keeps rows of a file or a hub dataset; inline 'rows' are the rows")
        return self


class Site(Record):
    """A component, narrowed: `layers` (an int, or `"all"`: one value, the layer axis first) on a block's
    component; at most one of `heads` and `units`, as the component allows."""
    component: str = Field(json_schema_extra={"enum": list(address.COMPONENTS)})
    layers: Swept[int | Literal["all"]] | None = None
    heads: Swept[list[int]] | None = None
    units: Swept[list[int]] | None = None

    @model_validator(mode="after")
    def _narrowing(self):
        if self.component not in address.COMPONENTS:
            raise ValueError(f"unknown component {self.component!r}; valid: {', '.join(address.COMPONENTS)}")
        meaning = address.COMPONENTS[self.component]
        if meaning["per_layer"] != (self.layers is not None):
            raise ValueError(f"{self.component} is a block's value and needs 'layers'" if meaning["per_layer"]
                             else f"{self.component} is the model's own value and takes no layers")
        narrowed = [k for k in ("heads", "units") if getattr(self, k) is not None]
        if len(narrowed) > 1:
            raise ValueError("a site narrows one axis: heads or units, not both")
        for k in narrowed:
            if k not in meaning["narrow"]:
                raise ValueError(f"{self.component} is narrowed by {', '.join(meaning['narrow'])}, not {k}")
        return self


class Scope(Record):
    """`{"variable": column}`: the row's own text of that column, located in its input; or `{"segment":
    "role[n]" | "role" | "eos"}`: a chat turn, or where a generated row stopped."""
    variable: str | None = None
    segment: str | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.variable is None) == (self.segment is None):
            raise ValueError("a scope is {'variable': column} or {'segment': 'role[n]' | 'role' | 'eos'}")
        return self


class Pos(Record):
    """`pos` names token positions within a scope.

    ```
    pos := int                                    # the one shorthand: {"index": int}
         | {"frame"?: "prompt" | "generated",     # default "prompt"
            "scope"?: {"variable": column} | {"segment": "role[n]" | "role" | "eos"},
            exactly one of "index": int | "span": [start, stop] | "all": true}
    ```

    Indices count the scope's tokens and may be negative (from its end); a span is half-open. The prompt's
    tokens are the row's tokens that are not special (BOS is never one) and not padding. `variable` scopes
    to the row's own text of that column, located in its input; `segment` to a chat turn (`user[1]` is the
    second user message) or, in the generated frame, to where the row stopped (`eos`). Text a scope names
    that occurs twice in the row is refused. The generated frame counts decode steps: index *k* is the
    forward that chose new token *k*, read at its last position. A turn has no meaning in the generated
    frame, nor `eos` in the prompt.

    A slice is rectangular when both ends count from the same side (or it runs to the end from a negative
    start): the same width on every row. Otherwise (`all`, or a span from the start to the end) it is
    ragged: a read then gives one entry per token, `{values, row, tok}`.
    """
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [k]} for k in ("index", "span", "all")]})
    frame: Literal["prompt", "generated"] = "prompt"
    scope: Scope | None = None
    index: Swept[int] | None = None
    span: Swept[tuple[int, int]] | None = None
    all: Literal[True] | None = None

    @model_validator(mode="after")
    def _grammar(self):
        if [self.index, self.span, self.all].count(None) != 2:
            raise ValueError("a position takes exactly one of 'index', 'span', 'all'")
        segment = self.scope.segment if self.scope else None
        if segment == "eos" and self.frame == "prompt":
            raise ValueError("segment 'eos' is where a row stopped generating; a prompt does not stop")
        if segment not in (None, "eos") and self.frame == "generated":
            raise ValueError(f"segment {segment!r} is a chat turn; a continuation has no turns")
        return self


class Op(Record):
    """An op inside a read or a write: `{"op": <name>, ...its arguments}` (see Ops). `featurizer` (and
    `features`, a subset of its latents) make any op run on the value's coordinates in that featurizer's
    space, the change written back."""
    model_config = ConfigDict(extra="allow")
    op: str = Field(json_schema_extra={"enum": list(ops.OPS)})
    featurizer: str | None = None
    features: Swept[list[int]] | None = None

    @model_validator(mode="after")
    def _arguments(self):
        ops.check(self.op, arguments(self))
        if self.features is not None and self.featurizer is None:
            raise ValueError("features are a featurizer's latents; name the 'featurizer' they belong to")
        return self


def arguments(op: "Op | OpStep") -> dict:
    """An op record's arguments by document spelling: its own fields, and `featurizer` when the op takes one
    (`encode`) rather than running in that featurizer's space."""
    given = dict(op.model_extra or {})
    if isinstance(op, Op) and op.op in ops.OPS and "featurizer" in ops.parameters(op.op):
        given["featurizer"] = op.featurizer
    return given


def space(op: "Op") -> str | None:
    """The featurizer whose space an op in a read or write runs in, if any."""
    return None if op.op in ops.OPS and "featurizer" in ops.parameters(op.op) else op.featurizer


class Access(Record):
    """A write or a read: the `site` it lands on (a declared name, or one inline), the positions `pos` names
    there, and the `ops` it runs on that selection. Several writes at one site compose in listed order on one
    copy of the value; a read at a written site observes the writes."""
    site: Annotated[str | Site, ByShape()]
    pos: Swept[int | Pos]
    ops: list[Op] = []
    dims: list[str] | None = derived(None, "the layout of what a read binds")


class Interventions(Record):
    """A bundle of `writes` and `reads` (each by name), pulled into a step by `"interventions": "<name>"` or a
    list of names. Its names resolve where a step pulls it in."""
    writes: dict[str, Access] = {}
    reads: dict[str, Access] = {}


class Forward(Record):
    """One forward over every row of a dataset input. `input` is `<dataset>.<column>` or
    `<dataset>.<column>[i]`, a text or a chat; `writes` land where they name before anything later reads them;
    each read binds `<step>.<read>`; `interventions` pulls named bundles of writes and reads in, in listed
    order, before the step's own.

    A chat input (a list of `{"role", "content"}` messages) is rendered through the checkpoint's own chat
    template without a generation prompt, so the rendered prompt ends with the template's close of the last
    message (`[/INST]` on Llama-2-style templates): `pos: -1` with no scope is that close's last token, and
    `{"scope": {"segment": "user[1]"}}` names the message's own tokens.

    Rows run in consecutive batches of the run's batch size, each padded on the left to the input's longest
    row, with position ids that count real tokens. A row's numbers then do not depend on which rows share its
    batch, but they do depend on the batch's shape: a trailing partial batch is a smaller matrix, and can
    differ in the last bits. Results reproduce to the bit at a fixed batch size."""
    kind: Literal["forward"]
    input: str
    writes: dict[str, Access] = {}
    reads: dict[str, Access] = {}
    interventions: str | list[str] | None = None
    at: dict[str, int] = derived({}, AT)
    dataset: str | None = derived(None, DATASET)

    @field_validator("input")
    @classmethod
    def _input(cls, text: str) -> str:
        data.parse_input(text)
        return text


class Generate(Forward):
    """Generation from every row of a dataset input. Writes land on the prompt; reads in the `generated` frame
    read decode steps. The step's own value is the new token ids, `[rows, new]`; `decode` makes them text."""
    kind: Literal["generate"]  # type: ignore[assignment]
    max_new_tokens: Swept[int]
    min_new_tokens: Swept[int] = 0
    do_sample: bool = False


class OpStep(Record):
    """An op over a value earlier steps produced: `of` names it, and the op's arguments sit beside it (see
    Ops). The step's value is the op's result; a step holds exactly one op, and chaining is the next step."""
    model_config = ConfigDict(extra="allow")
    kind: Literal["op"]
    op: str = Field(json_schema_extra={"enum": list(ops.OPS)})
    of: str
    at: dict[str, int] = derived({}, AT)
    dataset: str | None = derived(None, DATASET)
    dims: list[str] | None = derived(None, "the layout of the step's value")

    @model_validator(mode="after")
    def _arguments(self):
        ops.check(self.op, arguments(self))
        return self


class Optimizer(Record):
    """The fit's optimizer, over the trained featurizers' parameters only."""
    name: Literal["adamw", "adam", "sgd"]
    lr: Swept[float]
    weight_decay: Swept[float] = 0.0


class EarlyStop(Record):
    """End the fit after `patience` epochs without improvement of a body op's mean, scored on the training data."""
    metric: str
    mode: Literal["max", "min"]
    patience: Swept[int]


class Eval(Record):
    """After the fit, run the body once with each dataset `data` maps read as the other (`{"train": "test"}`)."""
    data: dict[str, str]


class Fit(Record):
    """Train the featurizers `train` names. Each epoch runs the body `steps` over minibatches of `batch_size`
    rows and minimizes the weighted sum `objective` (`{term: weight}`; a term is a body op, averaged, or a
    trained featurizer's scalar, `<featurizer>.<scalar>`). Inside the body a trained featurizer's name and
    `<fit>.<featurizer>` are the one being trained. Afterwards `<fit>.<featurizer>` is the trained one,
    `<fit>.<op>` a body op under `eval`, `<fit>.<step>.<read>` a body read, and `<fit>.history.<op>` the
    body op's mean per epoch."""
    kind: Literal["fit"]
    train: list[str]
    objective: dict[str, Swept[float]]
    epochs: Swept[int]
    batch_size: Swept[int]
    seed: Swept[int]
    optimizer: Optimizer
    steps: "dict[str, Entry]"
    early_stop: EarlyStop | None = None
    eval: Eval | None = None
    at: dict[str, int] = derived({}, AT)


KINDS = {"forward": Forward, "generate": Generate, "op": OpStep, "fit": Fit}


def _kind(value: Any) -> str | None:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return f"~{kind}" if kind in KINDS else None


def _points(value: Any) -> str:
    return "~points" if isinstance(value, list) else "~one"


# A `~` tag names a union branch; `PlanError`'s paths leave it out. A compiled entry that is swept is the list
# of its points.
Step = Annotated[Union[Annotated[Forward, Tag("~forward")], Annotated[Generate, Tag("~generate")],
                       Annotated[OpStep, Tag("~op")], Annotated[Fit, Tag("~fit")]],
                 Discriminator(_kind, custom_error_type="kind", custom_error_message="unknown step kind")]
Entry = Annotated[Union[Annotated[Step, Tag("~one")], Annotated[list[Step], Tag("~points")]], Discriminator(_points)]
Fit.model_rebuild()


class Featurizer(Record):
    """A space a read or write happens in: `{"kind": <kind>, ...its fields}` (see Featurizer kinds)."""
    model_config = ConfigDict(extra="allow")
    kind: str
    at: dict[str, int] = derived({}, AT)
    bundle: SkipJsonSchema[dict[str, Tensor] | None] = derived(
        None, "the tensors of the bundle `file_path` names, checked against its header", exclude=True)

    @model_validator(mode="after")
    def _fields(self):
        featurizers.check(self.kind, self.model_extra or {})
        return self


FeaturizerEntry = Annotated[Union[Annotated[Featurizer, Tag("~one")], Annotated[list[Featurizer], Tag("~points")]],
                            Discriminator(_points)]


class Axis(Record):
    """A sweep axis: the document path of its sweep and its values."""
    path: str
    values: list[Any]


class Run(Record):
    """How a plan ran, set by `execute`. `tokens` and `positions` describe values without holding any: how
    each dataset input tokenized (token strings per row), and the token index every positioned read and write
    resolved to, per row (keyed by coordinate when its step is swept)."""
    engine: str
    remote: bool | str
    batch_size: int
    versions: dict[str, str] = {}
    started: str = ""
    seconds: float = 0.0
    tokens: dict[str, list[list[str]]] = {}
    positions: dict[str, Any] = {}


class Plan(Record):
    """A complete experiment."""
    description: str = ""
    """what the experiment is, in prose"""
    model: Model
    """the checkpoint, and how to load it"""
    data: dict[str, Data]
    """each dataset by name"""
    sites: dict[str, Site] = {}
    """named locations, referenced by `site`"""
    interventions: dict[str, Interventions] = {}
    """named bundles of writes and reads, pulled into a step by `interventions`"""
    featurizers: dict[str, FeaturizerEntry] = {}
    """named spaces a read or write happens in"""
    steps: dict[str, Entry]
    """what runs, in order"""
    saves: list[str] = []
    """the values that come home, by name"""
    document: str | None = derived(None, "the document file the plan was read from; written relative to its "
                                         "corpus root, where data paths and bundles resolve")
    compiled: bool = derived(False, "whether this is the compiled plan")
    sweeps: dict[str, Axis] = derived({}, "each sweep axis by label, in axis order")
    results: SkipJsonSchema[dict[str, Any]] = derived(
        {}, "each saved value by name; a swept one is a dict keyed by coordinate", exclude=True)
    run: Run | None = derived(None, "how the plan ran, set by `execute`")

    def __init__(self, path: str | None = None, /, **fields: Any):
        if path is not None:
            with open(path) as f:
                fields = {**json.load(f), "document": os.path.abspath(path)}
        try:
            super().__init__(**fields)
        except ValidationError as e:
            raise PlanError("\n".join(message(error) for error in e.errors())) from None

    @model_validator(mode="before")
    @classmethod
    def _names_and_sweeps(cls, given: Any) -> Any:
        if isinstance(given, dict):
            if find_sweeps(given) and given.get("compiled"):
                raise ValueError("a compiled plan holds no sweep; its sweeps are expanded into points")
            for name in given.get("steps", {}):
                if "." in name:
                    raise ValueError(f"steps.{name}: a name has no dots; a dot separates a thing from its member")
                if name in given.get("data", {}) or name in given.get("featurizers", {}):
                    raise ValueError(f"steps.{name}: {name!r} is also a dataset or a featurizer")
        return given

    @model_validator(mode="after")
    def _bundles(self):
        # A bundle's file is its source of truth, and tensors never go in a document: a compiled plan read back
        # without its bundles loads them again from their `file_path`.
        for entry in self.featurizers.values() if self.compiled else []:
            for f in points(entry):
                fields = f.model_extra or {}
                if fields.get("file_path") is not None and f.bundle is None:
                    f.bundle = featurizers.bundle(corpus_root(self), f.kind, fields, self.model)
        return self

    @field_serializer("document", when_used="json")
    def _relative(self, document: str | None) -> str | None:
        return document and os.path.relpath(document, corpus_root(self))

    def __repr__(self) -> str:
        if self.run is not None:
            return f"Plan({name(self)}, run on {self.run.engine}, {len(self.results)} results)"
        if self.compiled:
            n = 1
            for axis in self.sweeps.values():
                n *= len(axis.values)
            return f"Plan({name(self)}, compiled, {n} point{'s' * (n != 1)})"
        return f"Plan({name(self)}, uncompiled)"

    def compile(self) -> "Plan":
        """This plan compiled: rows loaded, sites and named interventions inlined, bundles loaded, sweeps
        expanded into points, every name resolved and every value's layout and dataset known."""
        from .compiler import compile
        return compile(self)

    def dump(self, out: str) -> None:
        """A run's output directory: one file per save and `manifest.json`."""
        from .output import dump
        dump(self, out)

    @classmethod
    def load(cls, out: str) -> "Plan":
        """The filled plan an output directory holds: the inverse of `dump`."""
        from .output import load
        return load(out)

    def write(self, path: str) -> None:
        """The plan as a JSON document, in its current state, without its results and run."""
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json", exclude_defaults=True, exclude={"run", "document"}), f, indent=1)
            f.write("\n")

    def table(self, *names: str, columns: list[str] | None = None) -> list[dict]:
        """The saved values `names` as rows, joined wide on their shared axes (see `output.table`)."""
        from .output import table
        return table(self, names, columns)


def corpus_root(plan: Plan) -> str:
    """Where data paths and bundles resolve: the parent of the directory of the document `Plan(path)` read, or
    the working directory for a plan built from fields or loaded from a run (whose manifest records its
    document relative to the corpus root)."""
    return os.path.dirname(os.path.dirname(plan.document)) if plan.document and os.path.isabs(plan.document) \
        else os.getcwd()


def name(plan: Plan) -> str:
    """The plan's name: its document's file name, without the extension."""
    return os.path.splitext(os.path.basename(plan.document))[0] if plan.document else "plan"


def message(error: Any) -> str:
    """A validation error as `<document path>: <what>`."""
    path = ""
    for part in error["loc"]:
        if isinstance(part, int):
            path += f"[{part}]"
        elif not part.startswith("~"):
            path += f".{part}" if path else part
    text = error["msg"].removeprefix("Value error, ")
    if error["type"] == "kind":
        given = error["input"].get("kind") if isinstance(error["input"], dict) else error["input"]
        text = f"unknown step kind {given!r}; valid: {', '.join(KINDS)}"
    return f"{path}: {text}" if path else text


def points(entry: Any) -> list:
    """A plan entry as its list of points; `[entry]` when it is not swept."""
    return entry if isinstance(entry, list) else [entry]


def coordinate(plan: Plan, at: dict) -> Any:
    """A point's key in a swept result: its axis's value (a list as a tuple), or the tuple of its axes' values
    in axis order."""
    values = [tuple(v) if isinstance(v := plan.sweeps[label].values[i], list) else v for label, i in at.items()]
    return values[0] if len(values) == 1 else tuple(values)


def labelled(plan: Plan, at: dict, labels: bool = True) -> str:
    """A point's coordinate as text: `pos=-1`, `layers=0,pos=-1`; without its labels, `-1`, `0,-1`."""
    parts = [json.dumps(plan.sweeps[label].values[i], separators=(",", ":")) for label, i in at.items()]
    return ",".join(f"{label}={part}" for label, part in zip(at, parts)) if labels else ",".join(parts)


def dims_of(plan: Plan, name: str) -> list[str]:
    """The axes of the value `name`: a step's value, or a step's read."""
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if member:
        return step.reads[member].dims
    return ["rows", "new"] if step.kind == "generate" else step.dims


# -- sweeps ---------------------------------------------------------------------

UNSWEPT = ("kind", "op", "site", "component", "input", "interventions", "train", "featurizer", "path", "rows",
           "split", "file_path", "of", "hf", "revision", "saves")


def find_sweeps(document: dict) -> dict[tuple, str]:
    """Every `{"sweep": [...]}` standing for a value, by document path, labelled by the last key of its path.

    A sweep may stand for any value except under `description` and `model`, on a name, a kind, a dataset's
    source, or a `file_path`. Two sweeps with one label are refused; one sweep reached from two places (a swept site
    two steps use) is one axis, so its points zip rather than cross. A step that uses a swept thing, or
    references a value that varies, becomes a list of points, one per combination of its axes, each a whole
    step whose `at` gives its index on each axis; a step no sweep touches runs once. A saved value of a
    swept step is a dict keyed by coordinate: the axis's value, or a tuple of them in axis order.
    """
    found: dict[tuple, str] = {}

    def walk(value: Any, path: tuple) -> None:
        if isinstance(value, dict) and set(value) == {"sweep"}:
            where = ".".join(map(str, path))
            label = next(k for k in reversed(path) if isinstance(k, str))
            if path[0] in ("description", "model") or label in UNSWEPT:
                raise ValueError(f"{where}: a sweep cannot stand for {label!r}")
            for other, other_label in found.items():
                if other_label == label:
                    raise ValueError(f"{where}: label {label!r} is also the sweep at "
                                     f"{'.'.join(map(str, other))}; rename one")
            found[path] = label
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in ("results", "run", "sweeps"):
                    walk(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + (i,))

    walk(document, ())
    return found
