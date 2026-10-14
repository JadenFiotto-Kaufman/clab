"""The plan: one model that is a document as written and, compiled, the document resolved for running.

`Plan(path)` reads a document; `plan.compile()` returns it compiled (rows loaded, sites and named
interventions inlined, bundles loaded, sweeps expanded into points, names resolved); an engine's
`execute` fills `plan.results` and `plan.run`; `plan.dump(out)` writes a run's output directory and
`Plan.load(out)` reads one back. The records (sites, reads and writes, ops, positions, steps, datasets,
featurizers) are the same models in both states; a *derived* field is one compile or the run fills.

The plan's methods are views and serialization, never execution: it pickles, prints and means the same
thing wherever it lands, which is what lets it travel into a remote session and back.
"""

import inspect
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
    if get_origin(member) is Annotated:
        member = get_args(member)[0]
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
    """The pinned checkpoint: `key`, `revision` and `dtype` (torch's spelling). Every other key goes to the
    engine's loader untouched (`attn_implementation`, `device_map`, ...), and a loader refuses what it cannot
    honor. The attention interior needs `attn_implementation: "eager"`."""
    model_config = ConfigDict(extra="allow")
    key: str
    revision: str
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"


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
    component, and `select` (`{<axis>: [indices]}`) on the axes it publishes besides its rows and tokens."""
    component: str = Field(json_schema_extra={"enum": list(address.COMPONENTS)})
    layers: Swept[int | Literal["all"]] | None = None
    select: dict[str, Swept[list[int]]] | None = None

    @model_validator(mode="after")
    def _narrowing(self):
        if self.component not in address.COMPONENTS:
            raise ValueError(f"unknown component {self.component!r}; valid: {', '.join(address.COMPONENTS)}")
        if address.COMPONENTS[self.component]["per_layer"] != (self.layers is not None):
            raise ValueError(f"{self.component} is a block's value and needs 'layers'"
                             if address.COMPONENTS[self.component]["per_layer"]
                             else f"{self.component} is the model's own value and takes no layers")
        valid = address.selectable(self.component)
        for axis, index in (self.select or {}).items():
            if axis not in valid:
                raise ValueError(f"{self.component} has no axis {axis!r} to select; its axes: {', '.join(valid)}")
            if index == []:
                raise ValueError(f"select {axis}: an empty list selects nothing")
        return self


# A half-open `[start, stop]`, each end negative from the end.
Span = Annotated[list[int], Field(min_length=2, max_length=2)]


class Scope(Record):
    """`{"column": "<dataset>.<column>"}`: the row's own text of that column (an indexed input's variables are
    named whole: `pairs.counterfactual_inputs_variables[0].entity`); `{"text": "<literal>"}`: that text; or
    `{"turn": "<role>"}`: that role's messages of a chat, each an occurrence. `column` and `text` may each be
    a list: the scope's texts are then every one listed (a string is the one-element list), their occurrences
    together, ordered by where each ends. Every scope takes `occurrence`, which of its occurrences it is (the
    selector
    `frame` and `index` use: an int, negative from the last, a `[start, stop]` span or `"all"`; default 0,
    the first); `column` and `text` also take `strip`, whether each text is stripped of surrounding whitespace
    before it is located (a turn's contents are located as the template rendered them)."""
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [k]} for k in ("column", "text", "turn")]})
    column: str | list[str] | None = None
    text: str | list[str] | None = None
    turn: str | None = None
    occurrence: Swept[int | Span | Literal["all"]] = 0
    strip: bool = False

    @model_validator(mode="after")
    def _one(self):
        if [self.column, self.text, self.turn].count(None) != 2:
            raise ValueError("a scope is one of {'column': '<dataset>.<column>'}, {'text': '<literal>'}, "
                             "{'turn': '<role>'}")
        if [] in (self.column, self.text):
            raise ValueError("a scope's list of columns or texts names at least one")
        if self.turn is not None and "strip" in self.model_fields_set:
            raise ValueError("a turn's messages are located as the chat template rendered them; it takes no strip")
        if self.turn is not None and not self.turn.isidentifier():
            raise ValueError(f"turn {self.turn!r} is a role; its messages are its occurrences, so "
                             "`{'turn': 'user', 'occurrence': 1}` is the second user message")
        return self


class Pos(Record):
    """`pos` is three stages of narrowing: `frame` picks forward calls, `scope` picks text within their tokens,
    `index` picks positions within that.

    ```
    pos   := int                                   # shorthand: {"frame": 0, "index": int}
           | {"frame"?: frame, "scope"?: scope, "index"?: index}
    frame := int | [start, stop] | "generated" | "all" | "eos"         # default 0
    scope := {"column" | "text": one or a list, "occurrence"?: selector, "strip"?: bool}
           | {"turn": "<role>", "occurrence"?: selector}
    index := int | [start, stop] | "all"                               # default "all"
    selector := int | [start, stop] | "all"                            # occurrence's; default 0
    ```

    **frame**: 0 is the prefill, the only call a `forward` step has; k >= 1 is decode call k; a negative k
    counts from the row's last call; a span is half-open; `"generated"` is every decode call, `"all"` the
    prefill and every decode call (in a `forward`, its one call: so a named intervention written at `"all"`
    is on in every call of whichever step names it), `"eos"` the call where the row stopped (the one that
    chose its EOS). A
    position is where an input token sits, in a forward and in a generation alike: the prefill holds every
    prompt token, its last position the distribution over generated token 0; under the cache decode call k
    holds one position, generated token k-1 (its input), and its logits are the distribution over token k. A
    row's calls end with the one that chose its last token, so the token it stopped on is never an input. The
    prompt's tokens are the row's tokens that are not special (BOS is never one) and not padding; a
    generation's prefill computes the head (`lm_head_output`, `logits`) at its last position only, so there
    call 0 holds that one position.

    **scope** (default: every token the frame selected): `column` and `text` locate text in the selected
    calls' tokens, in the prompt by the tokenizer's offsets and in the generated tokens by decoding growing
    prefixes. A list of columns or texts locates every occurrence of each. Occurrences are ordered by where
    they end, then where they start: where an occurrence becomes known and where index -1 lands, so a write
    found online, a read and the run record count them alike (`b` inside `abc` comes before it). `occurrence`
    (default 0, the first of any) picks which of those occurrences among the tokens the
    scope is, with the selector `frame` and `index` use; the scope is the tokens of the picked ones, in order
    (their union where occurrences of different texts overlap), and `strip` strips each text of surrounding
    whitespace first. The prompt and the generated
    tokens are searched apart, so a text straddling the boundary is not found; and a row's last generated
    token is never an input to any call, so a word ending on it is `alignment_missing` (with
    `max_new_tokens: 1` a scope in the generated tokens never matches). `turn` is a role of the chat, in
    call 0 only: its messages are its occurrences, in order (`{"turn": "user", "occurrence": 1}` is the second
    user message, `"all"` every one of them); a role the row has no message of is refused by name.

    **index** (default `"all"`): an int counts the scope's tokens (negative from the end), a span is
    half-open. It counts over the tokens of every call the frame selected, in order: `{"frame": [0, 3],
    "index": -1}` is one position, the last token of call 2, the same as `{"frame": 2, "index": -1}`.

    A record is rectangular (the same width on every row, `(rows, pos, ...)`) when its frame is one call or
    a span of calls, its index is an int or a span whose ends count from the same side, and a `column`/`text`
    scope, if it has one, is one occurrence (an int) in the prefill; otherwise it is ragged, one entry per
    token, `{values, row, tok}`, `tok` the position. In the prefill (`frame: 0`) an occurrence a row does not
    have, or a rectangular index that does not fit, is refused by name before any forward: the data decided.
    Anywhere else it is a result, as is a call the row never reached: the row is `alignment_missing`, NaN in
    a rectangular read and absent from a ragged one.
    """
    frame: Swept[int | Span | Literal["generated", "all", "eos"]] = 0
    scope: Scope | None = None
    index: Swept[int | Span | Literal["all"]] = "all"

    @model_validator(mode="after")
    def _grammar(self):
        if self.scope is not None and self.scope.turn is not None and self.frame != 0 \
                and not isinstance(self.frame, Sweep):
            raise ValueError(f"scope turn {self.scope.turn!r} is a chat turn of the prompt; it takes frame 0")
        return self


class Op(Record):
    """An op inside a read or a write: `{"op": <name>, ...its arguments}` (see Ops). `featurizer` makes any op
    run on the value's coordinates in that featurizer's space, the change written back; `select`
    (`{"features": [indices]}`) narrows it to some of them. On the `featurizer` op it names the featurizer
    the op calls."""
    model_config = ConfigDict(extra="allow")
    op: str = Field(json_schema_extra={"enum": list(ops.OPS)})
    featurizer: str | None = None
    select: dict[str, Swept[list[int]]] | None = None

    @model_validator(mode="after")
    def _arguments(self):
        ops.check(self.op, arguments(self))
        if self.select is not None and space(self) is None:
            raise ValueError("select on an op narrows a featurizer's space; name the 'featurizer' it runs in")
        for axis, index in (self.select or {}).items():
            if axis != "features":
                raise ValueError(f"a featurizer's space has no axis {axis!r} to select; its axes: features")
            if index == []:
                raise ValueError(f"select {axis}: an empty list selects nothing")
        return self


def arguments(op: "Op | OpStep") -> dict:
    """An op record's arguments by document spelling: its own fields, and `featurizer` when the op takes one
    (the `featurizer` op) rather than running in that featurizer's space."""
    given = dict(op.model_extra or {})
    if isinstance(op, Op) and op.op in ops.OPS and "featurizer" in ops.parameters(op.op):
        given["featurizer"] = op.featurizer
    return given


def space(op: "Op") -> str | None:
    """The featurizer whose space an op in a read or write runs in, if any."""
    return None if op.op in ops.OPS and "featurizer" in ops.parameters(op.op) else op.featurizer


def parameters(plan: "Plan", name: str, args: dict) -> dict[str, inspect.Parameter]:
    """Op `name`'s parameters by document spelling, with those of the function its `options` choose for the
    kind of the featurizer `args` names."""
    ref = ops.featurizer_of(name, args)
    return ops.parameters(name, args, kind_of(plan, ref) if isinstance(ref, str) else None)


def encodes(op: Any) -> bool:
    """Whether an op record calls a featurizer in mode `encode`. (`mode` is the featurizer op's argument and
    `encode` its kinds' mode, so this asks that op by name.)"""
    return op.op == "featurizer" and arguments(op).get("mode") == "encode"


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
    `{"scope": {"turn": "user", "occurrence": 1}}` names the second user message's own tokens.

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
    """Generation from every row of a dataset input. It owns `input`, `writes`, `reads` and `interventions`, as
    a forward does; every other key goes to the engine's `generate` as-is (`max_new_tokens`, `min_new_tokens`,
    `do_sample`, ...), and may sweep. Its reads and writes name forward calls by `pos`'s `frame`: the
    prefill, and the decode calls, each at its one position. A write at decode call k takes effect from
    generated token k on, through the cache. A write lands as its call runs, so beyond the prefill what it
    selects is known by then: without a scope, a call, a span of calls or `generated` with an index counted
    from their start (or -1 of a single call), or the prefill with them and index `all`; with a `column` or
    `text` scope, decode calls only (a call, a span from 1 or `generated`), index -1 and an occurrence
    counted from the first. A scoped write is found online, decoding the inputs fed so far at each call: it
    lands at the call fed the last token of each occurrence it picks, as that occurrence appears (its
    earlier tokens were fed before the text was there to find), and a row with fewer occurrences than it
    picks is `alignment_missing`. No write takes `eos` or a frame counted from a row's last call. A row's
    calls are what the engine actually generated, up to the
    one that chose the row's first EOS: the step's own `eos_token_id` when it gives one, else the id the
    engine's generation stops on (its generation config's). The step's own value is the
    new token ids, `[rows, new]`; `decode` makes them text. Options that would change the rows or the value
    (`num_return_sequences`, `num_beams`, `return_dict_in_generate`, `use_cache`) are refused before anything
    runs; any other option is the engine's `generate`'s to accept, and one it rejects fails when the step
    runs, naming the step.

    Rows run in batches as a forward's do, and reproduce to the bit only at a fixed batch size: the prefill
    projects the head at the last position only and every decode call is a `[batch, 1, ...]` matmul, so
    batch sizes 4 and 2 already differ by about 1e-7 at call 0."""
    model_config = ConfigDict(extra="allow")
    kind: Literal["generate"]  # type: ignore[assignment]


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
    """End the fit after `patience` epochs without improvement (`mode` max or min) of body op `metric`'s mean,
    scored on the training rows after each epoch."""
    metric: str
    mode: Literal["max", "min"]
    patience: Swept[int]


class Fit(Record):
    """Train the featurizers `train` names, inside the run, through the frozen model.

    Each trained featurizer is made at the fit's start (drawn from its own `seed`, else the fit's), and the
    optimizer over its trainable tensors with it. Each epoch sets a gate's temperature, then runs the body
    `steps` over minibatches of `batch_size` rows of its datasets, shuffled by a generator seeded with `seed`
    and the epoch, with gradients on and gates soft, and steps the optimizer once per minibatch on the
    weighted sum `objective` (`{term: weight}`; a term is a body op averaged over every axis, or a trained
    featurizer's scalar, `<featurizer>.<scalar>`). Gradients are taken for the featurizers' tensors alone; the
    model's weights are never touched. After each optimizer step a featurizer with a constraint is projected
    back onto it (an sae's decoder rows to unit norm). After each epoch the body runs over every training row
    in scoring mode (gates hard, no gradients): each body op's mean and each trained featurizer's scalar is
    that epoch's entry of `<fit>.history.<op>` and `<fit>.history.<featurizer>.<scalar>`, and `early_stop`
    is judged on it. The best epoch's featurizer tensors (without `early_stop`, the last's) are restored, and
    the body runs once, scored, per named substitution of `eval`, in listed order: `{"held_out": {"train":
    "test"}, "train": {}}` runs it with dataset `train` read as `test`, then on the data it trained on. The
    body's forwards run in batches of `batch_size`, whatever the run's batch size.

    The body is a scope of its own that sees every value produced before the fit; a trained featurizer's
    name and `<fit>.<featurizer>` both mean the one being trained, and a value with a row per example it uses
    is one it makes. Nothing leaves it except through the fit's members: `<fit>.<featurizer>` the trained
    featurizer (saved as its bundle), `<fit>.<eval>.<op>` and `<fit>.<eval>.<step>.<read>` a body value from
    the run named `<eval>`, and `<fit>.history.<term>`, `[epochs]`, one entry per epoch run. A sweep over any
    of the fit's values (an objective weight, the learning rate) makes the fit, its members and their users a
    list of points."""
    kind: Literal["fit"]
    train: list[str]
    objective: dict[str, Swept[float]]
    epochs: Swept[int]
    batch_size: Swept[int]
    seed: Swept[int]
    optimizer: Optimizer
    steps: "dict[str, Entry]"
    early_stop: EarlyStop | None = None
    eval: dict[str, dict[str, str]] = {}
    """each named run of the body after training: a substitution `{<body dataset>: <dataset>}`, `{}` the data
    it trained on"""
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
    header: dict[str, str] | None = derived(None, "the header of the bundle `file_path` names, which preflight "
                                                  "checks the model's width against")

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
    each dataset input tokenized and what each generate step's rows generated (token strings per row, under
    the input or the step's name), and the position every read and write resolved to, per row (an index into
    the row's prompt tokens, then its generated ones; `alignment_missing` for a row it does not reach). A
    swept step's are keyed by coordinate."""
    engine: str
    remote: bool | str
    batch_size: int
    versions: dict[str, str] = {}
    started: str = ""
    seconds: float = 0.0
    tokens: dict[str, Any] = {}
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
                    classes = featurizers.labels({n: d.rows or [] for n, d in self.data.items()}, fields["classes"]) \
                        if "classes" in fields else None
                    f.bundle, f.header = featurizers.bundle(corpus_root(self), f.kind, fields, self.model, classes)
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


def agreeing(entry: Any, at: dict) -> Any:
    """The point of `entry` that agrees with the point `at` on the entry's own axes."""
    return next(p for p in points(entry) if all(at.get(k) == i for k, i in p.at.items()))


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
    """The axes of the value `name`: a step's value, a step's read, or a fit's member (a trained featurizer,
    saved as its bundle, has none; `history.<op>` is per epoch)."""
    return dims_in(plan.steps, name)


def dims_in(steps: dict, name: str) -> list[str]:
    thing, _, member = name.partition(".")
    step = points(steps[thing])[0]
    if step.kind == "fit":
        inner, _, rest = member.partition(".")
        return [] if inner in step.train else ["epochs"] if inner == "history" else dims_in(step.steps, rest)
    if member:
        return step.reads[member].dims
    return ["rows", "new"] if step.kind == "generate" else step.dims


def walk(steps: dict, prefix: str = ""):
    """Every step and point, a fit's body steps after the fit, each with its dotted name."""
    for name, entry in steps.items():
        for step in points(entry):
            yield prefix + name, step
            if step.kind == "fit":
                yield from walk(step.steps, f"{prefix}{name}.")


def declared(plan: Plan, ref: str) -> str:
    """The declared featurizer a reference names: itself, or the one `<fit>.<featurizer>` trains."""
    return ref if ref in plan.featurizers else ref.partition(".")[2]


def body_view(plan: Plan, fit: str, at: dict | None = None, swap: dict | None = None) -> Plan:
    """The plan as fit `fit`'s body sees it: the body's steps beside the root's, which they shadow (the body
    at the fit's point `at`; without one, a list of each step's points when the fit is swept), and each
    dataset a substitution `swap` maps read under the body's name for it."""
    entry = plan.steps[fit]
    fits = [agreeing(entry, at)] if at is not None else points(entry)
    one = at is not None or not isinstance(entry, list)
    steps = {n: fits[0].steps[n] if one else [p.steps[n] for p in fits] for n in fits[0].steps}
    swap = swap or {}
    return plan.model_copy(update={"steps": {**plan.steps, **steps},
                                   "data": {**plan.data, **{a: plan.data[b] for a, b in swap.items()}}})


def probe_of(plan: Plan, name: str) -> str:
    """The probe whose `encode` made value `name`'s `classes` axis, as referenced where it was encoded."""
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if step.kind == "op":
        return probe_of(plan, step.of)
    if step.kind == "fit":
        return probe_of(body_view(plan, thing), member.partition(".")[2])
    return next(o.featurizer for o in reversed(step.reads[member].ops) if encodes(o) and o.featurizer)


def kind_of(plan: Plan, ref: str) -> str:
    """The kind of the featurizer a reference names."""
    return points(plan.featurizers[declared(plan, ref)])[0].kind


def uses(plan: Plan) -> dict[str, list[tuple[str, Any, dict, "Op"]]]:
    """Every op that runs in a declared featurizer's space or calls it, by featurizer: the record's
    dotted name, its site, its step's point and the op."""
    found: dict[str, list] = {}
    for name, step in walk(plan.steps):
        for key, record in {**getattr(step, "writes", {}), **getattr(step, "reads", {})}.items():
            for o in record.ops:
                if o.featurizer is not None:
                    found.setdefault(declared(plan, o.featurizer), []).append(
                        (f"{name}.{key}", record.site, step.at, o))
    return found


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
