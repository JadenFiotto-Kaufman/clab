"""The plan: one model that is a document as written and, compiled, the document resolved for running.

`Plan(path)` reads a document; `plan.compile()` returns it compiled (rows loaded (a hub dataset's checked, then
left out), sites and named interventions inlined, bundles loaded, sweeps expanded into points, names resolved); an
engine's `execute` fills `plan.results` and `plan.run`; `plan.dump(out)` writes a run's output directory and
`Plan.load(out)` reads one back. The records (sites, interventions and their ops, positions, steps, datasets,
featurizers) are the same models in both states; a *derived* field is one compile or the run fills.

The plan's methods are views and serialization, never execution: it pickles, prints and means the same
thing wherever it lands, which is what lets it travel into a remote session and back.
"""

import json
import math
import os
import types
from collections.abc import Iterator
from typing import Annotated, Any, Literal, NoReturn, TypeVar, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    GetCoreSchemaHandler,
    SerializerFunctionWrapHandler,
    Tag,
    ValidationError,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import ErrorDetails, core_schema
from torch import Tensor

from clab import featurizers, ops
from clab.format import address, data, selectors
from clab.format.selectors import Selector
from clab.format.sweeps import agreeing, find_sweeps, points


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

    @model_serializer(mode="wrap")
    def _derived_last(self, handler: SerializerFunctionWrapHandler) -> Any:
        """The record as written: the document's fields first, then what compile or the run derived."""
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        later = [k for k in type(self).model_fields if k not in documented(type(self))]
        return {**{k: v for k, v in dumped.items() if k not in later}, **{k: v for k, v in dumped.items() if k in later}}


def documented(cls: type[BaseModel]) -> list[str]:
    """The fields a document writes: every field but the derived ones."""
    return [k for k, f in cls.model_fields.items()
            if not (isinstance(f.json_schema_extra, dict) and f.json_schema_extra.get("derived"))]


class Sweep(Record):
    """A value standing for several. Whatever uses it runs once per value, a point each, on the axis its
    `label` names (default: the field it stands for); sweeps with one label are one axis."""
    sweep: list[Any] = Field(min_length=1)
    label: str | None = None

    @field_validator("sweep")
    @classmethod
    def _flat(cls, values: list[Any]) -> list[Any]:
        # A point is keyed by its value, so each value is one a key can hold: a scalar, or a list of them.
        for value in values:
            if isinstance(value, dict):
                field, first = next(iter(value.items()), ("<field>", "..."))
                raise ValueError(f"a sweep's values are numbers, strings, booleans or lists of them, each point "
                                 f"keyed by its value; {value!r} is an object: sweep a scalar field inside it, "
                                 f"{{{field!r}: {{'sweep': [{first!r}, ...]}}}}")
            if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
                raise ValueError(f"a sweep's values are numbers, strings, booleans or lists of them, each point "
                                 f"keyed by its value; {value!r} holds a list or an object")
        return values


class ByShape:
    """A union validated by the one member the input's shape picks (a sweep, a dict, a string, a number, a
    list, null), so a wrong value gets one error, from the member it was meant for."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
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
    if get_origin(member) is dict:
        return isinstance(value, dict) and "sweep" not in value
    if get_origin(member) in (Union, types.UnionType):
        return any(shaped(m, value) for m in get_args(member))
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


# The quantized dtypes (nnsight's names for them) and the dtype each computes in: what its values, and the weights
# it leaves alone (norms, embeddings, the head), are held in.
QUANTIZED = {"nf4": "bfloat16", "fp4": "bfloat16", "int8": "float16", "fp8": "bfloat16"}


class Model(Record):
    """The pinned checkpoint: `key`, `revision` and `dtype`, torch's spelling of a float dtype or one of nnsight's
    names for a quantization: `nf4` and `fp4` hold the weights in bitsandbytes' 4 bits (NormalFloat, FloatPoint) and
    `int8` in its LLM.int8() 8 bits, on the CPU or on CUDA, with bitsandbytes installed; `fp8` holds them in
    transformers' block-wise 8-bit floats, on a GPU of compute capability 8.9 or later. A quantized model computes in
    `bfloat16` (`int8` in `float16`), which is the dtype its values have. Every other key goes to the
    engine's loader untouched (`attn_implementation`, ...), and a loader refuses what it cannot honor; where
    the model runs (`device`) is a run option, not the document's. The attention interior needs
    `attn_implementation: "eager"`."""
    model_config = ConfigDict(extra="allow")
    key: str
    revision: str
    dtype: Literal["float32", "float16", "bfloat16", "nf4", "fp4", "int8", "fp8"] = "float32"

    @property
    def computes(self) -> str:
        """The dtype the model computes in, which its values have: `dtype`, or a quantization's (`QUANTIZED`)."""
        return QUANTIZED.get(self.dtype, self.dtype)


class Data(Record):
    """A dataset: where its rows come from and how its text renders, never a computation on its values. Its rows
    come from one of three sources: `path` is `data/<path>.json` under the corpus root, `rows` are inline, and
    `hf` is a dataset on the Hugging Face hub, `<owner>/<name>` at `revision` (a commit, as `model` pins one; a
    branch is passed as written, and the digest catches what it moves to), in its subset `config` where it has
    several. A file's and inline rows travel in the compiled plan; a hub dataset's never do: compile loads them to
    check the document and records their digest, and the run loads them again where it runs (under `ndif`, the
    server) and refuses other rows before any forward. Then, in this order: `split` picks which split of the
    dataset, the hub dataset's split (left out: its only one) or the rows of a file whose `split` column is that;
    `shuffle` puts the rows in one fixed random order, the same on every machine (seed 0); `take` keeps the first
    N (after `shuffle`, a random sample of N); and `columns` renames them, `{"<name>": "<column as the source has
    it>"}`, each name that column and no longer under its own, every other column as it is (so one file serves a
    fit's `eval` substitution under the body's column names). `template` is passed to the checkpoint's
    `apply_chat_template` wherever a chat column of the dataset renders (`{"add_generation_prompt": true,
    "enable_thinking": false}`), so a chat can end in the generation prompt and keep its `turn` scopes; a
    checkpoint whose template reads the clock (`strftime_now`: Llama-3.2's writes today's date into the system
    header it adds) renders a chat only with its date pinned, `{"date_string": "26 Jul 2024"}`, since the rows'
    digest cannot say that the tokens moved with the day. A row is
    JSON. Refused by name: a second source, an `hf` that is not `<owner>/<name>`, `hf` without `revision`, and
    `revision` or `config` without `hf`; a hub dataset the hub does not give, or offline a commit `revision` other
    than the one its cache holds (offline, a branch reads as the cache holds it), a split it does not have (or, of
    several, none named), and a column of it whose values are not JSON (images, audio, bytes); `take` beyond the
    rows, a `columns` source the rows lack or a name that is a column they keep (a swap, `{"a": "b", "b": "a"}`,
    keeps neither), a `template` on a dataset no step reads a chat column of, and, before any forward, a chat of
    a clock-reading template without `date_string`. `sha256` and `n_rows` describe the rows these leave."""
    path: str | None = None
    rows: list[dict] | None = None
    hf: str | None = None
    revision: str | None = None
    config: str | None = None
    split: str | None = None
    shuffle: bool = False
    take: int | None = None
    columns: dict[str, str] | None = None
    template: dict[str, Any] | None = None
    sha256: str | None = derived(None, "the digest of the rows, which a re-materialization must match")
    n_rows: int | None = derived(None, "how many rows the dataset has")

    @model_validator(mode="after")
    def _one_source(self) -> "Data":
        given = [k for k in ("path", "rows", "hf") if getattr(self, k) is not None]
        # Compiled (compile alone sets `sha256`), a file's rows are beside its path.
        if not given or len(given) > 1 and (self.sha256 is None or "hf" in given):
            raise ValueError("a dataset's rows come from one of 'path', inline 'rows' and a hub dataset 'hf'"
                             + (f"; it gives {' and '.join(map(repr, given))}" if given else ""))
        if self.hf is not None and (len(self.hf.split("/")) != 2 or not all(self.hf.split("/"))):
            raise ValueError(f"hf {self.hf!r}: a hub dataset is named '<owner>/<name>', as 'nyu-mll/glue' is")
        if self.hf is not None and not self.revision:
            raise ValueError("'hf' needs 'revision', the commit of the hub dataset to load, as 'model' pins its "
                             "checkpoint's")
        for key, what in (("revision", "pins"), ("config", "is the subset of")):
            if self.hf is None and getattr(self, key) is not None:
                raise ValueError(f"'{key}' {what} a hub dataset ('hf'); a file's rows and inline ones have none")
        if self.path is None and self.hf is None and self.split is not None:
            raise ValueError("'split' is which split of a file or a hub dataset; inline 'rows' are the rows")
        if self.take is not None and self.take < 1:
            raise ValueError(f"take {self.take}: a dataset keeps at least one row")
        if "tokenize" in (self.template or {}):
            raise ValueError("template: a chat renders to text, which clab tokenizes; it takes no 'tokenize'")
        return self


class Site(Record):
    """A component, narrowed: `layers` (a selector, as `pos` has them, of the model's layers: an int is that
    layer's value; a slice or a list of picks is one value, a `layers` axis first, labelled with the layers
    it picks: `[0, 3, 7]` is three, `"-2:"` the last two) on a block's component, and `select`
    (`{<axis>: <selector>}`, which Components describes). A read and a write take `layers` alike: an entry is
    made at each layer they pick, in one forward."""
    component: str
    layers: Swept[Selector] | None = None
    select: dict[str, Swept[Selector]] | None = None

    @model_validator(mode="after")
    def _narrowing(self) -> "Site":
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
            if axis == "key" and not isinstance(index, Sweep) and not selectors.keyed(index):
                raise ValueError(f"select key {index!r}: the key axis is as long as each row, so a selector there "
                                 "picks a fixed number of keys counted from the start (0, [0, 2], \"0:3\")")
        return self


class Column(Record):
    """A column read as text, where a scope searches for it or a metric scores it: `"<dataset>.<column>"`, or
    `{"column": "<dataset>.<column>", "strip"?: bool, "lower"?: bool}` (an indexed input's variables are named
    whole: `pairs.counterfactual_inputs_variables[0].entity`). A row's value may be a list of strings: a set, any
    of which is the answer. The flags say how its text compares and never rewrite the data: `lower` compares it
    lowercased, and `strip` makes its surrounding whitespace optional.

    A scope searches the input for the text as the flags compare it, under `strip` without its surrounding
    whitespace, its positions the original's.

    An answer scored against logits is tokenized as written, and each of its tokens stands for its class: every
    token of the vocabulary whose text (what it adds after another token, so a leading space is its own) compares
    the same; without a flag, or for a token that is part of a character, the token alone. Under `strip`
    whitespace-only tokens at the end are dropped, since what follows an answer is not scored, and one at the
    start is kept, since a tokenizer that writes a number's space as its own token makes it a token the model
    says (`" 42"` is `▁ 4 2`, its `▁` itself alone); the first token then compares without leading whitespace and
    the last without trailing, a one-token answer's without either (` Yes`, ` yes`, `yes` and `YES` under both
    flags), and one between them keeps its whitespace, a whitespace-only one itself alone (`" New York "` is
    `▁New ▁York`, `▁New` standing for `New` too)."""
    column: str
    strip: bool = False
    lower: bool = False

    @model_serializer(mode="wrap")
    def _spelled(self, handler: SerializerFunctionWrapHandler) -> str | dict[str, Any]:
        """The string alone when no flag is set, as a document writes it."""
        return handler(self) if self.strip or self.lower else self.column


ColumnRef = Union[str, Column]


class Scope(Record):
    """`{"column": <column>}`: the row's own text of that column (a `Column`: the string or the object), each
    member of a set; `{"text": "<literal>"}`: that text; or `{"turn": "<role>"}`: that role's messages of a
    chat, each an occurrence. `column` and `text` may each be a list: the scope's texts are then every one
    listed (a string is the one-element list), their occurrences together, ordered by where each ends. Every
    scope takes `occurrence`, which of its occurrences it is (a selector; default 0, the first): a text's
    occurrence lies on token and word boundaries, as `pos` says (a token's own surrounding whitespace on either
    side, and no letter or digit adjoining a word it cuts); and `before`,
    whether it is the positions that predict its tokens rather than the tokens themselves. A `text` takes
    `strip` and `lower`, as a column's reference does (a turn's contents are located as the template rendered
    them)."""
    # One of column, text and turn; a text's flags are its own, a column's on its reference, a turn's none.
    column: ColumnRef | list[ColumnRef] | None = None
    text: str | list[str] | None = None
    turn: str | None = None
    occurrence: Swept[Selector] = 0
    strip: bool = False
    lower: bool = False
    before: bool = False

    @field_validator("column")
    @classmethod
    def _columns(cls, given: ColumnRef | list[ColumnRef] | None) -> Column | list[Column] | None:
        """Each column a `Column`, a string the one without flags."""
        def made(ref: ColumnRef) -> Column:
            return Column(column=ref) if isinstance(ref, str) else ref
        return [made(c) for c in given] if isinstance(given, list) else None if given is None else made(given)

    @model_validator(mode="after")
    def _one(self) -> "Scope":
        if [self.column, self.text, self.turn].count(None) != 2:
            raise ValueError("a scope is one of {'column': '<dataset>.<column>'}, {'text': '<literal>'}, "
                             "{'turn': '<role>'}")
        if [] in (self.column, self.text):
            raise ValueError("a scope's list of columns or texts names at least one")
        for flag in ("strip", "lower"):
            if self.turn is not None and flag in self.model_fields_set:
                raise ValueError(f"a turn's messages are located as the chat template rendered them; it takes no "
                                 f"{flag}")
            if self.column is not None and flag in self.model_fields_set:
                raise ValueError(f"a column's {flag} goes on its reference: {{'column': {{'column': "
                                 f"'<dataset>.<column>', {flag!r}: true}}}}")
        if self.turn is not None and not self.turn.isidentifier():
            raise ValueError(f"turn {self.turn!r} is a role; its messages are its occurrences, so "
                             "`{'turn': 'user', 'occurrence': 1}` is the second user message")
        return self


class Pos(Record):
    """`pos` is three stages of narrowing: `frame` picks forward calls, `scope` picks text within their tokens,
    `index` picks positions within that.

    **A row's positions are its input tokens, all of them, in order from 0: BOS where the tokenizer adds one,
    a chat template's markers, everything but padding, an appended EOS included** (`pos: 0` is BOS where there
    is one, `-1` the last token, `{"index": ":"}` every token). Every count of a row's tokens is this one: a
    record's `positions`, `clab table`'s `pos` and `key`, an attention pattern's `key` axis, and a
    refusal's token counts.

    ```
    pos      := int                                # shorthand: {"frame": 0, "index": int}
              | {"frame"?: frame, "scope"?: scope, "index"?: selector}     # index default ":"
    frame    := selector                                                  # default 0
    scope    := {"column": column or a list, "occurrence"?: selector, "before"?: bool}
              | {"text": one or a list, "occurrence"?: selector, "strip"?: bool, "lower"?: bool, "before"?: bool}
              | {"turn": "<role>", "occurrence"?: selector, "before"?: bool}   # occurrence default 0
    column   := "<dataset>.<column>" | {"column": "<dataset>.<column>", "strip"?: bool, "lower"?: bool}
    selector := int | "<start>:<stop>:<step>" | [int, ...]                # one, a slice, picks
    ```

    Each stage is a selector over what the stage before it left (`selectors.select`): an int counts one
    (negative from the last), a slice string is a range in Python notation, half-open, each part optional
    (`":"` every one, `"-4:-1"`, `"2:"`, `"::2"`), a list picks those, in the order written. A selector
    that picks nothing however many things there are (`"0:0"`, `"5:2"`) is refused, and so is an index that
    picks none of the tokens of a frame of a static number of decode calls (`{"frame": "1:3", "index":
    "5:"}`).

    **frame**: 0 is the prefill, the only call a `forward` step has; k >= 1 is decode call k; a negative k
    counts from the row's last call (`-1` is the call that chose its last token, whether that was EOS or the
    generation ran to `max_new_tokens`); `"1:"` is every decode call, `":"` the prefill and every decode
    call (in a `forward`, its one call: so a named intervention written at `":"` is on in every call of
    whichever step names it). A
    position is where an input token sits, in a forward and in a generation alike: the prefill holds every
    prompt token, its last position the distribution over generated token 0; under the cache decode call k
    holds one position, generated token k-1 (its input), and its logits are the distribution over token k. A
    row's calls end with the one that chose its last token, so the token it stopped on is never an input. A
    generation's prefill computes the head (`lm_head_output`, `logits`) at its last position only, so there
    call 0 holds that one position.

    **scope** (default: every token the frame selected): `column` and `text` locate text in the selected
    calls' tokens, in the prompt by the tokenizer's offsets and in the generated tokens by decoding growing
    prefixes: the tokens covering its characters, a chat template's marker among them where the text spans one
    (a BOS the tokenizer adds covers none; one a template writes covers its text, `<s>`). An occurrence lies on
    token and word boundaries: it starts where a token starts and ends where a token ends (a token's own leading
    or trailing whitespace on either side of the boundary; one that ends in the next token's leading whitespace
    ends where that token starts, so `Paris. ` does not cover the ` Paris` after it), and where its first or
    last character is a letter or a digit, none adjoins it: ` Sara` does not occur in ` Sarah`, ` Ann` not in
    ` Annabel`, `:` not in `):` (a write found online sees nothing past the token it is fed, so it takes an end
    there as a word's, whatever follows). A list of columns or texts locates every occurrence of each.
    Occurrences are ordered by where they end, then where they start: where an occurrence becomes known and
    where index -1 lands, so a write found online, a read and the run record count them alike (`b` inside `a b
    c` comes before it). `occurrence` (default 0, the first of any) picks which of those occurrences among the
    tokens the scope is; the scope is the tokens of the picked ones, in order
    (their union where occurrences of different texts overlap); `strip` and `lower` (a text's, or on a
    column's reference) compare each text stripped of surrounding whitespace and lowercased, the positions
    the original's. The prompt and the generated
    tokens are searched apart, so a text straddling the boundary is not found; and a row's last generated
    token is never an input to any call, so a word ending on it is `alignment_missing` (with
    `max_new_tokens: 1` a scope in the generated tokens never matches), unless the scope is `before`.
    `turn` is a role of the chat, in call 0 only: its messages are its occurrences, in order (`{"turn":
    "user", "occurrence": 1}` is the second user message, `":"` every one of them); a role the row has no
    message of is refused by name.

    **before** (default false): the positions that predict the scope's tokens, each the position just
    before one, the same width. The scope is searched among the tokens the frame's calls hold and, where
    the frame names the row's last call, the token that call chose (the row's last generated token, which
    no call holds); the record is the position before each, which may belong to a call the frame does not
    name: under `"frame": "1:"` a word said as generated token 0 is held by decode call 1 and recorded at
    the prefill's last position, call 0, which chose it. In the prefill that is the position before each
    token; in a generation, the call that chose each generated token, the last one included. A scope token
    at position 0 has no position before it, which makes the row `alignment_missing` (a refusal in the
    prefill). A write beyond the prefill lands as its call runs, before the token it would predict is
    chosen, so it takes no `before`.

    **index** (default `":"`): counts the scope's tokens, over the tokens of every call the frame selected,
    in order: `{"frame": "0:3", "index": -1}` is one position, the last token of call 2, the same as
    `{"frame": 2, "index": -1}`.

    A record has a static width (the same number of positions on every row, `(rows, pos, ...)`) when its frame
    names a static number of calls (an int, picks, or a slice with a static width, not `":"` or `"1:"`), its
    index has a static width (an int, picks, or a slice whose two ends count from the same side: `"-4:-1"`,
    `"-3:"`, `"2:5"`; see `selectors.fixed`), and a `column`/`text` scope, if it has one, is one occurrence in
    the prefill; otherwise its rows hold different numbers of positions, and the value is as wide as the widest
    row, NaN past a row's own. In the prefill (`frame: 0`) an occurrence a row does not have, or a static index
    that does not fit, is refused by name before any forward: the data decided. Anywhere else it is a result, as
    is a call the row never reached: the row is `alignment_missing`, NaN all the way.
    """
    frame: Swept[Selector] = 0
    scope: Scope | None = None
    index: Swept[Selector] = ":"

    @model_validator(mode="after")
    def _grammar(self) -> "Pos":
        frame, index = self.frame, self.index
        if self.scope is not None and self.scope.turn is not None and frame != 0 and not isinstance(frame, Sweep):
            raise ValueError(f"scope turn {self.scope.turn!r} is a chat turn of the prompt; it takes frame 0")
        if self.scope is not None or isinstance(frame, Sweep) or isinstance(index, Sweep) or selectors.from_end(frame):
            return self
        calls = selectors.fixed(frame)
        if calls is not None and 0 not in selectors.select(frame, 1) and not selectors.select(index, calls):
            raise ValueError(f"index {index!r} picks none of the {calls} tokens of frame {frame!r}'s decode "
                             "calls, one each")
        return self


class Op(Record):
    """An op inside an entry: `{"op": <name>, ...its arguments}` (see Ops)."""
    model_config = ConfigDict(extra="allow")
    op: str
    dims: list[str] | None = derived(None, "the layout of the op's value, at each layer the entry is made at (the "
                                           "op after it meets it; the first meets the site's selection)")
    space: list[dict] = derived([], "the coordinates the op's value is in: the encodes open after it, outermost "
                                    "first, each its `featurizer` and `select`; none in the site's own layout")

    @model_validator(mode="after")
    def _arguments(self) -> "Op":
        ops.check(self.op, arguments(self))
        return self


def arguments(op: "Op | OpStep") -> dict[str, Any]:
    """An op record's arguments by document spelling."""
    return dict(op.model_extra or {})


class Labelled(Record):
    """A value's labelled layout, derived: the fields the record of every value carries (an entry's, an op step's, a
    generation's), which compile writes as far as the document decides them and preflight completes with the model
    in hand, so that every reader of a value's axes, labels, rows or positions reads its record and nothing else."""
    dims: list[str] | None = derived(None, "the layout of the value: its axes, in order")
    dataset: str | None = derived(None, DATASET)
    layers: list[int] | None = derived(None, "the model layers the value is from, in the order its site picks them: "
                                             "the labels of its `layers` axis where it has one, the one layer it was "
                                             "read at where not; null for a value from no layer (a model's own, or "
                                             "one reduced over its layers); `[]` until preflight resolves a slice or "
                                             "a layer counted from the last")
    read: str | None = derived(None, "the entry whose value this is, or descends from through op steps, whose "
                                     "positions it holds where it has a `pos` axis; null for a value no read made")
    width: int | None = derived(None, "how many positions the value holds on every row, where it has a `pos` axis "
                                      "and that is one number: the static width its read's `pos` names, confirmed "
                                      "from the positions preflight resolves; null where rows hold different numbers "
                                      "of positions, or there is no `pos` axis")
    kept: dict[str, list[int]] = derived({}, "the indices a site's `select` keeps on each axis it selects, in their "
                                             "order (`key`: the keys it picks), an op step's those of the value it is "
                                             "over on the axes its own still has; an unselected axis is whole, and "
                                             "`[]` stands until preflight resolves a slice or a negative pick")
    space: list[dict] = derived([], "the coordinates the value is in: the encodes left open around it, outermost "
                                    "first, each its `featurizer` and `select`; none for the site's own layout")


# What an entry's value spelled `reads` and `writes` is now, said where a document still spells them.
SIDES = ("reads", "writes")
SIDED = ("a step's reads and writes are one 'interventions' value, and a bundle is one dict of entries, "
         "{<name>: {'site', 'pos'?, 'ops'?}}: an entry whose ops include an edit writes, one without reads")


class Intervention(Labelled):
    """An entry of an `interventions` value: the `site` it lands on (a declared name, or one inline), the
    positions `pos` names there (default every token of the prompt, `{"frame": 0, "index": ":"}`), and the `ops` it
    runs on that selection, in the order written.

    ```
    step.interventions := item | [item, ...]
    item               := "<bundle name>" | {<name>: entry, ...}
    entry              := {"site": site, "pos"?: pos, "ops"?: [op, ...]}
    root.interventions := {<bundle>: {<name>: entry, ...}}
    ```

    An entry whose ops include an edit (the edits group: <the edits group>) writes back: its value after its
    last edit replaces the live value at its selection, so later entries and the rest of the forward see it;
    one with none leaves the model alone. Every entry binds `<step>.<name>` to its value after all its ops, a
    write's included, so what a write wrote is readable. An edit comes before any op that changes the value's
    layout (`readout`, a metric, a reduction, an `encode` no `decode` closes): after one there is no live value
    to write. An entry's value leaves the model only when something uses it (`saves`, or a later entry or step
    names it); an entry nothing names only writes, and costs no memory.

    An entry is made at each layer its site's `layers` pick, in one forward: a write at `layers: "2:6"` runs its
    ops at each of layers 2 to 5 on that layer's live value, in the forward's order, exactly as four entries
    of one layer each (the same ops, in order) would. Every op of the entry runs at each layer, the ops after its
    last edit too, and the value it binds stacks what they made there, a `layers` axis first, labelled with the
    layers it picks, as a read there does (`[add, readout, token_prob]` at `layers: ":"` is a probability per
    layer); so no op inside an entry reduces over `layers`, which a step after the forward does. An operand
    meets those layers as Ops says.

    A step's entries run in the order written: the items of a list in turn, a dict's entries in turn, a bundle's
    where its name stands. Entries at different sites act as the forward reaches their sites; two at one site
    compose there in order, each on what the one before it left. A name binds once per step: two entries of one
    name are refused. An operand may name an earlier entry of the same forward (`patched.premix` in `patched`),
    at this entry's site or at one the forward reaches before it, at each layer this entry is made at: it is that
    entry's value in this batch, at its positions, and so an entry whose ops all write (none after its last
    edit), in the coordinates the operand meets (`space`). An operand read on another input at the write's own
    positions, with no scope, pairs its i-th position with the write's i-th, which on rows whose prompts differ
    in length is checked conservatively before any forward: each pair sits at one distance from the start within
    the tokens both prompts begin with alike, or from the end within those they end with alike (the first token
    past either counted in); any other is refused, and a scope names the token in each row. What an operand's
    layout and labels must be is Ops' one rule. A
    bundle at the root is a dict of entries pulled into a step by name, a template whose names resolve in the
    step that pulls it in: an operand naming one of the bundle's own entries bare (`"with": "premix"`) means that
    entry of whichever step pulls it in, and `<a step>.<one of its entries>` is refused, since pulled into
    another step it would still mean that step's. An operand from an entry of its own step is served in a
    forward; inside a generate it is not yet (a cross-site adapter in a generation is on the roadmap)."""
    site: Annotated[str | Site, ByShape()]
    pos: Swept[int | Pos] = Pos()
    ops: list[Op] = []
    gathered: bool = derived(False, "whether the run takes the entry's value off the model: `saves` names it, "
                                    "or a later entry or step uses it")
    positions: dict[str, list] = derived({}, "the positions the entry resolved to, per row (as `pos` counts them: an "
                                             "index into the row's prompt tokens, BOS first, then its generated ones; "
                                             "`alignment_missing` for a row it does not reach), under the dataset whose "
                                             "rows they are (a fit body's entry has them under each dataset an eval "
                                             "reads it over); the prefill's set by preflight, a generation's decode "
                                             "calls' by the run")


# The edits group as `ops` declares it, written into the docstring where `<the edits group>` stands, so a new edit is listed.
Intervention.__doc__ = (Intervention.__doc__ or "").replace("<the edits group>", ops.listed("edits"))

Entries = dict[str, Intervention]
Interventions = Annotated[Union[str, Entries, list[Annotated[Union[str, Entries], ByShape()]]], ByShape()]


def entries(step: "Entry") -> dict[str, Any]:
    """A compiled forward's or generation's entries (`Intervention`s, each site inlined) in execution order:
    compile lays every `interventions` value out as one dict. Empty for another step."""
    found = getattr(step, "interventions", {})
    assert isinstance(found, dict), "a compiled step's interventions are one dict of entries"
    return found


def edits(o: Op) -> bool:
    """Whether an op is an edit, which inside an entry writes back."""
    return ops.DECLARED[o.op]["group"] == "edits"


def writes(entry: Intervention) -> bool:
    """Whether an entry writes back: it holds an edit."""
    return any(edits(o) for o in entry.ops)


def gathers(entry: Intervention) -> list[Op]:
    """The ops of an entry that run on its value once it is taken off the model: those after its last edit."""
    last = max((i for i, o in enumerate(entry.ops) if edits(o)), default=-1)
    return entry.ops[last + 1:]


class TextPart(Record):
    """`{"text": "<literal>"}`: literal text in a list `input`."""
    text: str


class Forward(Record):
    """One forward over every row of a dataset input. `input` is `<dataset>.<column>` or
    `<dataset>.<column>[i]`, a text or a chat, or a list of text columns and `{"text": "<literal>"}`
    literals of one dataset, joined per row as they are, with nothing between (`["days.input",
    "days.answer"]` appends each row's answer to its prompt: teacher forcing); `interventions` (a bundle's
    name, a dict of entries, or a list of either) are what it reads and writes, each entry binding
    `<step>.<name>`, in the order written.

    A chat input (a list of `{"role", "content"}` messages) is rendered through the checkpoint's own chat
    template, given its dataset's `template` as arguments. Without one there is no generation prompt, so the
    rendered prompt ends with the template's close of the last message (`[/INST]` on Llama-2-style templates,
    `<|im_end|>` and a newline on Qwen3's): `pos: -1` with no scope is that close's last token, a marker or not.
    With `{"add_generation_prompt": true}` it ends with the assistant's header, whose tokens are positions like
    any other and belong to no `turn`, and a generate from it continues after the header. Either way
    `{"scope": {"turn": "user", "occurrence": 1}}` names the second user message's own tokens.

    Rows run in consecutive batches of the run's batch size, each padded on the left to the input's longest
    row, with position ids that count real tokens. At the thread count the suite pins (`tests/conftest.py`, 12;
    8 and 24 give the same bits) a row's numbers do not depend on which rows share its batch; at 4 or fewer they
    can differ by about 1e-8. They do depend on the batch's shape: a trailing partial batch is a smaller matrix,
    and can differ in the last bits. Results reproduce to the bit at a fixed batch size and thread count."""
    kind: Literal["forward"]
    input: str | list[str | TextPart]
    interventions: Interventions = {}
    at: dict[str, int] = derived({}, AT)
    dataset: str | None = derived(None, DATASET)

    @model_validator(mode="before")
    @classmethod
    def _one_value(cls, given: Any) -> Any:
        for key in SIDES if isinstance(given, dict) else ():
            if key in given:
                raise ValueError(f"{key!r}: {SIDED}")
        return given

    @field_validator("input")
    @classmethod
    def _input(cls, given: str | list[str | TextPart]) -> str | list[str | TextPart]:
        if isinstance(given, list) and not any(isinstance(part, str) for part in given):
            raise ValueError("a list input names at least one column, whose dataset its rows are")
        for column in data.input_columns(given):
            data.parse_input(column)
        return given


# The generate options a step cannot forward as they are, the one value each may have, and why.
FIXED = {
    "num_return_sequences": (1, "more sequences than rows would break the step's one row of ids per input row"),
    "num_beams": (1, "beams change the rows each forward call sees, so a read's row is no longer the input's"),
    "return_dict_in_generate": (False, "the step's value is the ids"),
    "use_cache": (True, "without the cache a decode step recomputes the prompt, and the prompt-call writes "
                        "would not reach it"),
}


class Generate(Forward, Labelled):
    """Generation from every row of a dataset input. It owns `input` and `interventions`, as a forward does;
    every other key goes to the engine's `generate` as-is (`max_new_tokens`, `min_new_tokens`, `do_sample`,
    ...), and may sweep. Its entries name forward calls by `pos`'s `frame`: the
    prefill, and the decode calls, each at its one position. A write at decode call k takes effect from
    generated token k on, through the cache. A write lands as its call runs, so beyond the prefill what it
    selects is known by then: its frame not counted from the row's last call, and with an
    index that names the same calls on a row that stops early as on one that runs on (not one counted from
    the end over several calls, nor any but `":"` over the prompt); a frame to the row's last call other than
    every decode call (`"2:"`, `"1::2"`) lands at the calls up to the step's `max_new_tokens`, which it then
    gives. With a `column` or `text` scope, decode calls only, index -1 and an occurrence counted from the
    first. A scoped write is found
    online, decoding the inputs fed so far at each call: it lands at the call fed the last token of each
    occurrence it picks, as that occurrence appears (its earlier tokens were fed before the text was there
    to find), and a row with fewer occurrences than it picks is `alignment_missing`. No write takes a frame
    counted from a row's last call. A row's calls are what the engine actually generated, up to the
    one that chose the row's first EOS: the step's own `eos_token_id` when it gives one, else the id the
    engine's generation stops on (its generation config's). The step's own value is the
    new token ids, `[rows, new]`; `detokenize` makes them text. Options that would change the rows or the value
    (`num_return_sequences`, `num_beams`, `return_dict_in_generate`, `use_cache`) are refused before anything
    runs; any other option is the engine's `generate`'s to accept, and one it rejects fails when the step
    runs, naming the step.

    Rows run in batches as a forward's do, and reproduce to the bit only at a fixed batch size: the prefill
    projects the head at the last position only and every decode call is a `[batch, 1, ...]` matmul, so
    batch sizes 4 and 2 already differ by about 1e-7 at call 0."""
    # Its other keys go to the engine, but `reads` and `writes` are refused, as on a forward (`SIDES`).
    model_config = ConfigDict(extra="allow")
    kind: Literal["generate"]  # type: ignore[assignment]
    dims: list[str] | None = derived(["rows", "new"], "the layout of the step's own value, its new token ids")

    @model_validator(mode="after")
    def _fixed(self) -> "Generate":
        """An option in `FIXED` at another value is refused; every other option is the engine's `generate`'s to
        accept or reject when the step runs."""
        for key, value in (self.model_extra or {}).items():
            if key in FIXED and value != FIXED[key][0]:
                raise ValueError(f"{key} {value!r}; it stays {FIXED[key][0]!r}: {FIXED[key][1]}")
        return self


class OpStep(Labelled):
    """An op over a value earlier steps produced: `of` names it, and the op's arguments sit beside it (see
    Ops). The step's value is the op's result; a step holds exactly one op, and chaining is the next step."""
    model_config = ConfigDict(extra="allow")
    kind: Literal["op"]
    op: str
    of: str
    at: dict[str, int] = derived({}, AT)

    @model_validator(mode="after")
    def _arguments(self) -> "OpStep":
        ops.check(self.op, arguments(self))
        return self


class Optimizer(Record):
    """The fit's optimizer, over the trained featurizers' parameters only."""
    name: Literal["adamw", "adam", "sgd"]
    lr: Swept[float]
    weight_decay: Swept[float] = 0.0

    @field_validator("name", mode="before")
    @classmethod
    def _named(cls, given: Any) -> Any:
        if isinstance(given, dict) and "sweep" in given:
            raise ValueError("a sweep cannot stand on a name, and the optimizer's name is one: a fit makes one "
                             "optimizer; write one fit per optimizer")
        return given


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
    `steps` over minibatches of `batch_size` rows of its datasets, shuffled by a generator seeded with `seed` and
    the epoch (a minibatch is its rows' ids in their datasets, so `noise` draws a row's noise in any minibatch
    as it does outside the fit), with gradients on and gates soft, and steps the optimizer once per minibatch on
    the weighted sum `objective` (`{term: weight}`; a term is a body op averaged over every axis, or a trained
    featurizer's scalar, `<featurizer>.<scalar>`). On a model not in float32 a metric term is a float32 loss, as
    every metric is (see Ops), so the fit's bundle differs from one trained when metrics computed in the model's
    dtype. Gradients are taken for the featurizers' tensors alone; the model's weights are never touched. After each
    optimizer step a featurizer with a constraint is projected back onto it (an sae's decoder rows to unit norm).
    After each epoch the body runs over every training row in scoring mode (gates hard, no gradients): each body
    op's mean and each trained featurizer's scalar is that epoch's entry of `<fit>.history.<op>` and
    `<fit>.history.<featurizer>.<scalar>`, and `early_stop` is judged on it. The best epoch's featurizer tensors
    (without `early_stop`, the last's) are restored, and the body runs once, scored, per named substitution of
    `eval`, in listed order: `{"held_out": {"train": "test"}, "train": {}}` runs it with dataset `train` read as
    `test`, then on the data it trained on (a dataset read as another renders its chats with its own `template`,
    so the two have the same one). The body's forwards run in batches of `batch_size`, whatever the run's batch
    size. `epochs: 0` trains nothing: the featurizers are scored as drawn (an untrained lora is the identity
    exactly, a null control), and each history is empty, `[0]`.

    The body is a scope of its own that sees every value produced before the fit; a trained featurizer's name and
    `<fit>.<featurizer>` both mean the one being trained. A value made before the fit with a row per example of
    one of the body's datasets (a harvest) comes into every run of the body at that run's rows, taken by their
    ids as the body's datasets are, and as it was made: on its device, in its dtype, with no gradient of its own;
    so the model runs while the fit trains only where the body has a forward. The body's datasets are those its
    forwards read (with none, the one its harvests are of): a harvest of another is refused, since a minibatch's
    ids would pair its rows with theirs. A reduction over rows inside the body is the minibatch's statistic; a
    statistic of the dataset is a step before the fit. An eval that reads a harvest's dataset as another is
    refused, since no step made the harvest over the other's rows: evaluate on other data in steps after the fit,
    through `<fit>.<featurizer>`. So is an objective none of whose terms depends on what the fit trains, which has
    no gradient to step on. Nothing leaves the body except through the fit's members: `<fit>.<featurizer>` the
    trained featurizer (saved as its bundle), `<fit>.<eval>.<op>` and `<fit>.<eval>.<step>.<entry>` a body value
    from the run named `<eval>`, and `<fit>.history.<term>`, `[epochs]`, one entry per epoch run. A sweep over any
    of the fit's values (an objective weight, the learning rate) makes the fit, its members and their users a list
    of points."""
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

    @model_validator(mode="after")
    def _counts(self) -> "Fit":
        if isinstance(self.epochs, int) and self.epochs < 0:
            raise ValueError(f"epochs {self.epochs}: a fit runs 0 or more epochs (0 scores the untrained draw)")
        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError(f"batch_size {self.batch_size}: a minibatch holds at least one row")
        return self


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
    """A space an entry's ops encode into, or a function `apply` calls: `{"kind": <kind>, ...its fields}` (see
    Featurizer kinds)."""
    model_config = ConfigDict(extra="allow")
    kind: str
    at: dict[str, int] = derived({}, AT)
    bundle: dict[str, Tensor] | None = derived(
        None, "the tensors of the bundle `file_path` names, checked against its header", exclude=True)
    header: dict[str, str] | None = derived(None, "the header of the bundle `file_path` names, which preflight "
                                                  "checks the model's width against")
    width: int | None = derived(None, "the feature width the featurizer has at the sites it is used at, one for all "
                                      "of them, which preflight finds among the model's sizes (null for one used "
                                      "nowhere); a saved bundle's header records it")

    @model_validator(mode="after")
    def _fields(self) -> "Featurizer":
        featurizers.check(self.kind, self.model_extra or {})
        return self


FeaturizerEntry = Annotated[Union[Annotated[Featurizer, Tag("~one")], Annotated[list[Featurizer], Tag("~points")]],
                            Discriminator(_points)]


class Axis(Record):
    """A sweep axis: the values its sweeps stand for, in order."""
    values: list[Any]


class Run(Record):
    """How a plan ran, set by `execute`: the run options (`engine`, `batch_size`, `device`, None where the
    engine does not place the model, as on an NDIF server, and every engine option after its defaults in
    `options`), what served it (`versions`: clab's version with its commit, the packages it runs on and python,
    where the model is: the server's, under `ndif`). `dtype` and `devices` are the
    distinct dtypes and devices of the model's parameters there, in the order first met (a quantized model's
    packed weights among them, `uint8` under `nf4`; every rank's devices under `tp`). `sizes` are the model's
    (`num_layers`, `hidden_size`, `num_heads`, ...), the axes a site's selectors count among: what a saved
    bundle's header records of an axis a site keeps whole. `tokens` describes values without holding any: how
    each dataset input tokenized and what each
    generate step's rows generated (token strings per row, under the input or the step's name; a swept step's
    keyed by coordinate). Where every read and write resolved is on its own record (`positions`), as a value's
    layout and labels are. `costs` holds what each step cost, by
    name (a swept step's keyed by coordinate): its wall time in `seconds`, the peak `memory` allocated during it on
    each of the model's devices, weights included, in bytes by device (null where the model is not on CUDA or the
    run is remote), the model calls it made in `forwards` (each forward, each call of a generation, its prefill
    and every decode call, and every forward a fit's body makes) and the input `tokens` they fed, padding excluded
    (a generation's rows each fed their generated tokens but the last, their EOS or the generation's final one)."""
    engine: str
    batch_size: int
    device: str | None
    options: dict[str, Any] = {}
    versions: dict[str, str] = {}
    dtype: list[str] = []
    devices: list[str] = []
    sizes: dict[str, int] = {}
    started: str = ""
    seconds: float = 0.0
    tokens: dict[str, Any] = {}
    costs: dict[str, Any] = {}


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
    interventions: dict[str, Entries] = {}
    """named bundles of entries, pulled into a step's `interventions` by name"""
    featurizers: dict[str, FeaturizerEntry] = {}
    """named spaces an entry encodes into, and functions it applies"""
    steps: dict[str, Entry]
    """what runs, in order"""
    saves: list[str] = []
    """the values that come home, by name"""
    document: str | None = derived(None, "the document file the plan was read from; written relative to its "
                                         "corpus root, where data paths and bundles resolve")
    compiled: bool = derived(False, "whether this is the compiled plan")
    sweeps: dict[str, Axis] = derived({}, "each sweep axis by label, in axis order, with its values")
    results: dict[str, Any] = derived(
        {}, "each saved value by name; a swept one is a dict keyed by coordinate", exclude=True)
    run: Run | None = derived(None, "how the plan ran, set by `execute`")

    def __init__(self, path: str | None = None, /, **fields: Any) -> None:
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
            # find_sweeps also refuses a sweep where none may stand, so every document read runs it.
            if find_sweeps(given) and given.get("compiled"):
                raise ValueError("a compiled plan holds no sweep; its sweeps are expanded into points")
            for bundle, listed in given.get("interventions", {}).items():
                for key in SIDES if isinstance(listed, dict) else ():
                    if key in listed:
                        raise ValueError(f"interventions.{bundle}.{key}: an entry named {key!r}: {SIDED}")
            for name in given.get("steps", {}):
                if "." in name:
                    raise ValueError(f"steps.{name}: a name has no dots; a dot separates a thing from its member")
                if name in given.get("data", {}) or name in given.get("featurizers", {}):
                    raise ValueError(f"steps.{name}: {name!r} is also a dataset or a featurizer")
        return given

    @model_validator(mode="after")
    def _bundles(self) -> "Plan":
        # A bundle's file is its source of truth, and tensors never go in a document: a compiled plan read back
        # without its bundles loads them again from their `file_path`. A manifest holds no rows a dataset read from
        # a file, so a probe's classes are then its bundle's header's, which the run checked.
        for entry in self.featurizers.values() if self.compiled else []:
            for f in points(entry):
                fields = f.model_extra or {}
                if fields.get("file_path") is not None and f.bundle is None:
                    held = "classes" in fields and self.data[fields["classes"].partition(".")[0]].rows is not None
                    classes = featurizers.labels({n: d.rows or [] for n, d in self.data.items()}, fields["classes"]) \
                        if held else None
                    f.bundle, f.header = featurizers.bundles.load(corpus_root(self), f.kind, fields, self.model,
                                                                  classes)
        return self

    @field_serializer("document", when_used="json")
    def _relative(self, document: str | None) -> str | None:
        return document and os.path.relpath(document, corpus_root(self))

    def __repr__(self) -> str:
        if self.run is not None:
            return f"Plan({name(self)}, run on {self.run.engine}, {len(self.results)} results)"
        if self.compiled:
            n = math.prod(len(axis.values) for axis in self.sweeps.values())
            return f"Plan({name(self)}, compiled, {n} point{'s' * (n != 1)})"
        return f"Plan({name(self)}, uncompiled)"

    def compile(self) -> "Plan":
        """This plan compiled: rows loaded (a hub dataset's checked, then left out), sites and named interventions
        inlined, bundles loaded, sweeps expanded into points, every name resolved and every value's layout and
        dataset known."""
        from clab.compile import compile
        return compile(self)

    def dump(self, out: str) -> None:
        """A run's output directory: one file per save and `manifest.json`."""
        from clab.output import dump
        dump(self, out)

    @classmethod
    def load(cls, out: str) -> "Plan":
        """The filled plan an output directory holds: the inverse of `dump`."""
        from clab.output import load
        return load(out)

    def write(self, path: str) -> None:
        """The plan as a JSON document, in its current state, without its results and run."""
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json", exclude_defaults=True, exclude={"run", "document"}), f, indent=1)
            f.write("\n")

    def table(self, *names: str, columns: list[str] | None = None) -> list[dict]:
        """The saved values `names` as rows, joined wide on their shared axes (see `table.table`)."""
        from clab.output.table import table
        return table(self, names, columns)


def corpus_root(plan: Plan) -> str:
    """Where data paths and bundles resolve: for the document `Plan(path)` read, the nearest directory above it
    that holds a `data/` folder (so `documents/real/x.json` shares `data/` with `documents/x.json`), else the
    parent of its own directory; the working directory for a plan built from fields or loaded from a run (whose
    manifest records its document relative to the corpus root)."""
    if not (plan.document and os.path.isabs(plan.document)):
        return os.getcwd()
    here = os.path.dirname(plan.document)
    while not os.path.isdir(os.path.join(here, "data")) and os.path.dirname(here) != here:
        here = os.path.dirname(here)
    return here if os.path.isdir(os.path.join(here, "data")) else os.path.dirname(os.path.dirname(plan.document))


def name(plan: Plan) -> str:
    """The plan's name: its document's file name, without the extension."""
    return os.path.splitext(os.path.basename(plan.document))[0] if plan.document else "plan"


def message(error: ErrorDetails) -> str:
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


def produced(steps: dict[str, Entry], declared: dict[str, FeaturizerEntry]) -> list[str]:
    """Every value `steps` produce, by name, in order: an op step's or a generation's own, each entry's
    `<step>.<entry>`, and a fit's members (`<fit>.history.<term>` and its eval members, `<fit>.<eval>.<member>`);
    not a trained featurizer, which is no value. `declared` are the plan's featurizers, whose kinds say the scalars
    a fit's history holds."""
    out: list[str] = []
    for name, entry in steps.items():
        step = points(entry)[0]
        if step.kind != "fit":
            out += [name] if step.kind != "forward" else []
            out += [f"{name}.{k}" for k in entries(step)]
            continue
        out += [f"{name}.history.{n}" for n, s in step.steps.items() if s.kind == "op"]
        out += [f"{name}.history.{f}.{scalar}" for f in step.train
                for scalar in featurizers.KINDS[points(declared[f])[0].kind].scalars]
        out += [f"{name}.{label}.{member}" for label in step.eval for member in produced(step.steps, declared)]
    return out


def walk(steps: dict[str, Entry], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every step and point, a fit's body steps after the fit, each with its dotted name."""
    for name, entry in steps.items():
        for step in points(entry):
            yield prefix + name, step
            if step.kind == "fit":
                yield from walk(step.steps, f"{prefix}{name}.")


def declared(plan: Plan, ref: str) -> str:
    """The declared featurizer a reference names: itself, or the one `<fit>.<featurizer>` trains."""
    return ref if ref in plan.featurizers else ref.partition(".")[2]


def kind_of(plan: Plan, ref: str) -> str:
    """The kind of the featurizer a reference names."""
    return points(plan.featurizers[declared(plan, ref)])[0].kind


def entry_of(steps: dict[str, Entry], ref: str, at: dict[str, int]) -> tuple[Forward, Intervention]:
    """The step point and the entry `<step>.<entry>` names among `steps` (or `<fit>.<eval>.<step>.<entry>`, in
    that fit's body), at the step's point that agrees with `at`: the one place a reference to an entry resolves,
    so a swept site resolves at every point."""
    thing, _, member = ref.partition(".")
    step = agreeing(steps[thing], at)
    if step.kind == "fit":
        return entry_of(step.steps, member.partition(".")[2], step.at)
    return step, entries(step)[member]


def record_of(steps: dict[str, Entry], ref: str, at: dict[str, int]) -> Labelled:
    """The record of the value `ref` names among `steps`, at the point agreeing with `at`, which carries its
    labelled layout: an entry's (`<step>.<entry>`), an op step's or a generation's own (`<step>`), or a fit
    member's: a body value under a named eval (`<fit>.<eval>.<member>`: the body's record with the dataset that
    eval reads it over, its read named from the root), a term's history (`<fit>.history.<term>`, `[epochs]`) or
    a trained featurizer (no axes)."""
    thing, _, member = ref.partition(".")
    step = agreeing(steps[thing], at)
    if step.kind != "fit":
        found = entries(step)[member] if member else step
        assert isinstance(found, Labelled), f"{ref} names no value"
        return found
    label, _, rest = member.partition(".")
    if label == "history" or label in step.train:
        return Labelled(dims=["epochs"] if label == "history" else [])
    inner = record_of(step.steps, rest, step.at)
    inside = inner.read is not None and inner.read.partition(".")[0] in step.steps
    return inner.model_copy(update={"dataset": step.eval[label].get(inner.dataset, inner.dataset),
                                    "read": f"{thing}.{label}.{inner.read}" if inside else inner.read})


def scope(plan: Plan, where: str) -> dict[str, Entry]:
    """The steps an entry `where` (`[<fit>.]<step>.<entry>`, as `uses` names it) sees by name: the root's, and
    a fit's body steps beside them when it stands in one."""
    if where.count(".") < 2:
        return plan.steps
    return {**plan.steps, **points(plan.steps[where.partition(".")[0]])[0].steps}


def horizon(step: Generate) -> int | None:
    """The most forward calls a generate step's rows make: its `max_new_tokens`, when it gives one."""
    most = (step.model_extra or {}).get("max_new_tokens")
    return most if isinstance(most, int) else None


def held_rows(plan: Plan, name: str) -> list[dict]:
    """Dataset `name`'s rows as the plan holds them, which the run's plan does (`steps.held`); one that holds none
    (a hub dataset's, a loaded plan's file rows) is refused by name."""
    rows = plan.data[name].rows
    if rows is None:
        fail(f"data.{name}", "the plan holds none of its rows; `compile.materialized` loads them, as the run does "
             "first (`steps.held`)")
    return rows


def body_view(plan: Plan, fit: str, at: dict[str, int] | None = None, swap: dict[str, str] | None = None) -> Plan:
    """The plan as fit `fit`'s body sees it: the body's steps beside the root's, which they shadow (the body
    at the fit's point `at`; without one, a list of each step's points when the fit is swept), and each
    dataset a substitution `swap` maps read under the body's name for it."""
    entry = plan.steps[fit]
    fits = [agreeing(entry, at)] if at is not None else points(entry)
    one = at is not None or not isinstance(entry, list)
    steps = {n: fits[0].steps[n] if one else [p.steps[n] for p in fits] for n in fits[0].steps}
    return plan.model_copy(update={"steps": {**plan.steps, **steps},
                                   "data": {**plan.data, **{a: plan.data[b] for a, b in (swap or {}).items()}}})


# -- an entry's brackets ---------------------------------------------------------------------------------


class Unbracketed(ValueError):
    """A `decode` with no matching encode open, at index `at` among the entry's ops."""

    def __init__(self, at: int, message: str) -> None:
        super().__init__(message)
        self.at = at


def bracketed(listed: list[Op]) -> list[dict]:
    """An entry's ops as their brackets nest: each op a node `{"at", "op"}`, an `encode` also holding the ops
    after it as `inner` up to the `decode` that closes it (`closed`, that decode's index) or to the end. A decode
    closes the most recent encode still open; one with none open, or of another featurizer, is `Unbracketed`."""
    top: list[dict] = []
    opened: list[dict] = []
    for i, o in enumerate(listed):
        ref = arguments(o).get("featurizer")
        if o.op == "decode":
            if not opened:
                raise Unbracketed(i, f"decode of {ref!r} closes no encode: none is open in this entry here; put the "
                                     f"encode of {ref!r} before it, the edits of its coordinates between them")
            theirs = arguments(opened[-1]["op"])["featurizer"]
            if theirs != ref:
                raise Unbracketed(i, f"decode of {ref!r}, and the open encode is of {theirs!r} "
                                     f"(ops[{opened[-1]['at']}]); close {theirs!r} first: a decode closes the most "
                                     "recent encode not yet closed")
            opened.pop()["closed"] = i
            continue
        node = {"at": i, "op": o}
        (opened[-1]["inner"] if opened else top).append(node)
        if o.op == "encode":
            node.update(inner=[], closed=None)
            opened.append(node)
    return top


def visited(nodes: list[dict], around: tuple[dict, ...] = ()) -> Iterator[tuple[dict, tuple[dict, ...]]]:
    """`bracketed` nodes in the order their ops run, each with the encode nodes open around it, outermost first."""
    for node in nodes:
        yield node, around
        yield from visited(node.get("inner", []), around + (node,))
