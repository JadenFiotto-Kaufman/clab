"""The plan: one model that is a document as written and, compiled, the document resolved for running.

`Plan(path)` reads a document; `plan.compile()` returns it compiled (rows loaded (a hub dataset's checked, then
left out), sites and named interventions inlined, bundles loaded, sweeps expanded into points, names resolved); an
engine's `execute` fills `plan.results` and `plan.run`; `plan.dump(out)` writes a run's output directory and
`Plan.load(out)` reads one back. The records (sites, interventions and their ops, positions, steps, datasets,
featurizers) are the same models in both states; a *derived* field is one compile or the run fills.

The plan's methods are views and serialization, never execution: it pickles, prints and means the same
thing wherever it lands, which is what lets it travel into a remote session and back.
"""

import functools
import json
import math
import os
import re
import types
from collections.abc import Iterator
from typing import Annotated, Any, Literal, NoReturn, TypeVar, Union, cast, get_args, get_origin

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
from pydantic_core import core_schema
from torch import Tensor

from clab import featurizers, ops
from clab.format import address, data, selectors
from clab.format.selectors import Selector
from clab.format.sweeps import agreeing_point, coordinate_spelling, find_sweeps, points, projection


class PlanError(ValueError):
    """A document that cannot mean anything; the message starts with the document path of what is wrong."""


def fail(path: str, message: str) -> NoReturn:
    raise PlanError(f"{path}: {message}" if path else message)


# The builtin types pydantic names a union's member by in an error's location; a model member goes by its class name.
MEMBER_TYPES = ("str", "int", "float", "bool", "dict", "list", "none", "tuple", "set")


def located(loc: tuple, given: Any, missing: bool) -> tuple[str, list[tuple[str, str]]]:
    """A pydantic error's location as a document path (`steps.patched.input[1]`), and the union members it went
    through, each as the path where the union stands and the member's tag. Pydantic names a union's member in the
    location (`str`, `list[union[str,TextPart]]`, `TextPart`, and this module's `~` tags), which the document does not
    hold. So a part that is no key of the document there and is spelled as a type is a member's tag. A `missing`
    error's last part is the field the document leaves out."""
    path, members, here = "", [], given
    for i, part in enumerate(loc):
        if isinstance(part, int):
            path += f"[{part}]"
            here = here[part] if isinstance(here, list) and 0 <= part < len(here) else None
        elif isinstance(here, dict) and part in here or missing and i == len(loc) - 1:
            path += f".{part}" if path else part
            here = here.get(part) if isinstance(here, dict) else None
        elif part.startswith("~") or "[" in part or part in MEMBER_TYPES or part[:1].isupper():
            members.append((path, part))
        else:
            path += f".{part}" if path else part
            here = None
    return path, members


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
                if key not in {f.alias or k for k, f in cls.model_fields.items()}:
                    raise ValueError(f"unknown field {key!r}; valid: "
                                     f"{', '.join(cls.model_fields[k].alias or k for k in documented(cls))}")
        return given

    @model_serializer(mode="wrap")
    def _derived_last(self, handler: SerializerFunctionWrapHandler) -> Any:
        """The record as written: the document's fields first, then what compile or the run derived."""
        dumped = handler(self)
        later = [k for k in type(self).model_fields if k not in documented(type(self))]
        return {**{k: v for k, v in dumped.items() if k not in later}, **{k: v for k, v in dumped.items() if k in later}}


@functools.cache
def documented(cls: type[BaseModel]) -> list[str]:
    """The fields a document writes: every field but the derived ones, worked out once per record type (every
    record checks its keys against them, and writes them first), and only read."""
    return [k for k, f in cls.model_fields.items()
            if not (isinstance(f.json_schema_extra, dict) and f.json_schema_extra.get("derived"))]


class Sweep(Record):
    """A value standing for several. Whatever uses it runs once per value, a point each, on the axis its
    `label` names (default: the field it stands for); sweeps with one label are one axis."""
    sweep: list[Any] = Field(min_length=1)
    label: str | None = None


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
    if get_origin(member) in (list, tuple):
        return isinstance(value, (list, tuple))
    if member is type(None):
        return value is None
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return {int: number and isinstance(value, int), float: number, str: isinstance(value, str)}.get(member, False)


T = TypeVar("T")
Swept = Annotated[Union[T, Sweep], ByShape()]

AT = "the point's index on each sweep axis it varies over"
DATASET = "the dataset whose rows the step's values are aligned to"


# The quantized dtypes (nnsight's names for them) and the dtype each computes in: what its values, and the weights
# it leaves alone (norms, embeddings, the head), are held in.
QUANTIZED = {"nf4": "bfloat16", "fp4": "bfloat16", "int8": "float16", "fp8": "bfloat16"}


# The loader keys a model block may not hold, and why: another spelling of a key the block owns (`key`, `revision`,
# `dtype`, a quantization in `dtype`, transformers' and vLLM's), or a run option (where the model runs, how it is
# sharded, and the vLLM engine arguments its options set), which no engine takes from the block. Compile refuses a document's by name, and an engine one in a block it is given; a plan read
# back from a run (`Plan.load`) is a record, which neither judges.
OWNED = {"device": "where the model runs is a run option: --device",
         "device_map": "where the model runs is a run option: --device",
         "dispatch": "the engine places its model itself",
         "torch_dtype": "the dtype is spelled `dtype`",
         "quantization_config": "a quantization is spelled in `dtype` (`nf4`, `fp4`, `int8`, `fp8`)",
         "load_in_4bit": "a quantization is spelled in `dtype` (`nf4`, `fp4`, `int8`, `fp8`)",
         "load_in_8bit": "a quantization is spelled in `dtype` (`nf4`, `fp4`, `int8`, `fp8`)",
         "quantization": "a quantization is spelled in `dtype` (`nf4`, `fp4`, `int8`, `fp8`)",
         "_commit_hash": "the revision is spelled `revision`",
         "pretrained_model_name_or_path": "the checkpoint is spelled `key`",
         "repo_id": "the checkpoint is spelled `key`",
         "tp_plan": "tensor parallelism is a run option: -o tp=N",
         "tp_size": "tensor parallelism is a run option: -o tp=N",
         "distributed_config": "tensor parallelism is a run option: -o tp=N",
         "device_mesh": "tensor parallelism is a run option: -o tp=N",
         "tensor_parallel_size": "tensor parallelism is a run option: -o tp=N",
         "gpu_memory_utilization": "the memory vLLM reserves is a run option: -o memory=F",
         "max_model_len": "the longest sequence vLLM serves is a run option: -o max_len=N"}


class Model(Record):
    """The pinned checkpoint: `key`, `revision` and `dtype`, torch's spelling of a float dtype or one of nnsight's
    names for a quantization: `nf4` and `fp4` hold the weights in bitsandbytes' 4 bits (NormalFloat, FloatPoint) and
    `int8` in its LLM.int8() 8 bits, on the CPU or on CUDA, with bitsandbytes installed; `fp8` holds them in
    transformers' block-wise 8-bit floats, on a GPU of compute capability 8.9 or later. A quantized model computes in
    `bfloat16` (`int8` in `float16`), which is the dtype its values have. A key that respells one of the three
    (`torch_dtype`, `quantization_config`, `_commit_hash`, ...) or names a run option (`device`, `device_map`:
    `--device`; `tp_plan`, `tensor_parallel_size`, ...: `-o tp=N`; vLLM's `gpu_memory_utilization` and
    `max_model_len`: `-o memory`, `-o max_len`) is refused by name at compile (`plan.OWNED`). Every other
    key goes to the engine's loader untouched (`attn_implementation`, ...), which refuses what it cannot honor as
    the engine loads, as the load refuses a checkpoint or revision the offline cache lacks. The attention interior
    needs `attn_implementation: "eager"`."""
    model_config = ConfigDict(extra="allow")
    key: str
    revision: str
    dtype: Literal["float32", "float16", "bfloat16", "nf4", "fp4", "int8", "fp8"] = "float32"

    @property
    def computes(self) -> str:
        """The dtype the model computes in, which its values have: `dtype`, or a quantization's (`QUANTIZED`)."""
        return QUANTIZED.get(self.dtype, self.dtype)


class TextPart(Record):
    """`{"text": "<text>"}`: a text, written as it is, in a list `input` and as a derived column's `after` or
    `before`: one string, its only field (no `strip` or `lower`, and no list, which a literal in an op's argument
    takes)."""
    text: str


class Derived(Record):
    """`{"from": "<column as the source has it>", "after"?: {"text": ...}, "before"?: {"text": ...}, "strip"?:
    bool}`: a column whose text on each row is cut from another's, text on text, decided at compile, never a
    computation on its values: the text of column `from` after the first occurrence of `after`'s text, then
    before the first occurrence of `before`'s in what is left (each compared exactly), then, under `strip`,
    without its surrounding whitespace. `{"from":
    "answer", "after": {"text": "#### "}, "strip": true}` is a GSM8K answer's number. `from` stays under its own
    name. Refused by name: a row whose `from` is not text, and one on which `after`'s or `before`'s text does not
    occur, naming the row."""
    model_config = ConfigDict(serialize_by_alias=True)
    source: str = Field(alias="from")
    after: TextPart | None = None
    before: TextPart | None = None
    strip: bool = False

    @field_validator("after", "before", mode="before")
    @classmethod
    def _text(cls, given: Any) -> Any:
        """A text is the object, as it is in a list `input`: a bare string is refused with it."""
        if isinstance(given, str):
            raise ValueError(f"{given!r} is a string; text written as it is is {{'text': {given!r}}}")
        return given


class Data(Record):
    """A dataset: where its rows come from and how its text renders, never a computation on its values. Its rows
    come from one of three sources: `path` is `data/<path>.json` under the corpus root, `rows` are inline, and
    `hf` is a dataset on the Hugging Face hub, `<owner>/<name>` at `revision` (a commit, as `model` pins one; a
    branch is passed as written, and the digest catches what it moves to), in its subset `config` where it has
    several. A file's and inline rows travel in the compiled plan; a hub dataset's never do: compile loads them to
    check the document and records their digest, and the run loads them again where it runs (under `ndif`, the
    server) and refuses other rows before any forward. Then, in this order: `split` picks which split of the
    dataset, the hub dataset's split (left out: its only one) or the rows of a file whose `split` column is that;
    `shuffle` puts the rows in one fixed random order, the same on every machine (seed 0); `skip` drops the first
    N; `take` keeps the first N of the rest (after `shuffle`, a random sample of N; after `skip` too, a shard of
    one order: `{"shuffle": true, "skip": 20000, "take": 10000}` is the third of 10,000, so a harvest too large
    for one run is made in shards); and `columns` renames them, `{"<name>": "<column as the source has it>"}`,
    each name that column and no longer under its own, every other column as it is (so one file serves a fit's
    `eval` substitution under the body's column names), or derives one, `{"<name>": {"from": "<column as the
    source has it>", ...}}`, text cut from another column's (`Derived`),
    which joins the rows and their digest as a column the source has does. `template` is passed to the checkpoint's
    `apply_chat_template` wherever a chat column of the dataset renders (`{"add_generation_prompt": true,
    "enable_thinking": false}`), so a chat can end in the generation prompt and keep its `turn` scopes; a
    checkpoint whose template reads the clock (`strftime_now`: Llama-3.2's writes today's date into the system
    header it adds) renders a chat only with its date pinned, `{"date_string": "26 Jul 2024"}`, since the rows'
    digest cannot say that the tokens moved with the day. A row is
    JSON. Refused by name: a second source, an `hf` that is not `<owner>/<name>`, `hf` without `revision`, and
    `revision` or `config` without `hf`; a hub dataset the hub does not give, or offline a commit `revision` other
    than the one its cache holds (offline, a branch reads as the cache holds it), a split it does not have (or, of
    several, none named), and a column of it whose values are not JSON (images, audio, bytes); a `skip` that
    leaves no row, `take` beyond the rows `skip` leaves, a `columns` source the rows lack or a name that is a
    column they keep (a swap, `{"a": "b", "b": "a"}`, keeps neither), a `template` on a dataset no step reads a
    chat column of, and, before
    any forward, a chat of a clock-reading template without `date_string`. `sha256` and `n_rows` describe the rows
    these leave."""
    path: str | None = None
    rows: list[dict] | None = None
    hf: str | None = None
    revision: str | None = None
    config: str | None = None
    split: str | None = None
    shuffle: bool = False
    skip: int | None = None
    take: int | None = None
    columns: dict[str, Annotated[Union[str, Derived], ByShape()]] | None = None
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
        if self.skip is not None and self.skip < 0:
            raise ValueError(f"skip {self.skip}: a dataset drops 0 rows or more from the start")
        if "tokenize" in (self.template or {}):
            raise ValueError("template: a chat renders to text, which clab tokenizes; it takes no 'tokenize'")
        return self


class Offset(Record):
    """`{"offset": k}` or `{"offset": [k, ...]}`: on an attention pattern's `key` axis, at each query position p the
    entry reads, key p + k of the row's own keys, one per offset in the order written (`{"offset": -1}` the previous
    token, `[-1, -2]` the two before it): the axis is as wide as the offsets are many, and an entry is missing (NaN)
    where p + k is not one of the row's keys (before its first or past its last). A read's alone: a write selects
    keys counted from the start."""
    offset: Swept[int | list[int]]

    @field_validator("offset")
    @classmethod
    def _distinct(cls, given: Any) -> Any:
        if isinstance(given, list) and (not given or len(set(given)) < len(given)):
            raise ValueError(f"offsets {given} name at least one, each once")
        return given


class Site(Record):
    """A component, narrowed: `layers` (a selector, as `pos` has them, of the model's layers: an int is that
    layer's value; a slice or a list of picks is one value, a `layers` axis first, labelled with the layers
    it picks: `[0, 3, 7]` is three, `"-2:"` the last two) on a block's component, and `select`
    (`{<axis>: <selector>}`, which Components describes; on `key`, a selector of keys counted from the start, or
    keys relative to each query, `{"offset": k}`, a key offset). A read and a write take `layers` alike: an entry is
    made at each layer they pick, in one forward."""
    component: str
    layers: Swept[Selector] | None = None
    select: dict[str, Swept[Annotated[Union[Selector, Offset], ByShape()]]] | None = None

    @model_validator(mode="after")
    def _narrowing(self) -> "Site":
        if self.component not in address.COMPONENTS:
            raise ValueError(f"unknown component {self.component!r}; the components clab addresses (`clab describe "
                             f"components` says what each is): {', '.join(address.COMPONENTS)}")
        if address.COMPONENTS[self.component]["per_layer"] != (self.layers is not None):
            raise ValueError(f"{self.component} is a block's value and needs 'layers'"
                             if address.COMPONENTS[self.component]["per_layer"]
                             else f"{self.component} is the model's own value and takes no layers")
        valid = address.selectable(self.component)
        for axis, index in (self.select or {}).items():
            if axis not in valid:
                raise ValueError(f"{self.component} has no axis {axis!r} to select; its axes: {', '.join(valid)}")
            if isinstance(index, Offset) and axis != "key":
                raise ValueError(f"select {axis} {index.model_dump()}: an offset is relative to the query position, "
                                 "on an attention pattern's key axis alone")
            if axis == "key" and not isinstance(index, (Sweep, Offset)) and not selectors.keyed(index):
                raise ValueError(f"select key {index!r}: the key axis is as long as each row, so a selector there "
                                 "picks a fixed number of keys counted from the start (0, [0, 2], \"0:3\"), or keys "
                                 "relative to each query ({'offset': -1})")
        return self


class Column(Record):
    """A column read as text, where a scope searches for it or a metric scores it: `"<dataset>.<column>"`, or
    `{"column": "<dataset>.<column>", "strip"?: bool, "lower"?: bool}` (an indexed input's variables are named
    whole: `pairs.counterfactual_inputs_variables[0].entity`). A row's value may be a list of strings: a set, any
    of which is the answer. The flags say how its text compares and never rewrite the data: `lower` compares it
    lowercased, and `strip` makes its surrounding whitespace optional.

    Where an op takes `Tokens` or `Text` (a metric's answer, `contains`'s text), a literal may stand in place of the
    column: `{"text": "</think>", "strip"?: bool, "lower"?: bool}`, its text a string or a list of them (a set), the
    same on every row of whatever value it goes with, and read exactly as a column holding it on every row is
    (tokenized as written, the flags as here). A scope names its text in its own `text` field.

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


ColumnRef = Annotated[Union[str, Column], ByShape()]


class Scope(Record):
    """`{"column": <column>}`: the row's own text of that column (a `Column`: the string or the object), each
    member of a set; `{"text": "<literal>"}`: that text; or `{"turn": "<role>"}`: that role's messages of a
    chat, each an occurrence. `column` and `text` may each be a list: the scope's texts are then every one
    listed (a string is the one-element list), their occurrences together, ordered by where each ends. Every
    scope takes `occurrence`, which of its occurrences it is (a selector; default 0, the first): a text's
    occurrence lies on token and word boundaries, as `pos` says; `until`, whether it is every token of the frame
    up to and including the occurrences it picks rather than their tokens alone (every token of the frame on a
    row without them); and `before`, whether it is the positions that predict its tokens rather than the tokens
    themselves. A `text` takes
    `strip` and `lower`, as a column's reference does (a turn's contents are located as the template rendered
    them)."""
    column: Annotated[Union[ColumnRef, list[ColumnRef], None], ByShape()] = None
    text: str | list[str] | None = None
    turn: str | None = None
    occurrence: Swept[Selector] = 0
    strip: bool = False
    lower: bool = False
    until: bool = False
    before: bool = False

    @field_validator("column", mode="before")
    @classmethod
    def _references(cls, given: Any) -> Any:
        """Each column a reference, the string or the object, so a number or a sweep is refused by name."""
        for ref in given if isinstance(given, list) else [] if given is None else [given]:
            if not isinstance(ref, (str, dict)) or "sweep" in ref:
                raise ValueError(f"{ref!r} is not a column reference, '<dataset>.<column>' or {{'column': "
                                 "'<dataset>.<column>', 'strip'?: bool, 'lower'?: bool}" + (
                                     "; a scope's column takes no sweep (its occurrence does)" if isinstance(ref, dict)
                                     else ""))
        return given

    @field_validator("column")
    @classmethod
    def _columns(cls, given: str | Column | list[str | Column] | None) -> Column | list[Column] | None:
        """Each column a `Column`, a string the one without flags."""
        def made(ref: str | Column) -> Column:
            return Column(column=ref) if isinstance(ref, str) else ref
        return [made(c) for c in given] if isinstance(given, list) else None if given is None else made(given)

    @field_validator("text", mode="before")
    @classmethod
    def _literal(cls, given: Any) -> Any:
        """A text is a literal; its `strip` and `lower` are the scope's own, beside it."""
        for text in given if isinstance(given, list) else [given]:
            if isinstance(text, dict):
                raise ValueError(f"{text!r} is an object; a text is a literal, and its strip and lower go on the scope "
                                 f"beside it: {{'text': {text.get('text', '<literal>')!r}, 'strip': true}}")
        return given

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
              | {"frame"?: frame, "scope"?: scope, "index"?: swept}        # index default ":"
    frame    := swept                                                     # default 0
    scope    := {"column": column or a list, "occurrence"?: swept, "until"?: bool, "before"?: bool}
              | {"text": one or a list, "occurrence"?: swept, "strip"?: bool, "lower"?: bool, "until"?: bool,
                 "before"?: bool}
              | {"turn": "<role>", "occurrence"?: swept, "until"?: bool, "before"?: bool}   # occurrence default 0
    column   := "<dataset>.<column>" | {"column": "<dataset>.<column>", "strip"?: bool, "lower"?: bool}
    selector := int | "<start>:<stop>:<step>" | [int, ...]                # one, a slice, picks
    swept    := selector | {"sweep": [selector, ...]}                     # a point per selector
    ```

    Each stage is a selector over what the stage before it left (`selectors.select`): an int counts one
    (negative from the last), a slice string is a range in Python notation, half-open, each part optional
    (`":"` every one, `"-4:-1"`, `"2:"`, `"::2"`), a list picks those, in the order written. A selector
    that picks nothing however many things there are (`"0:0"`, `"5:2"`) is refused, and so is an index that
    picks none of the tokens of a frame of a static number of decode calls (`{"frame": "1:3", "index":
    "5:"}`), or whose static width those tokens cannot fill (`{"frame": "1:3", "index": "1:3"}`, two wide over the
    one token it picks).

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

    **until** (default false): the scope is every token the frame selected up to and including the last token of
    the occurrences `occurrence` picks, rather than those occurrences' tokens; on a row that has fewer of them than
    it picks (the text never occurs), every token the frame selected, in the prefill too, so nothing is refused
    and the record says per row where it ended. So `{"frame": "1:", "scope": {"text": "</think>", "until": true}}`
    is each row's generated inputs through its first `</think>`, and `count` of a read there is the thinking's
    length in tokens (all its inputs on a row that never closes it). The scope is the frame's: an occurrence that
    holds none of the frame's tokens is not one of its own, so over `"2:"` a text said at call 1 is not found and
    until is the whole frame, as on a row that never says it. An until scope has no width compile can fix, and
    two values of different reads pair only at a fixed width, whether one forward or two made them. With `before`
    its frame is decode calls: over a frame that holds the prompt it would start before
    the frame's first token (position 0, or at a generation's head the prompt's last, before which its prefill
    computes no head) and name no position on any row, so compile refuses it. A write's positions beyond the
    prefill are decided call by call, as each runs, and until names positions up to an occurrence not yet seen
    then, so a write there takes no until.

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
    `"-3:"`, `"2:5"`; see `selectors.fixed`), and a scope, if it has one, is one occurrence in the prefill and not
    `until`; compile writes it. Any other record's width is open at compile, and a prefill record's is the number
    every row's positions hold where preflight resolves them and they agree; where rows hold different numbers the
    value is as wide as the widest row, NaN past a row's own. Only pairing with another read's value needs the
    width to be one number, since two values of different reads pair only at a fixed width, whether one forward or
    two made them: compile refuses the pair wherever the `pos` leaves it open, except where the prefill's rows
    decide it: an index of no static width over the prompt (`":"`, `"2:"`), or over one occurrence of a scope there,
    not `until` (the whole scope, or such an index over it). Where the rows agree the pair runs at their width;
    where they disagree it is refused by name, with each count's rows and a static span or index to read instead,
    in preflight, or at compile for a loaded read, which carries its rows' positions. In the prefill (`frame: 0`) an
    occurrence a row does not have, or a static index that does not fit, is refused by name before any forward: the data decided. Anywhere else it
    is a result, as is a call the row never reached, and as is an index that picks none of a row's tokens (`"5:"`
    on a shorter row), in the prefill too: the row is `alignment_missing`, NaN all the way but in a `count`, which
    is 0 there (`clab table` prints the row flagged, with no value, for the read and every value made from it, the
    `count` included), and a write there is refused by name, since
    it has no token to write. So is a later step's write that would take such a value as its operand on a row it
    lands on (a read at `{"frame": 3, "index": 0}` on a row whose generation made three calls, or
    an op step's value that takes it as any named value: `of`, `with`, a `factor`), which would write NaN there:
    refused by name, at compile where the read's frame is a decode call at or past its generation's
    `max_new_tokens`, which no row makes, and otherwise in the run, naming the row and the call, a forward's write
    before its batch's forward, one at a decode call once the batch's generation says which rows made that call; a
    generation in a fit's body records no positions, so a write in the body with such an operand writes NaN there.
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
        if calls is None or 0 in selectors.select(frame, 1):
            return self
        got, width = selectors.select(index, calls), selectors.fixed(index)
        if not got:
            raise ValueError(f"index {index!r} picks none of the {calls} tokens of frame {frame!r}'s decode "
                             "calls, one each")
        if width is not None and len(got) != width:
            raise ValueError(f"index {index!r} is {width} tokens wide and picks {len(got)} of the {calls} tokens of "
                             f"frame {frame!r}'s decode calls, one each, so it fits no row")
        return self


class Op(Record):
    """An op inside an entry: `{"op": <name>, ...its arguments}` (see Ops)."""
    model_config = ConfigDict(extra="allow")
    op: str
    dims: list[str] | None = derived(None, "the layout of the op's value, at each layer the entry is made at (the "
                                           "op after it meets it; the first meets the site's selection)")
    space: list[dict] = derived([], "the coordinates the op's value is in: the encodes open after it, outermost "
                                    "first, each its `featurizer` and `select`; none in the site's own layout, and "
                                    "none once an op changed the last axis they lie on (a sum over features)")

    @model_validator(mode="after")
    def _arguments(self) -> "Op":
        if self.op == "permute":
            raise ValueError("permute is no op of an entry, whose ops act on each row's live value: it is a step "
                             "over a read, {'kind': 'op', 'op': 'permute', 'of': '<step>.<entry>', 'seed': ...}, "
                             "whose value a later forward's entry swaps in")
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
                                             "read at where not. Null for a value from no layer (a model's own, or "
                                             "one reduced over its layers). `[]` until preflight resolves a slice or "
                                             "a layer counted from the last")
    read: str | None = derived(None, "the entry whose value this is, or descends from through op steps, whose "
                                     "positions it holds where it has a `pos` axis; null for a value no read made, "
                                     "and for one whose rows hold other rows' values (a `permute`'s and what "
                                     "descends from it)")
    width: int | None = derived(None, "how many positions the value holds on every row, where it has a `pos` axis "
                                      "and that is one number. Compile writes the static width its read's `pos` "
                                      "names. Where the `pos` leaves it open, it is the number every row's "
                                      "positions hold once they are resolved (a prefill read's by preflight, a "
                                      "loaded read's by its run), where they agree. Null where rows hold different "
                                      "numbers of positions, where none are resolved yet, or there is no `pos` "
                                      "axis. Only pairing with another read's value needs it to be one number")
    kept: dict[str, selectors.Kept] = derived({}, "the indices a site's `select` keeps on each axis it selects, in "
                                                  "their order (`key`: the keys it picks, or `{\"offset\": [...]}`, "
                                                  "those relative to each query). An op step's are those of the "
                                                  "value it is over, on the axes its own still has. An unselected "
                                                  "axis is whole. `[]` stands until preflight resolves a slice or a "
                                                  "negative pick")
    space: list[dict] = derived([], "the coordinates the value is in: the encodes left open around it, outermost "
                                    "first, each its `featurizer` and `select`; none for the site's own layout, and "
                                    "none once an op changed the last axis they lie on (a sum over features)")
    axes: list[str] = derived([], "the sweep axes the value depends on, in axis order: the sweeps on its own entry "
                                  "(its site, its positions, an op's literal argument), those of every value or "
                                  "featurizer its ops name, and, for an entry of a forward, the forward's "
                                  "(`schedule`); an op step's are its `of`'s and its operands'; a fit member's are "
                                  "the fit's. The value is one per distinct point on these axes, and its result is "
                                  "keyed by them")


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
    names it); an entry nothing names only writes, and costs no memory. An entry whose first op is `grad` (see
    Ops) writes nothing: its value is a gradient, and every op after `grad`, an edit too, acts on it once it is
    off the model.

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
    positions, with no scope, pairs its i-th position with the write's i-th. On rows whose prompts differ in
    length that is checked before any forward: a position pairs across prompts of different lengths only where
    every token between it and the end it is counted from is the same on both, whatever the two tokens it pairs.
    Any other is refused, and a scope names the token in each row. What an operand's
    layout and labels must be is Ops' one rule. A
    bundle at the root is a dict of entries pulled into a step by name, a template whose names resolve in the
    step that pulls it in: an operand naming one of the bundle's own entries bare (`"with": "premix"`) means that
    entry of whichever step pulls it in, and `<a step>.<one of its entries>` is refused, since pulled into
    another step it would still mean that step's. An operand from an entry of its own step is served in a
    forward, not inside a generate: a cross-site adapter inside a generate is refused."""
    site: Annotated[str | Site, ByShape()]
    pos: Swept[int | Pos] = Pos()
    ops: list[Op] = []
    gathered: bool = derived(False, "whether the run takes the entry's value off the model: `saves` names it, "
                                    "a later entry or step uses it, or a gathered `grad` entry is its gradient")
    per_call: int = derived(0, "how many of the ops after the entry's last edit a generation runs at each call it "
                               "reads the entry beyond its prefill, on that call's own value. The rest run once the "
                               "calls are arranged by position. Those counted are the leading ops that are "
                               "positionwise (Ops), keep the `pos` axis and take no operand that has one, up to the "
                               "first that scores an answer of several tokens, which preflight finds. 0 in a forward "
                               "and in the prefill")
    positions: dict[str, list] = derived({}, "the positions the entry resolved to, per row, under the dataset whose "
                                             "rows they are. A position counts as `pos` does: an index into the row's "
                                             "prompt tokens, BOS first, then its generated ones. `alignment_missing` "
                                             "stands for a row it does not reach. Preflight sets the prefill's, and "
                                             "the run a generation's decode calls'")

    @field_validator("pos", mode="before")
    @classmethod
    def _selectors(cls, given: Any) -> Any:
        """A string, a float or a boolean is not read as a number, in a sweep's values too, as no selector is: the
        int shorthand is an int, and a slice goes under `index`."""
        swept = isinstance(given, dict) and isinstance(given.get("sweep"), list)
        for value in given["sweep"] if swept else [given]:
            if isinstance(value, str):
                raise ValueError(f"{value!r} is a string, which is not read as a number; a pos is an int ({{'frame': 0, "
                                 f"'index': <int>}}) or an object, each of whose selectors is {selectors.FORMS}")
            if isinstance(value, (float, bool)):
                raise ValueError(f"{value!r} is not a selector: {selectors.FORMS}; a pos is an int ({{'frame': 0, "
                                 "'index': <int>}) or an object")
        return given


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


def gradient_of(entry: Intervention) -> str | None:
    """The entry of the same forward whose value a `grad` entry is the gradient of (its first op's `of`), or None
    for an entry that takes no gradient."""
    first = entry.ops[0] if entry.ops else None
    return arguments(first)["of"] if first is not None and first.op == "grad" else None


def writes(entry: Intervention) -> bool:
    """Whether an entry writes back: it holds an edit, and takes no gradient (`grad` first, whose edits act on the
    gradient)."""
    return gradient_of(entry) is None and any(edits(o) for o in entry.ops)


def gathers(entry: Intervention) -> list[Op]:
    """The ops of an entry that run on its value once it is taken off the model: those after its last edit, and
    every one of a `grad` entry's."""
    last = -1 if gradient_of(entry) is not None else max((i for i, o in enumerate(entry.ops) if edits(o)), default=-1)
    return entry.ops[last + 1:]


def open_encodes(steps: dict[str, Any], featurizer: str) -> list[str]:
    """The steps and entries of `steps` (a fit's body) that read `featurizer`'s coordinates with an encode no decode
    closes, by name: an encode step, and an entry whose ops hold one after its last edit (`gathers`). What they
    encode is what the body shows the featurizer (`Kind.seen`), which an sae's scalars are over."""
    def encodes(o: Any) -> bool:
        return o.op == "encode" and arguments(o)["featurizer"] == featurizer
    return [n for n, s in steps.items() if isinstance(s, OpStep) and encodes(s)] + \
        [f"{n}.{k}" for n, s in steps.items() for k, e in entries(s).items() if any(map(encodes, gathers(e)))]


class Forward(Record):
    """One forward over every row of a dataset input. `input` is `<dataset>.<column>` or
    `<dataset>.<column>[i]`, a text or a chat, or a list of text columns and texts, `{"text": "<text>"}`,
    of one dataset, joined per row as they are, with nothing between (`["days.input",
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

    Rows run in batches of the run's batch size, padded on the left as the run's `padding` says, with position
    ids that count real tokens: under `dataset`, the default, consecutive batches each padded to the input's
    longest row; under `batch` each padded to its own longest row; under `sorted` the rows longest first, each
    batch padded to its own, and every value put back in dataset row order. Padding is never a position, so a
    row's `positions` and its `key` axis (as wide as the input's longest row, NaN past the row's own keys) are the
    same under each. Under `dataset` a row's numbers depend on the width of the input's longest row, not on which
    rows share its batch. At some thread counts the batch's other rows can still move them in the last bits. They
    do depend on the batch's shape: a trailing partial batch is a smaller matrix, and a batch padded to its own
    width a narrower one. Either can differ in the last bits: in float32 a forward's values move by 6e-8 at most on
    the corpus (`tests/test_padding.py`), 1.2e-7 where a write takes what a fit trained on such values, and a real
    model's move more. A featurizer trained through such a forward
    can move by more than the last bits, since an optimizer such as Adam scales a rounding-sized gradient up to its
    step size. So a fit whose trained tensors are compared belongs under `dataset` padding. Results reproduce to the
    bit at a fixed batch size, padding, readout and thread count, each of which the run records."""
    kind: Literal["forward"]
    input: str | list[str | TextPart]
    interventions: Interventions = {}
    at: dict[str, int] = derived({}, AT)
    dataset: str | None = derived(None, DATASET)
    schedule: dict[str, Any] = derived({}, "the forward this point runs, or the point whose forward serves it. A "
                                           "forward depends on the step's options and on its writes alone (`axes`: "
                                           "each write's site, positions, ops up to its last edit and their operands). "
                                           "The points that agree on those axes are served by the forward of the "
                                           "first of them. Each of the others holds `serving`, the first one's "
                                           "coordinates, and nothing else; the points a forward serves are those that "
                                           "name it (`plan.serving`). A forward that reads entries only for its own "
                                           "writes holds `freed`, those entries, of whose batches the run keeps none. "
                                           "Compile writes those. Preflight, with the model in hand, writes the rest "
                                           "at each point that runs its forward. `sites` holds the sites the forward "
                                           "visits, in the engine's forward order, each its component and layer and "
                                           "the records done there in order. A write there holds its ops up to its "
                                           "last edit, each operand resolved to a producer's name and point or to the "
                                           "name of a read of the same forward. A read there holds the point whose "
                                           "entry it is and the name it feeds, and `readout` where it reads `logits` "
                                           "that the run's readout `at-reads` does at the last block's output. `stop` "
                                           "is the site the forward ends at (null for a generation, which runs "
                                           "whole). A forward with `grad` entries holds `grad`, one item per distinct "
                                           "value differentiated: `of` the names of its reads, one per layer, and "
                                           "`wrt` the names of the `grad` reads whose gradient it is. A generation "
                                           "holds each record's `calls` and `online`, `prompt` on a part over the "
                                           "prompt of an entry beyond the prefill, and the `loop` of calls the engine "
                                           "iterates (null: every one). The engines execute it and decide nothing")
    batch_size: int | None = derived(None, "the rows a batch of the input holds, whatever the run's batch size, where "
                                           "the step fixes it: a fit's eval step's are the fit's `batch_size`; null, "
                                           "the run's")

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
    "prefill_chunk_size": (None, "a chunked prefill runs the model's forward more than once for the prompt, so a "
                                 "prefill read or write would meet one chunk of it"),
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
    to find), and a row with fewer occurrences than it picks is `alignment_missing`. A row's calls are what
    the engine actually generated, up to the
    one that chose the row's first EOS: the step's own `eos_token_id` when it gives one, else the id the
    engine's generation stops on (its generation config's). The step's own value is the
    new token ids, `[rows, new]`; `detokenize` makes them text. Options that would change the rows, the value or
    the calls (`num_return_sequences`, `num_beams`, `return_dict_in_generate`, `use_cache`, `prefill_chunk_size`)
    are refused before anything runs; any other option is the engine's `generate`'s to accept, and one it rejects
    fails when the step runs, naming the step.

    A read beyond the prefill runs the ops after its last edit at each call it reads, on that call's value, up
    to the first that is not positionwise (<the per-call ops> are), removes `pos`, takes an operand with a `pos`
    axis or scores an answer of several tokens; those from there on run once the batch's generation ends, on its
    calls arranged by position (its record's `per_call` counts those it runs at each call). So a metric of the head at
    `{"frame": ":"}` holds one number a call, `[rows, calls]`, where a read with no op, or whose first op waits,
    holds `calls × vocab_size` values a row in the model's dtype until the generation ends (twice that while it is
    arranged). An op waits where its bits depend on the shape of the value it runs on, as a call's value and the
    whole one's differ in shape: a matmul (`readout`'s or an `encode`'s), a sum over a long axis (`kl`'s, `js`'s,
    `entropy`'s, `cosine`'s, `mean`'s, `sum`'s, `std`'s, `norm`'s, `dot`'s), and `pow`, which torch computes two
    ways. On the CPU either way a value is the same to the bit. On CUDA a per-call metric can differ in its last
    bits from the same metric over the whole read, since torch picks a softmax's kernel by the tensor's outer size,
    which a call's value and the whole one's differ in.

    Rows run in batches as a forward's do, and reproduce to the bit only at a fixed batch size: the prefill
    projects the head at the last position only and every decode call is a `[batch, 1, ...]` matmul, so two
    batch sizes can differ in the last bits from call 0."""
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


# The positionwise ops, which a generation runs at each call, as `ops` declares them and in the tables' order,
# written into the docstring where `<the per-call ops>` stands.
Generate.__doc__ = (Generate.__doc__ or "").replace(
    "<the per-call ops>", ", ".join(f"`{name}`" for name in ops.OPS if ops.DECLARED[name]["positionwise"]))


class OpStep(Labelled):
    """An op over a value earlier steps produced: `of` names it, and the op's arguments sit beside it (see
    Ops). The step's value is the op's result; a step holds exactly one op, and chaining is the next step."""
    model_config = ConfigDict(extra="allow")
    kind: Literal["op"]
    op: str
    of: str
    at: dict[str, int] = derived({}, AT)
    permutation: list[int] | None = derived(None, "a `permute`'s: the dataset row whose value each of its rows "
                                                  "holds, as the run drew it (in a fit's body, at the body's last "
                                                  "run, which scores every training row, where each training "
                                                  "minibatch drew its own over its rows); null before it runs and "
                                                  "for any other op")

    @model_validator(mode="after")
    def _arguments(self) -> "OpStep":
        if self.op == "grad":
            raise ValueError("grad is no step: it is the first op of an entry of a forward, whose value is the "
                             "gradient of a later entry of that forward ({'op': 'grad', 'of': '<entry>'})")
        ops.check(self.op, arguments(self))
        return self


class Load(Labelled):
    """A value an earlier run saved, brought into this document with its record: `from` is that run's output
    directory under the corpus root (where its `manifest.json` is), `value` a name in its `saves`. The step's value
    is that saved value, in the dtype it was saved in, where the model is, and its record is the one the source's
    manifest holds (its layout and labels, the sweep axes it depends on and, where it holds positions, the read they
    are of: that entry's site, `pos` and positions per row, with the input it read), so every step after it treats
    it as the value it was: a fit's body takes its rows by id, an op step reduces it, a featurizer encodes a read's
    value at its site, and `table` labels its positions with this run's tokens of that input, which preflight
    encodes as a forward does. Compile reads the manifest and the file, and refuses by name a directory with no
    manifest, a name it did not save, a source whose `model` block is not this document's (key, revision, dtype and
    every loader key), a dataset the value names whose rows are not, in this document, the rows the source read in
    the same order (its `sha256` and `n_rows` as the source recorded them) or whose chats render with another
    `template`, and a sweep the value depends on that this document declares under the same label over other values.
    A swept value loads at every point of its axes, keyed by them as the run that saved it keyed them: the load
    declares each such sweep this document does not, with the run's label and values, so a value swept over layers
    loads with no other field swept. So are a bundle (a `pca`'s value, a trained featurizer), which a featurizer loads by
    `file_path`; a value in a featurizer's coordinates (an encode no decode closes), whose featurizer is the
    source's; a value a generation made, or one whose positions are a generation's, which count tokens this run
    does not have; and a load inside a fit's body. It calls no model; its cost's seconds are the value's move to
    the model's device. The value travels inside the compiled plan, as a bundle's tensors do,
    wherever the run is: under `ndif`, in the request, so a server runs it as `remote="local"` does and reads no
    file of this machine's; a request carries at most 512 MiB of loaded tensors over a document's loads, and a
    document that loads more is refused by name before anything is sent."""
    model_config = ConfigDict(serialize_by_alias=True)
    kind: Literal["load"]
    from_: str = Field(alias="from")
    value: str
    at: dict[str, int] = derived({}, AT)
    source: Forward | None = derived(None, "the read whose positions the value holds, where it holds any: the "
                                           "forward that made it as the source's manifest records it, with that "
                                           "entry alone and its positions under the value's dataset, the entry "
                                           "named as the value's `read` names it (`<this step>.<entry>`)")
    loaded: Any = derived(None, "the value at this point, read from the source's file into a compiled plan that "
                                "has not run (`loaded_value`), which carries it wherever the run is, in a request "
                                "too, and the run puts where the model is; never in a document or a manifest",
                          exclude=True)


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
    op's mean and each trained featurizer's scalar the body computes (an sae's only over an encode of it no decode
    closes) is that epoch's entry of `<fit>.history.<op>` and `<fit>.history.<featurizer>.<scalar>`, and
    `early_stop` is judged on it. The best epoch's featurizer tensors
    (without `early_stop`, the last's) are restored, and the fit ends with them scoring (gates hard). Each named
    substitution of `eval` is the body once more as ordinary steps after the fit (`evals`), in listed order, each
    over every row of its dataset: `{"held_out": {"train": "test"}, "train": {}}` is the body with dataset `train`
    read as `test` (its inputs, columns and scopes), then the body over the data it trained on (a dataset read as
    another renders its chats with its own `template`, so the two have the same one). The body's forwards, and an
    eval's, run in batches of `batch_size`, whatever the run's batch size, padded as the run's `padding` says
    (under `sorted`, a minibatch's own rows longest first). `epochs: 0` trains nothing: the featurizers are scored
    as drawn (an untrained lora or affine is the identity exactly, a null control), and each history is empty,
    `[0]`.

    The body is a scope of its own that sees every value produced before the fit; a trained featurizer's name
    there is `<fit>.<featurizer>`, the one being trained, which compile writes for its bare name, and whose tensors
    the body may not name: they are values once the fit ends. A value made
    before the fit with a row per example of
    one of the body's datasets (a harvest) comes into every run of the body at that run's rows, taken by their
    ids as the body's datasets are, and as it was made: on its device, in its dtype, with no gradient of its own;
    so the model runs while the fit trains only where the body has a forward. The body's datasets are those its
    forwards read (with none, the one its harvests are of): a harvest of another is refused, since a minibatch's
    ids would pair its rows with theirs. A reduction over rows inside the body is the minibatch's statistic; a
    statistic of the dataset is a step before the fit. An eval that reads a harvest's dataset as another is
    refused, since no step made the harvest over the other's rows: evaluate on other data in steps after the fit,
    through `<fit>.<featurizer>`. So is an objective none of whose terms depends on what the fit trains, which has
    no gradient to step on. Nothing leaves the body except through the fit's members: `<fit>.<featurizer>` the
    trained featurizer (saved as its bundle), `<fit>.<featurizer>.<tensor>` one of its tensors as the fit left it, a
    value laid out as its kind lays it (see Featurizer kinds), `<fit>.<eval>.<op>` and `<fit>.<eval>.<step>.<entry>` a
    value of the eval named `<eval>`, and `<fit>.history.<term>`, `[epochs]`, one entry per epoch run. A sweep over
    any of the fit's values (an objective weight, the learning rate) makes the fit, its members and their users a
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
    at: dict[str, int] = derived({}, AT)
    evals: "dict[str, dict[str, Step]]" = derived(
        {}, "each eval as the steps it runs, by label, then by body step: the body's steps compiled again at the "
            "fit's point, after it. Each dataset the substitution names is read as the one it maps to. Each body "
            "name is `<fit>.<eval>.<name>`, and each trained featurizer `<fit>.<featurizer>`. Each step and value "
            "is on the fit's axes, and a forward runs in batches of the fit's `batch_size`")

    @model_validator(mode="after")
    def _counts(self) -> "Fit":
        if isinstance(self.epochs, int) and self.epochs < 0:
            raise ValueError(f"epochs {self.epochs}: a fit runs 0 or more epochs (0 scores the untrained draw)")
        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError(f"batch_size {self.batch_size}: a minibatch holds at least one row")
        return self


KINDS = {"forward": Forward, "generate": Generate, "op": OpStep, "fit": Fit, "load": Load}


def _kind(value: Any) -> str | None:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return f"~{kind}" if kind in KINDS else None


def _points(value: Any) -> str:
    return "~points" if isinstance(value, list) else "~one"


# A `~` tag names a union branch; `PlanError`'s paths leave it out. A compiled entry that is swept is the list
# of its points.
Step = Annotated[Union[Annotated[Forward, Tag("~forward")], Annotated[Generate, Tag("~generate")],
                       Annotated[OpStep, Tag("~op")], Annotated[Fit, Tag("~fit")], Annotated[Load, Tag("~load")]],
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
                                      "of them, which preflight finds among the model's sizes; null for one used "
                                      "nowhere. For a gate over axes it is how many units the gate has, the "
                                      "product of `kept`'s lengths. An affine over layers is as wide as its "
                                      "sites' last axis. A saved bundle's header records it")
    kept: dict[str, list[int]] = derived({}, "a featurizer over axes (a gate's units, an affine's maps): the indices "
                                             "it keeps on each axis it is over, in `over`'s order (`layers` by model "
                                             "layer). Preflight finds them as the sites it is used at select them, or "
                                             "as its bundle's header records them. Empty for any other featurizer")

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
    """How a plan ran, set by `execute`: the run options (`engine`, `batch_size`, `padding`, how its batches were
    padded, one of `data.PADDINGS`, `readout`, how its forwards computed `logits`, one of `schedule.READOUTS`, each
    of which a manifest records even where it is the default, `device`, None where the engine does not place the model,
    as on an NDIF server, and every engine option after its defaults in `options`), what served it (`versions`:
    clab's version with its commit, the packages it runs on and python, where the model is: the server's, under
    `ndif`; `threads`, torch's intra-op thread count there at the run's start, which with the batch shape sets a CPU
    matmul's last bits). `dtype` and `devices` are the distinct dtypes and
    devices of the model's parameters there, in the order first met (a quantized model's
    packed weights among them, `uint8` under `nf4`; every rank's devices under `tp`). `sizes` are the model's
    (`num_layers`, `hidden_size`, `num_heads`, ...), the axes a site's selectors count among: what a saved
    bundle's header records of an axis a site keeps whole. `tokens` describes values without holding any: how
    each dataset input tokenized and what each
    generate step's rows generated (token strings per row, under the input or the step's name; a swept step's
    keyed by coordinate); a run's directory holds them in `tokens.json` beside its manifest, not in it, and
    `Plan.load` reads them back here. Where every read and write resolved is on its own record (`positions`), as a value's
    layout and labels are. `costs` holds what each step cost, by name (a fit's eval steps' under `<fit>.<eval>.<step>`),
    one entry per run the step made (a swept step's keyed by the coordinate of the point that ran, each listing in
    `serves` the coordinates of the points it served, `serving`: those one forward serves, or those an op's value is one
    for): its wall time in `seconds`, the peak `memory` allocated during it on each of the model's devices, weights
    included, in bytes by device (null where the model is not on CUDA or the run is remote), the model calls it made in
    `forwards` (each forward, each call of a generation, its prefill and every decode call, and every forward a fit's
    body makes), the input `tokens` they fed, padding included (each call's input ids: a batch's padded prompts, then
    one per row at each decode call), and of those the rows' own, `real` (a generation's rows each fed their generated
    tokens but the last, their EOS or the generation's final one).
    `started` and `seconds` are when the run began and how long it took, the weights' load left out: a shell loads
    them before the clock starts (the nnterp engine's at its first run that calls the model, ndif's under
    `remote="local"`; the hooks engine loads its weights when it loads). `stopped` is why the run ended before its last step, when it did (a CUDA out-of-memory at a step: the
    refusal, naming the step and what the run held), its `results` and `costs` those of the steps that finished."""
    engine: str
    batch_size: int
    padding: str = "dataset"
    readout: str = "full"
    device: str | None
    options: dict[str, Any] = {}
    versions: dict[str, str] = {}
    threads: int | None = None
    dtype: list[str] = []
    devices: list[str] = []
    sizes: dict[str, int] = {}
    started: str = ""
    seconds: float = 0.0
    tokens: dict[str, Any] = {}
    costs: dict[str, Any] = {}
    stopped: str | None = None


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
        if isinstance(path, dict):
            raise TypeError("Plan(dict): a document's fields are keywords, Plan(**fields); Plan(path) reads a "
                            "document file, and Plan.load(out) a run's directory")
        if path is not None:
            try:
                with open(path) as f:
                    fields = {**json.load(f), "document": os.path.abspath(path)}
            except FileNotFoundError:
                raise FileNotFoundError(f"no document at {path!r}") from None
        try:
            super().__init__(**fields)
        except ValidationError as e:
            found = []                              # each as `<document path>: <what>`
            for error in e.errors():
                path, members = located(error["loc"], fields, error["type"] == "missing")
                text = error["msg"].removeprefix("Value error, ")
                if error["type"] == "kind":
                    given = error["input"].get("kind") if isinstance(error["input"], dict) else error["input"]
                    text = f"unknown step kind {given!r}; valid: {', '.join(KINDS)}"
                # ruled out: a union's member the value's shape does not fit, refused at the member itself
                ruled_out = error["type"].endswith("_type") and bool(members) and members[-1][0] == path
                found.append({"path": path, "text": text, "members": members, "ruled_out": ruled_out,
                              "input": error.get("input")})
            # At a union, a member the value's shape rules out says nothing where another member got further; where
            # every member rules it out, one line says what the value should be.
            for point in sorted({p for f in found for p, _ in f["members"]}, key=len):
                at = [f for f in found if any(p == point for p, _ in f["members"])]
                out = [f for f in at if f["ruled_out"] and f["members"][-1][0] == point]
                if out and len(out) < len(at):
                    found = [f for f in found if f not in out]
                elif len(out) > 1:
                    forms = [re.sub(r" or instance of \w+$", "", f["text"].removeprefix("Input should be "))
                             for f in out]
                    one = {**out[0], "text": f"{out[0]['input']!r} should be {', or '.join(forms)}", "members": []}
                    found = [one if f is out[0] else f for f in found if f not in out[1:]]
            raise PlanError("\n".join(f"{f['path']}: {f['text']}" if f["path"] else f["text"]
                                      for f in found)) from None

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
        # a file, so a probe's classes are then its bundle's header's, which the run checked. A load's value is read
        # so too, into a compiled plan that has not run (compile's own, or one read back), which then carries it
        # wherever it runs; a run's plan keeps the load's record alone.
        files: dict[tuple[str, str], tuple[list[str], Any]] = {}
        for entry in self.steps.values() if self.compiled and self.run is None else []:
            for s in points(entry):
                if isinstance(s, Load) and s.loaded is None:
                    s.loaded = loaded_value(self, s, files)
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
        """The plan in one line: its name and its state (uncompiled; compiled, with its points; or run, with its
        engine and how many results it holds), which `str` and `print` give too, never its records' fields or a
        bundle's tensors."""
        if self.run is not None:
            return f"Plan({name(self)}, run on {self.run.engine}, {len(self.results)} results)"
        if self.compiled:
            n = math.prod(len(axis.values) for axis in self.sweeps.values())
            return f"Plan({name(self)}, compiled, {n} point{'s' * (n != 1)})"
        return f"Plan({name(self)}, uncompiled)"

    __str__ = __repr__

    def compile(self) -> "Plan":
        """This plan compiled: rows loaded (a hub dataset's checked, then left out), sites and named interventions
        inlined, bundles loaded, sweeps expanded into points, every name resolved and every value's layout and
        dataset known."""
        from clab.compile import compile
        return compile(self)

    def dump(self, out: str) -> None:
        """A run's output directory: one file per save, `manifest.json` and `tokens.json` (each input's tokens per
        row, `run.tokens`, which `load` reads back)."""
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


def loaded_value(plan: Plan, step: Load, files: dict[tuple[str, str], tuple[list[str], Any]]) -> Any:
    """Load `step`'s value at its point, as the file of its run's directory under the corpus root holds it (the
    output directory's one reader, `output.read_saved_value`, once per value into `files`, with the axes its record
    names): a swept value's point is the one keyed by the point's coordinate on those axes, spelled in the order of the
    run that saved it, whose labels and values are this plan's sweeps (compile checked them)."""
    from clab.output import manifest, read_saved_value          # the output directory's one reader, a later stage's
    if (step.from_, step.value) not in files:
        out = os.path.join(corpus_root(plan), step.from_)
        d = manifest(out)[1][step.value]
        files[step.from_, step.value] = d["axes"], read_saved_value(out, step.value, d)
    axes, value = files[step.from_, step.value]
    return value[coordinate_spelling({a: plan.sweeps[a].values for a in axes}, {a: step.at[a] for a in axes})] \
        if axes else value


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


def produced(steps: dict[str, Entry], declared: dict[str, FeaturizerEntry]) -> list[str]:
    """Every value `steps` produce, by name, in order: an op step's or a generation's own, a `top_k` step's members
    (`<step>.ids`, ...: `top_k_members`), each entry's `<step>.<entry>`, and a fit's members (`<fit>.history.<term>`,
    each trained featurizer's tensors, `<fit>.<featurizer>.<tensor>`, and its eval members, `<fit>.<eval>.<member>`);
    not a trained featurizer, which is no value. `declared` are the plan's featurizers, whose kinds say the scalars a
    fit's history holds (those its body computes: an sae's over its `open_encodes`) and the tensors each holds."""
    out: list[str] = []
    for name, entry in steps.items():
        step = points(entry)[0]
        if step.kind != "fit":
            out += [name] if step.kind != "forward" else []
            out += [f"{name}.{m}" for m in top_k_members(step)] if top_k_of(step) is not None else []
            out += [f"{name}.{k}" for k in entries(step)]
            continue
        out += [f"{name}.history.{n}" for n, s in step.steps.items() if s.kind == "op"]
        kinds = {f: featurizers.KINDS[points(declared[f])[0].kind] for f in step.train}
        out += [f"{name}.history.{f}.{scalar}" for f in step.train for scalar in kinds[f].scalars
                if scalar not in kinds[f].over_encodes or open_encodes(step.steps, f"{name}.{f}")]
        out += [f"{name}.{f}.{t}" for f in step.train for t in kinds[f].held(points(declared[f])[0].bundle)]
        out += [f"{name}.{label}.{member}" for label in step.eval for member in produced(step.steps, declared)]
    return out


def trained_featurizers(steps: dict[str, Entry]) -> list[str]:
    """Every featurizer a fit among `steps` trains, by the name the steps after the fit use: `<fit>.<featurizer>`."""
    return [f"{n}.{f}" for n, e in steps.items() for s in points(e)[:1] if s.kind == "fit" for f in s.train]


def tensor_of(steps: dict[str, Entry], declared: dict[str, FeaturizerEntry], ref: str) -> tuple[str, str] | None:
    """The featurizer and the tensor `ref` names, where it names a featurizer's tensor: `<featurizer>.<tensor>`, one of
    the plan's featurizers (`declared`) as drawn or loaded, or `<fit>.<featurizer>.<tensor>`, one a fit among `steps`
    trains, as the fit left it; the featurizer as the name writes it (`probe`, `fit.probe`). None for any other
    name. The tensor is whatever the name says, which compile holds to those the kind holds (`Kind.held`)."""
    thing, _, member = ref.partition(".")
    if thing in declared:
        return (thing, member) if member else None
    step = points(steps[thing])[0] if thing in steps else None
    label, _, rest = member.partition(".")
    return (f"{thing}.{label}", rest) if step is not None and step.kind == "fit" and label in step.train and rest \
        else None


def tensors_named(plan: "Plan") -> set[str]:
    """Every featurizer's tensor (`tensor_of`) the plan names: an op step's `of`, an op's argument, a featurizer's
    field (a constant's `init`) or a save; those the run puts among its values, each a copy."""
    names = [*plan.saves, *(v for e in plan.featurizers.values() for f in points(e)
                            for v in (f.model_extra or {}).values())]
    for _, step in walk(plan.steps):
        listed = [*([step] if step.kind == "op" else []), *(o for e in entries(step).values() for o in e.ops)]
        names += [v for o in listed for v in [getattr(o, "of", None), *arguments(o).values()]]
    return {n for n in names if isinstance(n, str) and tensor_of(plan.steps, plan.featurizers, n) is not None}


def top_k_of(record: Any) -> "Op | OpStep | None":
    """The `top_k` whose whole value a record's is: an op step's own, or an entry's last op; None for any other value.
    That value is several tensors (`ids`, one tensor per axis over a list of them, beside `probs` or `values`, and as
    a step over `vocab_size` the picks' `tokens`), which no op takes whole."""
    if isinstance(record, OpStep):
        return record if record.op == "top_k" else None
    if isinstance(record, Intervention) and record.ops and record.ops[-1].op == "top_k":
        return record.ops[-1]
    return None


def top_k_members(step: "OpStep") -> list[str]:
    """The members of a `top_k` step's value, each a value of its own a later step names as `<step>.<member>`, as it
    names a fit's: `ids`, the picks' indices (over a list of axes an object of one tensor per axis, each `ids.<axis>`
    beside it), `probs` over a vocabulary or else `values`, and over `vocab_size` the picks' `tokens`, text. Each is
    laid out as the step's value is, with its labels."""
    over = arguments(step).get("over", ops.transforms.RANKED)
    axes = over if isinstance(over, list) else [over]
    return ["ids", *(f"ids.{a}" for a in axes if isinstance(over, list)),
            "probs" if set(axes) & set(ops.VOCABULARY) else "values",
            *(["tokens"] if ops.transforms.RANKED in axes else [])]


def member_of(steps: dict[str, Entry], ref: str, at: dict[str, int]) -> "tuple[OpStep, str] | None":
    """The `top_k` step whose value's member `ref` names (`<step>.<member>`, `top_k_members`), at the point agreeing
    with `at`, and the member; a fit's eval step's too (`<fit>.<eval>.<step>.<member>`). None for any other name."""
    thing, _, member = ref.partition(".")
    if thing not in steps or not member:
        return None
    step = agreeing_point(steps[thing], at)
    if step.kind == "fit":
        label, _, rest = member.partition(".")
        return member_of(step.evals[label], rest, step.at) if label in step.evals else None
    return (step, member) if top_k_of(step) is not None and member in top_k_members(step) else None


def walk(steps: dict[str, Entry], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every step and point, a fit's body steps after the fit, each with its dotted name."""
    for name, entry in steps.items():
        for step in points(entry):
            yield prefix + name, step
            if step.kind == "fit":
                yield from walk(step.steps, f"{prefix}{name}.")


def calls_model(steps: dict[str, Entry]) -> bool:
    """Whether a run of `steps` needs the model's weights: a forward or a generation, at the root or in a fit's body,
    or an op step reading the model's own head (`readout`) or its generation config's stop ids (`detokenize`). A
    document of loads and other op steps needs the tokenizer and the config alone."""
    return any(step.kind in ("forward", "generate") or step.kind == "op" and step.op in ("readout", "detokenize")
               for _, step in walk(steps))


def declared(featurizers_of: dict[str, FeaturizerEntry], ref: str) -> str:
    """The declared featurizer a reference names, among the plan's (`featurizers_of`): itself, or the one
    `<fit>.<featurizer>` trains."""
    return ref if ref in featurizers_of else ref.partition(".")[2]


def kind_of(plan: Plan, ref: str) -> str:
    """The kind of the featurizer a reference names."""
    return points(plan.featurizers[declared(plan.featurizers, ref)])[0].kind


def adapts(plan: Plan, o: "Op | OpStep") -> bool:
    """Whether op `o` is an `apply` of a featurizer whose `with` is its input, which may be read at another site
    (an adapter's: `Kind.adapts`), rather than an operand of the value (a gate's, which it interpolates toward)."""
    ref = arguments(o).get("featurizer")
    return o.op == "apply" and ref is not None and featurizers.KINDS[kind_of(plan, ref)].adapts


def over_of(f: Featurizer) -> list[str] | None:
    """The axes a featurizer is over (a gate's units, an affine's maps): its `over`, or, where a loading document
    leaves it out, its bundle's header's; None for a gate on the last axis of its site's selection, an affine with
    one map, and any other featurizer."""
    given = (f.model_extra or {}).get("over")
    return given if given is not None or f.header is None or "over" not in f.header else json.loads(f.header["over"])


def featurizer_uses(steps: dict[str, Entry], featurizers_of: dict[str, FeaturizerEntry]
                    ) -> Iterator[tuple[str, str, Intervention, dict[str, int], "Op | OpStep", list[Op]]]:
    """Every `encode` and `apply` of a featurizer among `steps`, in the order they are written (a fit's body after the
    fit): the featurizer among the plan's (`featurizers_of`) it reaches (`declared`), the entry's dotted name, the read
    it is of (the entry's own record; an op step's the read its `of` names, whose site and coordinates its featurizer
    is used at), its step's point, the op, and the encodes still open around it (outermost first), whose coordinates
    it is applied to."""
    for name, step in walk(steps):
        ref = arguments(step).get("featurizer") if step.kind == "op" else None
        if ref is not None:                 # a body's step sees the body's steps beside the root's
            fit = name.rpartition(".")[0]
            scope = {**steps, **agreeing_point(steps[fit], step.at).steps} if fit else steps
            read = entry_of(scope, cast(str, step.read), step.at)[1]
            yield (declared(featurizers_of, ref), name, read, step.at, step,
                   [Op(op="encode", **s) for s in read.space])
        for key, entry in entries(step).items():
            for node, around in visited(bracketed(entry.ops)):
                ref = arguments(node["op"]).get("featurizer")
                if ref is not None:
                    yield (declared(featurizers_of, ref), f"{name}.{key}", entry, step.at,
                           node["op"], [a["op"] for a in around])


def tensor_record(steps: dict[str, Entry], featurizers_of: dict[str, FeaturizerEntry], featurizer: str, tensor: str,
                  at: dict[str, int], axes: list[str]) -> Labelled | None:
    """The record of tensor `tensor` of `featurizer` (as a name writes it: one of the plan's, `featurizers_of`, or
    `<fit>.<featurizer>`, one a fit trains), at the point agreeing with `at`, on the sweep axes `axes`: laid out as
    its kind lays the tensor (`Kind.laid`) at the first use of the featurizer among `steps` (`featurizer_uses`) whose
    point agrees with `at`. Its `width` is that use's last axis (inside an encode, that encode's coordinates), with
    the indices the use's site keeps on it; an axis the featurizer is over is labelled as the use's selection is
    (`layers` by model layer); and it is in the coordinates of the encodes open around the use, then of the
    featurizer's own where the kind's coordinate axis is its last. None where no step among `steps` uses it."""
    name = declared(featurizers_of, featurizer)
    f = agreeing_point(featurizers_of[name], at)
    kind = featurizers.KINDS[f.kind]
    use = next((u for u in featurizer_uses(steps, featurizers_of)
                if u[0] == name and all(at.get(k, i) == i for k, i in u[3].items())), None)
    if use is None:
        return None
    _, _, read, _, _, around = use
    if around:
        outer = arguments(around[-1])["featurizer"]
        last = featurizers.KINDS[points(featurizers_of[declared(featurizers_of, outer)])[0].kind].axis
    else:
        last = address.dims(cast(Site, read.site).component)[-1]
    over, laid = over_of(f) or [], kind.laid[tensor]
    # Over axes, a stacked kind's tensors gain them first, and a gate's units lie on them in place of `width`.
    dims = [*(over if kind.stacked else []), *(a for axis in laid for a in (
        over if over and axis == "width" and not kind.stacked else [last] if axis == "width" else [axis]))]
    kept = {a: read.kept[a] for a in [*([] if around else [last]), *over] if a in read.kept and a in dims}
    space = [{k: v for k, v in arguments(o).items() if k in ("featurizer", "select")} for o in around]
    return Labelled(dims=dims, layers=read.layers if "layers" in dims else None, kept=kept, axes=axes,
                    space=space + ([{"featurizer": featurizer}] if laid[-1] == kind.axis else []))


def entry_of(steps: dict[str, Entry], ref: str, at: dict[str, int]) -> tuple[Forward, Intervention]:
    """The step point and the entry `<step>.<entry>` names among `steps` (or `<fit>.<eval>.<step>.<entry>`, among
    that fit's eval steps; or a load's read, the forward and entry it carries), at the step's point that agrees
    with `at`: the one place a reference to an entry resolves, so a swept site resolves at every point."""
    thing, _, member = ref.partition(".")
    step = agreeing_point(steps[thing], at)
    if step.kind == "fit":
        label, _, rest = member.partition(".")
        return entry_of(step.evals[label], rest, step.at)
    if step.kind == "load":
        assert step.source is not None, f"{ref} names no read"
        return step.source, entries(step.source)[member]
    return step, entries(step)[member]


def record_of(steps: dict[str, Entry], ref: str, at: dict[str, int], declared: dict[str, FeaturizerEntry]) -> Labelled:
    """The record of the value `ref` names among `steps`, at the point agreeing with `at`, which carries its
    labelled layout: an entry's (`<step>.<entry>`, a load's read among them), an op step's, a generation's or a
    load's own (`<step>`), a `top_k` step's member's (`<step>.ids`, `member_of`: the step's labels), a fit
    member's: an eval step's value (`<fit>.<eval>.<member>`, its record among the fit's `evals`), a term's history
    (`<fit>.history.<term>`, `[epochs]`), a trained featurizer (no axes) or one of its tensors
    (`<fit>.<featurizer>.<tensor>`, on the fit's axes), or a tensor of one of the plan's featurizers, `declared`
    (`<featurizer>.<tensor>`, on the featurizer's axes), each laid out at the featurizer's first use among `steps`
    (`tensor_record`). Every caller passes the plan's featurizers, which a tensor's layout needs."""
    thing, _, member = ref.partition(".")
    if thing in declared:
        found = tensor_record(steps, declared, thing, member, at, list(agreeing_point(declared[thing], at).at))
        assert found is not None, f"{ref}: no step uses {thing!r}, whose site its tensors are laid out at"
        return found
    step = agreeing_point(steps[thing], at)
    if step.kind == "op" and member:
        assert member_of(steps, ref, at) is not None, f"{ref} names no value"
        return Labelled(**{f: getattr(step, f) for f in Labelled.model_fields})
    if step.kind != "fit":
        host = step.source if step.kind == "load" else step
        found = entries(host)[member] if member else step
        assert isinstance(found, Labelled), f"{ref} names no value"
        return found
    label, _, rest = member.partition(".")
    if label == "history" or label in step.train and not rest:
        return Labelled(dims=["epochs"] if label == "history" else [], axes=list(step.at))
    if label in step.train:
        found = tensor_record(steps, declared, f"{thing}.{label}", rest, step.at, list(step.at))
        assert found is not None, f"{ref}: no step uses {thing}.{label}, whose site its tensors are laid out at"
        return found
    return record_of(step.evals[label], rest, step.at, declared)


def projections(plan: Plan, name: str) -> list[tuple[dict[str, int], Step]]:
    """The points of the value `name`, in order: each point of its step projected onto the value's own axes
    (`Labelled.axes`), each once, with the first point of the step it is the projection of; `[({}, <the step>)]`
    for a value that depends on no axis."""
    found: dict[tuple, tuple[dict[str, int], Step]] = {}
    for p in points(producer(plan, name)):
        at = projection(p.at, record_of(plan.steps, name, p.at, plan.featurizers).axes)
        found.setdefault(tuple(at.items()), (at, p))
    return list(found.values())


def saved_value_axes(plan: Plan, name: str) -> list[str]:
    """The sweep axes a saved value depends on, as its record says (at its step's first point)."""
    return list(record_of(plan.steps, name, points(producer(plan, name))[0].at, plan.featurizers).axes)


def producer(plan: Plan, name: str) -> Any:
    """The entry the first part of value `name` names: a step, or one of the plan's featurizers, whose tensors are
    values (`<featurizer>.<tensor>`)."""
    thing = name.partition(".")[0]
    return plan.featurizers[thing] if thing in plan.featurizers else plan.steps[thing]


def runnable(steps: dict[str, Entry]) -> dict[str, Entry]:
    """The steps that run, by name, in the order they run: each of `steps`, and after a fit each of its evals'
    steps under `<fit>.<eval>.<step>`, a point per point of the fit."""
    out: dict[str, Entry] = {}
    for name, entry in steps.items():
        out[name] = entry
        fits = points(entry)
        for label, evaluated in fits[0].evals.items() if fits[0].kind == "fit" else ():
            for n in evaluated:
                each = [f.evals[label][n] for f in fits]
                out[f"{name}.{label}.{n}"] = each if isinstance(entry, list) else each[0]
    return out


def made_by(plan: Plan, name: str) -> str:
    """The step that makes value `name`, by the name `runnable` gives it: an eval's member's its eval step
    (`fit.held_out.ce`'s itself, `fit.held_out.adapted.logits`'s `fit.held_out.adapted`), any other value's the
    step its first part names; a tensor of one of the plan's featurizers (`<featurizer>.<tensor>`), which no step
    makes, the featurizer, which the run makes before its first step."""
    thing, _, member = name.partition(".")
    if thing in plan.featurizers:
        return thing
    label, _, rest = member.partition(".")
    step = points(plan.steps[thing])[0]
    return f"{thing}.{label}.{rest.partition('.')[0]}" if step.kind == "fit" and label in step.evals else thing


def serving(plan: Plan, name: str) -> list[tuple[Step, list[Step]]]:
    """The points of step `name` (a name `runnable` lists: a fit's eval step's too) that run, in order, each with
    the points its run serves, itself first: a forward's or a generation's each point whose `schedule` holds no
    `serving`, with the points whose `serving` names it; an op's or a load's the points of its value
    (`projections`: one per distinct point on its `axes`), each with the step's points that agree with it there; a
    fit's each point, itself alone. Found in one pass over the points, each joining the run its coordinates name
    (an op's or a load's on its axes, a forward's its `serving` or its own), whose first point is the one that
    runs."""
    steps = points(runnable(plan.steps)[name])
    if steps[0].kind == "fit":
        return [(p, [p]) for p in steps]
    runs: dict[tuple, list[Step]] = {}
    for q in steps:
        at = projection(q.at, q.axes) if q.kind in ("op", "load") else q.schedule.get("serving", q.at)
        runs.setdefault(tuple(at.items()), []).append(q)
    return [(served[0], served) for served in runs.values()]


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
    """Dataset `name`'s rows as the plan holds them, which the run's plan does (`steps.plan_with_rows`); one that holds
    none (a hub dataset's, a loaded plan's file rows) is refused by name."""
    rows = plan.data[name].rows
    if rows is None:
        fail(f"data.{name}", "the plan holds none of its rows; `compile.dataset_rows` loads them, as the run does "
             "first (`steps.plan_with_rows`)")
    return rows


def body_view(plan: Plan, fit: str, at: dict[str, int]) -> Plan:
    """The plan as fit `fit`'s body sees it at the fit's point `at`: the body's steps beside the root's, which
    they shadow."""
    return plan.model_copy(update={"steps": {**plan.steps, **agreeing_point(plan.steps[fit], at).steps}})


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
