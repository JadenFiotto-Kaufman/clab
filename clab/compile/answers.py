"""Answers: a dataset column scored against a value's vocabulary (the tokenizer's against logits, or a probe's
classes), each row's answer tokenized as written and lined up with the read it scores; and a column read as text."""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import ValidationError
from torch import Tensor

from clab import ops
from clab.compile.state import Compiling, Found
from clab.format import data
from clab.ops.metrics import ONE_ID
from clab.plan import Column, Plan, declared, fail, held_rows

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def text_ref(cx: Compiling, value: object, path: str, dataset: str | None) -> str:
    """A column reference (`"<dataset>.<column path>"`, or `{"column", "strip"?, "lower"?}`) naming text on
    every row, of `dataset` (the step's) when one is given: a string, or a non-empty list of them (a set).
    Returns its dataset."""
    if isinstance(value, dict):
        try:
            Column(**value)
        except ValidationError as e:
            fail(path, "; ".join(err["msg"] if not err["loc"] else f"{err['loc'][0]}: {err['msg']}"
                                 for err in e.errors()))
    ref = data.column_of(value)
    if ref is None:
        fail(path, f"{value!r} is not a column: '<dataset>.<column>' or {{'column': '<dataset>.<column>', "
                   "'strip'?: bool, 'lower'?: bool}")
    name, _, column = ref[0].partition(".")
    column_ref(cx, f"{name}.{column.split('[')[0].split('.')[0]}", path, dataset)
    for i, row in enumerate(cx.rows[name]):
        try:
            found = data.lookup(row, column)
        except (KeyError, IndexError, TypeError):
            fail(path, f"{ref[0]!r} is not on row {i} of {name!r}")
        if not (isinstance(found, str) or isinstance(found, list) and found and all(isinstance(f, str) for f in found)):
            fail(path, f"{ref[0]!r} is {type(found).__name__} on row {i} of {name!r}; a column read as text is a "
                       "string, or a list of them (a set)")
    return name


def column_ref(cx: Compiling, value: object, path: str, dataset: str | None) -> str:
    """`<dataset>.<column>`, a column present on every row; of `dataset` when one is given. Returns the dataset."""
    name, _, column = value.partition(".") if isinstance(value, str) else ("", "", "")
    if name not in cx.rows:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx.rows)}")
    if not column:
        fail(path, f"{name!r} names a dataset, not a column reference, which is '<dataset>.<column>'; its columns: "
             f"{', '.join(cx.rows[name][0])}")
    if any(column not in r for r in cx.rows[name]):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(cx.rows[name][0])}")
    if dataset is not None and name != dataset:
        fail(path, f"{value!r} is a column of {name!r}, and the value it goes with has a row per example of "
             f"{dataset!r}")
    return name


def check_tokens(engine: Engine, plan: Plan, found: Found, where: str, name: str, args: dict, dims: list[str] | None,
                 placed: list | None, forced: list[list[int]] | None) -> None:
    """Every column op `name` scores against logits, resolved once into `found.answers` (`answer`): a class or
    a set of several ids is refused where the op reads one id's logit. A column with an answer of several
    tokens lines every row up with the read, one position per token: the value has a pos axis (when its axes
    `dims` are known) and, where the read's positions are known before the run (`placed`, per row), each row
    the read reaches has as many (a row it does not reach, the marker, is missing, not refused). Where the read
    is a scope that is `before` on the step's own input (`forced`: each row's input ids), the input's next token
    at each position is among the ids the answer's token there may be."""
    for key, p in ops.parameters(name).items():
        if p.annotation != ops.Tokens or key not in args:
            continue
        got = answer(engine, plan, found, args[key], where)
        values, sets, column, aligned = got.values, got.sets, got.column, got.aligned
        strip = (data.column_of(args[key]) or ("", False, False))[1]
        for r, (value, row) in enumerate(zip(values, sets)):
            if name in ONE_ID and any(len(ids) > 1 for ids in row):
                raise ValueError(f"{where}: {column} row {r}: {value!r} stands for {max(map(len, row))} ids (a set, "
                                 f"or a class under strip or lower), which have no one logit for {name}; "
                                 "token_prob scores them together")
            if aligned and dims is not None and "pos" not in dims:
                raise ValueError(f"{where}: {column} row {r}: {value!r} is {len(row)} token"
                                 f"{'s' * (len(row) != 1)}, a position each, and the value has no pos axis ({dims})")
            if aligned and placed is not None and isinstance(placed[r], list) and len(placed[r]) != len(row):
                spaced = blank(engine.tokenizer, value, strip, len(row), placed[r], None if forced is None else forced[r])
                raise ValueError(f"{where}: {unaligned(column, r, value, len(row), len(placed[r]))}{spaced}")
        if forced is None or placed is None:
            continue
        for r, at in enumerate(placed):
            for i, p in enumerate(at if isinstance(at, list) else []):
                ids, next_ = sets[r][i if aligned else 0], forced[r][p + 1]
                if next_ not in ids:
                    shown = engine.tokenizer.convert_ids_to_tokens([next_])[0]
                    raise ValueError(f"{where}: {column} row {r}: position {p} predicts {shown!r} in the input, which "
                                     f"is not token {i if aligned else 0} of {values[r]!r} as it tokenizes alone "
                                     f"({engine.tokenizer.convert_ids_to_tokens(ids[:3])}); the answer is scored as "
                                     "it tokenizes, so the input says it the same way")


@dataclass
class Answer:
    """A column reference's answers, resolved once per findings: each row's `values`, per token the ids it may
    be (`sets`), the `column`, and whether its rows line up with the read (`aligned`: some row is several
    tokens; else every row is one, which scores every position of its row)."""
    values: list
    sets: list
    column: str
    aligned: bool


def answer(engine: Engine, plan: Plan, found: Found, ref: str | dict, where: str) -> Answer:
    """The `Answer` of a column reference, resolved into `found.answers` the first time it is asked for."""
    key = refkey(ref)
    if key not in found.answers:
        column, strip, lower = data.column_of(ref) or ("", False, False)
        values = column_texts(plan, column, None)
        try:
            sets = answers(engine.tokenizer, values, column, strip, lower)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        found.answers[key] = Answer(values, sets, column, any(len(row) > 1 for row in sets))
    return found.answers[key]


def check_classes(plan: Plan, where: str, name: str, args: dict, probe: str, classes: dict, swap: dict) -> None:
    """Every column op `name` scores against a probe's class list holds only those classes, on every row of the
    dataset it reads (named as the run reads it: under eval, the substituted one)."""
    vocabulary = classes[declared(plan, probe)]
    for key, p in ops.parameters(name).items():
        if p.annotation == ops.Tokens and key in args:
            ref, strip, lower = data.column_of(args[key]) or ("", False, False)
            if strip or lower:
                raise ValueError(f"{where}: {key} names a class of probe {declared(plan, probe)!r}, which is compared "
                                 "exactly; strip and lower compare an answer's tokens")
            dataset, _, column = ref.partition(".")
            for i, row in enumerate(held_rows(plan, dataset)):
                if isinstance(row[column], list) or row[column] not in vocabulary:
                    raise ValueError(f"{where}: {key} is {column!r} of {swap.get(dataset, dataset)!r}, whose row {i} "
                                     f"is {row[column]!r}, not one of the classes of probe {declared(plan, probe)!r}: "
                                     f"{vocabulary}")


def indexed(plan: Plan, column: str, vocabulary: list, ids: list[int] | None) -> Tensor:
    """A column's values as their indices in a probe's class list."""
    return torch.tensor([vocabulary.index(v) for v in column_texts(plan, column, ids)])   # each one, preflight checked


def column_texts(plan: Plan, column: str, ids: list[int] | None) -> list:
    """A column's value on each of its dataset's rows `ids` (every one: None), a string or a list of them."""
    name, _, path = column.partition(".")
    return [data.lookup(r, path) for r in data.by_id(held_rows(plan, name), ids)]


def refkey(ref: str | dict) -> str:
    """A column reference as the key its resolved answers are found under."""
    return json.dumps(ref, sort_keys=True)


def answered(answer: Answer, of: Tensor | None, dims: list[str], ids: list[int] | None) -> Tensor:
    """Each row's answer (of dataset rows `ids`; every one, 0 to n-1 in order: None) laid against the value `of` a
    metric scores, as the token ids each position may be, padded with -1 (`metrics._aligned`). A column whose every
    row is one token scores it at every position of its row (`[rows, m]`); one with an answer of several tokens is
    `aligned`: token i of each row's at the row's position i (`[rows, pos, m]`), which needs every row the value
    reaches to have as many positions as its answer has tokens (`present`)."""
    column, chosen = answer.column, data.by_id(answer.sets, ids)
    most = max(len(t) for row in chosen for t in row)
    sets = [[t + [-1] * (most - len(t)) for t in row] for row in chosen]
    if not answer.aligned:
        return torch.tensor([row[0] for row in sets])
    if "pos" not in dims:
        raise ValueError(f"{column}: an answer of several tokens scores a position each, and the value has no pos "
                         f"axis ({dims})")
    assert of is not None, "only a write's ops are bound without the value, and with no dims"
    width = of.shape[dims.index("pos")]
    for r, value, row, n in zip(ids if ids is not None else range(len(sets)), data.by_id(answer.values, ids), sets,
                                present(of, dims)):
        if n != len(row) and n != 0:                    # a row the value does not reach is missing, not refused
            raise ValueError(unaligned(column, r, value, len(row), n))
    # `[rows, pos, m]` whatever `width` is: over a read no row reached it is 0, and the list alone would be `[rows, 0]`.
    return torch.tensor([[row[i] if i < len(row) else [-1] * most for i in range(width)]
                         for row in sets]).reshape(len(sets), width, most)


def present(of: Tensor, dims: list[str]) -> list[int]:
    """How many positions each row of `of` holds: those not NaN all the way (a row the value does not reach,
    or a position past a row's own, is NaN); 0 on every row of a value no row reached, which holds no position."""
    x = of.movedim((dims.index("rows"), dims.index("pos")), (0, 1))
    # The trailing size spelled out: a reshape with -1 is ambiguous on a value of no elements (a `pos` axis of 0).
    return (~x.reshape(*x.shape[:2], math.prod(x.shape[2:])).isnan().all(-1)).sum(1).tolist()


def unaligned(column: str, row: int, value: object, k: int, n: int) -> str:
    """Why a row's answer does not line up with the read a metric scores."""
    return (f"{column} row {row}: {value!r} is {k} token{'s' * (k != 1)}; the read has {n} position"
            f"{'s' * (n != 1)} on that row; "
            "token i of an answer scores the read's position i, so the read carries one per token (a scope with "
            "\"before\": true reads the positions that predict the answer's tokens)")


def blank(tokenizer: PreTrainedTokenizerBase, value: str | list[str], strip: bool, k: int, at: list[int],
          ids: list[int] | None) -> str:
    """What to do where a whitespace-only token is what an answer of `k` tokens and its read at positions `at`
    differ by: the answer's first, which it keeps and a scope under strip does not cover; or, where the read is
    `before` on the step's input (its `ids`), the token its first position predicts, which the answer lacks."""
    texts = vocabulary(tokenizer)
    first = trimmed(tokenizer, tokenizer(value[0] if isinstance(value, list) else value,
                                         add_special_tokens=False)["input_ids"], strip)[0]
    if k > len(at) and not texts[first].strip():
        return (f"; its token 0, {texts[first]!r}, is whitespace only, which an answer keeps and a scope under strip "
                "does not cover: scope the column without strip")
    if ids is not None and len(at) > k and not texts[ids[at[0] + 1]].strip():
        return (f"; the read's position {at[0]} predicts {texts[ids[at[0] + 1]]!r}, whitespace only, which the "
                "answer has no token for: write the answer with its space")
    return ""


def answers(tokenizer: PreTrainedTokenizerBase, values: list, column: str, strip: bool,
            lower: bool) -> list[list[list[int]]]:
    """Each row's answer as, per token, the ids it may be. A row's value is a string or a set of them (a list),
    each tokenized exactly as written without special tokens (`" Friday"` and `"Friday"` are different tokens
    on most tokenizers; the column says which), and under `strip` without its whitespace-only tokens at the end
    (`trimmed`); a set's members are one width, and its token i is any of theirs. Each token stands for its
    class (`spelled`). Each position's ids are in first-met order, each once."""
    out = []
    for row, value in enumerate(values):
        members = value if isinstance(value, list) else [value]
        tokenized = [trimmed(tokenizer, tokenizer(m, add_special_tokens=False)["input_ids"], strip) for m in members]
        if len({len(t) for t in tokenized}) > 1 or not tokenized[0]:
            empty = f"{value!r} is no tokens" + (" once strip drops the whitespace-only tokens at its end" if strip
                                                 else "")
            raise ValueError(f"{column} row {row}: " + (empty if not tokenized[0] else
                             "its members are " + ", ".join(f"{m!r} {len(t)}" for m, t in zip(members, tokenized))
                             + " tokens; a set's members are one width, the positions its read aligns"))
        out.append([list(dict.fromkeys(i for t in tokenized for i in spelled(tokenizer, t, k, strip, lower)))
                    for k in range(len(tokenized[0]))])
    return out


def trimmed(tokenizer: PreTrainedTokenizerBase, ids: list[int], strip: bool) -> list[int]:
    """`ids` without their whitespace-only tokens at the end under `strip` (`" New York "`: `▁New ▁York`), since
    what follows an answer is not scored; a whitespace-only token at the start is kept (`" 42"`: `▁ 4 2`)."""
    texts = vocabulary(tokenizer)
    while strip and ids and not texts[ids[-1]].strip():
        ids = ids[:-1]
    return ids


def spelled(tokenizer: PreTrainedTokenizerBase, ids: list[int], at: int, strip: bool, lower: bool) -> list[int]:
    """The ids token `at` of an answer's `ids` (`trimmed`) stands for: the one place a token's class is decided.
    Its class is every id whose text compares the same (`classed`): under `strip` the answer's first token
    without its leading whitespace and its last without its trailing (a one-token answer's without either),
    under `lower` lowercased. A token between the ends compares with its whitespace; under `strip` a
    whitespace-only token, and anywhere a token that is part of a character (its text is U+FFFD), stands for
    itself, as every token does without a flag."""
    text = vocabulary(tokenizer)[ids[at]]
    lead, trail = strip and at == 0, strip and at == len(ids) - 1
    if "\ufffd" in text or strip and not text.strip() or not (lead or trail or lower):
        return [ids[at]]
    return spellings(tokenizer, lead, trail, lower)[classed(text, lead, trail, lower)]


def classed(text: str, lead: bool, trail: bool, lower: bool) -> str:
    """A token's text as its class compares it: without its leading whitespace under `lead`, its trailing under
    `trail`, lowercased under `lower` (`data.compared`)."""
    text = text.lstrip() if lead else text
    return data.compared(text.rstrip() if trail else text, False, lower)


@functools.cache
def vocabulary(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Each id's text: what it decodes to after another token (`a`'s first), so a leading space (` Positive`,
    `▁Positive`) is its own. Made once per tokenizer."""
    anchor = tokenizer("a", add_special_tokens=False)["input_ids"][0]
    lead = len(tokenizer.decode([anchor]))
    return [text[lead:] for text in tokenizer.batch_decode([[anchor, i] for i in range(len(tokenizer))])]


@functools.cache
def spellings(tokenizer: PreTrainedTokenizerBase, lead: bool, trail: bool, lower: bool) -> dict[str, list[int]]:
    """Every id of the vocabulary but a part of a character by its text as its class compares it (`classed`):
    the classes an answer's tokens stand for. Made once per tokenizer and flags."""
    found: dict[str, list[int]] = {}
    for i, text in enumerate(vocabulary(tokenizer)):
        if "\ufffd" not in text:
            found.setdefault(classed(text, lead, trail, lower), []).append(i)
    return found
