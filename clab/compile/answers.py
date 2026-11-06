"""Answers: a dataset column scored against a value's vocabulary (the tokenizer's against logits, or a probe's
classes), each row's answer tokenized as written and lined up with the read it scores; and a column read as text."""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from pydantic import ValidationError
from torch import Tensor

from clab import ops
from clab.compile.state import Compiling, Findings
from clab.format import data
from clab.ops.metrics import ONE_ID
from clab.plan import Column, Plan, declared, fail, held_rows

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def check_text_ref(cx: Compiling, value: object, path: str, dataset: str | None) -> str:
    """A column reference (`"<dataset>.<column path>"`, or `{"column", "strip"?, "lower"?}`) naming text on
    every row, of `dataset` (the step's) when one is given: a string, or a non-empty list of them (a set).
    Returns its dataset."""
    if isinstance(value, str) and value.partition(".")[0] not in cx.rows:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx.rows)}; text "
                   f"written as it is, the same on every row, is {{'text': {value!r}}}")
    if isinstance(value, dict):
        try:
            Column(**value)
        except ValidationError as e:
            said = [(err["loc"], err["msg"].removeprefix("Value error, ")) for err in e.errors()]
            fail(path, "; ".join(f"{loc[0]}: {msg}" if loc else msg for loc, msg in said))
    ref = data.column_of(value)
    if ref is None:
        fail(path, f"{value!r} is not a column: '<dataset>.<column>' or {{'column': '<dataset>.<column>', "
                   "'strip'?: bool, 'lower'?: bool}")
    name, _, column = ref[0].partition(".")
    check_column_ref(cx, f"{name}.{column.split('[')[0].split('.')[0]}" if column else ref[0], path, dataset)
    for i, row in enumerate(cx.rows[name]):
        try:
            cell = data.lookup(row, column)
        except (KeyError, IndexError, TypeError):
            fail(path, f"{ref[0]!r} is not on row {i} of {name!r}")
        if not (isinstance(cell, str) or isinstance(cell, list) and cell and all(isinstance(f, str) for f in cell)):
            fail(path, f"{ref[0]!r} is {type(cell).__name__} on row {i} of {name!r}; a column read as text is a "
                       "string, or a list of them (a set)")
    return name


def is_literal(value: object, path: str) -> bool:
    """Whether an argument a column is read as text in is a literal, `{"text": <a string, or a non-empty list of
    them: a set>, "strip"?: bool, "lower"?: bool}`, the same text on every row of whatever dataset the value it goes
    with has rows of; one with another key, or a text or a flag of another type, is refused by name."""
    if data.literal_of(value) is None:
        return False
    assert isinstance(value, dict)
    for key in value:
        if key not in ("text", "strip", "lower"):
            fail(path, f"a literal takes no {key!r}: it is {{'text': <a string, or a list of them (a set)>, "
                       "'strip'?: bool, 'lower'?: bool}")
    text = value["text"]
    if not (isinstance(text, str) or isinstance(text, list) and text and all(isinstance(t, str) for t in text)):
        fail(f"{path}.text", f"{text!r} is not a literal's text: a string, or a non-empty list of them (a set)")
    for flag in ("strip", "lower"):
        if not isinstance(value.get(flag, False), bool):
            fail(f"{path}.{flag}", f"{value[flag]!r} is not a boolean")
    return True


def check_column_ref(cx: Compiling, value: object, path: str, dataset: str | None) -> str:
    """`<dataset>.<column>`, a column present on every row; of `dataset` when one is given. Returns the dataset."""
    name, _, column = value.partition(".") if isinstance(value, str) else ("", "", "")
    if name not in cx.rows:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx.rows)}")
    if not column:
        fail(path, f"{value!r} is not a column reference, '<dataset>.<column>'; dataset {name!r} has columns "
             f"{', '.join(cx.rows[name][0])}")
    if any(column not in r for r in cx.rows[name]):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(cx.rows[name][0])}")
    if dataset is not None and name != dataset:
        fail(path, f"{value!r} is a column of {name!r}, and the value it goes with has a row per example of "
             f"{dataset!r}")
    return name


def check_tokens(engine: Engine, plan: Plan, findings: Findings, where: str, name: str, args: dict,
                 dims: list[str] | None, row_positions: list | None, inputs: list[list[int]] | None,
                 before: bool) -> None:
    """Every column op `name` scores against logits, resolved once into `findings.answers` (`answer`): a class or
    a set of several ids is refused where the op reads one id's logit. A column with an answer of several
    tokens lines every row up with the read, one position per token: the value has a pos axis (when its axes
    `dims` are known) and, where the read's positions are known before the run (`row_positions`, per row), each row
    the read reaches has as many (a row it does not reach, the marker, is missing, not refused), the input's
    tokens there (`inputs`, each row's ids; under `before`, the ones those positions predict) saying what differs
    (`how_to_align`). Where the read is a scope that is `before` on the step's own input, the input's next token at
    each position is among the ids the answer's token there may be."""
    texts = vocabulary(engine.tokenizer)
    for key, p in ops.parameters(name).items():
        if p.annotation != ops.Tokens or key not in args:
            continue
        column_answer = answer(engine, plan, findings, args[key], where)
        values, sets, column, aligned = (column_answer.values, column_answer.sets, column_answer.column,
                                         column_answer.aligned)
        if column_answer.literal:                         # the same on every row the read holds
            n = len(row_positions) if row_positions is not None else 1
            values, sets = values * n, sets * n
        strip, lower = data.flags_of(args[key])
        for r, (value, row) in enumerate(zip(values, sets)):
            if name in ONE_ID and any(len(ids) > 1 for ids in row):
                raise ValueError(f"{where}: {column} row {r}: {value!r} stands for {max(map(len, row))} ids (a set, "
                                 f"or a class under strip or lower), which have no one logit for {name}; "
                                 "token_prob scores them together")
            if aligned and dims is not None and "pos" not in dims:
                raise ValueError(f"{where}: {column} row {r}: {value!r} is {len(row)} token"
                                 f"{'s' * (len(row) != 1)}, a position each, and the value has no pos axis ({dims})")
            if aligned and row_positions is not None and isinstance(row_positions[r], list) and \
                    len(row_positions[r]) != len(row):
                held_tokens = None if inputs is None else [inputs[r][p + 1 if before else p] for p in row_positions[r]]
                why = how_to_align(engine.tokenizer, value, strip, lower, len(row), row_positions[r], held_tokens,
                                   before)
                raise ValueError(f"{where}: "
                                 f"{unaligned_answer_refusal(column, r, value, len(row), len(row_positions[r]))}{why}")
        if not before or inputs is None or row_positions is None:
            continue
        for r, positions in enumerate(row_positions):
            for i, p in enumerate(positions if isinstance(positions, list) else []):
                ids, next_ = sets[r][i if aligned else 0], inputs[r][p + 1]
                if next_ not in ids:
                    raise ValueError(f"{where}: {column} row {r}: position {p} predicts {texts[next_]!r} in the input, "
                                     f"which is not token {i if aligned else 0} of {values[r]!r} as it tokenizes alone "
                                     f"({[texts[i] for i in ids[:3]]}); the answer is scored as it tokenizes, so the "
                                     "input says it the same way")


@dataclass
class Answer:
    """A column reference's answers, resolved once per findings: each row's `values`, per token the ids it may
    be (`sets`), the `column`, and whether its rows line up with the read (`aligned`: some row is several
    tokens; else every row is one, which scores every position of its row). A literal's (`literal`) is one row,
    the same on every row of whatever value it scores, `column` its JSON."""
    values: list
    sets: list
    column: str
    aligned: bool
    literal: bool = False


def answer(engine: Engine, plan: Plan, findings: Findings, ref: str | dict, where: str) -> Answer:
    """The `Answer` of a column reference or a literal, resolved into `findings.answers` the first time it is asked
    for; a literal's is tokenized as a column's value is."""
    key = refkey(ref)
    if key not in findings.answers:
        given = data.literal_of(ref)
        if given is not None:
            column, values = json.dumps(ref, ensure_ascii=False), [given[0]]
        else:
            column = (data.column_of(ref) or ("",))[0]
            values = column_texts(plan, column, None)
        strip, lower = data.flags_of(ref)
        try:
            sets = answer_sets(engine.tokenizer, values, column, strip, lower)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        findings.answers[key] = Answer(values, sets, column, any(len(row) > 1 for row in sets), given is not None)
    return findings.answers[key]


def check_classes(plan: Plan, where: str, name: str, args: dict, probe: str, classes: dict) -> None:
    """Every column op `name` scores against a probe's class list holds only those classes, on every row of the
    dataset it reads."""
    vocabulary = classes[declared(plan, probe)]
    for key, p in ops.parameters(name).items():
        if p.annotation == ops.Tokens and key in args:
            strip, lower = data.flags_of(args[key])
            if strip or lower:
                raise ValueError(f"{where}: {key} names a class of probe {declared(plan, probe)!r}, which is compared "
                                 "exactly; strip and lower compare an answer's tokens")
            given = data.literal_of(args[key])
            if given is not None:
                if isinstance(given[0], list) or given[0] not in vocabulary:
                    raise ValueError(f"{where}: {key} is the literal {given[0]!r}, not one of the classes of probe "
                                     f"{declared(plan, probe)!r}: {vocabulary}")
                continue
            ref = (data.column_of(args[key]) or ("",))[0]
            dataset, _, column = ref.partition(".")
            for i, row in enumerate(held_rows(plan, dataset)):
                if isinstance(row[column], list) or row[column] not in vocabulary:
                    raise ValueError(f"{where}: {key} is {column!r} of {dataset!r}, whose row {i} "
                                     f"is {row[column]!r}, not one of the classes of probe {declared(plan, probe)!r}: "
                                     f"{vocabulary}")


def column_texts(plan: Plan, column: str, ids: list[int] | None) -> list:
    """A column's value on each of its dataset's rows `ids` (every one: None), a string or a list of them."""
    name, _, path = column.partition(".")
    return [data.lookup(r, path) for r in data.by_id(held_rows(plan, name), ids)]


def texts_of(plan: Plan, ref: object, ids: list[int] | None, of: object, dims: list[str]) -> list:
    """A column reference's value on each of the dataset rows `ids` (`column_texts`), or a literal's text on each
    of them, the same on every row (every row of the value `of` it goes with, laid out `dims`: None)."""
    given = data.literal_of(ref)
    if given is not None:
        return [given[0]] * row_count(of, dims, ids)
    return column_texts(plan, (data.column_of(ref) or ("",))[0], ids)


def row_count(of: object, dims: list[str], ids: list[int] | None) -> int:
    """How many rows the value `of` (laid out `dims`; a list of texts is one per row) holds: `ids`'s, where given."""
    if ids is not None:
        return len(ids)
    assert of is not None, "a value bound without its rows is a write's, which a batch's ids are given for"
    return of.shape[dims.index("rows")] if isinstance(of, Tensor) else len(cast(list, of))


def refkey(ref: str | dict) -> str:
    """A column reference as the key its resolved answers are found under."""
    return json.dumps(ref, sort_keys=True)


def answer_ids(answer: Answer, of: Tensor | None, dims: list[str], ids: list[int] | None) -> Tensor:
    """Each row's answer (of dataset rows `ids`; every one, 0 to n-1 in order: None) laid against the value `of` a
    metric scores, as the token ids each position may be, padded with -1 (`metrics._aligned`). A column whose every
    row is one token scores it at every position of its row (`[rows, m]`); one with an answer of several tokens is
    `aligned`: token i of each row's at the row's position i (`[rows, pos, m]`), which needs every row the value
    reaches to have as many positions as its answer has tokens."""
    n = row_count(of, dims, ids) if answer.literal else 0
    column = answer.column
    chosen = answer.sets * n if answer.literal else data.by_id(answer.sets, ids)
    given = answer.values * n if answer.literal else data.by_id(answer.values, ids)
    most = max(len(t) for row in chosen for t in row)
    sets = [[t + [-1] * (most - len(t)) for t in row] for row in chosen]
    if not answer.aligned:
        return torch.tensor([row[0] for row in sets])
    if "pos" not in dims:
        raise ValueError(f"{column}: an answer of several tokens scores a position each, and the value has no pos "
                         f"axis ({dims})")
    assert of is not None, "only a write's ops are bound without the value, and with no dims"
    width = of.shape[dims.index("pos")]
    # A row's positions are those not NaN all the way (a row the value does not reach, or a position past a row's own,
    # is NaN); the trailing size is spelled out, since a reshape with -1 is ambiguous on a `pos` axis of 0.
    x = of.movedim((dims.index("rows"), dims.index("pos")), (0, 1))
    counts = (~x.reshape(*x.shape[:2], math.prod(x.shape[2:])).isnan().all(-1)).sum(1).tolist()
    for r, value, row, k in zip(ids if ids is not None else range(len(sets)), given, sets, counts):
        if k != len(row) and k != 0:                    # a row the value does not reach is missing, not refused
            raise ValueError(unaligned_answer_refusal(column, r, value, len(row), k))
    # `[rows, pos, m]` whatever `width` is: over a read no row reached it is 0, and the list alone would be `[rows, 0]`.
    return torch.tensor([[row[i] if i < len(row) else [-1] * most for i in range(width)]
                         for row in sets]).reshape(len(sets), width, most)


def unaligned_answer_refusal(column: str, row: int, value: object, k: int, n: int) -> str:
    """Why a row's answer does not line up with the read a metric scores."""
    return (f"{column} row {row}: {value!r} is {k} token{'s' * (k != 1)}; the read has {n} position"
            f"{'s' * (n != 1)} on that row; "
            "token i of an answer scores the read's position i, so the read carries one per token (a scope with "
            "\"before\": true reads the positions that predict the answer's tokens)")


def how_to_align(tokenizer: PreTrainedTokenizerBase, value: str | list[str], strip: bool, lower: bool, k: int,
                 positions: list[int], held_tokens: list[int] | None, before: bool) -> str:
    """What to do where an answer of `k` tokens and its read at positions `positions` differ, where the tokens say: the
    answer's first token is whitespace only, which it keeps and a scope under strip does not cover; the input's
    token at the read's first position (`held_tokens`: the tokens the positions hold, or predict under `before`, where
    known) is whitespace only, which the answer has no token for; or the input says the answer's own text there in
    other tokens than the column's text makes alone (` signs` one token, `signs` two): the answer is written as the
    input says it. Where the input's text there makes the answer's own tokens alone (`42` and ` 42` are both `▁`,
    `4`, `2` on a SentencePiece tokenizer), the answer's whitespace-only token 0 the read misses is named instead,
    and the scope is written without strip."""
    texts = vocabulary(tokenizer)
    first = value[0] if isinstance(value, list) else value
    written = drop_trailing_whitespace(tokenizer, tokenizer(first, add_special_tokens=False)["input_ids"], strip)
    if strip and k > len(positions) and not texts[written[0]].strip():
        return (f"; its token 0, {texts[written[0]]!r}, is whitespace only, which an answer keeps and a scope under "
                "strip does not cover: scope the column without strip")
    if held_tokens is None:
        return ""
    if len(positions) > k and not texts[held_tokens[0]].strip():
        return (f"; the read's position {positions[0]} {'predicts' if before else 'holds'} "
                f"{texts[held_tokens[0]]!r}, whitespace only, which the answer has no token for: write the answer "
                "with its space")
    joined = "".join(texts[i] for i in held_tokens)
    if data.compared(joined, True, lower) != data.compared(first, True, lower):
        return ""
    alone = drop_trailing_whitespace(tokenizer, tokenizer(joined, add_special_tokens=False)["input_ids"], strip)
    if alone != written:
        return (f"; the input says it as {joined!r}, {len(held_tokens)} token{'s' * (len(held_tokens) != 1)} "
                f"({[texts[i] for i in held_tokens]}), where {first!r} alone is {[texts[i] for i in written]}: "
                f"write the answer as the input says it, {joined!r}")
    if k > len(positions) and not texts[written[0]].strip():
        if first[:1].isspace():
            return (f"; its token 0, {texts[written[0]]!r}, is whitespace only, which an answer keeps and a scope "
                    "under strip does not cover: scope the column without strip")
        return (f"; its token 0, {texts[written[0]]!r}, is whitespace only, which {first!r} gains when tokenized "
                "alone and the read does not cover: write the answer with the space the input has before it, and scope the "
                "column without strip")
    return ""


def answer_sets(tokenizer: PreTrainedTokenizerBase, values: list, column: str, strip: bool,
                lower: bool) -> list[list[list[int]]]:
    """Each row's answer as, per token, the ids it may be. A row's value is a string or a set of them (a list),
    each tokenized exactly as written without special tokens (`" Friday"` and `"Friday"` are different tokens
    on most tokenizers; the column says which), and under `strip` without its whitespace-only tokens at the end
    (`drop_trailing_whitespace`); a set's members are one width, and its token i is any of theirs. Each token stands for
    its spelling (`spelling_ids`). Each position's ids are in first-met order, each once."""
    out = []
    for row, value in enumerate(values):
        members = value if isinstance(value, list) else [value]
        tokenized = [drop_trailing_whitespace(tokenizer, tokenizer(m, add_special_tokens=False)["input_ids"], strip)
                     for m in members]
        if len({len(t) for t in tokenized}) > 1 or not tokenized[0]:
            empty = f"{value!r} is no tokens" + (" once strip drops the whitespace-only tokens at its end" if strip
                                                 else "")
            raise ValueError(f"{column} row {row}: " + (empty if not tokenized[0] else
                             "its members are " + ", ".join(f"{m!r} {len(t)}" for m, t in zip(members, tokenized))
                             + " tokens; a set's members are one width, the positions its read aligns"))
        out.append([list(dict.fromkeys(i for t in tokenized for i in spelling_ids(tokenizer, t, k, strip, lower)))
                    for k in range(len(tokenized[0]))])
    return out


def drop_trailing_whitespace(tokenizer: PreTrainedTokenizerBase, ids: list[int], strip: bool) -> list[int]:
    """`ids` without their whitespace-only tokens at the end under `strip` (`" New York "`: `▁New ▁York`), since
    what follows an answer is not scored; a whitespace-only token at the start is kept (`" 42"`: `▁ 4 2`)."""
    texts = vocabulary(tokenizer)
    while strip and ids and not texts[ids[-1]].strip():
        ids = ids[:-1]
    return ids


def spelling_ids(tokenizer: PreTrainedTokenizerBase, ids: list[int], index: int, strip: bool, lower: bool) -> list[int]:
    """The ids token `index` of an answer's `ids` (`drop_trailing_whitespace`) stands for: the one place a token's
    spelling is decided. It stands for every id whose text compares the same (`spelling`): under `strip` the answer's
    first token without its leading whitespace and its last without its trailing (a one-token answer's without either),
    under `lower` lowercased. A token between the ends compares with its whitespace; under `strip` a whitespace-only
    token, and anywhere a token that is part of a character (its text is U+FFFD), stands for itself, as every token does
    without a flag."""
    text = vocabulary(tokenizer)[ids[index]]
    lead, trail = strip and index == 0, strip and index == len(ids) - 1
    if "\ufffd" in text or strip and not text.strip() or not (lead or trail or lower):
        return [ids[index]]
    return spellings(tokenizer, lead, trail, lower)[spelling(text, lead, trail, lower)]


def spelling(text: str, lead: bool, trail: bool, lower: bool) -> str:
    """A token's spelling, its text as it is compared: without its leading whitespace under `lead`, its trailing under
    `trail`, lowercased under `lower` (`data.compared`)."""
    text = text.lstrip() if lead else text
    return data.compared(text.rstrip() if trail else text, False, lower)


@functools.cache
def vocabulary(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Each id's text: what it decodes to after another token (`a`'s first), so a leading space (` Positive`,
    `▁Positive`) is its own. Made once per tokenizer. It is how a refusal shows a token, what a column or a text
    scope can hold (`' signs'`); an output (`pos_token`, `run.tokens`, `top_k`'s `tokens`) spells the tokenizer's
    own pieces (`Ġsigns`, `▁signs`)."""
    anchor = tokenizer("a", add_special_tokens=False)["input_ids"][0]
    lead = len(tokenizer.decode([anchor]))
    return [text[lead:] for text in tokenizer.batch_decode([[anchor, i] for i in range(len(tokenizer))])]


@functools.cache
def spellings(tokenizer: PreTrainedTokenizerBase, lead: bool, trail: bool, lower: bool) -> dict[str, list[int]]:
    """Every id of the vocabulary but a part of a character by its spelling (`spelling`): the spellings an
    answer's tokens stand for. Made once per tokenizer and flags."""
    by_spelling: dict[str, list[int]] = {}
    for i, text in enumerate(vocabulary(tokenizer)):
        if "\ufffd" not in text:
            by_spelling.setdefault(spelling(text, lead, trail, lower), []).append(i)
    return by_spelling
