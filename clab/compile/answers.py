"""Answers: a dataset column scored against a value's vocabulary (the tokenizer's against logits, or a probe's
classes), each row's answer tokenized as written and lined up with the read it scores; and a column read as text."""

import functools
import json
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from pydantic import ValidationError
from torch import Tensor

from clab import ops
from clab.compile import positions
from clab.compile.state import Compiling, Found
from clab.format import data
from clab.format.sweeps import lookup
from clab.ops.metrics import ONE_ID
from clab.plan import Column, Plan, arguments, body_view, declared, entries, fail, points


def probe_of(plan: Plan, name: str) -> str:
    """The probe whose `encode` made value `name`'s `classes` axis, as referenced where it was encoded."""
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    if step.kind == "op":
        return probe_of(plan, step.of)
    if step.kind == "fit":
        return probe_of(body_view(plan, thing), member.partition(".")[2])
    return next(arguments(o)["featurizer"] for o in reversed(entries(step)[member].ops) if o.op == "encode")


def text_ref(cx: Compiling, value, path: str, dataset: str | None) -> str:
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


def column_ref(cx: Compiling, value, path: str, dataset: str | None) -> str:
    """`<dataset>.<column>`, a column present on every row; of `dataset` when one is given. Returns the dataset."""
    name, _, column = value.partition(".") if isinstance(value, str) else ("", "", "")
    if name not in cx.rows:
        fail(path, f"{value!r} is not <dataset>.<column> over a dataset; datasets: {', '.join(cx.rows)}")
    if any(column not in r for r in cx.rows[name]):
        fail(path, f"dataset {name!r} has no column {column!r} on every row; columns: {', '.join(cx.rows[name][0])}")
    if dataset is not None and name != dataset:
        fail(path, f"{value!r} is a column of {name!r}, and the value it goes with has a row per example of "
             f"{dataset!r}")
    return name


def check_tokens(engine, plan: Plan, found: Found, where: str, name: str, args: dict, dims: list[str] | None,
                 placed: list | None, forced: list | None = None) -> None:
    """Every column op `name` scores against logits, resolved once into `found.answers` (`answer`): a class or
    a set of several ids is refused where the op reads one id's logit. A column with an answer of several
    tokens lines every row up with the read, one position per token: the value has a pos axis (when its axes
    `dims` are known) and, where the read's positions are known before the run (`placed`, per row), each row
    has as many. Where the read is a scope that is `before` on the step's own input (`forced`: each row's
    input ids), the input's next token at each position is among the ids the answer's token there may be."""
    for key, p in ops.parameters(name).items():
        if p.annotation != ops.Tokens or key not in args:
            continue
        got = answer(engine, plan, found, args[key], where)
        values, sets, column, aligned = got.values, got.sets, got.column, got.aligned
        for r, (value, row) in enumerate(zip(values, sets)):
            if name in ONE_ID and any(len(ids) > 1 for ids in row):
                raise ValueError(f"{where}: {column} row {r}: {value!r} stands for {max(map(len, row))} ids (a set, "
                                 f"or a class under strip or lower), which have no one logit for {name}; "
                                 "token_prob scores them together")
            if aligned and dims is not None and "pos" not in dims:
                raise ValueError(f"{where}: {column} row {r}: {value!r} is {len(row)} token"
                                 f"{'s' * (len(row) != 1)}, a position each, and the value has no pos axis ({dims})")
            if aligned and placed is not None and isinstance(placed[r], list) and len(placed[r]) != len(row):
                raise ValueError(f"{where}: {unaligned(column, r, value, len(row), len(placed[r]))}")
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

    def rows(self, part: slice) -> "Answer":
        return replace(self, values=self.values[part], sets=self.sets[part])


def answer(engine, plan: Plan, found: Found, ref, where: str) -> Answer:
    """The `Answer` of a column reference, resolved into `found.answers` the first time it is asked for."""
    key = refkey(ref)
    if key not in found.answers:
        try:
            values, sets, column = tokens(engine, plan, ref, None)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        found.answers[key] = Answer(values, sets, column, any(len(row) > 1 for row in sets))
    return found.answers[key]


def placed(plan: Plan, found: Found, name: str, at: dict) -> tuple[list | None, list | None]:
    """The positions per row of the read value `name`'s positions are (following `of` through op steps), where
    they are known before the run, else None; and each row's input ids where that read is a scope that is
    `before` (whose next tokens an answer scores), else None."""
    while True:
        thing, _, member = name.partition(".")
        step = points(plan.steps[thing])[0]
        if step.kind != "op":
            break
        name = step.of
    try:
        where = lookup(found.positions, name, at)
    except KeyError:
        return None, None
    record = entries(step).get(member) if step.kind in ("forward", "generate") else None
    scope = (positions.parse(record.pos)["scope"] or {}) if record is not None else {}
    return where, [e["ids"] for e in found.inputs[data.input_name(step.input)]["rows"]] \
        if scope.get("before") else None


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
            for i, row in enumerate(plan.data[dataset].rows or []):
                if isinstance(row[column], list) or row[column] not in vocabulary:
                    raise ValueError(f"{where}: {key} is {column!r} of {swap.get(dataset, dataset)!r}, whose row {i} "
                                     f"is {row[column]!r}, not one of the classes of probe {declared(plan, probe)!r}: "
                                     f"{vocabulary}")


def indexed(plan: Plan, column: str, vocabulary: list, rows: slice | None) -> Tensor:
    """A column's values as their indices in a probe's class list."""
    values = column_texts(plan, column, None)       # preflight checked each is a class
    return torch.tensor([vocabulary.index(v) for v in (values[rows] if rows is not None else values)])


def column_texts(plan: Plan, column: str, rows: slice | None) -> list:
    """A column's value on each of its dataset's rows (a string, or a list of them), sliced to `rows`."""
    name, _, path = column.partition(".")
    values = [data.lookup(r, path) for r in plan.data[name].rows or []]
    return values[rows] if rows is not None else values


def refkey(ref) -> str:
    """A column reference as the key its resolved answers are found under."""
    return json.dumps(ref, sort_keys=True)


def tokens(engine, plan: Plan, ref, rows: slice | None) -> tuple[list, list, str]:
    """A column reference's answer on each row, sliced to `rows`: its value, and per token the ids it may be
    (`answers`), and the column's name."""
    column, strip, lower = data.column_of(ref) or ("", False, False)
    values = column_texts(plan, column, rows)
    return values, answers(engine.tokenizer, values, column, strip, lower), column


def answered(values: list, sets: list, column: str, of: Any, dims: list[str], rows: slice | None,
             aligned: bool) -> Tensor:
    """Each row's answer laid against the value `of` a metric scores, as the ids each position may be, padded
    with -1 (`metrics._aligned`). A column whose every row is one token scores it at every position of its row
    (`[rows, m]`); one with an answer of several tokens is `aligned`: token i of each row's at the row's
    position i (`[rows, pos, m]`), which needs every row the value reaches to have as many positions as its
    answer has tokens (`present`)."""
    start = rows.start if rows is not None else 0
    most = max(len(ids) for row in sets for ids in row)
    sets = [[ids + [-1] * (most - len(ids)) for ids in row] for row in sets]
    if not aligned:
        return torch.tensor([row[0] for row in sets])
    if "pos" not in dims:
        raise ValueError(f"{column}: an answer of several tokens scores a position each, and the value has no pos "
                         f"axis ({dims})")
    width = cast(Tensor, of).shape[dims.index("pos")]
    lined(values, sets, present(cast(Tensor, of), dims), column, start)
    return torch.tensor([[row[i] if i < len(row) else [-1] * most for i in range(width)] for row in sets])


def present(of: Tensor, dims: list[str]) -> list[int]:
    """How many positions each row of `of` holds: those not NaN all the way (a row the value does not reach,
    or a position past a row's own, is NaN)."""
    x = of.movedim((dims.index("rows"), dims.index("pos")), (0, 1))
    return (~x.reshape(*x.shape[:2], -1).isnan().all(-1)).sum(1).tolist()


def lined(values: list, sets: list, widths: list[int], column: str, start: int) -> None:
    """Refuses a row that does not have one of the read's positions per token of its answer; a row with none is
    missing, not refused."""
    for r, (value, k, n) in enumerate(zip(values, (len(s) for s in sets), widths)):
        if n != k and n != 0:
            raise ValueError(unaligned(column, start + r, value, k, n))


def unaligned(column: str, row: int, value, k: int, n: int) -> str:
    """Why a row's answer does not line up with the read a metric scores."""
    return (f"{column} row {row}: {value!r} is {k} token{'s' * (k != 1)}; the read has {n} position"
            f"{'s' * (n != 1)} on that row; "
            "token i of an answer scores the read's position i, so the read carries one per token (a scope with "
            "\"before\": true reads the positions that predict the answer's tokens)")


def answers(tokenizer, values: list, column: str, strip: bool = False, lower: bool = False) -> list[list[list[int]]]:
    """Each row's answer as, per token, the ids it may be. A row's value is a string or a set of them (a list),
    each tokenized exactly as written without special tokens (`" Friday"` and `"Friday"` are different tokens
    on most tokenizers; the column says which), and under `strip` without its whitespace-only tokens at either
    end; a set's members are one width, and its token i is any of theirs. Under `strip` or `lower` a token
    stands for its class (`spellings`). Each position's ids are in first-met order, each once."""
    out = []
    for row, value in enumerate(values):
        members = value if isinstance(value, list) else [value]
        tokenized = [trimmed(tokenizer, tokenizer(m, add_special_tokens=False)["input_ids"], strip) for m in members]
        if len({len(t) for t in tokenized}) > 1 or not tokenized[0]:
            raise ValueError(f"{column} row {row}: " + (f"{value!r} is no tokens" if not tokenized[0] else
                             "its members are " + ", ".join(f"{m!r} {len(t)}" for m, t in zip(members, tokenized))
                             + " tokens; a set's members are one width, the positions its read aligns"))
        out.append([list(dict.fromkeys(i for t in tokenized for i in spelled(tokenizer, t[k], strip, lower)))
                    for k in range(len(tokenized[0]))])
    return out


def trimmed(tokenizer, ids: list[int], strip: bool) -> list[int]:
    """`ids` without the whitespace-only tokens at either end under `strip` (`"  New York "`: `ĠNew ĠYork`)."""
    texts = vocabulary(tokenizer)
    while strip and ids and not texts[ids[0]].strip():
        ids = ids[1:]
    while strip and ids and not texts[ids[-1]].strip():
        ids = ids[:-1]
    return ids


def spelled(tokenizer, token: int, strip: bool, lower: bool) -> list[int]:
    """The ids a token stands for: under `strip` or `lower`, its class, every id whose text compares the same
    (`compared`); else the token alone. A token that is part of a character (its text is U+FFFD) is its own."""
    text = vocabulary(tokenizer)[token]
    if not (strip or lower) or "\ufffd" in text:
        return [token]
    return spellings(tokenizer, strip, lower)[data.compared(text, strip, lower)]


@functools.cache
def vocabulary(tokenizer) -> list[str]:
    """Each id's text: what it decodes to after another token (`a`'s first), so a leading space (` Positive`,
    `▁Positive`) is its own. Made once per tokenizer."""
    anchor = tokenizer("a", add_special_tokens=False)["input_ids"][0]
    lead = len(tokenizer.decode([anchor]))
    return [text[lead:] for text in tokenizer.batch_decode([[anchor, i] for i in range(len(tokenizer))])]


@functools.cache
def spellings(tokenizer, strip: bool, lower: bool) -> dict[str, list[int]]:
    """Every id of the vocabulary but a part of a character by its text as `strip` and `lower` compare it: the
    classes an answer's tokens stand for. Made once per tokenizer and flags."""
    found: dict[str, list[int]] = {}
    for i, text in enumerate(vocabulary(tokenizer)):
        if "\ufffd" not in text:
            found.setdefault(data.compared(text, strip, lower), []).append(i)
    return found
