"""Rows at compile time; encoding, padding and token columns inside the run."""

import functools
import hashlib
import json
import os
import re

import torch

INPUT = re.compile(r"^(\w+)\.(\w+)(?:\[(\d+)\])?$")


def rows(root: str, path: str, split: str | None) -> list[dict]:
    """`data/<path>.json` under the corpus root, keeping the rows whose `split` column is `split` when it is given."""
    file = os.path.join(root, "data", path + ".json")
    if not os.path.exists(file):
        raise ValueError(f"no file data/{path}.json under the corpus root")
    with open(file) as f:
        found = json.load(f)
    return [r for r in found if r.get("split") == split] if split is not None else found


def digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_input(text: str) -> tuple[str, str, int | None]:
    """`<dataset>.<column>` or `<dataset>.<column>[i]` as `(dataset, column, i)`."""
    match = INPUT.match(text)
    if match is None:
        raise ValueError(f"{text!r} is not <dataset>.<column> or <dataset>.<column>[i]")
    return match.group(1), match.group(2), None if match.group(3) is None else int(match.group(3))


def input_columns(input) -> list[str]:
    """The columns an input names: itself, or a list's that are not `{"text": ...}` literals."""
    return [input] if isinstance(input, str) else [part for part in input if isinstance(part, str)]


def input_name(input, swap: dict | None = None) -> str:
    """An input as the run record names it: the column, or a list's parts joined by ` + ` (a literal as JSON),
    each column's dataset as `swap` renames it."""
    parts = [input] if isinstance(input, str) else input
    return " + ".join(renamed(p, swap or {}) if isinstance(p, str) else json.dumps(p.text, ensure_ascii=False)
                      for p in parts)


def column_of(ref) -> tuple[str, bool, bool] | None:
    """A column reference as `(<dataset>.<column>, strip, lower)`: the string, or `{"column", "strip"?,
    "lower"?}` as written or as a `Column`; None for anything else."""
    if isinstance(ref, str):
        return ref, False, False
    ref = ref if isinstance(ref, dict) else getattr(ref, "__dict__", {})
    if "column" not in ref:
        return None
    return ref["column"], bool(ref.get("strip")), bool(ref.get("lower"))


def renamed(name: str, swap: dict) -> str:
    """An input's name with each column's dataset as `swap` renames it."""
    parts = name.split(" + ")
    return " + ".join(p if p.startswith('"') or p.partition(".")[0] not in swap
                      else f"{swap[p.partition('.')[0]]}.{p.partition('.')[2]}" for p in parts)


def field(row: dict, input) -> str | list[dict]:
    """The row's input: its text, or its chat messages; a list's parts joined, each column's text as it is."""
    if not isinstance(input, str):
        return "".join(str(field(row, part)) if isinstance(part, str) else part.text for part in input)
    _, column, index = parse_input(input)
    return row[column] if index is None else row[column][index]


def lookup(row: dict, path: str) -> object:
    """The value a column path names in one row, after its dataset: `entity`, or into a column's entries and
    keys, `counterfactual_inputs_variables[0].entity`."""
    value: object = row
    for name, index in re.findall(r"(\w+)|\[(\d+)\]", path):
        value = value[name] if name else value[int(index)]  # type: ignore[index]
    return value


def encode(tokenizer, value: str | list[dict]) -> dict:
    """One row's tokens: `ids`, each token's character `offsets` in `text` (None on a tokenizer without an
    offset mapping), and, for a chat, each message's `turns` as `(role, start, end)` in the rendered text.

    A chat is rendered through the checkpoint's own template without a generation prompt, and the rendered
    text tokenized without adding special tokens, since the template writes its own.
    """
    turns = []
    if isinstance(value, str):
        text, special = value, True
    else:
        text, special, cursor = tokenizer.apply_chat_template(value, tokenize=False), False, 0
        for i, message in enumerate(value):
            start = text.find(message["content"], cursor)
            if start < 0:
                raise ValueError(f"message {i} ({message['role']}) {message['content']!r} is not where the chat "
                                 "template put the messages before it")
            cursor = start + len(message["content"])
            turns.append((message["role"], start, cursor))
    got = tokenizer(text, add_special_tokens=special, return_offsets_mapping=tokenizer.is_fast)
    return {"ids": got["input_ids"], "offsets": got.get("offset_mapping"), "text": text, "turns": turns}


def pad(tokenizer, ids: list[list[int]], width: int) -> dict:
    """Rows of ids padded on the left, by hand, to `width`.

    Padding by hand keeps the serving side's tokenizer unmutated. `position_ids` count real tokens only, and
    both engines pass them to the model.
    """
    value = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
    input_ids = torch.tensor([[value] * (width - len(x)) + x for x in ids])
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])
    return {"input_ids": input_ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}


def answers(tokenizer, values: list, column: str, strip: bool = False, lower: bool = False) -> list[list[list[int]]]:
    """Each row's answer as, per token, the ids it may be. A row's value is a string or a set of them (a list),
    each tokenized exactly as written without special tokens (`" Friday"` and `"Friday"` are different tokens
    on most tokenizers; the column says which); a set's members are one width, and its token i is any of
    theirs. Under `strip` or `lower` a token stands for its class: every id whose text compares the same
    (`spellings`). Each position's ids are in first-met order, each once."""
    out = []
    for row, value in enumerate(values):
        members = value if isinstance(value, list) else [value]
        tokenized = [tokenizer(m, add_special_tokens=False)["input_ids"] for m in members]
        if len({len(t) for t in tokenized}) > 1 or not tokenized[0]:
            raise ValueError(f"{column} row {row}: " + (f"{value!r} is no tokens" if not tokenized[0] else
                             "its members are " + ", ".join(f"{m!r} {len(t)}" for m, t in zip(members, tokenized))
                             + " tokens; a set's members are one width, the positions its read aligns"))
        classes = spellings(tokenizer, strip, lower) if strip or lower else None
        out.append([list(dict.fromkeys(i for t in tokenized for i in (
            classes[text_of(tokenizer, t[k], strip, lower)] if classes is not None else [t[k]])))
            for k in range(len(tokenized[0]))])
    return out


@functools.cache
def spellings(tokenizer, strip: bool, lower: bool) -> dict[str, list[int]]:
    """Every id of the tokenizer's vocabulary by its text as `strip` and `lower` compare it (`text_of`): the
    classes an answer's tokens stand for. Made once per tokenizer and flags."""
    anchor = anchored(tokenizer)
    found: dict[str, list[int]] = {}
    for i, text in enumerate(tokenizer.batch_decode([[anchor, i] for i in range(len(tokenizer))])):
        found.setdefault(compared(text[len(tokenizer.decode([anchor])):], strip, lower), []).append(i)
    return found


def text_of(tokenizer, token: int, strip: bool, lower: bool) -> str:
    """A token's text as `strip` and `lower` compare it: what it decodes to after another token, so a leading
    space (` Positive`, `▁Positive`) is its own."""
    anchor = anchored(tokenizer)
    return compared(tokenizer.decode([anchor, token])[len(tokenizer.decode([anchor])):], strip, lower)


def anchored(tokenizer) -> int:
    """The token a token is decoded after, so that its text is the text it adds: `a`'s first."""
    return tokenizer("a", add_special_tokens=False)["input_ids"][0]


def compared(text: str, strip: bool, lower: bool) -> str:
    """`text` stripped of surrounding whitespace under `strip`, lowercased under `lower`."""
    text = text.strip() if strip else text
    return text.lower() if lower else text
