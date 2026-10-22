"""Rows at compile time; encoding, padding and token columns inside the run."""

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
    swap = swap or {}
    return " + ".join(f"{swap.get(p.partition('.')[0], p.partition('.')[0])}.{p.partition('.')[2]}"
                      if isinstance(p, str) else json.dumps(p.text, ensure_ascii=False) for p in parts)


def column_of(ref) -> tuple[str, bool, bool] | None:
    """A column reference as a document writes it, `(<dataset>.<column>, strip, lower)`: the string, or
    `{"column", "strip"?, "lower"?}`; None for anything else."""
    if isinstance(ref, str):
        return ref, False, False
    if not isinstance(ref, dict) or "column" not in ref:
        return None
    return ref["column"], bool(ref.get("strip")), bool(ref.get("lower"))


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


# What holds a message's place while its template renders, to find where its content begins: private-use
# characters no template writes or changes.
MARKER = "\ue000clab\ue001"


def encode(tokenizer, value: str | list[dict]) -> dict:
    """One row's tokens: `ids`, each token's character `offsets` in `text` (None on a tokenizer without an
    offset mapping), and, for a chat, each message's `turns` as `(role, start, end)` in the rendered text.

    Text is tokenized as the tokenizer does by default, BOS added where it adds one. A chat is rendered through
    the checkpoint's own template without a generation prompt, and the rendered text tokenized adding nothing,
    since the template writes its own BOS and markers. Every token either way is a position. A message's text is
    searched for from where its own content begins, which the template rendered with that content held by a
    marker shows, so a message that says its role's name (`user`) is not found in its role's header.
    """
    turns = []
    if isinstance(value, str):
        text, adds = value, True
    else:
        text, adds, cursor = tokenizer.apply_chat_template(value, tokenize=False), False, 0
        for i, message in enumerate(value):
            held = [{**m, "content": MARKER} if j == i else m for j, m in enumerate(value)]
            own = tokenizer.apply_chat_template(held, tokenize=False).find(MARKER, cursor)
            start = text.find(message["content"], max(cursor, own))
            if start < 0:
                raise ValueError(f"message {i} ({message['role']}) {message['content']!r} is not where the chat "
                                 "template put the messages before it")
            cursor = start + len(message["content"])
            turns.append((message["role"], start, cursor))
    got = tokenizer(text, add_special_tokens=adds, return_offsets_mapping=tokenizer.is_fast)
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


def compared(text: str, strip: bool, lower: bool) -> str:
    """`text` as the flags compare it, wherever text is compared (a scope's search, an answer's class,
    `contains`): stripped of surrounding whitespace under `strip`, lowercased under `lower` a character at a
    time, each kept as it is where lowering would change its length, so a span of the result is the text's."""
    text = text.strip() if strip else text
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text) if lower else text
