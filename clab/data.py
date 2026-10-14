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


def field(row: dict, input: str) -> str | list[dict]:
    """The row's input: its text, or its chat messages."""
    _, column, index = parse_input(input)
    return row[column] if index is None else row[column][index]


def variable(row: dict, input: str, name: str) -> str | None:
    """The row's own text of variable `name`: its column of that name, or, for an indexed input, the matching
    entry of `<column>_variables` (`counterfactual_inputs_variables[0]["entity"]`)."""
    _, column, index = parse_input(input)
    return row.get(name) if index is None else row[f"{column}_variables"][index].get(name)


def encode(tokenizer, value: str | list[dict]) -> dict:
    """One row's tokens: `ids`, each token's character `offsets` in `text` (None on a tokenizer without an
    offset mapping), and, for a chat, each message's `turns` as `(role, start, end)` in the rendered text.

    A chat is rendered through the checkpoint's own template, and the rendered text tokenized without adding
    special tokens, since the template writes its own.
    """
    turns = []
    if isinstance(value, str):
        text, special = value, True
    else:
        text, special, cursor = tokenizer.apply_chat_template(value, tokenize=False), False, 0
        for message in value:
            start = text.index(message["content"], cursor)
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


def token_ids(tokenizer, texts: list[str], form: str, column: str) -> torch.Tensor:
    """One token id per row. `bare` spells the text as the first word of a text is, without its leading space."""
    ids = []
    for row, text in enumerate(texts):
        got = tokenizer(text if form == "spaced" else text.strip(), add_special_tokens=False)["input_ids"]
        if len(got) != 1:
            raise ValueError(f"{column} row {row}: {text!r} is {len(got)} tokens on this tokenizer; "
                             "an answer must be one")
        ids.append(got[0])
    return torch.tensor(ids)
