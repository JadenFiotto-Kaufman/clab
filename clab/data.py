"""Rows at compile time; encoding, padding and token columns inside the run."""

import json
import os

import torch


def load_rows(root: str, path: str) -> list[dict]:
    """`data/<path>.json` under the corpus root, filtered by `split` when the path ends in `#<split>`."""
    file, _, split = path.partition("#")
    with open(os.path.join(root, "data", file + ".json")) as f:
        rows = json.load(f)
    return [r for r in rows if r.get("split") == split] if split else rows


def key(input: dict) -> str:
    """One string per dataset input; every batch of it is padded to the same width."""
    index = "" if input["index"] is None else f"[{input['index']}]"
    return f"{input['data']}.{input['column']}{index}"


def field(row: dict, input: dict) -> str | list[dict]:
    """The row's input: its text, or its chat messages."""
    value = row[input["column"]]
    return value if input["index"] is None else value[input["index"]]


def lengths(tokenizer, texts: list) -> list[int]:
    return [len(_ids(tokenizer, t)) for t in texts]


def _ids(tokenizer, text) -> list[int]:
    if not isinstance(text, str):
        raise NotImplementedError("M2: chat inputs, rendered through the checkpoint's template")
    return tokenizer(text)["input_ids"]


def encode(tokenizer, texts: list, width: int) -> dict:
    """Tokenize each row and pad it on the left, by hand, to `width`.

    Padding by hand keeps the serving side's tokenizer unmutated. `position_ids`
    count real tokens only, and both engines pass them to the model; `special`
    marks the tokens no position names (the tokenizer's special ids, never
    padding).
    """
    ids = [_ids(tokenizer, t) for t in texts]
    pad = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
    input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids])
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])
    special = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids)) & mask.bool()
    return {"input_ids": input_ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0), "special": special}


def token_ids(tokenizer, texts: list[str], form: str, column: str) -> torch.Tensor:
    """One token id per row. `bare` spells the text as the first word of a text is, without its leading space."""
    ids = []
    for row, text in enumerate(texts):
        got = tokenizer(text if form == "spaced" else text.strip(), add_special_tokens=False)["input_ids"]
        if len(got) != 1:
            raise ValueError(f"{column} row {row}: {text!r} is {len(got)} tokens on this tokenizer; an answer must be one")
        ids.append(got[0])
    return torch.tensor(ids)
