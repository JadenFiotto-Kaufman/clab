"""Rows from a file or the hub; encoding, padding and token columns inside the run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.plan import Data, TextPart

INPUT = re.compile(r"^(\w+)\.(\w+)(?:\[(\d+)\])?$")


def rows(root: str, path: str, split: str | None) -> list[dict]:
    """`data/<path>.json` under the corpus root, keeping the rows whose `split` column is `split` when it is given."""
    file = os.path.join(root, "data", path + ".json")
    if not os.path.exists(file):
        raise ValueError(f"no file data/{path}.json under the corpus root")
    with open(file) as f:
        found = json.load(f)
    return [r for r in found if r.get("split") == split] if split is not None else found


def hub(d: Data) -> list[dict]:
    """Hub dataset `d.hf` at `d.revision`, in its subset `d.config`: the rows of its split `d.split` (left out, its
    only split), loaded whole by the `datasets` library, each a JSON object. Offline the library reads its cache at
    the commit it last cached, whatever the revision says: a commit `revision` other than that one is refused, and
    a branch reads as the cache holds it. What the hub or the cache refuses is said as it says it, after the
    dataset; a split it does not have, or several and none named, and a column whose values are not JSON (images,
    audio, bytes) are refused by name."""
    import datasets                     # with pyarrow and pandas, which only a hub dataset needs

    assert d.hf is not None and d.revision is not None     # the record's check: `hf` comes with its `revision`
    where = hub_name(d)
    try:
        found = cast("datasets.DatasetDict", datasets.load_dataset(d.hf, d.config, revision=d.revision))
    except Exception as e:              # whatever the hub, its cache or the library raises, named
        raise ValueError(f"{where}: {e}") from None
    splits = [str(s) for s in found]
    split = d.split if d.split is not None else splits[0] if len(splits) == 1 else None
    if split is None:
        raise ValueError(f"{where} has splits {', '.join(splits)}; 'split' names the one to read")
    if split not in found:
        raise ValueError(f"{where} has no such split; its splits: {', '.join(splits)}")
    # The library caches a hub dataset under the commit it loaded: offline, the latest it cached.
    cached = {os.path.basename(os.path.dirname(f["filename"])) for f in found[split].cache_files} - {d.revision}
    if cached and re.fullmatch(r"[0-9a-f]{40}", d.revision):
        raise ValueError(f"{where} is not in the cache, which holds it at {', '.join(sorted(cached))}")
    rows = found[split].to_list()
    for column in found[split].column_names:
        try:
            json.dumps([row[column] for row in rows])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: column {column!r} holds values that are not JSON (images, audio, bytes); a "
                             "row is JSON: text, numbers, booleans, null, and lists and objects of them") from None
    return rows


def hub_name(d: Data) -> str:
    """A hub dataset as a message names it: `nyu-mll/glue (config 'wnli', split 'validation') at revision <r>`."""
    given = [f"{k} {v!r}" for k, v in (("config", d.config), ("split", d.split)) if v is not None]
    return f"{d.hf}" + (f" ({', '.join(given)})" if given else "") + f" at revision {d.revision}"


def shuffled(rows: list[dict]) -> list[dict]:
    """`rows` in one fixed random order, the same on every machine and version: each row's index ranked by a
    blake2b hash of seed 0 and the index."""
    def key(i: int) -> bytes:
        return hashlib.blake2b(repr((0, i)).encode(), digest_size=8).digest()
    return [rows[i] for i in sorted(range(len(rows)), key=key)]


def renamed(rows: list[dict], columns: dict[str, str]) -> list[dict]:
    """Each row with each name `columns` maps the column it maps to, that column no longer under its own name,
    and every other column as it is."""
    sources = set(columns.values())
    return [{**{k: v for k, v in row.items() if k not in sources}, **{name: row[c] for name, c in columns.items()}}
            for row in rows]


def digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_input(text: str) -> tuple[str, str, int | None]:
    """`<dataset>.<column>` or `<dataset>.<column>[i]` as `(dataset, column, i)`."""
    match = INPUT.match(text)
    if match is None:
        raise ValueError(f"{text!r} is not <dataset>.<column> or <dataset>.<column>[i]")
    return match.group(1), match.group(2), None if match.group(3) is None else int(match.group(3))


def by_id(table: Any, ids: list[int] | None, among: list[int] | None = None, axis: int = 0) -> Any:
    """The rows of a per-row table for dataset row ids `ids` (None: every row, the table as it is), in their order:
    a list's, a tensor's along `axis` (on its device), each of a dict's. The one way per-row data is indexed: a
    dataset's rows, an input's encoded rows, a read's positions, an answer's tokens, a column's texts and a value's
    rows. A table holds every row of its dataset in order, or the rows `among`, in that order (a value a run of a
    fit's body made holds the rows that run was over: `batches.value_at`)."""
    if ids is None:
        return table
    at = ids if among is None else [among.index(i) for i in ids]
    if isinstance(table, dict):
        return {k: by_id(v, ids, among, axis) for k, v in table.items()}
    if isinstance(table, torch.Tensor):
        return table.index_select(axis, torch.tensor(at, dtype=torch.long, device=table.device))
    return [table[i] for i in at] if axis == 0 else [by_id(v, ids, among, axis - 1) for v in table]


def input_columns(input: str | list[str | TextPart]) -> list[str]:
    """The columns an input names: itself, or a list's that are not `{"text": ...}` literals."""
    return [input] if isinstance(input, str) else [part for part in input if isinstance(part, str)]


def input_name(input: str | list[str | TextPart], swap: dict[str, str] | None = None) -> str:
    """An input as the run record names it: the column, or a list's parts joined by ` + ` (a literal as JSON),
    each column's dataset as `swap` renames it."""
    parts = [input] if isinstance(input, str) else input
    swap = swap or {}
    return " + ".join(f"{swap.get(p.partition('.')[0], p.partition('.')[0])}.{p.partition('.')[2]}"
                      if isinstance(p, str) else json.dumps(p.text, ensure_ascii=False) for p in parts)


def column_of(ref: object) -> tuple[str, bool, bool] | None:
    """A column reference as a document writes it, `(<dataset>.<column>, strip, lower)`: the string, or
    `{"column", "strip"?, "lower"?}`; None for anything else."""
    if isinstance(ref, str):
        return ref, False, False
    if not isinstance(ref, dict) or "column" not in ref:
        return None
    return ref["column"], bool(ref.get("strip")), bool(ref.get("lower"))


def field(row: dict, input: str | list[str | TextPart]) -> str | list[dict]:
    """The row's input: its text, or its chat messages; a list's parts joined, each column's text as it is."""
    if not isinstance(input, str):
        return "".join(str(field(row, part)) if isinstance(part, str) else part.text for part in input)
    _, column, index = parse_input(input)
    return row[column] if index is None else row[column][index]


def lookup(row: dict, path: str) -> object:
    """The value a column path names in one row, after its dataset: `entity`, or into a column's entries and
    keys, `counterfactual_inputs_variables[0].entity`."""
    value: Any = row
    for name, index in re.findall(r"(\w+)|\[(\d+)\]", path):
        value = value[name] if name else value[int(index)]
    return value


# What holds a message's place while its template renders, to find where its content begins: private-use
# characters no template writes or changes.
MARKER = "\ue000clab\ue001"


def clocked(tokenizer: PreTrainedTokenizerBase, template: dict | None) -> bool:
    """Whether the checkpoint's chat template reads the clock (`strftime_now`, which Llama-3.2's calls to write
    today's date into the system header it adds) and `template` does not pin the date (`date_string`), so that
    the same chat renders to other tokens on another day, which the rows' digest cannot say."""
    source = getattr(tokenizer, "chat_template", None)
    text = json.dumps(source) if isinstance(source, dict) else source or ""
    return "strftime_now" in text and "date_string" not in (template or {})


def encode(tokenizer: PreTrainedTokenizerBase, value: str | list[dict], template: dict | None) -> dict:
    """One row's tokens: `ids`, each token's character `offsets` in `text` (None on a tokenizer without an
    offset mapping), and, for a chat, each message's `turns` as `(role, start, end)` in the rendered text.

    Text is tokenized as the tokenizer does by default, BOS added where it adds one. A chat is rendered through
    the checkpoint's own template, given its dataset's `template` as arguments (none by default: no generation
    prompt), and the rendered text tokenized adding nothing, since the template writes its own BOS and markers.
    Every token either way is a position, a generation prompt's among them, which belong to no message. A
    message's text is searched for from where its own content begins, which the template rendered (with the same
    arguments) with that content held by a marker shows, so a message that says its role's name (`user`) is not
    found in its role's header.
    """
    turns = []
    if isinstance(value, str):
        text, adds = value, True
    else:
        given = template or {}
        text, adds, cursor = cast(str, tokenizer.apply_chat_template(value, tokenize=False, **given)), False, 0
        for i, message in enumerate(value):
            held = [{**m, "content": MARKER} if j == i else m for j, m in enumerate(value)]
            own = cast(str, tokenizer.apply_chat_template(held, tokenize=False, **given)).find(MARKER, cursor)
            start = text.find(message["content"], max(cursor, own))
            if start < 0:
                raise ValueError(f"message {i} ({message['role']}) {message['content']!r} is not where the chat "
                                 "template put the messages before it")
            cursor = start + len(message["content"])
            turns.append((message["role"], start, cursor))
    got = tokenizer(text, add_special_tokens=adds, return_offsets_mapping=bool(tokenizer.is_fast))
    return {"ids": got["input_ids"], "offsets": got.get("offset_mapping"), "text": text, "turns": turns}


def pad(tokenizer: PreTrainedTokenizerBase, ids: list[list[int]], width: int) -> dict[str, torch.Tensor]:
    """Rows of ids padded on the left, by hand, to `width`.

    Padding by hand keeps the serving side's tokenizer unmutated. `position_ids` count real tokens only, and
    both engines pass them to the model.
    """
    value = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
    input_ids = torch.tensor([[value] * (width - len(x)) + x for x in ids])
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])
    return {"input_ids": input_ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}


def compared(text: str, strip: bool, lower: bool) -> str:
    """`text` as the flags compare it, wherever text is compared (a scope's search, `contains`, and the lowering
    of an answer's class): stripped of surrounding whitespace under `strip`, lowercased under `lower` a character
    at a time, each kept as it is where lowering would change its length, so a span of the result is the text's."""
    text = text.strip() if strip else text
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text) if lower else text
