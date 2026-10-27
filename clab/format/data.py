"""Rows from a file or the hub; encoding, padding, images and token columns inside the run."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.plan import Data, TextPart

INPUT = re.compile(r"^(\w+)\.(\w+)(?:\[(\d+)\])?$")

# What a vision-language checkpoint's encoding needs beside the tokenizer: the processor (which renders a chat with
# image parts, tokenizes it as the checkpoint does and makes the image tensors) and the id of the token the
# processor puts where an image's tokens go. An engine that loaded no processor has none (`engines.standard.vision`).
Vision = dict[str, Any]


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


def image_parts(value: object) -> list[dict]:
    """A chat's image parts, in order: each `{"type": "image", "image": "<path>"}` among its messages' content
    parts (`content` a list of parts, as the checkpoint's processor takes them). None for text, or for a chat
    whose every content is a string."""
    if not isinstance(value, list):
        return []
    return [part for message in value if isinstance(message.get("content"), list) for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image"]


def images(root: str, value: list[dict]) -> list[Any]:
    """The chat's images loaded, in order: each image part's `image` is a file under the corpus root's `data/`;
    a part without one, or one that is not there, is refused by name."""
    from PIL import Image

    out = []
    for i, part in enumerate(image_parts(value)):
        path = part.get("image")
        if not isinstance(path, str):
            raise ValueError(f"image part {i}: {part!r}; an image part is {{'type': 'image', 'image': '<file under "
                             "data/>'}}")
        file = os.path.join(root, "data", path)
        if not os.path.isfile(file):
            raise ValueError(f"image part {i}: no file data/{path} under the corpus root")
        with Image.open(file) as image:
            out.append(image.convert("RGB"))
    return out


def with_images(value: list[dict], loaded: list[Any]) -> list[dict]:
    """The chat with each image part's `image` replaced by its loaded image, in order, as the processor's chat
    template takes them."""
    given = iter(loaded)
    return [{**m, "content": [{**p, "image": next(given)} if isinstance(p, dict) and p.get("type") == "image" else p
                              for p in m["content"]]} if isinstance(m.get("content"), list) else m for m in value]


def encode(tokenizer: PreTrainedTokenizerBase, value: str | list[dict], template: dict | None,
           root: str | None = None, vision: Vision | None = None) -> dict:
    """One row's tokens: `ids`, each token's character `offsets` in `text` (None on a tokenizer without an
    offset mapping), for a chat each message's `turns` as `(role, start, end)` in the rendered text, and, on a
    checkpoint with a processor, its `images` (each where its tokens sit, `start` and `n`, and its token `grid`,
    rows by columns, where that is known) and the per-token tensors the processor made beside the ids
    (`per_token`: Qwen-VL's `mm_token_type_ids`, which the model needs back).

    Text is tokenized as the tokenizer does by default, BOS added where it adds one. A chat is rendered through
    the checkpoint's own template, given its dataset's `template` as arguments (none by default: no generation
    prompt), and the rendered text tokenized adding nothing, since the template writes its own BOS and markers.
    On a checkpoint loaded with its processor (`vision`), a chat renders and tokenizes through the processor
    instead, as the checkpoint does (it expands each image part into that image's tokens, as many as the image's
    size makes, and adds what it adds): the images are the files its parts name under the corpus root's `data/`.
    The tokens' offsets in the rendered text are then the plain tokenizer's, aligned token by token (`aligned`):
    an image's tokens all cover its placeholder, so a scope that spans the placeholder takes them.
    Every token either way is a position, a generation prompt's and an image's among them. A message's content
    begins where the template rendered (with the same arguments) with that content held by a marker shows, so a
    message that says its role's name (`user`) is not found in its role's header, and ends where the text after
    it begins; an image part's placeholder is inside its message's turn.
    """
    if isinstance(value, str):
        got = tokenizer(value, add_special_tokens=True, return_offsets_mapping=bool(tokenizer.is_fast))
        return {"ids": got["input_ids"], "offsets": got.get("offset_mapping"), "text": value, "turns": [],
                "images": [], "per_token": {}}
    given = template or {}
    parts = image_parts(value)
    if parts and vision is None:
        raise ValueError("the chat has an image part, and this model loaded no processor; load the checkpoint "
                         "with 'task': 'image-text-to-text' in the model block")
    renderer = vision["processor"] if vision is not None else tokenizer
    loaded = images(cast(str, root), value) if parts else []
    chat = with_images(value, loaded) if parts else value
    text = cast(str, renderer.apply_chat_template(chat, tokenize=False, **given))
    turns, cursor = [], 0
    for i, message in enumerate(value):
        # The content held by the marker, in the content's own shape: a string, or a list of parts.
        hold = [{"type": "text", "text": MARKER}] if isinstance(message["content"], list) else MARKER
        held = [{**m, "content": hold} if j == i else m for j, m in enumerate(chat)]
        marked = cast(str, renderer.apply_chat_template(held, tokenize=False, **given))
        own = marked.find(MARKER, cursor)
        end = len(text) - (len(marked) - own - len(MARKER))
        content = message["content"]
        if isinstance(content, str) and text[own:end] != content:
            own = text.find(content, max(cursor, own))
            end = own + len(content)
        if own < cursor or end < own:
            raise ValueError(f"message {i} ({message['role']}) {content!r} is not where the chat template put "
                             "the messages before it")
        turns.append((message["role"], own, end))
        cursor = end
    plain = tokenizer(text, add_special_tokens=False, return_offsets_mapping=bool(tokenizer.is_fast))
    if vision is None:
        return {"ids": plain["input_ids"], "offsets": plain.get("offset_mapping"), "text": text, "turns": turns,
                "images": [], "per_token": {}}
    got = vision["processor"].apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt", **given)
    ids = got["input_ids"][0].tolist()
    offsets = aligned(plain["input_ids"], plain["offset_mapping"], ids) if plain.get("offset_mapping") else None
    per_token = {k: v[0].tolist() for k, v in got.items() if k not in ("input_ids", "attention_mask")
                 and isinstance(v, torch.Tensor) and tuple(v.shape) == (1, len(ids)) and not v.is_floating_point()}
    found = runs(ids, vision["image_token_id"])
    placed = [{"path": part["image"], "start": start, "n": n, "grid": grid(got, i, n, vision["processor"])}
              for i, (part, (start, n)) in enumerate(zip(parts, found))] if len(found) == len(parts) else []
    note = None if len(found) == len(parts) else (f"its {sum(n for _, n in found)} image tokens come in "
                                                    f"{len(found)} runs for {len(parts)} image(s), so which are "
                                                    "which image's is unknown")
    return {"ids": ids, "offsets": offsets, "text": text, "turns": turns, "images": placed, "per_token": per_token,
            "chat": value, **({"image_note": note} if note else {})}


def aligned(plain: list[int], offsets: list[tuple[int, int]], ids: list[int]) -> list[tuple[int, int]]:
    """The processor's `ids` given character offsets in the rendered text from the plain tokenizer's `plain`
    tokens of it (with their `offsets`), aligned token by token: a token the two share keeps its span; one the
    processor put in place of a plain token (a newline it rewrote) takes the span of what it replaced; one it
    inserted (an image's tokens after its placeholder, a BOS it added) takes the span of the token before it, an
    empty span at the start where there is none; and a run of one repeated id that holds inserted tokens takes
    the span of the token the alignment matched, so an image's tokens all cover its placeholder whichever of
    them stood for it. A run the plain tokenizer has too keeps each token's own span."""
    out: list[tuple[int, int]] = [(0, 0)] * len(ids)
    kept = [False] * len(ids)                   # the tokens that stand for a plain token
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=plain, b=ids, autojunk=False).get_opcodes():
        if tag == "equal":
            out[j1:j2] = [tuple(o) for o in offsets[i1:i2]]
            kept[j1:j2] = [True] * (j2 - j1)
        elif tag == "replace":
            span = (min(o[0] for o in offsets[i1:i2]), max(o[1] for o in offsets[i1:i2]))
            out[j1:j2] = [span] * (j2 - j1)
            kept[j1:j2] = [True] * (j2 - j1)
        elif tag == "insert":
            span = out[j1 - 1] if j1 > 0 else (offsets[i1][0], offsets[i1][0]) if i1 < len(offsets) else (0, 0)
            out[j1:j2] = [span] * (j2 - j1)
    # A run of one repeated id that the processor put in (an image's tokens) is one thing: where the alignment
    # matched the placeholder to any token of the run, every token of the run takes the placeholder's span. A
    # run the plain tokenizer has too (repeated digits, newlines) keeps each token's own span.
    start = 0
    for end in range(1, len(ids) + 1):
        if end == len(ids) or ids[end] != ids[start]:
            found = next((out[j] for j in range(start, end) if kept[j]), None)
            if found is not None and not all(kept[start:end]):
                out[start:end] = [found] * (end - start)
            start = end
    return out


def runs(ids: list[int], image_token_id: int | None) -> list[tuple[int, int]]:
    """Each run of consecutive image tokens among `ids`, as `(start, n)`, in order: one per image where the
    processor writes an image's tokens together."""
    out, start = [], None
    for i, token in enumerate([*ids, None]):
        if token == image_token_id and start is None:
            start = i
        elif token != image_token_id and start is not None:
            out.append((start, i - start))
            start = None
    return out


def grid(got: Any, i: int, n: int, processor: Any) -> tuple[int, int] | None:
    """How image `i`'s `n` tokens tile the image, rows by columns, in raster order: from the processor's
    `image_grid_thw` (the Qwen ViTs: the patch grid, merged `merge_size` ways on each side) where it gives one,
    else a square where `n` is one (a fixed-resolution tower); None where neither holds."""
    thw = got.get("image_grid_thw")
    if thw is not None and i < len(thw):
        t, h, w = (int(x) for x in thw[i])
        merge = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
        rows, cols = t * (h // merge), w // merge
        return (rows, cols) if rows * cols == n else None
    side = math.isqrt(n)
    return (side, side) if side * side == n else None


def pixels(vision: Vision, root: str, rows: list[dict], template: dict | None) -> dict[str, torch.Tensor]:
    """The image tensors of a batch of encoded `rows` (each with its `chat`), made as `encode` made its ids,
    by the same processor call, every tensor but the ids' and the per-token ones concatenated along its first
    axis, row after row, the processor's own layout (pixels, grids, sizes); nothing for a batch without images."""
    out: dict[str, list[torch.Tensor]] = {}
    for row in rows:
        if not row.get("images") and not row.get("image_note"):
            continue
        chat = with_images(row["chat"], images(root, row["chat"]))
        got = vision["processor"].apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt",
                                                     **(template or {}))
        for key, value in got.items():
            if key in ("input_ids", "attention_mask") or key in row["per_token"] or not isinstance(value, torch.Tensor):
                continue
            out.setdefault(key, []).append(value)
    return {k: torch.cat(v) for k, v in out.items()}


def pad(tokenizer: PreTrainedTokenizerBase, ids: list[list[int]], width: int,
        per_token: list[dict[str, list[int]]] | None = None) -> dict[str, torch.Tensor]:
    """Rows of ids padded on the left, by hand, to `width`; the rows' `per_token` tensors (a processor's, beside
    the ids) padded the same way, with zeros.

    Padding by hand keeps the serving side's tokenizer unmutated. `position_ids` count real tokens only, and
    both engines pass them to the model (a model that lays its own positions out over an image's tokens is
    given none: `engines.standard.inputs`).
    """
    value = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
    input_ids = torch.tensor([[value] * (width - len(x)) + x for x in ids])
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])
    keys = sorted({key for row in per_token or [] for key in row})
    extra = {key: torch.tensor([[0] * (width - len(x)) + row.get(key, [0] * len(x))
                                for row, x in zip(cast(list, per_token), ids)]) for key in keys}
    return {"input_ids": input_ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
            **extra}


def compared(text: str, strip: bool, lower: bool) -> str:
    """`text` as the flags compare it, wherever text is compared (a scope's search, `contains`, and the lowering
    of an answer's class): stripped of surrounding whitespace under `strip`, lowercased under `lower` a character
    at a time, each kept as it is where lowering would change its length, so a span of the result is the text's."""
    text = text.strip() if strip else text
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text) if lower else text
