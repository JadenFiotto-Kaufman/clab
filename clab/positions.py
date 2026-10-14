"""Positions: a `pos` as a slice of a scope's tokens, and that slice resolved to token indices, row by row,
inside the run (where the tokenizer is)."""

import re


def parse(pos) -> dict:
    """A `pos` (an int, or a `Pos`) as `{"frame", "scope", "start", "stop"}`: a half-open slice of the scope's
    tokens, `stop` None for "to the end"."""
    if isinstance(pos, int):
        return {"frame": "prompt", "scope": None, "start": pos, "stop": None if pos == -1 else pos + 1}
    scope = None if pos.scope is None else pos.scope.model_dump(exclude_none=True)
    if pos.index is not None:
        start, stop = pos.index, (None if pos.index == -1 else pos.index + 1)
    elif pos.span is not None:
        start, stop = pos.span
    else:
        start, stop = 0, None
    return {"frame": pos.frame, "scope": scope, "start": start, "stop": stop}


def ragged(pos: dict) -> bool:
    """Whether the slice's width can differ between rows: one end counted from the start, one from the end."""
    start, stop = pos["start"], pos["stop"]
    return (start < 0) != (stop is None or stop < 0)


def resolve(pos: dict, row: dict, special: set, variable: str | None) -> list[int] | None:
    """The slice's token indices in one encoded row (`data.encode`), or None when its scope is not in the row.

    The prompt's tokens are the row's tokens that are not special (so BOS is never one). A `variable` scope
    is the prompt tokens overlapping the one place the row's own text of that column occurs in its input;
    a `segment` scope those overlapping a chat turn (`user[1]` is the second user message; a bare `user` is
    the only one). Text that occurs twice is ambiguous and refused. A rectangular slice must fit the scope;
    a ragged one takes what is there, possibly nothing.
    """
    tokens = [i for i, t in enumerate(row["ids"]) if t not in special]
    scope = pos["scope"]
    if scope is not None:
        if "segment" in scope:
            spans, what = turns(row, scope["segment"]), f"segment {scope['segment']!r}"
        else:
            spans = [(m.start(), m.end()) for m in re.finditer(re.escape(variable), row["text"])] if variable else []
            what = f"variable {scope['variable']!r} ({variable!r})"
        if not spans:
            return None
        if len(spans) > 1:
            raise ValueError(f"{what} occurs {len(spans)} times in the row's input; a scope is one place")
        if row["offsets"] is None:
            raise ValueError("this tokenizer has no offset mapping, which a scope needs")
        (a, b), = spans
        tokens = [i for i in tokens if row["offsets"][i][0] < b and row["offsets"][i][1] > a]
    n = len(tokens)
    start = pos["start"] + n if pos["start"] < 0 else pos["start"]
    stop = n if pos["stop"] is None else (pos["stop"] + n if pos["stop"] < 0 else pos["stop"])
    if not ragged(pos) and not 0 <= start < stop <= n:
        raise ValueError(f"tokens [{pos['start']}:{pos['stop']}] do not fit its {n} scope tokens")
    return tokens[max(start, 0):max(stop, 0)]


def turns(row: dict, segment: str) -> list[tuple[int, int]]:
    """The character spans of the chat turns `segment` names: `role[n]` the n-th of that role, `role` all of them."""
    role, n = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", segment).groups("")  # type: ignore[union-attr]
    spans = [(a, b) for r, a, b in row["turns"] if r == role]
    return spans[int(n):int(n) + 1] if n else spans
