"""Positions: a `pos` as a slice of a scope's tokens, and that slice resolved to token indices, row by row,
inside the run (where the tokenizer is): in the prompt frame from the row's encoding, in the generated frame
from what the row generated."""

import re

MISSING, AMBIGUOUS = "alignment_missing", "alignment_ambiguous"


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
    """Whether the slice's width can differ between rows: one end counted from the start, one from the end. In
    the generated frame every row's continuation has its own length, so a slice is the same width on every
    row only unscoped and counted from the start at both ends."""
    start, stop = pos["start"], pos["stop"]
    if pos["frame"] == "generated":
        return pos["scope"] is not None or start < 0 or stop is None or stop < 0
    return (start < 0) != (stop is None or stop < 0)


def asked(pos: dict) -> list[int] | None:
    """The decode steps a generated-frame slice reads: its own when it is rectangular; every step (None) when
    which ones it means depends on what each row said."""
    return None if ragged(pos) else list(range(pos["start"], pos["stop"]))


def bounds(pos: dict, n: int) -> tuple[int, int]:
    """The slice's ends as indices into a scope of `n` tokens."""
    start = pos["start"] + n if pos["start"] < 0 else pos["start"]
    stop = n if pos["stop"] is None else (pos["stop"] + n if pos["stop"] < 0 else pos["stop"])
    return start, stop


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
    start, stop = bounds(pos, n)
    if not ragged(pos) and not 0 <= start < stop <= n:
        raise ValueError(f"tokens [{pos['start']}:{pos['stop']}] do not fit its {n} scope tokens")
    return tokens[max(start, 0):max(stop, 0)]


def turns(row: dict, segment: str) -> list[tuple[int, int]]:
    """The character spans of the chat turns `segment` names: `role[n]` the n-th of that role, `role` all of them."""
    role, n = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", segment).groups("")  # type: ignore[union-attr]
    spans = [(a, b) for r, a, b in row["turns"] if r == role]
    return spans[int(n):int(n) + 1] if n else spans


def continuation(ids: list[int], eos: set[int]) -> list[int]:
    """What a row generated: its new ids up to and including its first EOS."""
    for i, token in enumerate(ids):
        if token in eos:
            return ids[:i + 1]
    return ids


def char_spans(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """The text `ids` decode to, and each token's character span in it, found by decoding growing prefixes.

    A token ends where its prefix's text stops agreeing with the whole text, so a character split across
    tokens (a multi-byte one, which a partial prefix decodes to `\ufffd`) is the span of the token that
    completes it, and the tokens before that one have empty spans at the character's start.
    """
    text = tokenizer.decode(ids)
    spans, start = [], 0
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1])
        end = next((j for j, (a, b) in enumerate(zip(prefix, text)) if a != b), min(len(prefix), len(text)))
        spans.append((start, max(start, end)))
        start = max(start, end)
    return text, spans


def generated(pos: dict, ids: list[int], eos: set[int], tokenizer, variable: str | None) -> list[int] | str:
    """The decode steps a generated-frame slice names in one row, from the row's new `ids`, or why it names
    none: `alignment_missing` or `alignment_ambiguous`, a result rather than an error.

    Step *k* is the forward that chose new token *k*, so the row's steps index its continuation (its new
    tokens up to and including its first EOS). A `variable` scope is the steps whose tokens overlap the one
    place the row's own text of that column occurs in what it said (its continuation decoded, without the
    EOS; a token with an empty span, part of a split character, covers the character it begins): missing
    where it never occurs, ambiguous where it occurs twice. `segment: eos` is the step that
    chose the row's first EOS, missing when the row never stopped. A slice the row's steps do not reach, or
    that holds none of them, is missing.
    """
    said = continuation(ids, eos)
    stopped = bool(said) and said[-1] in eos
    steps = list(range(len(said)))
    scope = pos["scope"] or {}
    if scope.get("segment") == "eos":
        steps = steps[-1:] if stopped else []
    elif "variable" in scope:
        text, spans = char_spans(tokenizer, said[:-1] if stopped else said)
        found = [(m.start(), m.end()) for m in re.finditer(re.escape(variable), text)] if variable else []
        if len(found) != 1:
            return AMBIGUOUS if found else MISSING
        (a, b), = found
        steps = [i for i, (x, y) in enumerate(spans) if x < b and max(y, x + 1) > a]
    start, stop = bounds(pos, len(steps))
    if not ragged(pos) and stop > len(steps):
        return MISSING
    return steps[max(start, 0):max(stop, 0)] or MISSING
