"""Positions: a `pos` as three stages of narrowing (frame, scope, index), resolved row by row inside the run,
where the tokenizer is, to the positions the record names.

A row's positions are its token indices in one sequence: the prompt's tokens (`0 .. P-1`, as encoded,
special ones included), then what it generated (`P + j` for generated token `j`). The prefill, call 0, holds
the prompt; decode call `k >= 1` holds generated token `k - 1`, its input. A row's calls run to the one that
chose its last token, so that token is never an input.
"""

import re

MISSING, AMBIGUOUS = "alignment_missing", "alignment_ambiguous"
TEXTUAL = ("column", "text")        # the scopes that locate text, which a row may or may not say


class Unaligned(Exception):
    """A row the record does not reach: its marker, and why, which in the prefill is a refusal."""

    def __init__(self, marker: str, why: str):
        super().__init__(why)
        self.marker = marker


def parse(pos) -> dict:
    """A `pos` (an int, or a `Pos`) as `{"frame", "scope", "index"}`."""
    if isinstance(pos, int):
        return {"frame": 0, "scope": None, "index": pos}
    scope = None if pos.scope is None else pos.scope.model_dump(exclude_none=True)
    return {"frame": pos.frame, "scope": scope, "index": pos.index}


def select(selector, n: int) -> list[int]:
    """The indices an int, a `[start, stop]` span (negative from the end, half-open) or `"all"` picks among
    `n` things."""
    if selector == "all":
        return list(range(n))
    start, stop = (selector, selector + 1 or n) if isinstance(selector, int) else selector
    start, stop = (start + n if start < 0 else start), (stop + n if stop < 0 else stop)
    return list(range(max(start, 0), min(stop, n)))


# -- what the run needs to know before it starts ------------------------------------------------------


def fixed(selector) -> int | None:
    """How many things a selector picks whatever their number: one for an int, a span's length when its ends
    count from the same side; None otherwise."""
    if isinstance(selector, int):
        return 1
    if isinstance(selector, list) and (selector[0] < 0) == (selector[1] < 0):
        return selector[1] - selector[0]
    return None


def width(pos: dict) -> int | None:
    """A record's width on every row, or None when it is ragged: its frame names calls (an int or a span,
    not a set each row has its own), its index picks a fixed number, and no text scope looks beyond the
    prefill."""
    textual = any(k in (pos["scope"] or {}) for k in TEXTUAL)
    if isinstance(pos["frame"], str) or textual and not prefill(pos):
        return None
    return fixed(pos["index"])


def reach(pos: dict) -> tuple[bool, list[int] | None, bool]:
    """Whether a record can name a prefill position on some row; the decode calls it can name (None: every
    one); and whether those are exactly the calls it names on each row that gets that far, which a write
    needs, landing as each call runs. Not exact: a frame counted from the row's last call or `eos`, an index
    into a frame holding the prompt (whose length differs by row), a negative index over several calls."""
    frame, index = pos["frame"], pos["index"]
    ends = {"generated": [1, None], "all": [0, None]}.get(frame) if isinstance(frame, str) else \
        [frame, frame + 1] if isinstance(frame, int) else frame
    if ends is None or ends[0] < 0 or (ends[1] or 0) < 0:
        return True, None, False
    start, stop = ends
    calls = None if stop is None else list(range(max(start, 1), stop))
    if start == 0 or index == "all":
        return start == 0, calls, index == "all" or calls == []
    single, ends = stop is not None and stop - start == 1, index if isinstance(index, list) else [index]
    if not single and min(ends) < 0:
        return False, calls, False
    picked = [start + i for i in select(index, 1 if single else max(ends) + 1)]
    return False, [k for k in picked if stop is None or k < stop], True


def prefill(pos: dict) -> bool:
    """Whether the record names prefill positions only, so the data alone decides it before any forward."""
    holds, calls, _ = reach(pos)
    return holds and calls == []


# -- the three stages ---------------------------------------------------------------------------------


def framed(frame, n: int, stopped: bool) -> list[int]:
    """Stage one: the calls `frame` picks among a row's `n` calls (`stopped`: its last one chose EOS)."""
    if frame == "generated":
        return list(range(1, n))
    if frame == "eos":
        return [n - 1] if stopped else []
    return select(frame, n)


def held(called: list[int], prompted: list[int], prompt: int) -> list[int]:
    """The positions the calls hold: the prefill's (`prompted`), and each decode call's input, in a row whose
    prompt is `prompt` tokens."""
    return [p for call in called for p in (prompted if call == 0 else [prompt + call - 1])]


def call_of(position: int, prompt: int) -> int:
    """The call that holds a position, the inverse of `held`."""
    return 0 if position < prompt else position - prompt + 1


def scoped(scope: dict | None, tokens: list[int], occurrences: list[set[int]]) -> list[int]:
    """Stage two: the tokens that overlap the one place the scope's text (or turn) occurs among them."""
    if scope is None:
        return tokens
    found = [o for o in occurrences if o & set(tokens)]
    what = f"{next(iter(scope))} {next(iter(scope.values()))!r}"
    if not found:
        raise Unaligned(MISSING, f"{what} is not in its input")
    if len(found) > 1:
        raise Unaligned(AMBIGUOUS, f"{what} occurs {len(found)} times in the row's input; a scope is one place")
    return [t for t in tokens if t in found[0]]


def indexed(index, tokens: list[int], width: int | None) -> list[int]:
    """Stage three: the positions `index` picks among the scope's tokens; a rectangular record's must all
    be there."""
    got = [tokens[i] for i in select(index, len(tokens))]
    if width is not None and len(got) != width:
        raise Unaligned(MISSING, f"tokens {index} do not fit its {len(tokens)} scope tokens")
    return got


def resolve(pos: dict, prompt: int, prompted: list[int], said: list[int], stopped: bool,
            occurrences: list[set[int]]) -> list[int] | str:
    """The positions a record names in one row, given its prompt's length, the positions its prefill holds
    (`prompted`), what it generated up to its first EOS (`said`, `stopped`), and where its scope's text occurs
    (`occurrences`); or why it names none, a marker. In the prefill that is a refusal: the data decided."""
    try:
        tokens = held(framed(pos["frame"], max(len(said), 1), stopped), prompted, prompt)
        got = indexed(pos["index"], scoped(pos["scope"], tokens, occurrences), width(pos))
    except Unaligned as e:
        if prefill(pos):
            raise ValueError(str(e)) from None
        return e.marker
    return got if got or prefill(pos) else MISSING


# -- where a scope's text occurs ----------------------------------------------------------------------


def occurrences(scope: dict | None, row: dict, said: list[int], tokenizer, text: str | None) -> list[set[int]]:
    """Each place the scope names, as the positions that overlap it: a chat turn of the prompt, or each
    occurrence of the text (`text`: the column's or the literal) in the prompt, by the tokenizer's offsets,
    and in the generated inputs, by decoding growing prefixes. A token with an empty span (part of a split
    character) covers the character it begins."""
    if scope is None:
        return []
    prompt, inputs = len(row["ids"]), said[:-1]
    spans = turns(row, scope["turn"]) if "turn" in scope else \
        [m.span() for m in re.finditer(re.escape(text), row["text"])] if text else []
    if spans and row["offsets"] is None:
        raise ValueError("this tokenizer has no offset mapping, which a scope in the prompt needs")
    found = [{t for t, (x, y) in enumerate(row["offsets"]) if x < b and y > a} for a, b in spans]
    if any(k in scope for k in TEXTUAL) and text and inputs:
        words, chars = char_spans(tokenizer, inputs)
        found += [{prompt + j for j, (x, y) in enumerate(chars) if x < m.end() and max(y, x + 1) > m.start()}
                  for m in re.finditer(re.escape(text), words)]
    return found


def lands(text: str, inputs: list[int], eos: set[int], tokenizer) -> bool:
    """Whether a write scoped to `text` lands at the call fed `inputs[-1]`, given every input the row was fed
    so far: the first place the text occurs in them ends on that token, and the row has not stopped."""
    if not text or any(t in eos for t in inputs):
        return False
    words, chars = char_spans(tokenizer, inputs)
    m = re.search(re.escape(text), words)
    return m is not None and max(j for j, (x, y) in enumerate(chars)
                                 if x < m.end() and max(y, x + 1) > m.start()) == len(inputs) - 1


def turns(row: dict, turn: str) -> list[tuple[int, int]]:
    """The character spans of the chat turns `turn` names: `role[n]` the n-th of that role, `role` all of them."""
    role, n = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", turn).groups("")  # type: ignore[union-attr]
    spans = [(a, b) for r, a, b in row["turns"] if r == role]
    return spans[int(n):int(n) + 1] if n else spans


def continuation(ids: list[int], eos: set[int]) -> list[int]:
    """What a row generated: its new ids up to and including its first EOS."""
    return next((ids[:i + 1] for i, token in enumerate(ids) if token in eos), ids)


def char_spans(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """The text `ids` decode to, and each token's character span in it, found by decoding growing prefixes.

    A token ends where its prefix's text stops agreeing with the whole text, so a character split across
    tokens (a multi-byte one, which a partial prefix decodes to `\\ufffd`) is the span of the token that
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
