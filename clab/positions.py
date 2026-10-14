"""Positions: a `pos` as three stages of narrowing (frame, scope, index), resolved row by row inside the run,
where the tokenizer is, to the positions the record names.

A row's positions are its token indices in one sequence: the prompt's tokens (`0 .. P-1`, as encoded,
special ones included), then what it generated (`P + j` for generated token `j`). The prefill, call 0, holds
the prompt; decode call `k >= 1` holds generated token `k - 1`, its input, at index `P + k - 1`. Calls up to
the one that chose the row's last token are the row's: calls `0 .. L-1` for a row that generated `L` tokens.
"""

import re

MISSING, AMBIGUOUS = "alignment_missing", "alignment_ambiguous"


def parse(pos) -> dict:
    """A `pos` (an int, or a `Pos`) as `{"frame", "scope", "index"}`, spans as lists."""
    if isinstance(pos, int):
        return {"frame": 0, "scope": None, "index": pos}
    def plain(value):
        return list(value) if isinstance(value, tuple) else value
    scope = None if pos.scope is None else pos.scope.model_dump(exclude_none=True)
    return {"frame": plain(pos.frame), "scope": scope, "index": plain(pos.index)}


def ragged(pos: dict) -> bool:
    """Whether a record's width can differ between rows, from its three selectors alone. It is rectangular
    when the frame is one call or a span of calls, the index an int or a span whose ends count from the same
    side, and no `column` or `text` scope looks outside the prefill."""
    frame, index, scope = pos["frame"], pos["index"], pos["scope"] or {}
    fixed = isinstance(index, int) or isinstance(index, list) and (index[0] < 0) == (index[1] < 0)
    texted = ("column" in scope or "text" in scope) and frame != 0
    return not (fixed and isinstance(frame, (int, list)) and not texted)


def width(pos: dict) -> int:
    """A rectangular record's width: one for an int index, the span's length otherwise."""
    index = pos["index"]
    return 1 if isinstance(index, int) else index[1] - index[0]


def prefill(pos: dict) -> bool:
    """Whether the record looks only at the prefill, so the data alone decides it, before any forward."""
    return pos["frame"] == 0


def calls(pos: dict) -> list[int] | None:
    """The decode calls a record must reach on every row, known before the run: a fixed frame's own calls
    (`[]` for the prefill alone), or None when which calls it means depends on how far each row got."""
    frame = pos["frame"]
    if isinstance(frame, int) and frame >= 0:
        return [frame] if frame else []
    if isinstance(frame, list) and min(frame) >= 0:
        return [k for k in range(*frame) if k]
    return None


def with_prefill(pos: dict) -> bool:
    """Whether the record can name a position of the prefill on some row."""
    frame = pos["frame"]
    if isinstance(frame, list):
        return frame[0] <= 0
    return frame == "all" or frame == "eos" or isinstance(frame, int) and frame <= 0


# -- the three stages ---------------------------------------------------------------------------------


def bounds(selector, n: int) -> tuple[int, int]:
    """An int or a `[start, stop]` span (negative from the end) as indices into `n` things."""
    start, stop = (selector, None if selector == -1 else selector + 1) if isinstance(selector, int) else selector
    start = start + n if start < 0 else start
    stop = n if stop is None else (stop + n if stop < 0 else stop)
    return start, stop


def framed(frame, n: int, stopped: bool) -> list[int]:
    """Stage one: the calls `frame` picks among a row's `n` calls; `stopped` says whether its last one chose
    EOS."""
    if frame == "generated":
        return list(range(1, n))
    if frame == "all":
        return list(range(n))
    if frame == "eos":
        return [n - 1] if stopped else []
    start, stop = bounds(frame, n)
    return list(range(max(start, 0), min(stop, n)))


def held(called: list[int], row: dict, said: list[int], special: set, head_only: bool) -> list[int]:
    """The positions the calls hold: the prefill's prompt tokens that are not special (only its last token,
    `head_only`, where the prefill computes the head at its last position alone), each decode call's input."""
    prompt = len(row["ids"])
    out = []
    for call in called:
        if call == 0:
            out += [prompt - 1] if head_only else [i for i, t in enumerate(row["ids"]) if t not in special]
        elif call - 1 < len(said):
            out.append(prompt + call - 1)
    return out


def scoped(scope: dict | None, tokens: list[int], row: dict, said: list[int], tokenizer, text: str | None,
           prefilled: bool) -> list[int] | str:
    """Stage two: the tokens of `tokens` that overlap the one place the scope's text occurs (`text`: the
    column's or the literal) or the chat turn it names. Text that is absent or occurs twice is refused in the
    prefill and a result (`MISSING`, `AMBIGUOUS`) elsewhere. The prompt's text is located by the tokenizer's
    offsets; the generated inputs' by decoding growing prefixes."""
    if scope is None:
        return tokens
    prompt, inputs, spans = len(row["ids"]), [], []
    if "turn" in scope:
        found, what = [("prompt", a, b) for a, b in turns(row, scope["turn"])], f"turn {scope['turn']!r}"
    else:
        what = f"{'column ' + repr(scope['column']) if 'column' in scope else 'text'} ({text!r})"
        looked = any(t < prompt for t in tokens)
        found = [("prompt", m.start(), m.end()) for m in re.finditer(re.escape(text), row["text"])] \
            if text and looked else []
        inputs = [t - prompt for t in tokens if t >= prompt]
        if inputs and text:
            words, spans = char_spans(tokenizer, [said[j] for j in inputs])
            found += [("said", m.start(), m.end()) for m in re.finditer(re.escape(text), words)]
    if len(found) != 1:
        if prefilled:
            raise ValueError(f"{what} occurs {len(found)} times in the row's input; a scope is one place"
                             if found else f"{what} is not in its input")
        return AMBIGUOUS if found else MISSING
    (part, a, b), = found
    if part == "prompt":
        if row["offsets"] is None:
            raise ValueError("this tokenizer has no offset mapping, which a scope in the prompt needs")
        return [t for t in tokens if t < prompt and row["offsets"][t][0] < b and row["offsets"][t][1] > a]
    # A token with an empty span (part of a split character) covers the character it begins.
    return [prompt + j for j, (x, y) in zip(inputs, spans) if x < b and max(y, x + 1) > a]


def indexed(index, tokens: list[int], rectangular: bool, prefilled: bool) -> list[int] | str:
    """Stage three: the positions `index` picks among the scope's tokens. A rectangular index that does not
    fit is refused in the prefill and missing elsewhere; a ragged one takes what is there."""
    if index == "all":
        got = tokens
    else:
        start, stop = bounds(index, len(tokens))
        if rectangular and not 0 <= start < stop <= len(tokens):
            if prefilled:
                raise ValueError(f"tokens {index} do not fit its {len(tokens)} scope tokens")
            return MISSING
        got = tokens[max(start, 0):max(stop, 0)]
    return got if got or prefilled else MISSING


def resolve(pos: dict, row: dict, said: list[int] | None, eos: set[int], special: set, tokenizer,
            text: str | None, head_only: bool = False) -> list[int] | str:
    """The positions a record names in one row, given its encoded prompt (`data.encode`) and, in a
    generation, its new ids `said` (None in a forward); or why it names none (`MISSING`, `AMBIGUOUS`)."""
    said = continuation(said or [], eos)
    stopped = bool(said) and said[-1] in eos
    called = framed(pos["frame"], max(len(said), 1), stopped)
    if not called:
        return MISSING
    tokens = held(called, row, said, special, head_only)
    tokens = scoped(pos["scope"], tokens, row, said, tokenizer, text, prefill(pos))
    if isinstance(tokens, str):
        return tokens
    return indexed(pos["index"], tokens, not ragged(pos), prefill(pos))


# -- the text a scope names ------------------------------------------------------------------------------


def turns(row: dict, turn: str) -> list[tuple[int, int]]:
    """The character spans of the chat turns `turn` names: `role[n]` the n-th of that role, `role` all of them."""
    role, n = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", turn).groups("")  # type: ignore[union-attr]
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
