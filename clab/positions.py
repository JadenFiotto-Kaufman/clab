"""Positions: a `pos` as three stages of narrowing (frame, scope, index), resolved row by row inside the run,
where the tokenizer is, to the positions the record names.

A row's positions are its token indices in one sequence: the prompt's tokens (`0 .. P-1`, as encoded,
special ones included), then what it generated (`P + j` for generated token `j`). The prefill, call 0, holds
the prompt; decode call `k >= 1` holds generated token `k - 1`, its input. A row's calls run to the one that
chose its last token, so that token is never an input.
"""

import re

MISSING = "alignment_missing"
TEXTUAL = ("column", "text")        # the scopes that locate text, which a row may or may not say
UNPROMPTED = {"ids": [], "offsets": [], "text": "", "turns": []}    # an encoded empty prompt


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
    not a set each row has its own), its index picks a fixed number, and a text scope, if any, is one
    occurrence in the prefill."""
    scope = pos["scope"] or {}
    textual = any(k in scope for k in TEXTUAL)
    if isinstance(pos["frame"], str) or textual and (not prefill(pos) or fixed(scope["occurrence"]) != 1):
        return None
    return fixed(pos["index"])


def reach(pos: dict) -> tuple[bool, list[int] | None, bool]:
    """Whether a record can name a prefill position on some row; the decode calls it can name (None: every
    one); and whether those are exactly the calls it names on each row that gets that far, which a write
    needs, landing as each call runs. Not exact: a frame counted from the row's last call or `eos`, an index
    into a frame holding the prompt (whose length differs by row), a negative index over several calls."""
    frame, index = pos["frame"], pos["index"]
    span = {"generated": [1, None], "all": [0, None]}.get(frame) if isinstance(frame, str) else \
        [frame, frame + 1] if isinstance(frame, int) else frame
    if span is None or span[0] < 0 or (span[1] or 0) < 0:
        return True, None, False
    start, stop = span
    calls = None if stop is None else list(range(max(start, 1), stop))
    if start == 0 or index == "all":
        return start == 0, calls, index == "all" or calls == []
    single, bounds = stop is not None and stop - start == 1, index if isinstance(index, list) else [index]
    if not single and min(bounds) < 0:
        return False, calls, False
    picked = [start + i for i in select(index, 1 if single else max(bounds) + 1)]
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
    """Stage two: the tokens of the occurrences `occurrence` picks among those of the scope's text among
    `tokens` (a turn is one place), in order."""
    if scope is None:
        return tokens
    found = [o for o in occurrences if o & set(tokens)]
    which = scope.get("occurrence", 0)
    picked = select(which, len(found))
    what = f"{next(iter(scope))} {next(iter(scope.values()))!r}"
    if not found:
        raise Unaligned(MISSING, f"{what} is not in its input")
    if "turn" in scope and len(found) > 1:
        raise Unaligned(MISSING, f"{what} names {len(found)} turns of the chat; a turn is one place")
    if not picked or fixed(which) is not None and len(picked) != fixed(which):
        raise Unaligned(MISSING, f"{what} occurs {len(found)} times in the row's input; it has no occurrence {which}")
    return [t for t in tokens if any(t in found[i] for i in picked)]


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


def occurrences(scope: dict | None, row: dict, said: list[int], tokenizer, texts: list[str]) -> list[set[int]]:
    """Each place the scope names, as the positions that overlap it: a chat turn of the prompt, or each
    occurrence of any of its texts (`texts`: the columns' or the literals) in the prompt, by the tokenizer's
    offsets, then in the generated inputs, by decoding growing prefixes; each part's in order of position
    (end, then start). A token with an empty span (part of a split character) covers the character it begins."""
    if scope is None:
        return []
    prompt, inputs = len(row["ids"]), said[:-1]
    spans = turns(row, scope["turn"]) if "turn" in scope else located(scope, texts, row["text"])
    if spans and row["offsets"] is None:
        raise ValueError("this tokenizer has no offset mapping, which a scope in the prompt needs")
    found = [{t for t, (x, y) in enumerate(row["offsets"]) if x < b and y > a} for a, b in spans]
    if "turn" not in scope and inputs:
        words, chars = char_spans(tokenizer, inputs)
        found += [covering(chars, a, b, prompt) for a, b in located(scope, texts, words)]
    return found


def located(scope: dict, texts: list[str], words: str) -> list[tuple[int, int]]:
    """Every character span in `words` where one of `texts` occurs (each stripped first under `strip`), in
    order of end, then start: where an occurrence becomes known, so a write found online, a read and the run
    record count them alike."""
    texts = [t.strip() if scope.get("strip") else t for t in texts]
    spans = [m.span() for t in texts if t for m in re.finditer(re.escape(t), words)]
    return sorted(spans, key=lambda span: (span[1], span[0]))


def covering(chars: list[tuple[int, int]], a: int, b: int, offset: int = 0) -> set[int]:
    """The tokens (`offset` + their index) whose spans overlap characters `a` to `b`; a token with an empty
    span covers the character it begins."""
    return {offset + j for j, (x, y) in enumerate(chars) if x < b and max(y, x + 1) > a}


def landing(pos: dict, texts: list[str], inputs: list[int], eos: set[int], tokenizer) -> bool:
    """Whether a write with a text scope lands at the call fed `inputs[-1]`, given every input the row was fed
    so far: some occurrence its `occurrence` picks resolves, with index -1, to that token, on the prefix and
    counted only within the frame's calls, as `resolve` counts them; so each picked occurrence lands at its own
    last token, as it appears. The prefix is the row's calls so far (the last is the one being fed, whose token
    is not chosen yet) after an empty prompt, so a row that stopped has no such call and input `j` sits at `j`."""
    said = continuation([*inputs, -1], eos)
    found = occurrences(pos["scope"], UNPROMPTED, said, tokenizer, texts)
    return any(resolve({**pos, "scope": {**pos["scope"], "occurrence": i}}, 0, [], said, False, found)
               == [len(inputs) - 1] for i in select(pos["scope"]["occurrence"], len(found)))


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
