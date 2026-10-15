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


class Unaligned(ValueError):
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


# -- selectors ----------------------------------------------------------------------------------------

SLICE = r"^(-?[0-9]+)?:(-?[0-9]+)?(:(-?[0-9]+)?)?$"
FORMS = "an int (2, -1), a slice in Python notation ('2:5', '-3:', ':') or a list of picks ([1, 3, 7])"


def checked(selector):
    """A selector as written, refused unless it is an int, a slice string or a list of distinct int picks."""
    if isinstance(selector, bool) or not isinstance(selector, (int, str, list)):
        raise ValueError(f"{selector!r} is not a selector: {FORMS}")
    if isinstance(selector, str):
        sliced(selector)
    if isinstance(selector, list):
        if not selector or not all(isinstance(i, int) and not isinstance(i, bool) for i in selector):
            raise ValueError(f"{selector!r} is not a selector: {FORMS}; a list of picks names at least one int")
        if len(set(selector)) < len(selector):
            raise ValueError(f"picks {selector} name {next(i for i in selector if selector.count(i) > 1)} twice")
    return selector


def sliced(text: str) -> slice:
    """A slice string as the `slice` Python reads it: `start:stop:step`, each part an int or empty."""
    if not re.match(SLICE, text):
        raise ValueError(f"{text!r} is not a selector: {FORMS}")
    parts = [int(x) if x else None for x in (text.split(":") + [""])[:3]]
    if parts[2] == 0:
        raise ValueError(f"{text!r}: a slice's step is not 0")
    return slice(*parts)


def select(selector, n: int) -> list[int]:
    """The indices a selector picks among `n` things, in the order it names them: an int, the one it counts
    (negative from the end), if there is one; a slice string, its range by Python's rules; a list of picks,
    each one there is, as written. Two picks that count the same thing (`[0, -2]` of two) are refused."""
    if isinstance(selector, str):
        return list(range(n)[sliced(selector)])
    picks = [selector] if isinstance(selector, int) else selector
    got = [i + n if i < 0 else i for i in picks if -n <= i < n]
    if len(set(got)) < len(got):
        raise Unaligned(MISSING, f"picks {selector} count one of these {n} twice")
    return got


def fixed(selector) -> int | None:
    """How many things a selector picks whatever their number (its static width), or None when that grows
    with the number. One for an int; a list of picks, its length; a slice, its length when its two ends
    count from the same side: both from the start (`"2:5"`, `":3"`, `"0:6:2"`) or both from the end
    (`"-4:-1"`, `"-3:"`), an omitted start counting from the start and an omitted stop from the end (with a
    negative step, both ends written). `":"` and `"2:"` have none. Where there are fewer things than a
    static width needs, the record does not fit that row, which is a refusal or `alignment_missing`."""
    if isinstance(selector, int):
        return 1
    if isinstance(selector, list):
        return len(selector)
    s = sliced(selector)
    start, stop, step = s.start, s.stop, s.step or 1
    if step > 0:
        start = start or 0
        if stop is None:                    # to the end: as wide as a start counted from the end leaves
            return len(range(start, 0, step)) if start < 0 else None
    if start is None or stop is None or (start < 0) != (stop < 0):
        return None
    return len(range(start, stop, step))


def from_end(selector) -> bool:
    """Whether what a selector picks depends on how many things there are: a negative int or pick, a slice with
    a negative end or step."""
    if isinstance(selector, int):
        return selector < 0
    if isinstance(selector, list):
        return min(selector) < 0
    s = sliced(selector)
    return any(x is not None and x < 0 for x in (s.start, s.stop, s.step))


def every(selector) -> bool:
    """Whether a selector picks every one of however many things there are (`":"`)."""
    if not isinstance(selector, str):
        return False
    s = sliced(selector)
    return s.start in (None, 0) and s.stop is None and s.step in (None, 1)


def extent(selector) -> int:
    """How many things a selector counted from the start (`from_end` false) and with an end reaches over."""
    if isinstance(selector, str):
        return sliced(selector).stop or 0
    return (max(selector) if isinstance(selector, list) else selector) + 1


# -- what the run needs to know before it starts ------------------------------------------------------


def bounded(frame) -> bool:
    """Whether a frame names calls by an end it gives, the same calls on every row that gets that far: an int,
    a list of picks, a slice with a stop; not `"generated"`, `"eos"` or a slice to the row's last call."""
    return not isinstance(frame, str) or frame not in ("generated", "eos") and sliced(frame).stop is not None


def width(pos: dict) -> int | None:
    """A record's width on every row, or None when it is ragged: its frame is `bounded`, its index has a
    static width (`fixed`), and a scope, if any, is one occurrence in the prefill."""
    scope = pos["scope"]
    if not bounded(pos["frame"]) or scope and (not prefill(pos) or fixed(scope.get("occurrence", 0)) != 1):
        return None
    return fixed(pos["index"])


def reach(pos: dict) -> tuple[bool, list[int] | None, bool]:
    """Whether a record can name a prefill position on some row; the decode calls it can name (None: every
    one), in order; and whether those are exactly the calls it names on each row that gets that far, which a
    write needs, landing as each call runs. Not exact: a frame counted from the row's last call or `eos`, an
    index into a frame holding the prompt (whose length differs by row), an index counted from the end over
    several calls, a frame to the row's last call from past call 1 or by a step, and an index without a stop
    over one."""
    frame, index = ("1:" if pos["frame"] == "generated" else pos["frame"]), pos["index"]
    if frame == "eos" or from_end(frame):
        return True, None, False
    if bounded(frame):
        called = select(frame, extent(frame))
        decode = sorted(k for k in called if k)
        if 0 in called or every(index):
            return 0 in called, decode, every(index) or decode == []
        if from_end(index) and len(called) > 1:
            return False, decode, False
        return False, sorted(called[i] for i in select(index, len(called))), True
    s = sliced(frame)                               # every call from the slice's start, by its step
    start = s.start or 0
    if s.step not in (None, 1) or start > 1:
        return start == 0, None, False
    if start == 0 or every(index):
        return start == 0, None, every(index)
    if from_end(index) or isinstance(index, str) and sliced(index).stop is None:
        return False, None, False
    return False, sorted(1 + i for i in select(index, extent(index))), True


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
    """Stage two: the tokens of the occurrences `occurrence` picks among those of the scope's text (or its
    role's messages) among `tokens`, in order."""
    if scope is None:
        return tokens
    found = [o for o in occurrences if o & set(tokens)]
    which = scope.get("occurrence", 0)
    picked = select(which, len(found))
    what = f"{next(iter(scope))} {next(iter(scope.values()))!r}"
    if not found:
        raise Unaligned(MISSING, f"the row has no {scope['turn']!r} message" if "turn" in scope else
                        f"{what} is not in its input")
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


def turns(row: dict, role: str) -> list[tuple[int, int]]:
    """The character spans of the row's messages of `role`, as the chat template rendered them: in order, so
    ordered by where each ends, as `located` orders text."""
    return [(a, b) for r, a, b in row["turns"] if r == role]


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
