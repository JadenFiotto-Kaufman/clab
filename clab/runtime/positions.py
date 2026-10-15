"""Positions: a `pos` as three stages of narrowing (frame, scope, index), resolved row by row inside the run,
where the tokenizer is, to the positions the record names.

A row's positions are its token indices in one sequence: the prompt's tokens (`0 .. P-1`, as encoded,
special ones included), then what it generated (`P + j` for generated token `j`). The prefill, call 0, holds
the prompt; decode call `k >= 1` holds generated token `k - 1`, its input. A row's calls run to the one that
chose its last token, so that token is never an input.
"""

import re
import sys

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

# A slice string: each part an int or empty, no `-0`, no step 0.
SLICE = r"^(0|-?[1-9][0-9]*)?:(0|-?[1-9][0-9]*)?(:(-?[1-9][0-9]*)?)?$"
FORMS = "an int (2, -1), a slice in Python notation ('2:5', '-3:', ':') or a list of picks ([1, 3, 7])"


def checked(selector):
    """A selector as written, refused unless it is an int, a slice string or a list of distinct int picks, and
    unless it can pick something: a slice whose static width is 0 (`"0:0"`, `"5:2"`) picks nothing ever."""
    if isinstance(selector, bool) or not isinstance(selector, (int, str, list)):
        raise ValueError(f"{selector!r} is not a selector: {FORMS}")
    if isinstance(selector, str) and not re.match(SLICE, selector):
        raise ValueError(f"{selector!r} is not a selector: {FORMS}")
    if isinstance(selector, list):
        if not selector or not all(isinstance(i, int) and not isinstance(i, bool) for i in selector):
            raise ValueError(f"{selector!r} is not a selector: {FORMS}; a list of picks names at least one int")
        if len(set(selector)) < len(selector):
            raise ValueError(f"picks {selector} name {next(i for i in selector if selector.count(i) > 1)} twice")
    if fixed(selector) == 0:
        raise ValueError(f"{selector!r} picks nothing, however many things there are")
    return selector


def select(selector, n: int) -> list[int]:
    """The indices a selector picks among `n` things, the one place a selector is read: an int, the one it
    counts (negative from the end), if there is one; a slice string, its range by Python's rules; a list of
    picks, each one there is, in the order written. Two picks that count the same thing (`[0, -2]` of two)
    are refused. The order matters where the picked things are listed (layers, forward calls, tokens); an
    occurrence's picks make a scope, which is a set of positions, so there the order does not."""
    if isinstance(selector, str):
        parts = [int(x) if x else None for x in (selector.split(":") + [""])[:3]]
        return list(range(n)[slice(*parts)])
    picks = [selector] if isinstance(selector, int) else selector
    got = [i + n if i < 0 else i for i in picks if -n <= i < n]
    if len(set(got)) < len(got):
        raise Unaligned(MISSING, f"picks {selector} count one of these {n} twice")
    return got


def kept(selects: dict | None, sizes: dict[str, int]) -> dict[str, list[int]]:
    """A site's or an op's `select` as the indices it keeps on each axis: the axis's selector among its size in
    `sizes`. The `key` axis is as long as each row, so a selector there names a fixed number of keys counted
    from the start (`keyed` says so), which no size changes."""
    return {axis: select(s, sys.maxsize if axis == "key" else sizes[axis]) for axis, s in (selects or {}).items()}


def keyed(selector) -> bool:
    """Whether a selector picks the same keys of every row long enough to hold them: a fixed number, counted
    from the start (`3`, `[0, 2]`, `"1:4"`; not `-1`, `"2:"`)."""
    return fixed(selector) is not None and not from_end(selector)


def probed(selector) -> tuple[int, list[int], list[int]]:
    """A number `n` past every number a selector is written with, and what it picks among `n` and `2n` things:
    where the two agree, what it picks does not depend on how many things there are."""
    numbers = [abs(int(x)) for x in re.findall(r"-?[0-9]+", str(selector))]
    n = 2 * (1 + max(numbers, default=0))
    return n, select(selector, n), select(selector, 2 * n)


def widths(selector) -> list[int]:
    """How many things a selector picks among 0, 1, ... up to `2n` of them (`n` as `probed` has it)."""
    n, _, _ = probed(selector)
    return [len(select(selector, k)) for k in range(2 * n + 1)]


def fixed(selector) -> int | None:
    """A selector's static width: how many things it picks whatever their number (once there are enough), or
    None when that grows with the number. An int is 1, a list of picks its length; a slice has one when its
    two ends count from the same side (`"2:5"`, `":3"`, `"-4:-1"`, `"-3:"`, `"3::-1"`), not `":"`, `"2:"`
    or `"-2:3"` (which picks fewer of more things): it picks no fewer of more things and, past every number
    it is written with, the same number.
    Where there are fewer things than a static width needs, the record does not fit that row: a refusal in
    the prefill, `alignment_missing` beyond it."""
    if not isinstance(selector, str):
        return 1 if isinstance(selector, int) else len(selector)
    counts = widths(selector)
    static = all(a <= b for a, b in zip(counts, counts[1:])) and counts[len(counts) // 2] == counts[-1]
    return counts[-1] if static else None


def from_end(selector) -> bool:
    """Whether a selector counts from the end: among more things it picks other ones of the first `n` than
    among `n` (`-1`, `[0, -1]`, `"-3:"`, `":-1"`), so what it picks of the first `n` is not decided by them."""
    n, few, many = probed(selector)
    return few != [i for i in many if i < n]


def every(selector) -> bool:
    """Whether a selector picks every one of however many things there are, in order (`":"`)."""
    n, few, many = probed(selector)
    return few == list(range(n)) and many == list(range(2 * n))


# -- what the run needs to know before it starts ------------------------------------------------------


def bounded(frame) -> bool:
    """Whether a frame names the same calls on every row that gets that far, a static number of them: not a
    frame to the row's last call (`":"`, `"1:"`, `"2:"`)."""
    return fixed(frame) is not None


def width(pos: dict) -> int | None:
    """A record's width on every row, or None when it is ragged: its frame is `bounded`, its index has a
    static width (`fixed`), and a scope, if any, is one occurrence in the prefill."""
    scope = pos["scope"]
    if not bounded(pos["frame"]) or scope and (not prefill(pos) or fixed(scope.get("occurrence", 0)) != 1):
        return None
    return fixed(pos["index"])


def reach(pos: dict, horizon: int | None = None) -> tuple[bool, list[int] | None, bool]:
    """Whether a record can name a prefill position on some row; the decode calls it can name, in order (None:
    every one, or ones not known before the run); and whether those are exactly the calls it names on each row
    that gets that far, which a write needs, landing as each call runs. `horizon` is how many calls a row makes
    at most (a generation's `max_new_tokens`), when known.

    A frame counted from the end depends on how far each row gets. Any other frame names call `k`
    on every row that makes it, as `k in select(frame, k + 1)` says. An index other than `":"` must not
    depend on how far the row gets either: over a frame holding the prompt it does (the prompt's length
    differs by row); over decode calls, each holding one position, the record is exact when a row that stops
    after `k` calls names exactly the calls below `k` that a row running to the horizon names (an index
    counted from the end over several calls does not). Without a horizon, a frame to the row's last call
    other than every decode call names calls only the run will know. A scope's index counts its text's
    tokens, which any of the frame's calls may hold, so a scoped record can name every call of its frame."""
    frame, index = pos["frame"], ":" if pos["scope"] else pos["index"]
    if from_end(frame):
        return True, None, False
    holds = 0 in select(frame, 1)
    if holds or every(index):
        calls = decode_calls(frame, horizon)
        known = calls is not None or covers(frame)
        return holds, calls, known and (every(index) or calls == [])
    pos = {**pos, "index": index}
    if horizon is None:
        k = enough(pos)
        if named(pos, k) != named(pos, 2 * k):
            return False, None, False                   # it names calls up to the row's last
        horizon = max(named(pos, k), default=0) + 1
    full = named(pos, horizon)
    exact = all(set(named(pos, k)) == {c for c in full if c < k} for k in range(1, horizon))
    return False, sorted(full) if exact else decode_calls(frame, horizon), exact


def decode_calls(frame, horizon: int | None) -> list[int] | None:
    """A frame's decode calls, in order: listed when it names a static number of them or when the horizon
    bounds them; None when it names every one, or calls to the row's last that only the run will know."""
    if fixed(frame) is not None:
        return sorted(k for k in probed(frame)[1] if k)
    if covers(frame) or horizon is None:
        return None
    return sorted(k for k in select(frame, horizon) if k)


def covers(frame) -> bool:
    """Whether a frame names every decode call (`":"`, `"1:"`)."""
    n, few, many = probed(frame)
    return set(range(1, n)) <= set(few) and set(range(1, 2 * n)) <= set(many)


def named(pos: dict, k: int) -> list[int]:
    """The decode calls an unscoped record whose frame holds no prompt names on a row that makes `k` calls: its
    index's picks among the frame's calls, each holding one position."""
    called = select(pos["frame"], k)
    return [called[i] for i in select(pos["index"], len(called))]


def enough(pos: dict) -> int:
    """A number of calls past every number a record's frame and index are written with, and past their
    product, so that what they name among that many is all a row can make them name."""
    numbers = [abs(int(x)) for x in re.findall(r"-?[0-9]+", f"{pos['frame']} {pos['index']}")]
    return (2 + max(numbers, default=0)) ** 2


def prefill(pos: dict) -> bool:
    """Whether the record names prefill positions only, so the data alone decides it before any forward."""
    holds, calls, _ = reach(pos)
    return holds and calls == []


# -- the three stages ---------------------------------------------------------------------------------


def framed(frame, n: int) -> list[int]:
    """Stage one: the calls `frame` picks among a row's `n` calls; `-1` is its last, whether it chose EOS or
    the generation ran out."""
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


def resolve(pos: dict, prompt: int, prompted: list[int], said: list[int],
            occurrences: list[set[int]]) -> list[int] | str:
    """The positions a record names in one row, given its prompt's length, the positions its prefill holds
    (`prompted`), what it generated up to its first EOS (`said`), and where its scope's text occurs
    (`occurrences`); or why it names none, a marker. In the prefill that is a refusal: the data decided."""
    try:
        tokens = held(framed(pos["frame"], max(len(said), 1)), prompted, prompt)
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
    return any(resolve({**pos, "scope": {**pos["scope"], "occurrence": i}}, 0, [], said, found)
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
