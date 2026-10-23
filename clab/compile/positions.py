"""Positions: a `pos` as three stages of narrowing (frame, scope, index), resolved row by row inside the run,
where the tokenizer is, to the positions the record names.

A row's positions are its input tokens, all of them, in order (`prompted`): the prompt's as encoded (`0 ..
P-1`, BOS and a chat template's markers included; padding is never one), then what it generated (`P + j` for
generated token `j`). The prefill, call 0, holds the prompt; decode call `k >= 1` holds generated token `k -
1`, its input. A row's calls run to the one that chose its last token, so that token is never an input. A
row's keys, on an attention pattern's `key` axis, are its prompt's positions, numbered the same way.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from clab.format import data, selectors
from clab.format.selectors import Selector, bound, covers, every, fixed, from_end, select
from clab.plan import Forward, Generate, Intervention, Pos, Site, Sweep, Swept, fail

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

MISSING = "alignment_missing"
UNPROMPTED = {"ids": [], "offsets": [], "text": "", "turns": []}    # an encoded empty prompt


class Unaligned(ValueError):
    """A row the record does not reach, and why: `alignment_missing` beyond the prefill, a refusal in it."""


def parse(pos: Swept[int | Pos]) -> dict:
    """A `pos` (an int, or a `Pos`) as `{"frame", "scope", "index"}`."""
    assert not isinstance(pos, Sweep), "a point's pos is one value"
    if isinstance(pos, int):
        return {"frame": 0, "scope": None, "index": pos}
    scope = None if pos.scope is None else pos.scope.model_dump(exclude_none=True)
    return {"frame": pos.frame, "scope": scope, "index": pos.index}


# -- what the run needs to know before it starts ------------------------------------------------------


def width(pos: dict) -> int | None:
    """A record's width on every row, or None when rows differ: its frame names a static number of calls, its
    index has a static width, and a scope, if any, is one occurrence in the prefill."""
    scope = pos["scope"]
    if fixed(pos["frame"]) is None or scope and (not prefill(pos) or fixed(scope.get("occurrence", 0)) != 1):
        return None
    return fixed(pos["index"])


def reach(pos: dict, horizon: int | None = None) -> tuple[bool, list[int] | None, bool]:
    """Whether a record can name a prefill position on some row; the decode calls it can name, in order (None:
    every one, or ones not known before the run); and whether those are exactly the calls it names on each row
    that gets that far, which a write needs, landing as each call runs. `horizon` is how many calls a row makes
    at most (a generation's `max_new_tokens`), when known.

    A frame counted from the end depends on how far each row gets. Any other frame names call `k` on every row
    that makes it. Over a frame holding the prompt any index but `":"` counts through the prompt, whose length
    differs by row. Over decode calls, each holding one position, the record is exact when a row that stops
    after `k` calls names exactly the calls below `k` that a row running to the horizon names (an index counted
    from the end over several calls does not, nor any index over picks out of order). Without a horizon, an
    open frame with an index that is open or counted from the end names calls only the run will know. A scope's
    index counts its text's tokens, which any of the frame's calls may hold, so a scoped record can name every
    call of its frame."""
    frame, index = pos["frame"], ":" if pos["scope"] else pos["index"]
    if pos["scope"] and pos["scope"].get("before") and select(frame, 2) != [0]:
        return True, None, False        # the position before a token may be any call's, the prefill's included
    if from_end(frame):
        return True, None, False
    holds = 0 in select(frame, 1)
    if holds or every(index):
        calls = decode_calls(frame, horizon)
        known = calls is not None or covers(frame)
        return holds, calls, known and (every(index) or calls == [])
    pos = {**pos, "index": index}
    if horizon is None:
        if fixed(frame) is None and (fixed(index) is None or from_end(index)):
            return False, None, False                   # it names calls up to the row's last
        horizon = max(named(pos, enough(pos)), default=0) + 1
    full = named(pos, horizon)
    exact = all(set(named(pos, k)) == {c for c in full if c < k} for k in range(1, horizon))
    return False, sorted(full) if exact else decode_calls(frame, horizon), exact


def decode_calls(frame: Selector, horizon: int | None) -> list[int] | None:
    """A frame's decode calls, in order: listed when it names a static number of them or when the horizon
    bounds them; None when it names every one, or calls to the row's last that only the run will know."""
    if fixed(frame) is not None:
        return sorted(k for k in select(frame, bound(frame)) if k)
    if covers(frame) or horizon is None:
        return None
    return sorted(k for k in select(frame, horizon) if k)


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


def framed(frame: Selector, n: int) -> list[int]:
    """Stage one: the calls `frame` picks among a row's `n` calls; `-1` is its last, whether it chose EOS or
    the generation ran out."""
    return select(frame, n)


def prompted(prompt: int, head_only: bool = False) -> list[int]:
    """The positions the prefill of a row whose prompt is `prompt` tokens holds, the one place a row's positions
    are decided: every token of its prompt, in order, BOS and a chat template's markers included (a row's ids are
    its own, never padding), or its last token alone (`head_only`) where the prefill computes only that (a
    generation's head)."""
    return [prompt - 1] if head_only else list(range(prompt))


def held(called: list[int], prompt: int, head_only: bool) -> list[int]:
    """The positions the calls hold, in a row whose prompt is `prompt` tokens: the prefill's (`prompted`), and
    each decode call's input."""
    return [p for call in called for p in (prompted(prompt, head_only) if call == 0 else [prompt + call - 1])]


def call_of(position: int, prompt: int) -> int:
    """The call that holds a position, the inverse of `held`."""
    return 0 if position < prompt else position - prompt + 1


def scoped(scope: dict | None, tokens: list[int], occurrences: list[set[int]]) -> list[int]:
    """Stage two: the tokens of the occurrences `occurrence` picks among those of the scope's text (or its
    role's messages) among `tokens`, in order. Under `before` an occurrence beginning the row (at position 0)
    has no position before it."""
    if scope is None:
        return tokens
    found = [o for o in occurrences if o & set(tokens)]
    which = scope.get("occurrence", 0)
    picked = select(which, len(found))
    key = next(k for k in ("column", "text", "turn") if k in scope)
    what = f"{key} {scope[key]['column'] if isinstance(scope[key], dict) else scope[key]!r}"
    if not found:
        raise Unaligned(f"the row has no {scope['turn']!r} message" if "turn" in scope else
                        f"{what} is not in its input")
    if not picked or fixed(which) is not None and len(picked) != fixed(which):
        raise Unaligned(f"{what} occurs {len(found)} times in the row's input; it has no occurrence {which}")
    if scope.get("before") and any(0 in found[i] for i in picked):
        raise Unaligned(f"{what} begins the row, at position 0, which no position comes before")
    return [t for t in tokens if any(t in found[i] for i in picked)]


def indexed(index: Selector, tokens: list[int], width: int | None) -> list[int]:
    """Stage three: the positions `index` picks among the scope's tokens; a record with a static width must
    hold all of them."""
    got = [tokens[i] for i in select(index, len(tokens))]
    if width is not None and len(got) != width:
        raise Unaligned(f"tokens {index} do not fit its {len(tokens)} scope tokens")
    return got


def resolve(pos: dict, prompt: int, said: list[int], occurrences: list[set[int]],
            head_only: bool = False) -> list[int] | str:
    """The positions a record names in one row, given its prompt's length, what it generated up to its first
    EOS (`said`), where its scope's text occurs (`occurrences`), and whether its prefill holds the last position
    alone (`head_only`, `prompted`); or why it names none, a marker. In the prefill that is a refusal: the data
    decided. A scope that is `before` picks its tokens as any scope does, the row's last generated token among
    them when the frame names its last call, and the record is the position before each."""
    before = bool((pos["scope"] or {}).get("before"))
    try:
        calls = framed(pos["frame"], max(len(said), 1))
        tokens = held(calls, prompt, head_only)
        if before and said and len(said) - 1 in calls:     # the token the last call chose, which no call holds
            tokens.append(prompt + len(said) - 1)
        got = indexed(pos["index"], scoped(pos["scope"], tokens, occurrences), width(pos))
        got = [t - 1 for t in got] if before else got
    except ValueError as e:
        if prefill(pos):
            raise ValueError(str(e)) from None
        return MISSING
    return got if got or prefill(pos) else MISSING


# -- where a scope's text occurs ----------------------------------------------------------------------


def occurrences(scope: dict | None, row: dict, said: list[int], tokenizer: PreTrainedTokenizerBase, texts: list[tuple],
                online: bool = False) -> list[set[int]]:
    """Each place the scope names, as the positions that overlap it: a chat turn of the prompt, or each
    occurrence of any of its texts (`texts`: `(text, strip, lower)`, the columns' or the literals) in the
    prompt, by the tokenizer's offsets, then in the generated inputs (under `before`, every generated token,
    the last included: the tokens the calls chose), by decoding growing prefixes; each part's in order of
    position (end, then start). A token with an empty span (part of a split character) covers the character
    it begins. `online`, a write's search as each call is fed, sees nothing past a generated token, so there an
    occurrence's end is not checked against the character after it (`located`)."""
    if scope is None:
        return []
    prompt, inputs = len(row["ids"]), said if scope.get("before") else said[:-1]
    spans = turns(row, scope["turn"]) if "turn" in scope else located(texts, row["text"], row["offsets"])
    if spans and row["offsets"] is None:
        raise ValueError("this tokenizer has no offset mapping, which a scope in the prompt needs")
    found = [{t for t, (x, y) in enumerate(row["offsets"]) if x < b and y > a} for a, b in spans]
    if "turn" not in scope and inputs:
        words, chars = char_spans(tokenizer, inputs)
        found += [covering(chars, a, b, prompt) for a, b in located(texts, words, chars, not online)]
    return found


def located(texts: list[tuple], words: str, tokens: list[tuple[int, int]] | None,
            ahead: bool = True) -> list[tuple[int, int]]:
    """Every character span in `words` where one of `texts` occurs on token and word boundaries, each `(text,
    strip, lower)` compared as its flags say (`data.compared`; `words` lowercased too under `lower`, its spans
    its own); in order of end, then start: where an occurrence becomes known, so a write found online, a read and
    the run record count them alike. An occurrence starts where a token starts and ends where a token ends
    (`bounds`, over the tokens' character spans `tokens`), and where its own first or last character is a letter
    or a digit, none adjoins it: ` Sara` is not in ` Sarah`, ` Ann` not in ` Annabel` (`ĠAnn abel`), `:` not in
    `):`. Without `ahead` the character after an occurrence is not looked at, as a write found online cannot.
    Without the spans (a tokenizer with no offsets, which `occurrences` then refuses) every character is a token
    boundary."""
    starts, ends = bounds(words, tokens) if tokens is not None else ({i: i for i in range(len(words) + 1)},) * 2
    spans = []
    for text, strip, lower in texts:
        text = data.compared(text, strip, lower)
        found = re.finditer(f"(?=({re.escape(text)}))", data.compared(words, False, lower)) if text else []
        last = 0                    # a text's occurrences do not overlap each other: each starts after the last
        for m in found:
            a, b = m.span(1)
            if a >= last and a in starts and b in ends and not word(words, a - 1, a) and \
                    not (ahead and word(words, b, b - 1)):
                spans.append((starts[a], ends[b]))
                last = b
    return sorted(spans, key=lambda span: (span[1], span[0]))


def word(words: str, outside: int, inside: int) -> bool:
    """Whether an occurrence's character at `inside` and the one just outside it, at `outside`, are both letters
    or digits: one word, which the occurrence would cut."""
    return 0 <= outside < len(words) and words[outside].isalnum() and words[inside].isalnum()


def bounds(words: str, tokens: list[tuple[int, int]]) -> tuple[dict[int, int], dict[int, int]]:
    """Where an occurrence may start and end among the tokens' character spans in `words`, each mapped to where
    its span then starts or ends: at a token's start or end, or within the whitespace a token begins or ends with
    (a text begins at ` Paris`'s `P` whichever way the tokenizer's offsets count the space). An occurrence that
    ends inside the next token's leading whitespace ends where that token starts, and one that starts inside the
    last token's trailing whitespace starts where that token ends, so neither covers that token (`Paris. ` does
    not cover the ` Paris` after it)."""
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for x, y in tokens:
        starts.update({x: x, y: y})
        ends.update({x: x, y: y})
    for x, y in tokens:
        piece = words[x:y]
        lead, trail = x + len(piece) - len(piece.lstrip()), y - len(piece) + len(piece.rstrip())
        starts.setdefault(lead, lead)
        ends.setdefault(lead, x)
        starts.setdefault(trail, y)
        ends.setdefault(trail, trail)
    return starts, ends


def covering(chars: list[tuple[int, int]], a: int, b: int, offset: int) -> set[int]:
    """The tokens (`offset` + their index) whose spans overlap characters `a` to `b`; a token with an empty
    span covers the character it begins."""
    return {offset + j for j, (x, y) in enumerate(chars) if x < b and max(y, x + 1) > a}


def landing(pos: dict, texts: list[tuple], inputs: list[int], eos: set[int], tokenizer: PreTrainedTokenizerBase
            ) -> bool:
    """Whether a write with a text scope lands at the call fed `inputs[-1]`, given every input the row was fed
    so far: some occurrence its `occurrence` picks resolves, with index -1, to that token, on the prefix and
    counted only within the frame's calls, as `resolve` counts them; so each picked occurrence lands at its own
    last token, as it appears. The prefix is the row's calls so far (the last is the one being fed, whose token
    is not chosen yet) after an empty prompt, so a row that stopped has no such call and input `j` sits at `j`."""
    said = continuation([*inputs, -1], eos)
    found = occurrences(pos["scope"], UNPROMPTED, said, tokenizer, texts, online=True)
    return any(resolve({**pos, "scope": {**pos["scope"], "occurrence": i}}, 0, said, found)
               == [len(inputs) - 1] for i in select(pos["scope"]["occurrence"], len(found)))


def turns(row: dict, role: str) -> list[tuple[int, int]]:
    """The character spans of the row's messages of `role`, as the chat template rendered them: in order, so
    ordered by where each ends, as `located` orders text."""
    return [(a, b) for r, a, b in row["turns"] if r == role]


def continuation(ids: list[int], eos: set[int]) -> list[int]:
    """What a row generated: its new ids up to and including its first EOS."""
    return next((ids[:i + 1] for i, token in enumerate(ids) if token in eos), ids)


def char_spans(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """The text `ids` decode to, and each token's character span in it, found by decoding growing prefixes.

    A token ends where its prefix's text stops agreeing with the whole text, so a character split across
    tokens (a multi-byte one, which a partial prefix decodes to `\\ufffd`) is the span of the token that
    completes it, and the tokens before that one have empty spans at the character's start.
    """
    text = cast(str, tokenizer.decode(ids))
    spans, start = [], 0
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1])
        end = next((j for j, (a, b) in enumerate(zip(prefix, text)) if a != b), min(len(prefix), len(text)))
        spans.append((start, max(start, end)))
        start = max(start, end)
    return text, spans


def indices(where: str, write: bool, record: Intervention, step: Forward | Generate, rows: list[dict],
            encoded: list[dict], tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    """A prefill record's positions in each row, which the data alone decides: a scope text that is absent or
    doubled, or a static index that does not fit, is refused by name, and so is a write whose positions
    hold no token of a row."""
    assert isinstance(record.site, Site), "a compiled entry's site is inlined"
    pos = parse(record.pos)
    found = []
    for i, (row, enc) in enumerate(zip(rows, encoded)):
        try:
            got = place(pos, row, enc, [], tokenizer, head(step, record))
        except ValueError as e:
            why = f"; a generation's prefill computes {record.site.component} at its last position only, so " \
                "that is all call 0 holds of it" if head(step, record) else ""
            raise ValueError(f"{where}: row {i}: {e}{why}") from None
        if write and not got:
            raise ValueError(f"{where}: row {i}: the write's positions hold no token of this row; a write needs "
                             "one to write")
        found.append(list(got))
    return found


def place(pos: dict, row: dict, encoded: dict, said: list[int], tokenizer: PreTrainedTokenizerBase,
          head_only: bool) -> list[int] | str:
    """`resolve` for one row (its dataset row and its encoding), with what that needs: where the
    scope's text occurs."""
    found = occurrences(pos["scope"], encoded, said, tokenizer, texts(pos, row))
    got = resolve(pos, len(encoded["ids"]), said, found, head_only)
    if head_only and (pos["scope"] or {}).get("before") and isinstance(got, list) and \
            any(p < len(encoded["ids"]) - 1 for p in got):
        return MISSING            # before a prompt token, where this prefill computes no head
    return got


def texts(pos: dict, row: dict) -> list[tuple[str, bool, bool]]:
    """The texts a record's scope names in one row, each `(text, strip, lower)`: its columns' (each member of
    a set), compared as each reference says, or the literals, as the scope says; a single one is the
    one-element case."""
    scope = pos["scope"] or {}
    if "column" in scope:
        out = []
        for ref in scope["column"] if isinstance(scope["column"], list) else [scope["column"]]:
            column, strip, lower = data.column_of(ref) or ("", False, False)
            value = data.lookup(row, column.partition(".")[2])
            out += [(str(v), strip, lower) for v in (value if isinstance(value, list) else [value])]
        return out
    given = scope.get("text")
    given = [] if given is None else given if isinstance(given, list) else [given]
    return [(t, bool(scope.get("strip")), bool(scope.get("lower"))) for t in given]


def head(step: Forward | Generate, record: Intervention) -> bool:
    """Whether the record reads the head in a generation, whose prefill computes it at its last position only."""
    assert isinstance(record.site, Site), "a compiled entry's site is inlined"
    return step.kind == "generate" and record.site.component in ("lm_head_output", "logits")


def decided(path: str, pos: dict, most: int | None) -> None:
    """A generation's write beyond the prefill lands at each call it selects as that call runs, so what it
    selects must be known by then. Without a scope its calls must be known before the run, up to the step's
    `max_new_tokens` (`most`) when it gives one (`reach`). With a text scope it is found online,
    call by call, in the inputs fed so far, and lands at the call fed the first occurrence's last token: so
    its frame is decode calls only, and its index -1, since the occurrence's earlier tokens were fed before
    the text was there to find; and which occurrences it means count from the first."""
    frame, index = pos["frame"], pos["index"]
    if pos["scope"] is not None:
        if reach({**pos, "index": ":"})[0]:
            fail(f"{path}.frame", f"frame {frame!r}: a scoped write beyond the prefill is found call by call in the "
                 "generated inputs, so its frame names decode calls only, not counted from the row's last call "
                 "(a call, picks, or a slice from 1 such as '1:')")
        if index != -1:
            fail(f"{path}.index", f"index {index!r}: a scoped write lands at the call fed its text's last token, the "
                 "first call that can know the text is there; earlier tokens were already fed, so its index is -1")
        which = pos["scope"].get("occurrence", 0)
        if selectors.from_end(which):
            fail(f"{path}.scope.occurrence", f"occurrence {which!r} counts from the last occurrence, which is known "
                 "only when the row ends; a write lands as its call runs, so it counts from the first")
        return
    if reach(pos, most)[2]:
        return
    holds = not selectors.from_end(frame) and 0 in selectors.select(frame, 1)
    if holds and selectors.covers(frame) and selectors.from_end(index):
        fail(path, f"frame {frame!r} with index {index!r} counts back from the row's last position, known only when "
             "the row stops, and a write lands as each call runs; for the prompt's last positions and every decode "
             f"call, write two entries: {{'frame': 0, 'index': {index!r}}} and {{'frame': '1:'}}")
    if holds:
        fail(path, f"frame {frame!r} with index {index!r}: over a frame that holds the prompt a write's index is "
             "':', since any other counts on through the prompt, whose length differs by row, into the decode "
             "calls, and a write lands as each call runs; write the prompt's positions you mean as an entry at "
             "frame 0 and the decode calls you mean as another at their own frame ('1:' is every one)")
    if most is None and reach(pos, enough(pos))[2]:
        fail(path, f"frame {frame!r} with index {index!r} names calls up to the row's last; a write lands at calls "
             "listed before the run, so the step gives max_new_tokens, which bounds them")
    fail(path, f"frame {frame!r} with index {index!r} names calls that depend on how far each row gets; a write "
         "lands as its call runs, so its frame is not counted from the row's last call, and its index "
         "names the same calls on a row that stops early as on one that runs on: ':' over the prompt, and not "
         "counted from the end over several decode calls; or it takes a text scope")
