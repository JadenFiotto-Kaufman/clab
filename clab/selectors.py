"""Selectors: which of a number of things (layers, forward calls, occurrences, tokens, keys) a range picks, an
int, a slice string in Python notation, or a list of distinct picks; and what can be known of a selector before
the number of things is."""

import re
import sys
from typing import Annotated, Union

from pydantic import BeforeValidator


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
        raise ValueError(f"picks {selector} count one of these {n} twice")
    return got


def kept(selects: dict | None, sizes: dict[str, int]) -> dict[str, list[int]]:
    """A site's `select` as the indices it keeps on each axis: the axis's selector among its size in `sizes`,
    and on `key` its `keys`."""
    return {axis: keys(s) if axis == "key" else select(s, sizes[axis]) for axis, s in (selects or {}).items()}


def keys(selector) -> list[int]:
    """The keys a selector on the `key` axis picks: the same of every row long enough to hold them, a fixed
    number counted from the start (`keyed` says so), so no row's length changes them."""
    return select(selector, sys.maxsize)


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


def covers(frame) -> bool:
    """Whether a frame names every decode call (`":"`, `"1:"`)."""
    n, few, many = probed(frame)
    return set(range(1, n)) <= set(few) and set(range(1, 2 * n)) <= set(many)


# Which of a number of things a range picks, as a document writes it: `pos` documents it, `select` resolves it.
Selector = Annotated[Union[int, str, list[int]], "selector", BeforeValidator(checked)]
