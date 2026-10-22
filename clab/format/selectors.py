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


def parsed(selector: str) -> tuple[int | None, int | None, int]:
    """A slice string's start, stop and step (1 when left out)."""
    start, stop, step = (int(x) if x else None for x in (selector.split(":") + [""])[:3])
    return start, stop, 1 if step is None else step


def bound(selector) -> int:
    """A number of things past every number the selector is written with: what it picks among that many is
    all a static selector ever picks."""
    return 2 * (1 + max((abs(int(x)) for x in re.findall(r"-?[0-9]+", str(selector))), default=0))


def fixed(selector) -> int | None:
    """A selector's static width: how many things it picks whatever their number (once there are enough), or
    None when that grows with the number. An int is 1, a list of picks its length; a slice has one when both
    its ends count from the same side, a bound written non-negative counting from the front and one left out
    counting from where Python starts or stops it (`"2:5"`, `":3"`, `"-4:-1"`, `"-3:"`, `"3::-1"`; not `":"`,
    `"2:"` or `"-2:3"`). Where there are fewer things than a static width needs, the record does not fit that
    row: a refusal in the prefill, `alignment_missing` beyond it."""
    if not isinstance(selector, str):
        return 1 if isinstance(selector, int) else len(selector)
    start, stop, step = parsed(selector)
    if (step > 0 and stop == 0) or (step < 0 and stop == -1):
        return 0                                        # empty, however many things there are
    front = (step > 0 if start is None else start >= 0, step < 0 if stop is None else stop >= 0)
    return len(select(selector, bound(selector))) if front[0] == front[1] else None


def from_end(selector) -> bool:
    """Whether what a selector picks among the first `n` things depends on `n`: a negative int or pick, a slice
    with a bound written negative, or a backward slice from the end unless it walks every thing (`"::-1"`)."""
    if isinstance(selector, int):
        return selector < 0
    if isinstance(selector, list):
        return any(i < 0 for i in selector)
    start, stop, step = parsed(selector)
    if stop is not None and stop < 0:
        return True
    return start is not None and start < 0 if step > 0 else not (step == -1 and start in (None, -1))


def every(selector) -> bool:
    """Whether a selector picks every one of however many things there are, in order (`":"`)."""
    return isinstance(selector, str) and parsed(selector) in ((None, None, 1), (0, None, 1))


def covers(frame) -> bool:
    """Whether a frame names every decode call (`":"`, `"1:"`), whichever way it walks them."""
    if not isinstance(frame, str):
        return False
    start, stop, step = parsed(frame)
    return (step == 1 and start in (None, 0, 1) and stop is None) or \
        (step == -1 and start in (None, -1) and stop in (None, 0))


# Which of a number of things a range picks, as a document writes it: `pos` documents it, `select` resolves it.
Selector = Annotated[Union[int, str, list[int]], "selector", BeforeValidator(checked)]


def made_at(site, n: int) -> list:
    """The layers a site is made at among the model's `n`, or `[None]` for a model's own value."""
    return [None] if site.layers is None else select(site.layers, n)


def picked(where: str, layers, n: int) -> list[int]:
    """The layers a site's `layers` select among the model's `n`; a layer outside them, or a slice that picks
    none, is refused by name."""
    for i in [layers] if isinstance(layers, int) else layers if isinstance(layers, list) else []:
        if not -n <= i < n:
            raise ValueError(f"{where}: layer {i}; the model has {n} layers")
    try:
        got = select(layers, n)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    if not got:
        raise ValueError(f"{where}: layers {layers!r} pick none of the model's {n} layers")
    return got


def selected(where: str, axis: str, selector, n: int, what: str) -> None:
    """Refuses a `select` on an axis of `n` (`what` says which) with a pick outside it, or that keeps none."""
    written = [selector] if isinstance(selector, int) else selector if isinstance(selector, list) else []
    try:
        got = kept({axis: selector}, {axis: n})[axis]
    except ValueError as e:
        raise ValueError(f"{where}: select {axis}: {e}") from None
    if any(not -n <= i < n for i in written) or any(i >= n for i in got):
        raise ValueError(f"{where}: select {axis} {selector!r} outside {what}")
    if not got:
        raise ValueError(f"{where}: select {axis} {selector!r} keeps none of {what}")
