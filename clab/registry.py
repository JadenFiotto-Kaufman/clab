"""How a function joins one of clab's tables (step kinds, featurizer kinds, ops): under its own name."""

from typing import Callable


def registers(table: dict) -> Callable:
    """A decorator that adds the function it decorates to `table` under the function's own name."""
    def add(fn: Callable) -> Callable:
        table[fn.__name__] = fn
        return fn
    return add
