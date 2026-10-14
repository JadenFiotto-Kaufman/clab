"""How a function joins one of clab's tables (step kinds, featurizer kinds, ops), and how a document's
arguments are checked against that function's signature."""

import inspect
import types
from typing import Annotated, Any, Callable, Literal, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError
from torch import Tensor

# How a document fills a parameter that names a thing the run binds: a named value or a number, a dataset
# column resolved through a vocabulary, a featurizer's name.
Value = Annotated[Tensor, "Value"]
Tokens = Annotated[Tensor, "Tokens"]
Featurizer = Annotated[dict, "Featurizer"]


def registers(table: dict) -> Callable:
    """A decorator that adds the function it decorates to `table` under the function's own name."""
    def add(fn: Callable) -> Callable:
        table[fn.__name__] = fn
        return fn
    return add


def spelled(fn: Callable, supplied: tuple) -> dict[str, inspect.Parameter]:
    """A function's parameters by document spelling (a trailing underscore dropped: `with_` is `"with"`),
    without the ones the run supplies and a `**` catch-all."""
    return {p.name.rstrip("_"): p for p in inspect.signature(fn).parameters.values()
            if p.name not in supplied and p.kind is not p.VAR_KEYWORD}


def check(fn: Callable, args: dict, supplied: tuple, what: str) -> None:
    """`args` (document spelling) bound to `fn`'s signature: every key a parameter, every required one given,
    and every literal of its annotated type. An `Annotated` parameter names a thing (a value, a column, a
    featurizer), which compile resolves; a `{"sweep": ...}` stands for a value and is checked per point. A
    function with a `**` catch-all takes any other key, which it passes on to be checked where it lands."""
    params = spelled(fn, supplied)
    try:
        inspect.signature(fn).bind(**{p: None for p in supplied if p in inspect.signature(fn).parameters},
                                   **{params[k].name if k in params else k: v for k, v in args.items()})
    except TypeError:
        unknown = [k for k in args if k not in params]
        if unknown:
            raise ValueError(f"{what} takes no {unknown[0]!r}; its arguments: {', '.join(params) or 'none'}") from None
        missing = [k for k, p in params.items() if p.default is p.empty and k not in args]
        raise ValueError(f"{what} needs {missing[0]!r}") from None
    for key, value in args.items():
        if key not in params:
            continue
        annotation = params[key].annotation
        if get_origin(annotation) is Annotated or annotation is Any or isinstance(value, dict) and "sweep" in value:
            continue
        try:
            TypeAdapter(annotation).validate_python(value, strict=True)
        except ValidationError:
            raise ValueError(f"{key}: {value!r} is not {spell(annotation)}") from None


def spell(annotation: Any, names: dict | None = None) -> str:
    """A type as a reader of the format says it: `int or 'all' or null`. `names` spells some types by name."""
    names = names or {}
    origin, args = get_origin(annotation), get_args(annotation)
    try:
        if annotation in names:
            return names[annotation]
    except TypeError:
        pass
    if origin is Literal:
        return "one of " + ", ".join(map(repr, args))
    if origin in (Union, types.UnionType):
        return " or ".join(spell(a, names) for a in args)
    if origin is Annotated:
        meta = annotation.__metadata__[0]
        return meta if isinstance(meta, str) else spell(args[0], names)
    if origin in (list, tuple):
        return f"list of {spell(args[0], names)}" if args else "list"
    if origin is dict:
        return f"object of {spell(args[1], names)}" if args else "object"
    if annotation is type(None):
        return "null"
    if isinstance(annotation, type) and issubclass(annotation, dict) and hasattr(annotation, "__required_keys__"):
        return "{" + ", ".join(f"{k}: {spell(v, names)}" for k, v in annotation.__annotations__.items()) + "}"
    return getattr(annotation, "__name__", str(annotation))
