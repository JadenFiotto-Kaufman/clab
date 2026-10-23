"""How a document's arguments are checked against the signature of the function they go to (an op, a
featurizer kind, an engine), and how its types are spelled to a reader."""

import inspect
import types
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Union, get_args, get_origin, is_typeddict

from pydantic import TypeAdapter, ValidationError
from torch import Tensor

if TYPE_CHECKING:       # a featurizer instance; its package imports this module, so only the checker sees it
    from clab.featurizers import Kind

# How a document fills a parameter that names a thing the run binds: a named value or a number, a dataset
# column resolved through a vocabulary, a dataset column as text, a featurizer's name.
Value = Annotated[Tensor, "Value"]
Tokens = Annotated[Tensor, "Tokens"]
Text = Annotated[list, "Text"]
Featurizer = Annotated["Kind", "Featurizer"]


def spelled(fn: Callable, supplied: tuple[str, ...]) -> dict[str, inspect.Parameter]:
    """A function's parameters by document spelling (a trailing underscore dropped: `with_` is `"with"`),
    without the ones the run supplies and a `**` catch-all."""
    return {p.name.rstrip("_"): p for p in inspect.signature(fn).parameters.values()
            if p.name not in supplied and p.kind is not p.VAR_KEYWORD}


def described(fn: Callable) -> dict[str, str]:
    """What each of a function's parameters takes, as its docstring says on a line of its own:
    `<name>: <what it takes>`."""
    names = inspect.signature(fn).parameters
    lines = (line.strip().partition(": ") for line in (inspect.getdoc(fn) or "").splitlines())
    return {name: text for name, sep, text in lines if sep and name in names}


def check(fn: Callable, args: dict[str, Any], supplied: tuple[str, ...], what: str) -> None:
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
        named = get_origin(thing(annotation)) is Annotated
        if named or annotation is Any or isinstance(value, dict) and "sweep" in value:
            continue
        try:
            TypeAdapter(annotation).validate_python(value, strict=True)
        except ValidationError:
            raise ValueError(f"{key}: {value!r} is not {spell(annotation)}") from None


def thing(annotation: Any) -> Any:
    """The thing an optional parameter names when given (`Value` of `Value | None`); any other annotation as it is."""
    named = [a for a in get_args(annotation) if get_origin(a) is Annotated]
    return named[0] if get_origin(annotation) in (Union, types.UnionType) and named else annotation


def spell(annotation: Any, names: dict[Any, str] | None = None, literal: Callable[[Any], str] = repr) -> str:
    """A type as a reader of the format says it: `float or null`. `names` spells some types by name, and
    `literal` a literal's values."""
    names = names or {}
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in names:
        return names[annotation]
    if origin is Literal:
        return "one of " + ", ".join(map(literal, args))
    if origin in (Union, types.UnionType):
        return " or ".join(spell(a, names, literal) for a in args)
    if origin is Annotated:
        meta = annotation.__metadata__[0]
        return meta if isinstance(meta, str) else spell(args[0], names, literal)
    if origin in (list, tuple):
        return f"list of {spell(args[0], names, literal)}" if args else "list"
    if origin is dict:
        return f"object of {spell(args[1], names, literal)}" if args else "object"
    if annotation is type(None):
        return "null"
    if is_typeddict(annotation):
        return "{" + ", ".join(f"{k}: {spell(v, names, literal)}" for k, v in annotation.__annotations__.items()) + "}"
    return getattr(annotation, "__name__", str(annotation))
