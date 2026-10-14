"""The one computation vocabulary: every op is a plain function over tensors.

An op runs in three places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a
  value earlier steps produced;
- inside a read, `"ops": [...]`, on the gathered selection before its name
  binds;
- inside a write, `"ops": [...]`, on the live selection at the site; what the
  last op returns is written back.

`@op(...)` registers the function under its own name and declares its axis
behaviour: the axes the input must carry (`needs`), the axes it removes
(`removes`: a tuple, or the name of the argument that lists them) and the axes
it appends (`adds`). Compile reads these to lay out every value, and refuses an
op inside a read or write that aggregates over `rows`: there it sees one batch,
so it must act per example (make it a step after the forward).

A parameter's annotation says how a document fills it: `Value` is a named
value or a number, `Tokens` a dataset column of one token per row (resolved
through the value's own vocabulary, inside the run), `Featurizer` a
featurizer's name; anything else is a literal. Three parameters are filled by
the run, never by a document: `engine`, `original` (inside a write, the value
at the site before its writes) and `dims` (the axis names of `of`). A trailing
underscore is dropped from a parameter's document spelling (`with_` is
`"with"`). Inside a read or write, `featurizer` and `features` are arguments of
every op: the op runs on the value's coordinates in that featurizer's space
(on the `features` given) and the change is written back.
"""

import inspect
import math
from typing import Annotated, Any, Callable, Literal

import torch
from torch import Tensor

from .address import EXAMPLE_AXES, pos_axis
from .registry import registers

Value = Annotated[Tensor, "a named value, or a number"]
Tokens = Annotated[Tensor, "a dataset column, one token per row"]
Featurizer = Annotated[dict, "a featurizer's name"]

# `engine`: the engine running the document. `original`: inside a write, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
SUPPLIED = ("engine", "original", "dims")

OPS: dict[str, Callable] = {}
AXES: dict[str, dict] = {}


def op(needs: tuple = (), removes: tuple | str = (), adds: tuple = ()):
    def register(fn: Callable) -> Callable:
        AXES[fn.__name__] = {"needs": needs, "removes": removes, "adds": adds}
        return registers(OPS)(fn)
    return register


def parameters(name: str) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones."""
    params = list(inspect.signature(OPS[name]).parameters.values())[1:]
    return {p.name.rstrip("_"): p for p in params if p.name not in SUPPLIED}


def call(name: str, value: Any, args: dict, **supplied: Any) -> Any:
    """Run op `name` on `value` with `args` (document spelling, already bound to tensors)."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    kwargs = {names[k]: v for k, v in args.items()}
    kwargs.update((k, v) for k, v in supplied.items() if k in names and k not in kwargs)
    return fn(value, **kwargs)


def out_dims(name: str, dims: list[str], args: dict) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`."""
    axes = AXES[name]
    for axis in axes["needs"]:
        if axis not in dims:
            raise ValueError(f"{name!r} needs a value with a {axis!r} axis; this one is {dims}")
    removes = axes["removes"]
    if isinstance(removes, str):
        removes = args.get(removes)
        for axis in removes or []:
            if axis not in dims:
                raise ValueError(f"{name!r} cannot remove {axis!r}; the value is {dims}")
        removes = removes if removes is not None else EXAMPLE_AXES
    return [a for a in dims if a not in removes] + list(axes["adds"])


# -- the site: selection, then ops, then write-back ---------------------------


def select(value: Tensor, component: str, record: dict) -> Tensor:
    """The record's positions and narrowing of `value`, laid out `(rows, pos, ...)`.

    `record["positions"]` is `[rows, width]` token indices; `record["select"]`
    maps a dimension of that layout to the indices kept along it.
    """
    x = value.movedim(pos_axis(component), 1)
    x = x[torch.arange(x.shape[0])[:, None], record["positions"]]
    for dim, index in record["select"].items():
        x = x.index_select(dim, index)
    return x


def write(value: Tensor, component: str, writes: list[dict]) -> Tensor:
    """`value` with `writes` applied in listed order, out of place.

    Each write's ops run on its selection of the current value; `original` is
    the selection as it was before any of these writes. A write narrows at most
    one dimension of its selection.
    """
    axis = pos_axis(component)
    before = value.movedim(axis, 1)
    x = before.clone()
    rows = torch.arange(x.shape[0])[:, None]
    for w in writes:
        if len(w["select"]) > 1:
            raise ValueError(f"write {w['name']!r} narrows {len(w['select'])} dimensions; a write narrows one")
        pos = w["positions"]
        current, original = x[rows, pos], before[rows, pos]
        new = current
        for dim, index in w["select"].items():
            new, original = new.index_select(dim, index), original.index_select(dim, index)
        for o in w["ops"]:
            if o["featurizer"] is not None:
                raise NotImplementedError(f"M4: op {o['op']!r} in the space of featurizer {o['featurizer']!r}")
            new = call(o["op"], new, o["args"], original=original, dims=o["dims"])
        for dim, index in w["select"].items():
            new = current.index_copy(dim, index, new.to(current.dtype))
        x[rows, pos] = new.to(x.dtype)
    return x.movedim(1, axis)


# -- ops that change a value in place of it ------------------------------------


@op()
def swap(of: Tensor, with_: Value) -> Tensor:
    """Replace the value with `with`, broadcast over its rows and positions."""
    return torch.as_tensor(with_, dtype=of.dtype, device=of.device).expand_as(of).clone()


@op()
def add(of: Tensor, with_: Value, scale: float = 1.0) -> Tensor:
    """`of + scale * with`."""
    return of + scale * torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op()
def sub(of: Tensor, with_: Value) -> Tensor:
    """`of - with`."""
    return of - torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op()
def scale(of: Tensor, factor: float) -> Tensor:
    """`of * factor`."""
    return of * factor


@op()
def renormalize(of: Tensor, original: Value) -> Tensor:
    """Each vector rescaled to the norm of `original`'s; inside a write, the value before the site's writes."""
    return of * (original.norm(dim=-1, keepdim=True) / of.norm(dim=-1, keepdim=True))


@op()
def normalize(of: Tensor) -> Tensor:
    """Each vector divided by its norm."""
    return of / of.norm(dim=-1, keepdim=True)


@op()
def project_out(of: Tensor, direction: Value) -> Tensor:
    """Each vector less its component along `direction`."""
    unit = direction / direction.norm(dim=-1, keepdim=True)
    return of - (of * unit).sum(-1, keepdim=True) * unit


@op(needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: Any) -> Tensor:
    """The model's own unembedding of a residual-stream value (the logit lens)."""
    return engine.readout(of)


@op(needs=("hidden_size",), removes=("hidden_size",), adds=("features",))
def encode(of: Tensor, featurizer: Featurizer) -> Tensor:
    """The value's coordinates in `featurizer`'s space."""
    raise NotImplementedError("M4")


# -- reductions ----------------------------------------------------------------


@op(removes="over")
def mean(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The mean over the axes `over` names; by default every axis but the feature axes."""
    over = over if over is not None else [a for a in dims if a in EXAMPLE_AXES]
    return of.mean(dim=[dims.index(a) for a in over])


@op(needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",))
def pca(of: Tensor, k: int, dims: list[str]) -> dict:
    """The top-`k` principal directions and the mean, as a bundle for a `pca` featurizer."""
    raise NotImplementedError("M2")


# -- metrics: logits in, one number per row and position out -------------------


def _pick(x: Tensor, ids: Tensor) -> Tensor:
    index = ids.reshape(*ids.shape, 1, 1).expand(*x.shape[:-1], 1)
    return x.gather(-1, index).squeeze(-1)


@op(needs=("vocab_size",), removes=("vocab_size",))
def match(of: Tensor, expected: Tokens, token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """1 where the most likely token is `expected`, else 0."""
    return (of.argmax(-1) == expected.reshape(*expected.shape, 1)).to(of.dtype)


@op(needs=("vocab_size",), removes=("vocab_size",))
def logit_diff(of: Tensor, a: Tokens, b: Tokens, token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `a` less the logit of `b`."""
    return _pick(of, a) - _pick(of, b)


@op(needs=("vocab_size",), removes=("vocab_size",))
def token_prob(of: Tensor, token: Tokens, token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The probability of `token`."""
    return _pick(of.softmax(-1), token)


@op(needs=("vocab_size",), removes=("vocab_size",))
def token_logit(of: Tensor, token: Tokens, token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `token`."""
    return _pick(of, token)


@op(needs=("vocab_size",), removes=("vocab_size",))
def cross_entropy(of: Tensor, target: Tokens, token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The negative log-probability of `target`, in nats."""
    return -_pick(of.log_softmax(-1), target)


@op(needs=("vocab_size",), removes=("vocab_size",))
def kl(of: Tensor, against: Value) -> Tensor:
    """KL(of ‖ against) between the two next-token distributions, in nats."""
    p, q = of.log_softmax(-1), against.log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


@op(needs=("vocab_size",), removes=("vocab_size",))
def js(of: Tensor, against: Value) -> Tensor:
    """The Jensen–Shannon divergence between the two distributions, in nats (at most log 2)."""
    p, q = of.log_softmax(-1), against.log_softmax(-1)
    m = torch.logaddexp(p, q) - math.log(2)
    return 0.5 * (p.exp() * (p - m)).sum(-1) + 0.5 * (q.exp() * (q - m)).sum(-1)


@op(needs=("vocab_size",), removes=("vocab_size",), adds=("k",))
def top_k(of: Tensor, k: int) -> dict:
    """The `k` most likely tokens and their probabilities, decoded."""
    raise NotImplementedError("M2")
