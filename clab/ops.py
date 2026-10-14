"""The one computation vocabulary: every op is a plain function over tensors.

`@op(group, ...)` registers the function under its own name and declares it: its group, the axes the input
must carry (`needs`), the axes it removes (`removes`: a tuple, or the name of the argument that lists them),
the axes it appends (`adds`), and, when its value is a bundle a featurizer loads, that featurizer's kind
(`bundle`). Compile reads the axes to lay out every value. A parameter's annotation says how a document
fills it (`Value`, `Tokens`, `Featurizer`, or a literal); `engine`, `original` and `dims` are filled by the
run, never by a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is
`"with"`).
"""

ABOUT = """An op runs in three places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a value earlier steps produced;
- inside a read, `"ops": [...]`, on the gathered selection before its name binds;
- inside a write, `"ops": [...]`, on the live selection at the site; what the last op returns is written back.

Inside a read or write an op sees one batch, so it must act per example: one that aggregates over rows is
refused there (make it a step after the forward).

Ops come in four groups: **metrics** score each example (one number per row and position: `match` is 0 or
1, and the accuracy is a `mean` step after it); **reductions** remove the rows axis; **edits** keep the
value's layout and change its contents; **transforms** turn each example's value into another of a
different layout (logits from a residual stream, tokens from logits, text from ids).

An argument is filled by a document as its type says: `Value` is a named value or a number; `Tokens` a
dataset column of one token per row, resolved through the value's own vocabulary; `Featurizer` a
featurizer's name; anything else is a literal. Inside a read or write, `featurizer` and `select` are
arguments of every op: the op runs on the value's coordinates in that featurizer's space (on the features
`{"select": {"features": [...]}}` keeps) and the change is written back. `renormalize`'s reference is the
value at the site before the step's writes.
"""

import inspect
import math
from typing import Annotated, Any, Callable, Literal

import torch
from torch import Tensor

from . import registry
from .address import EXAMPLE_AXES, pos_axis

Value = Annotated[Tensor, "Value"]
Tokens = Annotated[Tensor, "Tokens"]
Featurizer = Annotated[dict, "Featurizer"]

# `engine`: the engine running the document. `original`: inside a write, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
SUPPLIED = ("engine", "original", "dims")

OPS: dict[str, Callable] = {}
GROUPS = ("metrics", "reductions", "edits", "transforms")
DECLARED: dict[str, dict] = {}


def op(group: str, needs: tuple = (), removes: tuple | str = (), adds: tuple = (), bundle: str | None = None):
    """Register the decorated function as an op under its own name, with its declaration: its `group`
    (`metrics`, `reductions`, `edits` or `transforms`), its axes, and, for a value a featurizer loads, the
    featurizer kind its `bundle` is for."""
    assert group in GROUPS
    def register(fn: Callable) -> Callable:
        DECLARED[fn.__name__] = {"group": group, "needs": needs, "removes": removes, "adds": adds, "bundle": bundle}
        return registry.registers(OPS)(fn)
    return register


def parameters(name: str) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones."""
    return registry.spelled(OPS[name], SUPPLIED + ("of",))


def check(name: str, args: dict) -> None:
    """Op `name` exists and `args` bind to its signature."""
    if name not in OPS:
        raise ValueError(f"unknown op {name!r}; valid: {', '.join(OPS)}")
    registry.check(OPS[name], args, SUPPLIED + ("of",), repr(name))


def call(name: str, value: Any, args: dict, **supplied: Any) -> Any:
    """Run op `name` on `value` with `args` (document spelling, already bound to tensors)."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    kwargs = {names[k]: v for k, v in args.items()}
    kwargs.update((k, v) for k, v in supplied.items() if k in names and k not in kwargs)
    return fn(value, **kwargs)


def out_dims(name: str, dims: list[str], args: dict) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`."""
    axes = DECLARED[name]
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
    one dimension of its selection (compile refuses more). A ragged write's rows are padded to one
    width; only the positions its `keep` marks are written.
    """
    axis = pos_axis(component)
    before = value.movedim(axis, 1)
    x = before.clone()
    rows = torch.arange(x.shape[0])[:, None]
    for w in writes:
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
        keep = w.get("keep", torch.ones(pos.shape, dtype=torch.bool))
        x[rows.expand_as(pos)[keep], pos[keep]] = new.to(x.dtype)[keep]
    return x.movedim(1, axis)


# -- ops that change a value in place of it ------------------------------------


@op("edits")
def swap(of: Tensor, with_: Value) -> Tensor:
    """Replace the value with `with`, broadcast over its rows and positions."""
    return torch.as_tensor(with_, dtype=of.dtype, device=of.device).expand_as(of).clone()


@op("edits")
def add(of: Tensor, with_: Value, scale: float = 1.0) -> Tensor:
    """`of + scale * with`."""
    return of + scale * torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op("edits")
def sub(of: Tensor, with_: Value) -> Tensor:
    """`of - with`."""
    return of - torch.as_tensor(with_, dtype=of.dtype, device=of.device)


@op("edits")
def scale(of: Tensor, factor: float) -> Tensor:
    """`of * factor`."""
    return of * factor


@op("edits")
def renormalize(of: Tensor, original: Value) -> Tensor:
    """Each vector rescaled to the norm of `original`'s; inside a write, the value before the site's writes."""
    return of * (original.norm(dim=-1, keepdim=True) / of.norm(dim=-1, keepdim=True))


@op("edits")
def normalize(of: Tensor) -> Tensor:
    """Each vector divided by its norm."""
    return of / of.norm(dim=-1, keepdim=True)


@op("edits")
def project_out(of: Tensor, direction: Value) -> Tensor:
    """Each vector less its component along `direction`."""
    unit = direction / direction.norm(dim=-1, keepdim=True)
    return of - (of * unit).sum(-1, keepdim=True) * unit


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("vocab_size",))
def readout(of: Tensor, engine: Any) -> Tensor:
    """The model's own readout of a residual-stream value, its final norm and head (the logit lens)."""
    return engine.readout(of)


@op("transforms", needs=("hidden_size",), removes=("hidden_size",), adds=("features",))
def encode(of: Tensor, featurizer: Featurizer) -> Tensor:
    """The value's coordinates in `featurizer`'s space."""
    raise NotImplementedError("M4")


# -- reductions ----------------------------------------------------------------


@op("reductions", removes="over")
def mean(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The mean over the axes `over` names; by default every axis but the feature axes."""
    over = over if over is not None else [a for a in dims if a in EXAMPLE_AXES]
    return of.mean(dim=[dims.index(a) for a in over])


@op("reductions", needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",), bundle="pca")
def pca(of: Tensor, k: int, dims: list[str]) -> dict:
    """The top-`k` principal directions (`basis`, `[hidden_size, k]`, orthonormal columns, the largest variance
    first) and the `mean` they are centred on: the bundle a `pca` featurizer loads."""
    x = of.reshape(-1, of.shape[-1]).double()
    if x.shape[0] <= k:
        raise ValueError(f"pca of {x.shape[0]} vectors has no rank-{k} basis; it needs more than {k}")
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    # An SVD's singular vectors are unique only up to sign; fix it so the largest entry of each is positive.
    basis = vh[:k].T
    basis = basis * basis.gather(0, basis.abs().argmax(0, keepdim=True)).sign()
    return {"basis": basis.to(of.dtype).contiguous(), "mean": mean.to(of.dtype)}


# -- metrics: logits in, one number per row and position out -------------------


def _per_row(ids: Tensor, x: Tensor, dims: list[str]) -> Tensor:
    """One id per row (`[rows]`), shaped to broadcast against `x` along its rows (or entries) axis."""
    shape = [1] * x.dim()
    shape[next(i for i, a in enumerate(dims) if a in ("rows", "entries"))] = -1
    return ids.reshape(shape)


def _pick(x: Tensor, ids: Tensor, dims: list[str]) -> Tensor:
    return x.gather(-1, _per_row(ids, x, dims).expand(*x.shape[:-1], 1)).squeeze(-1)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def match(of: Tensor, expected: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """1 where the most likely token is `expected`, else 0."""
    return (of.argmax(-1, keepdim=True) == _per_row(expected, of, dims)).squeeze(-1).to(of.dtype)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def logit_diff(of: Tensor, a: Tokens, b: Tokens, dims: list[str],
               token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `a` less the logit of `b`."""
    return _pick(of, a, dims) - _pick(of, b, dims)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def token_prob(of: Tensor, token: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The probability of `token`."""
    return _pick(of.softmax(-1), token, dims)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def token_logit(of: Tensor, token: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `token`."""
    return _pick(of, token, dims)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def cross_entropy(of: Tensor, target: Tokens, dims: list[str],
                  token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The negative log-probability of `target`, in nats."""
    return -_pick(of.log_softmax(-1), target, dims)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def kl(of: Tensor, against: Value) -> Tensor:
    """KL(of ‖ against) between the two next-token distributions, in nats."""
    p, q = of.log_softmax(-1), against.log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


@op("metrics", needs=("vocab_size",), removes=("vocab_size",))
def js(of: Tensor, against: Value) -> Tensor:
    """The Jensen–Shannon divergence between the two distributions, in nats (at most log 2)."""
    p, q = of.log_softmax(-1), against.log_softmax(-1)
    m = torch.logaddexp(p, q) - math.log(2)
    return 0.5 * (p.exp() * (p - m)).sum(-1) + 0.5 * (q.exp() * (q - m)).sum(-1)


@op("transforms", needs=("vocab_size",), removes=("vocab_size",), adds=("k",))
def top_k(of: Tensor, k: int) -> dict:
    """The `k` most likely tokens, `{"ids", "probs"}`, most likely first; as a step, its value also holds their
    `tokens`, decoded."""
    probs, ids = of.softmax(-1).topk(k, dim=-1)
    return {"ids": ids, "probs": probs}


@op("transforms", needs=("vocab_size",), removes=("vocab_size",))
def argmax(of: Tensor) -> Tensor:
    """The id of the most likely token."""
    return of.argmax(-1)


@op("transforms", needs=("new",), removes=("new",))
def decode(of: Tensor, engine: Any, tokens: bool = False) -> list:
    """Each row's generated ids as text, stopped before the row's first EOS (the engine's, where its generation
    stops); with `tokens`, as its token strings."""
    tokenizer, eos = engine.tokenizer, set(engine.eos)
    out = []
    for ids in of.tolist():
        ids = next((ids[:i] for i, t in enumerate(ids) if t in eos), ids)
        out.append(tokenizer.convert_ids_to_tokens(ids) if tokens else tokenizer.decode(ids))
    return out
