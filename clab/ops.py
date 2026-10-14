"""The one computation vocabulary: every op is a plain function over tensors.

`@op(group, ...)` registers the function under its own name and declares it: its group, the axes the input
must carry (`needs`), the axes it removes (`removes`: a tuple, or the name of the argument that lists them),
the axes it appends (`adds`), and, when its value is a bundle a featurizer loads, that featurizer's kind
(`bundle`). Compile reads the axes to lay out every value; the `featurizer` op's are its mode's, which the
featurizer's kind declares. A parameter's annotation says how a document fills it (`Value`, `Tokens`,
`Featurizer`, or a literal); `engine`, `original` and `dims` are filled by the
run, never by a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is
`"with"`).
"""

ABOUT = """An op runs in three places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a value earlier steps produced;
- inside a read, `"ops": [...]`, on the gathered selection before its name binds;
- inside a write, `"ops": [...]`, on the live selection at the site; what the last op returns is written back.

Inside a read or write an op sees one batch, so it must act per example: one that aggregates over rows is
refused there (make it a step after the forward).

A row a read beyond the prefill did not reach is missing all the way down: in a rectangular value it is NaN,
every metric and edit of it is NaN, `argmax` and `top_k` give it id -1 (`top_k` NaN probabilities and no
token text). Reductions skip it: `mean` is the mean over the rows that have a value, and `pca` is fitted on them.

Ops come in four groups: **metrics** score each example (one number per row and position: `match` is 0 or
1, and the accuracy is a `mean` step after it); **reductions** remove the rows axis; **edits** keep the
value's layout and change its contents; **transforms** turn each example's value into another of a
different layout (logits from a residual stream, tokens from logits, text from ids).

An argument is filled by a document as its type says: `Value` is a named value or a number; `Tokens` a
dataset column, resolved through the value's own vocabulary: one token per row against logits, or one of a
probe's classes against its scores (a metric scores a `vocab_size` or a `classes` axis); `Featurizer` a
featurizer's name; anything else is a literal. Inside a read or write, `featurizer` and `select` are
arguments of every op: the op runs on the value's coordinates in that featurizer's space (on the features
`{"select": {"features": [...]}}` keeps; an operand is laid out on them) and the change is written back as a
delta, so an op that changes no coordinate changes nothing to the bit. The `featurizer` op calls a featurizer
instead, in one of its kind's modes (`encode` gives the coordinates themselves, `apply` the featurizer's
version of the value). `renormalize`'s reference is the value at the site before the step's writes.
"""

import inspect
import math
from typing import Annotated, Any, Callable, Literal

import torch
from torch import Tensor

from . import featurizers, registry
from .address import EXAMPLE_AXES, pos_axis
from .registry import Featurizer, Tokens, Value

# `engine`: the engine running the document. `original`: inside a write, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
# `eos`: the ids the generation `of` came from stops on.
SUPPLIED = ("engine", "original", "dims", "eos")
# A metric scores a vocabulary axis: the tokenizer's, or a probe's classes.
VOCABULARY = ("vocab_size", "classes")

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


def parameters(name: str, mode: str | None = None, kind: str | None = None) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones; the `featurizer` op's with
    those of its `mode` for a featurizer of `kind`, when both are given."""
    found = registry.spelled(OPS[name], SUPPLIED + ("of",))
    if name == "featurizer" and mode is not None and kind is not None:
        found.update(registry.spelled(featurizers.mode(kind, mode), ("f", "h")))
    return found


def check(name: str, args: dict) -> None:
    """Op `name` exists and `args` bind to its signature."""
    if name not in OPS:
        raise ValueError(f"unknown op {name!r}; valid: {', '.join(OPS)}")
    registry.check(OPS[name], args, SUPPLIED + ("of",), repr(name))


def call(name: str, value: Any, args: dict[str, Any], **supplied: Any) -> Any:
    """Run op `name` on `value` with `args` (document spelling, already bound to tensors); an argument the
    op's signature does not name goes to its `**` catch-all as spelled."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    kwargs = {names.get(k, k): v for k, v in args.items()}
    kwargs.update((k, v) for k, v in supplied.items() if k in names and k not in kwargs)
    return fn(value, **kwargs)


def out_dims(name: str, dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`; the `featurizer` op's are those its mode
    produces for a featurizer of `kind`."""
    if name == "featurizer":
        assert kind is not None
        return featurizers.layout(kind, args["mode"], dims)
    axes = DECLARED[name]
    for axis in axes["needs"]:
        if not set(axis if isinstance(axis, tuple) else (axis,)) & set(dims):
            raise ValueError(f"{name!r} needs a value with a {' or '.join(map(repr, axis)) if isinstance(axis, tuple) else repr(axis)} "
                             f"axis; this one is {dims}")
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
    maps a dimension of that layout to the indices kept along it: one list for
    every row, or (`[rows, n]`, on the `key` axis) each row's own.
    """
    x = value.movedim(pos_axis(component), 1)
    x = x[torch.arange(x.shape[0])[:, None], record["positions"]]
    for dim, index in record["select"].items():
        x = take(x, dim, index)
    return x


def take(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` at `index` along `dim`: the same indices on every row, or (`[rows, n]`) each row's own."""
    return x.index_select(dim, index) if index.dim() == 1 else x.gather(dim, per_row(x, dim, index))


def per_row(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Each row's `[rows, n]` indices expanded to `x`'s shape with `n` along `dim`, for gather and scatter."""
    shape = [1] * x.dim()
    shape[0], shape[dim] = index.shape
    size = list(x.shape)
    size[dim] = index.shape[1]
    return index.reshape(shape).expand(size)


def write(value: Tensor, component: str, writes: list[dict]) -> Tensor:
    """`value` with `writes` applied in listed order, out of place.

    Each write's ops run on its selection of the current value; `original` is
    the selection as it was before any of these writes. An op with a `featurizer` (an instance) runs on the
    selection's coordinates in its space, on the features its `select` keeps, and the change is written back
    (`featurizers.within`). A write narrows at most
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
            new, original = take(new, dim, index), take(original, dim, index)
        for o in w["ops"]:
            f, features = o["featurizer"], o.get("features")
            args = o["args"]
            if f is None:
                new = call(o["op"], new, args, original=original, dims=o["dims"])
                continue
            start = featurizers.encode(f, original)
            start = start if features is None else start.index_select(-1, features)
            new = featurizers.within(f, new, lambda z, args=args, o=o, start=start: call(
                o["op"], z, args, original=start, dims=o["dims"]), features)
        for dim, index in w["select"].items():
            new = new.to(current.dtype)
            new = current.index_copy(dim, index, new) if index.dim() == 1 else \
                current.scatter(dim, per_row(new, dim, index), new)
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


@op("transforms")
def featurizer(of: Tensor, featurizer: Featurizer, mode: str, **options: Any) -> Tensor:
    """`featurizer` called on the value in `mode`, one of its kind's modes, with the mode's own arguments
    beside (`with`, for a lora's `apply`): `encode` gives its coordinates (a probe's: its scores, one per
    class), `apply` its version of the value. This is the one op whose axes depend on an argument: they are
    the mode's, which the featurizer's kind declares (see Featurizer kinds), so it cannot declare them here."""
    return featurizers.call(featurizer, mode, of, options)


# -- reductions ----------------------------------------------------------------


@op("reductions", removes="over")
def mean(of: Tensor, dims: list[str], over: list[str] | None = None) -> Tensor:
    """The mean over the axes `over` names, of the rows that have a value (a missing row is NaN); by default
    over every axis but the feature axes."""
    over = over if over is not None else [a for a in dims if a in EXAMPLE_AXES]
    axes = [dims.index(a) for a in over]
    return of.nanmean(dim=axes) if of.isnan().any() else of.mean(dim=axes)


@op("reductions", needs=("hidden_size",), removes=EXAMPLE_AXES, adds=("k",), bundle="pca")
def pca(of: Tensor, k: int, dims: list[str]) -> dict:
    """The top-`k` principal directions (`basis`, `[hidden_size, k]`, orthonormal columns, the largest variance
    first) and the `mean` they are centred on: the bundle a `pca` featurizer loads."""
    x = of.reshape(-1, of.shape[-1]).double()
    x = x[~x.isnan().any(-1)]
    if x.shape[0] <= k:
        raise ValueError(f"pca of {x.shape[0]} vectors has no rank-{k} basis; it needs more than {k}")
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    # An SVD's singular vectors are unique only up to sign; fix it so the largest entry of each is positive.
    basis = vh[:k].T
    basis = basis * basis.gather(0, basis.abs().argmax(0, keepdim=True)).sign()
    return {"basis": basis.to(of.dtype).contiguous(), "mean": mean.to(of.dtype)}


# -- metrics: logits in, one number per row and position out -------------------


def _missing(of: Tensor) -> Tensor:
    """Where a value is missing, per vector: NaN anywhere along its last axis."""
    return of.isnan().any(-1)


def _per_row(ids: Tensor, x: Tensor, dims: list[str]) -> Tensor:
    """One id per row (`[rows]`), shaped to broadcast against `x` along its rows (or entries) axis."""
    shape = [1] * x.dim()
    shape[next(i for i, a in enumerate(dims) if a in ("rows", "entries"))] = -1
    return ids.reshape(shape)


def _pick(x: Tensor, ids: Tensor, dims: list[str]) -> Tensor:
    return x.gather(-1, _per_row(ids, x, dims).expand(*x.shape[:-1], 1)).squeeze(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def match(of: Tensor, expected: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """1 where the most likely token is `expected`, else 0."""
    hit = (of.argmax(-1, keepdim=True) == _per_row(expected, of, dims)).squeeze(-1).to(of.dtype)
    return hit.masked_fill(_missing(of), float("nan"))


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def logit_diff(of: Tensor, a: Tokens, b: Tokens, dims: list[str],
               token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `a` less the logit of `b`."""
    return _pick(of, a, dims) - _pick(of, b, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_prob(of: Tensor, token: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The probability of `token`."""
    return _pick(of.softmax(-1), token, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_logit(of: Tensor, token: Tokens, dims: list[str], token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The logit of `token`."""
    return _pick(of, token, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def cross_entropy(of: Tensor, target: Tokens, dims: list[str],
                  token_form: Literal["spaced", "bare"] = "spaced") -> Tensor:
    """The negative log-probability of `target`, in nats."""
    return -_pick(of.log_softmax(-1), target, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def kl(of: Tensor, against: Value) -> Tensor:
    """KL(of ‖ against) between the two next-token distributions, in nats."""
    p, q = of.log_softmax(-1), against.log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
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
    missing = _missing(of)[..., None]
    return {"ids": ids.masked_fill(missing, -1), "probs": probs.masked_fill(missing, float("nan"))}


@op("transforms", needs=("vocab_size",), removes=("vocab_size",))
def argmax(of: Tensor) -> Tensor:
    """The id of the most likely token (-1 where the value is missing)."""
    return of.argmax(-1).masked_fill(_missing(of), -1)


@op("transforms", needs=("new",), removes=("new",))
def decode(of: Tensor, engine: Any, eos: list[int], tokens: bool = False) -> list:
    """Each row's generated ids as text, stopped before the row's first EOS (the id its generation stops on);
    with `tokens`, as its token strings."""
    tokenizer, stops = engine.tokenizer, set(eos)
    out = []
    for ids in of.tolist():
        ids = next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
        out.append(tokenizer.convert_ids_to_tokens(ids) if tokens else tokenizer.decode(ids))
    return out
