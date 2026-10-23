"""Metrics: logits (or a probe's scores, or a residual-stream value for `cosine`) in, one number per row and
position out."""

import math

import torch
from torch import Tensor

from clab.format.data import compared
from clab.format.registry import Text, Tokens, Value
from clab.ops import VOCABULARY, op


def missing(of: Tensor) -> Tensor:
    """Where a value is missing, per vector: NaN anywhere along its last axis."""
    return of.isnan().any(-1)


def ranked(of: Tensor, k: int) -> Tensor:
    """The ids of the `k` largest values of the last axis, the largest first, an exact tie to the lower id: the
    one tie rule `match`, `argmax` and `top_k` share. A top-k finds them; where the k-th value ties with one it
    left out, which set it kept is the kernel's choice, so only those vectors are sorted whole, stably."""
    values, ids = of.topk(k, dim=-1)
    ids, order = ids.sort(dim=-1)                                   # by id, then stably by value, largest first
    ids = ids.gather(-1, values.gather(-1, order).sort(dim=-1, descending=True, stable=True).indices)
    kth = values[..., -1:]
    tied = (of == kth).sum(-1) > (values == kth).sum(-1)
    if tied.any():
        ids[tied] = of[tied].sort(dim=-1, descending=True, stable=True).indices[..., :k]
    return ids


# The metrics that read one id's logit, which a class or a set of several ids does not have.
ONE_ID = ("logit_diff", "token_logit")


def _aligned(ids: Tensor, x: Tensor, dims: list[str]) -> Tensor:
    """A token argument, on `x`'s device and expanded against `x`'s axes but its last: the ids each row may
    be, `[rows, m]` for the same answer at every position, or `[rows, pos, m]` for token i of the answer at
    position i; `m` is the most a row's set or class holds, padded with -1."""
    shape = [1] * (x.dim() - 1) + [ids.shape[-1]]
    shape[dims.index("rows")] = ids.shape[0]
    if ids.dim() == 3:
        shape[dims.index("pos")] = ids.shape[1]
    return ids.to(x.device).reshape(shape).expand(*x.shape[:-1], ids.shape[-1])


def _gather(x: Tensor, ids: Tensor, dims: list[str]) -> tuple[Tensor, Tensor]:
    """`x` at each of the ids along its last axis, and which of them are ids rather than padding."""
    index = _aligned(ids, x, dims)
    return x.gather(-1, index.clamp(min=0)), index >= 0


def _one(x: Tensor, ids: Tensor, dims: list[str]) -> Tensor:
    """`x` at the one id each position has."""
    return _gather(x, ids, dims)[0].squeeze(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def match(of: Tensor, expected: Tokens, dims: list[str]) -> Tensor:
    """1 where the most likely token is `expected` (any of its class or set), else 0; the most likely token is
    `argmax`'s, an exact tie to the lower id."""
    hit = (ranked(of, 1) == _aligned(expected, of, dims)).any(-1).to(of.dtype)
    return hit.masked_fill(missing(of), float("nan"))


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def logit_diff(of: Tensor, a: Tokens, b: Tokens, dims: list[str]) -> Tensor:
    """The logit of `a` less the logit of `b`."""
    return _one(of, a, dims) - _one(of, b, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_prob(of: Tensor, token: Tokens, dims: list[str]) -> Tensor:
    """The probability of `token` (of a class or a set, its members' summed)."""
    p, real = _gather(of.softmax(-1), token, dims)
    return p.squeeze(-1) if p.shape[-1] == 1 else p.masked_fill(~real, 0.0).sum(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_logit(of: Tensor, token: Tokens, dims: list[str]) -> Tensor:
    """The logit of `token`."""
    return _one(of, token, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def cross_entropy(of: Tensor, target: Tokens, dims: list[str]) -> Tensor:
    """The negative log-probability of `target` (of a class or a set, of its members' summed probability), in
    nats."""
    logp, real = _gather(of.log_softmax(-1), target, dims)
    return -(logp.squeeze(-1) if logp.shape[-1] == 1 else logp.masked_fill(~real, -math.inf).logsumexp(-1))


def distribution(args: dict) -> None:
    """`kl`'s and `js`'s `against` is a distribution: a named value, never a number or a flag."""
    against = args["against"]
    if not isinstance(against, str):
        raise ValueError(f"against {against!r} is not a name, and only a named value is a distribution; 'against' "
                         "names a value with the same vocabulary axis (another read's logits)")


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY, check=distribution)
def kl(of: Tensor, against: Value) -> Tensor:
    """KL(of ‖ against) between the two next-token distributions, in nats."""
    p, q = of.log_softmax(-1), against.to(of.device).log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY, check=distribution)
def js(of: Tensor, against: Value) -> Tensor:
    """The Jensen–Shannon divergence between the two distributions, in nats (at most log 2)."""
    p, q = of.log_softmax(-1), against.to(of.device).log_softmax(-1)
    m = torch.logaddexp(p, q) - math.log(2)
    return 0.5 * (p.exp() * (p - m)).sum(-1) + 0.5 * (q.exp() * (q - m)).sum(-1)


@op("metrics", needs=("hidden_size",), removes=("hidden_size",))
def cosine(of: Tensor, with_: Value) -> Tensor:
    """The cosine similarity of each vector and `with`'s, over `hidden_size`."""
    return torch.nn.functional.cosine_similarity(of, torch.as_tensor(with_, dtype=of.dtype, device=of.device), dim=-1)


@op("metrics", needs=("rows",))
def contains(of: list, text: Text, dims: list[str]) -> Tensor:
    """1 where a row's text (a `detokenize`d generation) contains `text`, the row's answer (any member of a set),
    compared as its reference says, else 0. Every row generated something, so none is missing."""
    hits = [any(compared(answer, strip, lower) in compared(said, False, lower) for answer in answers)
            for said, (answers, strip, lower) in zip(of, text)]
    return torch.tensor(hits, dtype=torch.float32)
