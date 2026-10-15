"""Metrics: logits (or a probe's scores, or a residual-stream value for `cosine`) in, one number per row and
position out."""

import math

import torch
from torch import Tensor

from ...registry import Tokens, Value
from . import VOCABULARY, op


def missing(of: Tensor) -> Tensor:
    """Where a value is missing, per vector: NaN anywhere along its last axis."""
    return of.isnan().any(-1)


def _per_row(ids: Tensor, x: Tensor, dims: list[str]) -> Tensor:
    """One id per row (`[rows]`), on `x`'s device, shaped to broadcast against `x` along its rows (or entries)
    axis."""
    shape = [1] * x.dim()
    shape[next(i for i, a in enumerate(dims) if a in ("rows", "entries"))] = -1
    return ids.to(x.device).reshape(shape)


def _pick(x: Tensor, ids: Tensor, dims: list[str]) -> Tensor:
    return x.gather(-1, _per_row(ids, x, dims).expand(*x.shape[:-1], 1)).squeeze(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def match(of: Tensor, expected: Tokens, dims: list[str]) -> Tensor:
    """1 where the most likely token is `expected`, else 0."""
    hit = (of.argmax(-1, keepdim=True) == _per_row(expected, of, dims)).squeeze(-1).to(of.dtype)
    return hit.masked_fill(missing(of), float("nan"))


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def logit_diff(of: Tensor, a: Tokens, b: Tokens, dims: list[str]) -> Tensor:
    """The logit of `a` less the logit of `b`."""
    return _pick(of, a, dims) - _pick(of, b, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_prob(of: Tensor, token: Tokens, dims: list[str]) -> Tensor:
    """The probability of `token`."""
    return _pick(of.softmax(-1), token, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def token_logit(of: Tensor, token: Tokens, dims: list[str]) -> Tensor:
    """The logit of `token`."""
    return _pick(of, token, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def cross_entropy(of: Tensor, target: Tokens, dims: list[str]) -> Tensor:
    """The negative log-probability of `target`, in nats."""
    return -_pick(of.log_softmax(-1), target, dims)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def kl(of: Tensor, against: Value) -> Tensor:
    """KL(of ‖ against) between the two next-token distributions, in nats."""
    p, q = of.log_softmax(-1), against.to(of.device).log_softmax(-1)
    return (p.exp() * (p - q)).sum(-1)


@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def js(of: Tensor, against: Value) -> Tensor:
    """The Jensen–Shannon divergence between the two distributions, in nats (at most log 2)."""
    p, q = of.log_softmax(-1), against.to(of.device).log_softmax(-1)
    m = torch.logaddexp(p, q) - math.log(2)
    return 0.5 * (p.exp() * (p - m)).sum(-1) + 0.5 * (q.exp() * (q - m)).sum(-1)


@op("metrics", needs=("hidden_size",), removes=("hidden_size",))
def cosine(of: Tensor, with_: Value) -> Tensor:
    """The cosine similarity of each vector and `with`'s, over `hidden_size`."""
    return torch.nn.functional.cosine_similarity(of, torch.as_tensor(with_, dtype=of.dtype, device=of.device), dim=-1)
