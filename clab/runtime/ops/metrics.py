"""Metrics: logits (or a probe's scores) in, one number per row and position out. Beside them, the
transforms over logits and generated ids (`top_k`, `argmax`, `decode`), which register after the metrics."""

import math
from typing import Any, Literal

import torch
from torch import Tensor

from ...plan.registry import Tokens, Value
from . import VOCABULARY, op


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
