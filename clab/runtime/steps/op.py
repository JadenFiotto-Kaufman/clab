"""The `op` step kind: an op over the whole of a value an earlier step produced."""

from torch import Tensor

from ...plan.model import Plan, arguments, dims_of, points, probe_of
from . import kind
from .batches import apply
from .generate import stops
from .state import bound, lookup, put


@kind
def op(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The op over the whole value `of` names. `top_k`'s value also holds its tokens, decoded."""
    of = lookup(state, step.of, step.at)
    producer = points(plan.steps[step.of.partition(".")[0]])[0]
    vocabulary = lookup(state, probe_of(plan, step.of), step.at)["classes"] if "classes" in dims_of(plan, step.of) \
        else None
    value = apply(engine, step.op, of, bound(engine, plan, step.op, arguments(step), state, step.at, None, of,
                                             vocabulary),
                  dims_of(plan, step.of), stops(engine, producer) if producer.kind == "generate" else engine.eos)
    if step.op == "top_k":
        value["tokens"] = decoded(engine.tokenizer, value["ids"])
    put(state, name, step.at, value)


def decoded(tokenizer, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are; None for a missing one (-1)."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    return [tokenizer.decode([i]) if i >= 0 else None for i in ids.tolist()]
