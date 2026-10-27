"""The `fit` step kind: train featurizers through the frozen model, then score its body under each eval."""

from __future__ import annotations

from collections import ChainMap, Counter
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from clab.compile.state import Found
from clab.compile.uses import instance
from clab.featurizers import Kind
from clab.format.sweeps import lookup, put
from clab.plan import Fit, Forward, OpStep, Plan, Step, agreeing, entries, held_rows
from clab.steps.batches import ROWS, home

if TYPE_CHECKING:
    from clab.engines import Engine

OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def fit(engine: Engine, plan: Plan, name: str, step: Fit, state: ChainMap, batch_size: int, found: Found) -> Counter:
    """Train the featurizers `train` names, then score the body once per named `eval` (`plan.Fit` says what
    each part means). The optimizer and every trained featurizer are made here, inside the run; each minibatch's
    gradients are `torch.autograd.grad` of the objective for the featurizers' tensors alone, so the model's
    weights never hold a gradient. The featurizers are made where the model is, so the optimizer runs there.
    Returns the model calls every run of its body made and the tokens they fed."""
    sources = lookup(found.bodies, name, step.at)
    # A compiled point's values are its own, never a sweep.
    epochs, seed, size = cast(int, step.epochs), cast(int, step.seed), cast(int, step.batch_size)
    trained = {f: instance(plan, spec, cast(int, spec.width), seed, home(engine))
               for f in step.train for spec in [agreeing(plan.featurizers[f], step.at)]}
    params = [p.requires_grad_() for f in trained.values() for p in f.parameters()]
    opt = OPTIMIZERS[step.optimizer.name](params, lr=step.optimizer.lr, weight_decay=step.optimizer.weight_decay)
    body_ops = [n for n, s in step.steps.items() if isinstance(s, OpStep)]
    n = len(held_rows(sources["train"][0], datasets(step)[0]))
    scalars = [(f, sc) for f, inst in trained.items() for sc in inst.scalars]
    history: dict = {**{o: [] for o in body_ops}, **{f"{f}.{sc}": [] for f, sc in scalars}}
    best, since, kept, count = None, 0, None, Counter()
    for epoch in range(epochs):
        for f in trained.values():
            f.anneal(epoch, epochs)
        # One integer per (seed, epoch): epochs stay below the prime, so no two pairs share a generator.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch)).tolist()
        for lo in range(0, n, size):
            with torch.enable_grad():
                for f in trained.values():
                    f.settle()
            body, made = body_run(engine, name, step, state, trained, sources["train"], order[lo:lo + size])
            count += made
            with torch.enable_grad():
                loss = sum(cast(float, weight) * term(body, trained, t, step.at)
                           for t, weight in step.objective.items())
            assert isinstance(loss, Tensor)
            for p, g in zip(params, torch.autograd.grad(loss, params)):
                p.grad = g
            opt.step()
            opt.zero_grad()
            for f in trained.values():
                f.project()
        for f in trained.values():
            f.settle()
        scored, made = body_run(engine, name, step, state, trained, sources["train"], None)
        count += made
        for o in body_ops:
            value = lookup(scored, o, step.at)
            if isinstance(value, Tensor):
                history[o].append(overall(value))
        for f, sc in scalars:
            history[f"{f}.{sc}"].append(trained[f].scalar(sc))
        if step.early_stop is None:
            kept = snapshot(trained)
            continue
        score, mode = history[step.early_stop.metric][-1], step.early_stop.mode
        if best is None or (score > best if mode == "max" else score < best):
            best, since, kept = score, 0, snapshot(trained)
        else:
            since += 1
            if since >= cast(int, step.early_stop.patience):
                break
    for f, inst in trained.items():
        vars(inst).update(kept[f] if kept else {})
        inst.settle()
    for label in step.eval:
        final, made = body_run(engine, name, step, state, trained, sources["evals"][label], None)
        count += made
        for key, s in step.steps.items():
            members = [key] if isinstance(s, OpStep) else []
            members += [f"{key}.{k}" for k, e in entries(s).items() if e.gathered]
            for member in members:
                put(state, f"{name}.{label}.{member}", step.at, lookup(final, member, step.at))
    for f, inst in trained.items():
        inst.seen = None
        put(state, f"{name}.{f}", step.at, inst)
    for key, values in history.items():         # `epochs: 0` scores the untrained draw: an empty history
        put(state, f"{name}.history.{key}", step.at, torch.stack(values) if values else torch.zeros(0))
    return count


def body_run(engine: Engine, name: str, step: Fit, state: ChainMap, trained: dict[str, Kind],
             source: tuple[Plan, Found], ids: list[int] | None) -> tuple[ChainMap, Counter]:
    """A fit's body over every row `source` holds (its training rows or eval's), or the minibatch of dataset row ids
    `ids`, in a scope of its own that falls back to `state` and carries those ids (`batches.ROWS`: what its values'
    rows are), each trained featurizer under its declared name and `<fit>.<name>`. With a minibatch it trains:
    gradients on, gates soft; otherwise it scores. Returns the body's namespace, and the model calls its steps made
    and the tokens they fed."""
    plan, found = source
    body = state.new_child({ROWS: ids} if ids is not None else {})
    for f, inst in trained.items():
        inst.training = ids is not None
        inst.seen = []
        put(body, f, {}, inst)
        put(body, f"{name}.{f}", {}, inst)
    from clab.steps import KINDS
    count = Counter()
    with torch.set_grad_enabled(ids is not None):
        for n, s in step.steps.items():
            count += KINDS[cast(Step, s).kind](engine, plan, n, s, body, step.batch_size, found)
    return body, count


def datasets(step: Fit) -> list[str]:
    """The datasets a fit's body runs over, which have as many rows as each other."""
    body = cast("dict[str, Forward | OpStep]", step.steps)          # a fit's body holds no fit (compile refuses one)
    return sorted({s.dataset for s in body.values() if s.dataset is not None})


def term(body: ChainMap, trained: dict[str, Kind], name: str, at: dict) -> Tensor:
    """An objective term: a body op's value averaged over every axis, or a trained featurizer's scalar."""
    thing, _, member = name.partition(".")
    if member and thing in trained:
        return trained[thing].scalar(member)
    return overall(lookup(body, name, at))


def overall(value: Tensor) -> Tensor:
    """A value's mean over every axis, of the entries that have one (a missing row is NaN)."""
    return value.float().nanmean() if value.isnan().any() else value.float().mean()


def snapshot(trained: dict[str, Kind]) -> dict[str, dict[str, Any]]:
    """Each trained featurizer as it is now, its tensors detached copies: what restoring the epoch puts back."""
    return {f: {k: v.detach().clone() if isinstance(v, Tensor) else v for k, v in vars(inst).items() if k != "seen"}
            for f, inst in trained.items()}
