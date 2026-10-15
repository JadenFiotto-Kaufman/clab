"""The `fit` step kind: train featurizers through the frozen model, then score its body under each eval."""

import torch
from torch import Tensor

from ...plan.model import Plan, agreeing
from . import KINDS, kind
from .preflight import instance
from .state import lookup, put


OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


@kind
def fit(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """Train the featurizers `train` names, then score the body once per named `eval` (`plan.Fit` says what
    each part means). The optimizer and every trained featurizer are made here, inside the run; each minibatch's
    gradients are `torch.autograd.grad` of the objective for the featurizers' tensors alone, so the model's
    weights never hold a gradient."""
    sources = lookup(found["bodies"], name, step.at)
    trained = {f: instance(plan, agreeing(plan.featurizers[f], step.at), found["widths"][f], step.seed)
               for f in step.train}
    params = [p.requires_grad_() for f in trained.values() for p in f.parameters()]
    optimizer = step.optimizer
    opt = OPTIMIZERS[optimizer.name](params, lr=optimizer.lr, weight_decay=optimizer.weight_decay)
    body_ops = [n for n, s in step.steps.items() if s.kind == "op"]
    n = len(sources["train"][0].data[next(iter(datasets(step)))].rows or [])
    scalars = [(f, sc) for f, inst in trained.items() for sc in inst.scalars]
    history: dict = {**{o: [] for o in body_ops}, **{f"{f}.{sc}": [] for f, sc in scalars}}
    best, since, kept = None, 0, None
    for epoch in range(step.epochs):
        for f in trained.values():
            f.anneal(epoch, step.epochs)
        # One integer per (seed, epoch): epochs stay below the prime, so no two pairs share a generator.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(step.seed * 1_000_003 + epoch)).tolist()
        for lo in range(0, n, step.batch_size):
            with torch.enable_grad():
                for f in trained.values():
                    f.settle()
            body = body_run(engine, name, step, state, trained, sources["train"], order[lo:lo + step.batch_size])
            with torch.enable_grad():
                loss = sum(weight * term(body, trained, t, step.at) for t, weight in step.objective.items())
            assert isinstance(loss, Tensor)
            for p, g in zip(params, torch.autograd.grad(loss, params)):
                p.grad = g
            opt.step()
            opt.zero_grad()
            for f in trained.values():
                f.project()
        for f in trained.values():
            f.settle()
        scored = body_run(engine, name, step, state, trained, sources["train"], None)
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
            if since >= step.early_stop.patience:
                break
    for f, inst in trained.items():
        vars(inst).update(kept[f] if kept else {})
        inst.settle()
    for label in step.eval:
        final = body_run(engine, name, step, state, trained, sources["evals"][label], None)
        for key, s in step.steps.items():
            members = ([key] if s.kind == "op" else []) + [f"{key}.{read}" for read in getattr(s, "reads", {})]
            for member in members:
                put(state, f"{name}.{label}.{member}", step.at, lookup(final, member, step.at))
    for f, inst in trained.items():
        inst.seen = None
        put(state, f"{name}.{f}", step.at, inst)
    for key, values in history.items():
        if values:
            put(state, f"{name}.history.{key}", step.at, torch.stack(values))


def body_run(engine, name: str, step, state: dict, trained: dict, source: tuple, index: list[int] | None) -> dict:
    """A fit's body over the rows `source` holds (its training rows or eval's), or the minibatch `index` of
    them, in a namespace of its own that falls back to `state`, each trained featurizer under its declared
    name and `<fit>.<name>`. With a minibatch it trains: gradients on, gates soft; otherwise it scores.
    Returns the body's namespace."""
    plan, found = source if index is None else narrowed(source[0], source[1], index, datasets(step))
    body = dict(state)
    for f, inst in trained.items():
        inst.training = index is not None
        inst.seen = []
        put(body, f, {}, inst)
        put(body, f"{name}.{f}", {}, inst)
    with torch.set_grad_enabled(index is not None):
        for n, s in step.steps.items():
            KINDS[s.kind](engine, plan, n, s, body, step.batch_size, found)
    return body


def datasets(step) -> list[str]:
    """The datasets a fit's body runs over, which have as many rows as each other."""
    return sorted({s.dataset for s in step.steps.values() if s.dataset is not None})


def narrowed(plan: Plan, found: dict, index: list[int], names: list[str]) -> tuple[Plan, dict]:
    """A body's plan and findings over the rows `index` picks of each of its datasets (`names`)."""
    data = {**plan.data, **{d: plan.data[d].model_copy(update={"rows": [(plan.data[d].rows or [])[i] for i in index]})
                            for d in names}}
    return plan.model_copy(update={"data": data}), {
        "inputs": {k: {**v, "rows": [v["rows"][i] for i in index]} for k, v in found["inputs"].items()},
        "positions": {w: {c: [p[i] for i in index] for c, p in by.items()} for w, by in found["positions"].items()},
        "said": {}, "widths": found["widths"]}


def term(body: dict, trained: dict, name: str, at: dict) -> Tensor:
    """An objective term: a body op's value averaged over every axis, or a trained featurizer's scalar."""
    thing, _, member = name.partition(".")
    if member and thing in trained:
        return trained[thing].scalar(member)
    return overall(lookup(body, name, at))


def overall(value: Tensor) -> Tensor:
    """A value's mean over every axis, of the entries that have one (a missing row is NaN)."""
    return value.float().nanmean() if value.isnan().any() else value.float().mean()


def snapshot(trained: dict) -> dict:
    """Each trained featurizer as it is now, its tensors detached copies: what restoring the epoch puts back."""
    return {f: {k: v.detach().clone() if isinstance(v, Tensor) else v for k, v in vars(inst).items() if k != "seen"}
            for f, inst in trained.items()}
