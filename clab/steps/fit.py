"""The `fit` step kind: train featurizers through the frozen model, then score its body under each eval."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
from torch import Tensor

from clab.compile.state import Found
from clab.compile.uses import instance
from clab.featurizers import Kind
from clab.format.sweeps import lookup, put
from clab.plan import Fit, Forward, OpStep, Plan, Step, agreeing, entries, held_rows
from clab.steps.batches import home

if TYPE_CHECKING:
    from clab.engines import Engine

OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def fit(engine: Engine, plan: Plan, name: str, step: Fit, state: dict, batch_size: int, found: Found) -> None:
    """Train the featurizers `train` names, then score the body once per named `eval` (`plan.Fit` says what
    each part means). The optimizer and every trained featurizer are made here, inside the run; each minibatch's
    gradients are `torch.autograd.grad` of the objective for the featurizers' tensors alone, so the model's
    weights never hold a gradient. The featurizers are made where the model is, so the optimizer runs there.
    The constant forwards' model calls are kept in `cache` (`planned`) while a later pass runs the same rows."""
    sources = lookup(found.bodies, name, step.at)
    # A compiled point's values are its own, never a sweep.
    epochs, seed, size = cast(int, step.epochs), cast(int, step.seed), cast(int, step.batch_size)
    trained = {f: instance(plan, agreeing(plan.featurizers[f], step.at), found.widths[f], seed, home(engine))
               for f in step.train}
    params = [p.requires_grad_() for f in trained.values() for p in f.parameters()]
    opt = OPTIMIZERS[step.optimizer.name](params, lr=step.optimizer.lr, weight_decay=step.optimizer.weight_decay)
    body_ops = [n for n, s in step.steps.items() if isinstance(s, OpStep)]
    n = len(held_rows(sources["train"][0], datasets(step)[0]))
    scalars = [(f, sc) for f, inst in trained.items() for sc in inst.scalars]
    history: dict = {**{o: [] for o in body_ops}, **{f"{f}.{sc}": [] for f, sc in scalars}}
    best, since, kept = None, 0, None
    # One integer per (seed, epoch): epochs stay below the prime, so no two pairs share a generator.
    orders = [torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch)).tolist()
              for epoch in range(epochs)]
    # Each pass is keyed by the eval substitution it reads its rows under and the rows, in order.
    evals = {label: (tuple(sorted(swap.items())), tuple(range(len(held_rows(sources["evals"][label][0],
                                                                            datasets(step)[0])))))
             for label, swap in step.eval.items()}
    cache = planned(step, orders, n, list(evals.values()))
    for epoch, order in enumerate(orders):
        for f in trained.values():
            f.anneal(epoch, epochs)
        for lo in range(0, n, size):
            with torch.enable_grad():
                for f in trained.values():
                    f.settle()
            minibatch = order[lo:lo + size]
            body = body_run(engine, name, step, state, trained, sources["train"], minibatch, ((), tuple(minibatch)),
                            cache)
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
        scored = body_run(engine, name, step, state, trained, sources["train"], None, ((), tuple(range(n))), cache)
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
    if cache is not None:                   # an early stop leaves the passes it skipped counted: only the evals run on
        cache["uses"] = Counter(evals.values())
        for done in [k for k in cache["calls"] if not cache["uses"][k[1]]]:
            del cache["calls"][done]
    for label in step.eval:
        final = body_run(engine, name, step, state, trained, sources["evals"][label], None, evals[label], cache)
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


def body_run(engine: Engine, name: str, step: Fit, state: dict, trained: dict[str, Kind], source: tuple[Plan, Found],
             index: list[int] | None, key: tuple, cache: dict | None) -> dict:
    """A fit's body over the rows `source` holds (its training rows or an eval's), or the minibatch `index` of
    them, in a namespace of its own that falls back to `state`, each trained featurizer under its declared name
    and `<fit>.<name>`. With a minibatch it trains: gradients on, gates soft; otherwise it scores. A constant step
    runs without gradients; a constant forward's model calls are those `cache` kept from an earlier pass the same
    `key` names (the eval substitution and the rows, in order), replayed, or are made and kept there when a later
    pass will reuse them (`planned`). Returns the body's namespace."""
    plan, found = source if index is None else narrowed(source[0], source[1], index, datasets(step))
    if cache is not None:
        cache["uses"][key] -= 1
    body = dict(state)
    for f, inst in trained.items():
        inst.training = index is not None
        inst.seen = []
        put(body, f, {}, inst)
        put(body, f"{name}.{f}", {}, inst)
    from clab.steps import KINDS
    with torch.set_grad_enabled(index is not None):
        for n, s in step.steps.items():
            s = cast(Step, s)
            if n not in step.constant:
                KINDS[s.kind](engine, plan, n, s, body, step.batch_size, found)
            elif s.kind == "op":
                with torch.no_grad():
                    KINDS["op"](engine, plan, n, s, body, step.batch_size, found)
            else:
                assert cache is not None, "planned keeps a cache whenever a forward is constant"
                replay = iter(cache["calls"][n, key]) if (n, key) in cache["calls"] else None
                kept = cache["calls"].setdefault((n, key), []) if replay is None and cache["uses"][key] > 0 else None

                def call(*args: Any, replay: Iterator | None = replay, kept: list | None = kept) -> dict[str, Tensor]:
                    if replay is not None:
                        return next(replay)
                    with torch.no_grad():
                        got = engine.forward(*args)
                    if kept is not None:
                        kept.append(got)
                    return got
                KINDS["forward"](engine, plan, n, s, body, step.batch_size, found, call=call)
    if cache is not None and cache["uses"][key] <= 0:         # no later pass runs these rows this way
        for done in [k for k in cache["calls"] if k[1] == key]:
            del cache["calls"][done]
    return body


def planned(step: Fit, orders: list[list[int]], n: int, evals: list[tuple]) -> dict | None:
    """A fit's cache, before its first pass, when a forward of its body is constant (else None): how many passes
    will run each key, the eval substitution they read their rows under and the rows in order (`uses`), from the
    shuffle of each epoch (`orders`) over the `n` training rows, each scoring pass and each eval's key (`evals`);
    and room for the kept model calls (`calls`, by step and key).

    A key is a whole pass's rows, never one row's: a row's numbers can depend on the rest of its batch. Measured
    on 2026-10-26 with transformers 5.20: on CPU (torch 2.14, float32, tiny Llama, both engines), at 12 threads a
    row moved from the first place to the second of a batch of two changed by up to 9e-8 (none at 1 or 4 threads,
    none alone against sizes 3 to 16, none ever on tiny GPT-2), and at 24 threads a row alone differed from itself
    in a batch of two or four, its residual stream by 2e-10; on an RTX A6000 (torch 2.13, Llama-3.2-1B, both
    engines), a row alone differed from itself in a batch of 16 by up to 4e-4 in float32 and 2 in bfloat16, and a
    row of the input's full width by up to 9e-5 and 0.16 between a batch of two with padding and one without. The
    same rows in the same order gave the same bits every time, which test_fit_cache.py checks."""
    if not any(cast(Step, step.steps[c]).kind == "forward" for c in step.constant):
        return None
    size = cast(int, step.batch_size)
    passes = [((), tuple(order[lo:lo + size])) for order in orders for lo in range(0, n, size)]
    passes += [((), tuple(range(n)))] * len(orders) + evals
    return {"uses": Counter(passes), "calls": {}}


def datasets(step: Fit) -> list[str]:
    """The datasets a fit's body runs over, which have as many rows as each other."""
    body = cast("dict[str, Forward | OpStep]", step.steps)          # a fit's body holds no fit (compile refuses one)
    return sorted({s.dataset for s in body.values() if s.dataset is not None})


def narrowed(plan: Plan, found: Found, index: list[int], names: list[str]) -> tuple[Plan, Found]:
    """A body's plan and findings over the rows `index` picks of each of its datasets (`names`)."""
    data = {**plan.data, **{d: plan.data[d].model_copy(update={"rows": [held_rows(plan, d)[i] for i in index]})
                            for d in names}}
    return plan.model_copy(update={"data": data}), Found(
        inputs={k: {**v, "rows": [v["rows"][i] for i in index]} for k, v in found.inputs.items()},
        positions={w: {c: [p[i] for i in index] for c, p in by.items()} for w, by in found.positions.items()},
        selects=found.selects,
        answers={k: replace(a, values=[a.values[i] for i in index], sets=[a.sets[i] for i in index])
                 if a.column.partition(".")[0] in names else a for k, a in found.answers.items()})


def term(body: dict, trained: dict[str, Kind], name: str, at: dict) -> Tensor:
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
