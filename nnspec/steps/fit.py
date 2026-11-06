"""The `fit` step kind: train featurizers through the frozen model; its evals are ordinary steps after it."""

from __future__ import annotations

from collections import ChainMap
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from nnspec.compile.schedule import values_read_by
from nnspec.compile.state import Findings
from nnspec.compile.uses import instance
from nnspec.featurizers import Kind
from nnspec.format.sweeps import agreeing_point, lookup, put
from nnspec.plan import Fit, Forward, OpStep, Plan, Step, body_view, held_rows, open_encodes, tensors_named
from nnspec.steps.batches import ROWS, model_device, schedule_binding

if TYPE_CHECKING:
    from nnspec.engines import Engine

OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def fit(engine: Engine, plan: Plan, name: str, step: Fit, state: ChainMap, batch_size: int, findings: Findings,
        cost: dict[str, Any]) -> None:
    """Train the featurizers `train` names (`plan.Fit` says what each part means), each body run in the plan as
    the body sees it (`plan.body_view`), over what preflight found (a generation's rows said nothing the run
    records). The optimizer and every trained featurizer are made here, inside the run; each minibatch's gradients
    are `torch.autograd.grad` of the objective for the featurizers' tensors alone, so the model's weights never
    hold a gradient. The featurizers are made where the model is, so the optimizer runs there. Every run of its
    body adds the model calls it made and the tokens they fed to the fit's `cost`. It ends with each trained
    featurizer scoring, as `<fit>.<featurizer>` in `state`, where the fit's eval steps after it find it, and each of
    its tensors the plan names beside it (`<fit>.<featurizer>.<tensor>`, a copy: `Kind.value`)."""
    view, findings = body_view(plan, name, step.at), replace(findings, said={})
    # A compiled point's values are its own, never a sweep.
    epochs, seed, size = cast(int, step.epochs), cast(int, step.seed), cast(int, step.batch_size)
    trained = {f: instance(plan, spec, cast(int, spec.width), seed, model_device(engine), state)
               for f in step.train for spec in [agreeing_point(plan.featurizers[f], step.at)]}
    params = [getattr(f, t).requires_grad_() for f in trained.values() for t in f.trains]
    opt = OPTIMIZERS[step.optimizer.name](params, lr=step.optimizer.lr, weight_decay=step.optimizer.weight_decay)
    body_ops = [n for n, s in step.steps.items() if isinstance(s, OpStep)]
    # What each body forward reads and the half of binding its schedule that depends on the step alone, once for every
    # run of the body: a body step is one point, which serves itself.
    values_read = {n: values_read_by([s]) for n, s in step.steps.items() if isinstance(s, Forward)}
    bindings = {n: schedule_binding(engine, view, n, s, values_read[n]) for n, s in step.steps.items()
                if isinstance(s, Forward)}
    # The rows of the first of the body's datasets, which have as many as each other; a body holds no fit (compile
    # refuses one).
    n = len(held_rows(plan, min(s.dataset for s in cast("dict[str, Forward | OpStep]", step.steps).values()
                                if s.dataset is not None)))
    # The scalars the body computes: an sae's are over its open encodes, which a body that applies it alone has none of.
    scalars = [(f, sc) for f, inst in trained.items() for sc in inst.scalars
               if sc not in inst.over_encodes or open_encodes(step.steps, f"{name}.{f}")]
    history: dict = {**{o: [] for o in body_ops}, **{f"{f}.{sc}": [] for f, sc in scalars}}
    best, since, kept = None, 0, None
    for epoch in range(epochs):
        for f in trained.values():
            f.anneal(epoch, epochs)
        # One integer per (seed, epoch): epochs stay below the prime, so no two pairs share a generator.
        shuffled_ids = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch)).tolist()
        for lo in range(0, n, size):
            with torch.enable_grad():
                for f in trained.values():
                    f.settle()
            body = run_body(engine, name, step, state, trained, view, findings, shuffled_ids[lo:lo + size], cost,
                            values_read, bindings)
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
            del body, loss          # a pass's values die before the next pass runs
        for f in trained.values():
            f.settle()
        scored = run_body(engine, name, step, state, trained, view, findings, None, cost, values_read, bindings)
        for o in body_ops:
            value = lookup(scored, o, step.at)
            if isinstance(value, Tensor):
                history[o].append(mean_over_axes(value))
        del scored
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
    named = tensors_named(plan)
    for f, inst in trained.items():
        vars(inst).update(kept[f] if kept else {})
        inst.settle()
        inst.training, inst.seen = False, None
        put(state, f"{name}.{f}", step.at, inst)
        for tensor in (t for t in inst.laid if f"{name}.{f}.{t}" in named):    # as the fit left it, a copy
            put(state, f"{name}.{f}.{tensor}", step.at, inst.value(tensor))
    for key, values in history.items():         # `epochs: 0` scores the untrained draw: an empty history
        put(state, f"{name}.history.{key}", step.at, torch.stack(values) if values else torch.zeros(0))


def run_body(engine: Engine, name: str, step: Fit, state: ChainMap, trained: dict[str, Kind], plan: Plan,
             findings: Findings, ids: list[int] | None, cost: dict[str, Any], values_read: dict[str, dict],
             bindings: dict[str, list[dict]]) -> ChainMap:
    """A fit's body over every training row, or the minibatch of dataset row ids `ids`, in `plan` (as the body sees
    it) over what preflight found (`findings`), in a scope of its own that falls back to `state` and carries those ids
    (`batches.ROWS`: what its values' rows are), each trained featurizer under `<fit>.<name>`, each body forward
    given the one point it serves, itself, what it reads (`values_read`, by name) and the half of binding its schedule
    that depends on the step alone (`bindings`, by name: `batches.schedule_binding`'s). With a minibatch it trains:
    gradients on, gates soft; otherwise it scores. Returns the body's namespace; the model calls its steps made and
    the tokens they fed add to the fit's `cost`."""
    body = state.new_child({ROWS: ids} if ids is not None else {})
    for f, inst in trained.items():
        inst.training = ids is not None
        inst.seen = []
        put(body, f"{name}.{f}", {}, inst)
    from nnspec.steps import KINDS
    with torch.set_grad_enabled(ids is not None):
        for n, s in step.steps.items():
            KINDS[cast(Step, s).kind](engine, plan, n, s, body, step.batch_size, findings, cost,
                                      *([[s], values_read[n], bindings[n]] if n in values_read else []))
    return body


def term(body: ChainMap, trained: dict[str, Kind], name: str, at: dict) -> Tensor:
    """An objective term: a body op's value averaged over every axis, or a trained featurizer's scalar."""
    thing, _, member = name.partition(".")
    if member and thing in trained:
        return trained[thing].scalar(member)
    return mean_over_axes(lookup(body, name, at))


def mean_over_axes(value: Tensor) -> Tensor:
    """A value's mean over every axis, of the entries that have one (a missing row is NaN)."""
    return value.float().nanmean() if value.isnan().any() else value.float().mean()


def snapshot(trained: dict[str, Kind]) -> dict[str, dict[str, Any]]:
    """Each trained featurizer as it is now, its tensors detached copies: what restoring the epoch puts back."""
    return {f: {k: v.detach().clone() if isinstance(v, Tensor) else v for k, v in vars(inst).items() if k != "seen"}
            for f, inst in trained.items()}
