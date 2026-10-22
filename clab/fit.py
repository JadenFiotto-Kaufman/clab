"""The `fit` step kind: train featurizers through the frozen model, then score its body under each eval."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import featurizers
from .featurizers.uses import instance
from .plan import Intervention, Plan, agreeing, arguments, entries, fail, gathers, kind_of, points
from .batches import home
from .sweeps import lookup, put

if TYPE_CHECKING:
    from .compile import Compiling


OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def fit(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """Train the featurizers `train` names, then score the body once per named `eval` (`plan.Fit` says what
    each part means). The optimizer and every trained featurizer are made here, inside the run; each minibatch's
    gradients are `torch.autograd.grad` of the objective for the featurizers' tensors alone, so the model's
    weights never hold a gradient. The featurizers are made where the model is, so the optimizer runs there."""
    sources = lookup(found["bodies"], name, step.at)
    device = home(engine)
    trained = {f: instance(plan, agreeing(plan.featurizers[f], step.at), found["widths"][f], step.seed, device)
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
            members = ([key] if s.kind == "op" else []) + [f"{key}.{k}" for k, e in entries(s).items() if e.gathered]
            for member in members:
                put(state, f"{name}.{label}.{member}", step.at, lookup(final, member, step.at))
    for f, inst in trained.items():
        inst.seen = None
        put(state, f"{name}.{f}", step.at, inst)
    for key, values in history.items():         # `epochs: 0` scores the untrained draw: an empty history
        put(state, f"{name}.history.{key}", step.at, torch.stack(values) if values else torch.zeros(0))


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
    from .steps import KINDS
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
        "said": {}, "selects": found["selects"], "widths": found["widths"],
        "answers": {k: {**a, "values": [a["values"][i] for i in index], "sets": [a["sets"][i] for i in index]}
                    if a["column"].partition(".")[0] in names else a for k, a in found["answers"].items()}}


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


def compile_fit(cx: Compiling, path: str, name: str, spec, plan: Plan) -> dict:
    """A fit's body runs over minibatches of its datasets' rows, the same rows of each: so its datasets have
    as many rows as each other (under each of `eval`'s substitutions too), and a value with a row per example it uses
    is one the body makes. It trains featurizers that have something to train, and holds no fit."""
    for i, f in enumerate(spec.train):
        if f not in plan.featurizers:
            fail(f"{path}.train[{i}]", f"no featurizer {f!r}; declared: {', '.join(plan.featurizers) or 'none'}")
        trainable = [k for k, c in featurizers.KINDS.items() if c.trains]
        if kind_of(plan, f) not in trainable:
            fail(f"{path}.train[{i}]", f"{f!r} is a {kind_of(plan, f)}, which a bundle fixes; it has nothing to "
                 f"train (trainable kinds: {', '.join(trainable)})")
        if any((p.model_extra or {}).get("file_path") for p in points(plan.featurizers[f])):
            fail(f"{path}.train[{i}]", f"{f!r} is loaded from a bundle, which fixes it; a fit trains a featurizer "
                 "it draws")
    within = replace(cx, names=dict(cx.names), aliases={f: f"{name}.{f}" for f in spec.train})
    for f in spec.train:
        within.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
    before = len(cx.refs)
    body = {}
    for n, s in spec.steps.items():
        if s.kind == "fit":
            fail(f"{path}.steps.{n}", "a fit's body holds no fit")
        if n in plan.data or n in plan.featurizers or "." in n:
            fail(f"{path}.steps.{n}", f"{n!r} is also a dataset or a featurizer, or has a dot; a body's names are "
                 "a root's")
        from .compile import step
        body[n] = step(within, f"{path}.steps.{n}", n, s, plan)
    inner = within.names
    for ref in cx.refs[before:]:
        if inner.get(ref) is cx.names.get(ref) and {"rows", "entries"} & set(cx.names[ref]["dims"]):
            fail(f"{path}.steps", f"the body uses {ref!r}, which has a row per example of the whole dataset; the "
                 "body runs over minibatches of its rows, so such a value is one it makes")
    body_ops = [n for n, s in body.items() if s["kind"] == "op"]
    for term in spec.objective:
        thing, _, scalar = term.partition(".")
        if term not in body_ops and not (thing in spec.train and scalar in featurizers.KINDS[
                kind_of(plan, thing)].scalars):
            fail(f"{path}.objective.{term}", f"not a body op or a trained featurizer's scalar; body ops: "
                 f"{', '.join(body_ops)}")
        if term not in body_ops:            # a scalar is over what the body's reads of its coordinates saw
            for n, s in body.items():
                for k, e in s.get("interventions", {}).items():
                    if any(o.op == "encode" and arguments(o)["featurizer"] in (thing, f"{name}.{thing}")
                           for o in gathers(Intervention(**e))):
                        cx.used.add(f"{n}.{k}")
    if spec.early_stop is not None and spec.early_stop.metric not in body_ops:
        fail(f"{path}.early_stop.metric",
             f"{spec.early_stop.metric!r} is not a body op; body ops: {', '.join(body_ops)}")
    for label, swap in spec.eval.items():
        where = f"{path}.eval.{label}"
        if label == "history" or label in spec.train or "." in label:
            fail(where, f"{label!r} names the fit's history or a featurizer it trains, or has a dot; an eval's name "
                 "is its own")
        for a, b in swap.items():
            for d in (a, b):
                if d not in cx.rows:
                    fail(where, f"no dataset {d!r}; datasets: {', '.join(cx.rows)}")
    for label, side in {"": {}, **spec.eval}.items():
        counts = {d: len(cx.rows[side.get(d, d)]) for d in {s.get("dataset") for s in body.values()} - {None}}
        if len(set(counts.values())) > 1:
            fail(f"{path}.steps", f"the body's datasets have {counts} rows" + (f" under eval {label!r}" if label else "")
                 + "; it runs over minibatches of the same rows of each, paired row by row, so they have as many rows: "
                 "repeat the smaller dataset's rows (or cut the larger's) to the same count, or fit on one")
    for f in spec.train:
        cx.names[f"{name}.{f}"] = {"dims": [], "data": None, "kind": "featurizer"}
        for scalar in featurizers.KINDS[kind_of(plan, f)].scalars:
            cx.names[f"{name}.history.{f}.{scalar}"] = {"dims": ["epochs"], "data": None, "kind": "value"}
    for n, s in body.items():
        if s["kind"] == "op":
            cx.names[f"{name}.history.{n}"] = {"dims": ["epochs"], "data": None, "kind": "value"}
        members = ([n] if s["kind"] == "op" else []) + [f"{n}.{k}" for k in s.get("interventions", {})]
        for label, swap in spec.eval.items():
            for member in members:
                info = inner[member]
                cx.names[f"{name}.{label}.{member}"] = {**info, "data": swap.get(info["data"], info["data"])}
    return {**spec.model_dump(exclude_defaults=True), "steps": body}
