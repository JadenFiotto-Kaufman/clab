"""Walk a compiled plan inside the run: preflight, every step and point in order, then the saved values into
`plan.results`.

This assumes only the engine protocol (`engines.Engine`): it asks the engine for forwards and generations,
its tokenizer, its sizes, its readout and the ids its generation stops on (EOS), and does everything else
itself, in plain torch: positions in the prefill from the encoded rows, and beyond it from the ids each row
generated, up to its first EOS. Under the nnterp engine it runs inside
one session; under the hooks engine, in plain Python. The cross-step state is a dict that lives in `run`'s
frame and never leaves the run: each value under its name, by the coordinates of the point that produced it.
"""

import importlib.metadata
import platform
import time
from typing import Any, cast

import torch
from torch import Tensor

from . import address, data, featurizers, ops, positions
from .plan import (Plan, Run, agreeing, arguments, body_view, coordinate, declared, dims_of, encodes, kind_of,
                   labelled, parameters, points, probe_of, space, uses, walk)
from .registry import registers

KINDS: dict = {}
kind = registers(KINDS)


def begin(engine, plan: Plan, batch_size: int, remote: bool | str) -> float:
    """What every engine's `execute` checks first, and the run record it starts; returns the start time."""
    if not plan.compiled:
        raise ValueError("plan is not compiled; call plan.compile()")
    if plan.model != engine.spec:
        raise ValueError(f"this engine loaded {engine.spec.key}@{engine.spec.revision}; the plan pins "
                         f"{plan.model.key}@{plan.model.revision}")
    plan.run = Run(engine=engine.name, remote=remote, batch_size=batch_size,
                   started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return time.time()


def run(engine, plan: Plan, batch_size: int) -> None:
    """Every step and point in document order, without gradients except where a fit trains; then each saved
    value into `plan.results` (a trained featurizer as its bundle's tensors)."""
    assert plan.run is not None
    plan.run.versions = versions()
    found = preflight(engine, plan)
    state = made(plan, found["widths"])
    with torch.no_grad():
        for name, entry in plan.steps.items():
            for step in points(entry):
                KINDS[step.kind](engine, plan, name, step, state, batch_size, found)
    for name in plan.saves:
        entry = plan.steps[name.partition(".")[0]]
        plan.results[name] = ({coordinate(plan, p.at): finished(saved(plan, state, name, p.at)) for p in entry}
                              if isinstance(entry, list) else finished(saved(plan, state, name, {})))
    positions = dict(found["positions"])
    for fit, by_point in found["bodies"].items():        # a body's records, as each eval run resolved them
        for sources in by_point.values():
            for label, (_, body) in sources["evals"].items():
                for where, resolved in body["positions"].items():
                    key = f"{fit}.{label}.{where}"
                    positions[key] = {**positions.get(key, {}), **resolved}
    plan.run.positions = {where: by_coordinate(plan, by_point) for where, by_point in positions.items()}
    plan.run.tokens.update({name: by_coordinate(plan, by_point) for name, by_point in found["said"].items()})


def saved(plan: Plan, state: dict, name: str, at: dict) -> Any:
    thing, _, member = name.partition(".")
    step = points(plan.steps[thing])[0]
    value = lookup(state, name, at)
    return featurizers.tensors(value) if step.kind == "fit" and member in step.train else value


def by_coordinate(plan: Plan, by_point: dict) -> Any:
    """A value recorded per point as the run record holds it: the one value of an unswept step, else keyed by
    the point's coordinate as text."""
    return by_point[()] if () in by_point else {labelled(plan, dict(c)): v for c, v in by_point.items()}


def preflight(engine, plan: Plan) -> dict:
    """Everything that needs the model, checked before any forward. Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs: each dataset input's encoded rows and padded width (`inputs`),
    every prefill record's positions per row (`positions`, which the run extends with the other records'),
    room for what each generate step's rows said (`said`), each featurizer's width (`widths`), and for each
    fit point the same for its body (`bodies`): over its training rows (`train`) and under each of its named
    `eval` substitutions (`evals`), each beside the plan the body runs in, which holds the body's steps and
    each substituted dataset's rows under the body's name for it.

    Refuses by name (`<step>.<record>`): a component the engine does not serve; a layer or a `select`
    outside the model's sizes (on `key`, outside the input's shortest row); a generate option its
    `generate` does not take, or one in `FIXED` at another value; a position that does not fit some row's
    scope, or whose scope is not in a row a write or a rectangular read needs; a token column that is not
    one token on some row; a write whose operand was selected at a different width or narrowing; a write to
    a component with a `key` axis (the attention pattern, the scores) whose operand was read on another
    input that does not tokenize to the same length on every row, which would misalign the keys; a `pca`
    over no more vectors than its `k`; a featurizer used at two widths, at a width its bundle's header does
    not say, or with a `select` outside its features; and a trained featurizer used at no site.
    """
    refuse(engine, plan)
    rows = {n: d.rows or [] for n, d in plan.data.items()}
    classes = {name: featurizers.labels(rows, (spec.model_extra or {})["classes"])
               for name, entry in plan.featurizers.items() for spec in points(entry)[:1] if spec.kind == "linear"}
    found = prepare(engine, plan, plan.steps, {}, classes)
    found["widths"] = widths(engine, plan)
    found["bodies"] = {}
    for name, entry in plan.steps.items():
        for step in points(entry):
            if step.kind != "fit":
                continue
            within = body_view(plan, name, step.at)
            sources: dict = {"train": (within, prepare(engine, within, step.steps, {}, classes)), "evals": {}}
            for label, swap in step.eval.items():
                held = body_view(plan, name, step.at, swap)
                sources["evals"][label] = (held, prepare(engine, held, step.steps, swap, classes)) if swap \
                    else sources["train"]
            for _, body in [sources["train"], *sources["evals"].values()]:
                body["widths"] = found["widths"]
            put(found["bodies"], name, step.at, sources)
    return found


def prepare(engine, plan: Plan, steps: dict, swap: dict, classes: dict) -> dict:
    """`preflight`'s checks and findings for `steps` (a fit is prepared apart), over the rows `plan` holds:
    the root's, or a body's. `swap` names each dataset as the run record knows it (an eval substitution's
    `{"train": "test"}` records `train.input` as `test.input`); `classes` is each probe's class list."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    found: dict = {"inputs": {}, "positions": {}, "said": {}}
    shapes: dict = {}
    for name, entry in steps.items():
        for step in points(entry):
            if step.kind == "fit":
                continue
            if step.kind == "op":
                if "classes" in dims_of(plan, step.of):
                    check_classes(plan, name, step.op, arguments(step), probe_of(plan, step.of), classes, swap)
                else:
                    check_tokens(engine, plan, name, step.op, arguments(step))
                if step.op == "pca" and step.of in found["positions"]:
                    n = sum(map(len, lookup(found["positions"], step.of, step.at)))
                    if n <= arguments(step)["k"]:
                        raise ValueError(f"{name}: pca of {n} vectors has no rank-{arguments(step)['k']} basis; "
                                         f"it needs more than {arguments(step)['k']}")
                continue
            if step.kind == "generate":
                generation_options(name, step.model_extra or {})
            rows = plan.data[step.dataset].rows or []
            if step.input not in found["inputs"]:
                encoded = []
                for i, r in enumerate(rows):
                    try:
                        encoded.append(data.encode(tokenizer, data.field(r, step.input)))
                    except ValueError as e:
                        raise ValueError(f"{name}: {step.input} row {i}: {e}") from None
                found["inputs"][step.input] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
                dataset, _, rest = step.input.partition(".")
                plan.run.tokens[f"{swap.get(dataset, dataset)}.{rest}"] = \
                    [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
            encoded = found["inputs"][step.input]["rows"]
            for side in ("writes", "reads"):
                for key, record in getattr(step, side).items():
                    where, site = f"{name}.{key}", record.site
                    if isinstance(site.layers, int) and not 0 <= site.layers < sizes["num_layers"]:
                        raise ValueError(f"{where}: layer {site.layers}; the model has {sizes['num_layers']} layers")
                    for axis, index in (site.select or {}).items():
                        shortest = min(len(e["ids"]) for e in encoded)
                        size, what = (shortest, f"the {shortest} tokens of the input's shortest row") \
                            if axis == "key" else (sizes[axis], f"the model's {sizes[axis]}")
                        if any(not 0 <= i < size for i in index):
                            raise ValueError(f"{where}: select {axis} {index} outside {what}")
                    probe = None                        # past a probe's encode, a column names its classes
                    for o in record.ops:
                        if probe is None:
                            check_tokens(engine, plan, where, o.op, arguments(o))
                        else:
                            check_classes(plan, where, o.op, arguments(o), probe, classes, swap)
                        if encodes(o) and kind_of(plan, o.featurizer or "") == "linear":
                            probe = o.featurizer
                    pos, select = positions.parse(record.pos), {a: len(v) for a, v in (site.select or {}).items()}
                    if positions.prefill(pos):
                        resolved = indices(where, side == "writes", record, step, rows, encoded, tokenizer)
                        put(found["positions"], where, step.at, resolved)
                        shape = (len(resolved[0]), select)
                    else:               # resolved in the run, from what each row generates; one position a call
                        shape = (positions.width(pos) or 1, select)
                    put(shapes, where, step.at, shape)
                    for o in record.ops if side == "writes" else []:
                        check_operands(plan, where, o, shape, step, site.component, found["inputs"], shapes)
    return found


def widths(engine, plan: Plan) -> dict[str, int]:
    """Each featurizer's width: the size of the last axis of every site it is used at, which is one size, and
    the width its bundle's header says. A `select` on its features stays within them, and a featurizer a fit
    trains is used somewhere."""
    sizes, out = engine.sizes(), {}
    for name, used in uses(plan).items():
        for where, site, at, o in used:
            last = address.dims(site.component, None, False)[-1]
            width = len((site.select or {})[last]) if last in (site.select or {}) else sizes.get(last)
            if width is None:
                raise ValueError(f"{where}: {site.component}'s last axis is {last}, which grows with the input; "
                                 f"featurizer {name!r} needs a feature axis")
            if out.setdefault(name, width) != width:
                raise ValueError(f"{where}: featurizer {name!r} is used at width {width} here and {out[name]} "
                                 "elsewhere; a featurizer has one width")
            f = agreeing(plan.featurizers[name], at)
            if f.header is not None and f.header.get("width") != str(width):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'width' is "
                                 f"{f.header.get('width')!r}; this document uses {name!r} at {site.component}, "
                                 f"whose width is {width}")
            features: Any = (o.select or {}).get("features")
            if features is not None:
                n = featurizers.size(instance(plan, f, width, None))
                if any(not 0 <= i < n for i in features):
                    raise ValueError(f"{where}: select features {features} outside the {n} features of {name!r}")
    for name, step in walk(plan.steps):
        for f in step.train if step.kind == "fit" else []:
            if f not in out:
                raise ValueError(f"{name}: {f!r} is trained but used at no site, so its width is unknown")
    return out


def instance(plan: Plan, spec, width: int, seed: int | None) -> dict:
    """A featurizer point `spec` made: from its bundle, or drawn from its seed, else `seed`; a probe's class
    list from its `classes` column."""
    fields = spec.model_extra or {}
    labels = featurizers.labels({n: d.rows or [] for n, d in plan.data.items()}, fields["classes"]) \
        if "classes" in fields else None
    return featurizers.make(spec.kind, fields, width, spec.bundle, seed, labels)


def made(plan: Plan, widths: dict) -> dict:
    """The run's state at its start: each declared featurizer used at a site, at each of its points, made
    where the model is."""
    state: dict = {}
    for name, entry in plan.featurizers.items():
        for spec in points(entry) if name in widths else []:
            put(state, name, spec.at, instance(plan, spec, widths[name], None))
    return state


# The generate options a step cannot forward as they are, the one value each may have, and why.
FIXED = {
    "num_return_sequences": (1, "more sequences than rows would break the step's one row of ids per input row"),
    "num_beams": (1, "beams change the rows each forward call sees, so a read's row is no longer the input's"),
    "return_dict_in_generate": (False, "the step's value is the ids"),
    "use_cache": (True, "without the cache a decode step recomputes the prompt, and the prompt-call writes "
                        "would not reach it"),
}


def generation_options(name: str, options: dict) -> None:
    """Refuses an option in `FIXED` at another value. Every other option is the engine's `generate`'s to
    accept or reject, when the step runs (`generated`)."""
    for key, value in options.items():
        if key in FIXED and value != FIXED[key][0]:
            raise ValueError(f"{name}: {key} {value!r}; it stays {FIXED[key][0]!r}: {FIXED[key][1]}")


def stops(engine, step) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def indices(where: str, write: bool, record, step, rows: list[dict], encoded: list[dict],
            tokenizer) -> list[list[int]]:
    """A prefill record's positions in each row, which the data alone decides: a scope text that is absent or
    doubled, or a rectangular index that does not fit, is refused by name, and so is a write whose positions
    hold no token of a row."""
    pos = positions.parse(record.pos)
    found = []
    for i, (row, enc) in enumerate(zip(rows, encoded)):
        try:
            got = place(pos, row, enc, [], False, tokenizer, head(step, record))
        except ValueError as e:
            why = f"; a generation's prefill computes {record.site.component} at its last position only, so " \
                "that is all call 0 holds of it" if head(step, record) else ""
            raise ValueError(f"{where}: row {i}: {e}{why}") from None
        if write and not got:
            raise ValueError(f"{where}: row {i}: the write's positions hold no token of this row; a write needs "
                             "one to write")
        found.append(list(got))
    return found


def prompted(encoded: dict, tokenizer, head_only: bool) -> list[int]:
    """The positions a row's prefill holds: its prompt tokens that are not special, or its last token alone
    (`head_only`) where the prefill computes only that."""
    ids, special = encoded["ids"], set(tokenizer.all_special_ids)
    return [len(ids) - 1] if head_only else [i for i, t in enumerate(ids) if t not in special]


def place(pos: dict, row: dict, encoded: dict, said: list[int], stopped: bool, tokenizer,
          head_only: bool) -> list[int] | str:
    """`positions.resolve` for one row (its dataset row and its encoding), with what that needs: the positions
    its prefill holds and where the scope's text occurs."""
    found = positions.occurrences(pos["scope"], encoded, said, tokenizer, texts(pos, row))
    return positions.resolve(pos, len(encoded["ids"]), prompted(encoded, tokenizer, head_only), said, stopped, found)


def texts(pos: dict, row: dict) -> list[str]:
    """The texts a record's scope names in one row: its columns' (each `<dataset>.<column path>`), or the
    literals; a single one is the one-element case."""
    scope = pos["scope"] or {}
    if "column" in scope:
        columns = scope["column"] if isinstance(scope["column"], list) else [scope["column"]]
        return [str(data.lookup(row, c.partition(".")[2])) for c in columns]
    given = scope.get("text")
    return [] if given is None else given if isinstance(given, list) else [given]


def head(step, record) -> bool:
    """Whether the record reads the head in a generation, whose prefill computes it at its last position only."""
    return step.kind == "generate" and record.site.component in ("lm_head_output", "logits")


def refuse(engine, plan: Plan) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name. `engine.unserved()`
    maps a component to None (served), why not, or why not per layer (nnterp's `support()` shape)."""
    reasons = engine.unserved()
    for name, step in walk(plan.steps):
        for key, record in {**getattr(step, "writes", {}), **getattr(step, "reads", {})}.items():
            reason, layers = reasons.get(record.site.component), record.site.layers
            if isinstance(reason, dict):
                reason = next((r for layer, r in reason.items() if r and layers in (layer, "all")), None)
            if reason:
                raise ValueError(f"{name}.{key}: the {engine.name} engine does not serve {record.site.component}: "
                                 f"{reason}")


def check_tokens(engine, plan: Plan, where: str, name: str, args: dict) -> None:
    for key, p in parameters(plan, name, args).items():
        if p.annotation == ops.Tokens and key in args:
            try:
                tokens(engine, plan, args[key], args.get("token_form", "spaced"), None)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None


def check_classes(plan: Plan, where: str, name: str, args: dict, probe: str, classes: dict, swap: dict) -> None:
    """Every column op `name` scores against a probe's class list holds only those classes, on every row of the
    dataset it reads (named as the run reads it: under eval, the substituted one)."""
    vocabulary = classes[declared(plan, probe)]
    for key, p in parameters(plan, name, args).items():
        if p.annotation == ops.Tokens and key in args:
            dataset, _, column = args[key].partition(".")
            for i, row in enumerate(plan.data[dataset].rows or []):
                if row[column] not in vocabulary:
                    raise ValueError(f"{where}: {key} is {column!r} of {swap.get(dataset, dataset)!r}, whose row {i} "
                                     f"is {row[column]!r}, not one of the classes of probe {declared(plan, probe)!r}: "
                                     f"{vocabulary}")


def check_operands(plan: Plan, where: str, o, shape: tuple, step, component: str, inputs: dict, shapes: dict) -> None:
    """A write's operands that another record selected: the same width and narrowing, and, on a `key` axis,
    inputs that line up token for token."""
    for key, p in parameters(plan, o.op, arguments(o)).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        if theirs != shape:
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer = points(plan.steps[value.partition(".")[0]])[0].input
        if "key" in address.COMPONENTS[component]["dims"] and producer != step.input:
            for row, (a, b) in enumerate(zip(inputs[producer]["rows"], inputs[step.input]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {step.input} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")


# -- the kinds ------------------------------------------------------------------------------------------


@kind
def forward(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.forward`; each read joined along rows.

    A `layers: "all"` read is one read per layer, its ops run on each, stacked layer-first; a ragged read is
    gathered at the batch's widest row and kept as `{values, row, tok}`: one entry per token, its row and its
    token index."""
    parts = []
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        sites, writes, reads = bind(engine, plan, name, step, step.reads, state, found, chunk, pads)
        got = engine.forward(batch, writes, reads)
        parts.append({key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk,
                                                 pads, state, step.at) for layer in layers(engine, record)])
                      for key, record in step.reads.items()})
    joined(state, name, step, parts)


@kind
def generate(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: dict) -> None:
    """The step's input in consecutive batches, each through `engine.generate` with the step's own keys as
    its options; each read joined along rows, and the new ids joined into the step's value.

    A record in the prefill (`frame: 0`) is a forward's. One beyond it goes to the engine in two parts: its
    prefill positions (for a read, the whole prefill, since which of its positions it means can depend on
    what the row says), and its decode calls, each at that call's one position (`calls`: the frame's own,
    or None for every decode call). A write with a text scope is found online: it goes to every decode call
    with each row's text (`online`), lands where `positions.landing` says, and is recorded at the calls it
    says so of afterwards. Every other record resolves after the generation from what each row generated
    (`positions.resolve`); a row that stopped before a call a write selects is not written there (the call
    computes nothing of that row's) and is recorded as missing. The resolution is recorded per row in
    `plan.run.positions` and each row's continuation, as token strings, in `plan.run.tokens[<step>]`. A
    batch whose rows all stopped early makes fewer calls; its ids are filled after their EOS to the longest
    batch's."""
    eos, rows = set(stops(engine, step)), plan.data[step.dataset].rows or []
    encoded = found["inputs"][step.input]["rows"]
    first = {k: r for k, r in step.reads.items() if positions.prefill(positions.parse(r.pos))}
    later = {k: r for k, r in step.reads.items() if k not in first}
    beyond = {k: w for k, w in step.writes.items() if not positions.prefill(positions.parse(w.pos))}
    parts, new, resolved = [], [], {f"{name}.{key}": [] for key in {**later, **beyond}}
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        n = chunk.stop - chunk.start
        sites, writes, reads = bind(engine, plan, name, step, first, state, found, chunk, pads, set(beyond))
        for key, record in beyond.items():
            pos = positions.parse(record.pos)
            holds, calls, _ = positions.reach(pos)
            edits = [edit(engine, plan, record, o, state, step.at, chunk) for o in record.ops]
            if holds:
                where = [prompted(e, engine.tokenizer, head(step, record)) for e in encoded[chunk]]
                writes.append({**site(record, where, pads, head(step, record)), "name": f"{key}#0",
                               "layer": record.site.layers, "ops": edits})
            online = {"online": {"pos": pos, "texts": [texts(pos, row) for row in rows[chunk]],
                                 "eos": sorted(eos)}} if pos["scope"] else {}
            writes.append({**decoding(record, n, pads), "name": key, "layer": record.site.layers, "ops": edits,
                           "calls": None if online else calls, **online})
        for key, record in later.items():
            holds, calls, _ = positions.reach(positions.parse(record.pos))
            for layer in layers(engine, record):
                if holds:
                    whole = [list(range(len(e["ids"]))) for e in encoded[chunk]]
                    reads.append({**site(record, whole, pads, head(step, record)), "name": f"{key}@{layer}#0",
                                  "layer": layer})
                reads.append({**decoding(record, n, pads), "name": f"{key}@{layer}", "layer": layer,
                              "calls": calls})
        ids, got = generated(engine, name, batch, writes, reads, dict(step.model_extra or {}))
        for record in reads:
            if "calls" in record:
                made = range(1, ids.shape[1]) if record["calls"] is None else \
                    [k for k in record["calls"] if k < ids.shape[1]]
                if len(got[record["name"]]) != len(made):
                    raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got[record['name']])} of the "
                                       f"{len(made)} decode calls it made; a read out of forward order shifts calls")
        out = {key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk, pads,
                                          state, step.at) for layer in layers(engine, record)])
               for key, record in first.items()}
        said = [positions.continuation(x, eos) for x in ids.tolist()]
        for key, record in {**later, **beyond}.items():
            pos = positions.parse(record.pos)
            if key in beyond and pos["scope"]:
                where = [[len(e["ids"]) + k - 1 for k in range(1, len(x))
                          if positions.landing(pos, texts(pos, row), x[:k], eos, engine.tokenizer)] or positions.MISSING
                         for row, e, x in zip(rows[chunk], encoded[chunk], said)]
            else:
                where = [place(pos, row, e, x, bool(x) and x[-1] in eos, engine.tokenizer, head(step, record))
                         for row, e, x in zip(rows[chunk], encoded[chunk], said)]
            resolved[f"{name}.{key}"] += where
            if key in later:
                out[key] = stacked(record, [at_calls(engine, plan, record, got.get(f"{key}@{layer}#0"),
                                                     got[f"{key}@{layer}"], where, encoded[chunk], pads, chunk,
                                                     state, step.at, head(step, record))
                                            for layer in layers(engine, record)])
        parts.append(out)
        new.append(ids)
    joined(state, name, step, parts)
    width = max(x.shape[1] for x in new)
    ids = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=min(eos)) for x in new])
    put(state, name, step.at, ids)
    for where, per_row in resolved.items():
        put(found["positions"], where, step.at, per_row)
    put(found["said"], name, step.at,
        [engine.tokenizer.convert_ids_to_tokens(positions.continuation(x, eos)) for x in ids.tolist()])


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
    params = [p.requires_grad_() for f in trained.values() for p in featurizers.parameters(f)]
    optimizer = step.optimizer
    opt = OPTIMIZERS[optimizer.name](params, lr=optimizer.lr, weight_decay=optimizer.weight_decay)
    body_ops = [n for n, s in step.steps.items() if s.kind == "op"]
    n = len(sources["train"][0].data[next(iter(datasets(step)))].rows or [])
    scalars = [(f, sc) for f, inst in trained.items() for sc in featurizers.SCALARS.get(inst["kind"], {})]
    history: dict = {**{o: [] for o in body_ops}, **{f"{f}.{sc}": [] for f, sc in scalars}}
    best, since, kept = None, 0, None
    for epoch in range(step.epochs):
        for f in trained.values():
            featurizers.anneal(f, epoch, step.epochs)
        # One integer per (seed, epoch): epochs stay below the prime, so no two pairs share a generator.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(step.seed * 1_000_003 + epoch)).tolist()
        for lo in range(0, n, step.batch_size):
            with torch.enable_grad():
                for f in trained.values():
                    featurizers.settle(f)
            body = body_run(engine, name, step, state, trained, sources["train"], order[lo:lo + step.batch_size])
            with torch.enable_grad():
                loss = sum(weight * term(body, trained, t, step.at) for t, weight in step.objective.items())
            assert isinstance(loss, Tensor)
            for p, g in zip(params, torch.autograd.grad(loss, params)):
                p.grad = g
            opt.step()
            opt.zero_grad()
            for f in trained.values():
                featurizers.project(f)
        for f in trained.values():
            featurizers.settle(f)
        scored = body_run(engine, name, step, state, trained, sources["train"], None)
        for o in body_ops:
            value = lookup(scored, o, step.at)
            if isinstance(value, Tensor):
                history[o].append(overall(value))
        for f, sc in scalars:
            history[f"{f}.{sc}"].append(featurizers.scalar(trained[f], sc))
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
        inst.update(kept[f] if kept else {})
        featurizers.settle(inst)
    for label in step.eval:
        final = body_run(engine, name, step, state, trained, sources["evals"][label], None)
        for key, s in step.steps.items():
            members = ([key] if s.kind == "op" else []) + [f"{key}.{read}" for read in getattr(s, "reads", {})]
            for member in members:
                put(state, f"{name}.{label}.{member}", step.at, lookup(final, member, step.at))
    for f, inst in trained.items():
        inst.pop("seen", None)
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
        inst["training"] = index is not None
        inst["seen"] = []
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
        return featurizers.scalar(trained[thing], member)
    return overall(lookup(body, name, at))


def overall(value: Tensor) -> Tensor:
    """A value's mean over every axis, of the entries that have one (a missing row is NaN)."""
    return value.float().nanmean() if value.isnan().any() else value.float().mean()


def snapshot(trained: dict) -> dict:
    """Each trained featurizer as it is now, its tensors detached copies: what restoring the epoch puts back."""
    return {f: {k: v.detach().clone() if isinstance(v, Tensor) else v for k, v in inst.items() if k != "seen"}
            for f, inst in trained.items()}


# -- binding a record to one batch ------------------------------------------------------------------------


def site(record, rows: list[list[int]], pads: list[int], last_only: bool = False) -> dict:
    """What the engine needs to reach a record's selection in this batch: its component, its token indices
    into the padded batch (`positions`, `[rows, width]`; a ragged row repeats its last index to the batch's
    widest, `keep` marking its own, `tok` the unpadded indices), and the part of the feature axes it keeps
    (`select`). Where the value
    holds the last position alone (`last_only`: a generation's head in the prefill) every index is -1."""
    width = max(1, *map(len, rows))
    index = [[i + pad for i in row] + [(row[-1] + pad) if row else 0] * (width - len(row))
             for row, pad in zip(rows, pads)]
    tok = [[i - pad for i in row] for row, pad in zip(index, pads)]
    index = [[-1] * width for _ in rows] if last_only else index
    return {"component": record.site.component, "positions": torch.tensor(index), "select": selection(record, pads),
            "keep": torch.tensor([[j < len(row) for j in range(width)] for row in rows]), "tok": torch.tensor(tok)}


def decoding(record, n: int, pads: list[int]) -> dict:
    """What the engine needs to reach a record at a decode call: the call's one position, the last."""
    return {"component": record.site.component, "positions": torch.full((n, 1), -1), "select": selection(record, pads)}


def selection(record, pads: list[int]) -> dict:
    """A record's `select` as dimension indices of its selection's layout, each with the indices it keeps. On
    the `key` axis an index counts the row's own tokens, so each row's is shifted by its padding (`[rows, n]`),
    as a position is."""
    layout = address.dims(record.site.component, None, False)
    return {layout.index(axis): torch.tensor([[i + pad for i in v] for pad in pads] if axis == "key" else v)
            for axis, v in (record.site.select or {}).items()}


def edit(engine, plan: Plan, record, o, state: dict, at: dict, rows: slice) -> dict:
    """A write's op bound to one batch, as `ops.write` runs it: in a featurizer's space, with its instance and
    the features its `select` keeps."""
    dims = address.dims(record.site.component, None, False)
    within = space(o)
    return {"op": o.op, "args": bound(engine, plan, o.op, arguments(o), state, at, rows),
            "featurizer": lookup(state, within, at) if within is not None else None,
            "features": features(o), "dims": dims[:-1] + ["features"] if within is not None else dims}


def features(o) -> Tensor | None:
    """The features an op in a featurizer's space keeps, if it narrows them."""
    return torch.tensor(o.select["features"]) if o.select else None


def generated(engine, name: str, batch: dict, writes: list[dict], reads: list[dict], options: dict):
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`."""
    try:
        return engine.generate(batch, writes, reads, options)
    except (TypeError, ValueError) as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None


def batches(engine, step, found: dict, batch_size: int):
    """The step's input in consecutive batches: each one's rows (a slice), its ids padded on the left to the
    input's longest row, and each row's padding."""
    encoded = found["inputs"][step.input]
    n = len(encoded["rows"])
    for lo in range(0, n, batch_size):
        chunk = slice(lo, min(lo + batch_size, n))
        ids = [e["ids"] for e in encoded["rows"][chunk]]
        yield chunk, data.pad(engine.tokenizer, ids, encoded["width"]), [encoded["width"] - len(x) for x in ids]


def bind(engine, plan: Plan, name: str, step, reads: dict, state: dict, found: dict, rows: slice,
         pads: list[int], beyond: frozenset | set = frozenset()) -> tuple[dict, list[dict], list[dict]]:
    """A step's prefill writes (all but `beyond`) and the prefill `reads` bound to one batch: each record's
    site, the writes with their ops, and the reads, one per layer (`<read>@<layer>`)."""
    writes_here = {k: w for k, w in step.writes.items() if k not in beyond}
    sites = {key: site(record, lookup(found["positions"], f"{name}.{key}", step.at)[rows], pads, head(step, record))
             for key, record in {**writes_here, **reads}.items()}
    writes = [{**sites[key], "name": key, "layer": record.site.layers,
               "ops": [edit(engine, plan, record, o, state, step.at, rows) for o in record.ops]}
              for key, record in writes_here.items()]
    return sites, writes, [{**sites[key], "name": f"{key}@{layer}", "layer": layer}
                           for key, record in reads.items() for layer in layers(engine, record)]


def layers(engine, record) -> list:
    """The layers a read is made at: every one under `layers: "all"`, else its own."""
    return list(range(engine.sizes()["num_layers"])) if record.site.layers == "all" else [record.site.layers]


def stacked(record, values: list) -> Any:
    """A read's per-layer values as its value: the one, or under `layers: "all"` stacked layer-first."""
    if record.site.layers != "all":
        return values[0]
    if isinstance(values[0], Tensor):
        return torch.stack(values)
    return {**values[0], "values": torch.stack([v["values"] for v in values])}


def joined(state: dict, name: str, step, parts: list[dict]) -> None:
    """Each read's per-batch values joined along rows (a ragged read's entries, after the layer axis)."""
    for key, record in step.reads.items():
        axis = 1 if record.site.layers == "all" else 0
        values = [p[key] for p in parts]
        put(state, f"{name}.{key}", step.at, torch.cat(values, axis) if isinstance(values[0], Tensor) else
            {k: torch.cat([v[k] for v in values], axis if k == "values" else 0) for k in values[0]})


def read(engine, plan: Plan, record, value: Any, site: dict, rows: slice, pads: list[int], state: dict,
         at: dict) -> Any:
    """One layer of a prefill read in one batch: its entries when it is ragged, then its ops."""
    if positions.width(positions.parse(record.pos)) is None:
        keep = site["keep"]
        value = {"values": value[keep], "row": torch.arange(rows.start, rows.stop)[:, None].expand(keep.shape)[keep],
                 "tok": site["tok"][keep]}
    return with_ops(engine, plan, record, value, state, at, rows)


def at_calls(engine, plan: Plan, record, whole: Tensor | None, values: list[Tensor], where: list, encoded: list[dict],
             pads: list[int], rows: slice, state: dict, at: dict, head_only: bool) -> Any:
    """One layer of a read beyond the prefill in one batch, arranged by the positions each row resolved to
    (`where`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (`whole`, `[rows, width, ...]`, column `p` the row's token `p`; its one column under
    `head_only`), a generated input's from
    its decode call (`values`, one `[rows, 1, ...]` per call the record asked for that the batch made); then
    its ops. A rectangular read is `[rows, width, ...]`, NaN on a row it does not reach; a ragged one
    `{values, row, tok}`, `tok` the position, with no entries for such a row."""
    pos = positions.parse(record.pos)
    _, asked, _ = positions.reach(pos)
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    sizes, select = engine.sizes(), record.site.select or {}
    rest = [len(select[a]) if a in select else sizes[a] for a in address.dims(record.site.component, None, False)[2:]]
    some = whole if whole is not None else values[0] if values else None
    dtype = some.dtype if some is not None else getattr(torch, plan.model.dtype)

    def at_position(i: int, p: int) -> Tensor:
        call = positions.call_of(p, len(encoded[i]["ids"]))
        if call == 0:
            assert whole is not None
            return whole[i, 0 if head_only else p]
        return by_call[call][i, 0]

    if (width := positions.width(pos)) is not None:
        blank = torch.full((width, *rest), float("nan"), dtype=dtype)
        value = torch.stack([blank if isinstance(got, str) else torch.stack([at_position(i, p) for p in got])
                             for i, got in enumerate(where)])
    else:
        kept = [(i, p) for i, got in enumerate(where) if not isinstance(got, str) for p in got]
        value = {"values": torch.stack([at_position(i, p) for i, p in kept]) if kept else
                 torch.zeros((0, *rest), dtype=dtype),
                 "row": torch.tensor([rows.start + i for i, _ in kept], dtype=torch.long),
                 "tok": torch.tensor([p for _, p in kept], dtype=torch.long)}
    return with_ops(engine, plan, record, value, state, at, rows)


def with_ops(engine, plan: Plan, record, value: Any, state: dict, at: dict, rows: slice) -> Any:
    """A read's value with its ops applied in listed order: one in a featurizer's space on the value's
    coordinates there, the change written back. Past a probe's `encode` a column names its classes."""
    dims = address.dims(record.site.component, None, positions.width(positions.parse(record.pos)) is None)
    vocabulary = None
    for o in record.ops:
        within = space(o)
        f = lookup(state, within, at) if within is not None else None
        args = bound(engine, plan, o.op, arguments(o), state, at, rows, value, vocabulary)
        value = apply(engine, o.op, value, args, dims[:-1] + ["features"] if f else dims, featurizer=f,
                      features=features(o))
        if f is None:
            dims = ops.out_dims(o.op, dims, arguments(o), args["featurizer"]["kind"] if o.op == "featurizer" else None)
        if encodes(o):
            vocabulary = args["featurizer"].get("classes")
    return value


def bound(engine, plan: Plan, name: str, args: dict, state: dict, at: dict, rows: slice | None,
          of: Any = None, vocabulary: list | None = None) -> dict:
    """An op's arguments bound to tensors: names looked up (a featurizer to its instance), columns resolved
    against the value's own vocabulary (the tokenizer's, or a probe's class list), sliced to `rows`. On a
    ragged value `of`, an argument with a row per example is taken at each entry's row."""
    params, out = parameters(plan, name, args), {}
    for key, value in args.items():
        annotation, per_row = params[key].annotation, False
        if annotation == ops.Value and isinstance(value, str):
            given, value = value, lookup(state, value, at)
            per_row = dims_of(plan, given)[:1] == ["rows"]
            value = value[rows] if rows is not None and per_row else value
        elif annotation == ops.Tokens:
            value, per_row = (tokens(engine, plan, value, args.get("token_form", "spaced"), rows) if vocabulary is None
                              else indexed(plan, value, vocabulary, rows)), True
        elif annotation == ops.Featurizer:
            value = lookup(state, value, at)
        if per_row and isinstance(of, dict) and "tok" in of:
            value = value[of["row"] - (rows.start if rows is not None else 0)]
        out[key] = value
    return out


def apply(engine, name: str, value: Any, args: dict, dims: list[str], eos: list[int] | None = None,
          featurizer: dict | None = None, features: Tensor | None = None) -> Any:
    """Op `name` on a value, or in `featurizer`'s space on its coordinates there (the `features` it keeps),
    the change written back; on a ragged value, on its entries, which stay ragged while the op keeps them."""
    if isinstance(value, dict) and "tok" in value:
        got = apply(engine, name, value["values"], args, dims, eos, featurizer, features)
        kind = args["featurizer"]["kind"] if name == "featurizer" else None
        return {**value, "values": got} if featurizer is not None or "entries" in ops.out_dims(name, dims, args, kind) \
            else got

    def fn(x: Any) -> Any:
        return ops.call(name, x, args, engine=engine, dims=dims, eos=eos)
    return fn(value) if featurizer is None else featurizers.within(featurizer, cast(Tensor, value), fn, features)


def indexed(plan: Plan, column: str, vocabulary: list, rows: slice | None) -> Tensor:
    """A column's values as their indices in a probe's class list."""
    name, _, col = column.partition(".")
    values = [r[col] for r in plan.data[name].rows or []]       # preflight checked each is a class
    return torch.tensor([vocabulary.index(v) for v in (values[rows] if rows is not None else values)])


def tokens(engine, plan: Plan, column: str, form: str, rows: slice | None) -> Tensor:
    name, _, col = column.partition(".")
    texts = [r[col] for r in plan.data[name].rows or []]
    return data.token_ids(engine.tokenizer, texts[rows] if rows is not None else texts, form, column)


def decoded(tokenizer, ids: Tensor) -> list:
    """Each token id as the text it decodes to, nested as the ids are; None for a missing one (-1)."""
    if ids.dim() > 1:
        return [decoded(tokenizer, x) for x in ids]
    return [tokenizer.decode([i]) if i >= 0 else None for i in ids.tolist()]


# -- state ------------------------------------------------------------------------------------------------


def put(state: dict, name: str, at: dict, value) -> None:
    # A new dict each time: a fit's body namespace is a copy of the state, whose entries it must not change.
    state[name] = {**state.get(name, {}), tuple(at.items()): value}


def lookup(state: dict, name: str, at: dict):
    """The value `name` at the producer's point that agrees with `at` on the producer's axes."""
    for coords, value in state[name].items():
        if all(at.get(k) == i for k, i in coords):
            return value
    raise KeyError(f"{name} at {at}")


def finished(value):
    """Detached CPU clones all the way down; a result never holds a view of a larger buffer."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: finished(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finished(v) for v in value]
    return value


def versions() -> dict[str, str]:
    """The installed versions where the model is."""
    found = {p: importlib.metadata.version(p) for p in ("clab", "nnsight", "nnterp", "transformers", "torch")}
    return {**found, "python": platform.python_version()}
