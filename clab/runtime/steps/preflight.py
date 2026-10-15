"""Everything that needs the model, checked before any forward: what the run needs found, and what it
cannot run refused by name."""

import math
from typing import Any

from ... import address
from ...plan.model import (Plan, agreeing, arguments, body_view, declared, dims_of, encodes, kind_of, parameters,
                           points, probe_of, uses, walk)
from .. import data, featurizers, ops, positions
from .state import lookup, put, tokens


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
    not say (or, with no header, whose tensors are not as wide), or with a `select` outside its features; and a
    trained featurizer used at no site.
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
            later = []                      # a write's operands, checked once the step's reads have their shapes
            for side in ("writes", "reads"):
                for key, record in getattr(step, side).items():
                    where, site = f"{name}.{key}", record.site
                    if site.layers is not None:
                        picked(where, site.layers, sizes["num_layers"])
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
                    if positions.prefill(pos) or step.kind == "forward":    # a forward's one call is its prefill
                        resolved = indices(where, side == "writes", record, step, rows, encoded, tokenizer)
                        put(found["positions"], where, step.at, resolved)
                        shape = (len(resolved[0]), select)
                    else:               # resolved in the run, from what each row generates; one position a call
                        shape = (positions.width(pos) or 1, select)
                    put(shapes, where, step.at, shape)
                    later += [(where, o, shape, site) for o in record.ops] if side == "writes" else []
            for where, o, shape, site in later:
                check_operands(engine, plan, where, o, shape, step, site, found, shapes)
    return found


def made_at(site, n: int) -> list:
    """The layers a site is made at among the model's `n`: those its `layers` select, or `[None]` for a
    model's own value, which has none."""
    return [None] if site.layers is None else positions.select(site.layers, n)


def picked(where: str, layers, n: int) -> list[int]:
    """The layers a site's `layers` select among the model's `n`, each one there: an int or a pick outside
    them, or a slice that picks none, is refused by name."""
    for i in [layers] if isinstance(layers, int) else layers if isinstance(layers, list) else []:
        if not -n <= i < n:
            raise ValueError(f"{where}: layer {i}; the model has {n} layers")
    try:
        got = positions.select(layers, n)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    if not got:
        raise ValueError(f"{where}: layers {layers!r} pick none of the model's {n} layers")
    return got


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
            layer = positions.select(site.layers, sizes["num_layers"]) if isinstance(site.layers, int) else None
            if f.header is not None and layer and f.header.get("layer") != str(layer[0]):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'layer' is "
                                 f"{f.header.get('layer')!r}; this document uses {name!r} at {site.component} "
                                 f"layer {site.layers}, which is layer {layer[0]}")
            if f.header is not None and f.header.get("width") != str(width):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'width' is "
                                 f"{f.header.get('width')!r}; this document uses {name!r} at {site.component}, "
                                 f"whose width is {width}")
            if f.header is None and f.bundle is not None and featurizers.KINDS[f.kind].width(f.bundle) != width:
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']} has no clab header and is "
                                 f"{featurizers.KINDS[f.kind].width(f.bundle)} wide; this document uses {name!r} at "
                                 f"{site.component}, whose width is {width}")
            features: Any = (o.select or {}).get("features")
            if features is not None:
                n = instance(plan, f, width, None).size()
                if any(not 0 <= i < n for i in features):
                    raise ValueError(f"{where}: select features {features} outside the {n} features of {name!r}")
    for name, step in walk(plan.steps):
        for f in step.train if step.kind == "fit" else []:
            if f not in out:
                raise ValueError(f"{name}: {f!r} is trained but used at no site, so its width is unknown")
    return out


def instance(plan: Plan, spec, width: int, seed: int | None) -> featurizers.Kind:
    """A featurizer point `spec` made: from its bundle, or drawn from its seed, else `seed`; a probe's class
    list from its `classes` column."""
    fields = spec.model_extra or {}
    labels = featurizers.labels({n: d.rows or [] for n, d in plan.data.items()}, fields["classes"]) \
        if "classes" in fields else None
    return featurizers.make(spec.kind, fields, width, spec.bundle, seed, labels, spec.header)


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


def indices(where: str, write: bool, record, step, rows: list[dict], encoded: list[dict],
            tokenizer) -> list[list[int]]:
    """A prefill record's positions in each row, which the data alone decides: a scope text that is absent or
    doubled, or a rectangular index that does not fit, is refused by name, and so is a write whose positions
    hold no token of a row."""
    pos = positions.parse(record.pos)
    found = []
    for i, (row, enc) in enumerate(zip(rows, encoded)):
        try:
            got = place(pos, row, enc, [], tokenizer, head(step, record))
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


def place(pos: dict, row: dict, encoded: dict, said: list[int], tokenizer,
          head_only: bool) -> list[int] | str:
    """`positions.resolve` for one row (its dataset row and its encoding), with what that needs: the positions
    its prefill holds and where the scope's text occurs."""
    found = positions.occurrences(pos["scope"], encoded, said, tokenizer, texts(pos, row))
    return positions.resolve(pos, len(encoded["ids"]), prompted(encoded, tokenizer, head_only), said, found)


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
                named = picked(f"{name}.{key}", layers, engine.sizes()["num_layers"])
                reason = next((r for layer, r in reason.items() if r and layer in named), None)
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


def check_operands(engine, plan: Plan, where: str, o, shape: tuple, step, site, found: dict, shapes: dict) -> None:
    """A write's operands that another record selected: the same width and narrowing; on a `key` axis, inputs
    that line up token for token. One from a read of the write's own forward is read at a site this engine
    reaches first. An argument of the function an op's `options` choose (a lora's `with`) is as wide per
    position, its axes after the positions flattened, as the value the op acts on, and, from the same forward,
    read at the write's own positions."""
    args, sizes = arguments(o), engine.sizes()
    chosen = set(parameters(plan, o.op, args)) - set(ops.parameters(o.op))
    for key, p in parameters(plan, o.op, args).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        read = points(plan.steps[value.partition(".")[0]])[0].reads[value.partition(".")[2]].site
        same = value.partition(".")[0] == where.partition(".")[0]
        if same and engine.rank(read.component, made_at(read, sizes["num_layers"])[0]) >= \
                engine.rank(site.component, made_at(site, sizes["num_layers"])[0]):
            raise ValueError(f"{where}: operand {value!r} is read at {read.component} layer {read.layers}, which the "
                             f"{engine.name} engine's forward does not reach before this write's {site.component} "
                             f"layer {site.layers}; an operand from the write's own forward is read at an earlier site")
        if key in chosen:
            per, own = (math.prod(len((s.select or {}).get(a, [])) or sizes[a]
                                  for a in address.dims(s.component, None, False)[2:]) for s in (read, site))
            if per != own:
                raise ValueError(f"{where}: operand {value!r} is {per} wide per position, its axes after the "
                                 f"positions flattened; this write's value is {own}")
            ours, at = (lookup(found["positions"], n, step.at) for n in (where, value)) if same else ([], [])
            for row, (a, b) in enumerate(zip(at, ours)):
                if a != b:
                    raise ValueError(f"{where}: operand {value!r} is read at positions {a} of row {row} and this "
                                     f"write lands at {b}; {key!r} from the write's own forward is read where it "
                                     "lands")
        if theirs[0] != shape[0] or (theirs[1] != shape[1] and key not in chosen):
            raise ValueError(f"{where}: operand {value!r} is {theirs[0]} token(s) wide, narrowed to {theirs[1]}; "
                             f"this write's selection is {shape[0]} wide, narrowed to {shape[1]}")
        producer = points(plan.steps[value.partition(".")[0]])[0].input
        if "key" in address.COMPONENTS[site.component]["dims"] and producer != step.input:
            for row, (a, b) in enumerate(zip(found["inputs"][producer]["rows"], found["inputs"][step.input]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {step.input} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
