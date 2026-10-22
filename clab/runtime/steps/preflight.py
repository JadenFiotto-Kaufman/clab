"""Everything that needs the model, checked before any forward: what the run needs found, and what it
cannot run refused by name."""

import math
from typing import Any

from ... import address
from ...plan.model import (Plan, agreeing, arguments, body_view, declared, dims_of, entries, gathers, kind_of,
                           operand_site, parameters, points, probe_of, uses, walk, writes)
from .. import data, featurizers, ops, positions
from ..ops.metrics import ONE_ID
from .state import lookup, put, refkey, tokens, unaligned


def preflight(engine, plan: Plan) -> dict:
    """Everything that needs the model, checked before any forward. Fills `plan.run.tokens` with each input's
    tokens, and returns what the run needs: each dataset input's encoded rows and padded width (`inputs`),
    every prefill record's positions per row (`positions`, which the run extends with the other records'),
    the indices every record's `select` keeps on each axis (`selects`, resolved here once, among the engine's
    sizes), room for what each generate step's rows said (`said`), each featurizer's width (`widths`), and for each
    fit point the same for its body (`bodies`): over its training rows (`train`) and under each of its named
    `eval` substitutions (`evals`), each beside the plan the body runs in, which holds the body's steps and
    each substituted dataset's rows under the body's name for it.

    Refuses by name (`<step>.<entry>`): a component the engine does not serve; a layer or a `select` outside the
    model's sizes (on `key`, outside the input's shortest row), or a `select` that keeps none of them; a generate
    option its `generate` does not take, or one in `FIXED` at another value; a position that does not fit some
    row's scope, or whose scope is not in a row a write or a rectangular read needs; a token column that is not
    one token on some row; a write whose operand was selected at a different width or narrowing; a write to a
    component with a `key` axis (the attention pattern, the scores) whose operand was read on another input that
    does not tokenize to the same length on every row, which would misalign the keys; a `pca` over no more vectors
    than its `k`; a featurizer used at two widths (at a site, the site's; inside an `encode`, that one's
    coordinates'), at a width its bundle's header does not say (or, with no header, whose tensors are not as
    wide), or encoded with a `select` outside its features or keeping none of them; and a trained featurizer
    used at no site.
    """
    assert plan.run is not None
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
            sources: dict = {"train": (within, inside(name, None, {}, lambda within=within: prepare(
                engine, within, step.steps, {}, classes))), "evals": {}}
            for label, swap in step.eval.items():
                held = body_view(plan, name, step.at, swap)
                sources["evals"][label] = (held, inside(name, label, swap, lambda held=held, swap=swap: prepare(
                    engine, held, step.steps, swap, classes))) if swap else sources["train"]
            for _, body in [sources["train"], *sources["evals"].values()]:
                body["widths"] = found["widths"]
            put(found["bodies"], name, step.at, sources)
    return found


def inside(fit: str, label: str | None, swap: dict, run) -> dict:
    """A fit body's `prepare` (`run`), a refusal naming the fit and, under an eval, which one and the datasets it
    reads in place of the body's, whose rows a row number counts."""
    try:
        return run()
    except ValueError as e:
        if label is None:
            raise ValueError(f"{fit}.{e}") from None
        read = ", ".join(f"dataset {a!r} read as {b!r}" for a, b in swap.items())
        raise ValueError(f"{fit}: eval {label!r}, with {read} (a row is the eval's dataset's): {e}") from None


def prepare(engine, plan: Plan, steps: dict, swap: dict, classes: dict) -> dict:
    """`preflight`'s checks and findings for `steps` (a fit is prepared apart), over the rows `plan` holds:
    the root's, or a body's. `swap` names each dataset as the run record knows it (an eval substitution's
    `{"train": "test"}` records `train.input` as `test.input`); `classes` is each probe's class list."""
    assert plan.run is not None
    sizes, tokenizer = engine.sizes(), engine.tokenizer
    found: dict = {"inputs": {}, "positions": {}, "said": {}, "selects": {}, "answers": {}}
    shapes: dict = {}
    for name, entry in steps.items():
        for step in points(entry):
            if step.kind == "fit":
                continue
            if step.kind == "op":
                if "classes" in dims_of(plan, step.of):
                    check_classes(plan, name, step.op, arguments(step), probe_of(plan, step.of), classes, swap)
                else:
                    check_tokens(engine, plan, found, name, step.op, arguments(step), dims_of(plan, step.of),
                                 *placed(plan, found, step.of, step.at))
                if step.op == "pca" and step.of in found["positions"]:
                    n = sum(map(len, lookup(found["positions"], step.of, step.at)))
                    if n <= arguments(step)["k"]:
                        raise ValueError(f"{name}: pca of {n} vectors has no rank-{arguments(step)['k']} basis; "
                                         f"it needs more than {arguments(step)['k']}")
                continue
            if step.kind == "generate":
                generation_options(name, step.model_extra or {})
            rows = plan.data[step.dataset].rows or []
            source = data.input_name(step.input)
            if source not in found["inputs"]:
                encoded = []
                for i, r in enumerate(rows):
                    try:
                        encoded.append(data.encode(tokenizer, data.field(r, step.input)))
                    except ValueError as e:
                        raise ValueError(f"{name}: {source} row {i}: {e}") from None
                found["inputs"][source] = {"rows": encoded, "width": max(len(e["ids"]) for e in encoded)}
                plan.run.tokens[data.input_name(step.input, swap)] = \
                    [tokenizer.convert_ids_to_tokens(e["ids"]) for e in encoded]
            encoded = found["inputs"][source]["rows"]
            later = []                      # a write's operands, checked once the step's entries have their shapes
            for key, record in entries(step).items():
                where, site, write = f"{name}.{key}", record.site, writes(record)
                if site.layers is not None:
                    picked(where, site.layers, sizes["num_layers"])
                for axis, index in (site.select or {}).items():
                    shortest = min(len(e["ids"]) for e in encoded)
                    size, what = (shortest, f"the {shortest} tokens of the input's shortest row") \
                        if axis == "key" else (sizes[axis], f"the model's {sizes[axis]}")
                    selected(where, axis, index, size, what)
                kept = positions.kept(site.select, sizes)
                put(found["selects"], where, step.at, kept)
                pos, select = positions.parse(record.pos), {a: len(v) for a, v in kept.items()}
                resolved = None
                if positions.prefill(pos) or step.kind == "forward":    # a forward's one call is its prefill
                    resolved = indices(where, write, record, step, rows, encoded, tokenizer)
                    put(found["positions"], where, step.at, resolved)
                    shape = (len(resolved[0]), select)
                else:               # resolved in the run, from what each row generates; one position a call
                    shape = (positions.width(pos) or 1, select)
                probe = None                        # past a probe's encode, a column names its classes
                for o in record.ops:
                    if probe is None:
                        forced = [e["ids"] for e in encoded] if (pos["scope"] or {}).get("before") else None
                        check_tokens(engine, plan, found, where, o.op, arguments(o), None, resolved, forced)
                    else:
                        check_classes(plan, where, o.op, arguments(o), probe, classes, swap)
                    if o.op == "encode" and kind_of(plan, arguments(o)["featurizer"]) == "linear":
                        probe = arguments(o)["featurizer"]
                put(shapes, where, step.at, shape)
                later += [(where, o, shape, site, kept) for o in record.ops[:len(record.ops) - len(gathers(record))]]
            for where, o, shape, site, kept in later:
                check_operands(engine, plan, where, o, shape, step, site, found, shapes)
                check_labels(plan, where, o, step, site, kept, sizes)
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


def selected(where: str, axis: str, selector, n: int, what: str) -> None:
    """Refuses by name a `select` on an axis of `n` (`what` says which) with a pick outside it, or that keeps
    none of it."""
    written = [selector] if isinstance(selector, int) else selector if isinstance(selector, list) else []
    try:
        got = positions.kept({axis: selector}, {axis: n})[axis]
    except ValueError as e:
        raise ValueError(f"{where}: select {axis}: {e}") from None
    if any(not -n <= i < n for i in written) or any(i >= n for i in got):
        raise ValueError(f"{where}: select {axis} {selector!r} outside {what}")
    if not got:
        raise ValueError(f"{where}: select {axis} {selector!r} keeps none of {what}")


def widths(engine, plan: Plan) -> dict[str, int]:
    """Each featurizer's width: the size of the last axis of every site it is used at, which is one size, and
    the width its bundle's header says. A `select` on its features stays within them, and a featurizer a fit
    trains is used somewhere."""
    sizes, out = engine.sizes(), {}
    for name, used in uses(plan).items():
        for where, site, at, o, around in used:
            last = address.dims(site.component, None, False)[-1]
            kept = positions.kept(site.select, sizes)
            width = len(kept[last]) if last in kept else sizes.get(last)
            if width is None:
                raise ValueError(f"{where}: {site.component}'s last axis is {last}, which grows with the input; "
                                 f"featurizer {name!r} needs a feature axis")
            for outer in around:                # the coordinates of each encode it is inside, outermost first
                ref = declared(plan, arguments(outer)["featurizer"])
                width = instance(plan, agreeing(plan.featurizers[ref], at), width, None).size()
                chosen = (arguments(outer).get("select") or {}).get("features")
                width = width if chosen is None else len(positions.select(chosen, width))
            if out.setdefault(name, width) != width:
                raise ValueError(f"{where}: featurizer {name!r} is used at width {width} here and {out[name]} "
                                 "elsewhere; a featurizer has one width")
            f = agreeing(plan.featurizers[name], at)
            for field in featurizers.KINDS[f.kind].counts:
                count = (f.model_extra or {}).get(field)
                if isinstance(count, int) and count > width:
                    raise ValueError(f"{where}: featurizer {name!r} has {field} {count}, and it is used at width "
                                     f"{width} ({site.component}), which has no more than {width} directions; "
                                     f"{field} is at most {width}")
            layer = positions.select(site.layers, sizes["num_layers"]) if isinstance(site.layers, int) else None
            stated = (f.model_extra or {}).get("layer")
            if isinstance(stated, int):
                made = positions.select(site.layers, sizes["num_layers"]) if site.layers is not None else []
                if made != positions.select(stated, sizes["num_layers"]):
                    raise ValueError(f"{where}: featurizer {name!r} says it was trained at layer {stated}; this "
                                     f"document uses it at {site.component} layers {site.layers!r}, which is layer "
                                     f"{made[0] if len(made) == 1 else made}: a dictionary is its layer's alone")
            if f.header is not None and layer and f.header.get("layer") != str(layer[0]):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'layer' is "
                                 f"{f.header.get('layer')!r}; this document uses {name!r} at {site.component} "
                                 f"layer {site.layers}, which is layer {layer[0]}")
            if f.header is not None:
                read_with(plan, where, name, f, o, sizes["num_layers"])
            if f.header is not None and f.header.get("width") != str(width):
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']}: header 'width' is "
                                 f"{f.header.get('width')!r}; this document uses {name!r} at {site.component}, "
                                 f"whose width is {width}")
            if f.header is None and f.bundle is not None and featurizers.KINDS[f.kind].width(f.bundle) != width:
                raise ValueError(f"{where}: {(f.model_extra or {})['file_path']} has no clab header and is "
                                 f"{featurizers.KINDS[f.kind].width(f.bundle)} wide; this document uses {name!r} at "
                                 f"{site.component}, whose width is {width}")
            features: Any = (arguments(o).get("select") or {}).get("features")
            if features is not None:
                n = instance(plan, f, width, None).size()
                selected(where, "features", features, n, f"the {n} features of {name!r}")
    for name, step in walk(plan.steps):
        for f in step.train if step.kind == "fit" else []:
            if f not in out:
                raise ValueError(f"{name}: {f!r} is trained but used at no site, so its width is unknown")
    return out


def read_with(plan: Plan, where: str, name: str, f, o, n: int) -> None:
    """A bundle made reading its input at another site (a cross-site adapter's `with`, which its header records as
    `with.component` and `with.layer`) is used reading it there, and one made without reads none: the width check
    alone would pass an o_proj adapter fed its own output, whose width can equal the premix's."""
    given, recorded = arguments(o).get("with"), f.header.get("with.component")
    file = (f.model_extra or {})["file_path"]
    if recorded is None and given is not None:
        raise ValueError(f"{where}: {file} was trained on the value at its own site, and this use gives with "
                         f"{given!r}; apply it without 'with'")
    if recorded is None:
        return
    layer = f.header.get("with.layer")
    if given is None:
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {layer} (with), and this "
                         f"use gives no 'with', so it would read {name!r}'s own site's value; give 'with' an entry of "
                         f"this forward at {recorded} layer {layer}")
    site = operand_site(plan, where, given)
    at = str(positions.select(site.layers, n)[0]) if isinstance(site.layers, int) else str(site.layers)
    if (site.component, at) != (recorded, layer):
        raise ValueError(f"{where}: {file} was trained reading its input at {recorded} layer {layer} (with), and this "
                         f"use's with {given!r} is {site.component} layer {at}; give 'with' an entry at "
                         f"{recorded} layer {layer}")


def instance(plan: Plan, spec, width: int, seed: int | None, device: str = "cpu") -> featurizers.Kind:
    """A featurizer point `spec` made on `device`: from its bundle, or drawn from its seed, else `seed`; a probe's
    class list from its `classes` column."""
    fields = spec.model_extra or {}
    labels = featurizers.labels({n: d.rows or [] for n, d in plan.data.items()}, fields["classes"]) \
        if "classes" in fields else None
    return featurizers.make(spec.kind, fields, width, spec.bundle, seed, labels, spec.header, device)


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
    got = positions.resolve(pos, len(encoded["ids"]), prompted(encoded, tokenizer, head_only), said, found)
    if head_only and (pos["scope"] or {}).get("before") and isinstance(got, list) and \
            any(p < len(encoded["ids"]) - 1 for p in got):
        return positions.MISSING            # before a prompt token, where this prefill computes no head
    return got


def texts(pos: dict, row: dict) -> list[tuple[str, bool, bool]]:
    """The texts a record's scope names in one row, each `(text, strip, lower)`: its columns' (each member of
    a set), compared as each reference says, or the literals, as the scope says; a single one is the
    one-element case."""
    scope = pos["scope"] or {}
    if "column" in scope:
        out = []
        for ref in scope["column"] if isinstance(scope["column"], list) else [scope["column"]]:
            column, strip, lower = data.column_of(ref) or ("", False, False)
            value = data.lookup(row, column.partition(".")[2])
            out += [(str(v), strip, lower) for v in (value if isinstance(value, list) else [value])]
        return out
    given = scope.get("text")
    given = [] if given is None else given if isinstance(given, list) else [given]
    return [(t, bool(scope.get("strip")), bool(scope.get("lower"))) for t in given]


def head(step, record) -> bool:
    """Whether the record reads the head in a generation, whose prefill computes it at its last position only."""
    return step.kind == "generate" and record.site.component in ("lm_head_output", "logits")


def refuse(engine, plan: Plan) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name. `engine.unserved()`
    maps a component to None (served), why not, or why not per layer (nnterp's `support()` shape)."""
    reasons = engine.unserved()
    for name, step in walk(plan.steps):
        for key, record in entries(step).items():
            reason, layers = reasons.get(record.site.component), record.site.layers
            if isinstance(reason, dict):
                named = picked(f"{name}.{key}", layers, engine.sizes()["num_layers"])
                reason = next((r for layer, r in reason.items() if r and layer in named), None)
            if reason:
                raise ValueError(f"{name}.{key}: the {engine.name} engine does not serve {record.site.component}: "
                                 f"{reason}")


def check_tokens(engine, plan: Plan, found: dict, where: str, name: str, args: dict, dims: list[str] | None,
                 placed: list | None, forced: list | None = None) -> None:
    """Every column op `name` scores against logits, resolved once into `found["answers"]` (`answer`): a class or
    a set of several ids is refused where the op reads one id's logit. A column with an answer of several
    tokens lines every row up with the read, one position per token: the value has a pos axis (when its axes
    `dims` are known) and, where the read's positions are known before the run (`placed`, per row), each row
    has as many. Where the read is a scope that is `before` on the step's own input (`forced`: each row's
    input ids), the input's next token at each position is among the ids the answer's token there may be."""
    for key, p in parameters(plan, name, args).items():
        if p.annotation != ops.Tokens or key not in args:
            continue
        got = answer(engine, plan, found, args[key], where)
        values, sets, column, aligned = got["values"], got["sets"], got["column"], got["aligned"]
        for r, (value, row) in enumerate(zip(values, sets)):
            if name in ONE_ID and any(len(ids) > 1 for ids in row):
                raise ValueError(f"{where}: {column} row {r}: {value!r} stands for {max(map(len, row))} ids (a set, "
                                 f"or a class under strip or lower), which have no one logit for {name}; "
                                 "token_prob scores them together")
            if aligned and dims is not None and not {"pos", "entries"} & set(dims):
                raise ValueError(f"{where}: {column} row {r}: {value!r} is {len(row)} token"
                                 f"{'s' * (len(row) != 1)}, a position each, and the value has no pos axis ({dims})")
            if aligned and placed is not None and isinstance(placed[r], list) and len(placed[r]) != len(row):
                raise ValueError(f"{where}: {unaligned(column, r, value, len(row), len(placed[r]))}")
        if forced is None or placed is None:
            continue
        for r, at in enumerate(placed):
            for i, p in enumerate(at if isinstance(at, list) else []):
                ids, next_ = sets[r][i if aligned else 0], forced[r][p + 1]
                if next_ not in ids:
                    shown = engine.tokenizer.convert_ids_to_tokens([next_])[0]
                    raise ValueError(f"{where}: {column} row {r}: position {p} predicts {shown!r} in the input, which "
                                     f"is not token {i if aligned else 0} of {values[r]!r} as it tokenizes alone "
                                     f"({engine.tokenizer.convert_ids_to_tokens(ids[:3])}); the answer is scored as "
                                     "it tokenizes, so the input says it the same way")


def answer(engine, plan: Plan, found: dict, ref, where: str) -> dict:
    """A column reference's answers, resolved once per findings (`state.tokens`): each row's `values`, per
    token the ids it may be (`sets`), the `column`, and whether its rows line up with the read (`aligned`:
    some row is several tokens; else every row is one, which scores every position of its row)."""
    key = refkey(ref)
    if key not in found["answers"]:
        try:
            values, sets, column = tokens(engine, plan, ref, None)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        found["answers"][key] = {"values": values, "sets": sets, "column": column,
                                 "aligned": any(len(row) > 1 for row in sets)}
    return found["answers"][key]


def placed(plan: Plan, found: dict, name: str, at: dict) -> tuple[list | None, list | None]:
    """The positions per row of the read value `name`'s positions are (following `of` through op steps), where
    they are known before the run, else None; and each row's input ids where that read is a scope that is
    `before` (whose next tokens an answer scores), else None."""
    while True:
        thing, _, member = name.partition(".")
        step = points(plan.steps[thing])[0]
        if step.kind != "op":
            break
        name = step.of
    try:
        where = lookup(found["positions"], name, at)
    except KeyError:
        return None, None
    record = entries(step).get(member) if step.kind in ("forward", "generate") else None
    scope = (positions.parse(record.pos)["scope"] or {}) if record is not None else {}
    return where, [e["ids"] for e in found["inputs"][data.input_name(step.input)]["rows"]] \
        if scope.get("before") else None


def check_classes(plan: Plan, where: str, name: str, args: dict, probe: str, classes: dict, swap: dict) -> None:
    """Every column op `name` scores against a probe's class list holds only those classes, on every row of the
    dataset it reads (named as the run reads it: under eval, the substituted one)."""
    vocabulary = classes[declared(plan, probe)]
    for key, p in parameters(plan, name, args).items():
        if p.annotation == ops.Tokens and key in args:
            ref, strip, lower = data.column_of(args[key]) or ("", False, False)
            if strip or lower:
                raise ValueError(f"{where}: {key} names a class of probe {declared(plan, probe)!r}, which is compared "
                                 "exactly; strip and lower compare an answer's tokens")
            dataset, _, column = ref.partition(".")
            for i, row in enumerate(plan.data[dataset].rows or []):
                if isinstance(row[column], list) or row[column] not in vocabulary:
                    raise ValueError(f"{where}: {key} is {column!r} of {swap.get(dataset, dataset)!r}, whose row {i} "
                                     f"is {row[column]!r}, not one of the classes of probe {declared(plan, probe)!r}: "
                                     f"{vocabulary}")


def check_operands(engine, plan: Plan, where: str, o, shape: tuple, step, site, found: dict, shapes: dict) -> None:
    """A write's operands that another entry selected: the same width and narrowing; on a `key` axis, inputs
    that line up token for token. One from an entry of the write's own forward is an earlier entry at the
    write's own site, or one at a site this engine reaches first (by site, not rank: the hooks engine ranks
    every component at one module boundary the same). An argument of the function an op's `options` choose (a
    lora's `with`) is as wide per position, its axes after the positions flattened, as the value the op acts on,
    and, from the same forward, read at the write's own positions."""
    args, sizes = arguments(o), engine.sizes()
    chosen = set(parameters(plan, o.op, args)) - set(ops.parameters(o.op))
    for key, p in parameters(plan, o.op, args).items():
        value = args.get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or value not in shapes:
            continue
        theirs = lookup(shapes, value, step.at)
        read = entries(points(plan.steps[value.partition(".")[0]])[0])[value.partition(".")[2]].site
        same = value.partition(".")[0] == where.partition(".")[0]
        theirs_at, ours_at = ((s.component, made_at(s, sizes["num_layers"])[0]) for s in (read, site))
        if same and theirs_at != ours_at and engine.rank(*theirs_at) >= engine.rank(*ours_at):
            raise ValueError(f"{where}: operand {value!r} is read at {read.component} layer {read.layers}, which the "
                             f"{engine.name} engine's forward reaches no earlier than this write's {site.component} "
                             f"layer {site.layers}; an operand from the write's own forward is an earlier entry at "
                             "this site, or at a site the forward reaches first")
        if key in chosen:
            per, own = (math.prod(kept.get(a) or sizes[a] for a in address.dims(s.component, None, False)[2:])
                        for s, kept in ((read, theirs[1]), (site, shape[1])))
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
        producer = data.input_name(points(plan.steps[value.partition(".")[0]])[0].input)
        consumer = data.input_name(step.input)
        if "key" in address.COMPONENTS[site.component]["dims"] and producer != consumer:
            for row, (a, b) in enumerate(zip(found["inputs"][producer]["rows"], found["inputs"][consumer]["rows"])):
                if len(a["ids"]) != len(b["ids"]):
                    raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                     f"{len(a['ids'])} tokens; {consumer} has {len(b['ids'])} there, so their "
                                     "key axes do not line up")
        if shape[0] > 1 and producer != consumer and not any(positions.parse(r.pos)["scope"] for r in (
                record_of(plan, where), record_of(plan, value))):
            windowed(engine, where, value, producer, consumer, found, step.at)


def windowed(engine, where: str, value: str, producer: str, consumer: str, found: dict, at: dict) -> None:
    """A window of several positions with no scope, swapped from one input into another, pairs the operand's
    i-th position with the write's i-th. On a row whose two prompts are as long, those are the same token's;
    on one where they are not, only where both sit among the tokens the prompts share, at the same distance
    from the side the window counts from (the common start counted from the start, the common end from the
    end). A window that reaches past them, into where the prompts differ in length, pairs different tokens,
    and is refused by name, as an attention pattern's keys are."""
    try:
        ours, theirs = lookup(found["positions"], where, at), lookup(found["positions"], value, at)
    except KeyError:
        return                  # beyond the prefill: no window there
    tokens = engine.tokenizer.convert_ids_to_tokens
    for row, (a, b) in enumerate(zip(found["inputs"][producer]["rows"], found["inputs"][consumer]["rows"])):
        x, y = a["ids"], b["ids"]
        if len(x) == len(y):
            continue
        start = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        end = next((i for i, (p, q) in enumerate(zip(x[::-1], y[::-1])) if p != q), min(len(x), len(y)))
        for i, (p, q) in enumerate(zip(theirs[row], ours[row])):
            if not (p == q < start or len(x) - p == len(y) - q <= end):
                raise ValueError(f"{where}: operand {value!r} was read on {producer}, row {row} of which is "
                                 f"{len(x)} tokens; {consumer} has {len(y)} there, so the window's position {i} "
                                 f"pairs {tokens([x[p]])[0]!r} with {tokens([y[q]])[0]!r}, past the tokens the two "
                                 "prompts share; read and write at a scope (the text the window means), one entry "
                                 "per position, or on inputs of one length")


def record_of(plan: Plan, name: str):
    """The entry `<step>.<entry>` names."""
    thing, _, member = name.partition(".")
    return entries(points(plan.steps[thing])[0])[member]


def check_labels(plan: Plan, where: str, o, step, site, kept: dict, sizes: dict) -> None:
    """A write's operand holds, on each axis its site narrows (`num_heads`, `hidden_size`, ...), the indices the
    write's selection does, in its order (`kept`, by axis; every one where it does not narrow): a `select` list is
    positional, so an operand read at heads `[1, 4]` swapped into a write at `[4, 1]` would trade the heads, and one
    head's value would broadcast over three. An operand's indices are those of the read it was taken over
    (`held`); a number, or a value no read made, has none to compare."""
    chosen = set(parameters(plan, o.op, arguments(o))) - set(ops.parameters(o.op))
    for key, p in parameters(plan, o.op, arguments(o)).items():
        value = arguments(o).get(key)
        if p.annotation != ops.Value or not isinstance(value, str) or key in chosen:
            continue
        theirs = held(plan, value, step.at, sizes)
        dims = dims_of(plan, value)
        for axis in address.selectable(site.component):
            if axis == "key" or axis not in dims or theirs is None or axis not in theirs:
                continue
            ours = kept.get(axis, list(range(sizes[axis])))
            if theirs[axis] == ours:
                continue
            if sorted(theirs[axis]) == sorted(ours):
                why = (f"a select list is positional, so its {axis} would land where this write's list puts its "
                       "own, trading them; select them in one order at both sites")
            elif len(theirs[axis]) == 1:
                why = (f"one {axis}'s value would be written into each of this write's, but an operand broadcasts "
                       f"over rows and positions only; read the {axis} this write selects, or write one per entry")
            else:
                why = f"an operand holds the {axis} this write selects, in its order"
            raise ValueError(f"{where}: operand {value!r} holds {axis} {theirs[axis]} and this write's selection "
                             f"holds {ours}; {why}")


def held(plan: Plan, name: str, at: dict, sizes: dict) -> dict[str, list[int]] | None:
    """The indices value `name` holds on each axis its read's site could narrow: what the site's `select` keeps,
    every index where it keeps them all; followed through op steps and into a fit's eval members. None for a
    value no read made (a fit's history, a generation's ids)."""
    thing, _, member = name.partition(".")
    step = agreeing(plan.steps[thing], at)
    if step.kind == "op":
        return held(plan, step.of, step.at, sizes)
    if step.kind == "fit":
        label, _, rest = member.partition(".")
        return held(body_view(plan, thing, step.at), rest, step.at, sizes) if label in step.eval and rest else None
    if not member:
        return None
    site = entries(step)[member].site
    chosen = positions.kept(site.select, sizes)
    return {a: chosen.get(a, list(range(sizes[a]))) for a in address.selectable(site.component)
            if a in chosen or a in sizes}