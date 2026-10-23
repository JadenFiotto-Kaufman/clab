"""The `generate` step kind, and its entries beyond the prefill."""

from typing import Any

import torch
from torch import Tensor

from clab.compile import positions
from clab.compile.positions import head, place, texts
from clab.compile.state import Found
from clab.format import address, data
from clab.format.sweeps import lookup, put
from clab.plan import Plan, entries, horizon, writes
from clab.steps.batches import (
    bind,
    chunked,
    decoding,
    edit,
    joined,
    layers,
    read,
    read_name,
    site,
    stacked,
    with_ops,
    write_name,
)


def stops(engine, step) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def generate(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: Found) -> None:
    """The step's input in consecutive batches, each through `engine.generate` with the step's own keys as its
    options; each gathered entry joined along rows, and the new ids joined into the step's value.

    An entry in the prefill (`frame: 0`) binds as a forward's (`bind`); one beyond it goes to the engine in
    parts (`beyond`) and resolves after the generation from what each row generated (`landed_at`); a row that
    stopped before a call a write selects is not written there and is recorded as missing. The resolution is
    recorded per row in `plan.run.positions` and each row's continuation, as token strings, in
    `plan.run.tokens[<step>]`. A batch whose rows all stopped early makes fewer calls; its ids are filled after
    their EOS to the longest batch's."""
    eos, rows = set(stops(engine, step)), plan.data[step.dataset].rows or []
    encoded, padded = (found.inputs[data.input_name(step.input)][k] for k in ("rows", "width"))
    every = entries(step)
    first = {k for k, r in every.items() if positions.prefill(positions.parse(r.pos))}
    after = {k: r for k, r in every.items() if k not in first}
    picks = {key: lookup(found.selects, f"{name}.{key}", step.at) for key in every}
    parts, new, resolved = [], [], {f"{name}.{key}": [] for key in after}
    most = horizon(step)
    for chunk, batch, pads in chunked(engine, step, found, batch_size):
        sites, records = bind(engine, plan, name, step, state, found, chunk, pads, first)
        for key, record in after.items():
            records += beyond(engine, plan, record, key, picks[key], step, state, found, chunk, pads, padded,
                              encoded[chunk], rows[chunk], eos, most)
        ids, got = generated(engine, name, batch, records, dict(step.model_extra or {}))
        out = {key: stacked(record, [read(engine, plan, record, got[read_name(key, layer)], sites[key], chunk, state,
                                          step.at, found.answers) for layer in layers(engine, record)])
               for key, record in every.items() if key in first and record.gathered}
        said = [positions.continuation(x, eos) for x in ids.tolist()]
        for key, record in after.items():
            where = landed_at(record, rows[chunk], encoded[chunk], said, eos, engine.tokenizer, head(step, record))
            resolved[f"{name}.{key}"] += where
            if record.gathered:
                out[key] = stacked(record, [at_calls(engine, plan, record, picks[key], got.get(read_name(key, layer, True)),
                                                     got[read_name(key, layer)], where, encoded[chunk], chunk, state,
                                                     step.at, head(step, record), most, found.answers)
                                            for layer in layers(engine, record)])
        parts.append(out)
        new.append(ids)
    joined(state, name, step, parts)
    width = max(x.shape[1] for x in new)
    ids = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=min(eos)) for x in new])
    put(state, name, step.at, ids)
    for where, per_row in resolved.items():
        put(found.positions, where, step.at, per_row)
    put(found.said, name, step.at,
        [engine.tokenizer.convert_ids_to_tokens(positions.continuation(x, eos)) for x in ids.tolist()])


def beyond(engine, plan: Plan, record, key: str, picks: dict, step, state: dict, found: Found, chunk: slice,
           pads: list[int], padded: int, encoded: list[dict], rows: list[dict], eos: set[int],
           most: int | None) -> list[dict]:
    """An entry past the prefill as the engine takes it in one batch, in parts: a write's prefill part, where
    its frame holds the prompt, and its decode part (`calls`: the frame's own, or None for every decode call;
    a write with a text scope goes to every call with each row's texts, `online`, and lands where
    `positions.landing` says); then a gathered entry's whole prefill (which of its positions it means can
    depend on what the row says) and its decode calls, one read per layer, each at the call's one position."""
    pos = positions.parse(record.pos)
    holds, calls, _ = positions.reach(pos, most)
    head_only, layer = head(step, record), layers(engine, record)[0]
    out = []
    if writes(record):
        edits = edit(engine, plan, record, state, step.at, chunk, answers=found.answers)
        if holds:
            where = [positions.prompted(len(e["ids"]), head_only) for e in encoded]
            out.append({**site(record, picks, where, pads, padded, head_only), "name": write_name(key, True),
                        "layer": layer, "ops": edits})
        online = {"online": {"pos": pos, "texts": [texts(pos, row) for row in rows], "eos": sorted(eos)}} \
            if pos["scope"] else {}
        out.append({**decoding(record, picks, [len(e["ids"]) for e in encoded]), "name": write_name(key),
                    "layer": layer, "ops": edits, "calls": None if online else calls, **online})
    for layer in layers(engine, record) if record.gathered else []:
        if holds:
            whole = [positions.prompted(len(e["ids"])) for e in encoded]
            out.append({**site(record, picks, whole, pads, padded, head_only), "name": read_name(key, layer, True),
                        "layer": layer})
        out.append({**decoding(record, picks, [len(e["ids"]) for e in encoded]), "name": read_name(key, layer),
                    "layer": layer, "calls": calls})
    return out


def landed_at(record, rows: list[dict], encoded: list[dict], said: list[list[int]], eos: set[int], tokenizer,
              head_only: bool) -> list:
    """Where an entry past the prefill resolved on each row, once what the row generated (`said`) is known: a
    write with a text scope at the calls `positions.landing` said so of as each input was fed, any other entry
    by `positions.resolve` (`place`); a marker for a row it does not reach."""
    pos = positions.parse(record.pos)
    if writes(record) and pos["scope"]:
        return [[len(e["ids"]) + k - 1 for k in range(1, len(x))
                 if positions.landing(pos, texts(pos, row), x[:k], eos, tokenizer)] or positions.MISSING
                for row, e, x in zip(rows, encoded, said)]
    return [place(pos, row, e, x, tokenizer, head_only) for row, e, x in zip(rows, encoded, said)]


def generated(engine, name: str, batch: dict, records: list[dict], options: dict) -> tuple[Tensor, dict]:
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`. Each read at decode calls is
    checked to hold one value per call the batch made (a read out of forward order shifts calls)."""
    try:
        ids, got = engine.generate(batch, records, options)
    except (TypeError, ValueError) as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None
    for record in records:
        if "calls" in record and "ops" not in record:
            made = range(1, ids.shape[1]) if record["calls"] is None else [k for k in record["calls"] if k < ids.shape[1]]
            if len(got[record["name"]]) != len(made):
                raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got[record['name']])} of the "
                                   f"{len(made)} decode calls it made; a read out of forward order shifts calls")
    return ids, got


def at_calls(engine, plan: Plan, record, picks: dict, whole: Tensor | None, values: list[Tensor], where: list,
             encoded: list[dict], rows: slice, state: dict, at: dict, head_only: bool, most: int | None,
             answers: dict) -> Any:
    """One layer of an entry's value beyond the prefill in one batch, arranged by the positions each row resolved
    to (`where`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (`whole`, `[rows, width, ...]`, column `p` the row's token `p`; its one column under
    `head_only`), a generated input's from its decode call (`values`, one `[rows, 1, ...]` per call the record
    asked for that the batch made); then the ops after its last edit; `picks` is what its `select` keeps. The
    value is `[rows, width, ...]`, `width` the record's, or the most positions a row resolved to, NaN past a
    row's own and on a row it does not reach. `most` is the step's `max_new_tokens`, which bounds the calls it
    asked for."""
    pos = positions.parse(record.pos)
    _, asked, _ = positions.reach(pos, most)
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    sizes = engine.sizes()
    rest = [len(picks[a]) if a in picks else sizes[a] for a in address.dims(record.site.component)[2:]]
    some = whole if whole is not None else values[0] if values else None
    dtype = some.dtype if some is not None else getattr(torch, plan.model.dtype)
    device = some.device if some is not None else None

    def at_position(i: int, p: int) -> Tensor:
        call = positions.call_of(p, len(encoded[i]["ids"]))
        if call == 0:
            assert whole is not None
            return whole[i, 0 if head_only else p]
        return by_call[call][i, 0]

    found = [got if isinstance(got, list) else [] for got in where]      # a marker is a row with none
    width = positions.width(pos) or max(map(len, found), default=0)
    value = torch.full((len(where), width, *rest), float("nan"), dtype=dtype, device=device)
    for i, got in enumerate(found):
        for j, p in enumerate(got):
            value[i, j] = at_position(i, p)
    places = torch.tensor([got + [-1] * (width - len(got)) for got in found], dtype=torch.long).reshape(len(where), width)
    return with_ops(engine, plan, record, value, state, at, rows, places, answers)
