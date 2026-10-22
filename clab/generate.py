"""The `generate` step kind, and its entries beyond the prefill."""

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from . import address, data, positions
from .batches import batches, decoding, edit, joined, layers, read, site, stacked, with_ops
from .plan import Plan, entries, horizon, writes
from .positions import head, place, texts
from .sweeps import lookup, put

if TYPE_CHECKING:
    from .preflight import Found


def stops(engine, step) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


def generate(engine, plan: Plan, name: str, step, state: dict, batch_size: int, found: "Found") -> None:
    """The step's input in consecutive batches, each through `engine.generate` with the step's own keys as
    its options; each gathered entry joined along rows, and the new ids joined into the step's value.

    An entry in the prefill (`frame: 0`) is a forward's. One beyond it goes to the engine in two parts: its
    prefill positions (for a read, the whole prefill, since which of its positions it means can depend on
    what the row says), and its decode calls, each at that call's one position (`calls`: the frame's own,
    or None for every decode call); an entry that writes and is gathered goes as its write, then its read.
    A write with a text scope is found online: it goes to every decode call with each row's text (`online`),
    lands where `positions.landing` says, and is recorded at the calls it says so of afterwards. Every other
    entry resolves after the generation from what each row generated (`positions.resolve`); a row that
    stopped before a call a write selects is not written there (the call computes nothing of that row's) and
    is recorded as missing. The resolution is recorded per row in `plan.run.positions` and each row's
    continuation, as token strings, in `plan.run.tokens[<step>]`. A batch whose rows all stopped early makes
    fewer calls; its ids are filled after their EOS to the longest batch's."""
    eos, rows = set(stops(engine, step)), plan.data[step.dataset].rows or []
    encoded, padded = (found.inputs[data.input_name(step.input)][k] for k in ("rows", "width"))
    every = entries(step)
    first = {k: r for k, r in every.items() if positions.prefill(positions.parse(r.pos))}
    after = {k: r for k, r in every.items() if k not in first}
    parts, new, resolved = [], [], {f"{name}.{key}": [] for key in after}
    most = horizon(step)
    picks = {key: lookup(found.selects, f"{name}.{key}", step.at) for key in every}
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        prompts = [len(e["ids"]) for e in encoded[chunk]]
        sites, records = {}, []
        for key, record in every.items():
            pos = positions.parse(record.pos)
            if key in first:
                if writes(record) or record.gathered:
                    sites[key] = site(record, picks[key], lookup(found.positions, f"{name}.{key}", step.at)[chunk],
                                      pads, padded, head(step, record))
                if writes(record):
                    records.append({**sites[key], "name": key, "layer": layers(engine, record)[0],
                                    "ops": edit(engine, plan, record, state, step.at, chunk, answers=found.answers)})
                if record.gathered:
                    records += [{**sites[key], "name": f"{key}@{layer}", "layer": layer}
                                for layer in layers(engine, record)]
                continue
            holds, calls, _ = positions.reach(pos, most)
            if writes(record):
                edits = edit(engine, plan, record, state, step.at, chunk, answers=found.answers)
                if holds:
                    where = [positions.prompted(len(e["ids"]), head(step, record)) for e in encoded[chunk]]
                    records.append({**site(record, picks[key], where, pads, padded, head(step, record)),
                                    "name": f"{key}#0", "layer": layers(engine, record)[0], "ops": edits})
                online = {"online": {"pos": pos, "texts": [texts(pos, row) for row in rows[chunk]],
                                     "eos": sorted(eos)}} if pos["scope"] else {}
                records.append({**decoding(record, picks[key], prompts), "name": key,
                                "layer": layers(engine, record)[0], "ops": edits, "calls": None if online else calls,
                                **online})
            for layer in layers(engine, record) if record.gathered else []:
                if holds:
                    whole = [positions.prompted(len(e["ids"])) for e in encoded[chunk]]
                    records.append({**site(record, picks[key], whole, pads, padded, head(step, record)),
                                    "name": f"{key}@{layer}#0", "layer": layer})
                records.append({**decoding(record, picks[key], prompts), "name": f"{key}@{layer}",
                                "layer": layer, "calls": calls})
        ids, got = generated(engine, name, batch, records, dict(step.model_extra or {}))
        for record in records:
            if "calls" in record and "ops" not in record:
                made = range(1, ids.shape[1]) if record["calls"] is None else \
                    [k for k in record["calls"] if k < ids.shape[1]]
                if len(got[record["name"]]) != len(made):
                    raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got[record['name']])} of the "
                                       f"{len(made)} decode calls it made; a read out of forward order shifts calls")
        out = {key: stacked(record, [read(engine, plan, record, got[f"{key}@{layer}"], sites[key], chunk, state,
                                          step.at, found.answers) for layer in layers(engine, record)])
               for key, record in first.items() if record.gathered}
        said = [positions.continuation(x, eos) for x in ids.tolist()]
        for key, record in after.items():
            pos = positions.parse(record.pos)
            if writes(record) and pos["scope"]:
                where = [[len(e["ids"]) + k - 1 for k in range(1, len(x))
                          if positions.landing(pos, texts(pos, row), x[:k], eos, engine.tokenizer)] or positions.MISSING
                         for row, e, x in zip(rows[chunk], encoded[chunk], said)]
            else:
                where = [place(pos, row, e, x, engine.tokenizer, head(step, record))
                         for row, e, x in zip(rows[chunk], encoded[chunk], said)]
            resolved[f"{name}.{key}"] += where
            if record.gathered:
                out[key] = stacked(record, [at_calls(engine, plan, record, picks[key], got.get(f"{key}@{layer}#0"),
                                                     got[f"{key}@{layer}"], where, encoded[chunk], chunk, state,
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


def generated(engine, name: str, batch: dict, records: list[dict], options: dict):
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`."""
    try:
        return engine.generate(batch, records, options)
    except (TypeError, ValueError) as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None


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


def now(records: list, call: int) -> list:
    """What a generation does at a site on forward call `call` (0 is the prefill), in order: a record with no
    `calls` acts on the prefill only, one with `calls` on those decode calls (None: every one)."""
    def wanted(record: dict) -> bool:
        if "calls" not in record:
            return call == 0
        return call > 0 and (record["calls"] is None or call in record["calls"])
    return [r for r in records if wanted(r)]


def landed(records: list[dict], seen: list, tokenizer) -> list[dict]:
    """The records at a decode call, each write stamped with the call (`call`, the number of inputs `seen` so
    far), a write with an `online` scope narrowed to the rows it lands on there (`keep`), as
    `positions.landing` decides from those inputs (one `[rows, 1]` per decode call); one that lands on no row
    is left out. A read is as it was."""
    out = []
    for w in records:
        if "ops" not in w:
            out.append(w)
            continue
        online, w = w.get("online"), {**w, "call": len(seen)}
        if online is None:
            out.append(w)
            continue
        rows = [positions.landing(online["pos"], texts, [int(t[r, 0]) for t in seen], set(online["eos"]), tokenizer)
                for r, texts in enumerate(online["texts"])]
        if any(rows):
            out.append({**w, "keep": torch.tensor(rows)[:, None]})
    return out
