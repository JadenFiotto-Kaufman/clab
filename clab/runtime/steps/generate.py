"""The `generate` step kind, and its reads beyond the prefill."""

from typing import Any

import torch
from torch import Tensor

from ... import address
from ...plan.model import Plan, horizon
from .. import positions
from . import kind
from .batches import batches, bind, decoding, edit, joined, layers, read, site, stacked, with_ops
from .preflight import head, place, prompted, texts
from .state import put


def stops(engine, step) -> list[int]:
    """The ids a generate step's rows stop on: its own `eos_token_id` when it gives one, else the engine's."""
    given = (step.model_extra or {}).get("eos_token_id")
    return engine.eos if given is None else [given] if isinstance(given, int) else list(given)


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
    most = horizon(step)
    for chunk, batch, pads in batches(engine, step, found, batch_size):
        prompts = [len(e["ids"]) for e in encoded[chunk]]
        sites, writes, reads = bind(engine, plan, name, step, first, state, found, chunk, pads, set(beyond))
        for key, record in beyond.items():
            pos = positions.parse(record.pos)
            holds, calls, _ = positions.reach(pos, most)
            edits = [edit(engine, plan, record, o, state, step.at, chunk) for o in record.ops]
            if holds:
                where = [prompted(e, engine.tokenizer, head(step, record)) for e in encoded[chunk]]
                writes.append({**site(record, where, pads, head(step, record)), "name": f"{key}#0",
                               "layer": layers(engine, record)[0], "ops": edits})
            online = {"online": {"pos": pos, "texts": [texts(pos, row) for row in rows[chunk]],
                                 "eos": sorted(eos)}} if pos["scope"] else {}
            writes.append({**decoding(record, prompts, pads), "name": key, "layer": layers(engine, record)[0],
                           "ops": edits, "calls": None if online else calls, **online})
        for key, record in later.items():
            holds, calls, _ = positions.reach(positions.parse(record.pos), most)
            for layer in layers(engine, record):
                if holds:
                    whole = [list(range(len(e["ids"]))) for e in encoded[chunk]]
                    reads.append({**site(record, whole, pads, head(step, record)), "name": f"{key}@{layer}#0",
                                  "layer": layer})
                reads.append({**decoding(record, prompts, pads), "name": f"{key}@{layer}", "layer": layer,
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
                where = [place(pos, row, e, x, engine.tokenizer, head(step, record))
                         for row, e, x in zip(rows[chunk], encoded[chunk], said)]
            resolved[f"{name}.{key}"] += where
            if key in later:
                out[key] = stacked(record, [at_calls(engine, plan, record, got.get(f"{key}@{layer}#0"),
                                                     got[f"{key}@{layer}"], where, encoded[chunk], pads, chunk,
                                                     state, step.at, head(step, record), most)
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


def at_calls(engine, plan: Plan, record, whole: Tensor | None, values: list[Tensor], where: list, encoded: list[dict],
             pads: list[int], rows: slice, state: dict, at: dict, head_only: bool, most: int | None) -> Any:
    """One layer of a read beyond the prefill in one batch, arranged by the positions each row resolved to
    (`where`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (`whole`, `[rows, width, ...]`, column `p` the row's token `p`; its one column under
    `head_only`), a generated input's from
    its decode call (`values`, one `[rows, 1, ...]` per call the record asked for that the batch made); then
    its ops. A rectangular read is `[rows, width, ...]`, NaN on a row it does not reach; a ragged one
    `{values, row, tok}`, `tok` the position, with no entries for such a row. `most` is the step's
    `max_new_tokens`, which bounds the calls it asked for."""
    pos = positions.parse(record.pos)
    _, asked, _ = positions.reach(pos, most)
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    sizes, select = engine.sizes(), record.site.select or {}
    rest = [len(select[a]) if a in select else sizes[a] for a in address.dims(record.site.component, None, False)[2:]]
    some = whole if whole is not None else values[0] if values else None
    dtype = some.dtype if some is not None else getattr(torch, plan.model.dtype)
    device = some.device if some is not None else None

    def at_position(i: int, p: int) -> Tensor:
        call = positions.call_of(p, len(encoded[i]["ids"]))
        if call == 0:
            assert whole is not None
            return whole[i, 0 if head_only else p]
        return by_call[call][i, 0]

    if (width := positions.width(pos)) is not None:
        blank = torch.full((width, *rest), float("nan"), dtype=dtype, device=device)
        value = torch.stack([blank if isinstance(got, str) else torch.stack([at_position(i, p) for p in got])
                             for i, got in enumerate(where)])
        places = torch.tensor([[-1] * width if isinstance(got, str) else got for got in where])
    else:
        kept = [(i, p) for i, got in enumerate(where) if not isinstance(got, str) for p in got]
        value = {"values": torch.stack([at_position(i, p) for i, p in kept]) if kept else
                 torch.zeros((0, *rest), dtype=dtype, device=device),
                 "row": torch.tensor([rows.start + i for i, _ in kept], dtype=torch.long),
                 "tok": torch.tensor([p for _, p in kept], dtype=torch.long)}
        places = None
    return with_ops(engine, plan, record, value, state, at, rows, places)
