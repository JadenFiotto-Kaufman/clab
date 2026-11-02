"""The `generate` step kind: a generation's schedule executed call by call, and its entries beyond the prefill
resolved from what each row generated."""

from __future__ import annotations

from collections import ChainMap
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from clab.compile import positions
from clab.compile.positions import head, place, texts
from clab.compile.schedule import gathered
from clab.compile.state import Found
from clab.format import address, data
from clab.format.sweeps import projected, put
from clab.plan import Forward, Generate, Intervention, Plan, Site, entries, held_rows, writes
from clab.steps.batches import bind, chunked, joined, layers, over, read, read_over, stacked, stops, with_ops

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def generate(engine: Engine, plan: Plan, name: str, step: Generate, state: ChainMap, batch_size: int, found: Found,
             cost: dict[str, Any], served: list[Forward], reading: dict | None = None) -> None:
    """The step's input in batches of dataset row ids padded as the run says (`chunked`, `plan.run.padding`), each
    through `engine.generate` with the point's schedule bound to it (`bind`) and the step's own keys as its options;
    each value the generation reads (`reading`, `gathered` over the points it serves, `served`, as a forward's)
    joined along rows from its own reads, and the new ids, one generation's, put as the step's value.

    An entry in the prefill (`frame: 0`) reads as a forward's (`read`); one beyond it is bound in parts (a part over
    the prompt, where its frame holds it, and its decode calls: `schedule.parts`) and resolves after the generation
    from what each row generated (`landed_at`, `at_calls`); a row that stopped before a call a write selects is not
    written there and is recorded as missing. The resolution is recorded per row on each point's record
    (`positions`) and each row's continuation, as token strings, in `plan.run.tokens[<step>]`, by a run over every
    row (not over a fit's minibatch). A batch whose rows all stopped early makes fewer calls; its ids are filled
    after their EOS to the longest batch's. Adds to `cost` the model calls it made, one per new token of each batch
    (its prefill, then a decode call per token after the first), and the tokens they fed: the prompts', then each
    row's generated tokens but its last, which is its EOS (fed on the call after it says it, and not counted, as
    padding is not) or the generation's last, which no call is fed."""
    assert plan.run is not None
    eos, rows = set(stops(engine, step)), held_rows(plan, cast(str, step.dataset))
    encoded = found.inputs[data.input_name(step.input)]["rows"]
    reading = gathered(served) if reading is None else reading
    # Every entry beyond the prefill lands somewhere on each row, gathered or not: by entry and the point's
    # coordinates on the entry's axes, with the record it is at the first such point.
    landing: dict[tuple[str, tuple], tuple[Intervention, dict[str, int], Any]] = {}
    for p in served:
        for key, record in entries(p).items():
            if not positions.prefill(positions.parse(record.pos)):
                landing.setdefault((key, tuple(projected(p.at, record.axes).items())), (record, projected(p.at, record.axes), p))
    parts: dict[tuple[str, tuple], list] = {k: [] for k in reading}
    landed: dict[tuple[str, tuple], list] = {k: [] for k in landing}
    new = []
    for ids, batch, pads in chunked(engine, step, state, found, batch_size, plan.run.padding):
        bound, binds = bind(engine, plan, step, state, found, ids, pads, reading)
        tokens, got = generated(engine, name, batch, bound, dict(step.model_extra or {}))
        inputs, examples = data.by_id(encoded, ids), data.by_id(rows, ids)
        said = [positions.continuation(x, eos) for x in tokens.tolist()]
        cost["forwards"] += tokens.shape[1]
        cost["tokens"] += batch["input_ids"].shape[1] * len(pads) - sum(pads) + sum(map(len, said)) - len(said)
        for k, (record, own, p) in landing.items():
            landed[k] += landed_at(record, examples, inputs, said, eos, engine.tokenizer, head(step, record))
        for k, (record, own, p) in reading.items():
            if k in landing:
                where = landed[k][-len(ids):]
                parts[k].append(stacked(record, [at_calls(engine, plan, record, got, binds[k + (layer,)], where, inputs,
                                                          ids, state, p.at, head(step, record), found, layer)
                                                 for layer in layers(engine, record)]))
            else:
                parts[k].append(stacked(record, [read(engine, plan, record, got[binds[k + (layer,)][0]["name"]],
                                                      binds[k + (layer,)][0], ids, state, p.at, found, layer)
                                                 for layer in layers(engine, record)]))
        new.append(tokens)
    width = max(x.shape[1] for x in new)
    tokens = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=min(eos)) for x in new])
    for k, (record, own, _) in reading.items():
        joined(state, f"{name}.{k[0]}", record, own, parts[k])
    put(state, name, projected(step.at, step.axes), tokens)
    if over(state) is None:          # a run over every row records where each record landed, on every record
        for p in served:
            for key, record in entries(p).items():
                k = (key, tuple(projected(p.at, record.axes).items()))
                if k in landing:
                    record.positions = {**record.positions, read_over(step, found): landed[k]}
        put(found.said, name, projected(step.at, step.axes),
            [engine.tokenizer.convert_ids_to_tokens(positions.continuation(x, eos)) for x in tokens.tolist()])


def landed_at(record: Intervention, rows: list[dict], encoded: list[dict], said: list[list[int]], eos: set[int],
              tokenizer: PreTrainedTokenizerBase, head_only: bool) -> list[list[int] | str]:
    """Where an entry past the prefill resolved on each row, once what the row generated (`said`) is known: a
    write with a text scope at the calls `positions.landing` said so of as each input was fed, any other entry
    by `positions.resolve` (`place`); a marker for a row it does not reach."""
    pos = positions.parse(record.pos)
    if writes(record) and pos["scope"]:
        return [[len(e["ids"]) + k - 1 for k in range(1, len(x))
                 if positions.landing(pos, texts(pos, row), x[:k], eos, tokenizer)] or positions.MISSING
                for row, e, x in zip(rows, encoded, said)]
    return [place(pos, row, e, x, tokenizer, head_only) for row, e, x in zip(rows, encoded, said)]


def generated(engine: Engine, name: str, batch: dict, bound: dict[str, Any], options: dict) -> tuple[Tensor, dict]:
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`, and a TypeError (an option of
    a type `GenerationConfig` did not catch at compile) as the step's with the options it gave; any other error
    passes unchanged. Each read at decode calls is checked to hold one value per call the batch made (a read out
    of forward order shifts calls)."""
    try:
        ids, got = engine.generate(batch, bound, options)
    except TypeError as e:
        raise ValueError(f"{name}: generate rejected its options {options!r}: {e}") from None
    except ValueError as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None
    for site in bound["sites"]:
        for record in site["records"]:
            if "calls" in record and "ops" not in record:
                made = range(1, ids.shape[1]) if record["calls"] is None else [k for k in record["calls"] if k < ids.shape[1]]
                if len(got.get(record["name"], [])) != len(made):
                    raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got.get(record['name'], []))} "
                                       f"of the {len(made)} decode calls it made; a read out of forward order shifts "
                                       "calls")
    return ids, got


def at_calls(engine: Engine, plan: Plan, record: Intervention, got: dict, parts: list[dict], where: list,
             encoded: list[dict], ids: list[int], state: ChainMap, at: dict, head_only: bool, found: Found,
             layer: int | None) -> Any:
    """One layer of an entry's value beyond the prefill in one batch, arranged by the positions each row resolved
    to (`where`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (the read's part over the prompt, `[rows, width, ...]`, column `p` the row's token `p`; its one
    column under `head_only`), a generated input's from its decode call (the read's decode part: one `[rows, 1,
    ...]` per call it asked for that the batch made, `parts` holding the bound records, the decode part last with
    its `calls`); then the ops after its last edit. The value is `[rows, width, ...]`, `width` the record's, or the
    most positions a row resolved to, NaN past a row's own and on a row it does not reach; `layer` is the layer
    this is the value at."""
    pos, decode = positions.parse(record.pos), parts[-1]
    whole: Tensor | None = got.get(parts[0]["name"]) if len(parts) > 1 else None
    values: list[Tensor] = got.get(decode["name"], [])
    asked = decode["calls"]
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    sizes, picks = engine.sizes(), record.kept
    rest = [len(picks[a]) if a in picks else sizes[a] for a in address.dims(cast(Site, record.site).component)[2:]]
    some = whole if whole is not None else values[0] if values else None
    dtype = some.dtype if some is not None else getattr(torch, plan.model.computes)
    device = some.device if some is not None else None

    def at_position(i: int, p: int) -> Tensor:
        call = positions.call_of(p, len(encoded[i]["ids"]))
        if call == 0:
            assert whole is not None
            return whole[i, 0 if head_only else p]
        return by_call[call][i, 0]

    hit = [got_ if isinstance(got_, list) else [] for got_ in where]      # a marker is a row with none
    width = positions.width(pos) or max(map(len, hit), default=0)
    value = torch.full((len(where), width, *rest), float("nan"), dtype=dtype, device=device)
    for i, got_ in enumerate(hit):
        for j, p in enumerate(got_):
            value[i, j] = at_position(i, p)
    places = torch.tensor([got_ + [-1] * (width - len(got_)) for got_ in hit], dtype=torch.long).reshape(len(where), width)
    return with_ops(engine, plan, record, value, state, at, ids, places, found, layer)
