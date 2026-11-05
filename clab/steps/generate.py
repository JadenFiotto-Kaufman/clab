"""The `generate` step kind: a generation's schedule executed call by call, and its entries beyond the prefill
resolved from what each row generated."""

from __future__ import annotations

from collections import ChainMap
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from clab.compile import positions
from clab.compile.positions import reads_head, resolve_row, scope_texts
from clab.compile.schedule import values_read_by
from clab.compile.state import Findings
from clab.format import address, data
from clab.format.sweeps import projection, put
from clab.plan import Forward, Generate, Intervention, Plan, Site, entries, gathers, held_rows, writes
from clab.steps.batches import (apply_bound_ops, apply_ops_after_edit, bind_schedule, in_run_order, input_batches,
                                join_batches, layers_made_at, refuse_unreached, schedule_binding, scope_ids,
                                stack_layers, stop_ids, value_in_prefill)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from clab.engines import Engine


def generate(engine: Engine, plan: Plan, name: str, step: Generate, state: ChainMap, batch_size: int,
             findings: Findings, cost: dict[str, Any], served: list[Forward], values_read: dict | None = None,
             binding: list[dict] | None = None) -> None:
    """The step's input in batches of dataset row ids padded as the run says (`input_batches`, `plan.run.padding`), each
    through `engine.generate` with the point's schedule bound to it (`bind_schedule`, from `binding`, as a forward's)
    and the step's own keys as its options; each value the generation reads (`values_read`, `values_read_by` over the
    points it serves, `served`, as a forward's) joined along rows from its own reads, and the new ids, one
    generation's, put as the step's value, each in the run's row order, as where each entry landed is
    (`in_run_order`).

    An entry in the prefill (`frame: 0`) reads as a forward's (`value_in_prefill`); one beyond it is bound in parts (a
    part over the prompt, where its frame holds it, and its decode calls: `schedule.parts`) and resolves after the
    generation from what each row generated (`positions_beyond_prefill`, `value_beyond_prefill`); a row that stopped
    before a call a write selects is not written there and is recorded as missing, and a write at a decode call whose
    operand holds no value on a row that made the call is refused once the batch's generation says so
    (`batches.refuse_unreached`). The resolution is recorded per row on
    each point's record (`positions`) and each row's continuation, as token strings, in `plan.run.tokens[<step>]`, by a
    run over every row (not over a fit's minibatch). A batch whose rows all stopped early makes fewer calls; its ids are
    filled after their EOS to the longest batch's. Adds to `cost` the model calls it made, one per new token of each
    batch (its prefill, then a decode call per token after the first), the tokens they fed (the padded prompts, then one
    per row at each decode call), and the rows' own of them (`real`): the prompts', then each row's generated tokens but
    its last, which is its EOS (fed on the call after it says it, and not counted, as padding is not) or the
    generation's last, which no call is fed."""
    assert plan.run is not None
    eos, rows = set(stop_ids(engine, step)), held_rows(plan, cast(str, step.dataset))
    encoded = findings.inputs[data.input_name(step.input)]["rows"]
    values_read = values_read_by(served) if values_read is None else values_read
    binding = schedule_binding(engine, plan, name, step, values_read) if binding is None else binding
    # Every entry beyond the prefill lands somewhere on each row, gathered or not: by entry and the point's
    # coordinates on the entry's axes, the record it is at the first such point.
    beyond_prefill: dict[tuple[str, tuple], Intervention] = {}
    for point in served:
        for key, record in entries(point).items():
            if not positions.prefill_only(positions.parse(record.pos)):
                beyond_prefill.setdefault((key, tuple(projection(point.at, record.axes).items())), record)
    parts: dict[tuple[str, tuple], list] = {k: [] for k in values_read}
    beyond_positions: dict[tuple[str, tuple], list] = {k: [] for k in beyond_prefill}
    new, batched_ids = [], []
    for ids, batch, pads in input_batches(engine, step, state, findings, batch_size, plan.run.padding):
        batched_ids += ids
        bound_schedule, binds = bind_schedule(engine, plan, step, binding, state, findings, ids, pads)
        tokens, got = run_engine_generate(engine, name, batch, bound_schedule, dict(step.model_extra or {}))
        inputs, batch_rows = data.by_id(encoded, ids), data.by_id(rows, ids)
        said = [positions.continuation(x, eos) for x in tokens.tolist()]
        cost["forwards"] += tokens.shape[1]
        cost["tokens"] += batch["input_ids"].numel() + len(ids) * (tokens.shape[1] - 1)
        cost["real"] += batch["input_ids"].numel() - sum(pads) + sum(map(len, said)) - len(said)
        for k, record in beyond_prefill.items():
            beyond_positions[k] += positions_beyond_prefill(record, batch_rows, inputs, said, eos, engine.tokenizer,
                                                            reads_head(step, record))
        for bound in (b for site in binding for b in site["records"] if "unreached" in b and "calls" in b["scheduled"]):
            landed = beyond_positions[bound["scheduled"]["write"], tuple(projection(step.at, bound["record"].axes)
                                                                         .items())][-len(ids):]
            refuse_unreached(plan, findings, bound, ids, [
                positions.call_of(at[0], len(row["ids"])) if isinstance(at, list) else None
                for at, row in zip(landed, inputs)])
        for k, (record, own, point) in values_read.items():
            if k in beyond_prefill:
                row_positions = beyond_positions[k][-len(ids):]
                parts[k].append(stack_layers(record, [
                    value_beyond_prefill(engine, plan, record, got, binds[k + (layer,)], row_positions, inputs, ids,
                                         state, point.at, reads_head(step, record), findings, layer)
                    for layer in layers_made_at(record)]))
            else:
                parts[k].append(stack_layers(record, [
                    value_in_prefill(engine, plan, record, got[binds[k + (layer,)][0]["name"]], binds[k + (layer,)][0],
                                     ids, state, point.at, findings, layer)
                    for layer in layers_made_at(record)]))
        new.append(tokens)
    width = max(x.shape[1] for x in new)
    tokens = in_run_order(state, torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=min(eos))
                                            for x in new]), batched_ids)
    beyond_positions = {k: in_run_order(state, v, batched_ids) for k, v in beyond_positions.items()}
    for k, (record, own, _) in values_read.items():
        join_batches(state, f"{name}.{k[0]}", record, own, parts[k], batched_ids)
    put(state, name, projection(step.at, step.axes), tokens)
    if scope_ids(state) is None:          # a run over every row records where each record landed, on every record
        for point in served:
            for key, record in entries(point).items():
                k = (key, tuple(projection(point.at, record.axes).items()))
                if k in beyond_prefill:
                    record.positions = {**record.positions, cast(str, step.dataset): beyond_positions[k]}
        put(findings.said, name, projection(step.at, step.axes),
            [engine.tokenizer.convert_ids_to_tokens(positions.continuation(x, eos)) for x in tokens.tolist()])


def positions_beyond_prefill(record: Intervention, rows: list[dict], encoded: list[dict], said: list[list[int]],
                             eos: set[int], tokenizer: PreTrainedTokenizerBase,
                             head_only: bool) -> list[list[int] | str]:
    """Where an entry past the prefill resolved on each row, once what the row generated (`said`) is known: a
    write with a text scope at the calls `positions.lands_at_call` said so of as each input was fed, any other entry
    by `positions.resolve` (`resolve_row`); a marker for a row it does not reach."""
    pos = positions.parse(record.pos)
    if writes(record) and pos["scope"]:
        return [[len(encoded_row["ids"]) + k - 1 for k in range(1, len(row_said))
                 if positions.lands_at_call(pos, scope_texts(pos, row), row_said[:k], eos, tokenizer)]
                or positions.MISSING
                for row, encoded_row, row_said in zip(rows, encoded, said)]
    return [resolve_row(pos, row, encoded_row, row_said, tokenizer, head_only)
            for row, encoded_row, row_said in zip(rows, encoded, said)]


def run_engine_generate(engine: Engine, name: str, batch: dict, bound_schedule: dict[str, Any],
                        options: dict) -> tuple[Tensor, dict]:
    """`engine.generate`, which validates its own options: an error it raises naming one of them is reported
    as the step's, `<step>: generate rejected '<option>': <the engine's message>`, and a TypeError (an option of
    a type `GenerationConfig` did not catch at compile) as the step's with the options it gave; any other error
    passes unchanged. Each read at decode calls is checked to hold one value per call the batch made (a read out
    of forward order shifts calls)."""
    try:
        ids, got = engine.generate(batch, bound_schedule, options)
    except TypeError as e:
        raise ValueError(f"{name}: generate rejected its options {options!r}: {e}") from None
    except ValueError as e:
        key = next((k for k in options if repr(k) in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"{name}: generate rejected {key!r}: {e}") from None
    for site in bound_schedule["sites"]:
        for record in site["records"]:
            if "calls" in record and "ops" not in record:
                calls_made = range(1, ids.shape[1]) if record["calls"] is None else \
                    [k for k in record["calls"] if k < ids.shape[1]]
                if len(got.get(record["name"], [])) != len(calls_made):
                    raise RuntimeError(f"{name}.{record['name']}: the engine read {len(got.get(record['name'], []))} "
                                       f"of the {len(calls_made)} decode calls it made; a read out of forward order "
                                       "shifts calls")
    return ids, got


def value_beyond_prefill(engine: Engine, plan: Plan, record: Intervention, got: dict, parts: list[dict],
                         row_positions: list, encoded: list[dict], ids: list[int], state: ChainMap, at: dict,
                         head_only: bool, findings: Findings, layer: int | None) -> Any:
    """One layer of an entry's value beyond the prefill in one batch, arranged by the positions each row resolved
    to (`row_positions`, per row a list of positions or an alignment marker): a prompt position from the prefill's
    whole value (the read's part over the prompt, `[rows, width, ...]`, column `p` the row's token `p`; its one
    column under `head_only`), a generated input's from its decode call (the read's decode part: one `[rows, 1,
    ...]` per call it asked for that the batch made, `parts` holding the bound records, the decode part last with
    its `calls`), each already through the ops the read runs at each call (`per_call`, which `visit_site` applied as it
    read them); then the ops after those. The value is `[rows, width, ...]`, `width` the record's, or the most
    positions a row resolved to, missing past a row's own and on a row it does not reach: there it holds what the
    ops run at each call give a missing vector (NaN, -1 in ids), computed once on one. `layer` is the layer this is
    the value at."""
    pos, decode = positions.parse(record.pos), parts[-1]
    whole: Any = got.get(parts[0]["name"]) if len(parts) > 1 else None
    values: list = got.get(decode["name"], [])
    asked = decode["calls"]
    by_call = dict(zip(asked if asked is not None else range(1, len(values) + 1), values))
    each = decode.get("per_call", [])
    sizes, picks = engine.sizes(), record.kept
    rest = [len(picks[a]) if a in picks else sizes[a] for a in address.dims(cast(Site, record.site).component)[2:]]
    some = whole if whole is not None else values[0] if values else None
    shown = next(iter(some.values())) if isinstance(some, dict) else some     # a top_k's is a dict of tensors
    dtype = some.dtype if isinstance(some, Tensor) and not each else getattr(torch, plan.model.computes)
    device = shown.device if shown is not None else None
    blank = torch.full((len(row_positions), 1, *rest), float("nan") if dtype.is_floating_point else -1, dtype=dtype,
                       device=device)
    missing = apply_bound_ops(blank, blank, each, got, None)
    # a marker is a row with none
    reached_positions = [reached if isinstance(reached, list) else [] for reached in row_positions]
    width = positions.width(pos) or max(map(len, reached_positions), default=0)

    def arranged(missing: Tensor, whole: Tensor | None, by_call: dict[int, Tensor], like: Tensor | None) -> Tensor:
        missing = missing if like is None else missing.to(dtype=like.dtype, device=like.device)
        value = missing.expand(missing.shape[0], width, *missing.shape[2:]).contiguous()
        for i, got_ in enumerate(reached_positions):
            for j, position in enumerate(got_):
                call = positions.call_of(position, len(encoded[i]["ids"]))
                if call == 0:
                    assert whole is not None
                    value[i, j] = whole[i, 0 if head_only else position]
                else:
                    value[i, j] = by_call[call][i, 0]
        return value
    if isinstance(missing, dict):
        value: Any = {k: arranged(missing[k], whole[k] if whole is not None else None,
                                  {c: v[k] for c, v in by_call.items()}, some[k] if some is not None else None)
                      for k in missing}
    else:
        value = arranged(missing, whole, by_call, shown)
    token_positions = torch.tensor([reached + [-1] * (width - len(reached)) for reached in reached_positions],
                                   dtype=torch.long).reshape(len(row_positions), width)
    return apply_ops_after_edit(engine, plan, record, value, state, at, ids, token_positions, findings, layer,
                                len(record.ops) - len(gathers(record)) + len(each))
