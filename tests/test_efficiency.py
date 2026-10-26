"""What a run costs, and where a forward ends.

`plan.run.costs` holds every step's and point's seconds, peak memory (null off CUDA), model calls and input
tokens, exact for a forward, a swept step, an op and a fit, and the server's under ndif, which come home in the run
record; `clab show` prints them. (A generation's are pinned in test_generate.py.)

A forward stops right after its deepest site, by the engine's `rank`, on the nnterp and hooks engines: nothing
after it runs (a hook on each block, MLP, the final norm and the head watches what does), and every value it reads
or writes is the whole forward's to the bit. A generation runs whole; a fit's body forward stops where its own
entries do. The ndif engine runs every forward whole, under `remote="local"` and through the stand-in server. A stopped forward leaves none of its frames (and so none of its
activations) for the garbage collector.
"""

import gc
import json
import types
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from conftest import BATCH_SIZE, TRIP, edited, plan_of, same, stand_in

from clab import Plan
from clab.cli import main
from clab.format.sweeps import labelled, points, put
from clab.plan import entries
from clab.steps.batches import edit

COST = {"seconds", "memory", "forwards", "tokens"}


def lengths(run, name: str) -> int:
    """The tokens an input or a generation recorded in the run record, every row's."""
    return sum(map(len, run.tokens[name]))


def test_every_step_and_point_has_a_cost_and_a_forward_one_call_per_batch(engines):
    """logit_lens: `lens` is swept over two layers, so its cost and its ops' are keyed by coordinate as
    `positions` are; its two points only read, so they share one forward, two batches of two of the four rows,
    each prompt token fed once, which the first point records and the second does not; an op calls no model; off
    CUDA no memory is measured."""
    plan = plan_of("logit_lens")
    run = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    assert list(run.costs) == list(plan.steps)
    for name, entry in plan.steps.items():
        costs = run.costs[name]
        assert list(costs) == [labelled(plan, p.at) for p in points(entry)] == ["layers=0", "layers=1"]
        for cost in costs.values():
            assert set(cost) == COST and cost["memory"] is None and cost["seconds"] >= 0
    prompts = lengths(run, "prompts.input")
    assert prompts == 38
    assert [(c["forwards"], c["tokens"]) for c in run.costs["lens"].values()] == [(2, prompts), (0, 0)]
    assert all((c["forwards"], c["tokens"]) == (0, 0) for name in ("p_answer", "top1")
               for c in run.costs[name].values())


def test_a_forward_feeds_the_same_tokens_at_every_batch_size(engines):
    """Padding is not fed: four rows in batches of 1, 2 and 4 make 4, 2 and 1 calls of the same 38 tokens."""
    plan = plan_of("logit_lens")
    engine = engines("hooks", plan.model)
    lens = {size: engine.execute(plan, batch_size=size).run.costs["lens"]["layers=0"] for size in (1, 2, 4)}
    assert {size: (c["forwards"], c["tokens"]) for size, c in lens.items()} == {1: (4, 38), 2: (2, 38), 4: (1, 38)}


def das(epochs: int = 3):
    """das with `epochs` epochs, no early stop, and minibatches of one row: two of its two training rows an
    epoch, each through the body's two forwards."""
    def change(d):
        fit = d["steps"]["fit"]
        fit.update(epochs=epochs, batch_size=1)
        del fit["early_stop"]
    return edited("das", change).compile()


def test_a_fit_counts_every_forward_its_body_makes(engines):
    """A fit's calls are its body's: per epoch, each minibatch through the body's two forwards (2 × 2) and the
    training rows scored after it (two rows, a batch of one each, through both: 4); then each eval's rows once
    (held_out's two test rows, train's two). Each pass over a dataset feeds each of its rows' tokens once: four
    passes over the training rows an epoch, two of them scoring, then one in the `train` eval and one over the
    test rows in `held_out`."""
    plan = das()
    run = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    train = lengths(run, "train.counterfactual_inputs[0]") + lengths(run, "train.input")
    test = lengths(run, "test.counterfactual_inputs[0]") + lengths(run, "test.input")
    epochs, minibatches, body = 3, 2, 2
    assert run.costs["fit"]["forwards"] == epochs * (minibatches * body + 2 * body) + 2 * body + 2 * body == 32
    assert run.costs["fit"]["tokens"] == epochs * 2 * train + train + test


def test_the_servers_costs_come_home_in_the_run_record(engines, monkeypatch):
    """Under ndif the steps run on the server, so their costs are measured there and come home with the run record,
    the same calls and tokens as a local run's, through `remote="local"` and the stand-in server alike."""
    local = engines("nnterp", das().model).execute(das(), batch_size=BATCH_SIZE).run
    there = engines("ndif", das().model, remote="local").execute(das(), batch_size=BATCH_SIZE).run
    came, trip = stand_in(engines, das(), monkeypatch)
    assert trip == TRIP
    assert came.run is not None
    for run in (there, came.run):
        assert list(run.costs) == ["fit"] and set(run.costs["fit"]) == COST and run.costs["fit"]["memory"] is None
        assert {k: run.costs["fit"][k] for k in ("forwards", "tokens")} == \
            {k: local.costs["fit"][k] for k in ("forwards", "tokens")}


def test_memory_is_each_cuda_devices_peak_and_a_remote_run_measures_none(engines, monkeypatch):
    """With the model on a CUDA device (stood in for here: torch's calls recorded, the device's peak 7 bytes), each
    step synchronizes it, resets its peak and reads it back, by device; under the ndif engine, even
    `remote="local"`, the run makes no CUDA call and `memory` is null."""
    called: list[str] = []
    for call in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, call, lambda *_, call=call: called.append(call) or 7)
    monkeypatch.setattr("clab.steps.gpus", lambda engine: ["cuda:0"])
    plan = plan_of("logit_lens")
    local = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    assert local.costs["lens"]["layers=0"]["memory"] == {"cuda:0": 7}
    assert called[:3] == ["synchronize", "reset_peak_memory_stats", "synchronize"]
    called.clear()
    remote = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE).run
    assert called == [] and all(c["memory"] is None for costs in remote.costs.values() for c in costs.values())


def test_what_the_model_does_once_is_done_before_the_first_step(monkeypatch):
    """A fresh ndif engine under `remote="local"`: its shell has loaded its weights, and nnterp measured its forward
    order, before the first step's timer starts, so that no step's cost holds either (one timer per group of
    points: `lens`'s two share a forward, then each op point's)."""
    from nnterp import order

    from clab.engines import load
    from clab.steps import measured
    seen: list[Any] = []
    compute = order.compute
    monkeypatch.setattr(order, "compute", lambda model: seen.append("order") or compute(model))
    monkeypatch.setattr("clab.steps.measured", lambda engine, call: seen.append(("step", engine.model.dispatched))
                        or measured(engine, call))
    plan = plan_of("logit_lens")
    load("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    assert seen == ["order"] + [("step", True)] * 5


def test_show_prints_each_steps_cost(engines, tmp_path, capsys):
    plan = plan_of("logit_lens")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[3].split() == ["step", "at", "seconds", "memory", "forwards", "tokens"]
    rows = [line.split() for line in shown[4:10]]
    assert [(r[0], r[1], *r[3:]) for r in rows] == [
        ("lens", "layers=0", "—", "2", "38"), ("lens", "layers=1", "—", "=", "layers=0", "0"),
        ("p_answer", "layers=0", "—", "0", "0"), ("p_answer", "layers=1", "—", "0", "0"),
        ("top1", "layers=0", "—", "0", "0"), ("top1", "layers=1", "—", "0", "0")]
    assert all(float(r[2]) >= 0 for r in rows) and float(rows[1][2]) == 0
    assert shown[10].split() == ["result", "step", "dims", "axes", "shape"]
    manifest = tmp_path / "manifest.json"                    # as a run that recorded no costs wrote it
    written = json.loads(manifest.read_text())
    del written["run"]["costs"]
    manifest.write_text(json.dumps(written))
    assert main(["show", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[3].split() == ["result", "step", "dims", "axes", "shape"]


def watched(engine) -> dict[Any, torch.nn.Module]:
    """The modules a test watches run, by tag: each block by its index, its MLP as `m<i>`, the final norm as `N` and
    the head as `H` (the hooks engine's native modules; under nnterp, its envoys' own)."""
    if engine.name == "hooks":
        paths = {c: engine.config["sites"][c]["paths"] for c in ("layer_output", "mlp_output", "final_norm_output",
                                                                 "lm_head_output")}
        module, n = engine.model.get_submodule, len(paths["layer_output"])
        return {**{i: module(paths["layer_output"][i]) for i in range(n)},
                **{f"m{i}": module(paths["mlp_output"][i]) for i in range(n)},
                "N": module(paths["final_norm_output"][None]), "H": module(paths["lm_head_output"][None])}
    layers = engine.model.layers
    return {**{i: layer._module for i, layer in enumerate(layers)}, **{f"m{i}": layer.mlp._module for i, layer in
            enumerate(layers)}, "N": engine.model.norm._module, "H": engine.model.lm_head._module}


@contextmanager
def counting(engine) -> Iterator[list[Any]]:
    """The tag of each watched module that is called on real tensors while inside, in the order they are called:
    not in nnterp's measure of its forward order, an nnsight scan on fake tensors (which a server makes once per
    request). A module is counted when it is called, before it runs, so a block the forward stops inside counts
    and its MLP shows whether it got that far."""
    ran: list[Any] = []

    def noted(tag: Any):
        def note(module, args, kwargs) -> None:
            if not any(isinstance(v, FakeTensor) for v in (*args, *kwargs.values())):
                ran.append(tag)
        return note
    handles = [m.register_forward_pre_hook(noted(tag), with_kwargs=True) for tag, m in watched(engine).items()]
    try:
        yield ran
    finally:
        for handle in handles:
            handle.remove()


def counted(engine, plan: Plan) -> tuple[Plan, list[Any]]:
    """`plan` run on `engine`, and each watched module that ran. It runs once first, uncounted, so that a meta shell
    (ndif under `remote="local"`, which loads its weights at its first forward) has the modules a hook watches."""
    engine.execute(plan, batch_size=BATCH_SIZE)
    with counting(engine) as ran:
        filled = engine.execute(plan, batch_size=BATCH_SIZE)
    return filled, ran


def through(n: int) -> list[Any]:
    """Blocks 0 to `n - 1` called, each with its MLP."""
    return [tag for i in range(n) for tag in (i, f"m{i}")]


WHOLE = through(5) + ["N", "H"]           # tiny GPT-2's whole forward


def at(component: str, layer: int | None = None, ops: list | None = None) -> dict:
    """An entry at the last token of `component` (at block `layer`, for a block's), with `ops`."""
    site = {"component": component, **({} if layer is None else {"layers": layer})}
    return {"site": site, "pos": -1, **({"ops": ops} if ops else {})}


def gpt2(entries: dict[str, dict]) -> Plan:
    """Tiny GPT-2 (five blocks) over gpt2_reach's four pairs: `early`, a forward with `entries`, then `whole`, the
    same forward reading the head as well, which leaves nothing after its deepest site; the read saved from both,
    and the head's. (An entry nothing saves or uses is not read, so it would not hold the forward up.)"""
    def change(d):
        d["steps"] = {"early": {"kind": "forward", "input": "pairs.input", "interventions": entries},
                      "whole": {"kind": "forward", "input": "pairs.input",
                                "interventions": {**entries, "logits": at("lm_head_output")}}}
        d["saves"] = ["early.read", "whole.read", "whole.logits"]
    return edited("gpt2_reach", change).compile()


ZERO = [{"op": "swap", "with": 0.0}]
# Each case: its entries, what of the forward runs per batch, and what runs after it (a `readout` runs the final
# norm and the head itself, on the value read).
CASES = {
    "a read at block 2": ({"read": at("layer_output", 2)}, through(3), []),
    "a write at block 1, then a read at block 3": ({"zero": at("layer_output", 1, ZERO),
                                                    "read": at("layer_output", 3)}, through(4), []),
    "a readout of a read at block 2": ({"read": at("layer_output", 2, [{"op": "readout"}])}, through(3), ["N", "H"]),
    "a read of block 2's attention, before its MLP": ({"read": at("attention_output", 2)}, through(2) + [2], []),
    "a read of the embeddings": ({"read": at("token_embeddings")}, [], []),
    "a read of the stream entering block 0": ({"read": at("layer_input", 0)}, [0], []),
    "a read of the final norm": ({"read": at("final_norm_output")}, through(5) + ["N"], []),
    "a read of the logits": ({"read": at("logits")}, WHOLE, []),
}
@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("case", CASES)
def test_a_forward_stops_right_after_its_deepest_site_and_reads_what_the_whole_one_does(engines, kind, case):
    """Each of the step's two batches runs up to its deepest site and no further (block 2 without its MLP after its
    attention, no block for the embeddings, block 0 without its body for the stream entering it, no head for the
    final norm), and the one that also reads the head runs whole. The values are the whole forward's to the bit: a write before the stop lands, and
    a `readout` runs the final norm and the head on the value read."""
    entries, ran_early, after = CASES[case]
    plan = gpt2(entries)
    filled, ran = counted(engines(kind, plan.model), plan)
    assert ran == (ran_early + after) * 2 + (WHOLE + after) * 2
    assert same(filled.results["early.read"], filled.results["whole.read"])


def test_the_write_before_the_stop_changes_the_read_after_it(engines):
    plan = gpt2(CASES["a write at block 1, then a read at block 3"][0])
    clean = gpt2({"read": at("layer_output", 3)})
    engine = engines("nnterp", plan.model)
    written, unwritten = (engine.execute(p, batch_size=BATCH_SIZE).results["early.read"] for p in (plan, clean))
    assert not torch.equal(written, unwritten)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_generation_runs_whole_at_every_call(engines, kind):
    """generated_answer reading block 0 in the prefill, on the tiny Llama (two blocks): its twelve calls (two
    batches of six) each run both blocks, the final norm and the head."""
    def change(d):
        d["steps"]["answer"]["interventions"] = {"early": at("layer_output", 0)}
        d["steps"] = {"answer": d["steps"]["answer"]}
        d["saves"] = ["answer.early"]
    plan = edited("generated_answer", change).compile()
    filled, ran = counted(engines(kind, plan.model), plan)
    assert filled.run is not None and filled.run.costs["answer"]["forwards"] == 12
    assert ran == (through(2) + ["N", "H"]) * 12


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_fit_bodys_forward_stops_at_its_own_deepest_site(engines, kind):
    """sae_train's body reads block 0 of the tiny Llama's two: every forward it makes, training and scoring, runs
    block 0 alone."""
    plan = plan_of("sae_train")
    filled, ran = counted(engines(kind, plan.model), plan)
    assert filled.run is not None and filled.run.costs["fit"]["forwards"] == 82 and ran == through(1) * 82


def test_the_ndif_engine_runs_every_forward_whole_with_the_same_values(engines, monkeypatch):
    """Under `remote="local"` and on the stand-in server's model (the nnterp engine's, which the server holds) the
    ndif engine's forwards run whole, the early step's as the whole one's, and what comes home is the nnterp
    engine's, which stops."""
    plan = gpt2(CASES["a write at block 1, then a read at block 3"][0])
    local = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    there, ran = counted(engines("ndif", plan.model, remote="local"), plan)
    assert ran == WHOLE * 4 and same(there.results, local.results)
    with counting(engines("nnterp", plan.model)) as ran:
        came, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP and ran == WHOLE * 4 and same(came.results, local.results)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_stopped_forward_leaves_no_frame_for_the_collector(engines, kind):
    """A forward stopped after block 0 on the tiny Llama: once it returns, nothing unreachable holds a frame, the
    model's (whose locals are its activations) or the run's (nnsight's stop holds both in cycles that the nnterp
    engine breaks; on a GPU, forwards of 16 rows of 512 tokens stopped at Llama-3.2-1B's block 8 ran it out of memory
    in a dozen). The engine has run once, so what it makes once is made."""
    plan = plan_of("sae_train")
    engine = engines(kind, plan.model)
    engine.execute(plan, batch_size=BATCH_SIZE)
    batch = {"input_ids": torch.tensor([[1, 5, 6, 7]] * 2), "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "position_ids": torch.arange(4).repeat(2, 1)}
    read = {"component": "layer_output", "layer": 0, "name": "read", "positions": torch.tensor([[3], [3]]),
            "select": {}}
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        got = engine.forward(batch, [read])
        gc.collect()
        frames = [f"{o.f_code.co_filename}:{o.f_code.co_name}" for o in gc.garbage if isinstance(o, types.FrameType)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.enable()
    assert frames == [] and got["read"].shape == (2, 1, 16)


def test_binding_a_write_leaves_nothing_for_the_collector():
    """`batches.edit` binds patching's swap to one batch from a namespace that nothing else holds once it returns
    (a fit's minibatch namespace, with the featurizers it trains and the values it made, is one): nothing of it,
    and no function, is left in a cycle for the collector."""
    plan = plan_of("patching")
    record = entries(points(plan.steps["patched"])[0])["patch"]

    def bound() -> None:
        state: dict = {}
        put(state, "counterfactual.v_cf", {}, torch.zeros(4, 1, 16))
        edit(plan, record, state, {}, slice(0, 2), {}, 0, {})
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        bound()
        gc.collect()
        left = [type(o).__name__ for o in gc.garbage if isinstance(o, (torch.Tensor, types.FunctionType))]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.enable()
    assert left == []
