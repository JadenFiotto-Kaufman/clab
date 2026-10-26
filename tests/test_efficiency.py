"""What a run costs, and where a forward ends.

`plan.run.steps` holds every step's and point's seconds, peak memory (null off CUDA), model calls and input
tokens, exact for a forward, a swept step, an op and a fit, and the server's under ndif, which come home in the run
record; `clab show` prints them. (A generation's are pinned in test_generate.py.)

A forward ends after the deepest block its entries are at (`batches.ending`), on every engine, under
`remote="local"` and through the stand-in server: the blocks after it never run (a hook on each block counts the
ones that do), and every value it reads or writes is the full forward's to the bit. A forward that reaches past the
blocks, and a generation, run whole; a fit's body forward ends where its own entries do. A stopped forward leaves
none of its frames (and so none of its activations) for the garbage collector.
"""

import gc
import types
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from conftest import BATCH_SIZE, TRIP, edited, plan_of, same, stand_in

from clab import Plan
from clab.cli import main
from clab.format.sweeps import labelled, points
from clab.steps.batches import PAST, ending

COST = {"seconds", "memory", "forwards", "tokens"}


def lengths(run, name: str) -> int:
    """The tokens an input or a generation recorded in the run record, every row's."""
    return sum(map(len, run.tokens[name]))


def test_every_step_and_point_has_a_cost_and_a_forward_one_call_per_batch(engines):
    """logit_lens: `lens` is swept over two layers, so its cost and its ops' are keyed by coordinate as
    `positions` are; each point of the forward is two batches of two of the four rows, each prompt token fed once;
    an op calls no model; off CUDA no memory is measured."""
    plan = plan_of("logit_lens")
    run = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    assert list(run.steps) == list(plan.steps)
    for name, entry in plan.steps.items():
        costs = run.steps[name]
        assert list(costs) == [labelled(plan, p.at) for p in points(entry)] == ["layers=0", "layers=1"]
        for cost in costs.values():
            assert set(cost) == COST and cost["memory"] is None and cost["seconds"] >= 0
    prompts = lengths(run, "prompts.input")
    assert prompts == 38
    assert [(c["forwards"], c["tokens"]) for c in run.steps["lens"].values()] == [(2, prompts)] * 2
    assert all((c["forwards"], c["tokens"]) == (0, 0) for name in ("p_answer", "top1")
               for c in run.steps[name].values())


def test_a_forward_feeds_the_same_tokens_at_every_batch_size(engines):
    """Padding is not fed: four rows in batches of 1, 2 and 4 make 4, 2 and 1 calls of the same 38 tokens."""
    plan = plan_of("logit_lens")
    engine = engines("hooks", plan.model)
    lens = {size: engine.execute(plan, batch_size=size).run.steps["lens"]["layers=0"] for size in (1, 2, 4)}
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
    assert run.steps["fit"]["forwards"] == epochs * (minibatches * body + 2 * body) + 2 * body + 2 * body == 32
    assert run.steps["fit"]["tokens"] == epochs * 2 * train + train + test


def test_the_servers_costs_come_home_in_the_run_record(engines, monkeypatch):
    """Under ndif the steps run on the server, so their costs are measured there and come home with the run record,
    the same calls and tokens as a local run's, through `remote="local"` and the stand-in server alike."""
    local = engines("nnterp", das().model).execute(das(), batch_size=BATCH_SIZE).run
    there = engines("ndif", das().model, remote="local").execute(das(), batch_size=BATCH_SIZE).run
    came, trip = stand_in(engines, das(), monkeypatch)
    assert trip == TRIP
    assert came.run is not None
    for run in (there, came.run):
        assert list(run.steps) == ["fit"] and set(run.steps["fit"]) == COST and run.steps["fit"]["memory"] is None
        assert {k: run.steps["fit"][k] for k in ("forwards", "tokens")} == \
            {k: local.steps["fit"][k] for k in ("forwards", "tokens")}


def test_show_prints_each_steps_cost(engines, tmp_path, capsys):
    plan = plan_of("logit_lens")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[3].split() == ["step", "at", "seconds", "memory", "forwards", "tokens"]
    rows = [line.split() for line in shown[4:10]]
    assert [(r[0], r[1], r[3], r[4], r[5]) for r in rows] == [
        ("lens", "layers=0", "—", "2", "38"), ("lens", "layers=1", "—", "2", "38"),
        ("p_answer", "layers=0", "—", "0", "0"), ("p_answer", "layers=1", "—", "0", "0"),
        ("top1", "layers=0", "—", "0", "0"), ("top1", "layers=1", "—", "0", "0")]
    assert all(float(r[2]) >= 0 for r in rows)
    assert shown[10].split() == ["result", "step", "dims", "axes", "shape"]


def blocks(engine) -> list[torch.nn.Module]:
    """The model's blocks, in order, as the engine runs them: the hooks engine's native modules, nnterp's layers'."""
    if engine.name == "hooks":
        paths = engine.config["sites"]["layer_output"]["paths"]
        return [engine.model.get_submodule(paths[i]) for i in sorted(paths)]
    return [layer._module for layer in engine.model.layers]


@contextmanager
def counting(engine) -> Iterator[list[int]]:
    """Each block that runs on real tensors while inside, by index, in the order they run: not in nnterp's
    measure of its forward order, an nnsight scan on fake tensors (which a server makes once per request)."""
    ran: list[int] = []

    def noted(i: int):
        def note(module, args, kwargs) -> None:
            if not any(isinstance(v, FakeTensor) for v in (*args, *kwargs.values())):
                ran.append(i)
        return note
    handles = [b.register_forward_pre_hook(noted(i), with_kwargs=True) for i, b in enumerate(blocks(engine))]
    try:
        yield ran
    finally:
        for handle in handles:
            handle.remove()


def counted(engine, plan: Plan) -> tuple[Plan, list[int]]:
    """`plan` run on `engine`, and each block that ran. It runs once first, uncounted, so that a meta shell (ndif
    under `remote="local"`, which loads its weights at its first forward) has the blocks a hook counts."""
    engine.execute(plan, batch_size=BATCH_SIZE)
    with counting(engine) as ran:
        filled = engine.execute(plan, batch_size=BATCH_SIZE)
    return filled, ran


def at(layer: int, ops: list | None = None) -> dict:
    """An entry at the last token of the residual stream leaving block `layer`, with `ops`."""
    return {"site": {"component": "layer_output", "layers": layer}, "pos": -1, **({"ops": ops} if ops else {})}


HEAD = {"site": {"component": "lm_head_output"}, "pos": -1}


def gpt2(entries: dict[str, dict]) -> Plan:
    """Tiny GPT-2 (five blocks) over gpt2_reach's four pairs: `early`, a forward with `entries`, then `whole`, the
    same forward reading the head as well, which runs every block; the read saved from both, and the head's. (An
    entry nothing saves or uses is not read, so it would not hold the forward up.)"""
    def change(d):
        d["steps"] = {"early": {"kind": "forward", "input": "pairs.input", "interventions": entries},
                      "whole": {"kind": "forward", "input": "pairs.input",
                                "interventions": {**entries, "logits": HEAD}}}
        d["saves"] = ["early.read", "whole.read", "whole.logits"]
    return edited("gpt2_reach", change).compile()


CASES = {
    "a read at block 2": ({"read": at(2)}, 2),
    "a write at block 1, then a read at block 3": ({"zero": at(1, [{"op": "swap", "with": 0.0}]), "read": at(3)}, 3),
    "a readout of a read at block 2": ({"read": at(2, [{"op": "readout"}])}, 2),
}
ENGINES = {"nnterp": {}, "hooks": {}, "ndif": {"remote": "local"}}


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("case", CASES)
def test_a_forward_stops_after_its_deepest_block_and_reads_what_the_whole_one_does(engines, kind, case):
    """Each of the step's two batches runs the blocks up to the deepest its entries are at, and no further; the one
    that also reads the head runs all five. The values are the whole forward's to the bit: a write before the stop
    lands, and a `readout` runs the final norm and the head itself, on the value read."""
    entries, deepest = CASES[case]
    plan = gpt2(entries)
    filled, ran = counted(engines(kind, plan.model, **ENGINES[kind]), plan)
    assert ran == [*range(deepest + 1)] * 2 + [*range(5)] * 2
    assert same(filled.results["early.read"], filled.results["whole.read"])


def test_the_write_before_the_stop_changes_the_read_after_it(engines):
    plan = gpt2(CASES["a write at block 1, then a read at block 3"][0])
    clean = gpt2({"read": at(3)})
    engine = engines("nnterp", plan.model)
    written, unwritten = (engine.execute(p, batch_size=BATCH_SIZE).results["early.read"] for p in (plan, clean))
    assert not torch.equal(written, unwritten)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("component", PAST)
def test_a_forward_that_reaches_past_the_blocks_runs_whole(engines, kind, component):
    """The final norm's output, the head's and the logits come after the last block: a forward that reads one
    beside a read at block 2 runs all five blocks."""
    def change(d):
        d["steps"] = {"past": {"kind": "forward", "input": "pairs.input", "interventions": {
            "read": at(2), "past": {"site": {"component": component}, "pos": -1}}}}
        d["saves"] = ["past.read", "past.past"]
    plan = edited("gpt2_reach", change).compile()
    _, ran = counted(engines(kind, plan.model), plan)
    assert ran == [*range(5)] * 2


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_generation_runs_every_block_at_every_call(engines, kind):
    """generated_answer reading block 0 in the prefill, on the tiny Llama (two blocks): its twelve calls (two
    batches of six) each run both blocks."""
    def change(d):
        d["steps"]["answer"]["interventions"] = {"early": at(0)}
        d["steps"] = {"answer": d["steps"]["answer"]}
        d["saves"] = ["answer.early"]
    plan = edited("generated_answer", change).compile()
    filled, ran = counted(engines(kind, plan.model), plan)
    assert filled.run is not None and filled.run.steps["answer"]["forwards"] == 12 and ran == [0, 1] * 12


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_fit_bodys_forward_stops_at_its_own_deepest_block(engines, kind):
    """sae_train's body reads block 0 of the tiny Llama's two: every forward it makes, training and scoring, runs
    block 0 alone."""
    plan = plan_of("sae_train")
    filled, ran = counted(engines(kind, plan.model), plan)
    assert filled.run is not None and ran == [0] * filled.run.steps["fit"]["forwards"] and len(ran) == 82


def test_the_stand_in_server_stops_where_the_client_would(engines, monkeypatch):
    """The server runs the same trace: on its model (the nnterp engine's) the blocks after the deepest never run,
    and what comes home is the local run's."""
    plan = gpt2(CASES["a write at block 1, then a read at block 3"][0])
    local = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    with counting(engines("nnterp", plan.model)) as ran:
        came, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP and ran == [*range(4)] * 2 + [*range(5)] * 2
    assert same(came.results, local.results)


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
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        engine.forward(batch, ending([]))
        gc.collect()
        frames = [f"{o.f_code.co_filename}:{o.f_code.co_name}" for o in gc.garbage if isinstance(o, types.FrameType)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.enable()
    assert frames == []


def record(component: str, layer: int | None) -> dict[str, Any]:
    return {"component": component, "layer": layer, "name": f"{component}@{layer}"}


def test_a_forward_ends_after_the_deepest_block_its_records_are_at():
    """`ending`, the one place it is decided: a stop at the output of the deepest block a record is at, whatever the
    component inside it and the order the records come in; at block 0 when only the embeddings are read, or
    nothing; none when a record is past the blocks."""
    stop = {"component": "layer_output", "layer": 3, "stop": True}
    inside = [record("attention_probabilities", 3), record("layer_output", 1), record("token_embeddings", None)]
    assert ending(inside) == [*inside, stop]
    assert ending([record("token_embeddings", None)])[-1] == {**stop, "layer": 0}
    assert ending([]) == [{**stop, "layer": 0}]
    for component in PAST:
        assert ending([*inside, record(component, None)]) == [*inside, record(component, None)]
