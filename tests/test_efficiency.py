"""What a run costs, and where a forward ends.

`plan.run.costs` holds every run of every step: its seconds, peak memory (null off CUDA), model calls, the input
tokens they fed with their padding and the rows' own of them (`real`), and the points it served, exact for a forward, a swept step, an op and a fit, and the server's under ndif,
which come home in the run record; `clab show` prints them. (A generation's are pinned in test_generate.py; the
points one forward serves in test_shared.py.)

A forward stops right after the schedule's stop site, its deepest, on the nnterp and hooks engines: nothing
after it runs (a hook on each block, MLP, the final norm and the head watches what does), and every value it reads
or writes is the whole forward's to the bit. A generation runs whole; a fit's body forward stops where its own
entries do. The ndif engine runs every forward whole, under `remote="local"` and through the stand-in server. A stopped forward leaves none of its frames (and so none of its
activations) for the garbage collector.
"""

import gc
import json
import types
from collections import ChainMap
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from conftest import BATCH_SIZE, TRIP, edited, plan_of, same, stand_in

from clab import Plan
from clab.cli import main
from clab.compile.state import Found
from clab.format.sweeps import labelled, points, put
from clab.plan import entries
from clab.steps.batches import edit

COST = {"seconds", "memory", "forwards", "tokens", "real"}


def lengths(run, name: str) -> int:
    """The tokens an input or a generation recorded in the run record, every row's."""
    return sum(map(len, run.tokens[name]))


def test_every_run_of_a_step_has_a_cost_and_a_forward_one_call_per_batch(engines):
    """logit_lens: `lens` is swept over two layers it only reads at, so its one forward, at the first point, serves
    both (`serves`), and its ops' costs are keyed by coordinate as `positions` are; the forward is two batches of
    two of the four rows, each prompt token fed once and each batch padded to the longest row, 11 tokens; an op
    calls no model; off CUDA no memory is measured."""
    plan = plan_of("logit_lens")
    run = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    assert list(run.costs) == list(plan.steps)
    assert list(run.costs["lens"]) == ["layers=0"] and run.costs["lens"]["layers=0"]["serves"] == ["layers=0", "layers=1"]
    for name in ("p_answer", "top1"):
        assert list(run.costs[name]) == [labelled(plan, p.at) for p in points(plan.steps[name])] == ["layers=0", "layers=1"]
        assert [c["serves"] for c in run.costs[name].values()] == [["layers=0"], ["layers=1"]]
    for costs in run.costs.values():
        for cost in costs.values():
            assert set(cost) == COST | {"serves"} and cost["memory"] is None and cost["seconds"] >= 0
    prompts = lengths(run, "prompts.input")
    assert prompts == 38
    assert [(c["forwards"], c["tokens"], c["real"]) for c in run.costs["lens"].values()] == [(2, 4 * 11, prompts)]
    assert all((c["forwards"], c["tokens"], c["real"]) == (0, 0, 0) for name in ("p_answer", "top1")
               for c in run.costs[name].values())


@pytest.mark.parametrize("padding, fed", [("dataset", {1: 44, 2: 44, 4: 44}), ("batch", {1: 38, 2: 40, 4: 44})])
def test_a_forward_feeds_its_rows_tokens_and_the_padding_at_every_batch_size(engines, padding, fed):
    """Four rows of 11, 9, 9 and 9 tokens in batches of 1, 2 and 4 make 4, 2 and 1 calls of the same 38 tokens of
    their own (`real`), and feed those and their padding (`tokens`): every batch as wide as the input's longest row
    under `dataset` padding, as its own under `batch`."""
    plan = plan_of("logit_lens")
    engine = engines("hooks", plan.model)
    lens = {size: engine.execute(plan, batch_size=size, padding=padding).run.costs["lens"]["layers=0"]
            for size in (1, 2, 4)}
    assert {size: (c["forwards"], c["real"]) for size, c in lens.items()} == {1: (4, 38), 2: (2, 38), 4: (1, 38)}
    assert {size: c["tokens"] for size, c in lens.items()} == fed


def das(epochs: int = 3):
    """das with `epochs` epochs, no early stop, and minibatches of one row: two of its two training rows an
    epoch, each through the body's forward."""
    def change(d):
        fit = d["steps"]["fit"]
        fit.update(epochs=epochs, batch_size=1)
        del fit["early_stop"]
    return edited("das", change).compile()


def test_a_fit_counts_every_forward_its_body_makes(engines):
    """A fit's calls are its body's: per epoch, each minibatch through the body's forward (2 × 1) and the
    training rows scored after it (two rows, a batch of one each: 2); then the `train` eval's two rows once. Each
    pass over a dataset feeds each of its rows' tokens once: two passes over the training prompts an epoch, one of
    them scoring, then one in the eval, each row padded to the longest, 11 tokens; the counterfactual prompts,
    harvested before the fit, are fed there once, in one call."""
    plan = das()
    run = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    train = lengths(run, "train.input")
    epochs, minibatches, body = 3, 2, 1
    assert run.costs["fit"]["forwards"] == epochs * (minibatches * body + 2 * body) + 2 * body == 14
    assert run.costs["fit"]["real"] == epochs * 2 * train + train
    assert run.costs["fit"]["tokens"] == (epochs * 2 + 1) * 2 * 11
    assert {k: run.costs["counterfactual"][k] for k in ("forwards", "tokens", "real")} == \
        {"forwards": 1, "tokens": 2 * 11, "real": lengths(run, "train.counterfactual_inputs[0]")}


def test_the_servers_costs_come_home_in_the_run_record(engines, monkeypatch):
    """Under ndif the steps run on the server, so their costs are measured there and come home with the run record,
    the same calls and tokens as a local run's, through `remote="local"` and the stand-in server alike."""
    local = engines("nnterp", das().model).execute(das(), batch_size=BATCH_SIZE).run
    there = engines("ndif", das().model, remote="local").execute(das(), batch_size=BATCH_SIZE).run
    came, trip = stand_in(engines, das(), monkeypatch)
    assert trip == TRIP
    assert came.run is not None
    for run in (there, came.run):
        assert list(run.costs) == list(das().steps)
        for step, cost in run.costs.items():
            assert set(cost) == COST and cost["memory"] is None
            assert {k: cost[k] for k in ("forwards", "tokens", "real")} == \
                {k: local.costs[step][k] for k in ("forwards", "tokens", "real")}


def test_memory_is_each_cuda_devices_peak_and_a_remote_run_measures_none(engines, monkeypatch):
    """With the model on a CUDA device (stood in for here: torch's calls recorded, the device's peak 7 bytes), each
    step resets its peak and reads it back, by device, and a step that calls the model synchronizes the device at
    both ends of its run, while an op step, which calls none, synchronizes nothing; under the ndif engine, even
    `remote="local"`, the run makes no CUDA call and `memory` is null."""
    called: list[str] = []
    for call in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, call, lambda *_, call=call: called.append(call) or 7)
    monkeypatch.setattr("clab.steps.gpus", lambda engine: ["cuda:0"])
    plan = plan_of("logit_lens")
    local = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).run
    assert all(c["memory"] == {"cuda:0": 7} for costs in local.costs.values() for c in costs.values())
    forward = ["synchronize", "reset_peak_memory_stats", "synchronize", "max_memory_allocated"]
    assert called == forward + ["reset_peak_memory_stats", "max_memory_allocated"] * 4     # two ops at two points
    called.clear()
    remote = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE).run
    assert called == [] and all(c["memory"] is None for costs in remote.costs.values() for c in costs.values())


def test_what_the_model_does_once_is_done_before_the_first_step(monkeypatch):
    """A fresh ndif engine under `remote="local"`: its shell has loaded its weights, and nnterp measured its forward
    order, before the first step's timer starts, so that no step's cost holds either."""
    from nnterp import order

    from clab.engines import load
    from clab.steps import measured
    seen: list[Any] = []
    compute = order.compute
    monkeypatch.setattr(order, "compute", lambda model: seen.append("order") or compute(model))
    plan = plan_of("logit_lens")
    engine = load("ndif", plan.model, remote="local")
    shell: Any = engine.model
    monkeypatch.setattr("clab.steps.measured", lambda devices, cost, call, synced: seen.append(
        ("step", shell.dispatched)) or measured(devices, cost, call, synced))
    engine.execute(plan, batch_size=BATCH_SIZE)
    assert seen == ["order"] + [("step", True)] * 5          # the lens's one forward, then two ops at two points


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("name, expected", [("logit_lens", 2), ("generated_answer", 12)])
def test_every_forward_the_model_runs_is_a_steps(kind, name, expected):
    """Counted at the model itself (a pre-hook on its root module), a fresh engine's first run makes exactly the
    forwards its steps' costs sum to (logit_lens's one forward of two batches, serving its two points,
    generated_answer's two batches of a prefill and five decode calls) plus, on nnterp, the one forward `ready`
    makes before the first step to measure the forward order (`rank`), once per engine; hooks measured it on a meta
    shell at load, so none."""
    from clab.engines import load

    plan = plan_of(name)
    engine: Any = load(kind, plan.model)
    root = engine.model._module if kind == "nnterp" else engine.model
    calls: list[int] = []
    handle = root.register_forward_pre_hook(lambda module, args: calls.append(1))
    try:
        run = engine.execute(plan, batch_size=BATCH_SIZE).run
    finally:
        handle.remove()
    assert run is not None
    counted = sum(c["forwards"] for by in run.costs.values() for c in (by.values() if "forwards" not in by else [by]))
    assert counted == expected and len(calls) == counted + (kind == "nnterp")


def test_show_prints_each_steps_cost(engines, tmp_path, capsys):
    plan = plan_of("logit_lens")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[3].split() == ["step", "at", "serves", "seconds", "memory", "forwards", "tokens", "real"]
    rows = [line.split(maxsplit=2) + line.split()[-5:] for line in shown[4:9]]      # the serves cell has spaces
    assert [(r[0], r[1], r[2].rsplit(maxsplit=5)[0], *r[4:]) for r in rows] == [
        ("lens", "layers=0", "all 2 points", "—", "2", "44", "38"),
        ("p_answer", "layers=0", "—", "—", "0", "0", "0"), ("p_answer", "layers=1", "—", "—", "0", "0", "0"),
        ("top1", "layers=0", "—", "—", "0", "0", "0"), ("top1", "layers=1", "—", "—", "0", "0", "0")]
    assert all(float(r[3]) >= 0 for r in rows)
    assert shown[9].split() == ["entry", "at", "positions"]                 # then where each entry resolved
    assert shown[10].split(maxsplit=1) == ["lens.logits", "all 2 points  [10], [8], [8], [8]"]   # one row, both layers
    assert shown[11].split() == ["result", "step", "dims", "axes", "shape"]
    manifest = tmp_path / "manifest.json"                    # as a run that recorded no costs wrote it
    written = json.loads(manifest.read_text())
    del written["run"]["costs"]
    manifest.write_text(json.dumps(written))
    assert main(["show", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[3].split() == ["entry", "at", "positions"]


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


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_stop_site_is_where_both_engines_end_the_forward(engines, kind, monkeypatch):
    """A schedule whose stop is the first of two sites (written so by hand, since preflight writes the last):
    both engines run block 0 alone (the first batch's; the refusal ends the step) and read nothing past it, and
    the read at block 3 the forward never reached is refused by name, the same on each; `stop` means one thing."""
    from clab.compile import preflight, schedule

    def early(engine, plan, name, step, served, trained):
        schedule.scheduled(engine, plan, name, step, served, trained)
        step.schedule["stop"] = ["layer_output", 0]
    monkeypatch.setattr(preflight, "scheduled", early)

    def change(d):
        d["steps"] = {"lens": {"kind": "forward", "input": "pairs.input",
                               "interventions": {"a": at("layer_output", 0), "b": at("layer_output", 3)}}}
        d["saves"] = ["lens.a", "lens.b"]
    plan = edited("gpt2_reach", change).compile()
    engine = engines(kind, plan.model)
    engine.execute(plan_of("gpt2_reach"), batch_size=BATCH_SIZE)
    with counting(engine) as ran, pytest.raises(RuntimeError, match=rf"^lens: the {kind} engine ended the forward at "
                                                                     r"\['layer_output', 0\] without reading b@3; the "
                                                                     r"schedule's stop site comes after its last read$"):
        engine.execute(plan, batch_size=BATCH_SIZE)
    assert ran == through(1)            # block 0 and its MLP, the stop site's; nothing after


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
    """sae_train with its harvest in the fit's body, the dictionary's encode in its entry, reads block 0 of the tiny
    Llama's two there: every forward the body makes, training and scoring, runs block 0 alone. As sae_train is,
    with the harvest before the fit, the body makes none and the harvest's two run block 0 alone."""
    def inside(d):
        harvest = d["steps"].pop("harvest")
        harvest["interventions"]["acts"]["ops"] = [{"op": "encode", "featurizer": "fit.dictionary"}]
        d["steps"]["fit"]["steps"] = {"harvest": harvest}
    plan = edited("sae_train", inside).compile()
    filled, ran = counted(engines(kind, plan.model), plan)
    assert filled.run is not None and filled.run.costs["fit"]["forwards"] == 82 and ran == through(1) * 82
    filled, ran = counted(engines(kind, plan.model), plan_of("sae_train"))
    assert filled.run is not None and [filled.run.costs[s]["forwards"] for s in ("harvest", "fit")] == [2, 0]
    assert ran == through(1) * 2


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
    model's (whose locals are its activations) or the run's (on a GPU, forwards of 16 rows of 512 tokens stopped at
    Llama-3.2-1B's block 8 that left their frames in cycles ran it out of memory in a dozen; nnsight after 0.8.1 leaves
    none). The engine has run once, so what it makes once is made."""
    plan = plan_of("sae_train")
    engine = engines(kind, plan.model)
    engine.execute(plan, batch_size=BATCH_SIZE)
    batch = {"input_ids": torch.tensor([[1, 5, 6, 7]] * 2), "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "position_ids": torch.arange(4).repeat(2, 1)}
    read = {"component": "layer_output", "layer": 0, "name": "read", "positions": torch.tensor([[3], [3]]),
            "select": {}}
    bound = {"sites": [{"component": "layer_output", "layer": 0, "records": [read]}], "stop": ["layer_output", 0],
             "grad": [], "loop": None, "online": False}
    gc.collect()
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        got = engine.forward(batch, bound)
        gc.collect()
        frames = [f"{o.f_code.co_filename}:{o.f_code.co_name}" for o in gc.garbage if isinstance(o, types.FrameType)]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.enable()
    assert frames == [] and got["read"].shape == (2, 1, 16)


def test_records_alike_are_resolved_checked_and_bound_once(monkeypatch):
    """patching with its swap made at a second layer too and a scale by one at both, every entry at the last token:
    preflight resolves the patched forward's positions once per row and checks the swap's operand once (a write's
    layers change nothing either finds), the engine's sizes are read at load and not during the run, and in each
    batch the records that reach one selection share its tensors, and every write the batch's rows, the swaps one
    copy of their operand at the batch's rows, and the scales one tensor of their number."""
    from clab.compile import positions, preflight
    from clab.engines import load, standard

    def change(d):
        p = d["steps"]["patched"]["interventions"]
        patch, logits = p.pop("patch"), p.pop("logits")
        p.update(patch=patch, again={**patch, "site": {"component": "layer_output", "layers": 1}},
                 scaled={"site": {"component": "layer_output", "layers": [0, 1]}, "pos": -1,
                         "ops": [{"op": "scale", "factor": 1.0}]}, logits=logits)
    plan = edited("patching", change).compile()
    engine: Any = load("nnterp", plan.model)
    placed, checked, sized, bounds = [], [], [], []
    place, check, forward = positions.place, preflight.check_operands, engine.forward
    monkeypatch.setattr(positions, "place", lambda *a: placed.append(a) or place(*a))
    monkeypatch.setattr(preflight, "check_operands", lambda *a: checked.append(a[2]) or check(*a))
    monkeypatch.setattr(standard, "sizes", lambda model: sized.append(model) or {})
    monkeypatch.setattr(engine, "forward", lambda batch, bound: bounds.append(bound) or forward(batch, bound))
    run = engine.execute(plan, batch_size=BATCH_SIZE).run
    assert run is not None and sized == []
    rows = len(run.tokens["pairs.input"])
    assert len(placed) == 2 * rows                  # the counterfactual's read, and the patched forward's four entries
    assert checked == ["patched.patch", "patched.scaled"]
    patched = [b for b in bounds if any("ops" in r for s in b["sites"] for r in s["records"])]
    assert len(patched) == -(-rows // BATCH_SIZE)
    for bound in patched:
        writes = [r for s in bound["sites"] for r in s["records"] if "ops" in r]
        assert [o["op"] for r in writes for o in r["ops"]] == ["swap", "scale", "swap", "scale"]
        assert len({id(r["positions"]) for r in writes}) == len({id(o["rows"]) for r in writes for o in r["ops"]}) == 1
        swaps, scales = ([o["args"][k] for r in writes for o in r["ops"] if o["op"] == op]
                         for op, k in (("swap", "with"), ("scale", "factor")))
        assert swaps[0].untyped_storage().data_ptr() == swaps[1].untyped_storage().data_ptr()     # each a view of it
        assert scales[0] is scales[1] and isinstance(scales[0], torch.Tensor)


def test_binding_a_write_leaves_nothing_for_the_collector():
    """`batches.edit` binds patching's swap to one batch from a namespace that nothing else holds once it returns
    (a fit's minibatch namespace, with the featurizers it trains and the values it made, is one): nothing of it,
    and no function, is left in a cycle for the collector."""
    plan = plan_of("patching")
    record = entries(points(plan.steps["patched"])[0])["patch"]

    def bound() -> None:
        state: ChainMap = ChainMap()
        put(state, "counterfactual.v_cf", {}, torch.zeros(4, 1, 16))
        edit(plan, record, state, {}, [0, 1], Found(), 0, {})
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


def test_clabs_version_is_worked_out_once_per_process(monkeypatch):
    """Every run records clab's version with its commit (`steps.versions`), which a checkout's `git describe`
    gives: a process asks for it once, whatever number of runs it makes."""
    import subprocess

    from clab import steps

    asked: list = []
    run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: asked.append(a) or run(*a, **k))
    steps.version.cache_clear()
    said = {steps.versions()["clab"] for _ in range(3)}
    assert len(said) == 1 and len(asked) <= 1
