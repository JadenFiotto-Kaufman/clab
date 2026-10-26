"""What a run costs: `plan.run.steps` holds every step's and point's seconds, peak memory (null off CUDA), model
calls and input tokens, exact for a forward, a swept step, an op and a fit, and the server's under ndif, which come
home in the run record; `clab show` prints them. (A generation's are pinned in test_generate.py.)"""

from conftest import BATCH_SIZE, TRIP, edited, plan_of, stand_in

from clab.cli import main
from clab.format.sweeps import labelled, points

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
