"""A fit's constant body steps: compile decides which they are from the document alone, their model calls run
without gradients, a pass over the same rows in the same order reuses an earlier pass's calls instead of making
them, what is kept goes once no pass will reuse it, and the fit is the same to the bit as one that makes every
call; and what that reuse rests on, that the same rows in the same order give the same bits.
"""

import json
import os

import pytest
import torch
from conftest import BATCH_SIZE, ROOT, edited, plan_of, same
from huggingface_hub import try_to_load_from_cache

from clab import Plan
from clab.format.sweeps import lookup, points

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"}
GPT2 = {"key": "hf-internal-testing/tiny-random-gpt2", "revision": "71034c5d8bde858ff824298bdedc65515b97d2b9",
        "dtype": "float32"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Saturday"]


def probe(batch_size: int = 4, epochs: int = 30) -> Plan:
    """test_fit.py's linear probe of layer 1's output: weekday or weekend."""
    rows = [{"input": f"If today is {d}", "label": "weekend" if d in ("Saturday", "Sunday") else "weekday"}
            for d in DAYS]
    scores = {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
              "ops": [{"op": "encode", "featurizer": "fit.probe"}]}
    return Plan(model=TINY, data={"days": {"rows": rows}},
                featurizers={"probe": {"kind": "linear", "classes": "days.label"}},
                steps={"fit": {"kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": epochs,
                               "batch_size": batch_size, "seed": 0, "optimizer": {"name": "adam", "lr": 0.5},
                               "eval": {"train": {}},
                               "steps": {"read": {"kind": "forward", "input": "days.input",
                                                  "interventions": {"scores": scores}},
                                         "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores",
                                                "target": "days.label"},
                                         "hit": {"kind": "op", "op": "match", "of": "read.scores",
                                                 "expected": "days.label"}}}},
                saves=["fit.history.ce", "fit.history.hit", "fit.train.hit", "fit.train.read.scores",
                       "fit.probe"]).compile()


def every_member(d: dict) -> None:
    """A fit document that saves every member its body has: each op's history and each eval's ops and entries."""
    fit = d["steps"]["fit"]
    for n, s in fit["steps"].items():
        listed = s.get("interventions", {})
        named = [k for item in (listed if isinstance(listed, list) else [listed])
                 for k in (d["interventions"][item] if isinstance(item, str) else item)]
        members = [n] if s["kind"] == "op" else [f"{n}.{k}" for k in named]
        d["saves"] += [f"fit.history.{n}"] * (s["kind"] == "op") + [f"fit.{e}.{m}" for e in fit["eval"]
                                                                    for m in members]
    d["saves"] = sorted(set(d["saves"]))


def reached(d: dict) -> None:
    """das with a body that shows the rule: `clean` reads the model alone, `drift` is an op over it, `steered`
    writes `drift`, `lp` scores `clean`, all constant; `cf_mean` is over the counterfactual's coordinates in the
    rotation being trained, `late` is read after the patch, so `late_mean` and `shifted`, which writes it, are
    not."""
    body = d["steps"]["fit"]["steps"]
    target, head = {"component": "layer_output", "layers": 0}, {"component": "lm_head_output"}
    d["steps"]["fit"]["steps"] = {
        "clean": {"kind": "forward", "input": "train.input", "interventions": {
            "h": {"site": target, "pos": -1}, "clean_logits": {"site": head, "pos": -1}}},
        "drift": {"kind": "op", "op": "mean", "of": "clean.h"},
        "steered": {"kind": "forward", "input": "train.input", "interventions": {
            "push": {"site": target, "pos": -1, "ops": [{"op": "add", "with": "drift", "scale": 0.5}]},
            "steered_logits": {"site": head, "pos": -1}}},
        "lp": {"kind": "op", "op": "token_prob", "of": "clean.clean_logits", "token": "train.cf_answer"},
        "counterfactual": body["counterfactual"],
        "cf_mean": {"kind": "op", "op": "mean", "of": "counterfactual.v_cf"},
        "patched": {"kind": "forward", "input": "train.input", "interventions": [
            "das", {"late": {"site": {"component": "layer_output", "layers": 1}, "pos": -1}}]},
        "late_mean": {"kind": "op", "op": "mean", "of": "patched.late"},
        "shifted": {"kind": "forward", "input": "train.input", "interventions": {
            "push": {"site": target, "pos": -1, "ops": [{"op": "add", "with": "late_mean"}]},
            "shifted_logits": {"site": head, "pos": -1}}},
        "steered_lp": {"kind": "op", "op": "token_prob", "of": "steered.steered_logits", "token": "train.cf_answer"},
        "shifted_lp": {"kind": "op", "op": "token_prob", "of": "shifted.shifted_logits", "token": "train.cf_answer"},
        "iia": body["iia"], "ce": body["ce"]}


def recomputed(plan: Plan) -> Plan:
    """`plan` with no fit step listed as constant: every pass makes every model call, with gradients where it trains,
    as before the cache."""
    plan = plan.model_copy(deep=True)
    for entry in plan.steps.values():
        for point in points(entry):
            if point.kind == "fit":
                point.constant = []
    return plan


def constants(plan: Plan) -> list[list[str]]:
    return [p.constant for p in points(plan.steps["fit"])]


# -- which steps are constant -----------------------------------------------------------------------------------


@pytest.mark.parametrize("make, expected", [
    (lambda: plan_of("das"), [["counterfactual"]]),
    (lambda: plan_of("dbm"), [["counterfactual"]] * 4),
    (lambda: plan_of("sae_train"), [["harvest"]]),
    (lambda: plan_of("lora_train"), [[]]),
    (lambda: plan_of("real/das"), [["counterfactual"]] * 4),
    (probe, [["read"]]),
])
def test_compile_lists_the_body_steps_nothing_the_fit_trains_reaches(make, expected):
    """das's and dbm's counterfactual is read through the trained featurizer, but only by the op after its read
    (`encode`), so the model's calls are constant; the patched forward writes through it, and the ops over its
    logits use what it wrote. sae_train's harvest and a probe's read are the same: the model alone, the trained
    featurizer after. lora_train's one forward writes through its adapter: nothing is constant."""
    assert constants(make()) == expected


def test_a_value_the_trained_featurizer_reaches_makes_whatever_uses_it_reached_and_an_op_over_constants_is_constant():
    plan = edited("das", reached).compile()
    assert constants(plan) == [["clean", "drift", "steered", "lp", "counterfactual", "steered_lp"]]


def test_the_manifest_lists_them_and_loads_them_back(engines, tmp_path):
    plan = plan_of("das")
    engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    with open(tmp_path / "manifest.json") as f:
        assert json.load(f)["steps"]["fit"]["constant"] == ["counterfactual"]
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)              # a loaded plan reads its rows again from the corpus root
        assert constants(Plan.load(str(tmp_path))) == [["counterfactual"]]
    finally:
        os.chdir(cwd)


# -- the model calls they make --------------------------------------------------------------------------------


def calls(engine, plan: Plan, monkeypatch) -> tuple[list[tuple[str, int, bool]], list[str], Plan]:
    """`plan` run on `engine`: each model call as (the body step that made it, the pass it is in, whether autograd
    was on), each pass, `train` (a minibatch) or `full` (every row: each scoring pass, then each eval), and the
    filled plan."""
    from clab import steps
    from clab.steps import fit

    made, passes, current = [], [], {}
    body_run, forward, call = fit.body_run, steps.KINDS["forward"], engine.forward

    def run(*args):
        passes.append("full" if args[6] is None else "train")
        return body_run(*args)

    def step(on, plan, name, *rest, **given):
        current["step"] = name
        return forward(on, plan, name, *rest, **given)

    monkeypatch.setattr(fit, "body_run", run)
    monkeypatch.setitem(steps.KINDS, "forward", step)
    engine.forward = lambda *args: made.append((current["step"], len(passes) - 1, torch.is_grad_enabled())) or \
        call(*args)
    try:
        filled = engine.execute(plan, batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    return made, passes, filled


def per_pass(made: list, passes: list[str], step: str) -> list[int]:
    """How many model calls `step` made in each pass."""
    return [sum(1 for s, i, _ in made if s == step and i == p) for p in range(len(passes))]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_das_makes_the_counterfactuals_calls_once_per_rows_in_order_and_patches_every_pass(engines, kind, monkeypatch):
    """das's two training rows are one minibatch an epoch, [0, 1] or [1, 0] as the epoch shuffles them, and every
    scoring pass is [0, 1]: the counterfactual is read once for each of the three ways its rows are run ([0, 1] and
    [1, 0] of `train`, [0, 1] of `test` in the held-out eval; the `train` eval is a scoring pass's rows), without
    gradients. The patched forward writes through the rotation: it calls the model on every pass, as before, with
    gradients while it trains."""
    plan = plan_of("das")
    made, passes, _ = calls(engines(kind, plan.model), plan, monkeypatch)
    assert passes == ["train", "full"] * 10 + ["full", "full"]
    assert per_pass(made, passes, "patched") == [1] * 22
    assert sum(per_pass(made, passes, "counterfactual")) == 3
    assert per_pass(made, passes, "counterfactual")[-2:] == [1, 0]           # held_out reads `test`; train is cached
    assert all(not grad for s, _, grad in made if s == "counterfactual")
    assert [grad for s, i, grad in made if s == "patched"] == [p == "train" for p in passes]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_sae_trains_on_its_harvest_read_once_per_minibatch_order_and_scores_from_the_first_scoring_pass(
        engines, kind, monkeypatch):
    """sae_train's four prompts run as two minibatches of two an epoch: the harvest calls the model once for each
    ordered pair the twenty shuffles draw (11 of the 12), once per batch of the first scoring pass, and never again:
    the other nineteen scoring passes and the `train` eval are that pass's rows."""
    plan = plan_of("sae_train")
    made, passes, _ = calls(engines(kind, plan.model), plan, monkeypatch)
    harvest = per_pass(made, passes, "harvest")
    full = [i for i, p in enumerate(passes) if p == "full"]
    assert len(full) == 21 and [harvest[i] for i in full] == [2] + [0] * 20
    assert sum(harvest) == 11 + 2


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_step_the_trained_featurizer_reaches_calls_the_model_every_pass(engines, kind, monkeypatch):
    """In `reached`'s body the constant forwards call the model once for each way das's rows are run (three) and
    `patched` and `shifted`, which the rotation reaches, on every pass."""
    plan = edited("das", reached).compile()
    made, passes, _ = calls(engines(kind, plan.model), plan, monkeypatch)
    assert {s: sum(per_pass(made, passes, s)) for s in ("clean", "steered", "counterfactual", "patched", "shifted")} \
        == {"clean": 3, "steered": 3, "counterfactual": 3, "patched": 22, "shifted": 22}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_epochs_0_scores_each_eval_once(engines, kind, monkeypatch):
    """With nothing trained, each eval runs once and each of its steps calls the model, as before: an eval of the
    training rows that no other pass runs keeps nothing."""
    plan = edited("das", lambda d: d["steps"]["fit"].update(epochs=0)).compile()
    made, passes, _ = calls(engines(kind, plan.model), plan, monkeypatch)
    assert passes == ["full", "full"]
    assert per_pass(made, passes, "counterfactual") == per_pass(made, passes, "patched") == [1, 1]


def test_each_eval_substitution_reads_its_own_rows(engines, monkeypatch):
    """Two evals that both read `train` as `test` run the same rows: the second reuses the first's calls; the
    scoring passes' calls, of `train`, serve neither."""
    def twice(d):
        d["steps"]["fit"]["eval"] = {"a": {"train": "test"}, "b": {"train": "test"}}
        d["saves"] = ["fit.a.iia", "fit.b.iia", "fit.b.counterfactual.v_cf", "fit.rot"]
    plan = edited("das", twice).compile()
    made, passes, filled = calls(engines("hooks", plan.model), plan, monkeypatch)
    assert per_pass(made, passes, "counterfactual")[-2:] == [1, 0]
    reference = engines("hooks", plan.model).execute(recomputed(plan), batch_size=BATCH_SIZE)
    assert same(filled.results, reference.results)


def test_after_an_early_stop_only_what_an_eval_reads_stays_kept(engines, monkeypatch):
    """das stopped after a few of its twelve epochs (patience 1): the minibatch [1, 0] it drew would have run again
    in an epoch it skipped, so it was kept; once training ends only the `train` eval's rows, a scoring pass's, are."""
    from clab.steps import fit

    def change(d):
        d["steps"]["fit"].update(epochs=12, early_stop={"metric": "iia", "mode": "max", "patience": 1})
        d["steps"]["fit"]["optimizer"]["lr"] = 0.5
    plan = edited("das", change).compile()
    held, body_run = [], fit.body_run

    def run(*args):
        held.append(sorted(rows for _, (_, rows) in args[8]["calls"]))
        return body_run(*args)
    monkeypatch.setattr(fit, "body_run", run)
    engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    assert len(held) < 2 * 12 + 2 and [(0, 1), (1, 0)] in held[:-2]
    assert held[-2:] == [[(0, 1)], [(0, 1)]]                # entering held_out, then train


# -- the same bits ----------------------------------------------------------------------------------------------


PLANS = {
    "das": lambda: edited("das", every_member).compile(),
    "das_batch_size_1": lambda: edited("das", lambda d: (d["steps"]["fit"].update(batch_size=1), every_member(d))
                                       ).compile(),
    "dbm": lambda: edited("dbm", every_member).compile(),
    "sae_train": lambda: edited("sae_train", every_member).compile(),
    "sae_train_batch_size_3": lambda: edited("sae_train", lambda d: (d["steps"]["fit"].update(batch_size=3, epochs=4),
                                                                     every_member(d))).compile(),
    "lora_train": lambda: edited("lora_train", every_member).compile(),
    "probe": probe,
    "probe_batch_size_3": lambda: probe(batch_size=3, epochs=5),             # a trailing minibatch of two
    "reached": lambda: edited("das", lambda d: (reached(d), every_member(d))).compile(),
}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("name", PLANS)
def test_a_fit_reusing_its_constant_calls_is_the_one_that_makes_every_call_to_the_bit(engines, name, kind):
    """Every member each fit has, its histories included, against the same fit with no step constant."""
    plan = PLANS[name]()
    engine = engines(kind, plan.model)
    cached = engine.execute(plan, batch_size=BATCH_SIZE)
    assert same(cached.results, engine.execute(recomputed(plan), batch_size=BATCH_SIZE).results)


@pytest.mark.parametrize("name", ["das", "sae_train"])
def test_remote_local_reuses_them_to_the_same_bits(engines, name):
    plan = PLANS[name]()
    there = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    assert same(there.results, engines("nnterp", plan.model).execute(recomputed(plan), batch_size=BATCH_SIZE).results)


def test_what_a_constant_call_gives_never_needs_a_gradient_and_the_trained_tensors_still_do(engines, monkeypatch):
    """In each training pass of das the counterfactual's reads come off the model as constants, its coordinates in
    the rotation being trained carry the gradient to the rotation's own tensor, a leaf that requires it, and the
    model's weights get none."""
    from clab.steps import fit

    seen = []
    body_run = fit.body_run

    def run(engine, name, step, state, trained, source, index, *rest):
        body = body_run(engine, name, step, state, trained, source, index, *rest)
        if index is not None:
            rot = trained["rot"]
            seen.append((lookup(body, "counterfactual.v_cf", {}).grad_fn is not None,
                         rot.theta.is_leaf and rot.theta.requires_grad))
        return body
    monkeypatch.setattr(fit, "body_run", run)
    plan = plan_of("das")
    engine = engines("hooks", plan.model)
    reads = []
    call = engine.forward
    engine.forward = lambda *args: reads.append(call(*args)) or reads[-1]
    try:
        engine.execute(plan, batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    assert seen == [(True, True)] * 10
    v_cf = [got["v_cf@0"] for got in reads if "v_cf@0" in got]
    assert v_cf and not any(v.requires_grad for v in v_cf)
    assert all(p.grad is None for p in engine.model.parameters())


# -- what the reuse rests on ---------------------------------------------------------------------------------------


with open(os.path.join(ROOT, "data", "weekdays_full", "data.json")) as f:
    ROWS = [{"input": r["input"]} for r in json.load(f)][:12]
ROWS += [{"input": r["input"] + " and then"} for r in ROWS[:4]]                 # prompts of two lengths
LLAMA = {"key": "meta-llama/Llama-3.2-1B", "revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08"}
ON_GPU = torch.cuda.is_available() and isinstance(
    try_to_load_from_cache(LLAMA["key"], "model.safetensors", revision=LLAMA["revision"]), str)
GPU = pytest.mark.skipif(not ON_GPU, reason=f"needs a CUDA device and {LLAMA['key']} in the local Hugging Face cache")


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("model, device", [(TINY, "cpu"),
                                           pytest.param({**LLAMA, "dtype": "float32"}, "cuda", marks=GPU),
                                           pytest.param({**LLAMA, "dtype": "bfloat16"}, "cuda", marks=GPU)])
def test_the_same_rows_in_the_same_order_give_the_same_bits(engines, kind, model, device):
    """What a fit's reuse rests on: a pass over the rows an earlier pass ran, in the same order and batches, gets
    that pass's bits, every residual stream and logit. Nothing looser is reused (`fit.planned` says why)."""
    plan = Plan(model=model, data={"d": {"rows": ROWS}},
                steps={"f": {"kind": "forward", "input": "d.input", "interventions": {
                    "h": {"site": {"component": "layer_output", "layers": ":"}, "pos": {"index": ":"}},
                    "logits": {"site": {"component": "lm_head_output"}, "pos": {"index": ":"}}}}},
                saves=["f.h", "f.logits"]).compile()
    engine = engines(kind, plan.model, device=device)
    assert same(engine.execute(plan, batch_size=2).results, engine.execute(plan, batch_size=2).results)
