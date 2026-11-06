"""A gradient as a value: `grad`, the first op of an entry of a forward, whose value is the gradient of a later
entry's sum at the entry's site. `documents/attribution_patching.json` is the corpus document; these are the
truths it rests on: the gradient is the central difference of the metric, the engines and a server take it to the
bit, a constant's is exactly zero, one after a write is of what it wrote, it moves no number the forward makes, the
model's weights take none, and what cannot be differentiated is refused by name."""

import copy

import pytest
import torch
from conftest import BATCH_SIZE, TRIP, edited, plan_of, same, stand_in

from clab.plan import Forward, Plan, PlanError

NAME = "attribution_patching"
METRIC = {"site": {"component": "lm_head_output"}, "pos": -1,
          "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}


def grad(of: str, *after: dict, site: str | dict = "target") -> dict:
    """An entry at `site`'s last token whose value is the gradient of `of`, then the ops `after`."""
    return {"site": site, "pos": -1, "ops": [{"op": "grad", "of": of}, *after]}


def forward(entries: dict, saves: list[str] | None = None, keep: tuple[str, ...] = ()) -> Plan:
    """attribution_patching.json's `patched` forward with `entries` in place of its own, after the steps `keep`
    names, saving `saves` (every entry's value when None)."""
    def change(d):
        d["steps"] = {**{k: d["steps"][k] for k in keep},
                      "patched": {**d["steps"]["patched"], "interventions": entries}}
        d["saves"] = [f"patched.{k}" for k in entries] if saves is None else saves
    return edited(NAME, change).compile()


def results(engine, plan: Plan) -> dict:
    """A run's results, copied out of the plan the next run fills again."""
    return copy.deepcopy(engine.execute(plan, batch_size=BATCH_SIZE).results)


def test_the_gradient_is_the_central_difference_of_the_metric(engines):
    """d logit_diff / d layer 0's last-token residual, along every element at once (`add`ing a number moves each
    alike, so the derivative is the gradient summed over `hidden_size`), against (metric(+e) - metric(-e)) / 2e:
    at e = 1e-3 in float32 within 1.2e-3 of the largest (it is 9.8e-4, a central difference's truncation), which
    falls as e squared, by 95 from e = 1e-2. The corpus document's own estimate, the derivative along the
    counterfactual's difference, is that central difference to 2e-5 (it is 1.4e-5 at e = 1e-3, float32's rounding
    of the metric over 2e, on values up to 3.9e-3)."""
    plan = forward({"g": grad("metric", {"op": "sum", "over": ["hidden_size"]}), "metric": METRIC})
    engine = engines("hooks", plan.model)
    g = results(engine, plan)["patched.g"].flatten()

    def central(eps: float, along: float | str) -> torch.Tensor:
        def nudged(e: float) -> torch.Tensor:
            write = {"site": "target", "pos": -1, "ops": [{"op": "add", "with": along, "scale": e}]}
            return results(engine, forward({"nudge": write, "metric": METRIC}, ["patched.metric"],
                                           ("counterfactual", "base", "difference")))["patched.metric"].flatten()
        return (nudged(eps) - nudged(-eps)) / (2 * eps)
    error = {eps: (central(eps, 1.0) - g).abs().max().item() for eps in (1e-2, 1e-3)}
    assert error[1e-3] <= 1.2e-3 * g.abs().max().item()
    assert 50 < error[1e-2] / error[1e-3] < 200
    estimate = results(engine, plan_of(NAME))["patched.h"].flatten()
    torch.testing.assert_close(estimate, central(1e-3, "difference"), atol=2e-5, rtol=0)


def test_the_engines_and_a_server_take_the_same_gradient_to_the_bit(engines, monkeypatch):
    """nnterp, hooks, the ndif engine's in-process trip and the stand-in for a server give the corpus document's
    numbers to the bit; the trip ships nothing by value, and the backward is no model call (two forwards of two
    rows each for the gradient's step)."""
    plan = plan_of(NAME)
    runs = {kind: engines(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE)
            for kind, options in (("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"}))}
    runs = {kind: (copy.deepcopy(run.results), run.run) for kind, run in runs.items()}
    served, trip = stand_in(engines, plan_of(NAME), monkeypatch)
    assert trip == TRIP
    for kind, (values, _) in runs.items():
        assert same(values, served.results), kind
    assert served.results["patched.h"].shape == (4, 1) and served.results["estimate"].shape == ()
    for _, record in runs.values():
        assert record is not None and record.costs["patched"]["forwards"] == 2


def test_the_schedule_takes_one_backward_per_value_differentiated():
    """Preflight writes what the forward differentiates into its schedule: the metric's read, and each `grad` read
    it serves, two entries naming it served by one backward."""
    from clab.engines import load

    plan = forward({"a": grad("metric"), "b": grad("metric", site={"component": "layer_output", "layers": 1}),
                    "metric": METRIC})
    load("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    step = plan.steps["patched"]
    assert isinstance(step, Forward) and step.schedule["grad"] == [{"of": ["metric@None"], "wrt": ["a@0", "b@1"]}]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_gradient_of_a_constant_is_exactly_zero(engines, kind):
    """`of` an entry whose ops end in `scale 0`: the logits times zero, whose sum depends on nothing."""
    zero = {"site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "scale", "factor": 0}]}
    plan = forward({"g": grad("zero"), "zero": zero}, ["patched.g"])
    g = results(engines(kind, plan.model), plan)["patched.g"]
    assert g.shape == (4, 1, 16) and torch.equal(g, torch.zeros_like(g))


def test_the_gradient_of_a_value_the_site_does_not_feed_is_exactly_zero(engines, monkeypatch):
    """A layer's attention scores are made from its queries and keys, not its values, so under eager attention their
    sum carries no graph from `attention_values` of the same layer: its gradient there is exactly zero, and no
    backward runs for it, as on a parallel block (Cohere) a layer's MLP output of its attention's."""
    def change(d):
        d["model"]["attn_implementation"] = "eager"
        d["steps"] = {"patched": {**d["steps"]["patched"], "interventions": {
            "g": grad("scores", site={"component": "attention_values", "layers": 0}),
            "scores": {"site": {"component": "attention_scores", "layers": 0}, "pos": -1}}}}
        d["saves"] = ["patched.g"]
    plan = edited(NAME, change).compile()
    engine = engines("nnterp", plan.model)
    backward, taken = torch.autograd.grad, []

    def counted(*args, **kwargs):
        taken.append(args)
        return backward(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", counted)
    g = results(engine, plan)["patched.g"]
    assert not taken and g.shape[0] == 4 and torch.equal(g, torch.zeros_like(g))


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_gradient_after_a_write_is_of_what_it_wrote(engines, kind):
    """At one site, a `grad` listed before a write of `scale 2` is twice the `grad` listed after it, to the bit in
    float32: the forward goes on with the written value, and the chain rule through the write multiplies by 2,
    which rounds nothing. Taking the gradients moves no number the forward makes: the metric is the write's alone."""
    double = {"site": "target", "pos": -1, "ops": [{"op": "scale", "factor": 2}]}
    plan = forward({"before": grad("metric"), "double": double, "after": grad("metric"), "metric": METRIC},
                   ["patched.before", "patched.after", "patched.metric"])
    engine = engines(kind, plan.model)
    got = results(engine, plan)
    assert torch.equal(got["patched.before"], 2 * got["patched.after"])
    assert not torch.equal(got["patched.before"], got["patched.after"])
    written = results(engine, forward({"double": double, "metric": METRIC}, ["patched.metric"]))
    assert torch.equal(got["patched.metric"], written["patched.metric"])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_gradient_over_a_layers_slice_is_each_layers_own_stacked(engines, kind):
    """One backward serves every layer and a head selection at once, and each layer's gradient is the one a forward
    taking it there alone gives, to the bit, wherever the graph starts."""
    every = forward({"g": {"site": {"component": "layer_output", "layers": ":"}, "pos": {"index": ":"},
                           "ops": [{"op": "grad", "of": "metric"}]},
                     "heads": grad("metric", site={"component": "attention_premix", "layers": 1,
                                                   "select": {"num_heads": [1, 3]}}),
                     "metric": METRIC}, ["patched.g", "patched.heads"])
    engine = engines(kind, every.model)
    got = results(engine, every)
    assert got["patched.g"].shape == (2, 4, 11, 16) and got["patched.heads"].shape == (4, 1, 2, 4)
    for layer in (0, 1):
        one = forward({"g": {"site": {"component": "layer_output", "layers": layer}, "pos": {"index": ":"},
                             "ops": [{"op": "grad", "of": "metric"}]}, "metric": METRIC}, ["patched.g"])
        assert same(results(engine, one)["patched.g"], got["patched.g"][layer]), layer


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
def test_no_weight_of_the_model_takes_a_gradient(engines, kind, options):
    """Every engine freezes the weights it holds (the ndif engine's, under `remote="local"`, once its shell has
    loaded them), so a gradient's graph keeps nothing for one."""
    plan = plan_of(NAME)
    engine = engines(kind, plan.model, **options)
    engine.execute(plan, batch_size=BATCH_SIZE)
    assert not any(p.requires_grad for p in engine.model.parameters())


def test_frozen_weights_move_no_value_of_a_fit(engines):
    """das.json trains a rotation through the model: with the weights frozen, as every engine holds them, its every
    result is the one it gives with them requiring grad, to the bit."""
    plan = plan_of("das")
    engine = engines("nnterp", plan.model)
    frozen = results(engine, plan)
    engine.model.requires_grad_(True)
    try:
        thawed = results(engine, plan)
    finally:
        engine.model.requires_grad_(False)
    assert same(frozen, thawed)


# -- what cannot be differentiated is refused by name ----------------------------------------------------------


def in_generate(d):
    d["steps"]["answer"]["interventions"]["g"] = grad("at_last", site="head")


def in_fit(d):
    d["steps"]["fit"]["steps"]["patched"]["interventions"]["g"] = grad("logits")


def as_step(d):
    d["steps"]["g"] = {"kind": "op", "op": "grad", "of": "patched.metric"}


@pytest.mark.parametrize("name, change, message", [
    ("generated_answer", in_generate, r"steps\.answer\.interventions\.g\.ops\[0\]: grad is taken in a forward, not a "
                                      r"generate, .*: save the generation's detokenize step"),
    ("das", in_fit, r"steps\.fit\.steps\.patched\.interventions\.g\.ops\[0\]: grad inside a fit's body is not served"),
    (NAME, as_step, r"steps\.g: grad is no step: it is the first op of an entry of a forward"),
])
def test_a_gradient_outside_a_forwards_entry_is_refused_by_name(name, change, message):
    with pytest.raises(PlanError, match=f"^{message}"):
        edited(name, change).compile()


ENTRY = r"steps\.patched\.interventions\.g\.ops"


@pytest.mark.parametrize("entries, message", [
    ({"g": {"site": "target", "pos": -1, "ops": [{"op": "readout"}, {"op": "grad", "of": "metric"}]}},
     rf"{ENTRY}\[1\]: grad is ops\[1\] here, and it is the first op of its entry"),
    ({"g": {"site": "target", "pos": -1, "ops": [{"op": "scale", "factor": 2}, {"op": "grad", "of": "metric"}]}},
     rf"{ENTRY}\[1\]: grad follows 'scale', an edit, and an entry that takes a gradient writes nothing: write in an "
     r"entry of its own listed before this one at this site"),
    ({"g": grad("nothing")},
     rf"{ENTRY}\[0\]\.of: 'nothing' names no entry of this forward; grad's of is an entry of its own forward, by its "
     r"bare name: g, metric"),
    ({"g": grad("patched.metric")}, rf"{ENTRY}\[0\]\.of: 'patched\.metric' names no entry of this forward"),
    ({"g": grad("h"), "h": grad("metric")},
     rf"{ENTRY}\[0\]\.of: 'h' is a grad entry, whose value is a gradient the forward never computes"),
    ({"g": grad("metric", {"op": "renormalize"})},
     rf"{ENTRY}\[1\]: 'renormalize' restores the value the site had before its writes, and after grad the value is a "
     r"gradient off the model"),
    ({"v": {"site": "target", "pos": -1}, "g": grad("metric", {"op": "scale", "factor": "patched.v"})},
     rf"{ENTRY}\[1\]\.factor: 'patched\.v' is an entry of this forward, and 'scale' here runs once the value is off "
     r"the model"),
    ({"g": grad("metric", {"op": "encode", "featurizer": "rot"}, {"op": "decode", "featurizer": "rot"})},
     rf"{ENTRY}\[2\]: decode closes an encode whose edits write back, and after grad nothing writes back"),
])
def test_a_misplaced_gradient_is_refused_at_compile(entries, message):
    def change(d):
        d["featurizers"] = {"rot": {"kind": "subspace", "k": 2, "seed": 0}}
        d["steps"] = {"patched": {**d["steps"]["patched"], "interventions": {**entries, "metric": METRIC}}}
        d["saves"] = ["patched.g"]
    with pytest.raises(PlanError, match=f"^{message}"):
        edited(NAME, change).compile()


NO_GRADIENT = r"and no gradient passes through it; of names an entry whose value is a differentiable number: a read, or " \
    r"a metric such as logit_diff$"


@pytest.mark.parametrize("ops, message", [
    ([{"op": "top_k", "k": 2}],
     rf"{ENTRY}\[0\]\.of: 'm' runs 'top_k', which gives ids with their probabilities or values as a dict, {NO_GRADIENT}"),
    ([{"op": "argmax"}], rf"{ENTRY}\[0\]\.of: 'm' runs 'argmax', which gives an index, {NO_GRADIENT}"),
    ([{"op": "argmax"}, {"op": "mean", "over": ["pos"]}],
     rf"{ENTRY}\[0\]\.of: 'm' runs 'argmax', which gives an index, {NO_GRADIENT}"),
    ([{"op": "match", "expected": "pairs.base_answer"}],
     rf"{ENTRY}\[0\]\.of: 'm' runs 'match', which gives 0 or 1, {NO_GRADIENT}"),
    ([{"op": "contains", "text": "pairs.base_answer"}],
     rf"{ENTRY}\[0\]\.of: 'm' runs 'contains', which gives 0 or 1, {NO_GRADIENT}"),
    ([{"op": "sign"}], rf"{ENTRY}\[0\]\.of: 'm' runs 'sign', which gives -1, 0 or 1, {NO_GRADIENT}"),
    ([*METRIC["ops"], {"op": "gt", "with": 0}], rf"{ENTRY}\[0\]\.of: 'm' runs 'gt', which gives 0 or 1, {NO_GRADIENT}"),
    ([{"op": "count"}], rf"{ENTRY}\[0\]\.of: 'm' runs 'count', which gives a count, {NO_GRADIENT}"),
    ([{"op": "detokenize"}],
     r"steps\.patched\.interventions\.m\.ops\[0\]: 'detokenize' needs a value with a 'new' axis; this one is "
     r"\['rows', 'pos', 'vocab_size'\]$"),
])
def test_an_of_no_gradient_passes_through_is_refused_at_compile(ops, message):
    """An `of` that runs an op whose value no gradient passes through is refused by name at compile, before any model
    loads, wherever the op is among its ops: `top_k`'s ids and probabilities, a dict, crashed the backward on it
    once the model had run, and `argmax`'s index, `match`'s 0 or 1, `sign`, a comparison and `count` gave an
    all-zero gradient. `detokenize` takes a generation's ids, which no forward's value has, so its own layout refuses
    it first."""
    def change(d):
        d["steps"] = {"patched": {**d["steps"]["patched"], "interventions": {
            "g": grad("m"), "m": {"site": {"component": "lm_head_output"}, "pos": -1, "ops": ops}}}}
        d["saves"] = ["patched.g"]
    with pytest.raises(PlanError, match=f"^{message}"):
        edited(NAME, change).compile()


@pytest.mark.parametrize("kind, ranks", [("nnterp", r"\(0, 11\) against \(2, 3\)"), ("hooks", "11 against 24")])
def test_a_gradient_of_an_earlier_site_is_refused_with_the_ranks(engines, kind, ranks):
    """`of` read where the forward has not yet reached the entry's site has no gradient there; preflight refuses it
    by the engine's ranks, whether or not anything uses the gradient."""
    plan = forward({"g": grad("v", site={"component": "lm_head_output"}), "v": {"site": "target", "pos": -1}}, [])
    with pytest.raises(ValueError, match=rf"^patched\.g: grad's of 'v' is read at layer_output layer 0, which the "
                                         rf"{kind} engine's forward reaches no later than this entry's lm_head_output "
                                         rf"\(rank {ranks}\); of is an entry at a site the forward reaches after "
                                         r"this one$"):
        engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_gradient_of_its_own_site_is_refused(engines, kind):
    plan = forward({"g": grad("v"), "v": {"site": "target", "pos": -1}})
    with pytest.raises(ValueError, match=rf"^patched\.g: grad's of 'v' is read at layer_output layer 0, which the "
                                         rf"{kind} engine's forward reaches no later than this entry's layer_output "
                                         "layer 0"):
        engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
