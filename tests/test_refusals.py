"""What cannot be served, or would change a document's meaning, is refused by name before any forward runs."""

import re

import pytest

from clab.plan import PlanError
from conftest import edited, plan_of

# -- the plan and compile ---------------------------------------------------------------------------------


def test_features_need_a_featurizer():
    def change(d):
        d["steps"]["patched"]["writes"]["patch"]["ops"][0]["features"] = [0]
    with pytest.raises(PlanError, match=r"ops\[0\]: features are a featurizer's latents"):
        edited("patching", change)


def test_an_operand_must_have_the_writes_layout():
    def change(d):
        d["steps"]["counterfactual"]["reads"]["logits"] = {"site": {"component": "lm_head_output"}, "pos": -1}
        d["steps"]["patched"]["writes"]["patch"]["ops"][0]["with"] = "counterfactual.logits"
    with pytest.raises(PlanError, match=r"'counterfactual\.logits' is laid out \['rows', 'pos', 'vocab_size'\]; "
                                        r"this write's selection is \['rows', 'pos', 'hidden_size'\]"):
        edited("patching", change).compile()


@pytest.mark.parametrize("where", ["read", "step"])
def test_renormalize_runs_only_inside_a_write(where):
    def change(d):
        if where == "read":
            d["steps"]["counterfactual"]["reads"]["v_cf"]["ops"] = [{"op": "renormalize"}]
        else:
            d["steps"]["iia"] = {"kind": "op", "op": "renormalize", "of": "counterfactual.v_cf"}
    with pytest.raises(PlanError, match="'renormalize' restores the value a site had before its writes"):
        edited("patching", change).compile()


def test_the_generated_frame_is_a_generate_steps():
    def change(d):
        d["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 0}
    with pytest.raises(PlanError, match=r"steps\.patched\.reads\.logits\.pos: the generated frame is a generate step's"):
        edited("patching", change).compile()


def test_readout_projects_only_the_residual_stream():
    def change(d):
        d["sites"]["resid"]["component"] = "mlp_activation"
    with pytest.raises(PlanError, match="readout projects a residual-stream value; mlp_activation is not one"):
        edited("logit_lens", change).compile()


# -- preflight, before any forward ----------------------------------------------------------------------------


def refused(engines, plan, kind: str = "hooks"):
    """Run `plan` on an engine that counts its forwards; return the refusal and the count."""
    engine = engines(kind, plan.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises((ValueError, NotImplementedError)) as caught:
            engine.execute(plan, batch_size=2)
    finally:
        del engine.forward
    return str(caught.value), len(calls)


@pytest.mark.parametrize("name, message", [
    ("random_subspace", "counterfactual.v_cf: op 'encode' lands in M4"),
    ("das", "fit: fit steps land in M4"),
    ("generated_answer", "answer: generate steps land in M3"),
])
def test_unserved_documents_refuse_before_any_forward(engines, name, message):
    text, forwards = refused(engines, plan_of(name))
    assert text.startswith(message) and forwards == 0, text


@pytest.mark.parametrize("name, change, message", [
    ("patching", lambda d: d["sites"]["target"].update(layers=99), "counterfactual.v_cf: layer 99; the model has 2 layers"),
    ("patching", lambda d: d["sites"]["target"].update(units=[999]),
     r"counterfactual.v_cf: units \[999\] outside the model's 16 \(hidden_size\)"),
    ("patching", lambda d: d["steps"]["patched"]["reads"].update(scored={
        "site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "match", "expected": "pairs.input"}]}),
     "patched.scored: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens"),
    ("pca_harvest", lambda d: d["steps"]["harvest"]["reads"]["acts"].update(pos=-1),
     "basis: pca of 4 vectors has no rank-4 basis"),
    ("entity_patch", lambda d: d["data"].update(pairs={"rows": [{
        "input": "If today is Monday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
     r"patched.patch: row 0: variable 'entity' \('Friday'\) is not in its input"),
    ("entity_patch", lambda d: d["data"].update(pairs={"rows": [{
        "input": "Friday or Friday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
     r"patched.patch: row 0: variable 'entity' \('Friday'\) occurs 2 times in the row's input; a scope is one place"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["writes"]["blank"]["pos"]["scope"].update(segment="user"),
     r"blanked.blank: row 0: segment 'user' occurs 2 times in the row's input"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["writes"]["blank"]["pos"]["scope"].update(segment="user[2]"),
     r"blanked.blank: row 0: segment 'user\[2\]' is not in its input"),
    ("patching", lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos=20),
     r"patched.patch: row 0: tokens \[20:21\] do not fit its 10 scope tokens"),
])
def test_what_needs_the_model_is_checked_before_any_forward(engines, name, change, message):
    text, forwards = refused(engines, edited(name, change).compile())
    assert re.match(message, text) and forwards == 0, text


def test_an_operand_of_another_width_is_refused(engines):
    def change(d):
        d["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = -1
    text, forwards = refused(engines, edited("window_patch", change).compile())
    assert text.startswith("patched.patch: operand 'counterfactual.v_cf' is 1 token(s) wide") and forwards == 0, text


def test_a_pattern_from_an_input_of_other_lengths_is_refused(engines):
    def change(d):
        d["data"]["pairs"] = {"path": "weekdays/train"}
    text, forwards = refused(engines, edited("attention_pattern_patching", change).compile(), kind="nnterp")
    assert forwards == 0
    assert text.startswith("patched.look_there: operand 'counterfactual.their_pattern' was read on "
                           "pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there"), text


def test_the_aligned_pattern_dataset_passes_the_check(engines):
    plan = plan_of("attention_pattern_patching")
    engines("nnterp", plan.model).execute(plan, batch_size=2)
