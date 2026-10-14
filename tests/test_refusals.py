"""What cannot be served, or would change a document's meaning, is refused by name before any forward runs."""

import re

import pytest

from clab.plan import PlanError
from conftest import edited, plan_of

# -- the plan and compile ---------------------------------------------------------------------------------


def test_select_on_an_op_needs_a_featurizer():
    def change(d):
        d["steps"]["patched"]["writes"]["patch"]["ops"][0]["select"] = {"features": [0]}
    with pytest.raises(PlanError, match=r"ops\[0\]: select on an op narrows a featurizer's space"):
        edited("patching", change)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["sites"]["target"].update(select={"heads": [0]}),
     r"sites\.target: layer_output has no axis 'heads' to select; its axes: hidden_size$"),
    (lambda d: d["sites"]["target"].update(component="attention_premix", select={"units": [0]}),
     r"sites\.target: attention_premix has no axis 'units' to select; its axes: num_heads, head_dim$"),
    (lambda d: d["sites"]["target"].update(heads=[0]),
     r"sites\.target: unknown field 'heads'; valid: component, layers, select$"),
])
def test_select_names_an_axis_the_component_publishes(change, message):
    with pytest.raises(PlanError, match=message):
        edited("patching", change)


@pytest.mark.parametrize("where", ["site", "op"])
def test_an_empty_select_is_refused(where):
    def change(d):
        if where == "site":
            d["sites"]["target"]["select"] = {"hidden_size": []}
        else:
            d["steps"]["patched"]["writes"]["patch"]["ops"][0].update(featurizer="rot", select={"features": []})
    with pytest.raises(PlanError, match=r"select (hidden_size|features): an empty list selects nothing"):
        edited("patching", change)


def test_a_write_selects_on_one_axis():
    def change(d):
        d["sites"]["target"].update(component="attention_premix", select={"num_heads": [0], "head_dim": [0]})
    with pytest.raises(PlanError, match=r"writes\.patch\.site\.select: a write selects on one axis"):
        edited("patching", change).compile()


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


def test_a_forward_makes_one_call():
    def change(d):
        d["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": 1, "index": -1}
    with pytest.raises(PlanError, match=r"steps\.patched\.reads\.logits\.pos\.frame: a forward makes one call, the "
                                        r"prefill; frame is 0"):
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


@pytest.mark.parametrize("name, change, message", [
    ("patching", lambda d: d["sites"]["target"].update(layers=99), "counterfactual.v_cf: layer 99; the model has 2 layers"),
    ("patching", lambda d: d["sites"]["target"].update(select={"hidden_size": [999]}),
     r"counterfactual.v_cf: select hidden_size \[999\] outside the model's 16"),
    ("patching", lambda d: d["steps"]["patched"]["reads"].update(scored={
        "site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "match", "expected": "pairs.input"}]}),
     "patched.scored: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens"),
    ("pca_harvest", lambda d: d["steps"]["harvest"]["reads"]["acts"].update(pos=-1),
     "basis: pca of 4 vectors has no rank-4 basis"),
    ("entity_patch", lambda d: d["data"].update(pairs={"rows": [{
        "input": "If today is Monday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
     r"patched.patch: row 0: column 'pairs.entity' is not in its input"),
    ("entity_patch", lambda d: (d["data"].update(pairs={"rows": [{
        "input": "Friday or Friday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
        d["steps"]["patched"]["writes"]["patch"]["pos"]["scope"].update(occurrence=2)),
     r"patched.patch: row 0: column 'pairs.entity' occurs 2 times in the row's input; it has no occurrence 2"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["writes"]["blank"]["pos"]["scope"].update(turn="system"),
     r"blanked.blank: row 0: the row has no 'system' message"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["writes"]["blank"]["pos"]["scope"].update(occurrence=2),
     r"blanked.blank: row 0: turn 'user' occurs 2 times in the row's input; it has no occurrence 2"),
    ("patching", lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"index": [9, -1]},
                                                                           ops=[{"op": "swap", "with": 0.0}]),
     r"patched.patch: row 0: the write's positions hold no token of this row; a write needs one to write"),
    ("chat_turn", lambda d: d["data"].update(chats={"rows": [{"chat": [{"role": "user", "content": "  padded  "}],
                                                             "base_answer": " Friday"}]}),
     r"blanked: chats.chat row 0: message 0 \(user\) '  padded  ' is not where the chat template put the messages"),
    ("patching", lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos=20),
     r"patched.patch: row 0: tokens 20 do not fit its 10 scope tokens"),
])
def test_what_needs_the_model_is_checked_before_any_forward(engines, name, change, message):
    text, forwards = refused(engines, edited(name, change).compile())
    assert re.match(message, text) and forwards == 0, text


def test_a_select_on_key_outside_the_shortest_row_is_refused(engines):
    plan = edited("attention_knockout", lambda d: d["sites"]["pattern"].update(select={"key": [9]})).compile()
    text, forwards = refused(engines, plan, kind="nnterp")
    assert text == "clean.pattern: select key [9] outside the 9 tokens of the input's shortest row" and forwards == 0


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


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("key, why", [("device", "the engine places its model itself"),
                                      ("device_map", "the engine places its model itself"),
                                      ("dispatch", "the engine places its model itself"),
                                      ("torch_dtype", "the dtype is spelled `dtype`")])
def test_a_loader_refuses_a_model_key_it_cannot_honor(kind, key, why):
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update({key: "cpu"}))
    with pytest.raises(ValueError, match=f"^model.{key}: the {kind} engine cannot honor it: {why}$"):
        ENGINES[kind](plan.model)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_key_the_loader_rejects_is_reported_as_that_key(kind):
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update(attn_implementation="nonsense"))
    with pytest.raises(ValueError, match=f"^model.attn_implementation: the {kind} engine's loader rejects it: "):
        ENGINES[kind](plan.model)


def test_the_model_block_passes_its_other_keys_through_and_spells_dtype_as_torch():
    plan = plan_of("attention_knockout")
    assert plan.model.dtype == "float32" and plan.model.model_extra == {"attn_implementation": "eager"}
    with pytest.raises(PlanError, match=r"model\.dtype: Input should be 'float32', 'float16' or 'bfloat16'"):
        edited("patching", lambda d: d["model"].update(dtype="fp32"))
