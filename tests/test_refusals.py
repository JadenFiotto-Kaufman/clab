"""What cannot be served, or would change a document's meaning, is refused by name before any forward runs."""

import json
import re

import pytest

from clab.compiler import CompileError, compile
from conftest import document, plan_of


def doc_of(name: str) -> dict:
    with open(document(name)) as f:
        return json.load(f)


def compiled(name: str, change) -> dict:
    doc = doc_of(name)
    change(doc)
    return compile(doc, document(name))


# -- compile ----------------------------------------------------------------------


def test_features_need_a_featurizer():
    def change(d):
        d["steps"]["patched"]["writes"]["patch"]["ops"][0]["features"] = [0]
    with pytest.raises(CompileError, match=r"ops\[0\]\.features: features are a featurizer's latents"):
        compiled("patching", change)


def test_an_operand_must_have_the_writes_layout():
    def change(d):
        d["steps"]["counterfactual"]["reads"]["logits"] = {"site": {"component": "lm_head_output"}, "pos": -1}
        d["steps"]["patched"]["writes"]["patch"]["ops"][0]["with"] = "counterfactual.logits"
    with pytest.raises(CompileError, match=r"'counterfactual\.logits' is laid out \['rows', 'pos', 'vocab_size'\]; "
                                           r"this write's selection is \['rows', 'pos', 'hidden_size'\]"):
        compiled("patching", change)


@pytest.mark.parametrize("where", ["read", "step"])
def test_renormalize_runs_only_inside_a_write(where):
    def change(d):
        if where == "read":
            d["steps"]["counterfactual"]["reads"]["v_cf"]["ops"] = [{"op": "renormalize"}]
        else:
            d["steps"]["iia"] = {"kind": "op", "op": "renormalize", "of": "counterfactual.v_cf"}
    with pytest.raises(CompileError, match="'renormalize' restores the value a site had before its writes"):
        compiled("patching", change)


# -- preflight, before any forward ---------------------------------------------------


def refused(engines, plan: dict, kind: str = "hooks"):
    """Run `plan` on an engine that counts its forwards; return the refusal and the count."""
    engine = engines(kind, plan["model"])
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
    ("logit_lens", "lens.logits: op 'readout' lands in M2"),
    ("random_subspace", "counterfactual.v_cf: op 'encode' lands in M4"),
    ("pca_harvest", "basis: op 'pca' lands in M2"),
    ("readout", "top3: op 'top_k' lands in M2"),
    ("logit_lens_all", "lens.logits: layers 'all' lands in M2"),
    ("entity_patch", "counterfactual.v_cf: M2: scoped and ragged positions"),
    ("chat_turn", "blanked: M2: chat inputs"),
    ("das", "fit: fit steps land in M4"),
])
def test_unserved_documents_refuse_before_any_forward(engines, name, message):
    text, forwards = refused(engines, plan_of(name))
    assert text.startswith(message) and forwards == 0, text


@pytest.mark.parametrize("change, message", [
    (lambda d: d["sites"]["target"].update(layers=99), "counterfactual.v_cf: layer 99; the model has 2 layers"),
    (lambda d: d["sites"]["target"].update(units=[999]),
     r"counterfactual.v_cf: units \[999\] outside the model's 16 \(hidden_size\)"),
    (lambda d: d["steps"]["patched"]["reads"].update(scored={
        "site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "match", "expected": "pairs.input"}]}),
     "patched.scored: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens"),
])
def test_sizes_and_token_columns_are_checked_before_any_forward(engines, change, message):
    text, forwards = refused(engines, compiled("patching", change))
    assert re.match(message, text) and forwards == 0, text


def test_an_operand_of_another_width_is_refused(engines):
    def change(d):
        d["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = -1
    text, forwards = refused(engines, compiled("window_patch", change))
    assert text.startswith("patched.patch: operand 'counterfactual.v_cf' is 1 token(s) wide") and forwards == 0, text


def test_a_pattern_from_an_input_of_other_lengths_is_refused(engines):
    def change(d):
        d["data"]["pairs"]["path"] = "weekdays/train"
    plan = compiled("attention_pattern_patching", change)
    text, forwards = refused(engines, plan, kind="nnterp")
    assert forwards == 0
    assert text.startswith("patched.look_there: operand 'counterfactual.their_pattern' was read on "
                           "pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there"), text


def test_the_aligned_pattern_dataset_passes_the_check(engines):
    plan = plan_of("attention_pattern_patching")
    engines("nnterp", plan["model"]).execute(plan, batch_size=2)
