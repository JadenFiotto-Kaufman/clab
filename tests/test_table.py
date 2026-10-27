"""`table`: saved values as rows, several joined on the keys they share (their sweep axes by coordinate, their
axis columns), and what has no rows to table."""

import pytest
from conftest import BATCH_SIZE, document

from clab import Plan

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"}
HEAD = {"component": "lm_head_output"}
PAIRS = {"pairs": {"path": "weekdays/train"}}


def plan(**fields) -> Plan:
    """A document on the tiny Llama over the weekday pairs, read as if from the corpus (so `data/` resolves)."""
    return Plan(model=TINY, data=PAIRS, document=document("patching"), **fields).compile()


def run(engines, compiled: Plan) -> Plan:
    return engines("nnterp", compiled.model).execute(compiled, batch_size=BATCH_SIZE)


def logits(pos: int = -1) -> dict:
    return {"site": HEAD, "pos": pos}


def test_values_over_different_sweep_axes_join_on_the_axes_they_share(engines):
    """A value over `src` and `dst` (the patch, a 2x2 grid) beside one over `src` alone (the hit of a read at `src`
    on the same input): the two are joined by `src`, `row` and `pos`, not by their whole coordinates, so each row of
    the grid takes the hit of its `src`, which repeats across `dst`, and the joined `at` holds both coordinates; in
    either order. The counterfactual's own hit is read on another input: beside the grid its `pos` counts other
    prompts' positions, which is refused by name."""
    def site(label: str) -> dict:
        return {"component": "layer_output", "layers": {"sweep": [0, 1], "label": label}}
    filled = run(engines, plan(
        sites={"src": site("src"), "dst": site("dst")},
        steps={"counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                                  "interventions": {"v_cf": {"site": "src", "pos": -1}, "logits": logits()}},
               "cf_hit": {"kind": "op", "op": "match", "of": "counterfactual.logits", "expected": "pairs.cf_answer"},
               "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
                   "patch": {"site": "dst", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
                   "logits": logits()}},
               "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
               "at_src": {"kind": "forward", "input": "pairs.input",
                          "interventions": {"r": {"site": "src", "pos": -1}, "logits": logits()}},
               "src_hit": {"kind": "op", "op": "match", "of": "at_src.logits", "expected": "pairs.cf_answer"}},
        saves=["iia", "cf_hit", "src_hit"]))
    hits = {(t["at"]["src"], t["row"]): t["value"] for t in filled.table("src_hit")}
    assert len(hits) == 8
    for names in (("iia", "src_hit"), ("src_hit", "iia")):
        joined = filled.table(*names)
        assert {(t["at"]["src"], t["at"]["dst"], t["row"]) for t in joined} == {(s, d, r) for s in (0, 1) for d in (0, 1)
                                                                                for r in range(4)}
        assert len(joined) == 16 and list(joined[0])[0] == "at"
        assert all(t["src_hit"] == hits[t["at"]["src"], t["row"]] for t in joined)
        assert all(t["iia"] == filled.results["iia"][t["at"]["src"], t["at"]["dst"]][t["row"], 0] for t in joined)
    with pytest.raises(ValueError, match=r"^'iia' has a pos column counting positions of pairs\.input, and 'cf_hit' one "
                                         r"counting positions of pairs\.counterfactual_inputs\[0\]: the column does not "
                                         r"mean one thing for both, so they do not join; table them one at a time$"):
        filled.table("iia", "cf_hit")


def test_rowless_values_with_a_layers_axis_join_by_layer_and_a_rowed_one_beside_them_too(engines):
    """Two means over everything but `layers` are keyed by `layer`, as `elements` prints it: joined, one row per
    layer with each value's; a value laid out `[layers, rows, pos]` beside one of them takes its layer's number on
    every row."""
    def site(component: str) -> dict:
        return {"component": component, "layers": [0, 1]}
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input",
                     "interventions": {"h": {"site": site("layer_output"), "pos": -1}, "i": {"site": site("layer_input"), "pos": -1}}},
               "m1": {"kind": "op", "op": "mean", "of": "f.h", "over": ["rows", "pos", "hidden_size"]},
               "m2": {"kind": "op", "op": "mean", "of": "f.i", "over": ["rows", "pos", "hidden_size"]},
               "n": {"kind": "op", "op": "mean", "of": "f.h", "over": ["hidden_size"]}}, saves=["m1", "m2", "n"]))
    assert filled.table("m1", "m2") == [{"layer": layer, "m1": filled.results["m1"][layer].item(),
                                         "m2": filled.results["m2"][layer].item()} for layer in (0, 1)]
    joined = filled.table("n", "m1")
    assert [(t["layer"], t["row"]) for t in joined] == [(layer, row) for layer in (0, 1) for row in range(4)]
    assert all(t["m1"] == filled.results["m1"][t["layer"]].item() for t in joined)


def test_a_value_with_fewer_keys_broadcasts_over_the_other_s(engines):
    """As an operand broadcasts over an axis it leaves out: a `mean` over everything, unswept, sits beside every
    row; swept over `L`, each point's number sits beside every row of an unswept value, at that point, in the order
    of the value with more keys (the rows'), whichever name comes first."""
    swept = {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "L"}}
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {"h": {"site": swept, "pos": -1},
                                                                                  "logits": logits()}},
               "hit": {"kind": "op", "op": "match", "of": "f.logits", "expected": "pairs.cf_answer"},
               "acc": {"kind": "op", "op": "mean", "of": "hit"},
               "g": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": logits()}},
               "hit2": {"kind": "op", "op": "match", "of": "g.logits", "expected": "pairs.cf_answer"},
               "acc2": {"kind": "op", "op": "mean", "of": "hit2"}}, saves=["hit", "acc", "hit2", "acc2"]))
    plain = filled.table("hit2", "acc2")
    assert [t["row"] for t in plain] == [0, 1, 2, 3] and {t["acc2"] for t in plain} == {filled.results["acc2"].item()}
    for names in (("acc", "hit2"), ("hit2", "acc")):
        joined = filled.table(*names)
        assert [(t["at"], t["row"]) for t in joined] == [({"L": layer}, row) for row in range(4) for layer in (0, 1)]
        assert all(t["acc"] == filled.results["acc"][t["at"]["L"]].item() for t in joined)


def test_a_shared_token_column_means_one_thing_for_both_or_is_refused(engines):
    """Two reads of one input at different positions share `pos` and join on it, to nothing where they agree
    nowhere; a value with no rows axis prints `pos` as a plain index, which beside a token position is refused by
    name."""
    layer0 = {"component": "layer_output", "layers": 0}
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
                   "a": logits(-1), "b": logits(-2), "h": {"site": layer0, "pos": {"index": "-3:"}}}},
               "v": {"kind": "op", "op": "mean", "of": "f.h", "over": ["hidden_size"]},
               "r": {"kind": "op", "op": "mean", "of": "f.h", "over": ["rows", "hidden_size"]}},
        saves=["f.a", "f.b", "v", "r"]))
    assert filled.table("f.a", "f.b") == []
    assert [t["pos"] for t in filled.table("r")] == [0, 1, 2]
    with pytest.raises(ValueError, match=r"^'v' has a pos column counting positions of pairs\.input, and 'r' one "
                                         r"counting a plain index along it \(a value with no rows axis\): the column "
                                         r"does not mean one thing for both, so they do not join; table them one at a "
                                         r"time$"):
        filled.table("v", "r")


def test_a_value_with_no_rows_axis_and_several_tensors_says_so(engines):
    """A `top_k` over a value with no rows axis is a dict of one list each, and a `pca`'s value a bundle: neither
    has rows to table, and the refusal says which it is."""
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
                   "logits": logits(), "h": {"site": {"component": "layer_output", "layers": 0}, "pos": -1}}},
               "m": {"kind": "op", "op": "mean", "of": "f.logits"},
               "t": {"kind": "op", "op": "top_k", "of": "m", "k": 2},
               "b": {"kind": "op", "op": "pca", "of": "f.h", "k": 2}}, saves=["t", "b"]))
    with pytest.raises(ValueError, match=r"^'t' has no rows to table: it is laid out \['k'\], a top_k over a value with "
                                         r"no rows axis, its ids and probabilities one list each; read it from "
                                         r"plan\.results$"):
        filled.table("t")
    with pytest.raises(ValueError, match=r"^'b' has no rows to table: it is a bundle \(a pca's value\); read it from "
                                         r"plan\.results$"):
        filled.table("b")
