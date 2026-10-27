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
    either order. The counterfactual's own hit is read on another input, whose prompts tokenize to positions of
    their own: beside the grid its `pos` counts other prompts, which is refused by name."""
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
    with pytest.raises(ValueError, match=r"^'iia' is read on pairs\.input and 'cf_hit' on pairs\.counterfactual_inputs"
                                         r"\[0\], whose rows tokenize to positions of their own, so the two pos columns "
                                         r"count different prompts; table them one at a time$"):
        filled.table("iia", "cf_hit")


def test_a_value_of_one_number_sits_beside_every_row(engines):
    """A `mean` over everything has no key at all: it is the same beside every row of a value with rows."""
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": logits()}},
               "hit": {"kind": "op", "op": "match", "of": "f.logits", "expected": "pairs.cf_answer"},
               "acc": {"kind": "op", "op": "mean", "of": "hit"}}, saves=["hit", "acc"]))
    joined = filled.table("hit", "acc")
    assert [t["row"] for t in joined] == [0, 1, 2, 3] and {t["acc"] for t in joined} == {filled.results["acc"].item()}


def test_two_reads_that_share_pos_but_agree_nowhere_are_refused_rather_than_joined_to_nothing(engines):
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {"a": logits(-1), "b": logits(-2)}}},
        saves=["f.a", "f.b"]))
    with pytest.raises(ValueError, match=r"^'f\.a' and 'f\.b' share row, pos and agree on no row of it \(two reads at "
                                         r"different positions share pos but never the same one\); table them one at a "
                                         r"time$"):
        filled.table("f.a", "f.b")


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
