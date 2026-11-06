"""`table`: saved values as rows, several joined on the keys they share (their sweep axes by coordinate, their
axis columns), and what has no rows to table."""

import json

import pytest
from conftest import BATCH_SIZE, ROOT, document, edited, same

from clab import Plan
from clab.plan import entries

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
    """A value over `src` and `dst` (the patch, a 2x2 grid) beside one over `src` alone (the hit of a readout at
    `src` on the same input): the two are joined by `src`, `row` and `pos`, not by their whole coordinates, so each
    row of the grid takes the hit of its `src`, which repeats across `dst`, and the joined `at` holds both
    coordinates; in either order. The counterfactual's own hit is read on another input: beside the grid its `pos`
    counts other prompts' positions, which is refused by name."""
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
                          "interventions": {"r": {"site": "src", "pos": -1, "ops": [{"op": "readout"}]}}},
               "src_hit": {"kind": "op", "op": "match", "of": "at_src.r", "expected": "pairs.cf_answer"}},
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
    """Two means over everything but `layers` are keyed by `layer`, as `value_rows` prints it: joined, one row per
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


def test_a_selected_axis_is_labelled_by_the_index_kept(engines):
    """A mean over everything but a selected `num_heads` axis is keyed by the head kept, as a `layers` axis is by
    the model's layer, and a `top_k` over rows of the selected value names the head kept too; the tensors count
    within the selection."""
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
                   "z": {"site": {"component": "attention_premix", "layers": 0, "select": {"num_heads": [3, 2]}},
                         "pos": -1, "ops": [{"op": "norm", "over": ["head_dim"]}]}}},
               "m": {"kind": "op", "op": "mean", "of": "f.z", "over": ["rows", "pos"]},
               "top": {"kind": "op", "op": "top_k", "of": "f.z", "over": "rows", "k": 1}}, saves=["m", "top"]))
    assert [(t["num_heads"], t["value"]) for t in filled.table("m")] == \
        [(head, filled.results["m"][i].item()) for i, head in enumerate((3, 2))]
    assert [t["num_heads"] for t in filled.table("top")] == [3, 2]


def test_a_value_with_fewer_keys_broadcasts_over_the_other_s(engines):
    """As an operand broadcasts over an axis it leaves out: a `mean` over everything, unswept, sits beside every
    row; swept over `L` (a readout at `L`, scored), each point's number sits beside every row of an unswept value,
    at that point, in the order of the value with more keys (the rows'), whichever name comes first."""
    swept = {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "L"}}
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input",
                     "interventions": {"h": {"site": swept, "pos": -1, "ops": [{"op": "readout"}]}}},
               "hit": {"kind": "op", "op": "match", "of": "f.h", "expected": "pairs.cf_answer"},
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
    has rows to table, and the refusal says which it is. A `top_k` over `rows` (T10's repro) tables a row per pick,
    beside the picked row's columns."""
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
                   "logits": logits(), "h": {"site": {"component": "layer_output", "layers": 0}, "pos": -1}}},
               "m": {"kind": "op", "op": "mean", "of": "f.logits"},
               "t": {"kind": "op", "op": "top_k", "of": "m", "k": 2},
               "b": {"kind": "op", "op": "pca", "of": "f.h", "k": 2},
               "hmax": {"kind": "op", "op": "max", "of": "f.h", "over": ["pos"]},
               "top": {"kind": "op", "op": "top_k", "of": "hmax", "k": 3, "over": "rows"}}, saves=["t", "b", "top", "hmax"]))
    with pytest.raises(ValueError, match=r"^'t' has no rows to table: it is laid out \['k'\], a top_k over a value with "
                                         r"no rows axis, its ids, probabilities and tokens one list each; read it from "
                                         r"plan\.results$"):
        filled.table("t")
    top = filled.results["top"]
    assert [(t["hidden_size"], t["k"], t["row"], t["value"]) for t in filled.table("top")] == \
        [(h, j, top["ids"][h, j].item(), top["values"][h, j].item()) for h in range(16) for j in range(3)]
    assert len(filled.table("hmax")) == 4
    with pytest.raises(ValueError, match=r"^'b' has no rows to table: it is a bundle \(a pca's value\); read it from "
                                         r"plan\.results$"):
        filled.table("b")


def test_a_swept_fits_history_tables_each_points_own_epochs(engines):
    """T8's repro: a fit swept over `epochs` [0, 2] keeps a history per point, of no epochs and of two, and its table
    is a row per epoch of each point, labelled with the point, where it ended in a bare StopIteration."""
    filled = run(engines, plan(
        sites={"target": {"component": "layer_output", "layers": 0}},
        featurizers={"lora": {"kind": "lora", "rank": 1, "alpha": 1}},
        steps={"fit": {"kind": "fit", "train": ["lora"], "objective": {"ce": 1.0}, "epochs": {"sweep": [0, 2]},
                       "batch_size": 2, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01}, "steps": {
                           "adapted": {"kind": "forward", "input": "pairs.input", "interventions": {
                               "adapt": {"site": "target", "pos": -1, "ops": [{"op": "apply", "featurizer": "fit.lora"}]},
                               "logits": logits()}},
                           "ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits",
                                  "target": "pairs.cf_answer"}}}},
        saves=["fit.history.ce"]))
    history = filled.results["fit.history.ce"]
    assert filled.table("fit.history.ce") == [{"at": {"epochs": 2}, "epochs": epoch, "value": history[2][epoch].item()}
                                              for epoch in (0, 1)]


def test_reads_that_resolved_alike_share_one_positions_list_in_the_manifest(engines, tmp_path, monkeypatch):
    """Two reads at one `pos` and a third at another: the manifest holds the two distinct positions lists once
    each, in its `positions`, each read naming its list by index; read back, every read's positions are the ran
    plan's, the two at one `pos` hold one list again, and the table is the ran plan's."""
    def at(layer: int, pos) -> dict:
        return {"site": {"component": "layer_output", "layers": layer}, "pos": pos}
    filled = run(engines, plan(
        steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
            "a": at(0, {"index": "1:"}), "b": at(1, {"index": "1:"}), "c": at(1, -1)}}},
        saves=["f.a", "f.b", "f.c"]))
    filled.dump(str(tmp_path))
    with open(tmp_path / "manifest.json") as f:
        held = json.load(f)
    ran = entries(filled.steps["f"])
    assert {k: e["positions"] for k, e in held["steps"]["f"]["interventions"].items()} == \
        {"a": {"pairs": 0}, "b": {"pairs": 0}, "c": {"pairs": 1}}
    assert held["positions"] == [ran["a"].positions["pairs"], ran["c"].positions["pairs"]]
    monkeypatch.chdir(ROOT)                 # a loaded plan's corpus root is where it is read
    back = Plan.load(str(tmp_path))
    got = entries(back.steps["f"])
    assert all(same(got[k].positions, ran[k].positions) for k in ran)
    assert got["a"].positions["pairs"] is got["b"].positions["pairs"] is not got["c"].positions["pairs"]
    assert back.table("f.a", "f.b") == filled.table("f.a", "f.b") and back.table("f.c") == filled.table("f.c")


def test_a_tables_rows_at_its_corners_are_the_ones_written_out_by_hand(engines):
    """A generation's read over layers 1 then 0 of every token of the word each row said: row 0 said it in one
    token, row 1 in two (row 0 is shorter than the widest), rows 2 and 3 never (`alignment_missing`). The table is a
    row per layer, row and position of the row's own, labelled with the model's layer, the token and the call that
    holds it, and one row, with no value, for each row the read did not reach; each row's keys in this order."""
    def change(d):
        d["steps"] = {"answer": {**d["steps"]["answer"], "interventions": {"said": {
            "site": {"component": "layer_output", "layers": [1, 0]},
            "pos": {"frame": "1:", "scope": {"column": "prompts.said"}}, "ops": [{"op": "sum", "over": ["hidden_size"]}]}}}}
        d["saves"] = ["answer.said"]
    filled = run(engines, edited("generated_answer", change).compile())
    value = filled.results["answer.said"]
    assert value.shape == (2, 4, 2)
    said = [("substr", "If today is Thursday, tomorrow is"), ("energ", "If today is Friday, tomorrow is"),
            ("Friday", "If today is Saturday, tomorrow is"), ("Saturday", "If today is Sunday, tomorrow is")]
    expected = []
    for i, layer in enumerate((1, 0)):
        def row(r: int, **labels) -> dict:
            return {"example_id": said[r][0], "layer": layer, "row": r, **labels, "input": said[r][1]}
        expected += [{**row(0, pos=12, pos_token="▁substr", call=2), "value": value[i, 0, 0].item()},
                     {**row(1, pos=11, pos_token="▁energ", call=3), "value": value[i, 1, 0].item()},
                     {**row(1, pos=12, pos_token="calc", call=4), "value": value[i, 1, 1].item()},
                     {**row(2, alignment_missing=True)}, {**row(3, alignment_missing=True)}]
    assert [list(t.items()) for t in filled.table("answer.said")] == [list(t.items()) for t in expected]


def test_a_top_k_over_rows_and_positions_tables_a_row_per_pick_labelled_as_the_value_it_ranks(engines):
    """T10's max-activating examples: a `top_k` over `["rows", "pos"]` of an SAE's latents at every token after BOS
    tables a row per feature and pick: `features` and `k`, then the picked row with its dataset columns and the
    picked position with its token, through the read the latents are of, then the latent there. Over `rows` alone
    the value keeps `pos`, which it labels the same way, through the picked row. Beside another name it is
    refused: its rows are its picks."""
    def change(d):
        d["steps"]["base"]["interventions"]["latents"]["pos"] = {"index": "1:"}
        d["steps"]["top"] = {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": ["rows", "pos"]}
        d["steps"]["top_rows"] = {"kind": "op", "op": "top_k", "of": "base.latents", "k": 2, "over": "rows"}
        d["saves"] = ["top", "top_rows", "base.latents"]
    filled = run(engines, edited("sae_published_jumprelu", change).compile())
    top, latents = filled.results["top"], filled.results["top"]["values"]
    assert filled.run is not None and filled.data["prompts"].rows is not None
    rows, tokens = filled.data["prompts"].rows, filled.run.tokens["prompts.input"]
    positions = entries(filled.steps["base"])["latents"].positions["prompts"]
    picked = [(f, j, top["ids"]["rows"][f, j].item(), top["ids"]["pos"][f, j].item())
              for f in range(latents.shape[0]) for j in range(3)]
    expected = [{"features": f, "k": j, "row": r, "pos": positions[r][i], "pos_token": tokens[r][positions[r][i]],
                 "input": rows[r]["input"], "value": latents[f, j].item()} for f, j, r, i in picked]
    assert [list(t.items()) for t in filled.table("top")] == [list(t.items()) for t in expected]
    assert len({t["pos_token"] for t in expected}) > 1 and len({t["row"] for t in expected}) > 1
    first = filled.table("top_rows")[0]
    assert list(first) == ["features", "k", "row", "pos", "pos_token", "input", "value"]
    assert first["pos"] == positions[first["row"]][0] and first["pos_token"] == tokens[first["row"]][first["pos"]]
    for names, top in ((("top", "base.latents"), "top"), (("base.latents", "top_rows"), "top_rows")):
        with pytest.raises(ValueError, match=rf"^'{top}' is a top_k over rows, whose rows are its picks, and a top_k "
                                             r"over rows tables alone; table it by itself$"):
            filled.table(*names)


@pytest.mark.parametrize("key", [{"offset": [-1, 0]}, "0:3"])
def test_a_top_k_over_rows_and_keys_labels_each_pick_as_the_patterns_own_table_does(engines, key):
    """Over `["rows", "key"]` of an attention pattern, keys relative to each query or selected, each pick's row is
    the pattern's own table row at the pick: the query position the read resolved to, its token, and the key
    through the read's selection, relative to that query or selected, with its token; its value is that row's
    value at the pick's head."""
    pattern = {"component": "attention_probabilities", "layers": 0, "select": {"key": key}}
    filled = run(engines, Plan(model={**TINY, "attn_implementation": "eager"}, data=PAIRS, document=document("patching"),
                               steps={"f": {"kind": "forward", "input": "pairs.input", "interventions": {
                                          "p": {"site": pattern, "pos": {"index": "1:"}}}},
                                      "top": {"kind": "op", "op": "top_k", "of": "f.p", "k": 5, "over": ["rows", "key"]}},
                               saves=["f.p", "top"]).compile())
    own = {(t["row"], t["pos"], t["key"]): t for t in filled.table("f.p")}
    picks = filled.table("top")
    assert len(picks) == (filled.results["top"]["ids"]["rows"] != -1).sum() > 100       # each pick that picks one
    assert list(picks[0])[:3] == ["num_heads", "k", "row"]
    for t in picks:
        mine = own[t["row"], t["pos"], t["key"]]
        assert (t["pos_token"], t["key_token"], t["value"]) == (mine["pos_token"], mine["key_token"],
                                                               mine["value"][t["num_heads"]])
