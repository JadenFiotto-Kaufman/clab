"""Every forward-only document runs on each engine that serves it, its numbers pinned per engine at one batch
size, the engines agreeing to the bit; and the numbers that pin what padding, position ids, prompt positions,
scopes, windows, ragged reads and `layers: "all"` mean.

`patching.json` alone cannot pin the padding: at its last position on Llama neither the padding width, the
position ids nor where the prompt starts changes a number. `from_the_start` (an interchange at a position
counted from the start of prompts of different lengths) and `mixed_lengths` (GPT-2's absolute positions on
inline rows of mixed lengths) do.
"""

import json
import os

import pytest
import torch

from clab import Plan
from clab.cli import main
from clab.plan import points
from conftest import BATCH_SIZE, FORWARD, INSIDE, close, document, edited, plain, plan_of, same, snapshot

ROWS = [
    {"input": "If the count is one, next is", "counterfactual_inputs": ["two"], "base_answer": " two", "cf_answer": " three"},
    {"input": "one", "counterfactual_inputs": ["If the count is three and it goes on, next is"],
     "base_answer": " three", "cf_answer": " four"},
    {"input": "four, then", "counterfactual_inputs": ["The count is four"], "base_answer": " one", "cf_answer": " two"},
    {"input": "If the count is two, then the one after it is", "counterfactual_inputs": ["one, two"],
     "base_answer": " four", "cf_answer": " one"},
]


def mixed_lengths(rows: list[dict]) -> Plan:
    """GPT-2 at layer 1, the last token of one prompt swapped into another's, over inline rows of mixed lengths."""
    def change(d):
        d["data"] = {"pairs": {"rows": rows}}
        d["sites"]["target"]["layers"] = 1
        d["steps"]["patched"]["reads"]["resid"] = {"site": "target", "pos": -1}
        d["saves"] = ["logit_diff", "patched.resid", "counterfactual.v_cf"]
    return edited("gpt2_reach", change).compile()


def from_the_start() -> Plan:
    """patching.json at prompt token 3, counted from the start (BOS and padding are not prompt tokens): the
    first token of the day's name, on prompts 11 and 9 tokens long. Tokens 0 to 2 are "If today is" on
    every row, so a count that included BOS or padding would land on a token both prompts share."""
    def change(d):
        d["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = 3
        d["steps"]["patched"]["writes"]["patch"]["pos"] = 3
    return edited("patching", change).compile()


PLANS = {**{name: (lambda name=name: plan_of(name)) for name in FORWARD},
         "from_the_start": from_the_start, "mixed_lengths": lambda: mixed_lengths(ROWS)}
SERVED = [(name, kind) for name in PLANS for kind in ("nnterp", "hooks") if not (kind == "hooks" and name in INSIDE)]


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    out = {}
    for name, kind in SERVED:
        plan = PLANS[name]()
        out[name, kind] = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    return out


@pytest.mark.parametrize("name, kind", SERVED)
def test_numbers_match_the_snapshot(filled, name, kind):
    if name == "patching":
        return                      # test_patching.py pins it
    values = plain(filled[name, kind].results)
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE
    close(values, expected["values"])


@pytest.mark.parametrize("name", [n for n in PLANS if n not in INSIDE])
def test_the_engines_agree_to_the_bit(filled, name):
    assert same(filled[name, "nnterp"].results, filled[name, "hooks"].results)
    assert filled[name, "nnterp"].run.positions == filled[name, "hooks"].run.positions


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_rows_numbers_do_not_depend_on_its_batch_companions(engines, kind):
    """At one batch shape, a row padded to its input's longest row gets the same numbers whichever rows share its
    batch: regrouped from (0, 1), (2, 3) into (0, 2), (1, 3), every row's results are the same to the bit."""
    order = [0, 2, 1, 3]
    together, regrouped = mixed_lengths(ROWS), mixed_lengths([ROWS[i] for i in order])
    engine = engines(kind, together.model)
    a = engine.execute(together, batch_size=BATCH_SIZE).results
    b = engine.execute(regrouped, batch_size=BATCH_SIZE).results
    for key in a:
        assert torch.equal(a[key][order], b[key]), key


def test_the_swapped_vector_is_the_counterfactuals_on_padded_rows(filled):
    for kind in ("nnterp", "hooks"):
        values = filled["mixed_lengths", kind].results
        assert torch.equal(values["patched.resid"], values["counterfactual.v_cf"]), kind


def test_a_window_and_three_single_writes_give_the_same_numbers(filled):
    for kind in ("nnterp", "hooks"):
        assert torch.equal(filled["window_patch", kind].results["logit_diff"],
                           filled["multi_position_patch", kind].results["logit_diff"]), kind


def test_renormalize_restores_the_norm_steering_changed(filled):
    for kind in ("nnterp", "hooks"):
        values = filled["steer_renormalize", kind].results
        before = values["base.resid"].norm(dim=-1)
        torch.testing.assert_close(values["steered_renormed.resid"].norm(dim=-1), before)
        assert not torch.allclose(values["steered.resid"].norm(dim=-1), before)


def test_layers_all_is_the_layer_sweep_stacked(filled):
    """logit_lens_all.json's one value, layer axis first, is logit_lens.json's two points, to the bit."""
    for kind in ("nnterp", "hooks"):
        every, swept = filled["logit_lens_all", kind].results, filled["logit_lens", kind].results
        for key in ("p_answer", "top1"):
            assert every[key].shape == (2, 4, 1)
            assert torch.equal(every[key], torch.stack([swept[key][0], swept[key][1]])), (kind, key)


def test_the_logit_lens_at_the_last_layer_is_the_models_logits(engines):
    def change(d):
        d["sites"]["resid"]["layers"] = 1
        d["steps"]["lens"]["reads"]["model"] = {"site": {"component": "logits"}, "pos": -1}
        d["saves"] = ["lens.logits", "lens.model"]
    plan = edited("logit_lens", change).compile()
    for kind in ("nnterp", "hooks"):
        values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
        torch.testing.assert_close(values["lens.logits"], values["lens.model"], atol=1e-6, rtol=0)


def test_premix_and_head_outputs_are_one_tensor(engines):
    """head_patching.json's claim: under eager attention, `attention_head_outputs` gives `attention_premix`'s
    numbers bit for bit; under sdpa nnterp does not serve it, and the run says so before any forward."""
    def eager(component):
        def change(d):
            d["model"]["attn_implementation"] = "eager"
            d["sites"]["target"]["component"] = component
        return edited("head_patching", change).compile()
    premix, outputs = eager("attention_premix"), eager("attention_head_outputs")
    engine = engines("nnterp", premix.model)
    assert same(engine.execute(premix, batch_size=BATCH_SIZE).results,
                engine.execute(outputs, batch_size=BATCH_SIZE).results)
    plan = edited("head_patching", lambda d: d["sites"]["target"].update(component="attention_head_outputs")).compile()
    with pytest.raises(ValueError, match=r"counterfactual\.v_cf: the nnterp engine does not serve "
                                         r"attention_head_outputs: read inside the eager attention forward"):
        engines("nnterp", plan.model).execute(plan)


def test_a_variable_scope_is_the_rows_own_text(filled):
    """entity_patch.json: the last token of each row's entity, ' Thursday' three tokens and the others one."""
    run = filled["entity_patch", "nnterp"].run
    at = [run.tokens["pairs.input"][r][i] for r, (i,) in enumerate(run.positions["patched.patch"])]
    cf = [run.tokens["pairs.counterfactual_inputs[0]"][r][i] for r, (i,) in enumerate(run.positions["counterfactual.v_cf"])]
    assert at == ["day", "▁Friday", "▁Saturday", "▁Sunday"]
    assert cf == ["▁Saturday", "▁Sunday", "day", "▁Friday"]


def test_a_ragged_read_is_one_entry_per_token(engines, filled):
    """entity_mean_ablation.json's harvest: every token of each row's entity, three for Thursday, one otherwise;
    the mean is over those six vectors, and pca_harvest.json's is over the 34 prompt tokens, BOS not among them."""
    def change(d):
        d["saves"] = ["harvest.acts", "mean"]
    plan = edited("entity_mean_ablation", change).compile()
    values = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    acts = values["harvest.acts"]
    assert acts["values"].shape == (6, 16) and acts["row"].tolist() == [0, 0, 0, 1, 2, 3]
    assert acts["tok"].tolist() == [4, 5, 6, 4, 4, 4]
    assert torch.equal(values["mean"], acts["values"].mean(0))
    harvested = filled["pca_harvest", "nnterp"].run.positions["harvest.acts"]
    assert sum(map(len, harvested)) == 34 and all(0 not in row for row in harvested)
    basis = filled["pca_harvest", "nnterp"].results["basis"]
    torch.testing.assert_close(basis["basis"].T @ basis["basis"], torch.eye(4), atol=1e-5, rtol=0)


def test_a_chat_turn_scope_is_the_turns_own_tokens(filled):
    """chat_turn.json: the last token of the second user turn is the question's own last token, not `[/INST]`;
    `pos: -1` with no scope is the rendered prompt's last token."""
    run = filled["chat_turn", "nnterp"].run
    tokens = run.tokens["chats.chat"]
    assert tokens[0][:4] == ["<s>", "[", "INST", "]"]
    assert [tokens[r][i] for r, (i,) in enumerate(run.positions["blanked.blank"])] == ["▁is"] * 4
    assert [tokens[r][i] for r, (i,) in enumerate(run.positions["blanked.logits"])] == ["]"] * 4


def test_top_k_saves_its_tokens(filled, tmp_path):
    plan = filled["readout", "nnterp"]
    top3 = plan.results["top3"]
    assert top3["ids"].shape == top3["probs"].shape == (4, 1, 3)
    assert len(top3["tokens"][0][0]) == 3 and all(isinstance(t, str) for t in top3["tokens"][0][0])
    plan.dump(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["manifest.json", "p_answer.safetensors", "top3.json"]
    back = Plan.load(str(tmp_path)).results["top3"]
    assert same(back, top3)
    row = plan.table("top3")[0]
    assert row["value"]["tokens"] == top3["tokens"][0][0] and row["pos_token"] == "▁is"


def test_a_swept_value_is_keyed_by_coordinate(filled, tmp_path):
    plan = filled["pos_sweep", "hooks"]
    assert list(plan.results["iia"]) == [-1, -2, -3] and plan.results["iia"][-2].shape == (4, 1)
    plan.dump(str(tmp_path))
    from safetensors import safe_open
    with safe_open(str(tmp_path / "iia.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["pos=-1", "pos=-2", "pos=-3"]
    back = Plan.load(str(tmp_path))
    assert same(back.results, plan.results)
    rows = back.table("iia")
    assert [r["at"] for r in rows] == [{"pos": p} for p in (-1, -2, -3) for _ in range(4)]
    assert {r["pos_token"] for r in rows} == {"▁is"}     # the value's own position, apart from the sweep's coordinate
    heads = filled["head_patching", "hooks"]
    assert list(heads.results["iia"]) == [(0,), (1,), (2,), (3,)]
    heads.dump(str(tmp_path / "heads"))
    with safe_open(str(tmp_path / "heads" / "iia.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["heads=[0]", "heads=[1]", "heads=[2]", "heads=[3]"]


def test_tables_join_wide_and_label_layers_and_keys(filled):
    rows = filled["two_observes", "nnterp"].table("patching_ld", "ablation_ld", "clean_ld")
    assert len(rows) == 4 and {"patching_ld", "ablation_ld", "clean_ld"} <= set(rows[0])
    lens = filled["logit_lens_all", "nnterp"].table("p_answer", "top1")
    assert [(r["layer"], r["row"]) for r in lens] == [(l, r) for l in range(2) for r in range(4)]
    pattern = filled["attention_knockout", "nnterp"].table("clean.pattern")
    first = [r for r in pattern if r["row"] == 1]
    assert [r["key_token"] for r in first][:2] == ["<s>", "▁If"] and len(first) == 9     # padding keys are left out
    assert len(first[0]["value"]) == 4                                                   # one probability per head


def test_a_bundle_carries_the_header_its_featurizer_checks(filled, tmp_path):
    filled["pca_harvest", "hooks"].dump(str(tmp_path))
    from safetensors import safe_open
    with safe_open(str(tmp_path / "basis.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["basis", "mean"]
        assert f.metadata() == {"format": "clab/1", "kind": "pca", "k": "4",
                                "model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                                "model.revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}


def test_the_cli_runs_remote_local(tmp_path, filled):
    out = tmp_path / "steer"
    assert main(["run", document("steer_renormalize"), "--remote", "local", "--batch-size", str(BATCH_SIZE),
                 "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["remote"] == "local"
    assert same(Plan.load(str(out)).results, filled["steer_renormalize", "nnterp"].results)


def test_every_point_has_at():
    for name in FORWARD:
        for entry in plan_of(name).steps.values():
            swept = isinstance(entry, list)
            assert all(bool(p.at) == swept for p in points(entry)), name
