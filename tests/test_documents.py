"""Every forward-only document runs on each engine that serves it, its numbers pinned per engine at one batch
size, the engines agreeing to the bit; and the numbers that pin what padding, position ids, prompt positions,
scopes, windows, ragged reads and a `layers` slice mean.

`patching.json` alone cannot pin the padding: at its last position on Llama neither the padding width, the
position ids nor where the prompt starts changes a number. `from_the_start` (an interchange at a position
counted from the start of prompts of different lengths) and `mixed_lengths` (GPT-2's absolute positions on
inline rows of mixed lengths) do.
"""

import json
import os

import pytest
import torch
from conftest import (BATCH_SIZE, FORWARD, GEMMA2, INSIDE, THREADS, close, document, edited, on_each, placed, plain,
                      plan_of, same, snapshot)

from clab import Plan
from clab.cli import main
from clab.format.sweeps import points
from clab.ops import edits
from clab.plan import PlanError

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
        d["steps"]["patched"]["interventions"]["resid"] = {"site": "target", "pos": -1}
        d["saves"] = ["logit_diff", "patched.resid", "counterfactual.v_cf"]
    return edited("gpt2_reach", change).compile()


def from_the_start() -> Plan:
    """patching.json at position 4, counted from the row's start (BOS is position 0; padding is never a
    position): the first token of the day's name, on prompts 11 and 9 tokens long. Positions 1 to 3 are "If
    today is" on every row, so a count that started in the padding, or after BOS, would land on another token."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = 4
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = 4
    return edited("patching", change).compile()


def softcapped_lens() -> Plan:
    """logit_lens_all on the tiny Gemma-2, the one tiny model that softcaps its logits, so that an engine whose
    readout has no cap disagrees with one whose readout has it."""
    return edited("logit_lens_all", lambda d: d.update(model=GEMMA2)).compile()


PLANS = {**{name: (lambda name=name: plan_of(name)) for name in FORWARD},
         "from_the_start": from_the_start, "mixed_lengths": lambda: mixed_lengths(ROWS),
         "softcapped_lens": softcapped_lens}
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
    assert placed(filled[name, "nnterp"]) == placed(filled[name, "hooks"])


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


def test_the_suite_runs_at_the_thread_count_the_snapshots_were_made_with():
    """conftest pins torch's intra-op threads, whatever OMP_NUM_THREADS or the host's cores say: at 4 threads or
    fewer the BLAS splits a matmul's reduction otherwise, and the test above fails by about 1e-8."""
    assert torch.get_num_threads() == THREADS == 12


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
        torch.testing.assert_close(values["steered_renormed.restore"].norm(dim=-1), before)
        assert not torch.allclose(values["steered.steer"].norm(dim=-1), before)


def test_every_layer_is_the_layer_sweep_stacked(filled):
    """logit_lens_all.json's one value, layer axis first, is logit_lens.json's two points, to the bit."""
    for kind in ("nnterp", "hooks"):
        every, swept = filled["logit_lens_all", kind].results, filled["logit_lens", kind].results
        for key in ("p_answer", "top1"):
            assert every[key].shape == (2, 4, 1)
            assert torch.equal(every[key], torch.stack([swept[key][0], swept[key][1]])), (kind, key)


def test_a_layers_selector_picks_and_labels_the_model_layers(engines, filled, tmp_path):
    """A pick list is the layers in the order written, `"-1:"` the last, `-1` the last as one layer's value; the
    run records the model's number of layers, and the table, after `Plan.load` too, says which model layer each
    index of the `layers` axis is."""
    every = filled["logit_lens_all", "nnterp"]
    for layers, picked in (([1, 0], [1, 0]), ("-1:", [1])):
        plan = edited("logit_lens_all", lambda d: d["sites"]["resid"].update(layers=layers)).compile()
        got = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
        assert got.run.sizes["num_layers"] == 2
        assert torch.equal(got.results["p_answer"], every.results["p_answer"][picked])
        assert [r["layer"] for r in got.table("p_answer")] == [l for l in picked for _ in range(4)]
        got.dump(str(tmp_path / str(layers)))
        assert Plan.load(str(tmp_path / str(layers))).table("p_answer") == got.table("p_answer")
    last = edited("logit_lens", lambda d: d["sites"]["resid"].update(layers=-1)).compile()
    got = engines("hooks", last.model).execute(last, batch_size=BATCH_SIZE)
    assert torch.equal(got.results["p_answer"], every.results["p_answer"][1])


def test_layers_outside_the_model_are_refused_by_name(engines):
    for layers, message in ((2, "layer 2; the model has 2 layers"), ([0, -3], "layer -3; the model has 2 layers"),
                            ("5:", "layers '5:' pick none of the model's 2 layers"),
                            ([1, -1], "picks \\[1, -1\\] count one of these 2 twice")):
        plan = edited("logit_lens_all", lambda d: d["sites"]["resid"].update(layers=layers)).compile()
        with pytest.raises(ValueError, match=f"^lens.logits: {message}$"):
            engines("hooks", plan.model).execute(plan)


def test_cosine_is_one_where_the_patch_cannot_reach(filled):
    """cosine_metric.json: at layer 0 (before the write) and at the second-to-last token (before it in the
    sequence) the patched and clean values are the same to the bit, and their cosine is 1 to within rounding;
    the last token at layers 2 and 4 is where the patch shows. Its axes are labelled with the layers picked."""
    for kind in ("nnterp", "hooks"):
        got = filled["cosine_metric", kind]
        cos = got.results["similarity"]
        assert cos.shape == (3, 4, 2) and got.run.sizes["num_layers"] == 5
        torch.testing.assert_close(cos[0], torch.ones(4, 2), atol=2e-7, rtol=0)
        torch.testing.assert_close(cos[:, :, 0], torch.ones(3, 4), atol=2e-7, rtol=0)
        assert (cos[1:, :, 1] < 1 - 1e-4).all()
        assert sorted({r["layer"] for r in got.table("similarity")}) == [0, 2, 4]
        assert [r["pos"] for r in got.table("similarity")][:2] == [10, 11]


def test_clamp_bounds_what_it_writes(filled):
    for kind in ("nnterp", "hooks"):
        values = filled["clamp_write", kind].results
        assert values["clamped.resid"].abs().max() <= 0.01 and values["clean.resid"].abs().max() > 0.01
        assert torch.equal(values["clamped.resid"], values["clean.resid"].clamp(-0.01, 0.01))
        assert not torch.equal(values["clamped_ld"], values["clean_ld"])


def test_noise_is_drawn_per_row_and_position_whatever_the_batch(engines, filled):
    """noise_write.json: `jitter` is the clean value plus 0.5 times each row's draw from a CPU generator seeded by
    (7, row, its index along pos); the write's noise moves the answer. A write of zeros then noise, saved, is
    the noise alone: at each row's position the draw seeded by (7, row, token position), the same at batch
    sizes 1 and 4, in a forward and at each decode call of a generation, never the same twice."""
    for kind in ("nnterp", "hooks"):
        values = filled["noise_write", kind].results
        drawn = torch.stack([torch.randn(1, 16, generator=edits.seeded(7, r, 0)) for r in range(4)])
        assert torch.equal(values["jitter"], values["clean.resid"] + 0.5 * drawn)
        assert not torch.equal(values["noisy_ld"], values["clean_ld"])

    def pure(d, step, frame):
        site = {"component": "layer_output", "layers": 0}
        d["steps"] = {step: d["steps"][step]}
        d["steps"][step]["interventions"] = {"n": {"site": site, "pos": {"frame": frame, "index": ":"}, "ops": [
            {"op": "swap", "with": 0.0}, {"op": "noise", "scale": 1.0, "seed": 7}]}}
        d["saves"] = [f"{step}.n"]
    for name, step, frame in (("noise_write", "noisy", 0), ("generated_answer", "answer", "1:")):
        plan = edited(name, lambda d: pure(d, step, frame)).compile()
        runs = [filled for b in (1, 4) for filled in on_each(engines, plan, batch_size=b).values()]
        assert all(same(r.results, runs[0].results) for r in runs)
        got, where = runs[0].results[f"{step}.n"], placed(runs[0])[f"{step}.n"]
        drawn = []
        for r, places in enumerate(where):
            expected = torch.stack([torch.randn(16, generator=edits.seeded(7, r, t)) for t in places])
            assert torch.equal(got[r, :len(places)], expected) and got[r, len(places):].isnan().all()
            drawn += [tuple(v.tolist()) for v in expected]
        assert len(set(drawn)) == len(drawn)


@pytest.mark.parametrize("name", list(PLANS))
def test_ndif_remote_local_is_nnterp_to_the_bit(engines, filled, name):
    plan = PLANS[name]()
    there = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    assert there.run.options == {"remote": "local"} and same(there.results, filled[name, "nnterp"].results)
    assert placed(there) == placed(filled[name, "nnterp"])


def test_the_logit_lens_at_the_last_layer_is_the_models_logits(engines):
    def change(d):
        d["sites"]["resid"]["layers"] = 1
        d["steps"]["lens"]["interventions"]["model"] = {"site": {"component": "logits"}, "pos": -1}
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


def test_a_column_scope_is_the_rows_own_text(filled):
    """entity_patch.json: the last token of each row's entity, ' Thursday' three tokens and the others one."""
    run, where = filled["entity_patch", "nnterp"].run, placed(filled["entity_patch", "nnterp"])
    at = [run.tokens["pairs.input"][r][i] for r, (i,) in enumerate(where["patched.patch"])]
    cf = [run.tokens["pairs.counterfactual_inputs[0]"][r][i] for r, (i,) in enumerate(where["counterfactual.v_cf"])]
    assert at == ["day", "▁Friday", "▁Saturday", "▁Sunday"]
    assert cf == ["▁Saturday", "▁Sunday", "day", "▁Friday"]


def test_a_read_of_every_token_of_a_scope_is_as_wide_as_the_widest_row(engines, filled):
    """entity_mean_ablation.json's harvest: every token of each row's entity, three for Thursday, one otherwise,
    NaN past a row's own; the mean is over those six vectors, and pca_harvest.json's is over the 34 tokens
    after BOS (`"1:"`)."""
    def change(d):
        d["saves"] = ["harvest.acts", "mean"]
    plan = edited("entity_mean_ablation", change).compile()
    done = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    values, acts = done.results, done.results["harvest.acts"]
    assert placed(done)["harvest.acts"] == [[4, 5, 6], [4], [4], [4]]
    assert acts.shape == (4, 3, 16) and not acts[0].isnan().any() and not acts[1:, 0].isnan().any()
    assert acts[1:, 1:].isnan().all()
    assert torch.equal(values["mean"], acts.nanmean(dim=(0, 1)))
    harvested = placed(filled["pca_harvest", "nnterp"])["harvest.acts"]
    assert sum(map(len, harvested)) == 34 and all(0 not in row for row in harvested)
    basis = filled["pca_harvest", "nnterp"].results["basis"]
    torch.testing.assert_close(basis["basis"].T @ basis["basis"], torch.eye(4), atol=1e-5, rtol=0)


def test_a_chat_turn_scope_is_the_turns_own_tokens(filled):
    """chat_turn.json: the last token of the second user turn is the question's own last token, not `[/INST]`;
    `pos: -1` with no scope is the rendered prompt's last token."""
    run, where = filled["chat_turn", "nnterp"].run, placed(filled["chat_turn", "nnterp"])
    tokens = run.tokens["chats.chat"]
    assert tokens[0][:4] == ["<s>", "[", "INST", "]"]
    assert [tokens[r][i] for r, (i,) in enumerate(where["blanked.blank"])] == ["▁is"] * 4
    assert [tokens[r][i] for r, (i,) in enumerate(where["blanked.logits"])] == ["]"] * 4


def test_top_k_saves_its_tokens_as_text(filled, tmp_path):
    plan = filled["readout", "nnterp"]
    top3 = plan.results["top3"]
    assert top3["ids"].shape == top3["probs"].shape == (4, 1, 3)
    assert len(top3["tokens"][0][0]) == 3 and "▁" not in "".join(top3["tokens"][0][0])     # decoded text
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
    assert list(heads.results["iia"]) == [0, 1, 2, 3]
    heads.dump(str(tmp_path / "heads"))
    with safe_open(str(tmp_path / "heads" / "iia.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["num_heads=0", "num_heads=1", "num_heads=2", "num_heads=3"]


def test_a_saved_read_that_selects_a_head_tables_per_point(engines):
    """head_patching.json's counterfactual read, saved: `select: {num_heads: <int>}` keeps the head axis at
    size 1, and its table is a row per point, example and position whose value is that head's result."""
    plan = edited("head_patching", lambda d: d["saves"].append("counterfactual.v_cf")).compile()
    filled = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    rows = filled.table("counterfactual.v_cf")
    assert [r["at"] for r in rows] == [{"num_heads": h} for h in range(4) for _ in range(4)]
    assert {r["pos_token"] for r in rows} == {"▁is"}
    head_dim = filled.results["counterfactual.v_cf"][2].shape[-1]
    assert all(len(r["value"]) == 1 and len(r["value"][0]) == head_dim for r in rows)
    assert rows[8]["value"] == filled.results["counterfactual.v_cf"][2][0, 0].tolist()


def test_tables_join_wide_and_label_layers_and_keys(filled):
    rows = filled["two_observes", "nnterp"].table("patching_ld", "ablation_ld", "clean_ld")
    assert len(rows) == 4 and {"patching_ld", "ablation_ld", "clean_ld"} <= set(rows[0])
    lens = filled["logit_lens_all", "nnterp"].table("p_answer", "top1")
    assert [(r["layer"], r["row"]) for r in lens] == [(l, r) for l in range(2) for r in range(4)]
    pattern = filled["attention_knockout", "nnterp"].table("clean.pattern")
    first = [r for r in pattern if r["row"] == 1]
    assert [r["key_token"] for r in first][:2] == ["<s>", "▁If"] and len(first) == 9     # padding keys are left out
    assert len(first[0]["value"]) == 4                                                   # one probability per head


def test_a_value_with_positions_but_no_rows_tables_a_row_per_position(engines):
    """A mean over rows keeps the token axis; like any value with no rows it tables a row per element, and its
    positions are no one row's, so each is labelled by its index alone."""
    def change(d):
        d["steps"]["patched"]["interventions"]["logits"]["pos"] = {"index": "-2:"}
        d["steps"]["avg"] = {"kind": "op", "op": "mean", "of": "patched.logits", "over": ["rows"]}
        d["saves"] = ["avg"]
    plan = edited("patching", change).compile()
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    rows, avg = filled.table("avg"), filled.results["avg"]
    assert len(rows) == avg.numel() and "pos_token" not in rows[0]
    vocab = avg.shape[-1]
    assert (rows[vocab]["pos"], rows[vocab]["vocab_size"], rows[vocab]["value"]) == (1, 0, avg[1, 0].item())


def test_a_bundle_carries_the_header_its_featurizer_checks(filled, tmp_path):
    filled["pca_harvest", "hooks"].dump(str(tmp_path))
    from safetensors import safe_open
    with safe_open(str(tmp_path / "basis.safetensors"), "pt") as f:
        assert sorted(f.keys()) == ["basis", "mean"]
        assert f.metadata() == {"format": "clab/1", "kind": "pca", "k": "4",
                                "model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                                "model.revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
                                "model.dtype": "float32", "component": "layer_output", "layer": "0", "width": "16",
                                "document": "pca_harvest"}


def test_a_bundle_is_the_same_bytes_every_time_it_is_written(filled, tmp_path):
    """safetensors orders a header as it likes; clab writes it with sorted keys."""
    files = []
    for out in ("a", "b", "c"):
        filled["pca_harvest", "hooks"].dump(str(tmp_path / out))
        files.append((tmp_path / out / "basis.safetensors").read_bytes())
    assert files[0] == files[1] == files[2]
    n = int.from_bytes(files[0][:8], "little")
    header = json.loads(files[0][8:8 + n])
    assert list(header) == sorted(header) and list(header["__metadata__"]) == sorted(header["__metadata__"])


def test_the_cli_runs_remote_local(tmp_path, filled):
    out = tmp_path / "steer"
    assert main(["run", document("steer_renormalize"), "--engine", "ndif", "-o", "remote=local", "--batch-size",
                 str(BATCH_SIZE), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["engine"] == "ndif" and manifest["run"]["options"] == {"remote": "local"}
    assert same(Plan.load(str(out)).results, filled["steer_renormalize", "nnterp"].results)


def test_every_point_has_at():
    for name in FORWARD:
        for entry in plan_of(name).steps.values():
            swept = isinstance(entry, list)
            assert all(bool(p.at) == swept for p in points(entry)), name


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_ragged_write_lands_only_on_each_rows_own_tokens(engines, kind):
    """A window from position 6 to the last is four tokens on the 11-token row and two on the others: zeros
    land there, and every other position (BOS included) keeps its value."""
    def change(d, write):
        entries = d["steps"]["patched"]["interventions"]
        zero = {"zero": {"site": "target", "pos": {"index": "6:-1"}, "ops": [{"op": "swap", "with": 0.0}]}}
        d["steps"]["patched"]["interventions"] = {**(zero if write else {}), "logits": entries["logits"],
                                                  "whole": {"site": "target", "pos": {"index": ":"}}}
        d["saves"] = ["patched.whole"]
    written, clean = (edited("patching", lambda d, w=w: change(d, w)).compile() for w in (True, False))
    engine = engines(kind, written.model)
    done = engine.execute(written, batch_size=BATCH_SIZE)
    after, where = done.results["patched.whole"], placed(done)["patched.whole"]
    before = engine.execute(clean, batch_size=BATCH_SIZE).results["patched.whole"]
    zeroed = (after == 0).all(-1)
    assert [p for r, row in enumerate(where) for j, p in enumerate(row) if zeroed[r, j]] == \
        [6, 7, 8, 9, 6, 7, 6, 7, 6, 7]                                           # positions 6 to -1
    assert torch.equal(after.nan_to_num(7.0)[~zeroed], before.nan_to_num(7.0)[~zeroed])


def test_a_swept_value_joins_an_unswept_one_beside_each_point(engines):
    def change(d):
        d["steps"]["clean"] = {"kind": "forward", "input": "pairs.input",
                               "interventions": {"logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}
        d["steps"]["clean_ld"] = {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer",
                                  "b": "pairs.base_answer"}
        d["saves"] = ["logit_diff", "clean_ld"]
    plan = edited("pos_sweep", change).compile()
    filled = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    rows = filled.table("logit_diff", "clean_ld", columns=[])
    assert len(rows) == 12 and [r["at"] for r in rows] == [{"pos": p} for p in (-1, -2, -3) for _ in range(4)]
    clean = filled.results["clean_ld"].flatten().tolist()
    assert [r["clean_ld"] for r in rows] == clean * 3
    assert filled.table("clean_ld", "logit_diff", columns=[]) == rows


def test_a_token_column_is_tokenized_as_written_on_gpt2(engines):
    """On GPT-2 " one" and "one" (and " is"/"is", " a"/"a") are different single tokens, and a token column is
    tokenized exactly as written: a column without the leading space scores the token without it."""
    def change(d):
        d["data"]["pairs"] = {"rows": [{**row, "cf_answer": a, "bare_answer": a.strip(), "base_answer": " one"}
                                       for row, a in zip(ROWS, (" one", " is", " a", " one"))]}
        d["steps"]["spaced"] = {"kind": "op", "op": "token_prob", "of": "patched.logits", "token": "pairs.cf_answer"}
        d["steps"]["bare"] = {**d["steps"]["spaced"], "token": "pairs.bare_answer"}
        d["saves"] = ["patched.logits", "spaced", "bare"]
    plan = edited("gpt2_reach", change).compile()
    engine = engines("nnterp", plan.model)
    values = engine.execute(plan, batch_size=BATCH_SIZE).results
    probs = values["patched.logits"][:, 0].softmax(-1)
    rows = plan.data["pairs"].rows or []
    assert len({engine.tokenizer(r["bare_answer"])["input_ids"][0] for r in rows}) == 3
    for name, column in (("spaced", "cf_answer"), ("bare", "bare_answer")):
        ids = [engine.tokenizer(r[column], add_special_tokens=False)["input_ids"] for r in rows]
        assert all(len(i) == 1 for i in ids)
        expected = probs[torch.arange(len(rows)), torch.tensor([i[0] for i in ids])]
        torch.testing.assert_close(values[name][:, 0], expected)
    assert not torch.allclose(values["spaced"], values["bare"])


def test_pca_control_compiles_against_the_basis_pca_harvest_writes(engines, tmp_path, filled):
    import shutil
    shutil.copytree(os.path.join(os.path.dirname(document("patching"))), tmp_path / "documents")
    os.symlink(os.path.join(os.path.dirname(os.path.dirname(document("patching"))), "data"), tmp_path / "data")
    filled["pca_harvest", "nnterp"].dump(str(tmp_path / "documents" / "artifacts"))
    compiled = Plan(str(tmp_path / "documents" / "pca_control.json")).compile()
    basis = filled["pca_harvest", "nnterp"].results["basis"]
    assert same(points(compiled.featurizers["rot"])[0].bundle, basis)
    compiled.write(str(tmp_path / "documents" / "written.json"))
    assert same(points(Plan(str(tmp_path / "documents" / "written.json")).featurizers["rot"])[0].bundle, basis)


def test_a_bundle_made_at_a_layer_counted_from_the_last_says_the_layer_it_is(engines, tmp_path):
    """pca_harvest at layer -2 of the 2-layer model writes a header for layer 0, which pca_control loads at 0 and
    at -2, and refuses at -1 before any forward, naming the layer that is."""
    import shutil
    shutil.copytree(os.path.dirname(document("patching")), tmp_path / "documents")
    os.symlink(os.path.join(os.path.dirname(os.path.dirname(document("patching"))), "data"), tmp_path / "data")
    harvest = edited("pca_harvest", lambda d: d["sites"]["target"].update(layers=-2)).compile()
    engine = engines("hooks", harvest.model)
    engine.execute(harvest, batch_size=BATCH_SIZE).dump(str(tmp_path / "documents" / "artifacts"))

    def control(layers):
        with open(tmp_path / "documents" / "pca_control.json") as f:
            doc = json.load(f)
        doc["sites"]["target"]["layers"] = layers
        return Plan(**doc, document=str(tmp_path / "documents" / "pca_control.json")).compile()
    assert points(control(0).featurizers["rot"])[0].header["layer"] == "0"
    engine.execute(control(-2), batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"header 'layer' is '0'; this document uses 'rot' at layer_output layer "
                                         r"-1, which is layer 1"):
        engine.execute(control(-1), batch_size=BATCH_SIZE)


def test_a_metrics_operand_is_laid_out_as_the_value_it_scores():
    """A metric (`cosine` here, `kl` and `js` alike) scores a value against another laid out as it is at each
    layer, or against one vector (a layers axis of either aligns by label, as test_layers.py pins); another
    layout, or a ragged operand beside a rectangular value, is refused by name."""
    def change(d, op, operand_site, pos):
        d["steps"]["clean"]["interventions"]["other"] = {"site": operand_site, "pos": pos}
        d["steps"]["similarity"] = {"kind": "op", "op": op, "of": "patched.resid", "with": "clean.other"}
    heads = {"component": "attention_premix", "layers": [0, 2]}
    for site, pos, message in ((heads, {"index": "-2:"}, r"is laid out \['layers', 'rows', 'pos', 'num_heads', "
                                                          r"'head_dim'\]; 'cosine' scores a value laid out \['rows', "
                                                          r"'pos', 'hidden_size'\]"),
                               ({"component": "layer_output", "layers": 0}, {"index": ":"},
                                r"holds a different number of positions on each row, which pair with no other "
                                r"value's; read both at a static index")):
        def one(d, site=site, pos=pos):
            d["sites"]["stream"]["layers"] = 2
            change(d, "cosine", site, pos)
        with pytest.raises(PlanError, match=r"steps\.similarity\.with: 'clean\.other' " + message):
            edited("cosine_metric", one).compile()


def test_select_on_key_counts_each_rows_own_tokens(engines):
    """On attention_knockout.json's rows (11, 9, 9 and 9 tokens, so padded by 0, 2, 2, 2) the key axis is each
    row's own keys: unselected, key k is the row's position k (BOS key 0) and a shorter row's keys past its own
    are NaN, never its padding's; `select: {key: "0:2"}` is the same first two keys (BOS and the first prompt
    token), as the picks `[0, 1]` are; and a write at `{key: 0}` zeroes exactly the first, wherever each row's
    padding puts it, the rest of the row as it was. A table labels the keys a read selected, and the ones it did
    not, with their tokens, and has no key past a row's own."""
    def change(d):
        pattern = {"component": "attention_probabilities", "layers": 0}
        d["steps"] = {
            "clean": {"kind": "forward", "input": "pairs.input",
                      "interventions": {"whole": {"site": pattern, "pos": -1},
                                        "first": {"site": {**pattern, "select": {"key": "0:2"}}, "pos": -1},
                                        "picks": {"site": {**pattern, "select": {"key": [0, 1]}}, "pos": -1}}},
            "zeroed": {"kind": "forward", "input": "pairs.input",
                       "interventions": {"zero": {"site": {**pattern, "select": {"key": 0}}, "pos": -1,
                                                  "ops": [{"op": "swap", "with": 0.0}]},
                                         "whole": {"site": pattern, "pos": -1}}}}
        d["saves"] = ["clean.whole", "clean.first", "clean.picks", "zeroed.whole"]
    plan = engines("nnterp", plan_of("attention_knockout").model).execute(
        edited("attention_knockout", change).compile(), batch_size=BATCH_SIZE)
    values, tokens = plan.results, plan.run.tokens["pairs.input"]
    lengths = [len(t) for t in tokens]
    assert lengths == [11, 9, 9, 9] and values["clean.whole"].shape == (4, 1, 4, 11)
    assert torch.equal(values["clean.first"], values["clean.picks"])
    for r, n in enumerate(lengths):
        whole, zeroed = values["clean.whole"][r], values["zeroed.whole"][r]
        assert not whole[..., :n].isnan().any() and whole[..., n:].isnan().all()
        torch.testing.assert_close(whole[..., :n].sum(-1), torch.ones(1, 4))        # the row's keys hold it all
        assert torch.equal(values["clean.first"][r], whole[..., :2])
        assert zeroed[..., 0].eq(0).all() and not whole[..., 0].eq(0).all()
        assert torch.equal(zeroed[..., 1:n], whole[..., 1:n]) and zeroed[..., n:].isnan().all()
    assert [r["key_token"] for r in plan.table("clean.first") if r["row"] == 1] == ["<s>", "▁If"]
    table = plan.table("clean.whole")
    for r, n in enumerate(lengths):
        own = [(t["key"], t["key_token"]) for t in table if t["row"] == r]
        assert own == [(k, tokens[r][k]) for k in range(n)]
