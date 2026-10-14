"""Numbers that pin what padding, position ids and prompt positions mean, on both engines.

`patching.json` alone cannot: at its last position on Llama neither the padding width, the position ids
nor where the prompt starts changes a number. These documents do: a swept position, composed writes, an
interchange at a position counted from the start of prompts of different lengths, and GPT-2 (absolute
positions) on inline rows of mixed lengths.
"""

import json
import os

import pytest
import torch

from clab.cli import main
from clab.compiler import compile, points
from clab.output import load, table
from conftest import document, plan_of, snapshot

BATCH_SIZE = 2
ROWS = [
    {"input": "If the count is one, next is", "counterfactual_inputs": ["two"], "base_answer": " two", "cf_answer": " three"},
    {"input": "one", "counterfactual_inputs": ["If the count is three and it goes on, next is"],
     "base_answer": " three", "cf_answer": " four"},
    {"input": "four, then", "counterfactual_inputs": ["The count is four"], "base_answer": " one", "cf_answer": " two"},
    {"input": "If the count is two, then the one after it is", "counterfactual_inputs": ["one, two"],
     "base_answer": " four", "cf_answer": " one"},
]


def mixed_lengths(rows: list[dict]) -> dict:
    """GPT-2 at layer 1, the last token of one prompt swapped into another's, over inline rows of mixed lengths."""
    with open(document("gpt2_reach")) as f:
        doc = json.load(f)
    doc["data"] = {"pairs": {"rows": rows}}
    doc["sites"]["target"]["layers"] = 1
    doc["steps"]["patched"]["reads"]["resid"] = {"site": "target", "pos": -1}
    doc["steps"]["saves"] = {"logit_diff": "logit_diff.json", "patched.resid": "resid.safetensors",
                             "counterfactual.v_cf": "v_cf.safetensors"}
    return compile(doc, document("gpt2_reach"))


def from_the_start() -> dict:
    """patching.json at prompt token 3, counted from the start (BOS and padding are not prompt tokens): the
    first token of the day's name, on prompts 11 and 9 tokens long. Tokens 0 to 2 are "If today is" on
    every row, so a count that included BOS or padding would land on a token both prompts share."""
    with open(document("patching")) as f:
        doc = json.load(f)
    doc["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = 3
    doc["steps"]["patched"]["writes"]["patch"]["pos"] = 3
    return compile(doc, document("patching"))


PLANS = {"pos_sweep": lambda: plan_of("pos_sweep"), "steer_renormalize": lambda: plan_of("steer_renormalize"),
         "from_the_start": from_the_start, "mixed_lengths": lambda: mixed_lengths(ROWS)}


def saved(plan: dict) -> dict:
    return {f"{key}@{json.dumps(p['at'], sort_keys=True)}": slot["value"]
            for entry in plan["steps"].values() for p in points(entry) for key, slot in p["result"].items()}


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    out = {}
    for name, make in PLANS.items():
        plan = make()
        for kind in ("nnterp", "hooks"):
            out[name, kind] = engines(kind, plan["model"]).execute(make(), batch_size=BATCH_SIZE)
    return out


@pytest.mark.parametrize("name", PLANS)
@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_numbers_match_the_snapshot(filled, name, kind):
    values = {k: v.tolist() for k, v in saved(filled[name, kind]).items()}
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE and set(values) == set(expected["values"])
    for key, value in values.items():
        torch.testing.assert_close(torch.tensor(value), torch.tensor(expected["values"][key]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", PLANS)
def test_the_engines_agree_to_the_bit(filled, name):
    a, b = saved(filled[name, "nnterp"]), saved(filled[name, "hooks"])
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_rows_numbers_do_not_depend_on_its_batch_companions(engines, kind):
    """At one batch shape, a row padded to its input's longest row gets the same numbers whichever rows share its
    batch: regrouped from (0, 1), (2, 3) into (0, 2), (1, 3), every row's results are the same to the bit."""
    order = [0, 2, 1, 3]
    together, regrouped = mixed_lengths(ROWS), mixed_lengths([ROWS[i] for i in order])
    engine = engines(kind, together["model"])
    a = saved(engine.execute(together, batch_size=BATCH_SIZE))
    b = saved(engine.execute(regrouped, batch_size=BATCH_SIZE))
    for key in a:
        assert torch.equal(a[key][order], b[key]), key


def test_the_swapped_vector_is_the_counterfactuals_on_padded_rows(filled):
    for kind in ("nnterp", "hooks"):
        values = saved(filled["mixed_lengths", kind])
        assert torch.equal(values["patched.resid@{}"], values["counterfactual.v_cf@{}"]), kind


def test_a_window_and_three_single_writes_give_the_same_numbers(engines):
    for kind in ("nnterp", "hooks"):
        window = plan_of("window_patch")
        engine = engines(kind, window["model"])
        a = saved(engine.execute(window, batch_size=BATCH_SIZE))["logit_diff@{}"]
        b = saved(engine.execute(plan_of("multi_position_patch"), batch_size=BATCH_SIZE))["logit_diff@{}"]
        assert torch.equal(a, b), kind


def test_renormalize_restores_the_norm_steering_changed(filled):
    for kind in ("nnterp", "hooks"):
        values = saved(filled["steer_renormalize", kind])
        before = values["base.resid@{}"].norm(dim=-1)
        torch.testing.assert_close(values["steered_renormed.resid@{}"].norm(dim=-1), before)
        assert not torch.allclose(values["steered.resid@{}"].norm(dim=-1), before)


def test_a_swept_output_tables_every_point(tmp_path):
    out = tmp_path / "pos_sweep"
    assert main(["run", document("pos_sweep"), "--engine", "hooks", "--batch-size", str(BATCH_SIZE),
                 "--out", str(out)]) == 0
    assert sorted(os.listdir(out)) == ["manifest.json", "pos=-1", "pos=-2", "pos=-3"]
    rows = table(load(str(out)), "iia")
    assert [r["at"] for r in rows] == [{"pos": p} for p in (-1, -2, -3) for _ in range(4)]
    assert {r["pos"] for r in rows} == {0}           # the value's own pos axis, apart from the sweep's coordinate


def test_the_cli_runs_remote_local(tmp_path, filled):
    out = tmp_path / "steer"
    assert main(["run", document("steer_renormalize"), "--remote", "local", "--batch-size", str(BATCH_SIZE),
                 "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["remote"] == "local"
    back = saved(load(str(out)))
    for key, value in saved(filled["steer_renormalize", "nnterp"]).items():
        assert torch.equal(back[key], value), key
