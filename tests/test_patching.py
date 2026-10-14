"""patching.json end to end: its numbers per engine, the engines' agreement, remote, the way home, the output."""

import json
import os

import pytest
import torch

import clab
from clab import Plan
from clab.cli import main
from conftest import BATCH_SIZE, ROOT, document, edited, plan_of, same, snapshot, stand_in

SAVED = ["iia", "logit_diff"]


def run(engines, kind: str, name: str = "patching", **kwargs) -> Plan:
    plan = plan_of(name)
    return engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE, **kwargs)


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    return {kind: run(engines, kind) for kind in ("nnterp", "hooks")}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_numbers_match_the_snapshot(filled, kind):
    values = {k: v.tolist() for k, v in filled[kind].results.items()}
    expected = snapshot(f"patching.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE and list(values) == SAVED
    for key in SAVED:
        torch.testing.assert_close(torch.tensor(values[key]), torch.tensor(expected["values"][key]), atol=1e-6, rtol=0)


def test_the_engines_agree_to_the_bit(filled):
    assert same(filled["nnterp"].results, filled["hooks"].results)


def test_results_are_detached_cpu_clones(filled):
    for value in filled["nnterp"].results.values():
        assert value.device.type == "cpu" and not value.requires_grad and value._base is None


def test_the_run_record_says_how_it_ran(filled):
    run = filled["nnterp"].run
    assert (run.engine, run.remote, run.batch_size) == ("nnterp", False, BATCH_SIZE)
    assert {"clab", "nnsight", "nnterp", "transformers", "torch", "python"} <= set(run.versions)
    assert run.tokens["pairs.input"][0] == ["<s>", "▁If", "▁today", "▁is", "▁Th", "urs", "day", ",", "▁tom",
                                            "orrow", "▁is"]
    assert run.positions["patched.patch"] == [[10], [8], [8], [8]]       # the last token, BOS-prefixed rows


def test_execute_refuses_an_uncompiled_plan_and_dump_one_without_results(engines, tmp_path):
    plan = Plan(document("patching"))
    with pytest.raises(ValueError, match="plan is not compiled; call plan.compile()"):
        engines("nnterp", plan.model).execute(plan)
    with pytest.raises(ValueError, match="the plan has not run"):
        plan.compile().dump(str(tmp_path))


def test_an_engine_refuses_a_plan_for_another_checkpoint(engines):
    with pytest.raises(ValueError, match="this engine loaded hf-internal-testing/tiny-random-LlamaForCausalLM"):
        engines("nnterp", plan_of("patching").model).execute(plan_of("gpt2_reach"))


def test_remote_local_is_bit_equal_to_local(engines, filled):
    there = run(engines, "nnterp", remote="local")
    assert there.run is not None and there.run.remote == "local" and same(there.results, filled["nnterp"].results)


def test_the_way_home_through_a_stand_in_server(engines, filled, monkeypatch):
    """The way home, against this design's own model of a server (`conftest.stand_in`), not nnsight's server
    code. It shows a filled `Plan` survives that trip whole, pickled by class reference with nothing of clab
    shipped by value; only a real server (M5) shows the rest."""
    import cloudpickle

    plan = plan_of("patching")
    home, trip = stand_in(engines("nnterp", plan.model), plan, monkeypatch)
    assert trip == {"by_reference": True, "names": ["plan"], "types": ["Plan"]}
    assert not any(m.split(".")[0] in ("clab", "nnterp") for m in cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES)
    assert home is not plan and plan.results == {}                 # what came back is the server's copy
    assert same(home.results, filled["nnterp"].results)
    assert home.run is not None and home.run.positions == filled["nnterp"].run.positions


def test_dump_and_load_are_inverses(filled, tmp_path):
    plan = filled["hooks"]
    plan.dump(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["iia.safetensors", "logit_diff.safetensors", "manifest.json"]
    back = Plan.load(str(tmp_path))
    assert same(back.results, plan.results)
    assert back.model_dump(exclude={"data", "document"}) == plan.model_dump(exclude={"data", "document"})
    assert back.document == "documents/patching.json"             # recorded relative to the corpus root
    assert back.data["pairs"].rows is None and back.data["pairs"].sha256 == plan.data["pairs"].sha256
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["results"]["iia"] == {"file": "iia.safetensors", "step": "iia", "dims": ["rows", "pos"], "axes": [],
                                          "shape": [4, 1], "dtype": "float32"}
    assert "rows" not in manifest["data"]["pairs"]
    assert manifest["model"]["dtype"] == "float32"               # recorded though it is the default


def test_a_table_labels_each_row(filled, tmp_path, monkeypatch):
    filled["hooks"].dump(str(tmp_path))
    monkeypatch.chdir(ROOT)                                   # a loaded plan's corpus root is where it is read
    rows = Plan.load(str(tmp_path)).table("logit_diff")       # the rows come back from the data file
    assert rows[0] == {"row": 0, "pos": 10, "pos_token": "▁is", "value": filled["hooks"].results["logit_diff"][0, 0].item(),
                       "input": "If today is Thursday, tomorrow is", "cf_answer": " Sunday", "base_answer": " Friday"}
    assert [r["value"] for r in rows] == filled["hooks"].results["logit_diff"].flatten().tolist()
    wide = filled["hooks"].table("iia", "logit_diff", columns=[])
    assert list(wide[1]) == ["row", "pos", "pos_token", "iia", "logit_diff"]
    with pytest.raises(ValueError, match="'patched.logits' was not saved; add it to saves"):
        filled["hooks"].table("patched.logits")


def test_clab_run_takes_a_path_or_a_plan_and_a_name_or_an_engine(engines, filled, tmp_path):
    plan = clab.run(document("patching"), engine="hooks", out=str(tmp_path / "a"), batch_size=BATCH_SIZE)
    again = clab.run(Plan(document("patching")), engine=engines("hooks", plan.model), out=str(tmp_path / "b"),
                     batch_size=BATCH_SIZE)
    assert same(plan.results, filled["hooks"].results) and same(again.results, plan.results)
    assert repr(again) == "Plan(patching, run on hooks, 2 results)"


def test_the_cli_runs(tmp_path, capsys):
    out = tmp_path / "patching"
    assert main(["run", document("patching"), "--engine", "hooks", "--batch-size", str(BATCH_SIZE),
                 "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "iia.safetensors" in printed and "logit_diff.safetensors" in printed
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["engine"] == "hooks" and manifest["run"]["batch_size"] == BATCH_SIZE
    assert list(manifest["results"]) == ["iia", "logit_diff"]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_read_at_a_written_site_observes_the_write(engines, kind):
    def change(d):
        d["steps"]["patched"]["reads"]["after"] = {"site": "target", "pos": -1}
        d["saves"] = ["patched.after", "counterfactual.v_cf"]
    plan = edited("patching", change).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert torch.equal(values["patched.after"], values["counterfactual.v_cf"])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_engine_never_sees_a_sweep_or_at(engines, kind, monkeypatch):
    plan = plan_of("pos_sweep")
    engine = engines(kind, plan.model)
    seen = []
    forward = engine.forward

    def recording(batch, writes, reads):
        seen.append(writes + reads)
        return forward(batch, writes, reads)

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {k for v in value.values() for k in keys(v)}
        if isinstance(value, list):
            return {k for v in value for k in keys(v)}
        return set()

    monkeypatch.setattr(engine, "forward", recording)
    engine.execute(plan, batch_size=BATCH_SIZE)
    assert len(seen) == 2 + 3 * 2         # two batches of counterfactual, then of each of three patched points
    assert not keys(seen) & {"sweep", "at"}


def test_unserved_kinds_refuse_by_name_before_anything_runs(engines):
    plan = plan_of("das")
    with pytest.raises(NotImplementedError, match="fit: fit steps land in M4"):
        engines("hooks", plan.model).execute(plan)


@pytest.mark.parametrize("name, component", [("attention_query", "attention_queries"),
                                             ("attention_knockout", "attention_probabilities"),
                                             ("attention_pattern_patching", "attention_probabilities")])
def test_hooks_refuses_what_lives_inside_a_forward(engines, name, component):
    plan = plan_of(name)
    with pytest.raises(ValueError, match=f"does not serve {component}: nnterp reads it inside a module's forward"):
        engines("hooks", plan.model).execute(plan)
    with pytest.raises(ValueError, match="remote runs need the nnterp engine"):
        engines("hooks", plan.model).execute(plan, remote="local")
