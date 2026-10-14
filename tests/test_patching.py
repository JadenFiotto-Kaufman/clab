"""patching.json end to end: its numbers per engine, the engines' agreement, remote, the way home, the output."""

import io
import json
import os

import pytest
import torch

import clab
from clab.cli import main
from clab.compile import compile, points
from clab.output import write
from conftest import document, plan_of, snapshot

BATCH_SIZE = 2      # results are reproducible to the bit only at a fixed batch size
SAVED = ("iia", "logit_diff")


def run(engines, kind: str, name: str = "patching", **kwargs) -> dict:
    plan = plan_of(name)
    return engines(kind, plan["model"]).execute(plan, batch_size=BATCH_SIZE, **kwargs)


def saved(plan: dict) -> dict:
    return {key: slot["value"] for entry in plan["steps"].values() for p in points(entry)
            for key, slot in p["result"].items()}


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    return {kind: run(engines, kind) for kind in ("nnterp", "hooks")}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_numbers_match_the_snapshot(filled, kind):
    values = {k: v.tolist() for k, v in saved(filled[kind]).items()}
    expected = snapshot(f"patching.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE
    for key in SAVED:
        torch.testing.assert_close(torch.tensor(values[key]), torch.tensor(expected["values"][key]), atol=1e-6, rtol=0)


def test_the_engines_agree_to_the_bit(filled):
    a, b = saved(filled["nnterp"]), saved(filled["hooks"])
    assert set(a) == set(b) == set(SAVED)
    for key in SAVED:
        assert torch.equal(a[key], b[key]), key


def test_slots_hold_detached_cpu_clones(filled):
    for value in saved(filled["nnterp"]).values():
        assert value.device.type == "cpu" and not value.requires_grad and value._base is None


def test_remote_local_is_bit_equal_to_local(engines, filled):
    there = saved(run(engines, "nnterp", remote="local"))
    for key in SAVED:
        assert torch.equal(there[key], saved(filled["nnterp"])[key]), key


def test_the_way_home_through_a_stand_in_server(engines, filled, monkeypatch):
    """The way home, against this design's own model of a server (DESIGN.md section 10(b)), not nnsight's server
    code: serialize the session, run the *deserialized* tracer with the model's persistent objects, take what its
    outermost boundary pushes, round-trip that through torch.save/torch.load as a response does, then mark it and
    push it into the caller's frame as RemoteBackend._push does. It shows a filled plan survives that trip whole;
    only a real server (M5) shows the rest."""
    from nnsight.schema.request import RequestModel
    from nnsight.tracing.backend import Backend
    from nnsight.tracing.tracer import _saves, mark
    from nnsight.tracing.util import push

    engine = engines("nnterp", plan_of("patching")["model"])
    trip = {}

    class StandIn(Backend):
        def __call__(self, tracer):
            blob = RequestModel.serialize(tracer, compress=False)
            server = RequestModel.deserialize(blob, engine.model._remoteable_persistent_objects(), compress=False)
            assert server.info is not None
            server.execute(server.info.code)
            result = {k: v for k, v in server.info.frame.f_locals.items() if id(v) in _saves()}
            buffer = io.BytesIO()
            torch.save(result, buffer)
            buffer.seek(0)
            result = torch.load(buffer, weights_only=False)
            trip.update(names=sorted(result))
            for value in result.values():
                mark(value)
            assert tracer.info is not None
            push(tracer.info.frame, result)

    monkeypatch.setattr(type(engine.model), "_remote_backend", lambda self, *args: StandIn())
    plan = plan_of("patching")
    home = engine.execute(plan, batch_size=BATCH_SIZE, remote=True)
    assert trip["names"] == ["plan"]
    assert home is not plan and saved(plan)["iia"] is None        # what came back is the server's copy
    for key in SAVED:
        assert torch.equal(saved(home)[key], saved(filled["nnterp"])[key]), key


def test_load_round_trips_the_filled_plan(filled, tmp_path):
    plan = filled["hooks"]
    write(plan, str(tmp_path), {"engine": "hooks", "remote": False, "batch_size": BATCH_SIZE,
                                     "started": "", "seconds": 0})
    back = clab.load(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["iia.json", "logit_diff.json", "manifest.json"]
    for key in ("format", "document", "description", "model", "featurizers", "sweeps"):
        assert back[key] == plan[key], key
    assert {n: d["n_rows"] for n, d in back["data"].items()} == {n: len(d["rows"]) for n, d in plan["data"].items()}

    def same(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    assert same(plan["steps"], back["steps"])
    rows = clab.table(back, "logit_diff")
    assert [r["row"] for r in rows] == [0, 1, 2, 3]
    assert [r["value"] for r in rows] == saved(plan)["logit_diff"].flatten().tolist()


def test_the_cli_runs(tmp_path, capsys):
    out = tmp_path / "patching"
    assert main(["run", document("patching"), "--engine", "hooks", "--batch-size", str(BATCH_SIZE),
                 "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "iia.json" in printed and "logit_diff.json" in printed
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["engine"] == "hooks" and manifest["batch_size"] == BATCH_SIZE
    assert [s["file"] for s in manifest["saves"]] == ["iia.json", "logit_diff.json"]
    table = json.loads((out / "iia.json").read_text())
    assert table["dims"] == ["rows", "pos"] and len(table["rows"]) == 4


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_read_at_a_written_site_observes_the_write(engines, kind):
    with open(document("patching")) as f:
        doc = json.load(f)
    doc["steps"]["patched"]["reads"]["after"] = {"site": "target", "pos": -1}
    doc["steps"]["saves"] = {"patched.after": "after.safetensors", "counterfactual.v_cf": "v_cf.safetensors"}
    plan = compile(doc, document("patching"))
    filled = engines(kind, plan["model"]).execute(plan, batch_size=BATCH_SIZE)
    values = saved(filled)
    assert torch.equal(values["patched.after"], values["counterfactual.v_cf"])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_engine_never_sees_a_sweep_or_at(engines, kind, monkeypatch):
    plan = plan_of("pos_sweep")
    engine = engines(kind, plan["model"])
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
    engine = engines("hooks", plan["model"])
    with pytest.raises(NotImplementedError, match="step 'fit' is a fit step; fit steps are served from M4"):
        engine.execute(plan)
    plan = plan_of("generated_answer")
    with pytest.raises(NotImplementedError, match="step 'answer' is a generate step; generate steps are served from M3"):
        engines("hooks", plan["model"]).execute(plan)


def test_hooks_refuses_what_lives_inside_a_forward(engines):
    plan = plan_of("attention_query")
    with pytest.raises(ValueError, match="does not serve attention_queries: nnterp reads it inside a module's forward"):
        engines("hooks", plan["model"]).execute(plan)
    with pytest.raises(ValueError, match="remote runs need the nnterp engine"):
        engines("hooks", plan["model"]).execute(plan, remote="local")
