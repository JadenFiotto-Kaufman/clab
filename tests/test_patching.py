"""patching.json end to end: its numbers per engine, the engines' agreement, remote, the way home, the output."""

import json
import os
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, ROOT, TRIP, by_reference, document, edited, plan_of, same, snapshot, stand_in

import clab
from clab import Plan
from clab.cli import main
from clab.format.sweeps import points
from clab.ops import edits
from clab.plan import PlanError, entries

SAVED = ["iia", "logit_diff"]


def run(engines, kind: str, name: str = "patching", **options) -> Plan:
    plan = plan_of(name)
    return engines(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE)


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
    assert (run.engine, run.batch_size, run.device, run.options) == ("nnterp", BATCH_SIZE, "cpu", {})
    assert (run.dtype, run.devices) == (["float32"], ["cpu"])               # the model's parameters, where it is
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
    there = run(engines, "ndif", remote="local")
    assert there.run is not None and there.run.options == {"remote": "local"}
    assert same(there.results, filled["nnterp"].results)


def test_the_way_home_through_a_stand_in_server(engines, filled, monkeypatch):
    """The way home, against this design's own model of a server (`conftest.stand_in`), not nnsight's server
    code. The ndif engine's request names clab's plan, engine and steps by class reference with nothing of clab
    shipped by value, and what comes home is the results and the run record, never the plan, which the
    client assigns into its own plan."""
    import cloudpickle

    plan = plan_of("patching")
    home, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP
    assert not any(m.split(".")[0] in ("clab", "nnterp") for m in cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES)
    assert home is plan and same(home.results, filled["nnterp"].results)
    assert home.run is not None and home.run.positions == filled["nnterp"].run.positions


def test_the_by_reference_check_sees_a_function_shipped_by_value(engines, monkeypatch):
    """A scratch session body on the ndif shell that names the plan, the engine and the steps passes the check;
    the same body calling a lambda it closes over (which nnsight ships by value, from its source) fails it."""
    from nnsight.schema.request import RequestModel
    from nnsight.tracing.backend import Backend

    from clab import steps

    plan = plan_of("patching")
    client = engines("ndif", plan.model, remote=True)
    blobs = []

    class Capture(Backend):
        def __call__(self, tracer):
            blobs.append(RequestModel.serialize(tracer, compress=False))

    monkeypatch.setattr(type(client.model), "_remote_backend", lambda self, *args: Capture())
    helper = lambda x: x  # noqa: E731
    with client.model.session(remote=True):           # serialized, never run: the backend only keeps the request
        kept = [plan.model, steps.versions, client.spec]  # noqa: F841
    with client.model.session(remote=True):
        kept = [plan.model, steps.versions, client.spec, helper(1)]  # noqa: F841
    assert by_reference(blobs[0]) and b"_load_function" in blobs[1] and not by_reference(blobs[1])


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


def test_a_loaded_plan_dumps_again(filled, tmp_path):
    """A loaded plan holds no rows a file gave, and dumps the same manifest again."""
    filled["hooks"].dump(str(tmp_path / "a"))
    Plan.load(str(tmp_path / "a")).dump(str(tmp_path / "b"))
    assert (tmp_path / "a" / "manifest.json").read_text() == (tmp_path / "b" / "manifest.json").read_text()


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


def test_a_loaded_plan_tables_the_rows_a_datasets_options_left(engines, tmp_path, monkeypatch):
    """A file read shuffled, taken and renamed is materialized again the same way when a loaded plan tables it,
    and checked against the digest the run recorded."""
    def change(d):
        d["data"]["pairs"].update(shuffle=True, take=3, columns={"target": "cf_answer"})
        d["steps"]["iia"]["expected"] = d["steps"]["logit_diff"]["a"] = "pairs.target"
    plan = edited("patching", change).compile()
    filled = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    filled.dump(str(tmp_path))
    monkeypatch.chdir(ROOT)
    back = Plan.load(str(tmp_path))
    assert back.data["pairs"].rows is None and back.data["pairs"].n_rows == 3
    rows = back.table("iia")
    assert [r["input"] for r in rows] == [r["input"] for r in plan.data["pairs"].rows or []]
    assert [r["target"] for r in rows] == [r["target"] for r in plan.data["pairs"].rows or []]


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
        d["steps"]["patched"]["interventions"]["after"] = {"site": "target", "pos": -1}
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

    def recording(batch, records):
        seen.append(records)
        return forward(batch, records)

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


@pytest.mark.parametrize("name, component", [("attention_query", "attention_queries"),
                                             ("attention_knockout", "attention_probabilities"),
                                             ("attention_pattern_patching", "attention_probabilities")])
def test_hooks_refuses_what_lives_inside_a_forward(engines, name, component):
    plan = plan_of(name)
    with pytest.raises(ValueError, match=f"does not serve {component}: nnterp reads it inside a module's forward"):
        engines("hooks", plan.model).execute(plan)


def test_an_option_the_engine_does_not_declare_is_refused_before_anything_loads(monkeypatch):
    """`load` checks the options against the engine's constructor before calling it: an unknown key is named
    with the ones the engine takes, and a value of another type is refused."""
    from clab.engines import load, options

    model = plan_of("patching").model
    for where in ("nnterp", "ndif"):
        monkeypatch.setattr(f"clab.engines.{where}.StandardizedTransformer", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="unknown engine 'vllm'; valid: nnterp, hooks, ndif"):
        load("vllm", model)
    assert list(options("nnterp")) == [] and list(options("ndif")) == ["remote"]
    with pytest.raises(ValueError, match="the nnterp engine takes no 'remote'; its options: device$"):
        load("nnterp", model, remote="local")
    for kind in ("nnterp", "hooks"):            # where the model goes is `device`, a device map as it is
        with pytest.raises(ValueError, match=f"the {kind} engine takes no 'device_map'; its options: device$"):
            load(kind, model, device_map="auto")
    with pytest.raises(ValueError, match="the ndif engine takes no 'device_map'; its options: device, remote$"):
        load("ndif", model, device_map="auto")
    with pytest.raises(ValueError, match="remote: 'yes' is not one of True, 'local'"):
        load("ndif", model, remote="yes")


def test_an_option_the_engine_cannot_honor_is_refused_by_name():
    with pytest.raises(ValueError, match="device: the ndif engine cannot honor 'cuda': the server places the model"):
        clab.run(document("patching"), engine="ndif", device="cuda")
    with pytest.raises(ValueError, match=r"device: the ndif engine places a local model on one torch device \('cpu', "
                                         r"'cuda:0'\), not 'auto'"):
        clab.run(document("patching"), engine="ndif", device="auto", remote="local")
    with pytest.raises(ValueError, match=r"model.device_map: the hooks engine cannot honor it: where the model runs is "
                                         r"a run option: --device"):
        clab.run(edited("patching", lambda d: d["model"].update(device_map="auto")), engine="hooks")


def test_a_loaded_engine_takes_no_device_or_option(engines):
    plan = plan_of("patching")
    with pytest.raises(ValueError, match="the loaded ndif engine runs on cpu with {'remote': 'local'}; device and "
                                         "options are given when an engine loads"):
        clab.run(plan, engine=engines("ndif", plan.model, remote="local"), remote=True)
    with pytest.raises(ValueError, match="the loaded hooks engine runs on cpu"):
        clab.run(plan, engine=engines("hooks", plan.model), device="cuda")


def test_the_run_record_names_the_versions_and_where_ndif_ran(engines, monkeypatch):
    """`versions` is what served the run where the model is (here this process), clab's with its commit; under
    remote=True the server places the model, so the run's device is None."""
    from clab.cli import how
    from clab.steps import version

    run = engines("nnterp", plan_of("patching").model).execute(plan_of("patching"), batch_size=2).run
    assert run is not None and list(run.versions) == ["clab", "nnsight", "nnterp", "torch", "transformers", "python"]
    assert run.versions["clab"] == version() and version().startswith("0.0.1+")    # a checkout: git describe's commit
    home, _ = stand_in(engines, plan_of("patching"), monkeypatch)
    assert home.run is not None and (home.run.device, home.run.devices) == (None, ["cpu"])
    assert how(home.run) == "device placed by the server, remote true"


def test_the_cli_says_a_refusal_from_the_server_in_one_line(monkeypatch, capsys):
    from nnsight.intervention.backends.remote import RemoteError

    def refused(*args, **kwargs):
        raise RemoteError("Traceback (most recent call last):\n  File ...\nValueError: patched.patch: row 0: no")
    monkeypatch.setattr(clab, "run", refused)
    assert main(["run", document("patching"), "--engine", "ndif"]) == 1
    assert capsys.readouterr().err == "clab: ValueError: patched.patch: row 0: no\n"


def test_the_cli_takes_the_run_options(tmp_path, capsys, engines):
    out = tmp_path / "patching"
    assert main(["run", document("patching"), "--engine", "ndif", "-o", "remote=local", "--device", "cpu",
                 "--batch-size", str(BATCH_SIZE), "--out", str(out)]) == 0
    assert "patching on ndif (batch size 2, device cpu, remote \"local\")" in capsys.readouterr().out
    run = Plan.load(str(out)).run
    assert run is not None and (run.engine, run.device, run.options, run.dtype, run.devices) == \
        ("ndif", "cpu", {"remote": "local"}, ["float32"], ["cpu"])
    assert main(["run", document("patching"), "-o", "remote=local", "--out", str(out)]) == 1
    assert "clab: the nnterp engine takes no 'remote'; its options: device" in capsys.readouterr().err
    assert main(["run", document("patching"), "-o", "device_map=auto", "--out", str(out)]) == 1
    assert "clab: the nnterp engine takes no 'device_map'; its options: device" in capsys.readouterr().err
    assert main(["run", document("patching"), "-o", "device=cuda", "--out", str(out)]) == 1
    assert "clab: -o device: it is given as --device" in capsys.readouterr().err
    assert main(["run", document("patching"), "-o", "remote", "--out", str(out)]) == 1
    assert "clab: -o 'remote': an option is KEY=VALUE" in capsys.readouterr().err


def test_an_engine_option_goes_to_its_constructor_and_is_recorded(monkeypatch, tmp_path, capsys):
    """An engine's options are its constructor's keyword parameters: given by name to `clab.run` or `-o`, refused
    when undeclared or of another type, and recorded in the run after their defaults."""
    from clab.engines import ENGINES, engine, load
    from clab.engines.hooks import HooksEngine

    class Scratch(HooksEngine):
        def __init__(self, model, *, device: str = "cpu", width: int = 1):
            super().__init__(model, device=device)
            self.options = {"width": width}

    monkeypatch.setitem(ENGINES, "scratch", Scratch)
    engine("scratch")(Scratch)
    filled = clab.run(document("patching"), engine="scratch", out=str(tmp_path / "a"), batch_size=BATCH_SIZE, width=3)
    assert filled.run is not None and filled.run.options == {"width": 3}
    assert Plan.load(str(tmp_path / "a")).run.options == {"width": 3}  # type: ignore
    loaded: Any = load("scratch", plan_of("patching").model)
    again = clab.run(document("patching"), engine=loaded, out=str(tmp_path / "b"), batch_size=BATCH_SIZE)
    assert again.run is not None and again.run.options == {"width": 1}
    assert main(["run", document("patching"), "--engine", "scratch", "-o", "width=2",
                 "--batch-size", str(BATCH_SIZE), "--out", str(tmp_path / "c")]) == 0
    assert "(batch size 2, device cpu, width 2)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="the scratch engine takes no 'nope'; its options: device, width"):
        clab.run(document("patching"), engine="scratch", nope=1)
    with pytest.raises(ValueError, match="width: 'x' is not int"):
        clab.run(document("patching"), engine="scratch", width="x")
    with pytest.raises(ValueError, match="the loaded scratch engine runs on cpu .*; device and options are given "
                                         "when an engine loads"):
        clab.run(document("patching"), engine=loaded, width=2)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_writes_value_is_what_a_read_after_it_at_the_site_takes(engines, kind):
    """Every entry binds its value after its ops, a write's included: `patched.patch` is the swapped activation,
    to the bit the counterfactual's and what a read at the site after it takes. An op after the last edit runs
    on that value once it is off the model, and the model still sees the write: `lensed` swaps, then reads out."""
    def change(d):
        entries = d["steps"]["patched"]["interventions"]
        entries["after"] = {"site": "target", "pos": -1}
        entries["after_lens"] = {"site": "target", "pos": -1, "ops": [{"op": "readout"}]}
        d["steps"]["lensed"] = {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"},
                                                            {"op": "readout"}]},
            "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}
        d["saves"] = ["patched.patch", "patched.after", "patched.after_lens", "patched.logits", "counterfactual.v_cf",
                      "lensed.patch", "lensed.logits"]
    plan = edited("patching", change).compile()
    assert entries(points(plan.steps["lensed"])[0])["patch"].dims == ["rows", "pos", "vocab_size"]
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert torch.equal(values["patched.patch"], values["patched.after"])
    assert torch.equal(values["patched.patch"], values["counterfactual.v_cf"])
    assert torch.equal(values["lensed.patch"], values["patched.after_lens"])
    assert torch.equal(values["lensed.logits"], values["patched.logits"])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_two_entries_at_one_site_compose_in_order_each_binding_its_own_value(engines, kind):
    """steer_renormalize.json's `steered_renormed` runs `steer` then `restore` at one site: `steer`'s value is
    the steered activation before `restore` renormalizes it (`steered.steer`'s, to the bit), and `restore`'s is
    what a read after both takes."""
    def change(d):
        d["steps"]["steered_renormed"]["interventions"][-1]["resid"] = {"site": "target", "pos": -1}
        d["saves"] = ["steered.steer", "steered_renormed.steer", "steered_renormed.restore", "steered_renormed.resid"]
    plan = edited("steer_renormalize", change).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert torch.equal(values["steered_renormed.steer"], values["steered.steer"])
    assert torch.equal(values["steered_renormed.restore"], values["steered_renormed.resid"])
    assert not torch.equal(values["steered_renormed.restore"], values["steered_renormed.steer"])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_an_entry_nothing_uses_is_not_gathered(engines, kind, monkeypatch):
    """A value leaves the model only when something uses it: `patch` writes and is not taken off, an unused read
    is not either (nor do its ops run), and what is saved or consumed is; the numbers are the document's."""
    def change(d):
        d["steps"]["patched"]["interventions"]["unused"] = {"site": "target", "pos": -1, "ops": [{"op": "readout"}]}
    plan = edited("patching", change).compile()
    flags = {f"{name}.{k}": e.gathered for name in ("counterfactual", "patched")
             for k, e in entries(points(plan.steps[name])[0]).items()}
    assert flags == {"counterfactual.v_cf": True, "patched.patch": False, "patched.logits": True,
                     "patched.unused": False}
    engine = engines(kind, plan.model)
    seen, forward = [], engine.forward

    def recording(batch, records):
        seen.extend(r["name"] for r in records if "ops" not in r and "stop" not in r)
        return forward(batch, records)
    monkeypatch.setattr(engine, "forward", recording)
    filled = engine.execute(plan, batch_size=BATCH_SIZE)
    assert sorted(set(seen)) == ["logits@None", "v_cf@0"]
    assert same(filled.results, engine.execute(plan_of("patching"), batch_size=BATCH_SIZE).results)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_an_earlier_entry_at_the_writes_own_site_is_its_operand(engines, kind):
    """`h` reads layer 0 at the last token, then `w` adds twice `s.h` at the same site: `s.h` keeps the value
    from before the write (the clean one), `w` is `h + 2 h` to the bit and is what the site holds after it, on
    both engines. An operand listed after the write is refused."""
    entry = {"site": "target", "pos": -1}

    def change(d):
        d["steps"]["clean"] = {"kind": "forward", "input": "pairs.input", "interventions": {"h": entry}}
        d["steps"]["s"] = {"kind": "forward", "input": "pairs.input", "interventions": {
            "h": entry, "w": {**entry, "ops": [{"op": "add", "with": "s.h", "scale": 2.0}]}, "after": entry}}
        d["saves"] = ["clean.h", "s.h", "s.w", "s.after"]
    plan = edited("patching", change).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert torch.equal(values["s.h"], values["clean.h"])
    assert torch.equal(values["s.w"], edits.add(values["s.h"], values["s.h"], scale=2.0))
    assert torch.equal(values["s.after"], values["s.w"])

    def later(d):
        change(d)
        d["steps"]["s"]["interventions"] = {"w": {**entry, "ops": [{"op": "add", "with": "s.h", "scale": 2.0}]},
                                            "h": entry}
    with pytest.raises(PlanError, match=r"'s\.h' is this entry or one this forward makes after it"):
        edited("patching", later).compile()
