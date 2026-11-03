"""A value one run saved, loaded into another document by a `load` step: every step after it treats it as the
value it was, so a loading document gives the numbers of the document that made the value to the bit, on every
engine and under `remote="local"`; and what a load cannot mean is refused by name before any forward.

The runs a loading document reads are made first, on the nnterp engine, into `out/` under a corpus root of this
module's own: `pca_harvest.json` and `sae_train.json` saving their harvests too, `patching.json` saving its
counterfactual read, a read swept over two layers, a read of chats, and a run that itself loads pca_harvest's
harvest (with an op's value over it).
"""

import json
import os
import shutil
from typing import Any, cast

import pytest
from conftest import BATCH_SIZE, ROOT, document, same
from safetensors import safe_open

from clab import Plan
from clab.format.sweeps import points
from clab.output import load
from clab.plan import Load, PlanError

LOAD = {"kind": "load", "from": "out/pca_harvest", "value": "harvest.acts"}


def written(root: str, name: str, doc: dict[str, Any]) -> Plan:
    """`doc` written as `documents/<name>.json` under `root`, and read back from there, compiled."""
    path = os.path.join(root, "documents", f"{name}.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return Plan(path).compile()


def corpus(name: str, change=None) -> dict[str, Any]:
    """Corpus document `name` as JSON, with `change` applied."""
    with open(document(name)) as f:
        doc = json.load(f)
    if change is not None:
        change(doc)
    return doc


def loading(doc: dict[str, Any], steps: dict[str, Any], saves: list[str], **more: Any) -> dict[str, Any]:
    """A document on `doc`'s model and data with `steps` and `saves`."""
    return {"model": doc["model"], "data": doc["data"], "steps": steps, "saves": saves, **more}


def ran(engines, root: str, name: str, doc: dict[str, Any], kind: str = "nnterp", **options: Any) -> Plan:
    """`doc` written under `root`, run on engine `kind`, and its output written to `out/<name>`."""
    plan = written(root, name, doc)
    filled = engines(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "out", name))
    return filled


def harvested(doc: dict[str, Any]) -> None:
    doc["saves"] = ["harvest.acts", *doc["saves"]]


@pytest.fixture(scope="module")
def root(tmp_path_factory, engines) -> str:
    """A corpus root with the corpus's data and, under `out/`, the runs the loading documents read."""
    here = tmp_path_factory.mktemp("load")
    os.symlink(os.path.join(ROOT, "data"), here / "data")
    os.makedirs(here / "documents")
    here = str(here)
    ran(engines, here, "pca_harvest", corpus("pca_harvest", harvested))
    ran(engines, here, "sae_train", corpus("sae_train", harvested))
    ran(engines, here, "patching", corpus("patching", lambda doc: doc.update(saves=["counterfactual.v_cf", "iia"])))
    swept = loading(corpus("pca_harvest"), {"harvest": {"kind": "forward", "input": "prompts.input", "interventions": {
        "acts": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1]}}, "pos": -1}}}}, ["harvest.acts"])
    ran(engines, here, "swept", swept)
    ran(engines, here, "chat", CHAT)
    ran(engines, here, "chained", loading(corpus("pca_harvest"), {
        "cached": LOAD, "m": {"kind": "op", "op": "mean", "of": "cached", "over": ["pos"]}}, ["cached", "m"]))
    return here


# A read of each chat's last token, its chats rendered ending in the generation prompt.
CHAT = {**corpus("chat_turn"), "data": {"chats": {"path": "weekdays/chat", "template": {"add_generation_prompt": True}}},
        "sites": {}, "saves": ["read.acts"], "steps": {"read": {
            "kind": "forward", "input": "chats.chat",
            "interventions": {"acts": {"site": {"component": "layer_output", "layers": 0}, "pos": -1}}}}}


def from_cache(root: str) -> Plan:
    """pca_harvest.json's basis, taken over its harvest loaded from its run."""
    return written(root, "from_cache", loading(corpus("pca_harvest"), {
        "harvest": LOAD, "basis": {"kind": "op", "op": "pca", "of": "harvest", "k": 4}}, ["basis", "harvest"]))


def control(basis: str) -> dict[str, Any]:
    """pca_control.json through the basis at `basis`."""
    return corpus("pca_control", lambda doc: doc["featurizers"]["rot"].update(file_path=basis))


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_loaded_harvest_gives_pca_control_its_numbers_to_the_bit(engines, root, kind, monkeypatch):
    """The basis a `pca` takes over a loaded harvest is the one pca_harvest.json took over its own, to the bit and
    with its header (the site, the layer and the width come with the load's record), and pca_control.json through
    it scores exactly what it scores through the original; the loaded value is the saved one, and `table` labels
    its positions as the source run's labels its read."""
    plan = from_cache(root)
    filled = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "out", f"from_cache_{kind}"))
    source = load(os.path.join(root, "out", "pca_harvest"))
    assert same(filled.results["basis"], source.results["basis"])
    assert same(filled.results["harvest"], source.results["harvest.acts"])

    def header(out: str) -> dict:
        with safe_open(os.path.join(root, out, "basis.safetensors"), "pt") as f:
            return {k: v for k, v in (f.metadata() or {}).items() if k != "document"}
    assert header(f"out/from_cache_{kind}") == header("out/pca_harvest")
    through = {}
    for out in ("pca_harvest", f"from_cache_{kind}"):
        doc = written(root, f"control_{out}", control(f"out/{out}/basis.safetensors"))
        through[out] = engines(kind, doc.model).execute(doc, batch_size=BATCH_SIZE).results.copy()
    assert same(through["pca_harvest"], through[f"from_cache_{kind}"])
    monkeypatch.chdir(root)                 # a table reads the data files again, from the corpus root
    tabled = load(os.path.join("out", f"from_cache_{kind}")).table("harvest")
    assert tabled == source.table("harvest.acts") and {"▁If", "▁today"} <= {row["pos_token"] for row in tabled}


def test_a_fit_takes_a_loaded_harvests_rows_by_id(engines, root):
    """sae_train.json with its harvest loaded from its own run, the body encoding the loaded value: the dictionary
    and both histories are the ones the document that harvested trained, to the bit."""
    def change(doc):
        doc["steps"]["harvest"] = {**LOAD, "from": "out/sae_train"}
        doc["steps"]["fit"]["steps"]["latents"]["of"] = "harvest"
    plan = written(root, "sae_from_cache", corpus("sae_train", change))
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    source = load(os.path.join(root, "out", "sae_train"))
    for name in ("fit.dictionary", "fit.history.dictionary.reconstruction", "fit.history.dictionary.l1"):
        assert same(filled.results[name], source.results[name]), name


def test_a_chained_load_encodes_as_the_read_it_carries(engines, root):
    """`cached`, a run's load of pca_harvest's harvest, loaded again from that run: its record is the entry it
    carries, the one pca_harvest read, so an `encode` of it runs, and gives what the encode of the harvest loaded
    directly gives, to the bit. An op's value over it, loaded, is not a read's own, and its encode is refused."""
    def encoding(step: dict[str, Any]) -> Plan:
        return written(root, "encoding", loading(corpus("pca_harvest"), {
            "again": step, "enc": {"kind": "op", "op": "encode", "of": "again", "featurizer": "pc"}}, ["enc"],
            featurizers={"pc": {"kind": "subspace", "k": 2, "seed": 0}}))
    got = {}
    for how, step in (("chained", {"kind": "load", "from": "out/chained", "value": "cached"}), ("direct", LOAD)):
        plan = encoding(step)
        got[how] = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results["enc"]
    assert same(got["chained"], got["direct"])
    with pytest.raises(PlanError, match=r"^steps\.enc\.of: 'again' is an op's value, which no one site made, and a "
                                        r"featurizer's encode acts on a value at its site"):
        encoding({"kind": "load", "from": "out/chained", "value": "m"})


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_loaded_counterfactual_patches_as_the_read_it_was(engines, root, kind):
    """patching.json with its counterfactual read loaded from its own run, the write's operand the loaded value:
    the interchange scores what the document that read it scores, to the bit."""
    def change(doc):
        doc["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "counterfactual"
        doc["steps"]["counterfactual"] = {"kind": "load", "from": "out/patching", "value": "counterfactual.v_cf"}
        doc["saves"] = ["iia"]
    plan = written(root, "patched_from_cache", corpus("patching", change))
    filled = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    assert same(filled.results["iia"], load(os.path.join(root, "out", "patching")).results["iia"])


def test_a_loaded_chat_read_renders_with_the_runs_template(engines, root, monkeypatch):
    """A read of chats loads where the dataset renders them with the template the run did, and `table` labels it
    with the tokens of those renderings; another template is refused, since the rows' digest does not cover it."""
    plan = written(root, "chat_from_cache", {**CHAT, "steps": {"read": {**LOAD, "from": "out/chat",
                                                                         "value": "read.acts"}}, "saves": ["read"]})
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "out", "chat_from_cache"))
    monkeypatch.chdir(root)
    source = load(os.path.join("out", "chat"))
    assert load(os.path.join("out", "chat_from_cache")).table("read") == source.table("read.acts")
    assert refused(root, {**LOAD, "from": "out/chat", "value": "read.acts"}, data={"chats": {"path": "weekdays/chat"}},
                   model=CHAT["model"]) == (
        "steps.harvest.value: 'read.acts' was made over the rows of dataset 'chats', whose chats out/chat rendered "
        "with template {'add_generation_prompt': True}, and this document's 'chats' has None")


def test_a_load_under_remote_local_agrees_and_carries_no_tensor(engines, root):
    """Under `remote="local"` the request holds the load's record without its value, and the run reads the file
    where it runs: the same basis as on nnterp, to the bit. A load costs no model call."""
    plan = from_cache(root)
    here = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results.copy()
    plan = from_cache(root)
    assert cast(Load, plan.steps["harvest"]).loaded is not None
    there = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    assert same(there.results, here)
    assert cast(Load, there.steps["harvest"]).loaded is None
    assert {k: v for k, v in there.run.costs["harvest"].items() if k != "seconds"} == \
        {"memory": None, "forwards": 0, "tokens": 0, "real": 0}


def test_a_load_is_refused_by_an_ndif_server_before_anything_is_sent(engines, root):
    plan = from_cache(root)
    with pytest.raises(ValueError, match=r"^harvest: a load reads out/pca_harvest where the run is, and an NDIF "
                                         r"server cannot see this machine's directories; run the document on the "
                                         r"nnterp or hooks engine, or under remote=\"local\"$"):
        engines("ndif", plan.model, remote=True).execute(plan, batch_size=BATCH_SIZE)


def test_a_swept_value_loads_at_each_point_of_its_axes(engines, root):
    """A read swept over two layers loads at both points of a document that sweeps `layers` over the same values,
    each point the saved one; its record names the layer of each."""
    doc = loading(corpus("pca_harvest"), {
        "harvest": {**LOAD, "from": "out/swept"},
        "again": {"kind": "forward", "input": "prompts.input", "interventions": {
            "acts": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1]}}, "pos": -1}}}},
        ["harvest", "again.acts"])
    plan = written(root, "swept_from_cache", doc)
    assert [p.layers for p in points(plan.steps["harvest"])] == [[0], [1]]
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    source = load(os.path.join(root, "out", "swept"))
    assert same(filled.results["harvest"], source.results["harvest.acts"])
    assert same(filled.results["harvest"], filled.results["again.acts"])


def refused(root: str, step: dict[str, Any], **more: Any) -> str:
    """The compile refusal of a document loading with `step`, on pca_harvest.json's model and data."""
    doc = {**loading(corpus("pca_harvest"), {"harvest": step}, ["harvest"]), **more}
    with pytest.raises(PlanError) as e:
        written(root, "refused", doc)
    return str(e.value)


def test_a_load_from_another_model_is_refused_naming_the_key(root):
    model = {**corpus("pca_harvest")["model"], "attn_implementation": "eager"}
    assert refused(root, LOAD, model=model) == (
        "steps.harvest.from: out/pca_harvest ran another model than this document's: attn_implementation None where "
        "this document's is 'eager'")


def test_a_load_over_other_rows_is_refused_by_their_digest(root):
    """One row's text edited, as many rows: the digest says they are not the rows the run read."""
    with open(os.path.join(ROOT, "data", "weekdays", "train.json")) as f:
        rows = json.load(f)
    rows[1] = {**rows[1], "input": rows[1]["input"] + " "}
    assert refused(root, LOAD, data={"prompts": {"rows": rows}}) == (
        "steps.harvest.value: 'harvest.acts' was made over the rows of dataset 'prompts', and this document's "
        "'prompts' is not the rows out/pca_harvest read, in their order: as many rows, whose digest differs")
    assert refused(root, LOAD, data={"prompts": {"rows": rows[:3]}}).endswith(": 3 rows where it read 4")
    assert refused(root, LOAD, data={"other": {"rows": rows}}) == (
        "steps.harvest.value: 'harvest.acts' was made over the rows of dataset 'prompts', which this document does "
        "not declare; declare it as out/pca_harvest did")


def test_a_name_the_run_did_not_save_and_a_directory_with_no_run_are_refused(root):
    assert refused(root, {**LOAD, "value": "harvest.nope"}) == (
        "steps.harvest.value: out/pca_harvest saved no 'harvest.nope'; it saved harvest.acts, basis")
    assert refused(root, {**LOAD, "from": "out/nothing"}) == (
        "steps.harvest.from: no run at out/nothing under the corpus root (no manifest.json); run the document that "
        "saves the value there first, with --out out/nothing")


def test_a_manifest_from_before_value_axes_is_refused(root):
    """A manifest whose value records hold no `axes` (written by an older clab) cannot say which sweeps a value
    depends on; taken as unswept, a swept value's points would load as one value, so it is refused by name."""
    old = os.path.join(root, "out", "noaxes")
    shutil.copytree(os.path.join(root, "out", "pca_harvest"), old, dirs_exist_ok=True)
    with open(os.path.join(old, "manifest.json")) as f:
        held = json.load(f)
    for record in held["results"].values():
        del record["axes"]
    with open(os.path.join(old, "manifest.json"), "w") as f:
        json.dump(held, f)
    assert refused(root, {**LOAD, "from": "out/noaxes"}) == (
        "steps.harvest.from: out/noaxes/manifest.json predates the sweep axes a saved value's record holds ('axes'), "
        "so it does not say which sweeps 'harvest.acts' depends on; rerun the document that saves it, with --out "
        "out/noaxes")


def test_a_swept_value_without_its_sweep_is_refused(root):
    assert refused(root, {**LOAD, "from": "out/swept"}) == (
        "steps.harvest.value: 'harvest.acts' is swept over 'layers', [0, 1] in out/swept, and this document declares "
        "no sweep labelled 'layers'; a swept value loads at each of its points, so a sweep of this document is "
        "labelled 'layers' with those values")
    other = {"kind": "forward", "input": "prompts.input", "interventions": {
        "acts": {"site": {"component": "layer_output", "layers": {"sweep": [1, 0]}}, "pos": -1}}}
    with pytest.raises(PlanError, match=r"^steps\.harvest\.value at layers=1: 'harvest\.acts' is swept over "
                                        r"'layers', \[0, 1\] in out/swept, and this document sweeps 'layers' over "
                                        r"\[1, 0\]; "):
        written(root, "refused", loading(corpus("pca_harvest"), {
            "harvest": {**LOAD, "from": "out/swept"}, "other": other}, ["harvest"]))


def test_a_bundle_and_a_load_inside_a_fit_are_refused(root):
    assert refused(root, {**LOAD, "value": "basis"}) == (
        "steps.harvest.value: 'basis' is a bundle, which a featurizer of its kind loads, with 'file_path': "
        "'out/pca_harvest/basis.safetensors'")
    doc = corpus("sae_train")
    doc["steps"]["fit"]["steps"]["harvest"] = {**LOAD, "from": "out/sae_train"}
    with pytest.raises(PlanError, match=r"^steps\.fit\.steps\.harvest: a fit's body holds no load: load the value "
                                        r"before the fit"):
        written(root, "refused", doc)


def test_a_load_names_its_fields_as_a_document_writes_them():
    with pytest.raises(PlanError, match=r"^steps\.harvest: unknown field 'from_'; valid: kind, from, value$"):
        Plan(**{**corpus("pca_harvest"), "steps": {"harvest": {"kind": "load", "from_": "out/x", "value": "y"}}})
    plan = Plan(**{**corpus("pca_harvest"), "steps": {"harvest": LOAD}, "saves": []})
    assert plan.model_dump(mode="json", exclude_defaults=True)["steps"]["harvest"] == LOAD
