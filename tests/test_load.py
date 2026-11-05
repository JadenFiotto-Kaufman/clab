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
from conftest import BATCH_SIZE, ROOT, TRIP, document, plan_of, same, stand_in
from safetensors import safe_open

from clab import Plan
from clab.cli import main
from clab.format.sweeps import points
from clab.output import load, manifest
from clab.plan import Load, PlanError, entries

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


def test_an_eval_members_read_loads_as_the_eval_step_read_it(engines, root):
    """A fit's eval is steps after it, so a member of one is a value of an ordinary step: lora_train's held-out
    logits, saved and loaded from its run, carry the eval forward's read of `test.input` (named after the load,
    with its positions under `test`), and a cross entropy over them gives the eval's own `fit.held_out.ce`, to the
    bit."""
    ran(engines, root, "lora_train",
        corpus("lora_train", lambda doc: doc["saves"].append("fit.held_out.adapted.logits")))
    plan = written(root, "held_from_cache", loading(corpus("lora_train"), {
        "held": {"kind": "load", "from": "out/lora_train", "value": "fit.held_out.adapted.logits"},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "held", "target": "test.cf_answer"}}, ["ce"]))
    step = cast(Load, plan.steps["held"])
    assert step.read == "held.logits" and step.source is not None and step.source.input == "test.input"
    assert list(entries(step.source)["logits"].positions) == ["test"]
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    assert same(filled.results["ce"], load(os.path.join(root, "out", "lora_train")).results["fit.held_out.ce"])



def generated(engines, root: str) -> str:
    """lora_train with a generation of two tokens in its body, so in each eval, and the same written at the root
    after the fit over `test`, both saved: run once, into `out/lora_gen`."""
    def change(doc):
        gen = {"kind": "generate", "input": "train.input", "max_new_tokens": 2, "do_sample": False}
        doc["steps"]["fit"]["steps"]["gen"] = gen
        doc["steps"]["r_gen"] = {**gen, "input": "test.input"}
        doc["saves"] += ["fit.held_out.gen", "r_gen"]
    out = os.path.join(root, "out", "lora_gen")
    if not os.path.exists(out):
        ran(engines, root, "lora_gen", corpus("lora_train", change))
    return out


def test_a_load_of_an_eval_generation_is_refused_as_the_roots_is(engines, root):
    """An eval's generation is made by its eval step, a generate step like the root's, so a load of
    `fit.held_out.gen` is refused as a load of the same generation at the root is, where the fit was taken as its
    maker and the load was compiled."""
    generated(engines, root)
    said = ("is a generation's value or holds its positions, which count the tokens that run generated; a load brings "
            "a forward's values and what op steps and fits made of them")
    for value in ("fit.held_out.gen", "r_gen"):
        assert refused(root, {"kind": "load", "from": "out/lora_gen", "value": value}) == \
            f"steps.harvest.value: {value!r} {said}"


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


def test_a_loaded_whole_scope_whose_rows_disagree_is_refused_at_compile_with_their_counts(engines, root):
    """A run saves entity_patch's counterfactual read over its whole scope, three tokens on its Thursday row and one
    on the others, which pairs with nothing there. Loaded as the write's operand, its rows' positions come with it,
    so compile already knows the counts and refuses the pair with them, as preflight does an unloaded read's."""
    def whole(doc):
        doc["steps"] = {"counterfactual": doc["steps"]["counterfactual"]}
        del doc["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"]["index"]
        doc["saves"] = ["counterfactual.v_cf"]
    ran(engines, root, "whole_scope", corpus("entity_patch", whole))

    def change(doc):
        doc["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "counterfactual"
        doc["steps"]["counterfactual"] = {"kind": "load", "from": "out/whole_scope", "value": "counterfactual.v_cf"}
        doc["saves"] = ["iia"]
    with pytest.raises(PlanError) as e:
        written(root, "patched_from_whole_scope", corpus("entity_patch", change))
    assert str(e.value) == (
        "steps.patched.interventions.patch.ops[0].with: 'counterfactual' holds 1 position on rows 0-1, 3 and 3 "
        "positions on row 2; a scoped read pairs with another forward's value when every row's scope holds the same "
        "number of tokens: pick one with 'index' (-1, or '-2:'), or reduce it")


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


def test_a_loads_value_travels_inside_the_plan_and_the_run_reads_no_file(engines, root, monkeypatch):
    """Under `remote="local"` and through the stand-in server the request carries the load's value inside the plan,
    as it carries a bundle's tensors: the plan the server deserializes holds the client's value, no run reads a
    file, and the basis is the nnterp engine's to the bit. A load costs no model call, and the request's value does
    not come home: the client keeps its own."""
    from nnsight.schema.request import RequestModel

    from clab import output
    plan = from_cache(root)
    here = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results.copy()
    local, served = from_cache(root), from_cache(root)
    mine = cast(Load, served.steps["harvest"])
    sent, serialize = [], RequestModel.serialize

    def kept(cls, tracer, compress=False):
        sent.append(serialize(tracer, compress))
        return sent[-1]
    monkeypatch.setattr(output, "read_saved_value", lambda *args: pytest.fail("the run read a file"))
    there = engines("ndif", local.model, remote="local").execute(local, batch_size=BATCH_SIZE)
    assert same(there.results, here)
    assert {k: v for k, v in there.run.costs["harvest"].items() if k != "seconds"} == \
        {"memory": None, "forwards": 0, "tokens": 0, "real": 0}
    monkeypatch.setattr(RequestModel, "serialize", classmethod(kept))
    home, trip = stand_in(engines, served, monkeypatch)
    assert trip == TRIP and same(home.results, here)
    shipped = RequestModel.deserialize(sent[0], engines("nnterp", served.model).model._remoteable_persistent_objects(),
                                       compress=False).info.frame.f_locals["plan"]
    assert mine.loaded is not None and same(cast(Load, shipped.steps["harvest"]).loaded, mine.loaded)
    assert home.steps["harvest"] is mine


def test_a_request_carries_at_most_512_mib_of_loaded_values(engines, root, monkeypatch):
    """A document whose loads hold more than 512 MiB of tensors (stubbed: the tiny model's harvest holds a few KiB)
    is refused by name before anything is sent, under either `remote`, with each load's size; the nnterp engine,
    where the file is, runs it."""
    from clab import steps
    monkeypatch.setattr(steps, "nbytes", lambda value: 600 * 2**20)
    for remote in (True, "local"):
        plan = from_cache(root)
        with pytest.raises(ValueError) as e:
            engines("ndif", plan.model, remote=remote).execute(plan, batch_size=BATCH_SIZE)
        assert str(e.value) == (
            "harvest: the values this document loads hold 600.0 MiB of tensors (harvest 600.0 MiB), and a request "
            "carries at most 512 MiB inside its plan; run it where the file is, on the nnterp or hooks engine")
    plan = from_cache(root)
    assert "basis" in engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results


def test_a_compiled_plan_read_back_reads_its_loads_values_again(root):
    """A compiled plan written as JSON holds no tensor, and read back, or rebuilt from its fields, reads each load's
    value again from its run's file at each point, as it reads a bundle's; a run's plan (`Plan.load`) keeps the
    load's record alone."""
    compiled = written(root, "swept_back", loading(corpus("pca_harvest"), {"harvest": {**LOAD, "from": "out/swept"}},
                                                   ["harvest"]))
    path = os.path.join(root, "documents", "swept_back.compiled.json")
    compiled.write(path)
    with open(path) as f:
        assert all("loaded" not in p for p in json.load(f)["steps"]["harvest"])
    for again in (Plan(path), Plan(**compiled.model_dump())):
        assert [same(p.loaded, q.loaded) for p, q in zip(points(again.steps["harvest"]),
                                                         points(compiled.steps["harvest"]))] == [True, True]
    assert cast(Load, load(os.path.join(root, "out", "chained")).steps["cached"]).loaded is None


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


def test_an_op_over_a_loaded_top_k_value_is_refused_at_compile(engines, root):
    """A load of readout.json's `top3` brings its whole value, several tensors, which an op over it met at the run with
    a raw error: refused by name at compile, as the run that saved it recorded the value."""
    ran(engines, root, "readout", corpus("readout"))
    doc = loading(corpus("readout"), {"top": {"kind": "load", "from": "out/readout", "value": "top3"},
                                      "m": {"kind": "op", "op": "mean", "of": "top"}}, ["m"])
    with pytest.raises(PlanError, match=r"^steps\.m\.of: 'top' is a top_k's value, several tensors \(its ids, their probs "
                                        r"and their tokens\), and 'mean' takes one: rank what it needs with an op over "
                                        r"the value the top_k ranks \(argmax gives the first of its ids\)$"):
        written(root, "loaded_top", doc)


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


def test_a_manifest_from_before_value_axes_is_refused(root, capsys):
    """A manifest whose value records hold no `axes` (written by an older clab) cannot say which sweeps a value
    depends on; taken as unswept, a swept value's points would load as one value, so `Plan.load`, `clab show`,
    `clab table` and a load from it refuse it by name, in the one place that reads a manifest."""
    old = os.path.join(root, "out", "noaxes")
    shutil.copytree(os.path.join(root, "out", "pca_harvest"), old, dirs_exist_ok=True)
    with open(os.path.join(old, "manifest.json")) as f:
        held = json.load(f)
    for record in held["results"].values():
        del record["axes"]
    with open(os.path.join(old, "manifest.json"), "w") as f:
        json.dump(held, f)
    said = ("manifest.json predates the sweep axes a saved value's record holds ('axes'), so it does not say which "
            "sweeps its values depend on; rerun the document that wrote it")
    with pytest.raises(ValueError) as e:
        Plan.load(old)
    assert str(e.value) == f"{old}/{said}"
    for command in (["show", old], ["table", old, "harvest.acts"]):
        assert main(command) == 1 and capsys.readouterr().err == f"clab: {old}/{said}\n"
    assert refused(root, {**LOAD, "from": "out/noaxes"}) == f"steps.harvest.from: out/noaxes/{said}"


def test_a_run_directory_without_its_tokens_is_refused_by_name_where_a_load_still_reads_it(root, capsys):
    """A run this clab writes keeps its tokens in `tokens.json` beside the manifest; a directory with neither that
    file nor tokens in the manifest would load with no tokens and table with no `pos_token`, so `Plan.load`, `clab
    show` and `clab table` refuse it by name. A load step reads the manifest and a value's file alone, so it still
    reads such a directory."""
    old = os.path.join(root, "out", "notokens")
    shutil.copytree(os.path.join(root, "out", "pca_harvest"), old, dirs_exist_ok=True)
    os.remove(os.path.join(old, "tokens.json"))
    said = (f"{old}/tokens.json is missing, and the manifest holds no tokens; a run this clab wrote has one beside "
            "its manifest, so rerun the document that wrote it")
    with pytest.raises(FileNotFoundError) as e:
        Plan.load(old)
    assert str(e.value) == said
    for command in (["show", old], ["table", old, "harvest.acts"]):
        assert main(command) == 1 and capsys.readouterr().err == f"clab: {said}\n"
    held, described = manifest(old)
    assert "harvest.acts" in described and held["run"] is not None



def test_a_manifest_from_before_a_fits_evals_were_steps_is_refused_naming_the_fit(engines, root, capsys):
    """A manifest whose fit has an `eval` and no `evals` (written by an older clab) holds no record of an eval's
    values, so `Plan.load`, `clab show` and `clab table` through it, and a load from it refuse it by name, naming
    the fit, as a manifest from before the sweep axes is refused, where each raised a bare KeyError on the eval's
    label."""
    old = os.path.join(root, "out", "noevals")
    shutil.copytree(generated(engines, root), old, dirs_exist_ok=True)
    with open(os.path.join(old, "manifest.json")) as f:
        held = json.load(f)
    del held["steps"]["fit"]["evals"]
    with open(os.path.join(old, "manifest.json"), "w") as f:
        json.dump(held, f)
    said = ("manifest.json predates a fit's evals as steps ('evals'), so its fit 'fit' does not hold its evals' "
            "records; rerun the document that wrote it")
    with pytest.raises(ValueError) as e:
        Plan.load(old)
    assert str(e.value) == f"{old}/{said}"
    for command in (["show", old], ["table", old, "fit.held_out.ce"]):
        assert main(command) == 1 and capsys.readouterr().err == f"clab: {old}/{said}\n"
    assert refused(root, {"kind": "load", "from": "out/noevals", "value": "fit.held_out.ce"}) == \
        f"steps.harvest.from: out/noevals/{said}"


def test_a_swept_value_loads_on_its_own_with_the_sweep_it_depends_on(engines, root):
    """A read swept over two layers loads in a document that sweeps nothing: the load declares the run's `layers`
    sweep, so the plan sweeps it, the load has a point at each layer, and its value is the run's, to the bit, keyed
    `layers=0` and `layers=1` in its file as the run's is."""
    plan = written(root, "swept_alone", loading(corpus("pca_harvest"), {"harvest": {**LOAD, "from": "out/swept"}},
                                                ["harvest"]))
    assert {label: axis.values for label, axis in plan.sweeps.items()} == {"layers": [0, 1]}
    assert [(p.at, p.layers) for p in points(plan.steps["harvest"])] == [({"layers": 0}, [0]), ({"layers": 1}, [1])]
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "out", "swept_alone"))
    assert same(filled.results["harvest"], load(os.path.join(root, "out", "swept")).results["harvest.acts"])
    files = {}
    for out, name in (("swept_alone", "harvest"), ("swept", "harvest.acts")):
        with safe_open(os.path.join(root, "out", out, f"{name}.safetensors"), "pt") as f:
            files[out] = {k: f.get_tensor(k) for k in f.keys()}
    assert sorted(files["swept_alone"]) == ["layers=0", "layers=1"] and same(files["swept_alone"], files["swept"])


def test_a_sweep_under_the_loads_label_over_other_values_is_refused(root):
    other = {"kind": "forward", "input": "prompts.input", "interventions": {
        "acts": {"site": {"component": "layer_output", "layers": {"sweep": [1, 0]}}, "pos": -1}}}
    with pytest.raises(PlanError) as e:
        written(root, "refused", loading(corpus("pca_harvest"), {
            "harvest": {**LOAD, "from": "out/swept"}, "other": other}, ["harvest"]))
    assert str(e.value) == (
        "steps.harvest.value: 'harvest.acts' is swept over 'layers', [0, 1] in out/swept, and this document sweeps "
        "'layers' over [1, 0]; a load declares the sweeps its value depends on, and a sweep of this document under "
        "the same label has the same values")


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


def test_a_value_a_reduction_took_out_of_a_featurizers_coordinates_loads(engines, root):
    """T10's repro: a per-token L0 (`[encode, gt 0, sum over features]`, laid out `[rows, pos]`) is in no
    featurizer's coordinates once the sum removed the axis they lie on, so a load of it compiles, and its mean is
    the harvesting run's mean of it, to the bit; the latents themselves, still in the coordinates, are refused."""
    site, pos = {"component": "layer_output", "layers": 0}, {"index": "1:"}
    made = ran(engines, root, "l0", loading(corpus("pca_harvest"), {
        "harvest": {"kind": "forward", "input": "prompts.input", "interventions": {
            "l0": {"site": site, "pos": pos, "ops": [{"op": "encode", "featurizer": "sae"}, {"op": "gt", "with": 0},
                                                     {"op": "sum", "over": ["features"]}]},
            "latents": {"site": site, "pos": pos, "ops": [{"op": "encode", "featurizer": "sae"}]}}},
        "m": {"kind": "op", "op": "mean", "of": "harvest.l0"}}, ["harvest.l0", "harvest.latents", "m"],
        featurizers={"sae": {"kind": "sae", "latents": 32, "seed": 0}}))
    l0 = next(iter(entries(made.steps["harvest"]).values()))
    assert l0.dims == ["rows", "pos"] and l0.space == [] and l0.ops[0].space == [{"featurizer": "sae"}]
    plan = written(root, "l0_mean", loading(corpus("pca_harvest"), {
        "l0": {"kind": "load", "from": "out/l0", "value": "harvest.l0"},
        "m": {"kind": "op", "op": "mean", "of": "l0"}}, ["m"]))
    assert same(engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results["m"], made.results["m"])
    assert refused(root, {"kind": "load", "from": "out/l0", "value": "harvest.latents"}) == (
        "steps.harvest.value: 'harvest.latents' is in the coordinates of 'sae', a featurizer of out/l0's; load the "
        "value it encodes and encode it here")


def test_a_document_of_loads_and_op_steps_loads_no_weights(engines, root):
    """T10's P5: a document whose steps are a load and op steps calls no model, so a fresh nnterp engine stays the
    meta shell it loads as (the tokenizer and the config, no weights) and gives the numbers of an engine that has
    its weights, to the bit; a document with a forward loads them before its first step."""
    from clab.engines import load as load_engine

    plan = written(root, "mean_of_loaded", loading(corpus("pca_harvest"), {
        "acts": LOAD, "m": {"kind": "op", "op": "mean", "of": "acts", "over": ["rows", "pos"]}}, ["m"]))
    fresh = load_engine("nnterp", plan.model)
    shell: Any = fresh.model
    filled = fresh.execute(plan, batch_size=BATCH_SIZE)
    assert not shell.dispatched and filled.run is not None and filled.run.devices == ["meta"]
    assert same(filled.results["m"], engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results["m"])
    fresh.execute(plan_of("patching"), batch_size=BATCH_SIZE)
    assert shell.dispatched and not any(p.requires_grad for p in shell.parameters())


def test_a_document_of_loads_and_op_steps_under_a_device_map_loads_the_weights(root):
    """Under a device map (`auto`) only the loaded weights say where the run's tensors go, so a document of loads and
    op steps loads them before its first step and its load's value goes where the map put the first parameter, where
    the meta shell's device, `auto`, is no torch device and the load step crashed in `Tensor.to`."""
    from clab.engines import load as load_engine

    plan = written(root, "mean_of_loaded_auto", loading(corpus("pca_harvest"), {
        "acts": LOAD, "m": {"kind": "op", "op": "mean", "of": "acts", "over": ["rows", "pos"]}}, ["m"]))
    filled = load_engine("nnterp", plan.model, "auto").execute(plan, batch_size=BATCH_SIZE)
    assert filled.run is not None and filled.run.devices == ["cpu"]
