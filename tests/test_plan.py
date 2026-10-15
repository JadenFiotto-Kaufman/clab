"""The plan: every document compiles to the plan its snapshot holds; the plan is pure data with views; errors
name the document path of what is wrong and the valid choices."""

import json
import os
import pickle
from typing import Any

import pytest

from clab import Plan
from clab.plan.model import PlanError, points
from conftest import NAMES, document, edited, plan_of, snapshot


def step(plan: Plan, name: str) -> Any:
    """An unswept step, untyped: the tests know its kind."""
    return plan.steps[name]


def compiled_form(plan: Plan) -> dict:
    """What a compiled plan holds, as written: rows aside (they are the data files')."""
    dump = plan.model_dump(mode="json", exclude_defaults=True)
    for d in dump["data"].values():
        if "path" in d:
            del d["rows"]
    return dump


def test_the_corpus_is_the_31_documents_and_the_examples_since():
    assert len(NAMES) == 31 + 4 and {"sae_train", "sae_train_ablation", "lora_train", "lora_apply"} <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_document_compiles_to_its_snapshot(name, corpus):
    plan = Plan(os.path.join(corpus, "documents", f"{name}.json")).compile()
    assert plan.compiled
    assert compiled_form(plan) == snapshot(f"plans/{name}.json", compiled_form(plan))
    assert '"sweep"' not in json.dumps(compiled_form(plan))


def test_a_missing_bundle_is_refused_naming_its_file(corpus, tmp_path):
    """In a copy of the corpus whose artifacts hold no basis (the live tree's may, once pca_harvest has run)."""
    import shutil
    shutil.copytree(os.path.join(corpus, "documents"), tmp_path / "documents")
    os.symlink(os.path.join(corpus, "data"), tmp_path / "data")
    os.remove(tmp_path / "documents" / "artifacts" / "basis.safetensors")
    with pytest.raises(PlanError, match="featurizers.rot.file_path: documents/artifacts/basis.safetensors does not exist"):
        Plan(str(tmp_path / "documents" / "pca_control.json")).compile()


def test_a_bundle_header_is_checked_key_by_key(corpus):
    with open(os.path.join(corpus, "documents", "pca_control.json")) as f:
        doc = json.load(f)
    doc["featurizers"]["rot"]["k"] = 8
    with pytest.raises(PlanError, match="header 'k' is '4'; this document expects '8'"):
        Plan(**doc, document=os.path.join(corpus, "documents", "pca_control.json")).compile()


def test_a_plan_pickles_and_rebuilds_from_its_dump():
    plan = plan_of("pos_sweep")
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert Plan(**plan.model_dump()) == plan


def test_a_root_featurizer_is_not_a_save():
    with pytest.raises(PlanError, match=r"saves\[0\]: 'rot' names no step's value"):
        edited("random_subspace", lambda d: d.update(saves=["rot"])).compile()


def test_a_compiled_plan_holds_no_sweep():
    with pytest.raises(PlanError, match="a compiled plan holds no sweep"):
        Plan(**{**Plan(document("pos_sweep")).model_dump(), "compiled": True})


def test_a_written_or_dumped_document_path_is_relative_to_the_corpus():
    plan = plan_of("patching")
    assert os.path.isabs(plan.document or "")
    assert plan.model_dump(mode="json")["document"] == "documents/patching.json"
    assert plan.data["pairs"].n_rows == 4


@pytest.mark.parametrize("name", ["pca_control", "das_apply"])
def test_a_compiled_plan_loads_its_bundles_again_from_their_files(corpus, name):
    path = os.path.join(corpus, "documents", f"{name}.json")
    compiled = Plan(path).compile()
    bundle = points(compiled.featurizers["rot"])[0].bundle
    assert bundle
    written = os.path.join(corpus, "documents", f"{name}.compiled.json")
    compiled.write(written)
    with open(written) as f:
        assert all("bundle" not in x for x in json.load(f)["featurizers"].values())   # no tensors in a document
    for again in (Plan(written), Plan(**compiled.model_dump())):
        assert again.compiled
        back = points(again.featurizers["rot"])[0].bundle
        assert back is not None and back.keys() == bundle.keys() and all(back[k].equal(bundle[k]) for k in bundle)


def test_repr_says_the_state():
    plan = Plan(document("pos_sweep"))
    assert repr(plan) == "Plan(pos_sweep, uncompiled)"
    assert repr(plan.compile()) == "Plan(pos_sweep, compiled, 3 points)"
    assert repr(plan_of("patching")) == "Plan(patching, compiled, 1 point)"


def test_write_keeps_the_document(tmp_path):
    plan = Plan(document("steer_renormalize"))
    plan.write(str(tmp_path / "copy.json"))
    again = Plan(str(tmp_path / "copy.json"))
    assert again.model_dump(exclude={"document"}) == plan.model_dump(exclude={"document"})
    plan_of("patching").write(str(tmp_path / "compiled.json"))
    assert Plan(str(tmp_path / "compiled.json")).compiled


def test_sweeps_become_points():
    plan = plan_of("pos_sweep")
    assert {label: axis.values for label, axis in plan.sweeps.items()} == {"pos": [-1, -2, -3]}
    assert not isinstance(plan.steps["counterfactual"], list)         # no sweep touches it: it runs once
    patched = plan.steps["patched"]
    assert isinstance(patched, list) and [p.at for p in patched] == [{"pos": 0}, {"pos": 1}, {"pos": 2}]
    assert [p.writes["patch"].pos for p in points(step(plan, "patched"))] == [-1, -2, -3]


def test_one_sweep_reached_from_two_steps_is_one_axis():
    plan = plan_of("head_patching")
    reads = [(p.reads["v_cf"].site.select or {})["num_heads"] for p in points(plan.steps["counterfactual"])]
    writes = [(p.writes["patch"].site.select or {})["num_heads"] for p in points(plan.steps["patched"])]
    assert reads == writes == [[h] for h in range(4)]


def test_a_reference_carries_its_producers_axes():
    plan = edited("dbm", lambda d: d["steps"].update(               # the step's only link to the swept fit
        accuracy={"kind": "op", "op": "mean", "of": "fit.held_out.iia"})).compile()
    assert [p.at for p in points(plan.steps["accuracy"])] == [{"mask.mask": i} for i in range(4)]
    assert [p.objective["mask.mask"] for p in points(plan.steps["fit"])] == [0.0001, 0.002, 0.01, 0.05]


def test_every_value_knows_its_dataset_and_layout():
    plan = plan_of("patching")
    assert step(plan, "patched").dataset == "pairs"
    assert step(plan, "patched").reads["logits"].dims == ["rows", "pos", "vocab_size"]
    assert (step(plan, "iia").dataset, step(plan, "iia").dims) == ("pairs", ["rows", "pos"])
    mean = step(plan_of("mean_ablation"), "mean")
    assert (mean.dataset, mean.dims) == ("prompts", ["hidden_size"])


def test_two_datasets_in_one_op_are_refused():
    def change(d):
        d["data"]["other"] = {"path": "weekdays/train"}
        d["steps"]["iia"]["expected"] = "other.cf_answer"
    with pytest.raises(PlanError, match=r"steps\.iia: 'patched\.logits' has a row per example of 'pairs' and "
                                        r"'other\.cf_answer' of 'other'; an op takes one dataset's rows"):
        edited("patching", change).compile()


def test_an_inline_site_and_a_named_intervention_are_inlined():
    steered = step(plan_of("steer_renormalize"), "steered_renormed")
    assert steered.interventions is None and list(steered.writes) == ["steer", "restore"]
    assert steered.writes["steer"].site.component == "layer_output" and steered.writes["steer"].site.layers == 0


def test_a_hub_dataset_is_declared_but_not_loaded_yet():
    def hub(d):
        d["data"]["pairs"] = {"hf": "org/name", "revision": "abc", "split": "train", "take": 4}
    with pytest.raises(PlanError, match="data.pairs.hf: hub datasets land later"):
        edited("patching", hub).compile()
    with pytest.raises(PlanError, match="data.pairs: a hub dataset is pinned"):
        edited("patching", lambda d: d["data"].update(pairs={"hf": "org/name"}))


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["iia"].update(op="matches"), r"steps\.iia: unknown op 'matches'; valid: swap, add"),
    (lambda d: d["sites"]["target"].update(component="block_output"),
     r"sites\.target: unknown component 'block_output'; valid: token_embeddings"),
    (lambda d: d["steps"]["patched"].update(field="input"),
     r"steps\.patched: unknown field 'field'; valid: kind, input, writes, reads, interventions"),
    (lambda d: d["steps"]["iia"].update(token_form="spelled"),
     r"steps\.iia: token_form: 'spelled' is not one of 'spaced', 'bare'"),
    (lambda d: d["steps"]["iia"].update(expect="pairs.cf_answer"),
     r"steps\.iia: 'match' takes no 'expect'; its arguments: expected, token_form"),
    (lambda d: d["steps"]["patched"].update(kind="trace"),
     r"steps\.patched: unknown step kind 'trace'; valid: forward, generate, op, fit"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"index": 1, "scope": {"turn": "user", "text": "x"}}),
     r"steps\.patched\.writes\.patch\.pos\.scope: a scope is one of \{'column'"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"frame": "prompt"}),
     r"steps\.patched\.writes\.patch\.pos\.frame: Input should be 'generated', 'all' or 'eos'$"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"frame": 1, "scope": {"turn": "user"}}),
     r"steps\.patched\.writes\.patch\.pos: scope turn 'user' is a chat turn of the prompt; it takes frame 0$"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"idx": -1}),
     r"steps\.patched\.writes\.patch\.pos: unknown field 'idx'; valid: frame, scope, index$"),
    (lambda d: d["sites"]["target"].update(layers="two"), r"sites\.target\.layers: Input should be 'all'$"),
    (lambda d: d["steps"]["patched"]["reads"]["logits"].update(site={"component": "nope"}),
     r"steps\.patched\.reads\.logits\.site: unknown component 'nope'; valid: token_embeddings, [a-z_, ]+, logits$"),
    (lambda d: d["data"]["pairs"].update(path=None, rows=[{"input": "x"}], split="train"),
     r"data\.pairs: 'split' keeps rows of a file or a hub dataset; inline 'rows' are the rows$"),
    (lambda d: d["sites"]["target"].update(component={"sweep": ["layer_output"]}),
     r"sites\.target\.component: a sweep cannot stand for 'component'"),
    (lambda d: d["data"]["pairs"].update(path="weekdays/train#test"), None),
])
def test_a_document_error_names_the_path_and_the_choices(change, message):
    if message is None:      # the `#split` form is gone: the path names a file that does not exist
        with pytest.raises(PlanError, match=r"data\.pairs\.path: no file data/weekdays/train#test\.json"):
            edited("patching", change).compile()
        return
    with pytest.raises(PlanError, match=message):
        edited("patching", change)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["counterfactual"]["reads"]["v_cf"].update(ops=[{"op": "mean"}]),
     r"steps\.counterfactual\.reads\.v_cf\.ops\[0\]: 'mean' aggregates over rows.*make it a step after the forward"),
    (lambda d: d["steps"]["iia"].update(expected="pairs.answers"),
     r"steps\.iia\.expected: dataset 'pairs' has no column 'answers'"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"]["ops"][0].update({"with": "counterfactual.v"}),
     r"steps\.patched\.writes\.patch\.ops\[0\]\.with: 'counterfactual\.v' names no value produced before this step; "
     r"known: counterfactual\.v_cf"),
    (lambda d: d.update(saves=["iia", "nothing"]), r"saves\[1\]: 'nothing' names no step's value; a save is a step "
                                                   r"or a step's member: counterfactual\.v_cf, iia, logit_diff, "
                                                   r"patched\.logits$"),
    (lambda d: d.update(saves=["iia", "iia"]), r"saves\[1\]: 'iia' is saved twice$"),
    (lambda d: d.update(saves=["pairs"]), r"saves\[0\]: 'pairs' is a dataset; hub datasets and dataset saves land later$"),
    (lambda d: d["sites"]["target"].update(layers="all"),
     r"steps\.patched\.writes\.patch\.site: a write names its layer"),
])
def test_a_compile_error_names_the_path(change, message):
    with pytest.raises(PlanError, match=message):
        edited("patching", change).compile()
