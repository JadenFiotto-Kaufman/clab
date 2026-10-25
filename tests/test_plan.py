"""The plan: every document compiles to the plan its snapshot holds; the plan is pure data with views; errors
name the document path of what is wrong and the valid choices."""

import json
import os
import pickle
from typing import Any

import pytest
from conftest import NAMES, REAL, document, edited, plan_of, snapshot

from clab import Plan
from clab.compile import positions
from clab.format import selectors
from clab.format.sweeps import points
from clab.plan import PlanError


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
    assert len(NAMES) == 31 + 13 and {"sae_train", "sae_train_ablation", "lora_train", "lora_apply", "noise_write",
                                      "clamp_write", "cosine_metric", "forced_answer", "answer_spellings",
                                      "generated_contains", "generated_confidence"} <= set(NAMES)
    assert REAL == ["real/behavior", "real/das", "real/head_patching", "real/layer_sweep", "real/pattern_patching"]


@pytest.mark.parametrize("name", NAMES + REAL)
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
    real = plan_of("real/behavior")         # a document in a directory under documents/ shares the corpus's data/
    assert real.model_dump(mode="json")["document"] == "documents/real/behavior.json"
    assert real.data["days"].n_rows == 42


@pytest.mark.parametrize("root, where", [
    ("", "documents/real/sub/x.json"),                  # deeper under documents/
    ("", "exps/x.json"),                                # under no documents/ at all
    ("documents/proj", "exps/x.json"),                  # a project inside someone's ~/documents
])
def test_the_corpus_root_is_the_nearest_directory_above_the_document_holding_data(tmp_path, root, where):
    """patching.json, written at `where` under `root`, beside the corpus's `data/`."""
    os.makedirs(tmp_path / root, exist_ok=True)
    os.symlink(os.path.join(os.path.dirname(os.path.dirname(document("patching"))), "data"), tmp_path / root / "data")
    path = tmp_path / root / where
    os.makedirs(path.parent)
    with open(document("patching")) as f:
        path.write_text(f.read())
    plan = Plan(str(path)).compile()
    assert plan.data["pairs"].n_rows == 4 and plan.model_dump(mode="json")["document"] == where


def test_a_document_with_no_data_folder_above_it_resolves_beside_its_own_directory(tmp_path):
    os.makedirs(tmp_path / "a" / "b")
    path = tmp_path / "a" / "b" / "x.json"
    with open(document("patching")) as f:
        path.write_text(f.read())
    assert Plan(str(path)).model_dump(mode="json")["document"] == "b/x.json"


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
    assert [p.interventions["patch"].pos for p in points(step(plan, "patched"))] == [-1, -2, -3]


def test_one_sweep_reached_from_two_steps_is_one_axis():
    plan = plan_of("head_patching")
    reads = [(p.interventions["v_cf"].site.select or {})["num_heads"] for p in points(plan.steps["counterfactual"])]
    writes = [(p.interventions["patch"].site.select or {})["num_heads"] for p in points(plan.steps["patched"])]
    assert reads == writes == list(range(4))


def test_a_reference_carries_its_producers_axes():
    plan = edited("dbm", lambda d: d["steps"].update(               # the step's only link to the swept fit
        accuracy={"kind": "op", "op": "mean", "of": "fit.held_out.iia"})).compile()
    assert [p.at for p in points(plan.steps["accuracy"])] == [{"mask.mask": i} for i in range(4)]
    assert [p.objective["mask.mask"] for p in points(plan.steps["fit"])] == [0.0001, 0.002, 0.01, 0.05]


def test_every_value_knows_its_dataset_and_layout():
    plan = plan_of("patching")
    assert step(plan, "patched").dataset == "pairs"
    assert step(plan, "patched").interventions["logits"].dims == ["rows", "pos", "vocab_size"]
    assert step(plan, "patched").interventions["patch"].dims == ["rows", "pos", "hidden_size"]
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
    """A step's `interventions` compile to one dict of entries in execution order: the bundles where their names
    stand, then the step's own."""
    steered = step(plan_of("steer_renormalize"), "steered_renormed")
    assert list(steered.interventions) == ["steer", "restore", "logits"]
    entry = steered.interventions["steer"]
    assert entry.site.component == "layer_output" and entry.site.layers == 0


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["iia"].update(op="matches"), r"steps\.iia: unknown op 'matches'; valid: swap, add"),
    (lambda d: d["sites"]["target"].update(component="block_output"),
     r"sites\.target: unknown component 'block_output'; valid: token_embeddings"),
    (lambda d: d["steps"]["patched"].update(field="input"),
     r"steps\.patched: unknown field 'field'; valid: kind, input, interventions"),
    (lambda d: d["steps"]["iia"].update(expect="pairs.cf_answer"),
     r"steps\.iia: 'match' takes no 'expect'; its arguments: expected$"),
    (lambda d: d["steps"]["patched"].update(kind="trace"),
     r"steps\.patched: unknown step kind 'trace'; valid: forward, generate, op, fit"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(
        pos={"index": 1, "scope": {"turn": "user", "text": "x"}}),
     r"steps\.patched\.interventions\.patch\.pos\.scope: a scope is one of \{'column'"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": "prompt"}),
     r"steps\.patched\.interventions\.patch\.pos\.frame: 'prompt' is not a selector: an int \(2, -1\), a slice "
     r"in Python notation \('2:5', '-3:', ':'\) or a list of picks \(\[1, 3, 7\]\)$"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": 1, "scope": {"turn": "user"}}),
     r"steps\.patched\.interventions\.patch\.pos: scope turn 'user' is a chat turn of the prompt; it takes frame 0$"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"idx": -1}),
     r"steps\.patched\.interventions\.patch\.pos: unknown field 'idx'; valid: frame, scope, index$"),
    (lambda d: d["sites"]["target"].update(layers="two"), r"sites\.target\.layers: 'two' is not a "
     r"selector: an int \(2, -1\), a slice in Python notation \('2:5', '-3:', ':'\) or a list of picks \(\[1, 3, 7\]\)$"),
    (lambda d: d["sites"]["target"].update(layers=[1, 1]), r"sites\.target\.layers: picks \[1, 1\] name 1 "
     r"twice$"),
    (lambda d: d["sites"]["target"].update(layers="5:2"), r"sites\.target\.layers: '5:2' picks nothing, however "
     r"many things there are$"),
    (lambda d: d["steps"]["patched"]["interventions"]["logits"].update(site={"component": "nope"}),
     r"steps\.patched\.interventions\.logits\.site: unknown component 'nope'; valid: token_embeddings, [a-z_, ]+, logits$"),
    (lambda d: d["data"]["pairs"].update(path=None, rows=[{"input": "x"}], split="train"),
     r"data\.pairs: 'split' keeps rows of a file; inline 'rows' are the rows$"),
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
    (lambda d: d["steps"]["counterfactual"]["interventions"]["v_cf"].update(ops=[{"op": "mean"}]),
     r"steps\.counterfactual\.interventions\.v_cf\.ops\[0\]: 'mean' aggregates over rows.*make it a step after the forward"),
    (lambda d: d["steps"]["iia"].update(expected="pairs.answers"),
     r"steps\.iia\.expected: dataset 'pairs' has no column 'answers'"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"]["ops"][0].update({"with": "counterfactual.v"}),
     r"steps\.patched\.interventions\.patch\.ops\[0\]\.with: 'counterfactual\.v' names no value produced before this step; "
     r"known: counterfactual\.v_cf"),
    (lambda d: d.update(saves=["iia", "nothing"]), r"saves\[1\]: 'nothing' names no step's value; a save is a step "
                                                   r"or a step's member: counterfactual\.v_cf, iia, logit_diff, "
                                                   r"patched\.logits, patched\.patch$"),
    (lambda d: d.update(saves=["iia", "iia"]), r"saves\[1\]: 'iia' is saved twice$"),
    (lambda d: d.update(saves=["pairs"]), r"saves\[0\]: 'pairs' names no step's value; a save is a step or a step's member"),
    (lambda d: d["sites"]["target"].update(layers="0:0"),
     r"sites\.target\.layers: '0:0' picks nothing, however many things there are"),
])
def test_a_compile_error_names_the_path(change, message):
    with pytest.raises(PlanError, match=message):
        edited("patching", change).compile()


# -- selectors -------------------------------------------------------------------------------------------


def test_a_selector_picks_as_python_slices_and_lists_do():
    n = list(range(6))
    for text in (":", "2:5", "-3:", ":4", "::2", "-4:-1", "1::-2", "5:1:-1", "9:", "-9:2"):
        assert selectors.select(text, 6) == n[slice(*[int(x) if x else None for x in (text.split(":") + [""])[:3]])]
    assert selectors.select(-1, 6) == [5] and selectors.select(6, 6) == [] and selectors.select(-7, 6) == []
    assert selectors.select([1, 3, -1], 6) == [1, 3, 5] and selectors.select([4, 0], 6) == [4, 0]
    assert selectors.select([1, 7], 6) == [1]                   # a pick that is not there; a static width refuses it
    with pytest.raises(ValueError, match=r"picks \[0, -2\] count one of these 2 twice"):
        selectors.select([0, -2], 2)


def test_a_selectors_static_width_needs_its_ends_on_one_side():
    widths = {0: 1, -1: 1, "[1, 3, 7]": 3, "2:5": 3, ":3": 3, "0:6:2": 3, "-4:-1": 3, "-3:": 3, "-6::2": 3,
              "5:1:-1": 4, ":-3:-1": 2, "3::-1": 4, ":": None, "2:": None, "-2:3": None, "1:-1": None, "::2": None,
              "::-1": None}
    for selector, width in widths.items():
        selector = [1, 3, 7] if selector == "[1, 3, 7]" else selector
        assert selectors.fixed(selector) == width, selector
        if width is not None:
            assert all(len(selectors.select(selector, n)) == width for n in (8, 9, 12)), selector


@pytest.mark.parametrize("pos, reach", [
    ({"frame": "1:4", "index": 1}, (False, [2], True)),
    ({"frame": [3, 1], "index": ":"}, (False, [1, 3], True)),
    ({"frame": [3, 1], "index": 0}, (False, [1, 3], False)),       # a row that stops after call 2 names call 1
    ({"frame": "1:", "index": 2}, (False, [3], True)),
    ({"frame": "1:", "index": [2, 0]}, (False, [1, 3], True)),
    ({"frame": "1:", "index": -1}, (False, None, False)),
    ({"frame": "2:", "index": ":"}, (False, None, False)),
    ({"frame": ":", "index": ":"}, (True, None, True)),
    ({"frame": "0:3", "index": -1}, (True, [1, 2], False)),
    ({"frame": "1:4", "index": "-2:"}, (False, [1, 2, 3], False)),
    ({"frame": 2, "index": -1}, (False, [2], True)),
    ({"frame": -1, "index": -1}, (True, None, False)),
])
def test_what_a_record_reaches_before_the_run(pos, reach):
    assert positions.reach({"scope": None, **pos}) == reach


@pytest.mark.parametrize("value", ["all", "1:2:3:4", "a:b", "1", "", [], True, 1.5, "::0", "-0:", "01:", ":-0"])
def test_anything_but_the_three_forms_is_refused_naming_them(value):
    with pytest.raises(PlanError, match=r"sites\.target\.layers: .* is not a selector: an int \(2, -1\), a slice "
                                        r"in Python notation \('2:5', '-3:', ':'\) or a list of picks \(\[1, 3, 7\]\)"):
        edited("patching", lambda d: d["sites"]["target"].update(layers=value))


@pytest.mark.parametrize("pos, message", [
    ({"index": "0:0"}, r"\.index: '0:0' picks nothing, however many things there are"),
    ({"index": ":0"}, r"\.index: ':0' picks nothing"),
    ({"index": "5:2"}, r"\.index: '5:2' picks nothing"),
    ({"index": "-1:-3"}, r"\.index: '-1:-3' picks nothing"),
    ({"frame": "3:1"}, r"\.frame: '3:1' picks nothing"),
    ({"frame": "1:3", "index": "5:"}, r": index '5:' picks none of the 2 tokens of frame '1:3''s decode calls"),
    ({"frame": "1:3", "index": -3}, r": index -3 picks none of the 2 tokens of frame '1:3''s decode calls"),
    ({"frame": [2], "index": [1, 2]}, r": index \[1, 2\] picks none of the 1 tokens"),
])
def test_a_selector_that_picks_nothing_is_refused_naming_it(pos, message):
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.patch\.pos" + message):
        edited("generate_probe", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=pos))


def test_a_slice_that_picks_fewer_of_more_things_is_not_refused():
    """`"-2:3"` picks something of three things and nothing of five: it has no static width, and is a selector."""
    assert selectors.fixed("-2:3") is None and selectors.select("-2:3", 3) == [1, 2]
    edited("generate_probe", lambda d: d["sites"]["target"].update(layers="-2:3"))
