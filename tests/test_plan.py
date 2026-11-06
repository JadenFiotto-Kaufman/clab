"""The plan: every document compiles to the plan its snapshot holds; the plan is pure data with views; errors
name the document path of what is wrong and the valid choices."""

import inspect
import json
import os
import pickle
import re
from typing import Any, cast

import pytest
from conftest import BATCH_SIZE, NAMES, REAL, ROOT, document, edited, placed, plan_of, same, snapshot
from huggingface_hub import try_to_load_from_cache

from clab import Plan, ops
from clab.compile import make_dataset_rows, positions
from clab.format import data, registry, selectors
from clab.format.sweeps import agreeing_point, lookup, points
from clab.plan import Data, PlanError


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


def test_the_corpus_is_what_documents_holds_and_the_readme_counts_it():
    """The corpus is every document under `documents/`, which the README counts, and the real-model documents are
    those under `documents/real/`."""
    assert {"direction_ablation", "sae_train", "sae_train_ablation", "lora_train", "lora_apply", "noise_write",
            "clamp_write", "cosine_metric", "forced_answer", "answer_spellings", "generated_contains",
            "generated_confidence"} <= set(NAMES)
    with open(os.path.join(ROOT, "README.md")) as f:
        assert f"({len(NAMES)} in all)" in f.read()
    assert REAL == ["real/attribution_patching", "real/behavior", "real/das", "real/head_patching", "real/layer_sweep",
                    "real/pattern_patching"]


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


def test_a_dict_is_refused_pointing_to_the_keyword_form():
    with open(document("patching")) as f:
        fields = json.load(f)
    with pytest.raises(TypeError, match=r"^Plan\(dict\): a document's fields are keywords, Plan\(\*\*fields\); "
                                        r"Plan\(path\) reads a document file, and Plan.load\(out\) a run's directory$"):
        Plan(fields)  # type: ignore[arg-type]


def test_a_root_featurizer_is_not_a_save():
    with pytest.raises(PlanError, match=r"saves\[0\]: 'rot' names no step's value"):
        edited("random_subspace", lambda d: d.update(saves=["rot"])).compile()



@pytest.mark.parametrize("field, value", [("batch_size", 1), ("at", {"seed": 0}),
                                          ("schedule", {"stop": ["layer_output", 0]})])
@pytest.mark.parametrize("name, path", [("patching", ("counterfactual",)), ("generated_contains", ("answer",)),
                                        ("lora_train", ("fit", "steps", "adapted"))])
def test_a_derived_field_a_step_sets_is_refused_by_name(name, path, field, value):
    """A field compile derives, on a forward or a generate (a fit body's too), is refused by name where a document
    sets it, which compile would write over: `batch_size` (which it writes for a fit's eval steps alone; set by a
    document, it ran the step at that size whatever the run's `--batch-size`), `at` and `schedule`."""
    def change(d):
        step = d["steps"][path[0]]
        for key in path[1:]:
            step = step[key]
        step[field] = value
    where = re.escape(".".join(("steps", *path, field)))
    with pytest.raises(PlanError, match=rf"^{where}: '{field}' is a derived field, which compile writes; a step sets "
                                        r"none$"):
        edited(name, change).compile()


@pytest.mark.parametrize("field, value, where", [("positions", {"pairs": [[0]]}, ""), ("gathered", True, ""),
                                                 ("width", 3, ""), ("dims", ["rows"], ".ops[0]")])
def test_a_derived_field_an_entry_or_its_op_sets_is_refused_by_name(field, value, where):
    """An entry's derived fields (`positions`, `gathered`, `width`) and its ops' (`dims`) are compile's, as a step's
    are, so a document's is refused by name at its path."""
    def change(d):
        patch = d["steps"]["patched"]["interventions"]["patch"]
        (patch["ops"][0] if where else patch)[field] = value
    with pytest.raises(PlanError, match=rf"^steps\.patched\.interventions\.patch{re.escape(where)}\.{field}: "
                                        rf"'{field}' is a derived field, which compile writes; an entry sets none$"):
        edited("patching", change).compile()


def test_a_fits_minibatch_holds_at_least_one_row():
    with pytest.raises(PlanError, match=r"^steps\.fit: batch_size 0: a minibatch holds at least one row$"):
        edited("das", lambda d: d["steps"]["fit"].update(batch_size=0))


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


def test_print_gives_the_one_line_repr_whatever_the_plan_holds(engines, capsys):
    """`str` is `repr`, so `print(plan)` is one line, not every field: a compiled plan that loads a bundle
    (sae_feature_ablation's dictionary) holds its tensors and prints none of them, and a run's prints its state."""
    compiled = plan_of("sae_feature_ablation")
    assert any(points(f)[0].bundle for f in compiled.featurizers.values())
    patching = plan_of("patching")
    filled = engines("nnterp", patching.model).execute(patching, batch_size=BATCH_SIZE)
    for plan, said in ((Plan(document("pos_sweep")), "Plan(pos_sweep, uncompiled)"),
                       (compiled, "Plan(sae_feature_ablation, compiled, 4 points)"),
                       (filled, "Plan(patching, run on nnterp, 2 results)")):
        print(plan)
        assert capsys.readouterr().out == f"{said}\n" and str(plan) == repr(plan) == said


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
        accuracy={"kind": "op", "op": "mean", "of": "fit.train.iia"})).compile()
    assert [p.at for p in points(plan.steps["accuracy"])] == [{"mask.mask": i} for i in range(4)]
    assert [p.objective["mask.mask"] for p in points(plan.steps["fit"])] == [0.0001, 0.002, 0.01, 0.05]


class Unscanned(list):
    """A step's points that refuse to be scanned: indexed, they are found; iterated, the test fails."""

    def __iter__(self):
        raise AssertionError("the points were scanned")


class Unscannable(dict):
    """A value's points by coordinate that refuse to be scanned, as `Unscanned` does."""

    def items(self):
        raise AssertionError("the points were scanned")


def test_a_point_is_found_by_its_index_on_each_axis_not_by_a_scan():
    """A swept step's points are the product of its axes in axis order, the last fastest, so the point agreeing
    with a consumer's coordinates (its own, or a later step's on more axes) is found by its index on each axis,
    without a scan; points in another order are scanned, to the same point."""
    plan = edited("pos_sweep", lambda d: d["sites"]["target"].update(layers={"sweep": [0, 1], "label": "src"}))
    plan = plan.compile()
    patched, counterfactual = points(plan.steps["patched"]), points(plan.steps["counterfactual"])
    assert [p.at for p in patched] == [{"src": s, "pos": i} for s in range(2) for i in range(3)]
    for p in patched:
        assert agreeing_point(Unscanned(patched), p.at) is p and agreeing_point(patched[::-1], p.at) is p
        assert agreeing_point(Unscanned(counterfactual), p.at) is counterfactual[p.at["src"]]


def test_a_value_is_found_by_the_consumers_coordinates_on_its_axes_not_by_a_scan():
    """A value's points share its axes, so the one agreeing with a consumer's coordinates, on more axes than its
    own, is keyed by them on the value's, without a scan."""
    state = {"v": Unscannable({(("src", s), ("pos", i)): (s, i) for s in range(2) for i in range(3)})}
    assert [lookup(state, "v", {"src": s, "other": 0, "pos": i}) for s in range(2) for i in range(3)] == \
        [(s, i) for s in range(2) for i in range(3)]


def test_a_functions_signature_is_taken_once_however_many_points_are_checked_against_it(monkeypatch):
    """Compile checks every op at every point of a sweep against its function's signature and spells its
    parameters by it (`registry.signature`, `ops._parameters`): each function's is taken once, so a compile from
    cold takes none twice and the next compile none at all."""
    registry.signature.cache_clear()
    ops._parameters.cache_clear()
    taken, signature = [], inspect.signature
    monkeypatch.setattr(inspect, "signature", lambda fn, *args, **kwargs: taken.append(fn) or signature(fn, *args,
                                                                                                        **kwargs))
    plan_of("head_patching")
    assert taken and len(taken) == len(set(taken))
    taken.clear()
    plan_of("head_patching")
    assert taken == []


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
    with pytest.raises(PlanError, match=r"steps\.iia\.expected: 'patched\.logits' has a row per example of 'pairs' and "
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
     r"sites\.target: unknown component 'block_output'; the components clab addresses \(`clab describe components` "
     r"says what each is\): token_embeddings"),
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
     r"steps\.patched\.interventions\.logits\.site: unknown component 'nope'; the components clab addresses "
     r"\(`clab describe components` says what each is\): token_embeddings, [a-z_, ]+, logits$"),
    (lambda d: d["data"]["pairs"].update(path=None, rows=[{"input": "x"}], split="train"),
     r"data\.pairs: 'split' is which split of a file or a hub dataset; inline 'rows' are the rows$"),
    (lambda d: d["sites"]["target"].update(component={"sweep": ["layer_output"]}),
     r"sites\.target\.component: a sweep cannot stand for 'component'$"),
    (lambda d: d["model"].update(revision={"sweep": ["main", "9fb191250dd56d0ba7ec9785a025ed29c03d5998"]}),
     r"model\.revision: a sweep cannot stand for 'revision'; a run loads one model: write one document per revision$"),
    (lambda d: d["data"]["pairs"].update(take={"sweep": [1, 2]}),
     r"data\.pairs\.take: a sweep cannot stand for 'take'; a dataset's rows are compiled once, and a step reads them "
     r"by name: write one dataset per value$"),
    (lambda d: d["data"]["pairs"].update(template={"sweep": [{"add_generation_prompt": True}, {}]}),
     r"data\.pairs\.template: a sweep cannot stand for 'template'; a dataset's rows are compiled once"),
    (lambda d: d["data"]["pairs"].update(path="weekdays/train#test"), None),
    (lambda d: d["sites"]["target"].pop("layers"),
     r"^sites\.target: layer_output is a block's value and needs 'layers'$"),
    (lambda d: d["steps"]["patched"]["interventions"]["logits"]["site"].update(layers=0),
     r"^steps\.patched\.interventions\.logits\.site: lm_head_output is the model's own value and takes no layers$"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"column": []}}),
     r"^steps\.patched\.interventions\.patch\.pos\.scope: a scope's list of columns or texts names at least one$"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"text": []}}),
     r"^steps\.patched\.interventions\.patch\.pos\.scope: a scope's list of columns or texts names at least one$"),
    (lambda d: d["steps"]["patched"].update(input="pairs"),
     r"^steps\.patched\.input: 'pairs' is not <dataset>\.<column> or <dataset>\.<column>\[i\]$"),
    (lambda d: d["steps"].update({"i.ia": d["steps"].pop("iia")}),
     r"^steps\.i\.ia: a name has no dots; a dot separates a thing from its member$"),
    (lambda d: d["steps"]["patched"].update(input=["pairs.input", {"text": "x", "bogus": 1}]),
     r"^steps\.patched\.input\[1\]: unknown field 'bogus'; valid: text$"),
    (lambda d: d["steps"]["patched"].update(input=5),
     r"^steps\.patched\.input: 5 should be a valid string, or a valid list$"),
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
    (lambda d: d["steps"]["iia"].update(expected="pairs"),
     r"steps\.iia\.expected: 'pairs' is not a column reference, '<dataset>\.<column>'; dataset 'pairs' has columns "
     r"answer, base_answer, "),
    (lambda d: d["steps"]["iia"].update(expected="pairs."),
     r"steps\.iia\.expected: 'pairs\.' is not a column reference, '<dataset>\.<column>'; dataset 'pairs' has columns"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"column": "pairs"}}),
     r"steps\.patched\.interventions\.patch\.pos\.scope\.column: 'pairs' is not a column reference"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"]["ops"][0].update({"with": "counterfactual.v"}),
     r"steps\.patched\.interventions\.patch\.ops\[0\]\.with: 'counterfactual\.v' names no value produced before this step; "
     r"known: counterfactual\.v_cf"),
    (lambda d: d.update(saves=["iia", "nothing"]), r"saves\[1\]: 'nothing' names no step's value; a save is a step "
                                                   r"or a step's member: counterfactual\.v_cf, iia, logit_diff, "
                                                   r"patched\.logits, patched\.patch$"),
    (lambda d: d.update(saves=["iia", "iia"]), r"saves\[1\]: 'iia' is saved twice$"),
    (lambda d: d.update(steps={**d["steps"], "tokens": d["steps"].pop("logit_diff")}, saves=["iia", "tokens"]),
     r"saves\[1\]: 'tokens' is the name of a run directory's own tokens\.json, which a save beside it would be "
     r"written over or write over; give the step another name$"),
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
    ({"frame": "1:3", "index": "1:3"}, r": index '1:3' is 2 tokens wide and picks 1 of the 2 tokens of frame '1:3''s "
                                       r"decode calls, one each, so it fits no row$"),
    ({"frame": [1, 2, 3], "index": [0, 5]}, r": index \[0, 5\] is 2 tokens wide and picks 1 of the 3 tokens"),
])
def test_a_selector_that_picks_nothing_is_refused_naming_it(pos, message):
    """A selector that can pick nothing, and an index whose static width the frame's decode calls can never fill
    (every row would be `alignment_missing`), are refused when the document is read."""
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.patch\.pos" + message):
        edited("generate_probe", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=pos))
    for fits in ({"frame": "1:4", "index": "1:3"}, {"frame": "1:3", "index": "-2:"}, {"frame": "1:3", "index": ":"}):
        edited("generate_probe", lambda d: d["steps"]["patched"]["interventions"]["logits"].update(pos=fits))


def test_a_pos_that_is_not_an_int_is_not_read_as_one():
    """A string, a float or a boolean is refused, in a sweep's values too, as `index`, `frame` and `layers` refuse
    them; nothing is coerced to a position."""
    where = r"^steps\.patched\.interventions\.patch\.pos: "
    forms = r"an int \(2, -1\), a slice in Python notation \('2:5', '-3:', ':'\) or a list of picks \(\[1, 3, 7\]\)"
    for pos in ("-1", {"sweep": ["-1", -2]}):
        with pytest.raises(PlanError, match=where + r"'-1' is a string, which is not read as a number; a pos is an int "
                                            r"\(\{'frame': 0, 'index': <int>\}\) or an object, each of whose selectors "
                                            r"is " + forms + "$"):
            edited("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=pos))
    for pos in (1.0, True, {"sweep": [1.0, -2]}):
        with pytest.raises(PlanError, match=where + r"(1\.0|True) is not a selector: " + forms
                                            + r"; a pos is an int \(\{'frame': 0, 'index': <int>\}\) or an object$"):
            edited("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=pos))
    with pytest.raises(PlanError, match=r"^steps\.patched\.interventions\.patch\.pos\.index: '-1' is not a selector"):
        edited("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"index": "-1"}))


def test_a_sweep_over_pos_objects_names_the_fields_that_take_one():
    """A sweep's example sweeps a scalar the pos takes one on (its frame, its index, a scope's occurrence), not
    the scope object again."""
    def swept(d):
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = {"sweep": [
            {"scope": {"column": "pairs.entity"}}, {"scope": {"column": "pairs.entity"}, "index": -1}]}
    with pytest.raises(PlanError, match=r"^steps\.patched\.interventions\.patch\.pos: a sweep's values are numbers, "
                                        r"strings, booleans or lists of them, each point keyed by its value; \{'scope': "
                                        r"\{'column': 'pairs\.entity'\}\} is an object: sweep a scalar inside the pos, "
                                        r"which takes one on frame, index and a scope's occurrence \(\{'frame': 0, "
                                        r"'index': \{'sweep': \[-1, -2\]\}\}, or \{'scope': \{'column': \.\.\., "
                                        r"'occurrence': \{'sweep': \[0, 1\]\}\}\}\)$"):
        edited("entity_patch", swept)
    with pytest.raises(PlanError, match=r"^steps\.patched\.generation_config: a sweep's values are numbers, strings, "
                                        r"booleans or lists of them, each point keyed by its value; \{'a': \{'b': 1\}\} is "
                                        r"an object: sweep a scalar field inside it, \{'a': \{'b': \{'sweep': \[1, "
                                        r"\.\.\.\]\}\}\}$"):
        edited("generate_probe", lambda d: d["steps"]["patched"].update(generation_config={"sweep": [{"a": {"b": 1}}]}))


@pytest.mark.parametrize("scope, message", [
    ({"column": {"colum": "pairs.entity"}}, r"column: unknown field 'colum'; valid: column, strip, lower$"),
    ({"column": [{"colum": "pairs.entity"}]}, r"column\[0\]: unknown field 'colum'; valid: column, strip, lower$"),
    ({"text": {"text": "Friday", "strip": True}},
     r"text: \{'text': 'Friday', 'strip': True\} is an object; a text is a literal, and its strip and lower go on the "
     r"scope beside it: \{'text': 'Friday', 'strip': true\}$"),
    ({"text": ["Friday", {"text": "Monday"}]}, r"text: \{'text': 'Monday'\} is an object; a text is a literal"),
    ({"column": 3}, r"column: 3 is not a column reference, '<dataset>\.<column>' or \{'column': '<dataset>\.<column>', "
                    r"'strip'\?: bool, 'lower'\?: bool\}$"),
    ({"column": ["pairs.entity", 3]}, r"column: 3 is not a column reference"),
    ({"column": {"sweep": ["pairs.entity", "pairs.answer"]}},
     r"column: \{'sweep': \['pairs\.entity', 'pairs\.answer'\]\} is not a column reference, .*; a scope's column takes "
     r"no sweep \(its occurrence does\)$"),
])
def test_a_scope_object_error_names_the_field_and_its_keys(scope, message):
    """A column reference inside a scope is validated as the one thing its shape says it is, so an unknown key
    names the keys a reference takes; `strip` inside a text object says where a text's flags go."""
    with pytest.raises(PlanError, match=r"^steps\.patched\.interventions\.patch\.pos\.scope\." + message):
        edited("entity_patch", lambda d: d["steps"]["patched"]["interventions"]["patch"]["pos"].update(scope=scope))


def test_a_slice_that_picks_fewer_of_more_things_is_not_refused():
    """`"-2:3"` picks something of three things and nothing of five: it has no static width, and is a selector."""
    assert selectors.fixed("-2:3") is None and selectors.select("-2:3", 3) == [1, 2]
    edited("generate_probe", lambda d: d["sites"]["target"].update(layers="-2:3"))


# -- a dataset's options --------------------------------------------------------------------------------------


def with_data(d: dict, **more) -> Plan:
    """patching.json with dataset `pairs` as `d`."""
    def change(doc):
        doc["data"]["pairs"] = d
        doc.update(more)
    return edited("patching", change)


def file_rows(path: str, split: str | None = None) -> list[dict]:
    with open(os.path.join(os.path.dirname(os.path.dirname(document("patching"))), "data", f"{path}.json")) as f:
        found = json.load(f)
    return [r for r in found if split is None or r["split"] == split]


def test_take_keeps_the_first_rows_and_the_digest_is_of_those():
    every = file_rows("weekdays/data")
    taken = with_data({"path": "weekdays/data", "take": 3}).compile()
    assert taken.data["pairs"].rows == every[:3] and taken.data["pairs"].n_rows == 3
    assert taken.data["pairs"].sha256 == data.digest(every[:3])
    assert compiled_form(taken)["data"]["pairs"] == {"path": "weekdays/data", "take": 3,
                                                     "sha256": data.digest(every[:3]), "n_rows": 3}
    split = with_data({"path": "weekdays/data", "split": "test", "take": 1}).compile()
    assert split.data["pairs"].rows == file_rows("weekdays/data", "test")[:1]
    inline = with_data({"rows": every, "take": 2}).compile()
    assert inline.data["pairs"].rows == every[:2]


def test_shuffle_is_one_fixed_permutation_then_take_samples_from_it():
    """Seed 0, by a hash of each row's index alone: the same order on every machine, twice the same, not the
    file's; `take` after it is a random sample of N. The plan shows the flag, the digest the rows' order."""
    def permutation(n: int) -> list[int]:
        return [r["i"] for r in data.shuffled([{"i": i} for i in range(n)])]
    assert permutation(8) == [0, 3, 2, 4, 7, 6, 5, 1]
    every = file_rows("weekdays/data")
    shuffled = with_data({"path": "weekdays/data", "shuffle": True}).compile()
    again = with_data({"path": "weekdays/data", "shuffle": True}).compile()
    order = permutation(len(every))
    assert shuffled.data["pairs"].rows == again.data["pairs"].rows == [every[i] for i in order]
    assert order != sorted(order) and shuffled.data["pairs"].sha256 != data.digest(every)
    assert compiled_form(shuffled)["data"]["pairs"]["shuffle"] is True
    sample = with_data({"path": "weekdays/data", "shuffle": True, "take": 2}).compile()
    assert sample.data["pairs"].rows == [every[i] for i in order[:2]] and sample.data["pairs"].n_rows == 2


def test_skip_drops_the_first_rows_of_the_order_and_take_keeps_a_shard_of_it():
    """`skip` after `shuffle` and before `take`: the rows are the slice of the shuffled order a hand would cut,
    `[skip:skip + take]`, and its digest theirs; the shards of one order are disjoint and together the whole."""
    every = file_rows("weekdays_full/data")
    order = data.shuffled(every)
    shards = [with_data({"path": "weekdays_full/data", "shuffle": True, "skip": skip, "take": 14}).compile()
              for skip in (0, 14, 28)]
    for i, shard in enumerate(shards):
        assert shard.data["pairs"].rows == order[14 * i:14 * i + 14] and shard.data["pairs"].n_rows == 14
        assert shard.data["pairs"].sha256 == data.digest(order[14 * i:14 * i + 14])
    assert [r for shard in shards for r in cast(list, shard.data["pairs"].rows)] == order != every
    assert compiled_form(shards[1])["data"]["pairs"] == {"path": "weekdays_full/data", "shuffle": True, "skip": 14,
                                                         "take": 14, "sha256": data.digest(order[14:28]), "n_rows": 14}
    rest = with_data({"path": "weekdays_full/data", "skip": len(every) - 1}).compile()
    assert rest.data["pairs"].rows == every[-1:]


def test_columns_renames_and_keeps_every_other_column():
    rows = [{"prompt": "a", "target": " b", "label": 1}, {"prompt": "c", "target": " d", "label": 0}]
    read = {"kind": "forward", "input": "pairs.input",
            "interventions": {"x": {"site": {"component": "lm_head_output"}, "pos": -1}}}
    plan = with_data({"rows": rows, "columns": {"input": "prompt", "answer": "target"}}, steps={
        "read": read, "p": {"kind": "op", "op": "token_prob", "of": "read.x", "token": "pairs.answer"}},
        saves=["p"]).compile()
    assert plan.data["pairs"].rows == [{"label": 1, "input": "a", "answer": " b"},
                                       {"label": 0, "input": "c", "answer": " d"}]
    assert compiled_form(plan)["data"]["pairs"]["columns"] == {"input": "prompt", "answer": "target"}
    swapped = make_dataset_rows("", "pairs", Data(rows=rows, columns={"prompt": "target", "target": "prompt"}))
    assert swapped == [{"label": 1, "prompt": " b", "target": "a"}, {"label": 0, "prompt": " d", "target": "c"}]
    with pytest.raises(PlanError, match=r"steps\.p\.token: dataset 'pairs' has no column 'target' on every row"):
        with_data({"rows": rows, "columns": {"input": "prompt", "answer": "target"}}, steps={
            "read": read, "p": {"kind": "op", "op": "token_prob", "of": "read.x", "token": "pairs.target"}},
            saves=["p"]).compile()


@pytest.mark.parametrize("d, message", [
    ({"path": "weekdays/data", "split": "test", "take": 3},
     r"^data\.pairs\.take: take 3, and the dataset has 2 rows in split 'test'; take keeps at most as many as there "
     r"are$"),
    ({"path": "weekdays/data", "take": 0}, r"^data\.pairs: take 0: a dataset keeps at least one row$"),
    ({"path": "weekdays/data", "split": "test", "skip": 2},
     r"^data\.pairs\.skip: skip 2, and the dataset has 2 rows in split 'test'; skip leaves at least one$"),
    ({"path": "weekdays/data", "split": "test", "skip": 1, "take": 2},
     r"^data\.pairs\.take: take 2, and the dataset has 1 rows in split 'test' after skip 1; take keeps at most as "
     r"many as there are$"),
    ({"path": "weekdays/data", "skip": -1}, r"^data\.pairs: skip -1: a dataset drops 0 rows or more from the start$"),
    ({"path": "weekdays/data", "columns": {"input": "prompt"}},
     r"^data\.pairs\.columns\.input: 'prompt' is not a column of row 0; its columns: answer, base_answer, "),
    ({"path": "weekdays/data", "columns": {"answer": "cf_answer"}},
     r"^data\.pairs\.columns\.answer: 'answer' is a column the rows keep, which this rename would overwrite; "
     r"name the new column otherwise$"),
    ({"path": "weekdays/data", "columns": {"gold": {"from": "answer", "after": {"text": "####"}}}},
     r"^data\.pairs\.columns\.gold\.after: row 0: '####' is not in its 'answer', ' Friday'$"),
    ({"path": "weekdays/data", "columns": {"gold": {"from": "input", "after": {"text": "today is "},
                                                    "before": {"text": "?"}}}},
     r"^data\.pairs\.columns\.gold\.before: row 0: '\?' is not in its 'input' after the first 'today is ', "),
    ({"path": "weekdays/data", "columns": {"gold": {"from": "counterfactual_inputs", "strip": True}}},
     r"^data\.pairs\.columns\.gold\.from: row 0: 'counterfactual_inputs' is list, \['If today is "),
    ({"path": "weekdays/data", "columns": {"gold": {"from": "gold", "strip": True}}},
     r"^data\.pairs\.columns\.gold\.from: 'gold' is not a column of row 0; its columns: answer, "),
    ({"path": "weekdays/data", "columns": {"answer": {"from": "input", "strip": True}}},
     r"^data\.pairs\.columns\.answer: 'answer' is a column the rows keep, which this derived column would "
     r"overwrite; name the new column otherwise$"),
    ({"path": "weekdays/data", "columns": {"gold": {"from": "answer", "after": "#### "}}},
     r"^data\.pairs\.columns\.gold\.after: '#### ' is a string; text written as it is is \{'text': '#### '\}"),
    ({"path": "weekdays/data", "columns": {"gold": {"of": "answer"}}},
     r"^data\.pairs\.columns\.gold: unknown field 'of'; valid: from, after, before, strip$"),
    ({"path": "weekdays/data", "template": {"add_generation_prompt": True}},
     r"^data\.pairs\.template: no step reads a chat column of 'pairs', and a template renders a chat \(a column "
     r"of \{'role', 'content'\} messages\) only$"),
    ({"path": "weekdays/data", "template": {"tokenize": True}},
     r"^data\.pairs: template: a chat renders to text, which clab tokenizes; it takes no 'tokenize'$"),
])
def test_a_dataset_option_that_cannot_apply_is_refused_by_name(d, message):
    with pytest.raises(PlanError, match=message):
        with_data(d).compile()


# GSM8K-style rows, whose answer ends in `#### <the number>`, and the same rows with the derived columns written in
# by hand.
WORKED = [{"input": "If today is Thursday, tomorrow is", "answer": "Thursday and one day #### Friday\n"},
          {"input": "If today is Saturday, tomorrow is", "answer": "Saturday and one day #### Sunday"}]
CUT = {"gold": {"from": "answer", "after": {"text": "#### "}, "strip": True},
       "day": {"from": "answer", "before": {"text": " and"}},
       "middle": {"from": "answer", "after": {"text": "and "}, "before": {"text": " ####"}},
       "raw": {"from": "answer", "after": {"text": "#### "}}}
BY_HAND = [{**WORKED[0], "gold": "Friday", "day": "Thursday", "middle": "one day", "raw": "Friday\n"},
           {**WORKED[1], "gold": "Sunday", "day": "Saturday", "middle": "one day", "raw": "Sunday"}]
SCORED = {"read": {"kind": "forward", "input": "pairs.input", "interventions": {
    "x": {"site": {"component": "logits"}, "pos": -1},
    "at": {"site": {"component": "layer_output", "layers": 0}, "pos": {"scope": {"column": "pairs.day"}}}}},
    "p": {"kind": "op", "op": "token_prob", "of": "read.x", "token": {"column": "pairs.gold", "strip": True}}}


def test_a_derived_column_is_the_text_cut_from_its_source_and_joins_the_digest():
    """A derived column is its source's text after `after`'s literal, then before `before`'s, then stripped under
    `strip`, its source kept: the rows equal the rows with those columns written by hand, and so does their digest;
    the compiled plan spells it as the document does."""
    derived = with_data({"rows": WORKED, "columns": CUT}, steps=SCORED, saves=["p"]).compile()
    by_hand = with_data({"rows": BY_HAND}, steps=SCORED, saves=["p"]).compile()
    assert derived.data["pairs"].rows == BY_HAND and \
        make_dataset_rows("", "pairs", Data.model_validate({"rows": WORKED, "columns": CUT})) == BY_HAND
    assert derived.data["pairs"].sha256 == by_hand.data["pairs"].sha256 == data.digest(BY_HAND)
    assert compiled_form(derived)["data"]["pairs"]["columns"] == CUT


def test_a_derived_column_scores_and_scopes_as_the_hand_built_one_does(engines):
    """A metric's answer and a scope's column read from a derived column give the bits the hand-built column gives,
    on both engines and under `remote="local"`, which agree."""
    runs = {}
    for name, d in (("derived", {"rows": WORKED, "columns": CUT}), ("by_hand", {"rows": BY_HAND})):
        for kind, options in (("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})):
            plan = with_data(d, steps=SCORED, saves=["p", "read.at"]).compile()
            runs[name, kind] = engines(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE)
    want = runs["by_hand", "nnterp"]
    assert placed(want)["read.at"] == [[4, 5, 6], [4]] and not want.results["p"].isnan().any()   # `▁Th urs day`, `▁Saturday`
    for key, filled in runs.items():
        assert same(filled.results, want.results) and placed(filled) == placed(want), key


# A tiny Llama-3.2, whose chat template calls `strftime_now` to write today's date into the system header it adds.
LLAMA32 = {"key": "yujiepan/llama-3.2-tiny-random", "revision": "0bf108f80b7c6cdf6e7e85698f0464bfbf7f9ce5"}
LLAMA32_CACHED = isinstance(try_to_load_from_cache(LLAMA32["key"], "model.safetensors", revision=LLAMA32["revision"]), str)


@pytest.mark.skipif(not LLAMA32_CACHED, reason=f"needs {LLAMA32['key']} in the local Hugging Face cache")
def test_a_chat_template_that_reads_the_clock_renders_only_with_its_date_pinned(engines):
    """Llama-3.2's template writes `Today Date: <today>` into the system header it adds, so the same chat tokenizes
    differently on another day while the rows' digest stays the same: a chat of it is refused by name before any
    forward unless the dataset's `template` pins `date_string`, and with it the date in the prompt is the one given."""
    def chat(template: dict) -> Plan:
        return Plan(model=LLAMA32, data={"d": {"rows": [{"chat": [{"role": "user", "content": "Hello there."}]}],
                                               "template": template}},
                    steps={"a": {"kind": "forward", "input": "d.chat",
                                 "interventions": {"l": {"site": {"component": "logits"}, "pos": -1}}}},
                    saves=["a.l"]).compile()
    engine = engines("nnterp", chat({}).model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=r"^a: d\.chat: the checkpoint's chat template calls strftime_now, which writes "
                                             r"today's date into the prompt, so these rows render to other tokens on "
                                             r"another day and nothing in the run record says so; pin it in "
                                             r"data\.d\.template, \{\"date_string\": \"26 Jul 2024\"\}$"):
            engine.execute(chat({"add_generation_prompt": True}), batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    assert not calls
    pinned = engine.execute(chat({"add_generation_prompt": True, "date_string": "26 Jul 2024"}), batch_size=BATCH_SIZE)
    tokens = pinned.run.tokens["d.chat"][0]
    assert "".join(tokens[tokens.index("Today"):]).startswith("TodayĠDate:Ġ26ĠJulĠ2024")
