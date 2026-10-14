"""Every document compiles to the plan its snapshot holds; the plan is pure data; errors name where and what."""

import ast
import glob
import json
import os
import pickle
import pprint

import pytest

from clab.compiler import CompileError, compile, points
from conftest import DOCUMENTS, ROOT, document, plan_of, snapshot

NAMES = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DOCUMENTS, "*.json")))
# These load a bundle another document writes (or, for the SAE, a fixture M4 commits).
BUNDLES = {"das_apply": "rot.safetensors", "pca_control": "pca.safetensors", "sae_feature_ablation": "sae.safetensors"}


def without_rows(plan: dict) -> dict:
    return {**plan, "data": {n: {k: v for k, v in d.items() if k != "rows"} for n, d in plan["data"].items()}}


def test_the_corpus_is_31_documents():
    assert len(NAMES) == 31


@pytest.mark.parametrize("name", NAMES)
def test_document_compiles_to_its_snapshot(name):
    if name in BUNDLES:
        with pytest.raises(CompileError, match=f"file_path: documents/artifacts/{BUNDLES[name]} does not exist"):
            plan_of(name)
        return
    plan = without_rows(plan_of(name))
    assert plan == snapshot(f"plans/{name}.json", plan)


def test_a_plan_pickles_and_prints():
    plan = plan_of("pos_sweep")
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert ast.literal_eval(pprint.pformat(plan)) == plan     # what it prints is what it is


def test_sweeps_are_gone_from_what_runs():
    plan = plan_of("pos_sweep")
    assert '"sweep"' not in json.dumps(plan["steps"])
    assert plan["sweeps"] == {"pos": {"path": "steps.patched.writes.patch.pos", "values": [-1, -2, -3]}}
    assert isinstance(plan["steps"]["counterfactual"], dict)          # no sweep touches it: it runs once
    patched = plan["steps"]["patched"]
    assert [p["at"] for p in patched] == [{"pos": 0}, {"pos": 1}, {"pos": 2}]
    assert [p["writes"][0]["pos"]["start"] for p in patched] == [-1, -2, -3]


def test_one_sweep_reached_from_two_steps_is_one_axis():
    plan = plan_of("head_patching")
    reads = [p["reads"][0]["select"] for p in plan["steps"]["counterfactual"]]
    writes = [p["writes"][0]["select"] for p in plan["steps"]["patched"]]
    assert reads == writes == [{"heads": [h]} for h in range(4)]


def test_a_reference_carries_its_producers_axes():
    plan = plan_of("dbm")       # the root counterfactual's only link to the swept fit is featurizer fit.mask
    assert [p["at"] for p in plan["steps"]["counterfactual"]] == [{"mask.mask": i} for i in range(4)]
    assert [p["objective"]["mask.mask"] for p in plan["steps"]["fit"]] == [0.0001, 0.002, 0.01, 0.05]


def test_data_rows_keep_only_the_columns_used():
    plan = plan_of("patching")
    assert plan["data"]["pairs"]["columns"] == ["base_answer", "cf_answer", "counterfactual_inputs", "input"]
    assert set(plan["data"]["pairs"]["rows"][0]) == set(plan["data"]["pairs"]["columns"])


def broken(change) -> dict:
    with open(document("patching")) as f:
        doc = json.load(f)
    change(doc)
    return doc


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["iia"].update(op="matches"),
     r"steps\.iia\.op: unknown op 'matches'; valid: swap, add"),
    (lambda d: d["sites"]["target"].update(component="block_output"),
     r"sites\.target\.component: unknown component 'block_output'; valid: token_embeddings"),
    (lambda d: d["steps"]["patched"].update(field="input"),
     r"steps\.patched\.field: unknown field; valid: input, writes, reads, interventions"),
    (lambda d: d["steps"]["counterfactual"]["reads"]["v_cf"].update(ops=[{"op": "mean"}]),
     r"steps\.counterfactual\.reads\.v_cf\.ops\[0\]\.op: 'mean' aggregates over rows.*make it a step after the forward"),
    (lambda d: d["steps"]["iia"].update(expected="pairs.answers"),
     r"steps\.iia\.expected: dataset 'pairs' has no column 'answers'"),
    (lambda d: d["steps"]["iia"].update(token_form="spelled"),
     r"steps\.iia\.token_form: 'spelled' is not one of 'spaced', 'bare'"),
    (lambda d: d["steps"]["patched"]["writes"]["patch"]["ops"][0].update({"with": "counterfactual.v"}),
     r"steps\.patched\.writes\.patch\.ops\[0\]\.with: 'counterfactual\.v' names no value produced before this step; "
     r"known: counterfactual\.v_cf"),
    (lambda d: d["steps"]["patched"].update(kind="trace"),
     r"steps\.patched\.kind: unknown step kind 'trace'; valid: forward, generate, op, fit"),
])
def test_a_compile_error_names_the_path_and_the_choices(change, message):
    with pytest.raises(CompileError, match=message):
        compile(broken(change), document("patching"))


def test_document_path_is_relative_to_the_corpus():
    assert plan_of("patching")["document"]["path"] == "documents/patching.json"
    assert os.path.isdir(os.path.join(ROOT, "data"))


def test_every_point_has_at():
    for name in NAMES:
        if name in BUNDLES:
            continue
        for entry in plan_of(name)["steps"].values():
            assert all("at" in p for p in points(entry))
