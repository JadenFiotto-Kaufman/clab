"""The six documents that train or load a featurizer run from `clab run` on both engines, their numbers pinned
per engine at one batch size, the engines agreeing to the bit; and the numbers that pin what a fit, a bundle
and an op in a featurizer's space mean.

`das` and `pca_harvest` run first, into the corpus copy's `documents/artifacts`, because `das_apply` and
`pca_control` load what they write.
"""

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from clab import Plan, featurizers
from clab.cli import main
from clab.plan import PlanError, dims_of
from conftest import BATCH_SIZE, DOCUMENTS, FIT, ROOT, bundle_header, close, edited, plain, same, snapshot, stand_in

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"}


def corpus_copy(root) -> str:
    shutil.copytree(DOCUMENTS, root / "documents")
    os.symlink(os.path.join(ROOT, "data"), root / "data")
    return str(root)


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> dict:
    """Each fit document run from the CLI on each engine and loaded back, `das` and `pca_harvest` (on nnterp)
    into `documents/artifacts` first."""
    root = corpus_copy(tmp_path_factory.mktemp("fits"))
    out = {}

    def run(name: str, kind: str, to: str) -> Plan:
        assert main(["run", os.path.join(root, "documents", f"{name}.json"), "--engine", kind,
                     "--batch-size", str(BATCH_SIZE), "--out", to]) == 0
        cwd = os.getcwd()
        try:
            os.chdir(root)            # a loaded plan reads its bundles again from the corpus root
            return Plan.load(to)
        finally:
            os.chdir(cwd)

    out["root"] = root
    artifacts = os.path.join(root, "documents", "artifacts")
    out["das", "nnterp"] = run("das", "nnterp", artifacts)
    run("pca_harvest", "nnterp", artifacts)
    for name in FIT:
        for kind in ("nnterp", "hooks"):
            if (name, kind) not in out:
                out[name, kind] = run(name, kind, os.path.join(root, "out", f"{name}-{kind}"))
    return out


@pytest.mark.parametrize("name, kind", [(n, k) for n in FIT for k in ("nnterp", "hooks")])
def test_numbers_match_the_snapshot(ran, name, kind):
    values = plain(ran[name, kind].results)
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE
    close(values, expected["values"])


@pytest.mark.parametrize("name", FIT)
def test_the_engines_agree_to_the_bit(ran, name):
    assert same(ran[name, "nnterp"].results, ran[name, "hooks"].results)


def with_history(change=None) -> Plan:
    def edit(d):
        d["saves"] += ["fit.history.ce", "fit.history.iia", "fit.ce"]
        if change:
            change(d)
    return edited("das", edit).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_das_fit_lowers_ce_and_its_history_says_so(engines, kind):
    """`fit.history.ce` is the training rows' ce after each epoch, scored; it falls every epoch. The best epoch
    (by iia, which rises every epoch) is the last, restored: the root's ce, scored with `fit.rot` on the
    training rows, is the history's last entry. The model's weights get no gradient."""
    engine = engines(kind, with_history().model)
    values = engine.execute(with_history(), batch_size=BATCH_SIZE).results
    ce, iia = values["fit.history.ce"], values["fit.history.iia"]
    assert ce.shape == iia.shape == (10,)
    assert (ce[1:] < ce[:-1]).all() and (iia[1:] > iia[:-1]).all()
    assert torch.equal(ce[-1], values["ce"].mean())
    assert all(p.grad is None for p in engine.model.parameters())


def test_early_stopping_restores_the_best_epoch(engines):
    """With iia's mode turned to min, the best epoch is the first (iia rises every epoch), so patience 3 ends
    the fit after four epochs and restores the first epoch's rotation: the root's ce is the history's first."""
    def change(d):
        d["steps"]["fit"]["early_stop"]["mode"] = "min"
    plan = with_history(change)
    values = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert values["fit.history.ce"].shape == (4,)
    assert torch.equal(values["fit.history.ce"][0], values["ce"].mean())


def test_a_fit_is_the_same_remote_local_and_through_the_stand_in_server(engines, monkeypatch):
    engine = engines("nnterp", with_history().model)
    local = engine.execute(with_history(), batch_size=BATCH_SIZE).results
    assert same(engine.execute(with_history(), batch_size=BATCH_SIZE, remote="local").results, local)
    came, trip = stand_in(engine, with_history(), monkeypatch)
    assert trip["by_reference"] and trip["names"] == ["plan"] and trip["types"] == ["Plan"]
    assert same(came.results, local)


def test_das_apply_scores_the_saved_rotation_as_das_scored_it_held_out(ran):
    """das_apply loads the bundle das wrote and runs das's held-out body: the same numbers, to the bit."""
    das, applied = ran["das", "nnterp"], ran["das_apply", "nnterp"]
    assert same(applied.results["iia"], das.results["fit.iia"])
    assert same(load_file(os.path.join(ran["root"], "documents", "artifacts", "fit.rot.safetensors")),
                das.results["fit.rot"])


def test_random_subspace_draws_a_different_subspace_per_seed(ran):
    """Each seed draws its own Haar-random rotation; the three rank-4 bases are orthonormal and span three
    different subspaces, and score differently."""
    bases = [featurizers.basis(featurizers.make("subspace", {"k": 4, "seed": s}, 16, None, None, None))
             for s in (0, 1, 2)]
    projections = [b @ b.T for b in bases]
    for b in bases:
        torch.testing.assert_close(b.T @ b, torch.eye(4), atol=1e-6, rtol=0)
    assert all(not torch.allclose(projections[i], projections[j], atol=1e-3) for i, j in ((0, 1), (0, 2), (1, 2)))
    ld = ran["random_subspace", "nnterp"].results["logit_diff"]
    assert not torch.equal(ld[0], ld[1]) and not torch.equal(ld[1], ld[2]) and not torch.equal(ld[0], ld[2])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_ablating_a_latent_that_was_already_zero_changes_nothing(engines, kind):
    """sae_feature_ablation.json's claim: on a row where latent j was already zero, zeroing it leaves the
    logits bit-identical to an unpatched forward (the delta keeps the error term exactly); where it was not,
    they move. The fixture has both among latents 0 to 3."""
    def change(d):
        d["steps"]["clean"] = {"kind": "forward", "input": "prompts.input", "reads": {"logits": {"site": "head", "pos": -1}}}
        d["saves"] = ["base.latents", "zeroed.logits", "clean.logits"]
    plan = edited("sae_feature_ablation", change).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    latents, clean, seen = values["base.latents"][:, 0], values["clean.logits"], set()
    for j in range(4):
        zeroed = values["zeroed.logits"][(j,)]
        for r in range(4):
            seen.add(bool(latents[r, j] == 0))
            assert torch.equal(zeroed[r], clean[r]) == bool(latents[r, j] == 0), (j, r)
    assert seen == {True, False}


def test_dbms_hard_mask_keeps_whole_units_per_sweep_point(ran):
    """The gate is scored hard (θ > 0), so each L1 weight keeps a whole number of the 16 units."""
    masks = ran["dbm", "nnterp"].results["fit.mask"]
    kept = {str(w): int((m["theta"] > 0).sum()) for w, m in masks.items()}
    assert kept == snapshot("dbm.units.json", kept)


@pytest.mark.parametrize("name, key, value, message", [
    ("das_apply", "layer", "1", r"featurizers\.rot\.file_path: documents/artifacts/fit\.rot\.safetensors: header "
                                r"'layer' is '1'; this document uses 'rot' at layer_output layer 0"),
    ("das_apply", "model.key", "gpt2", r"header 'model\.key' is 'gpt2'; this document expects "
                                       r"'hf-internal-testing/tiny-random-LlamaForCausalLM'"),
    ("das_apply", "k", "4", r"header 'k' is '4'; this document expects '8'"),
    ("pca_control", "layer", "1", r"basis\.safetensors: header 'layer' is '1'"),
    ("pca_control", "component", "mlp_output", r"header 'component' is 'mlp_output'; this document uses 'rot' at "
                                               r"layer_output"),
])
def test_a_bundle_from_another_site_or_model_is_refused_naming_the_key(corpus, tmp_path, name, key, value, message):
    root = corpus_copy(tmp_path)
    shutil.copytree(os.path.join(corpus, "documents", "artifacts"), os.path.join(root, "documents", "artifacts"),
                    dirs_exist_ok=True)
    file = os.path.join(root, "documents", "artifacts", "fit.rot.safetensors" if name == "das_apply" else
                        "basis.safetensors")
    from safetensors import safe_open
    with safe_open(file, "pt") as f:
        header = {**(f.metadata() or {}), key: value}
    save_file(load_file(file), file, metadata=header)
    with pytest.raises(PlanError, match=message):
        Plan(os.path.join(root, "documents", f"{name}.json")).compile()


@pytest.mark.parametrize("name", ["das_apply", "pca_control"])
def test_a_bundle_of_another_width_is_refused_before_any_forward(engines, corpus, tmp_path, name):
    root = corpus_copy(tmp_path)
    shutil.copytree(os.path.join(corpus, "documents", "artifacts"), os.path.join(root, "documents", "artifacts"),
                    dirs_exist_ok=True)
    file = os.path.join(root, "documents", "artifacts", "fit.rot.safetensors" if name == "das_apply" else
                        "basis.safetensors")
    save_file(load_file(file), file, metadata=bundle_header(**{
        "kind": "subspace", "k": "8", "parametrization": "cayley", "width": "32"} if name == "das_apply" else
        {"kind": "pca", "k": "4", "width": "32"}))
    plan = Plan(os.path.join(root, "documents", f"{name}.json")).compile()
    with pytest.raises(ValueError, match=r"counterfactual\.v_cf: .*: header 'width' is '32'; this document uses "
                                         r"'rot' at layer_output, whose width is 16"):
        engines("hooks", plan.model).execute(plan)


# -- a probe ----------------------------------------------------------------------------------------------

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Saturday"]


def probe(change=None) -> Plan:
    """A linear probe of layer 1's output at the prompt's last token (its day's), trained to say whether the day
    is a weekday or the weekend; its classes are the label column's values, sorted."""
    doc = {
        "model": TINY,
        "data": {"days": {"rows": [{"input": f"If today is {d}", "label": "weekend" if d in ("Saturday", "Sunday")
                                    else "weekday"} for d in DAYS]}},
        "featurizers": {"probe": {"kind": "linear", "classes": "days.label"}},
        "steps": {"fit": {
            "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 30, "batch_size": 4, "seed": 0,
            "optimizer": {"name": "adam", "lr": 0.5},
            "steps": {
                "read": {"kind": "forward", "input": "days.input", "reads": {"scores": {
                    "site": {"component": "layer_output", "layers": 1}, "pos": -1,
                    "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
                "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "days.label"},
                "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "days.label"}}}},
        "saves": ["fit.history.ce", "fit.hit", "fit.read.scores", "fit.probe"]}
    if change:
        change(doc)
    return Plan(**doc).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_probe_learns_its_classes(engines, kind):
    """Its scores are one per class, `[rows, pos, classes]`; cross_entropy and match resolve the label column
    against the class list, so the training ce falls and the trained probe calls every row right."""
    plan = probe()
    assert dims_of(plan, "fit.read.scores") == ["rows", "pos", "classes"]
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert values["fit.read.scores"].shape == (8, 1, 2) and values["fit.probe"]["W"].shape == (16, 2)
    assert values["fit.history.ce"][-1] < values["fit.history.ce"][0]
    assert values["fit.hit"].eq(1).all()
    assert values["fit.hit"].tolist() == (values["fit.read.scores"].argmax(-1) == torch.tensor(
        [[int(d in ("Saturday", "Sunday"))] for d in DAYS])).float().tolist()       # weekday, weekend: sorted


def test_the_probe_runs_the_same_on_both_engines(engines):
    got = [engines(kind, probe().model).execute(probe(), batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")]
    assert same(*got)


# -- featurizers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["subspace", "pca", "gate", "sae"])
def test_writing_back_a_values_own_coordinates_changes_nothing_to_the_bit(kind):
    g = torch.Generator().manual_seed(1)
    h = torch.randn(3, 2, 16, generator=g)
    bundle = {"pca": {"basis": torch.linalg.qr(torch.randn(16, 4, generator=g))[0], "mean": torch.randn(16, generator=g)},
              "sae": {"W_enc": torch.randn(16, 32, generator=g), "b_enc": torch.randn(32, generator=g),
                      "W_dec": torch.randn(32, 16, generator=g), "b_dec": torch.randn(16, generator=g)}}.get(kind)
    fields = {"subspace": {"k": 4}, "pca": {"k": 4, "file_path": "x"}, "sae": {"file_path": "x"}}.get(kind, {})
    f = featurizers.make(kind, fields, 16, bundle, 0, None)
    assert torch.equal(featurizers.within(f, h, lambda z: z, None), h)
    assert torch.equal(featurizers.within(f, h, lambda z: z, torch.tensor([1, 3])), h)
    moved = featurizers.within(f, h, lambda z: z * 0, torch.tensor([1]))
    assert not torch.equal(moved, h) or kind == "sae"         # an sae latent at zero on every vector moves nothing


def test_a_gate_is_soft_while_trained_hard_when_scored_and_anneals_linearly():
    f = featurizers.make("gate", {"temperature": {"start": 1.0, "end": 0.05}}, 4, None, None, None)
    f["theta"] = torch.tensor([2.0, -1.0, 0.5, -0.1])
    f["training"] = True
    torch.testing.assert_close(featurizers.mask(f), torch.sigmoid(f["theta"]))
    f["training"] = False
    assert featurizers.mask(f).tolist() == [1.0, 0.0, 1.0, 0.0]
    temps = []
    for epoch in range(20):
        featurizers.anneal(f, epoch, 20)
        temps.append(f["T"])
    assert temps[0] == 1.0 and abs(temps[-1] - 0.05) < 1e-12 and abs(temps[10] - (1 - 0.95 * 10 / 19)) < 1e-12
    torch.testing.assert_close(featurizers.scalar(f, "mask"), torch.sigmoid(f["theta"] / 0.05).mean())


def test_a_trained_bundle_carries_its_site_and_document(ran):
    from safetensors import safe_open
    with safe_open(os.path.join(ran["root"], "documents", "artifacts", "fit.rot.safetensors"), "pt") as f:
        assert f.metadata() == bundle_header(kind="subspace", k="8", parametrization="cayley", document="das")


def test_a_fit_member_tables_over_the_eval_rows(ran):
    """`fit.iia` is the body's iia under eval: one row per held-out example, labelled with its own tokens."""
    cwd = os.getcwd()
    try:
        os.chdir(ran["root"])
        rows = ran["das", "nnterp"].table("fit.iia")
    finally:
        os.chdir(cwd)
    assert [r["row"] for r in rows] == [0, 1] and {r["pos_token"] for r in rows} == {"▁is"}
    assert [r["input"] for r in rows] == [r["input"] for r in json.load(open(os.path.join(ROOT, "data", "weekdays",
                                                                                          "data.json")))
                                          if r["split"] == "test"]


# -- refusals -------------------------------------------------------------------------------------------------


def train_an_sae(d):
    d["featurizers"]["dictionary"] = {"kind": "sae", "file_path": "documents/artifacts/sae.safetensors"}
    d["steps"]["fit"]["train"] = ["dictionary"]


def use_a_root_read_in_the_body(d):
    d["steps"] = {"early": {"kind": "forward", "input": "train.input", "reads": {"logits": {"site": "head", "pos": -1}}},
                  **d["steps"]}
    d["steps"]["fit"]["steps"]["iia"] = {"kind": "op", "op": "sub", "of": "patched.logits", "with": "early.logits"}


@pytest.mark.parametrize("change, message", [
    (train_an_sae, r"steps\.fit\.train\[0\]: 'dictionary' is a sae, which a bundle fixes; it has nothing to train"),
    (use_a_root_read_in_the_body,
     r"steps\.fit\.steps: the body uses 'early\.logits', which has a row per example of the whole dataset"),
    (lambda d: d["steps"]["fit"]["steps"].update(rot={"kind": "op", "op": "mean", "of": "patched.logits"}),
     r"steps\.fit\.steps\.rot: 'rot' is also a dataset or a featurizer"),
])
def test_a_fit_that_cannot_mean_anything_is_refused_at_compile(change, message):
    with pytest.raises(PlanError, match=message):
        edited("das", change).compile()


def test_a_write_through_a_probe_is_refused():
    def change(d):
        d["steps"]["fit"]["steps"]["read"]["writes"] = {"w": {"site": {"component": "layer_output", "layers": 0},
                                                             "pos": -1, "ops": [{"op": "scale", "factor": 0.0,
                                                                                 "featurizer": "fit.probe"}]}}
    with pytest.raises(PlanError, match=r"'fit\.probe' is a probe, whose space has no way back into the value"):
        probe(change)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["zeroed"]["writes"]["ablate"]["ops"][0].update(select={"features": [40]}),
     r"zeroed\.ablate: select features \[40\] outside the 32 features of 'dictionary'"),
])
def test_a_featurizer_used_outside_its_features_is_refused_before_any_forward(engines, change, message):
    plan = edited("sae_feature_ablation", change).compile()
    with pytest.raises(ValueError, match=message):
        engines("hooks", plan.model).execute(plan)


def test_a_readout_in_the_body_carries_the_gradient_on_both_engines(engines):
    """das with its logits read as the logit lens of layer 1's output (`readout`, the engine's own norm and
    head) rather than at the head: the fit still lowers ce, and the engines agree to the bit."""
    def change(d):
        d["interventions"]["das"]["reads"]["logits"] = {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                                                        "ops": [{"op": "readout"}]}
    plan = with_history(change)
    got = [engines(kind, plan.model).execute(with_history(change), batch_size=BATCH_SIZE).results
           for kind in ("nnterp", "hooks")]
    assert same(*got)
    assert got[0]["fit.history.ce"][-1] < got[0]["fit.history.ce"][0]
