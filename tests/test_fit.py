"""The documents that train or load a featurizer run from `clab run` on both engines, their numbers pinned
per engine at one batch size, the engines agreeing to the bit; and the numbers that pin what a fit, a bundle
and an edit of a featurizer's coordinates mean.

`das`, `pca_harvest`, `sae_train` and `lora_train` run first, into the corpus copy's `documents/artifacts`,
because `das_apply`, `pca_control`, `sae_train_ablation` and `lora_apply` load what they write.
"""

import functools
import json
import os
import shutil
from typing import cast

import pytest
import torch
from conftest import (
    BATCH_SIZE,
    DOCUMENTS,
    FIT,
    ROOT,
    TRIP,
    bundle_header,
    close,
    dims,
    edited,
    placed,
    plain,
    same,
    snapshot,
    stand_in,
)
from safetensors.torch import load_file, save_file

from clab import Plan, featurizers, ops
from clab.cli import main
from clab.featurizers.gate import Gate
from clab.featurizers.subspace import Subspace
from clab.plan import Model, PlanError

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"}
TINY_MODEL = Model.model_validate(TINY)


def corpus_copy(root) -> str:
    shutil.copytree(DOCUMENTS, root / "documents")
    os.symlink(os.path.join(ROOT, "data"), root / "data")
    return str(root)


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> dict:
    """Each fit document run from the CLI on each engine (ndif under `remote="local"`) and loaded back, `das`,
    `sae_train`, `lora_train` and `pca_harvest` (on nnterp) into `documents/artifacts` first."""
    root = corpus_copy(tmp_path_factory.mktemp("fits"))
    out = {}

    def run(name: str, kind: str, to: str) -> Plan:
        options = ["-o", "remote=local"] if kind == "ndif" else []
        assert main(["run", os.path.join(root, "documents", f"{name}.json"), "--engine", kind, *options,
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
    out["sae_train", "nnterp"] = run("sae_train", "nnterp", artifacts)
    out["lora_train", "nnterp"] = run("lora_train", "nnterp", artifacts)
    run("pca_harvest", "nnterp", artifacts)
    for name in FIT:
        for kind in ("nnterp", "hooks", "ndif"):
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
    assert same(ran[name, "nnterp"].results, ran[name, "ndif"].results)       # ndif under remote="local"


def chain(*changes):
    """The changes to a document's JSON, applied in turn."""
    def change(d):
        for c in changes:
            c(d)
    return change


def inside(*names):
    """A change that moves the root steps `names` (harvests made before the fit) into the fit's body, first: the
    form in which the body makes them itself, on every minibatch, which the form that reads them is checked
    against."""
    def change(d):
        d["steps"]["fit"]["steps"] = {**{n: d["steps"].pop(n) for n in names}, **d["steps"]["fit"]["steps"]}
    return change


def held_ce(d):
    """das's held-out cross entropy beside its iia, after the fit."""
    d["steps"]["held_ce"] = {"kind": "op", "op": "cross_entropy", "of": "held_patched.logits",
                             "target": "test.cf_answer"}


def with_history(change=None) -> Plan:
    def edit(d):
        held_ce(d)
        d["saves"] += ["fit.history.ce", "fit.history.iia", "held_ce"]
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
    assert torch.equal(ce[-1], values["fit.train.ce"].mean())
    assert all(p.grad is None for p in engine.model.parameters())


def test_early_stopping_restores_the_best_epoch_before_the_eval_run(engines):
    """With iia's mode turned to min, the best epoch is the first (iia rises every epoch), so patience 3 ends
    the fit after four epochs and restores the first epoch's rotation: the train eval's ce is the history's
    first, and the held-out ce after the fit is that of a fit that ran one epoch, so the eval and the steps
    after the fit read the restored rotation."""
    def change(d):
        d["steps"]["fit"]["early_stop"]["mode"] = "min"

    def one_epoch(d):
        d["steps"]["fit"]["epochs"] = 1
    plan = with_history(change)
    engine = engines("hooks", plan.model)
    values = engine.execute(plan, batch_size=BATCH_SIZE).results
    assert values["fit.history.ce"].shape == (4,)
    assert torch.equal(values["fit.history.ce"][0], values["fit.train.ce"].mean())
    assert torch.equal(values["held_ce"], engine.execute(with_history(one_epoch), batch_size=BATCH_SIZE).results["held_ce"])


def test_a_fit_steps_once_per_minibatch_in_an_order_its_seed_and_epoch_shuffle(engines, monkeypatch):
    """das with `batch_size: 1` makes two minibatches of its two training rows an epoch: 20 optimizer steps
    over 10 epochs, its history pinned. With the rotation's own seed fixed, another fit seed changes only the
    order the rows are shuffled in, and that changes the trajectory."""
    from clab.steps.fit import OPTIMIZERS

    def change(d, seed=0):
        d["steps"]["fit"].update(batch_size=1, seed=seed)
        d["featurizers"]["rot"]["seed"] = 0
    counted = []

    class Counting(torch.optim.AdamW):
        def step(self, *args, **kwargs):            # type: ignore[override]
            counted.append(1)
            return super().step(*args, **kwargs)
    monkeypatch.setitem(OPTIMIZERS, "adamw", Counting)
    plan = with_history(change)
    engine = engines("hooks", plan.model)
    values = engine.execute(plan, batch_size=BATCH_SIZE).results
    assert len(counted) == 10 * 2
    history = plain({k: values[k] for k in ("fit.history.ce", "fit.history.iia")})
    close(history, snapshot("das.batch_size_1.json", history))
    other = engine.execute(with_history(lambda d: change(d, seed=1)), batch_size=BATCH_SIZE).results
    assert not torch.equal(other["fit.history.ce"], values["fit.history.ce"])


def test_a_pca_encodes_its_mean_as_zero():
    g = torch.Generator().manual_seed(0)
    bundle = {"basis": torch.linalg.qr(torch.randn(16, 4, generator=g))[0], "mean": torch.randn(16, generator=g)}
    f = featurizers.make("pca", {"k": 4, "file_path": "x"}, 16, bundle, None, None)
    assert torch.equal(f.call("encode", bundle["mean"][None], {}), torch.zeros(1, 4))


def test_a_fit_is_the_same_remote_local_and_through_the_stand_in_server(engines, monkeypatch):
    local = engines("nnterp", with_history().model).execute(with_history(), batch_size=BATCH_SIZE).results
    there = engines("ndif", with_history().model, remote="local").execute(with_history(), batch_size=BATCH_SIZE)
    assert same(there.results, local)
    came, trip = stand_in(engines, with_history(), monkeypatch)
    assert trip == TRIP and same(came.results, local)


def test_das_apply_scores_the_saved_rotation_as_das_scored_it_held_out(ran):
    """das_apply loads the bundle das wrote and runs the steps das runs after its fit: the same numbers, to
    the bit."""
    das, applied = ran["das", "nnterp"], ran["das_apply", "nnterp"]
    assert same(applied.results["iia"], das.results["held_iia"])
    assert same(load_file(os.path.join(ran["root"], "documents", "artifacts", "fit.rot.safetensors")),
                das.results["fit.rot"])


def test_random_subspace_draws_a_different_subspace_per_seed(ran):
    """Each seed draws its own Haar-random rotation; the three rank-4 bases are orthonormal and span three
    different subspaces, and score differently."""
    bases = [Subspace(16, k=4, seed=s).basis for s in (0, 1, 2)]
    projections = [b @ b.T for b in bases]
    for b in bases:
        torch.testing.assert_close(b.T @ b, torch.eye(4), atol=1e-6, rtol=0)
    assert all(not torch.allclose(projections[i], projections[j], atol=1e-3) for i, j in ((0, 1), (0, 2), (1, 2)))
    ld = ran["random_subspace", "nnterp"].results["logit_diff"]
    assert not torch.equal(ld[0], ld[1]) and not torch.equal(ld[1], ld[2]) and not torch.equal(ld[0], ld[2])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("name", ["sae_feature_ablation", "sae_train_ablation"])
def test_ablating_a_latent_that_was_already_zero_changes_nothing(engines, ran, name, kind):
    """The ablation documents' claim: on a row where latent j was already zero, zeroing it leaves the logits
    bit-identical to an unpatched forward (the delta keeps the error term exactly); where it was not, they
    move. The fixture and the dictionary sae_train trains both have each among latents 0 to 3."""
    def change(d):
        d["steps"]["clean"] = {"kind": "forward", "input": "prompts.input",
                               "interventions": {"logits": {"site": "head", "pos": -1}}}
        d["saves"] = ["base.latents", "zeroed.logits", "clean.logits"]
    with open(os.path.join(ran["root"], "documents", f"{name}.json")) as f:
        doc = json.load(f)
    change(doc)
    plan = Plan(**doc, document=os.path.join(ran["root"], "documents", f"{name}.json")).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    latents, clean, seen = values["base.latents"][:, 0], values["clean.logits"], set()
    for j in range(4):
        zeroed = values["zeroed.logits"][j]
        for r in range(4):
            seen.add(bool(latents[r, j] == 0))
            assert torch.equal(zeroed[r], clean[r]) == bool(latents[r, j] == 0), (j, r)
    assert seen == {True, False}


def sae_train(change=None) -> Plan:
    def edit(d):
        d["saves"].append("fit.train.latents")
        if change:
            change(d)
    return edited("sae_train", edit).compile()


def test_training_a_dictionary_lowers_its_reconstruction_the_same_everywhere(engines, monkeypatch):
    """sae_train's reconstruction falls every epoch; nnterp, hooks, remote="local" and the stand-in server give
    the same numbers to the bit, and the model's weights get no gradient. The decoder's rows stay unit-norm."""
    engine = engines("nnterp", sae_train().model)
    values = engine.execute(sae_train(), batch_size=BATCH_SIZE).results
    reconstruction = values["fit.history.dictionary.reconstruction"]
    assert reconstruction.shape == (20,) and (reconstruction[1:] < reconstruction[:-1]).all()
    assert same(engines("hooks", sae_train().model).execute(sae_train(), batch_size=BATCH_SIZE).results, values)
    there = engines("ndif", sae_train().model, remote="local").execute(sae_train(), batch_size=BATCH_SIZE)
    assert same(there.results, values)
    came, trip = stand_in(engines, sae_train(), monkeypatch)
    assert trip == TRIP and same(came.results, values)
    torch.testing.assert_close(values["fit.dictionary"]["W_dec"].norm(dim=1), torch.ones(32), atol=1e-6, rtol=0)
    assert all(p.grad is None for p in engine.model.parameters())


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_dictionary_trains_from_a_harvest_as_from_the_bodys_own_encoding_read(engines, kind):
    """sae_train with the harvest in its body, encoded inside its entry (whose reads the scalars are over, so the
    run takes it off the model though nothing names it), trains the dictionary that sae_train's encode step of
    the harvest trains, to the bit, with the harvest one batch of the 4 prompts and the minibatch all 4."""
    def every_row(d):
        d["steps"]["fit"]["batch_size"] = 4

    def in_entry(d):
        harvest = d["steps"].pop("harvest")
        harvest["interventions"]["acts"]["ops"] = [{"op": "encode", "featurizer": "fit.dictionary"}]
        d["steps"]["fit"]["steps"] = {"harvest": harvest}
    outside, within = (edited("sae_train", chain(every_row, *more)).compile() for more in ((), (in_entry,)))
    engine = engines(kind, outside.model)
    assert same(engine.execute(outside, batch_size=4).results, engine.execute(within, batch_size=4).results)


def test_a_trained_dictionary_loads_and_encodes_as_the_trained_one_did(engines, tmp_path):
    """sae_train_ablation's dictionary, loaded from the bundle sae_train wrote, gives the latents of every
    token after BOS bit-equal to the trained instance's own encode of the harvest in sae_train's `train` eval."""
    root = corpus_copy(tmp_path)
    engine = engines("hooks", sae_train().model)
    filled = engine.execute(sae_train(), batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "documents", "artifacts"))
    with open(os.path.join(root, "documents", "sae_train_ablation.json")) as f:
        doc = json.load(f)
    doc["steps"]["base"]["interventions"]["latents"]["pos"] = {"index": "1:"}
    doc["saves"] = ["base.latents"]
    plan = Plan(**doc, document=os.path.join(root, "documents", "sae_train_ablation.json")).compile()
    assert same(engine.execute(plan, batch_size=BATCH_SIZE).results["base.latents"],
                filled.results["fit.train.latents"])


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


def probe(*changes) -> Plan:
    """A linear probe of layer 1's output at the prompt's last token (its day's), trained to say whether the day
    is a weekday or the weekend; its classes are the label column's values, sorted. The read is harvested before
    the fit (`read.h`), and the body is the probe's `encode` of it (`scores`) and the metrics over that."""
    doc = {
        "model": TINY,
        "data": {"days": {"rows": [{"input": f"If today is {d}", "label": "weekend" if d in ("Saturday", "Sunday")
                                    else "weekday"} for d in DAYS]}},
        "featurizers": {"probe": {"kind": "linear", "classes": "days.label"}},
        "steps": {
            "read": {"kind": "forward", "input": "days.input", "interventions": {"h": {
                "site": {"component": "layer_output", "layers": 1}, "pos": -1}}},
            "fit": {
                "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 30, "batch_size": 4, "seed": 0,
                "optimizer": {"name": "adam", "lr": 0.5}, "eval": {"train": {}},
                "steps": {
                    "scores": {"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "read.h"},
                    "ce": {"kind": "op", "op": "cross_entropy", "of": "scores", "target": "days.label"},
                    "hit": {"kind": "op", "op": "match", "of": "scores", "expected": "days.label"}}}},
        "saves": ["fit.history.ce", "fit.train.hit", "fit.train.scores", "fit.probe"]}
    chain(*changes)(doc)
    return Plan(**doc).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_probe_learns_its_classes(engines, kind):
    """Its scores are one per class, `[rows, pos, classes]`; cross_entropy and match resolve the label column
    against the class list, so the training ce falls and the trained probe calls every row right."""
    plan = probe()
    assert dims(plan, "fit.train.scores") == ["rows", "pos", "classes"]
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert values["fit.train.scores"].shape == (8, 1, 2) and values["fit.probe"]["W"].shape == (16, 2)
    assert values["fit.history.ce"][-1] < values["fit.history.ce"][0]
    assert values["fit.train.hit"].eq(1).all()
    assert values["fit.train.hit"].tolist() == (values["fit.train.scores"].argmax(-1) == torch.tensor(
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
    assert torch.equal(f.within(h, lambda z: z, None), h)
    assert torch.equal(f.within(h, lambda z: z, torch.tensor([1, 3])), h)
    moved = f.within(h, lambda z: z * 0, torch.tensor([1]))
    assert not torch.equal(moved, h) or kind == "sae"         # an sae latent at zero on every vector moves nothing


def test_a_gate_is_soft_while_trained_hard_when_scored_and_anneals_linearly():
    f = Gate(4, temperature={"start": 1.0, "end": 0.05})
    f.theta = torch.tensor([2.0, -1.0, 0.5, -0.1])
    f.training = True
    torch.testing.assert_close(f.mask(), torch.sigmoid(f.theta))
    f.training = False
    assert f.mask().tolist() == [1.0, 0.0, 1.0, 0.0]
    temps = []
    for epoch in range(20):
        f.anneal(epoch, 20)
        temps.append(f.T)
    assert temps[0] == 1.0 and abs(temps[-1] - 0.05) < 1e-12 and abs(temps[10] - (1 - 0.95 * 10 / 19)) < 1e-12
    torch.testing.assert_close(f.scalar("mask"), torch.sigmoid(f.theta / 0.05).mean())


def test_a_trained_bundle_carries_its_site_and_document(ran):
    from safetensors import safe_open
    with safe_open(os.path.join(ran["root"], "documents", "artifacts", "fit.rot.safetensors"), "pt") as f:
        assert f.metadata() == bundle_header(kind="subspace", k="8", parametrization="cayley", document="das")


def test_both_named_evals_run_and_each_member_tables_over_its_own_rows(ran):
    """lora_train names two evals: `held_out` (the body with `test` for `train`) and `train` (the data it
    trained on). Both run, and each member tables over its own eval's rows, labelled with their own tokens and
    text."""
    lora = ran["lora_train", "nnterp"]
    assert lora.results["fit.held_out.ce"].shape == lora.results["fit.train.ce"].shape == (2, 1)
    assert not torch.equal(lora.results["fit.held_out.ce"], lora.results["fit.train.ce"])
    cwd = os.getcwd()
    try:
        os.chdir(ran["root"])
        tables = {label: lora.table(f"fit.{label}.ce") for label in ("held_out", "train")}
    finally:
        os.chdir(cwd)
    with open(os.path.join(ROOT, "data", "weekdays", "data.json")) as f:
        rows = json.load(f)
    for label, split in (("held_out", "test"), ("train", "train")):
        table = tables[label]
        assert [r["row"] for r in table] == [0, 1] and {r["pos_token"] for r in table} == {"▁is"}
        assert [r["input"] for r in table] == [r["input"] for r in rows if r["split"] == split]
        assert [r["value"] for r in table] == lora.results[f"fit.{label}.ce"].flatten().tolist()


def test_two_histories_join_epoch_by_epoch_and_one_beside_a_member_with_rows_broadcasts(engines):
    """`fit.history.iia` and `fit.history.ce` are over `epochs`, a value with no rows axis each: joined, they give one
    row per epoch, each term's entry of that epoch beside the other's, not every pair of epochs; beside a value with
    a row per example (`fit.train.iia`) a history shares no key, so each epoch's entry sits beside every row, as an
    operand broadcasts over an axis it leaves out; and a trained featurizer (`fit.rot`) is a bundle with no rows."""
    def change(d):
        d["steps"]["fit"]["epochs"] = 3
        d["saves"] = ["fit.history.iia", "fit.history.ce", "fit.train.iia", "fit.rot"]
    filled = engines("nnterp", TINY_MODEL).execute(edited("das", change).compile(), batch_size=BATCH_SIZE)
    joined = filled.table("fit.history.iia", "fit.history.ce")
    assert [t["epochs"] for t in joined] == [0, 1, 2]
    for term in ("iia", "ce"):
        assert [t[f"fit.history.{term}"] for t in joined] == filled.results[f"fit.history.{term}"].tolist()
    beside = filled.table("fit.history.ce", "fit.train.iia")
    assert [(t["epochs"], t["row"]) for t in beside] == [(epoch, row) for row in range(2) for epoch in range(3)]
    assert all(t["fit.history.ce"] == filled.results["fit.history.ce"][t["epochs"]].item() and
               t["fit.train.iia"] == filled.results["fit.train.iia"][t["row"], 0].item() for t in beside)
    with pytest.raises(ValueError, match=r"^'fit\.rot' is a trained featurizer's bundle, which has no rows to table; "
                                         r"read it from plan\.results$"):
        filled.table("fit.rot")


def test_an_op_over_a_fit_members_value_tables_over_the_evals_rows(engines):
    """A step after the fit over `fit.held_out.ce`, its mean over `pos`: tabled, its rows are the held-out split's,
    with that split's text, and its values the member's; the chain from the op to the body's read through the
    eval resolves in the records (it once crashed in `table`'s provenance walk)."""
    def change(d):
        d["steps"]["fit"]["epochs"] = 1
        d["steps"]["m"] = {"kind": "op", "op": "mean", "of": "fit.held_out.ce", "over": ["pos"]}
        d["saves"] = ["fit.held_out.ce", "m"]
    filled = engines("nnterp", TINY_MODEL).execute(edited("lora_train", change).compile(), batch_size=BATCH_SIZE)
    with open(os.path.join(ROOT, "data", "weekdays", "data.json")) as f:
        held_out = [r["input"] for r in json.load(f) if r["split"] == "test"]
    table = filled.table("m")
    assert [(t["row"], t["input"]) for t in table] == list(enumerate(held_out))
    assert [t["value"] for t in table] == filled.results["m"].tolist() == filled.results["fit.held_out.ce"].flatten().tolist()


# -- refusals -------------------------------------------------------------------------------------------------


def train_an_sae(d):
    d["featurizers"]["dictionary"] = {"kind": "sae", "file_path": "documents/artifacts/sae.safetensors"}
    d["steps"]["fit"]["train"] = ["dictionary"]


@pytest.mark.parametrize("change, message", [
    (train_an_sae, r"steps\.fit\.train\[0\]: 'dictionary' is loaded from a bundle, which fixes it"),
    (lambda d: d["steps"]["fit"]["steps"].update(rot={"kind": "op", "op": "mean", "of": "patched.logits"}),
     r"steps\.fit\.steps\.rot: 'rot' is also a dataset or a featurizer"),
    (lambda d: d["steps"]["fit"]["eval"].update(rot={}),
     r"steps\.fit\.eval\.rot: 'rot' names the fit's history or a featurizer it trains"),
])
def test_a_fit_that_cannot_mean_anything_is_refused_at_compile(change, message):
    with pytest.raises(PlanError, match=message):
        edited("das", change).compile()


def test_a_write_through_a_probe_is_refused():
    def change(d):
        d["steps"]["fit"]["steps"]["read"]["interventions"] = {"w": {
            "site": {"component": "layer_output", "layers": 0}, "pos": -1,
            "ops": [{"op": "encode", "featurizer": "fit.probe"}, {"op": "scale", "factor": 0.0},
                    {"op": "decode", "featurizer": "fit.probe"}]}}
    with pytest.raises(PlanError, match=r"'fit\.probe' is a 'linear' featurizer, which has no decode: it serves encode; it "
                                        r"has no way back into the value"):
        probe(inside("read"), change)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["zeroed"]["interventions"]["ablate"]["ops"][0].update(select={"features": [40]}),
     r"zeroed\.ablate: select features \[40\] outside the 32 features of 'dictionary'"),
    (lambda d: d["steps"]["zeroed"]["interventions"]["ablate"]["ops"][0].update(select={"features": "40:"}),
     r"zeroed\.ablate: select features '40:' keeps none of the 32 features of 'dictionary'"),
])
def test_a_featurizer_used_outside_its_features_is_refused_before_any_forward(engines, change, message):
    plan = edited("sae_feature_ablation", change).compile()
    with pytest.raises(ValueError, match=message):
        engines("hooks", plan.model).execute(plan)


def test_a_readout_in_the_body_carries_the_gradient_on_both_engines(engines):
    """das with its logits read as the logit lens of layer 1's output (`readout`, the engine's own norm and
    head) rather than at the head: the fit still lowers ce, and the engines agree to the bit."""
    def change(d):
        d["steps"]["fit"]["steps"]["patched"]["interventions"]["logits"] = {
            "site": {"component": "layer_output", "layers": 1}, "pos": -1, "ops": [{"op": "readout"}]}
    plan = with_history(change)
    got = [engines(kind, plan.model).execute(with_history(change), batch_size=BATCH_SIZE).results
           for kind in ("nnterp", "hooks")]
    assert same(*got)
    assert got[0]["fit.history.ce"][-1] < got[0]["fit.history.ce"][0]


# -- review findings: joins, swept bundles, probe labels, root values in a body, gate and probe bundles ---------


def test_a_table_refuses_to_join_values_of_two_datasets(ran):
    """`held_iia` has a row per held-out example and `fit.train.iia` one per training example: row 1 of one is not
    row 1 of the other, so the join is refused naming both datasets."""
    with pytest.raises(ValueError, match=r"'held_iia' has a row per example of 'test' and "
                                         r"'fit\.train\.iia' of 'train'"):
        ran["das", "nnterp"].table("held_iia", "fit.train.iia")


def test_a_swept_fits_bundle_is_refused_naming_its_points(engines, tmp_path):
    root = corpus_copy(tmp_path)
    swept = edited("das", lambda d: d["steps"]["fit"]["optimizer"].update(lr={"sweep": [0.001, 0.01]})).compile()
    engines("hooks", swept.model).execute(swept, batch_size=BATCH_SIZE).dump(
        os.path.join(root, "documents", "artifacts"))
    with pytest.raises(PlanError, match=r"fit\.rot\.safetensors holds one bundle per point of a sweep "
                                        r"\(lr=0\.001, lr=0\.01\)"):
        Plan(os.path.join(root, "documents", "das_apply.json")).compile()


def test_a_root_value_without_rows_is_the_bodys_to_use_whole(engines):
    """A root mean over a `layers: ":"` read is `[layers, hidden_size]`: no row per example, so every run of the
    body reads it whole, as it was made."""
    def change(d):
        d["steps"] = {"lens": {"kind": "forward", "input": "train.input",
                               "interventions": {"h": {"site": {"component": "layer_output", "layers": ":"}, "pos": -1}}},
                      "avg": {"kind": "op", "op": "mean", "of": "lens.h", "over": ["rows", "pos"]}, **d["steps"]}
        d["steps"]["fit"]["steps"]["spread"] = {"kind": "op", "op": "mean", "of": "avg", "over": ["hidden_size"]}
        d["saves"] += ["avg", "fit.train.spread", "fit.history.spread"]
    plan = edited("das", change).compile()
    assert dims(plan, "avg") == ["layers", "hidden_size"]
    values = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert torch.equal(values["fit.train.spread"], values["avg"].mean(-1))
    assert torch.equal(values["fit.history.spread"], values["fit.train.spread"].mean().expand(10))


def held_out(label: str):
    """The probe with the harvest in its body and an eval over a copy of its rows, the first relabelled `label`."""
    def change(d):
        held = [dict(r) for r in d["data"]["days"]["rows"]]
        held[0]["label"] = label
        d["data"]["held"] = {"rows": held}
        d["steps"]["fit"]["eval"] = {"train": {"days": "held"}}
    return chain(inside("read"), change)


def test_a_probes_labels_need_not_be_text():
    def ints(d):
        for row in d["data"]["days"]["rows"]:
            row["label"] = int(row["label"] == "weekend")
    plan = probe(ints)
    assert plan.compiled


def test_a_probe_class_column_is_compared_exactly(engines):
    plan = probe(lambda d: d["steps"]["fit"]["steps"]["ce"].update(target={"column": "days.label", "lower": True}))
    with pytest.raises(ValueError, match=r"ce: target names a class of probe 'probe', which is compared exactly; "
                                         r"strip and lower compare an answer's tokens"):
        engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)


def test_a_probe_label_outside_its_classes_is_refused_before_any_forward(engines):
    """Under eval the label column is the held-out dataset's; a label there the probe has no class for is
    refused naming that dataset, before the fit's first forward."""
    plan = probe(held_out("holiday"))
    engine = engines("hooks", plan.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=r"ce: target is 'label' of 'held', whose row 0 is 'holiday', not one of "
                                             r"the classes of probe 'probe': \['weekday', 'weekend'\]"):
            engine.execute(plan, batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    assert not calls


def applied(corpus_root: str, name: str, featurizer: dict, steps: dict, data: dict) -> Plan:
    """A document that loads `featurizer` from the corpus's artifacts and runs `steps` with it."""
    doc = {"model": TINY, "data": data, "featurizers": {name: featurizer}, "steps": steps,
           "saves": [next(iter(steps)) + "." + next(iter(next(iter(steps.values()))["interventions"]))]}
    with open(os.path.join(corpus_root, "documents", "applied.json"), "w") as f:
        json.dump(doc, f)
    return Plan(os.path.join(corpus_root, "documents", "applied.json")).compile()


def test_a_trained_probe_loads_from_its_bundle_and_checks_its_classes(engines, tmp_path):
    root = corpus_copy(tmp_path)
    trained = probe()
    engine = engines("hooks", trained.model)
    filled = engine.execute(trained, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "documents", "artifacts"))
    data = probe().model_dump(mode="json", exclude_defaults=True)["data"]
    data["days"] = {"rows": data["days"]["rows"]}
    read = {"read": {"kind": "forward", "input": "days.input", "interventions": {"scores": {
        "site": {"component": "layer_output", "layers": 1}, "pos": -1, "ops": [{"op": "encode", "featurizer": "probe"}]}}}}
    spec = {"kind": "linear", "classes": "days.label", "file_path": "documents/artifacts/fit.probe.safetensors"}
    plan = applied(root, "probe", spec, read, data)
    assert same(engine.execute(plan, batch_size=BATCH_SIZE).results["read.scores"], filled.results["fit.train.scores"])
    for row in data["days"]["rows"]:
        row["label"] = "rest" if row["label"] == "weekend" else row["label"]
    with pytest.raises(PlanError, match=r"header 'classes' is '\[\"weekday\", \"weekend\"\]'; this document expects "
                                        r"'\[\"rest\", \"weekday\"\]'"):
        applied(root, "probe", spec, read, data)


def test_a_trained_gate_loads_from_its_bundle(engines, tmp_path):
    """dbm at one L1 weight, its gate loaded by a document that runs dbm's body at the root through it: the
    numbers of its `train` eval, to the bit."""
    root = corpus_copy(tmp_path)
    one = edited("dbm", lambda d: d["steps"]["fit"]["objective"].update({"mask.mask": 0.01})).compile()
    engine = engines("hooks", one.model)
    filled = engine.execute(one, batch_size=BATCH_SIZE)
    filled.dump(os.path.join(root, "documents", "artifacts"))

    def loaded(d):
        d["featurizers"]["mask"]["file_path"] = "documents/artifacts/fit.mask.safetensors"
        d["steps"] = {"counterfactual": d["steps"]["counterfactual"], **d["steps"]["fit"]["steps"]}
        for s in d["steps"].values():
            for o in [s] + [o for e in s.get("interventions", {}).values() for o in e.get("ops", [])]:
                if "featurizer" in o:
                    o["featurizer"] = "mask"
        d["saves"] = ["iia", "ce"]
    plan = edited("dbm", loaded)
    plan = Plan(**plan.model_dump(exclude_defaults=True, exclude={"document"}),
                document=os.path.join(root, "documents", "dbm.json")).compile()
    assert same(engine.execute(plan, batch_size=BATCH_SIZE).results,
                {k: filled.results[f"fit.train.{k}"] for k in ("iia", "ce")})


def test_inside_the_body_a_trained_featurizers_two_names_are_one_space(engines):
    """In das's body `rot` and `fit.rot` both name the rotation being trained: the counterfactual encoded as
    `rot` (by an encode step) is an operand of the bracket written `fit.rot`, and the fit runs to the bit as das
    does."""
    def change(d):
        d["steps"]["fit"]["steps"]["v_cf"]["featurizer"] = "rot"
    plan = edited("das", change).compile()
    engine = engines("hooks", plan.model)
    assert same(engine.execute(plan, batch_size=BATCH_SIZE).results,
                engine.execute(edited("das", lambda d: None).compile(), batch_size=BATCH_SIZE).results)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_fits_eval_reads_a_file_whose_columns_are_renamed_to_the_bodys(engines, tmp_path, kind):
    """das's held-out rows as a file of other column names, renamed back by `columns`: the rows it materializes
    are the split's, digest included, and the steps after the fit score them the same, to the bit."""
    shutil.copytree(DOCUMENTS, tmp_path / "documents")
    shutil.copytree(os.path.join(ROOT, "data"), tmp_path / "data")
    names = {"input": "prompt", "cf_answer": "target", "base_answer": "baseline", "counterfactual_inputs": "cfs"}
    with open(tmp_path / "data" / "weekdays" / "data.json") as f:
        held = [r for r in json.load(f) if r["split"] == "test"]
    os.makedirs(tmp_path / "data" / "renamed")
    with open(tmp_path / "data" / "renamed" / "held.json", "w") as f:
        json.dump([{names.get(k, k): v for k, v in r.items()} for r in held], f)        # `prompt` for `input`, ...

    def das(change=None) -> Plan:
        with open(tmp_path / "documents" / "das.json") as f:
            doc = json.load(f)
        if change:
            change(doc)
        return Plan(**doc, document=str(tmp_path / "documents" / "das.json")).compile()
    renamed = das(lambda d: d["data"].update(test={"path": "renamed/held", "columns": names}))
    original = das()
    assert renamed.data["test"].rows == original.data["test"].rows == held
    assert renamed.data["test"].sha256 == original.data["test"].sha256
    engine = engines(kind, original.model)
    assert same(engine.execute(renamed, batch_size=BATCH_SIZE).results,
                engine.execute(original, batch_size=BATCH_SIZE).results)


# -- a body that reads a harvest made before the fit ------------------------------------------------------------


def counted(engine, plan: Plan, batch_size: int) -> tuple[dict, list[int]]:
    """`plan` run on `engine`, and the rows of each batch the model was called on, in order."""
    batches: list[int] = []
    original = engine.forward
    engine.forward = lambda batch, records: batches.append(len(batch["input_ids"])) or original(batch, records)
    try:
        return engine.execute(plan, batch_size=batch_size).results, batches
    finally:
        del engine.forward


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_probe_calls_the_model_once_to_harvest_and_never_while_it_trains(engines, kind):
    """The probe's read is one harvest before the fit: one model call over its 8 rows in one batch of 8, and
    none while the fit trains, scores each epoch or runs its `train` eval, where the body is its encode of
    each minibatch's rows of the harvest. The trained probe calls every row right. Made by the body, the read
    runs on every minibatch and every scoring pass: 30 epochs of two minibatches and two batches, and two for
    the eval."""
    values, batches = counted(engines(kind, probe().model), probe(), 8)
    assert batches == [8]
    assert values["fit.train.hit"].eq(1).all()
    assert len(counted(engines(kind, probe().model), probe(inside("read")), 8)[1]) == 30 * (2 + 2) + 2


def oracle(name: str, *changes) -> tuple[Plan, Plan, int]:
    """Fit document `name` (`probe` for the probe) with its harvest before the fit and with it inside the body,
    each with `changes`, and the batch size that makes the harvest one batch of every row."""
    root = {"das": "counterfactual", "dbm": "counterfactual", "probe": "read"}[name]
    def plan(*more) -> Plan:
        return probe(*changes, *more) if name == "probe" else edited(name, chain(*changes, *more)).compile()
    return plan(), plan(inside(root)), 8 if name == "probe" else 2


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("name", ["probe", "das"])
def test_a_body_reads_its_rows_of_a_harvest_exactly_as_it_would_make_them(engines, kind, name):
    """The equivalence oracle. With the harvest made in one batch of every row and the fit's minibatch every
    row (one batch, the same rows), the body that reads the harvest gives what the body that makes it gives,
    to the bit: every saved value, the history and the trained featurizer. So each run of the body reads the
    harvest cut to its rows exactly, in the minibatch's shuffled order and in the scoring pass's."""
    def every_row(d):
        d["steps"]["fit"]["batch_size"] = 8 if name == "probe" else 2
    outside, within, size = oracle(name, every_row)
    engine = engines(kind, outside.model)
    assert same(engine.execute(outside, batch_size=size).results, engine.execute(within, batch_size=size).results)


VARIANTS = {
    # name, change, batch size: the harvest's batches as the minibatches, so their shapes are the body's
    "a trailing partial minibatch": ("probe", lambda d: d["steps"]["fit"].update(batch_size=3), 3),
    "batch_size 1": ("das", lambda d: d["steps"]["fit"].update(batch_size=1), 1),
    "epochs 0": ("das", lambda d: d["steps"]["fit"].update(epochs=0), 2),
    "a swept fit": ("dbm", lambda d: None, 2),
    "early stopping with a restore": ("das", lambda d: d["steps"]["fit"]["early_stop"].update(mode="min"), 2),
    "another seed": ("das", lambda d: (d["steps"]["fit"].update(seed=1), d["featurizers"]["rot"].update(seed=0)), 2),
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_fit_that_reads_a_harvest_matches_the_one_that_makes_it(engines, variant):
    """Each shape of fit, its harvest made in batches as large as its minibatches, gives what it gives with the
    harvest in its body, to the bit: 8 rows in minibatches of 3 (the last of 2); one row a minibatch; no epoch
    (the untrained draw scored); dbm's four points; das stopped early, its first epoch restored; and another
    order of rows."""
    name, change, size = VARIANTS[variant]
    outside, within, _ = oracle(name, change)
    engine = engines("hooks", outside.model)
    got, made = (engine.execute(p, batch_size=size).results for p in (outside, within))
    assert same(got, made)
    if variant == "epochs 0":
        assert engine.execute(edited("das", chain(change, lambda d: d["saves"].append("fit.history.ce"))).compile(),
                              batch_size=size).results["fit.history.ce"].shape == (0,)
    if variant == "a swept fit":
        assert set(got["fit.mask"]) == {0.0001, 0.002, 0.01, 0.05}


def test_a_fits_members_keep_their_forms_when_its_body_reads_a_harvest(engines):
    """das reads its counterfactual harvest in the body: `fit.history.<op>` is one entry per epoch run, a
    `train` eval member a row per training pair, `fit.rot` the trained rotation (saved as its bundle) and what
    the steps after the fit read; the early stop is judged on the history."""
    plan = with_history(lambda d: d["saves"].append("fit.train.v_cf"))
    assert [dims(plan, n) for n in ("fit.history.ce", "fit.train.iia", "fit.train.v_cf", "fit.rot", "held_iia")] \
        == [["epochs"], ["rows", "pos"], ["rows", "pos", "features"], [], ["rows", "pos"]]
    values = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert values["fit.history.ce"].shape == values["fit.history.iia"].shape == (10,)
    assert values["fit.train.iia"].shape == values["held_iia"].shape == (2, 1)
    assert values["fit.train.v_cf"].shape == (2, 1, 8) and values["fit.rot"]["basis"].shape == (16, 8)


def held_out_eval(d):
    """das's held-out evaluation as an eval of a body that makes its own harvest, beside the train eval."""
    inside("counterfactual")(d)
    d["steps"]["fit"]["eval"] = {"held_out": {"train": "test"}, "train": {}}
    d["saves"] += ["fit.held_out.iia", "fit.held_out.hit"]


def hits(d):
    """A `match` beside das's iia, in the body and after the fit."""
    d["steps"]["fit"]["steps"]["hit"] = {"kind": "op", "op": "match", "of": "patched.logits",
                                         "expected": "train.cf_answer"}
    d["steps"]["held_hit"] = {"kind": "op", "op": "match", "of": "held_patched.logits", "expected": "test.cf_answer"}
    d["saves"].append("held_hit")


def test_an_eval_that_reads_a_harvests_dataset_as_another_is_refused_naming_both():
    """das's body reads the counterfactual harvest, made over `train`; an eval with `test` for `train` would
    need it over `test`, which no step made, so it is refused at compile, naming the eval, the value and the
    dataset, and saying where such an evaluation goes."""
    with pytest.raises(PlanError, match=r"^steps\.fit\.eval\.held_out: the body uses 'counterfactual\.h', made before "
                                        r"the fit with a row per example of 'train', and this eval reads 'train' as "
                                        r"'test', over whose rows no step made it; evaluate on 'test' in steps after "
                                        r"the fit, which reach what it trained as 'fit\.rot'$"):
        edited("das", lambda d: d["steps"]["fit"]["eval"].update(held_out={"train": "test"})).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_the_held_out_steps_after_the_fit_give_what_a_held_out_eval_gave(engines, kind):
    """das's held-out evaluation in steps after the fit (a counterfactual read of `test` and a patched forward
    through `fit.rot`) gives, to the bit, the `held_out` eval of a body that makes its own harvest: its match and
    its iia, with the fit's minibatch every training row."""
    every_row = lambda d: d["steps"]["fit"].update(batch_size=2)        # noqa: E731
    after, evaluated = (edited("das", chain(every_row, hits, *more)).compile() for more in ((), (held_out_eval,)))
    engine = engines(kind, after.model)
    got, held = (engine.execute(p, batch_size=2).results for p in (after, evaluated))
    assert same(got["held_hit"], held["fit.held_out.hit"]) and same(got["held_iia"], held["fit.held_out.iia"])
    assert same(got["fit.rot"], held["fit.rot"])


def other_dataset(d):
    """das with a read of the held-out prompts before the fit, which the body names."""
    d["steps"] = {"other": {"kind": "forward", "input": "test.input",
                            "interventions": {"logits": {"site": "head", "pos": -1}}}, **d["steps"]}


def few(d):
    """The probe with a harvest over 4 other prompts before the fit, which its body averages."""
    d["data"]["few"] = {"rows": [{"input": f"If today is {day}"} for day in DAYS[:4]]}
    d["steps"] = {"few_read": {"kind": "forward", "input": "few.input", "interventions": {"h": {
        "site": {"component": "layer_output", "layers": 1}, "pos": -1}}}, **d["steps"]}
    d["steps"]["fit"]["steps"]["spread"] = {"kind": "op", "op": "mean", "of": "few_read.h"}


def every_day(d):
    """`few` as many rows as the probe's own dataset, so that only what pairs them is left to refuse."""
    d["data"]["few"]["rows"] = [{"input": f"If today is {day}"} for day in DAYS]


@pytest.mark.parametrize("plan, message", [
    (lambda: edited("das", chain(other_dataset, lambda d: d["steps"]["fit"]["steps"].update(gap={
        "kind": "op", "op": "sub", "of": "patched.logits", "with": "other.logits"}))).compile(),
     r"steps\.fit\.steps\.gap\.with: 'patched\.logits' has a row per example of 'train' and 'other\.logits' of 'test'; "
     r"an op takes one dataset's rows"),
    (lambda: probe(few), r"steps\.fit\.steps: the body's datasets have \{.*'few': 4.*\} rows; it runs over minibatches "
                         r"of the same rows of each, paired row by row"),
    (lambda: edited("das", chain(other_dataset, lambda d: d["steps"]["fit"]["steps"].update(o={
        "kind": "op", "op": "token_prob", "of": "other.logits", "token": "test.cf_answer"}))).compile(),
     r"^steps\.fit\.steps\.o: the body uses 'other\.logits', made before the fit with a row per example of 'test', "
     r"which no forward of the body reads; the body runs over the rows of 'train', so a minibatch would take rows of "
     r"'test' at the ids of its rows of 'train', pairing the two row by row: read 'test' in a forward of the body, or "
     r"make 'other\.logits' over 'train'$"),
    (lambda: probe(few, every_day), r"^steps\.fit\.steps\.spread: the body uses 'few_read\.h', made before the fit "
                                    r"with a row per example of 'few', which no forward of the body reads; the body "
                                    r"runs over the rows of 'days'"),
])
def test_a_harvest_of_a_dataset_the_body_does_not_read_is_refused(plan, message):
    """A harvest's rows are a dataset's, and the body's datasets are those its forwards read (with none, the one its
    first harvest is of): a harvest of another beside the body's own rows in one op is refused as any op over two
    datasets is, one of another row count by the row rule, and one of as many rows, which a minibatch's ids would
    pair with the body's row by row, by name, in a body with a forward (das) and in one without (the probe)."""
    with pytest.raises(PlanError, match=message):
        plan()


def test_a_featurizer_step_over_an_ops_value_is_refused_by_name():
    """An encode step meets a read, at its site; one over an op's value (layer 0's read less layer 1's) has no one
    site, so it is refused, naming the value."""
    def two_layers(d):
        d["steps"] = {"counterfactual": d["steps"]["counterfactual"],
                      "upper": {"kind": "forward", "input": "train.counterfactual_inputs[0]", "interventions": {
                          "h": {"site": {"component": "layer_output", "layers": 1}, "pos": -1}}},
                      "diff": {"kind": "op", "op": "sub", "of": "counterfactual.h", "with": "upper.h"},
                      **{k: v for k, v in d["steps"].items() if k != "counterfactual"}}
        d["steps"]["fit"]["steps"]["v_cf"]["of"] = "diff"
    with pytest.raises(PlanError, match=r"^steps\.fit\.steps\.v_cf\.of: 'diff' is an op's value, which no one site "
                                        r"made, and a featurizer's encode acts on a value at its site: encode a read "
                                        r"\(an entry's value\), or encode inside the entry that reads$"):
        edited("das", two_layers).compile()


@pytest.mark.parametrize("name, change, terms", [
    ("das", lambda d: (d["steps"]["fit"]["steps"].update(plain={"kind": "op", "op": "mean", "of": "counterfactual.h"}),
                       d["steps"]["fit"].update(objective={"plain": 1.0})), r"plain\) depends on what the fit trains \(rot\)"),
    ("lora_train", lambda d: d["steps"]["fit"]["steps"]["adapted"]["interventions"].pop("adapt"),
     r"ce\) depends on what the fit trains \(lora\)"),
])
def test_an_objective_that_reaches_nothing_the_fit_trains_is_refused_naming_its_terms(name, change, terms):
    """An objective none of whose terms a trained featurizer reaches has no gradient: a mean of the harvest, or
    a forward that no longer runs the adapter, is refused at compile, naming the terms and what the fit trains."""
    with pytest.raises(PlanError, match=r"steps\.fit\.objective: none of its terms \(" + terms):
        edited(name, change).compile()


# -- a row is a dataset row id, in a fit's minibatch too ---------------------------------------------------------


WEEK = [*DAYS, "Monday", "Friday", "Tuesday", "Sunday"]
JITTER = {"site": {"component": "layer_output", "layers": 0}, "pos": -1,
          "ops": [{"op": "swap", "with": 0.0}, {"op": "noise", "scale": 1.0, "seed": 7}]}
EVERY = {"site": {"component": "layer_output", "layers": 1}, "pos": {"index": ":"}}


def jittered() -> Plan:
    """A probe of layer 1's output at the last token, 12 prompts in minibatches of 4 for two epochs, whose body's
    forward replaces layer 0's output there by noise (`jitter`) and reads every token of layer 1 (`every`, as many
    positions as the row has tokens); and a plain forward over the same rows that does the same, before the fit."""
    rows = [{"input": f"If today is {d}", "label": "weekend" if d in ("Saturday", "Sunday") else "weekday"}
            for d in WEEK]
    return Plan(**{
        "model": TINY, "data": {"days": {"rows": rows}},
        "featurizers": {"probe": {"kind": "linear", "classes": "days.label"}},
        "steps": {
            "plain": {"kind": "forward", "input": "days.input", "interventions": {"jitter": JITTER, "every": EVERY}},
            "fit": {"kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 2, "batch_size": 4,
                    "seed": 0, "optimizer": {"name": "adam", "lr": 0.1}, "eval": {"train": {}}, "steps": {
                        "noisy": {"kind": "forward", "input": "days.input", "interventions": {
                            "jitter": JITTER, "every": EVERY,
                            "scores": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                                       "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
                        "ce": {"kind": "op", "op": "cross_entropy", "of": "noisy.scores", "target": "days.label"}}}},
        "saves": ["plain.jitter", "plain.every", "fit.train.noisy.jitter", "fit.train.noisy.every"]}).compile()


def test_noise_in_a_fit_body_draws_by_dataset_row_in_any_minibatch(engines, monkeypatch):
    """A minibatch is its rows' dataset row ids, which `noise` is given: each training minibatch's rows are the
    epoch's shuffle of the 12 rows in fours (not rows 0 to 3 of a minibatch renumbered from 0), each row gets the
    same draw in whatever minibatch it lands in, in each epoch's scoring pass, in the eval and in the plain forward
    over the same rows, and no two rows share one; so the eval's noise is the plain forward's, to the bit."""
    noise, drawn = ops.OPS["noise"], []

    @functools.wraps(noise)
    def recorded(of, scale, dims, seed=0, rows=None, positions=None):
        out = noise(of, scale, dims, seed, rows, positions)
        drawn.append((cast(torch.Tensor, rows).tolist(), out.detach().clone()))
        return out
    monkeypatch.setitem(ops.OPS, "noise", recorded)
    plan = jittered()
    values = engines("hooks", plan.model).execute(plan, batch_size=4).results
    every = [list(range(lo, lo + 4)) for lo in range(0, 12, 4)]
    shuffled = [torch.randperm(12, generator=torch.Generator().manual_seed(epoch)).tolist() for epoch in (0, 1)]
    minibatches = [[order[lo:lo + 4] for lo in range(0, 12, 4)] for order in shuffled]
    assert [rows for rows, _ in drawn] == every + minibatches[0] + every + minibatches[1] + every + every
    by_row: dict = {}
    for rows, out in drawn:
        for r, draw in zip(rows, out):
            by_row.setdefault(r, []).append(draw)
    assert all(len(d) == 6 and all(torch.equal(x, d[0]) for x in d) for d in by_row.values())
    assert len({tuple(d[0].flatten().tolist()) for d in by_row.values()}) == 12
    assert same(values["fit.train.noisy.jitter"], values["plain.jitter"])
    assert same(values["fit.train.noisy.every"], values["plain.every"])


def test_a_fits_eval_records_and_tables_by_dataset_row(engines):
    """A record's `positions` and `clab table` number a fit's eval values by dataset row as they number a forward's: the
    eval's read of every token resolves on each row to that row's own positions, the plain forward's, and its
    table's rows are the plain forward's, row, position and token, with the same values."""
    plan = jittered()
    filled = engines("hooks", plan.model).execute(plan, batch_size=4)
    positions = placed(filled)
    assert positions["fit.train.noisy.every"] == positions["plain.every"]
    assert len({len(p) for p in positions["plain.every"]}) > 1          # rows of different lengths
    fit, plain = filled.table("fit.train.noisy.every"), filled.table("plain.every")
    assert [(r["row"], r["pos"], r["pos_token"]) for r in fit] == [(r["row"], r["pos"], r["pos_token"]) for r in plain]
    assert {r["row"] for r in fit} == set(range(12)) and same([r["value"] for r in fit], [r["value"] for r in plain])
