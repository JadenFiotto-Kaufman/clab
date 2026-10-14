"""The lora kind: an adapter that starts as the identity exactly, trains through the frozen model the same on
every path, loads from its bundle as it was trained, and applies across sites with `with`; and lora_apply's
always-on adapter as a named intervention, on forwards and through a generation."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from clab import Plan, featurizers
from clab.plan import PlanError, points
from conftest import BATCH_SIZE, bundle_header, edited, same, stand_in
from test_fit import TINY, corpus_copy

G = torch.Generator().manual_seed(5)


def lora_train(change=None) -> Plan:
    def edit(d):
        d["saves"].append("fit.held_out.adapted.logits")
        if change:
            change(d)
    return edited("lora_train", edit).compile()


def test_an_untrained_adapter_is_the_identity_to_the_bit():
    f = featurizers.make("lora", {"rank": 4, "alpha": 8}, 16, None, 0, None)
    h, other = torch.randn(3, 2, 16, generator=G), torch.randn(3, 2, 4, 4, generator=G)
    assert torch.equal(featurizers.call(f, "apply", h, {}), h)
    assert torch.equal(featurizers.call(f, "apply", h, {"with": other}), h)
    assert f["A"].shape == (16, 4) and f["B"].eq(0).all() and f["scale"] == 2.0


def test_apply_is_h_plus_the_scaled_low_rank_update():
    f = {"kind": "lora", "A": torch.randn(16, 4, generator=G), "B": torch.randn(4, 16, generator=G), "scale": 2.0}
    h, other = torch.randn(3, 2, 16, generator=G), torch.randn(3, 2, 4, 4, generator=G)
    assert torch.equal(featurizers.call(f, "apply", h, {}), h + 2.0 * ((h @ f["A"]) @ f["B"]))
    assert torch.equal(featurizers.call(f, "apply", h, {"with": other}),
                       h + 2.0 * ((other.flatten(2) @ f["A"]) @ f["B"]))


def test_training_an_adapter_lowers_ce_the_same_everywhere(engines, monkeypatch):
    """lora_train's ce falls over the fit; nnterp, hooks, remote="local" and the stand-in server give the same
    numbers to the bit, and the model's weights get no gradient."""
    engine = engines("nnterp", lora_train().model)
    values = engine.execute(lora_train(), batch_size=BATCH_SIZE).results
    ce = values["fit.history.ce"]
    assert ce.shape == (20,) and ce[-1] < ce[0] and (ce[1:] < ce[:-1]).all()
    assert torch.equal(ce[-1], values["fit.train.ce"].mean())
    assert same(engines("hooks", lora_train().model).execute(lora_train(), batch_size=BATCH_SIZE).results, values)
    assert same(engine.execute(lora_train(), batch_size=BATCH_SIZE, remote="local").results, values)
    came, trip = stand_in(engine, lora_train(), monkeypatch)
    assert trip["by_reference"] and trip["names"] == ["plan"] and same(came.results, values)
    assert not values["fit.lora"]["B"].eq(0).all()
    assert all(p.grad is None for p in engine.model.parameters())


def test_a_loaded_adapter_applies_as_the_trained_one_did(engines, tmp_path):
    """lora_apply loads the bundle lora_train wrote, rank and alpha from its header: its adapted logits on the
    held-out prompts are lora_train's held-out eval's, to the bit, and its apply on any value the trained
    instance's."""
    root = corpus_copy(tmp_path)
    engine = engines("hooks", lora_train().model)
    trained = engine.execute(lora_train(), batch_size=BATCH_SIZE)
    trained.dump(os.path.join(root, "documents", "artifacts"))
    with open(os.path.join(root, "documents", "lora_apply.json")) as f:
        doc = json.load(f)
    doc["saves"].append("adapted.logits")
    plan = Plan(**doc, document=os.path.join(root, "documents", "lora_apply.json")).compile()
    spec = points(plan.featurizers["lora"])[0]
    loaded = featurizers.make("lora", {"file_path": "x"}, 16, spec.bundle, None, None, spec.header)
    assert loaded["scale"] == 2.0 and (spec.header or {})["rank"] == "4"
    h = torch.randn(4, 16, generator=G)
    f = {**trained.results["fit.lora"], "kind": "lora", "scale": 8 / 4}
    assert torch.equal(featurizers.call(loaded, "apply", h, {}), featurizers.call(f, "apply", h, {}))
    applied = engine.execute(plan, batch_size=BATCH_SIZE).results
    assert same(applied["adapted.logits"], trained.results["fit.held_out.adapted.logits"])


def test_the_adapter_is_on_in_every_step_that_names_it(engines, tmp_path):
    """lora_apply: the adapted forward's logits differ from the clean one's, and the generation runs with the
    adapter at the prompt and at every decode call: its decode-call logits differ from a generation whose
    adapter is written in the prefill only."""
    root = corpus_copy(tmp_path)
    engine = engines("nnterp", lora_train().model)
    engine.execute(lora_train(), batch_size=BATCH_SIZE).dump(os.path.join(root, "documents", "artifacts"))

    def compiled(frame) -> Plan:
        with open(os.path.join(root, "documents", "lora_apply.json")) as f:
            doc = json.load(f)
        doc["interventions"]["adapted"]["writes"]["adapt"]["pos"]["frame"] = frame
        doc["steps"]["answer"]["reads"] = {"later": {"site": "head", "pos": {"frame": [1, 4], "index": [0, 3]}}}
        doc["saves"] += ["adapted.logits", "clean.logits", "answer.later"]
        return Plan(**doc, document=os.path.join(root, "documents", "lora_apply.json")).compile()
    on, prefill_only = (engine.execute(compiled(frame), batch_size=BATCH_SIZE).results for frame in ("all", 0))
    assert not torch.equal(on["adapted.logits"], on["clean.logits"])
    assert not torch.equal(on["adapted_ld"], on["clean_ld"])
    assert len(on["said"]) == 2 and all(isinstance(text, str) for text in on["said"])
    assert on["answer.later"].shape == (2, 3, 32000)
    for k in range(3):
        assert not torch.equal(on["answer.later"][:, k], prefill_only["answer.later"][:, k]), k
    assert same(engines("hooks", lora_train().model).execute(compiled("all"), batch_size=BATCH_SIZE).results, on)


# -- an adapter across sites: o_proj, read at attention_premix and applied at attention_output ----------------


def cross_site(root: str, steps: dict, saves: list[str], train: bool = False) -> Plan:
    """A document with a rank-2 adapter whose input is layer 0's `attention_premix` and whose update lands on
    its `attention_output`: drawn to be trained, or loaded from a random bundle made at that site."""
    spec = {"kind": "lora", "rank": 2, "alpha": 4} if train else \
        {"kind": "lora", "file_path": "documents/artifacts/oproj.safetensors"}
    if not train:
        save_file({"A": torch.randn(16, 2, generator=G), "B": torch.randn(2, 16, generator=G)},
                  os.path.join(root, "documents", "artifacts", "oproj.safetensors"),
                  metadata=bundle_header(kind="lora", rank="2", alpha="4", component="attention_output",
                                         document="oproj"))
    doc = {"model": TINY, "data": {"train": {"path": "weekdays/data", "split": "train"}},
           "featurizers": {"oproj": spec}, "steps": steps, "saves": saves}
    with open(os.path.join(root, "documents", "oproj.json"), "w") as f:
        json.dump(doc, f)
    return Plan(os.path.join(root, "documents", "oproj.json")).compile()


def adapted(name: str, featurizer: str, pos) -> dict:
    """A forward that reads layer 0's premix and adapts its attention output with it, reading that after."""
    premix, out = ({"component": c, "layers": 0} for c in ("attention_premix", "attention_output"))
    return {"kind": "forward", "input": "train.input",
            "writes": {"adapt": {"site": out, "pos": pos, "ops": [
                {"op": "featurizer", "featurizer": featurizer, "mode": "apply", "with": f"{name}.premix"}]}},
            "reads": {"premix": {"site": premix, "pos": pos}, "out": {"site": out, "pos": pos},
                      "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}


@pytest.mark.parametrize("pos", [-1, {"index": "all"}])
def test_a_cross_site_adapter_adds_its_update_to_the_live_value_on_both_engines(engines, tmp_path, pos):
    root = corpus_copy(tmp_path)
    clean = {"kind": "forward", "input": "train.input",
             "reads": {"out": {"site": {"component": "attention_output", "layers": 0}, "pos": pos}}}
    plan = cross_site(root, {"clean": clean, "patched": adapted("patched", "oproj", pos)},
                      ["clean.out", "patched.premix", "patched.out"])
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")]
    assert same(*got)
    values = got[0]
    bundle = points(plan.featurizers["oproj"])[0].bundle
    assert bundle is not None
    unwrap = (lambda v: v["values"]) if isinstance(pos, dict) else (lambda v: v)
    premix = unwrap(values["patched.premix"])
    expected = unwrap(values["clean.out"]) + 2.0 * ((premix.flatten(-2) @ bundle["A"]) @ bundle["B"])
    torch.testing.assert_close(unwrap(values["patched.out"]), expected, atol=1e-6, rtol=0)
    assert not torch.allclose(unwrap(values["patched.out"]), unwrap(values["clean.out"]))


def test_a_cross_site_adapter_trains_the_same_on_both_engines(engines, tmp_path):
    root = corpus_copy(tmp_path)
    fit = {"kind": "fit", "train": ["oproj"], "objective": {"ce": 1.0}, "epochs": 5, "batch_size": 2, "seed": 0,
           "optimizer": {"name": "adam", "lr": 0.05}, "eval": {"train": {}},
           "steps": {"patched": adapted("patched", "fit.oproj", -1),
                     "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"}}}
    plan = cross_site(root, {"fit": fit}, ["fit.history.ce", "fit.oproj"], train=True)
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")]
    assert same(*got)
    assert got[0]["fit.history.ce"][-1] < got[0]["fit.history.ce"][0]


# -- refusals -------------------------------------------------------------------------------------------------


def premix_after(d):
    """lora_train's write reads `with` from a read at layer 1, which the forward reaches after layer 0."""
    body = d["steps"]["fit"]["steps"]["adapted"]
    body["reads"]["later"] = {"site": {"component": "layer_output", "layers": 1}, "pos": {"index": "all"}}
    body["writes"]["adapt"]["ops"][0]["with"] = "adapted.later"


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["fit"]["steps"]["adapted"]["writes"]["adapt"]["ops"][0].update(mode="encode"),
     r"featurizer 'fit\.lora': kind 'lora' has no mode 'encode'; its modes: apply"),
    (lambda d: d["featurizers"]["lora"].update(file_path="documents/artifacts/fit.lora.safetensors"),
     r"featurizers\.lora: 'lora' \(lora\) is loaded from its bundle \('file_path'\) or drawn to be trained "
     r"\('rank' and 'alpha'\): one of them"),
    (lambda d: d["featurizers"]["lora"].pop("alpha"), r"'lora' \(lora\) is loaded from its bundle"),
    (lambda d: d["steps"]["fit"]["steps"]["adapted"]["writes"]["adapt"]["ops"].__setitem__(
        0, {"op": "scale", "factor": 2.0, "featurizer": "fit.lora"}),
     r"'fit\.lora' is a 'lora' featurizer, which has no coordinates to edit and write back into the value; call it with the "
     r"featurizer op \(modes: apply\)"),
    (premix_after, r"adapt\.ops\[0\]\.with: 'adapted\.later' is read at layer_output layer 1, which the forward does "
                   r"not reach before this write's layer_output layer 0"),
])
def test_what_an_adapter_cannot_be_is_refused_at_compile(change, message):
    with pytest.raises(PlanError, match=message):
        edited("lora_train", change).compile()


def test_a_generations_writes_take_no_operand_from_its_own_reads(corpus):
    with open(os.path.join(corpus, "documents", "lora_apply.json")) as f:
        doc = json.load(f)
    doc["steps"]["answer"]["reads"] = {"h": {"site": "target", "pos": {"index": "all"}}}
    doc["interventions"]["adapted"]["writes"]["adapt"]["ops"][0]["with"] = "answer.h"
    with pytest.raises(PlanError, match=r"interventions\.adapted\.writes\.adapt\.ops\[0\]\.with: 'answer\.h' names no "
                                        r"value produced before this step"):
        Plan(**doc, document=os.path.join(corpus, "documents", "lora_apply.json")).compile()


def test_a_with_of_another_width_is_refused_before_any_forward(engines, tmp_path):
    """One head of the premix is 4 wide per position; the adapter's value, attention_output, is 16."""
    root = corpus_copy(tmp_path)
    step = adapted("patched", "oproj", -1)
    step["reads"]["premix"]["site"]["select"] = {"num_heads": [0]}
    plan = cross_site(root, {"patched": step}, ["patched.out"])
    with pytest.raises(ValueError, match=r"patched\.adapt: operand 'patched\.premix' is 4 wide per position, its axes "
                                         r"after the positions flattened; this write's value is 16"):
        engines("hooks", plan.model).execute(plan)
