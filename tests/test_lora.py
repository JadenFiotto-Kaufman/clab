"""The lora kind: an adapter that starts as the identity exactly, trains through the frozen model the same on
every path, loads from its bundle as it was trained, and applies across sites with `with`; and lora_apply's
always-on adapter as a named intervention, on forwards and through a generation."""

import json
import os
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, TRIP, bundle_header, edited, same, stand_in
from safetensors.torch import save_file
from test_fit import TINY, corpus_copy

from clab import Plan
from clab.featurizers.lora import Lora
from clab.format.plan import PlanError
from clab.format.sweeps import points

G = torch.Generator().manual_seed(5)


def lora_train(change=None) -> Plan:
    def edit(d):
        d["saves"].append("fit.held_out.adapted.logits")
        if change:
            change(d)
    return edited("lora_train", edit).compile()


def test_an_untrained_adapter_is_the_identity_to_the_bit():
    f = Lora(16, rank=4, alpha=8, seed=0)
    h, other = torch.randn(3, 2, 16, generator=G), torch.randn(3, 2, 4, 4, generator=G)
    assert torch.equal(f.call("apply", h, {}), h)
    assert torch.equal(f.call("apply", h, {"with": other}), h)
    assert f.A.shape == (16, 4) and f.B.eq(0).all() and f.scale == 2.0


def test_apply_is_h_plus_the_scaled_low_rank_update():
    f = Lora(16, bundle={"A": torch.randn(16, 4, generator=G), "B": torch.randn(4, 16, generator=G)},
             header={"rank": "4", "alpha": "8"})
    h, other = torch.randn(3, 2, 16, generator=G), torch.randn(3, 2, 4, 4, generator=G)
    assert torch.equal(f.call("apply", h, {}), h + 2.0 * ((h @ f.A) @ f.B))
    assert torch.equal(f.call("apply", h, {"with": other}), h + 2.0 * ((other.flatten(2) @ f.A) @ f.B))


def test_training_an_adapter_lowers_ce_the_same_everywhere(engines, monkeypatch):
    """lora_train's ce falls over the fit; nnterp, hooks, remote="local" and the stand-in server give the same
    numbers to the bit, and the model's weights get no gradient."""
    engine = engines("nnterp", lora_train().model)
    values = engine.execute(lora_train(), batch_size=BATCH_SIZE).results
    ce = values["fit.history.ce"]
    assert ce.shape == (20,) and ce[-1] < ce[0] and (ce[1:] < ce[:-1]).all()
    assert torch.equal(ce[-1], values["fit.train.ce"].mean())
    assert same(engines("hooks", lora_train().model).execute(lora_train(), batch_size=BATCH_SIZE).results, values)
    there = engines("ndif", lora_train().model, remote="local").execute(lora_train(), batch_size=BATCH_SIZE)
    assert same(there.results, values)
    came, trip = stand_in(engines, lora_train(), monkeypatch)
    assert trip == TRIP and same(came.results, values)
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
    loaded = Lora(16, bundle=spec.bundle, header=spec.header)
    assert loaded.scale == 2.0 and (spec.header or {})["rank"] == "4"
    h = torch.randn(4, 16, generator=G)
    f = Lora(16, bundle=trained.results["fit.lora"], header={"rank": "4", "alpha": "8"})
    assert torch.equal(loaded.call("apply", h, {}), f.call("apply", h, {}))
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
        doc["interventions"]["adapted"]["adapt"]["pos"]["frame"] = frame
        doc["steps"]["answer"]["interventions"] = ["adapted", {"later": {"site": "head",
                                                                      "pos": {"frame": "1:4", "index": "0:3"}}}]
        doc["saves"] += ["adapted.logits", "clean.logits", "answer.later"]
        return Plan(**doc, document=os.path.join(root, "documents", "lora_apply.json")).compile()
    on, prefill_only = (engine.execute(compiled(frame), batch_size=BATCH_SIZE).results for frame in (":", 0))
    assert not torch.equal(on["adapted.logits"], on["clean.logits"])
    assert not torch.equal(on["adapted_ld"], on["clean_ld"])
    assert len(on["said"]) == 2 and all(isinstance(text, str) for text in on["said"])
    assert on["answer.later"].shape == (2, 3, 32000)
    for k in range(3):
        assert not torch.equal(on["answer.later"][:, k], prefill_only["answer.later"][:, k]), k
    assert same(engines("hooks", lora_train().model).execute(compiled(":"), batch_size=BATCH_SIZE).results, on)


# -- an adapter across sites: o_proj, read at attention_premix and applied at attention_output ----------------


def cross_site(root: str, steps: dict, saves: list[str], train: bool = False,
               interventions: dict | None = None) -> Plan:
    """A document with a rank-2 adapter whose input is layer 0's `attention_premix` and whose update lands on
    its `attention_output`: drawn to be trained, or loaded from a random bundle made at that site."""
    spec = {"kind": "lora", "rank": 2, "alpha": 4} if train else \
        {"kind": "lora", "file_path": "documents/artifacts/oproj.safetensors"}
    if not train:
        save_file({"A": torch.randn(16, 2, generator=G), "B": torch.randn(2, 16, generator=G)},
                  os.path.join(root, "documents", "artifacts", "oproj.safetensors"),
                  metadata=bundle_header(kind="lora", rank="2", alpha="4", component="attention_output",
                                         document="oproj", **{"with.component": "attention_premix",
                                                              "with.layer": "0"}))
    doc = {"model": TINY, "data": {"train": {"path": "weekdays/data", "split": "train"}},
           "featurizers": {"oproj": spec}, "interventions": interventions or {}, "steps": steps, "saves": saves}
    with open(os.path.join(root, "documents", "oproj.json"), "w") as f:
        json.dump(doc, f)
    return Plan(os.path.join(root, "documents", "oproj.json")).compile()


def adapted(name: str, featurizer: str, pos) -> dict:
    """A forward that reads layer 0's premix and adapts its attention output with it, reading that after."""
    premix, out = ({"component": c, "layers": 0} for c in ("attention_premix", "attention_output"))
    return {"kind": "forward", "input": "train.input", "interventions": {
        "premix": {"site": premix, "pos": pos},
        "adapt": {"site": out, "pos": pos, "ops": [{"op": "apply", "featurizer": featurizer, "with": f"{name}.premix"}]},
        "out": {"site": out, "pos": pos}, "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}


@pytest.mark.parametrize("pos", [-1, {"index": ":"}])
def test_a_cross_site_adapter_adds_its_update_to_the_live_value_on_both_engines(engines, tmp_path, pos):
    root = corpus_copy(tmp_path)
    clean = {"kind": "forward", "input": "train.input",
             "interventions": {"out": {"site": {"component": "attention_output", "layers": 0}, "pos": pos}}}
    plan = cross_site(root, {"clean": clean, "patched": adapted("patched", "oproj", pos)},
                      ["clean.out", "patched.premix", "patched.out"])
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")]
    assert same(*got)
    values = got[0]
    bundle = points(plan.featurizers["oproj"])[0].bundle
    assert bundle is not None
    unwrap = (lambda v: v.nan_to_num(0.0)) if isinstance(pos, dict) else (lambda v: v)     # NaN past a row's own
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


# -- a cross-site adapter as a named intervention ---------------------------------------------------------------

OPROJ = {"premix": {"site": {"component": "attention_premix", "layers": 0}, "pos": -1},
         "adapt": {"site": {"component": "attention_output", "layers": 0}, "pos": -1, "ops": [
             {"op": "apply", "featurizer": "oproj", "with": "premix"}]}}


def test_an_intervention_names_its_own_read_bare_and_each_step_uses_its_own(engines, tmp_path):
    """`oproj` reads the premix and adapts the attention output with `with: "premix"`; pulled into two forwards
    on different inputs, each one's output is its own clean output plus the update of its own premix."""
    root = corpus_copy(tmp_path)
    out = {"site": {"component": "attention_output", "layers": 0}, "pos": -1}
    steps = {}
    for name, column in (("a", "train.input"), ("b", "train.counterfactual_inputs[0]")):
        steps[f"{name}_clean"] = {"kind": "forward", "input": column, "interventions": {"out": out}}
        steps[name] = {"kind": "forward", "input": column, "interventions": ["oproj", {"out": out}]}
    saves = [f"{n}.{r}" for n in ("a", "b") for r in ("premix", "out")] + ["a_clean.out", "b_clean.out"]
    plan = cross_site(root, steps, saves, interventions={"oproj": OPROJ})
    step: Any = plan.steps["b"]
    assert [o.model_extra for o in step.interventions["adapt"].ops] == [{"featurizer": "oproj", "with": "b.premix"}]
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")]
    assert same(*got)
    bundle = points(plan.featurizers["oproj"])[0].bundle
    assert bundle is not None
    values = got[0]
    assert not torch.equal(values["a.premix"], values["b.premix"])
    for name in ("a", "b"):
        update = (values[f"{name}.premix"].flatten(-2) @ bundle["A"]) @ bundle["B"]
        expected = values[f"{name}_clean.out"] + 2.0 * update
        torch.testing.assert_close(values[f"{name}.out"], expected, atol=1e-6, rtol=0)


def test_an_intervention_naming_a_steps_read_as_its_own_is_refused(tmp_path):
    root = corpus_copy(tmp_path)
    oproj = json.loads(json.dumps(OPROJ))
    oproj["adapt"]["ops"][0]["with"] = "a.premix"
    with pytest.raises(PlanError, match=r"interventions\.oproj\.adapt\.ops\[0\]\.with: 'a\.premix' names step 'a''s "
                                        r"entry 'premix', which is also this bundle's own entry; name the bundle's "
                                        r"own entry bare \('premix'\)"):
        cross_site(root, {"a": {"kind": "forward", "input": "train.input", "interventions": "oproj"}}, ["a.premix"],
                   interventions={"oproj": oproj})


# -- refusals -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["fit"]["steps"]["adapted"]["interventions"]["adapt"]["ops"].__setitem__(
        0, {"op": "encode", "featurizer": "fit.lora"}),
     r"ops\[0\]\.featurizer: 'fit\.lora' is a 'lora' featurizer, which has no encode: it serves apply$"),
    (lambda d: d["featurizers"]["lora"].pop("alpha"),
     r"featurizers\.lora: a 'lora' featurizer drawn without a bundle needs 'alpha'; or load one with 'file_path'"),
    (lambda d: d["steps"]["fit"]["steps"]["adapted"]["interventions"]["adapt"].update(ops=[
        {"op": "decode", "featurizer": "fit.lora"}]),
     r"ops\[0\]: decode of 'fit\.lora' closes no encode: none is open in this entry here; put the encode of "
     r"'fit\.lora' before it"),
    (lambda d: d["steps"]["fit"]["steps"]["adapted"]["interventions"]["adapt"]["ops"][0].update({"with": 0.5}),
     r"adapt\.ops\[0\]\.with: 0\.5 is not a name; 'with' names a value read at the positions the op acts on"),
])
def test_what_an_adapter_cannot_be_is_refused_at_compile(change, message):
    with pytest.raises(PlanError, match=message):
        edited("lora_train", change).compile()


def test_a_generations_writes_take_no_operand_from_its_own_reads(corpus):
    with open(os.path.join(corpus, "documents", "lora_apply.json")) as f:
        doc = json.load(f)
    doc["steps"]["answer"]["interventions"] = {
        "h": {"site": {"component": "layer_input", "layers": 0}, "pos": -1},
        "again": {"site": "target", "pos": -1, "ops": [{"op": "apply", "featurizer": "lora", "with": "answer.h"}]}}
    with pytest.raises(PlanError, match=r"steps\.answer\.interventions\.again\.ops\[0\]\.with: 'answer\.h' is an entry "
                                        r"of this generation; an operand from an entry of its own step is served "
                                        r"in a forward, not yet inside a generate"):
        Plan(**doc, document=os.path.join(corpus, "documents", "lora_apply.json")).compile()


def at_layer_1(step: dict) -> None:
    step["interventions"]["premix"]["site"] = {"component": "attention_output", "layers": 1}


def one_head(step: dict) -> None:
    step["interventions"]["premix"]["site"]["select"] = {"num_heads": [0]}


def at_the_first_token(step: dict) -> None:
    step["interventions"]["premix"]["pos"] = 1         # position 0 is BOS


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("change, message", [
    (at_layer_1, r"patched\.adapt: operand 'patched\.premix' is read at attention_output layer 1, which the \w+ "
                 r"engine's forward reaches no earlier than this write's attention_output layer 0; an operand from the "
                 r"write's own forward is an earlier entry at this site, or at a site the forward reaches first"),
    (one_head, r"patched\.adapt: operand 'patched\.premix' is 4 wide per position, its axes after the positions "
               r"flattened; this write's value is 16"),
    (at_the_first_token, r"patched\.adapt: operand 'patched\.premix' is read at positions \[1\] of row 0 and this "
                         r"write lands at \[10\]; 'with' from the write's own forward is read where it lands"),
])
def test_a_same_forward_with_that_cannot_mean_anything_is_refused_before_any_forward(engines, tmp_path, kind,
                                                                                     change, message):
    """Read after the write (at a later layer), of another width per position, or at other positions than the
    write lands on: each refused by name before the first forward, the order by the engine's own measured
    forward order. (An earlier entry at the write's own site is an operand: `test_ops`' same-site test.)"""
    root = corpus_copy(tmp_path)
    step = adapted("patched", "oproj", -1)
    change(step)
    plan = cross_site(root, {"patched": step}, ["patched.out"])
    engine = engines(kind, plan.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=message):
            engine.execute(plan)
    finally:
        del engine.forward
    assert not calls


def test_each_engine_ranks_the_premix_before_the_attention_output(engines):
    for kind in ("nnterp", "hooks"):
        engine = engines(kind, lora_train().model)
        ranks = [engine.rank(c, layer) for c, layer in (("token_embeddings", None), ("attention_premix", 0),
                                                         ("attention_output", 0), ("layer_output", 0),
                                                         ("attention_premix", 1), ("lm_head_output", None))]
        assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks), kind


def test_a_live_operand_the_engine_had_not_read_is_named():
    from clab.steps import batches
    with pytest.raises(ValueError, match=r"operand 'premix@0' is a read of this forward the engine had not made"):
        batches.live({}, {"live": "premix@0"})


# -- loading: a document may state the kind's declaration fields, and each it states is checked ---------------


def loading(corpus: str, name: str, featurizer: str, fields: dict) -> Plan:
    with open(os.path.join(corpus, "documents", f"{name}.json")) as f:
        doc = json.load(f)
    doc["featurizers"][featurizer].update(fields)
    return Plan(**doc, document=os.path.join(corpus, "documents", f"{name}.json")).compile()


@pytest.mark.parametrize("name, featurizer, fields", [
    ("lora_apply", "lora", {}), ("lora_apply", "lora", {"rank": 4, "alpha": 8}), ("lora_apply", "lora", {"alpha": 8.0}),
    ("das_apply", "rot", {"k": 8}), ("pca_control", "rot", {}),
])
def test_a_loading_document_may_state_its_declaration_fields(corpus, name, featurizer, fields):
    assert loading(corpus, name, featurizer, fields).compiled


@pytest.mark.parametrize("name, featurizer, fields, message", [
    ("lora_apply", "lora", {"rank": 2}, r"header 'rank' is '4'; this document expects '2'"),
    ("lora_apply", "lora", {"alpha": 16}, r"header 'alpha' is '8'; this document expects '16'"),
    ("das_apply", "rot", {"k": 4}, r"header 'k' is '8'; this document expects '4'"),
    ("sae_feature_ablation", "dictionary", {"latents": 32}, r"header 'latents' is None; this document expects '32'"),
])
def test_a_stated_field_the_header_disagrees_with_is_refused_naming_it(corpus, name, featurizer, fields, message):
    with pytest.raises(PlanError, match=message):
        loading(corpus, name, featurizer, fields)
