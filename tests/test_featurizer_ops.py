"""`encode`, `decode` and `apply`: each kind's ops against a few lines of torch, the layout each produces, the
bracket an `encode` and its `decode` make (the identity to the bit, nesting, a `select`), and the refusals of an
op a kind does not serve and an argument its `apply` does not take."""

import os
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from clab import Plan
from clab.docs import describe
from clab.plan import PlanError, dims_of
from clab.sweeps import points
from clab import batches, featurizers, ops
from clab.featurizers.gate import Gate
from conftest import BATCH_SIZE, bundle_header, edited, same
from test_featurizers import every_kind
from test_fit import TINY, corpus_copy

G = torch.Generator().manual_seed(3)
H = torch.randn(3, 2, 16, generator=G)
BASIS = torch.linalg.qr(torch.randn(16, 4, generator=G))[0]
MEAN = torch.randn(16, generator=G)
SAE = {"W_enc": torch.randn(16, 32, generator=G), "b_enc": torch.randn(32, generator=G),
       "W_dec": torch.randn(32, 16, generator=G), "b_dec": torch.randn(16, generator=G)}
DIMS = ["rows", "pos", "hidden_size"]
SITE = {"component": "layer_output", "layers": 0}
HEAD = {"component": "lm_head_output"}


def made(kind: str) -> Any:
    fields = {"subspace": {"k": 4}, "pca": {"k": 4, "file_path": "x"}, "sae": {"file_path": "x"}}.get(kind, {})
    bundle = {"subspace": {"basis": BASIS}, "pca": {"basis": BASIS, "mean": MEAN}, "sae": SAE}.get(kind)
    return featurizers.make(kind, fields, 16, bundle, 0, None)


def test_apply_is_each_kinds_version_of_the_value():
    """subspace: the projection onto the basis; pca: the same through the mean; sae: the decoding of the
    latents, without the error term; gate: the value times its mask, hard when scored and soft while trained."""
    projection = BASIS @ BASIS.T
    expected = {"subspace": H @ projection, "pca": MEAN + (H - MEAN) @ projection,
                "sae": torch.relu((H - SAE["b_dec"]) @ SAE["W_enc"] + SAE["b_enc"]) @ SAE["W_dec"] + SAE["b_dec"]}
    for kind, value in expected.items():
        torch.testing.assert_close(ops.call("apply", H, {"featurizer": made(kind)}), value, atol=1e-5, rtol=0)
    gate = Gate(16)
    gate.theta = torch.linspace(-1, 1, 16)
    assert torch.equal(gate.call("apply", H, {}), H * (gate.theta > 0).float())
    gate.training = True
    assert torch.equal(gate.call("apply", H, {}), H * torch.sigmoid(gate.theta))


def test_encode_is_the_coordinates_on_the_kinds_axis_and_select_keeps_some():
    for kind in ("subspace", "pca", "sae", "gate"):
        f = made(kind)
        assert torch.equal(ops.call("encode", H, {"featurizer": f}), f.encode(H))
        assert torch.equal(ops.call("encode", H, {"featurizer": f, "select": {"features": "1:3"}}), f.encode(H)[..., 1:3])
        assert ops.out_dims("encode", DIMS, {}, kind) == ["rows", "pos", "features"]
    assert ops.out_dims("encode", DIMS, {}, "linear") == ["rows", "pos", "classes"]
    assert [featurizers.KINDS[k].ops() for k in featurizers.KINDS] == [
        ["encode", "decode", "apply"]] * 4 + [["encode"], ["apply"]]


def test_an_op_the_kind_does_not_serve_is_refused_by_name():
    with pytest.raises(ValueError, match=r"a 'linear' featurizer has no apply; it serves encode$"):
        featurizers.KINDS["linear"].method("apply")
    with pytest.raises(ValueError, match=r"a 'lora' featurizer has no decode; it serves apply$"):
        featurizers.KINDS["lora"].method("decode")


def bracket(f, features=None, between=()) -> list[dict]:
    """An encode of `f` holding the bound ops `between`, which its decode closes over, as `batches.edit` binds
    them."""
    return [{"op": "encode", "args": {"featurizer": f}, "dims": DIMS, "inner": [
        {**o, "dims": ["rows", "pos", f.axis]} for o in between], **({} if features is None else {"features": features})}]


@pytest.mark.parametrize("kind", ["subspace", "gate", "pca", "sae", "jumprelu", "topk"])
def test_encode_then_decode_is_the_identity_to_the_bit_on_every_kind(kind):
    """The delta rule is what decode is: `[encode x, decode x]` writes back `h` itself, on every coordinate or
    the ones a `select` keeps, and so does an edit that changes no coordinate (a swap with themselves)."""
    f = every_kind()[kind]
    record = {"positions": torch.tensor([[0, 1]] * 3), "select": {}}
    for features in (None, torch.tensor([1, 2])):
        for between in ((), [{"op": "scale", "args": {"factor": 1.0}}]):
            out, wrote = batches.visit(H, "layer_output", [{**record, "ops": bracket(f, features, between)}], {})
            assert wrote and torch.equal(out, H), (kind, features, between)


def test_a_decode_scatters_back_only_the_coordinates_its_encode_kept():
    f = made("subspace")
    record = {"positions": torch.tensor([[0, 1]] * 3), "select": {}}
    out, _ = batches.visit(H, "layer_output", [{**record, "ops": bracket(f, torch.tensor([2]), [
        {"op": "swap", "args": {"with": 0.0}}])}], {})
    z, kept = f.encode(H), f.encode(out)
    torch.testing.assert_close(kept[..., 2], torch.zeros(3, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(kept[..., [0, 1, 3]], z[..., [0, 1, 3]], atol=1e-6, rtol=0)


def bracketed(root: str, featurizer: dict, file: dict | None = None) -> Plan:
    """The tiny model's prompts read at layer 0 and at the head, clean and through `[encode f, decode f]` at
    layer 0's last position; `file` is a bundle `f` loads, written under the corpus copy at `root`."""
    if file is not None:
        save_file(file["tensors"], os.path.join(root, "documents", "artifacts", "f.safetensors"),
                  metadata=bundle_header(**file["header"]))
        featurizer = {**featurizer, "file_path": "documents/artifacts/f.safetensors"}
    entry = {"site": SITE, "pos": -1, "ops": [{"op": "encode", "featurizer": "f"}, {"op": "decode", "featurizer": "f"}]}
    doc = {"model": TINY, "data": {"train": {"path": "weekdays/data", "split": "train"}}, "featurizers": {"f": featurizer},
           "steps": {"clean": {"kind": "forward", "input": "train.input", "interventions": {
                         "h": {"site": SITE, "pos": -1}, "logits": {"site": HEAD, "pos": -1}}},
                     "through": {"kind": "forward", "input": "train.input", "interventions": {
                         "same": entry, "h": {"site": SITE, "pos": -1}, "logits": {"site": HEAD, "pos": -1}}}},
           "saves": ["clean.h", "clean.logits", "through.same", "through.h", "through.logits"]}
    return Plan(**doc, document=os.path.join(root, "documents", "bracketed.json")).compile()


@pytest.mark.parametrize("kind, featurizer, file", [
    ("subspace", {"kind": "subspace", "k": 4, "seed": 2}, None),
    ("gate", {"kind": "gate"}, None),
    ("sae", {"kind": "sae", "latents": 32, "seed": 1}, None),
    ("pca", {"kind": "pca"}, {"tensors": {"basis": BASIS.contiguous(), "mean": MEAN},
                              "header": {"kind": "pca", "k": "4"}}),
])
def test_an_entry_that_encodes_and_decodes_changes_nothing_on_the_model(engines, tmp_path, kind, featurizer, file):
    """Run on the tiny model, an entry `[encode f, decode f]` writes back what was there: the head's logits and
    the value at the site are the clean run's to the bit, and so is the entry's own value, on both engines."""
    plan = bracketed(corpus_copy(tmp_path), featurizer, file)
    got = [engines(k, plan.model).execute(plan, batch_size=BATCH_SIZE).results for k in ("nnterp", "hooks")]
    assert same(*got)
    values = got[0]
    assert torch.equal(values["through.same"], values["clean.h"]) and torch.equal(values["through.h"], values["clean.h"])
    assert torch.equal(values["through.logits"], values["clean.logits"])


def reconstructed(more: dict | None = None) -> Plan:
    """sae_feature_ablation's base step with, at layer 0's last position, a read of the value, then `recon`
    (the dictionary's `apply`, an edit, so it writes back), then a read of the site after it."""
    def change(d):
        entries = d["steps"]["base"]["interventions"]
        entries["plain"] = {"site": "target", "pos": -1}
        entries["recon"] = {"site": "target", "pos": -1, "ops": [{"op": "apply", "featurizer": "dictionary",
                                                                   **(more or {})}]}
        entries["after"] = {"site": "target", "pos": -1}
        d["saves"] = ["base.latents", "base.plain", "base.recon", "base.after"]
    return edited("sae_feature_ablation", change).compile()


def test_apply_writes_the_featurizers_version_back_on_both_engines(engines):
    plan = reconstructed()
    assert dims_of(plan, "base.recon") == ["rows", "pos", "hidden_size"]
    assert dims_of(plan, "base.latents") == ["rows", "pos", "features"]
    got = [engines(kind, plan.model).execute(reconstructed(), batch_size=BATCH_SIZE).results
           for kind in ("nnterp", "hooks")]
    assert same(*got)
    f = featurizers.make("sae", {"file_path": "x"}, 16, points(plan.featurizers["dictionary"])[0].bundle, None, None)
    torch.testing.assert_close(got[0]["base.recon"], f.call("apply", got[0]["base.plain"], {}), atol=1e-6, rtol=0)
    assert torch.equal(got[0]["base.after"], got[0]["base.recon"])


def test_an_argument_the_kinds_apply_does_not_take_is_refused_at_compile():
    with pytest.raises(PlanError, match=r"steps\.base\.interventions\.recon\.ops\[0\]: featurizer 'dictionary': an sae's "
                                        r"`apply` takes no 'scale'; its arguments: none"):
        reconstructed({"scale": 2.0})


def test_describe_shows_the_ops_and_each_kinds_ops():
    text = describe()
    assert "encode(featurizer: Featurizer, select: {features: selector} = null)  [the last axis becomes" in text
    assert "decode(featurizer: Featurizer)  [the layout its `encode` had]" in text
    assert "detokenize(tokens: bool = false)  [needs new; removes new]" in text
    assert "apply(featurizer: Featurizer, ...)  [keeps the layout]" in text
    assert "ops: encode gives its coordinates, the last axis becoming classes\n" in text
    assert "ops: apply(with: Value or null = null) gives its version of the value" in text


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_write_after_one_on_coordinates_sees_the_sites_original_value(engines, kind):
    """Two writes at one site: a swap of a subspace's coordinates, then `renormalize`, whose reference is the
    value before the site's writes. It restores the norm the swap changed; it must not see the first write's
    coordinates (a `[rows, pos, k]` tensor) as that value."""
    doc = {"model": TINY, "data": {"train": {"path": "weekdays/data", "split": "train"}},
           "featurizers": {"rot": {"kind": "subspace", "k": 4, "seed": 0}},
           "steps": {"clean": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": SITE, "pos": -1}}},
                     "patched": {"kind": "forward", "input": "train.input", "interventions": {
                         "zero": {"site": SITE, "pos": -1, "ops": [{"op": "encode", "featurizer": "rot"},
                                                                   {"op": "swap", "with": 0.0},
                                                                   {"op": "decode", "featurizer": "rot"}]},
                         "renorm": {"site": SITE, "pos": -1, "ops": [{"op": "renormalize"}]}}}},
           "saves": ["clean.h", "patched.renorm"]}
    plan = Plan(**doc).compile()
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert not torch.allclose(values["patched.renorm"], values["clean.h"])
    torch.testing.assert_close(values["patched.renorm"].norm(dim=-1), values["clean.h"].norm(dim=-1), atol=1e-6, rtol=0)


# -- a chain: a gate over PCA coordinates (MIB's "+PCA") -----------------------------------------------------------


def chained(root: str, steps: dict, saves: list[str], mask: str = "mask") -> Plan:
    """A pca of layer 0's output (a random orthonormal basis and mean, as a bundle) and a gate `mask` over its
    4 coordinates: `cf` reads the counterfactual's coordinates through both, `patch` swaps them in and decodes
    back out, innermost first."""
    save_file({"basis": BASIS.contiguous(), "mean": MEAN}, os.path.join(root, "documents", "artifacts", "pc.safetensors"),
              metadata=bundle_header(kind="pca", k="4", document="pca_harvest"))
    enc = [{"op": "encode", "featurizer": "pc"}, {"op": "encode", "featurizer": mask}]
    dec = [{"op": "decode", "featurizer": mask}, {"op": "decode", "featurizer": "pc"}]
    doc = {"model": TINY, "data": {"train": {"path": "weekdays/data", "split": "train"}},
           "featurizers": {"pc": {"kind": "pca", "file_path": "documents/artifacts/pc.safetensors"},
                           "mask": {"kind": "gate"}},
           "interventions": {
               "cf": {"v_cf": {"site": SITE, "pos": -1, "ops": enc}, "v": {"site": SITE, "pos": -1}},
               "patch": {"patch": {"site": SITE, "pos": -1, "ops": [*enc, {"op": "swap", "with": "counterfactual.v_cf"},
                                                                    *dec]},
                         "logits": {"site": HEAD, "pos": -1}}},
           "steps": steps, "saves": saves}
    return Plan(**doc, document=os.path.join(root, "documents", "chained.json")).compile()


def test_a_gate_over_pca_coordinates_patches_through_both(engines, tmp_path):
    """The chain on the tiny model: the counterfactual's coordinates are the gate's of the pca's (4 of them, the
    gate as wide as the pca's k), and the patch writes back `h + (z_cf - z) Bᵀ` where the gate passes every
    coordinate (an untrained gate's hard mask is all ones): the base read through the pca alone, swapped and
    decoded, on both engines to the bit."""
    root = corpus_copy(tmp_path)
    steps = {"clean": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": SITE, "pos": -1}}},
             "counterfactual": {"kind": "forward", "input": "train.counterfactual_inputs[0]", "interventions": "cf"},
             "patched": {"kind": "forward", "input": "train.input", "interventions": "patch"}}
    plan = chained(root, steps, ["clean.h", "counterfactual.v", "counterfactual.v_cf", "patched.patch"])
    assert dims_of(plan, "counterfactual.v_cf") == ["rows", "pos", "features"]
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")]
    assert same(got[0].results, got[1].results)
    values = got[0].results
    pc: Any = featurizers.make("pca", {"file_path": "x"}, 16, {"basis": BASIS, "mean": MEAN}, None, None)
    torch.testing.assert_close(values["counterfactual.v_cf"], pc.encode(values["counterfactual.v"]), atol=1e-6, rtol=0)
    expected = values["clean.h"] + (pc.encode(values["counterfactual.v"]) - pc.encode(values["clean.h"])) @ BASIS.T
    torch.testing.assert_close(values["patched.patch"], expected, atol=1e-5, rtol=0)


def test_a_gate_over_pca_coordinates_trains_on_both_engines(engines, tmp_path):
    """MIB's "+PCA" row as a fit: the gate trains over the pca's 4 coordinates (its bundle 4 wide, its header
    the site's), the same on both engines."""
    root = corpus_copy(tmp_path)
    fit = {"kind": "fit", "train": ["mask"], "objective": {"ce": 1.0, "mask.mask": 0.01}, "epochs": 3, "batch_size": 2,
           "seed": 0, "optimizer": {"name": "adam", "lr": 0.1}, "eval": {"train": {}},
           "steps": {"counterfactual": {"kind": "forward", "input": "train.counterfactual_inputs[0]",
                                        "interventions": "cf"},
                     "patched": {"kind": "forward", "input": "train.input", "interventions": "patch"},
                     "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"}}}
    plan = chained(root, {"fit": fit}, ["fit.mask", "fit.history.ce"], mask="fit.mask")
    got = [engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")]
    assert same(got[0].results, got[1].results)
    assert got[0].results["fit.mask"]["theta"].shape == (4,)
    assert not torch.equal(got[0].results["fit.mask"]["theta"], torch.ones(4))
    got[0].dump(str(tmp_path / "out"))
    with safe_open(str(tmp_path / "out" / "fit.mask.safetensors"), "pt") as f:
        header = f.metadata()
    assert (header["width"], header["component"], header["layer"]) == ("4", "layer_output", "0")
