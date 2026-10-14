"""The `featurizer` op: each kind's modes against a few lines of torch, the layout each mode produces, and the
refusals of a mode a kind lacks, a mode no kind has and an argument a mode does not take."""

import pytest
import torch

from clab import Plan, featurizers
from clab.cli import describe
from clab.plan import PlanError, dims_of, points
from conftest import BATCH_SIZE, edited, same

G = torch.Generator().manual_seed(3)
H = torch.randn(3, 2, 16, generator=G)
BASIS = torch.linalg.qr(torch.randn(16, 4, generator=G))[0]
MEAN = torch.randn(16, generator=G)
SAE = {"W_enc": torch.randn(16, 32, generator=G), "b_enc": torch.randn(32, generator=G),
       "W_dec": torch.randn(32, 16, generator=G), "b_dec": torch.randn(16, generator=G)}


def made(kind: str) -> dict:
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
        torch.testing.assert_close(featurizers.call(made(kind), "apply", H, {}), value, atol=1e-5, rtol=0)
    gate = made("gate")
    gate["theta"] = torch.linspace(-1, 1, 16)
    assert torch.equal(featurizers.call(gate, "apply", H, {}), H * (gate["theta"] > 0).float())
    gate["training"] = True
    assert torch.equal(featurizers.call(gate, "apply", H, {}), H * torch.sigmoid(gate["theta"]))


def test_encode_is_the_coordinates_on_the_kinds_axis():
    for kind in ("subspace", "pca", "sae", "gate"):
        assert torch.equal(featurizers.call(made(kind), "encode", H, {}), featurizers.encode(made(kind), H))
        assert featurizers.layout(kind, "encode", ["rows", "pos", "hidden_size"]) == ["rows", "pos", "features"]
        assert featurizers.layout(kind, "apply", ["rows", "pos", "hidden_size"]) == ["rows", "pos", "hidden_size"]
    assert featurizers.layout("linear", "encode", ["rows", "pos", "hidden_size"]) == ["rows", "pos", "classes"]


def test_a_mode_the_kind_lacks_is_refused_by_name():
    with pytest.raises(ValueError, match=r"kind 'linear' has no mode 'apply'; its modes: encode"):
        featurizers.mode("linear", "apply")
    with pytest.raises(ValueError, match=r"kind 'sae' has no mode 'decode'; its modes: encode, apply"):
        featurizers.mode("sae", "decode")


def reconstructed(mode: str = "apply", **more) -> Plan:
    """sae_feature_ablation with a read of the dictionary's reconstruction of layer 0's output at the last
    position, beside the plain read."""
    def change(d):
        d["steps"]["base"]["reads"]["recon"] = {"site": "target", "pos": -1, "ops": [
            {"op": "featurizer", "featurizer": "dictionary", "mode": mode, **more}]}
        d["steps"]["base"]["reads"]["plain"] = {"site": "target", "pos": -1}
        d["saves"] = ["base.latents", "base.recon", "base.plain"]
    return edited("sae_feature_ablation", change).compile()


def test_a_read_through_apply_keeps_the_sites_layout_on_both_engines(engines):
    plan = reconstructed()
    assert dims_of(plan, "base.recon") == ["rows", "pos", "hidden_size"]
    assert dims_of(plan, "base.latents") == ["rows", "pos", "features"]
    got = [engines(kind, plan.model).execute(reconstructed(), batch_size=BATCH_SIZE).results
           for kind in ("nnterp", "hooks")]
    assert same(*got)
    f = featurizers.make("sae", {"file_path": "x"}, 16, points(plan.featurizers["dictionary"])[0].bundle, None, None)
    torch.testing.assert_close(got[0]["base.recon"], featurizers.call(f, "apply", got[0]["base.plain"], {}),
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize("make, message", [
    (lambda: reconstructed("decode"), r"steps\.base\.reads\.recon\.ops\[0\]: featurizer 'dictionary': kind 'sae' has "
                                      r"no mode 'decode'; its modes: encode, apply"),
    (lambda: reconstructed(scale=2.0), r"featurizer 'dictionary': 'sae_apply' takes no 'scale'; its arguments: "
                                       r"none"),
])
def test_a_mode_or_argument_the_kind_lacks_is_refused_at_compile(make, message):
    with pytest.raises(PlanError, match=message):
        make()


def test_a_probe_has_no_apply_and_an_encode_is_no_write():
    def probe_apply(d):
        d["featurizers"]["probe"] = {"kind": "linear", "classes": "prompts.label"}
        d["steps"]["base"]["reads"]["latents"]["ops"] = [{"op": "featurizer", "featurizer": "probe", "mode": "apply"}]
    with pytest.raises(PlanError, match=r"featurizer 'probe': kind 'linear' has no mode 'apply'; its modes: encode"):
        edited("sae_feature_ablation", probe_apply).compile()

    def encode_in_a_write(d):
        d["steps"]["zeroed"]["writes"]["ablate"]["ops"] = [
            {"op": "featurizer", "featurizer": "dictionary", "mode": "encode"}]
    with pytest.raises(PlanError, match=r"'featurizer' turns \['rows', 'pos', 'hidden_size'\] into \['rows', 'pos', "
                                        r"'features'\]; a write's ops keep the value's layout"):
        edited("sae_feature_ablation", encode_in_a_write).compile()


def test_describe_shows_the_op_and_each_kinds_modes():
    text = describe()
    assert "featurizer(featurizer: Featurizer, mode: str, ...)  [its mode's (see Featurizer kinds)]" in text
    assert "modes: encode gives its coordinates, the last axis becoming classes\n" in text
    assert "encode(" not in text.replace("featurizer(", "")
