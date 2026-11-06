"""Featurizer math is device- and autocast-safe, and a published SAE (a safetensors file with no nnspec header, as
SAELens and Gemma Scope publish one) loads: its tensor names and shapes checked, its activation the document's,
its model and site unchecked; an nnspec-written sae bundle records its activation and round-trips."""

import json
import os

import pytest
import torch
from conftest import BATCH_SIZE, DOCUMENTS, ROOT, TRIP, bundle_header, edited, same, stand_in
from safetensors import safe_open
from safetensors.torch import save_file

from nnspec import Plan, featurizers
from nnspec.featurizers.gate import Gate
from nnspec.featurizers.sae import Sae
from nnspec.format.sweeps import points
from nnspec.plan import PlanError

G = torch.Generator().manual_seed(5)
H = torch.randn(3, 2, 16, generator=G)
BASIS = torch.linalg.qr(torch.randn(16, 4, generator=G))[0]
TENSORS = {"W_enc": torch.randn(16, 32, generator=G), "b_enc": torch.randn(32, generator=G),
           "W_dec": torch.randn(32, 16, generator=G), "b_dec": torch.randn(16, generator=G)}
THRESHOLD = torch.rand(32, generator=G)
# What a document loading a published sae states, since the file does not say it: how it encodes, and the site
# it was trained at.
WHERE = {"site": "target"}
STATED = {"activation": "relu", "centered": True, **WHERE}


def every_kind() -> dict[str, featurizers.Kind]:
    """One instance of each kind, its tensors on the CPU in float32."""
    return {"subspace": featurizers.make("subspace", {"k": 4}, 16, None, 0, None),
            "gate": featurizers.make("gate", {}, 16, {"theta": torch.linspace(-1, 1, 16)}, None, None),
            "pca": featurizers.make("pca", {"file_path": "x"}, 16, {"basis": BASIS, "mean": TENSORS["b_dec"]}, None, None),
            "sae": featurizers.make("sae", {"file_path": "x"}, 16, TENSORS, None, None),
            "jumprelu": Sae(16, file_path="x", activation="jumprelu", centered=False,
                            bundle={**TENSORS, "threshold": THRESHOLD}),
            "topk": Sae(16, file_path="x", activation="topk", k=3, bundle=TENSORS),
            "linear": featurizers.make("linear", {"classes": "d.c"}, 16,
                                       {"W": torch.randn(16, 3, generator=G), "b": torch.randn(3, generator=G)}, None,
                                       ["a", "b", "c"]),
            "lora": featurizers.make("lora", {"file_path": "x"}, 16, {"A": torch.randn(16, 4, generator=G),
                                                                        "B": torch.randn(4, 16, generator=G)},
                                     None, None, {"rank": "4", "alpha": "8"}),
            "affine": featurizers.make("affine", {"file_path": "x"}, 16, {"W": torch.randn(16, 16, generator=G),
                                                                            "b": torch.randn(16, generator=G)},
                                       None, None),
            "constant": featurizers.make("constant", {"file_path": "x"}, 16, {"vector": torch.linspace(-1, 1, 16)},
                                         None, None)}


def outputs(f: featurizers.Kind, h: torch.Tensor) -> list[torch.Tensor]:
    """Everything a featurizer computes on a value: its encode and apply, and an edit of its coordinates decoded."""
    out = [f.call(op, h, {}) for op in f.ops() if op != "decode"]
    if "decode" in f.ops():
        out.append(f.within(h, lambda z: z * 2, torch.tensor([1, 2])))
    return out


def mixed(z: torch.Tensor) -> torch.Tensor:
    """An edit op that does a matmul: every coordinate replaced by a third of their sum."""
    return z @ torch.ones(z.shape[-1], z.shape[-1], device=z.device) / 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_featurizer_math_runs_on_the_values_device():
    """Each kind's tensors made on the CPU meet a value on the GPU: its ops and an edit of its coordinates run there,
    the tensors themselves stay where they were made, and the numbers are the CPU's (to float32 rounding)."""
    h = H.to("cuda")
    for name, f in every_kind().items():
        on_cpu = outputs(f, H)
        on_gpu = outputs(f, h)
        assert all(t.device == h.device for t in on_gpu), name
        for a, b in zip(on_gpu, on_cpu):
            torch.testing.assert_close(a.cpu(), b, atol=1e-5, rtol=1e-5)
        assert all(t.device.type == "cpu" for t in f.tensors().values()), name
    sae = every_kind()["sae"]
    sae.seen = [(h, sae.call("encode", h, {}))]
    assert sae.scalar("reconstruction").device == h.device


def test_featurizer_math_inside_autocast_is_the_same_bits_as_outside():
    """A model run under autocast (a bfloat16 server) leaves a featurizer's float32 math in float32: every op, an
    edit of its coordinates and an sae's reconstruction give the same bits inside an enabled autocast region as
    outside it."""
    for name, f in every_kind().items():
        outside = outputs(f, H)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.mm(H[0], BASIS).dtype == torch.bfloat16       # the region is on
            inside = outputs(f, H)
        assert same(inside, outside), name
    sae = every_kind()["sae"]
    sae.seen = [(H, sae.call("encode", H, {}))]
    outside = sae.scalar("reconstruction")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(sae.scalar("reconstruction"), outside)


def test_an_edit_in_a_featurizers_space_runs_outside_autocast_too():
    """`within` (an encode and the decode that closes it) turns autocast off for the edit between as well: a
    matmul on a gate's coordinates (whose own math has none) gives the bits inside an enabled autocast region
    that it gives outside."""
    gate = Gate(16, bundle={"theta": torch.linspace(-1, 1, 16)})
    outside = gate.within(H, mixed, None), gate.within(H, mixed, torch.tensor([1, 2, 5]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert mixed(H).dtype == torch.bfloat16                         # the region is on
        inside = gate.within(H, mixed, None), gate.within(H, mixed, torch.tensor([1, 2, 5]))
    assert same(inside, outside)


def test_the_published_activations_are_their_formulas():
    """relu (the default) subtracts `b_dec` first; topk keeps the relu of the `k` largest pre-activations of each
    vector; jumprelu is the relu of a pre-activation above its threshold, so a negative pre-activation above a
    negative threshold is zero, not itself."""
    kinds = every_kind()
    pre = (H - TENSORS["b_dec"]) @ TENSORS["W_enc"] + TENSORS["b_enc"]
    assert torch.equal(kinds["sae"].call("encode", H, {}), torch.relu(pre))
    jump = Sae(1, file_path="x", activation="jumprelu", centered=False,
               bundle={"W_enc": torch.zeros(1, 3), "b_enc": torch.tensor([-0.5, 0.5, -2.0]), "W_dec": torch.zeros(3, 1),
                       "b_dec": torch.zeros(1), "threshold": torch.tensor([-1.0, 0.0, -1.0])})
    assert jump.encode(torch.zeros(1, 1)).tolist() == [[0.0, 0.5, 0.0]]
    topk = kinds["topk"].call("encode", H, {})
    assert ((topk != 0).sum(-1) <= 3).all()
    kept = pre.topk(3, dim=-1)
    assert torch.equal(topk.gather(-1, kept.indices), torch.relu(kept.values))
    for name in ("jumprelu", "topk"):
        f = kinds[name]
        assert torch.equal(f.call("apply", H, {}), f.call("encode", H, {}) @ TENSORS["W_dec"] + TENSORS["b_dec"])
        assert torch.equal(f.within(H, lambda z: z, None), H)


def published(tmp_path, tensors: dict, featurizer: dict, site: dict | None = None,
              metadata: dict | None = None) -> Plan:
    """sae_published_jumprelu with its dictionary loaded from `tensors`, written with `metadata` (none by
    default), declared as `featurizer`, at `site`."""
    os.makedirs(tmp_path / "documents" / "artifacts", exist_ok=True)
    if not os.path.exists(tmp_path / "data"):
        os.symlink(os.path.join(ROOT, "data"), tmp_path / "data")
    save_file(tensors, str(tmp_path / "documents" / "artifacts" / "published.safetensors"), metadata=metadata)
    with open(os.path.join(DOCUMENTS, "sae_published_jumprelu.json")) as f:
        doc = json.load(f)
    doc["featurizers"]["dictionary"] = {"kind": "sae", "file_path": "documents/artifacts/published.safetensors",
                                        **featurizer}
    doc["sites"]["target"] = site or doc["sites"]["target"]
    return Plan(**doc, document=str(tmp_path / "documents" / "sae_published_jumprelu.json"))


@pytest.mark.parametrize("tensors, featurizer, message", [
    ({**TENSORS, "W_gate": TENSORS["W_enc"].clone()}, STATED,
     r"published\.safetensors has no nnspec header and holds W_dec, W_enc, W_gate, b_dec, b_enc; a published sae "
     r"holds W_enc, W_dec, b_enc, b_dec, threshold \(threshold optional\)"),
    ({**TENSORS, "W_enc": torch.zeros(16)}, STATED,
     r"published\.safetensors: W_enc is \[16\]; it is \[width, latents\]"),
    ({**TENSORS, "b_enc": torch.zeros(31)}, STATED,
     r"published\.safetensors: b_enc is \[31\]; with W_enc \[16, 32\] it is \[latents 32\]"),
    ({**TENSORS, "W_dec": torch.zeros(32, 15)}, STATED,
     r"W_dec is \[32, 15\]; with W_enc \[16, 32\] it is \[latents 32, width 16\]"),
    (TENSORS, {"activation": "jumprelu", "centered": False, **WHERE},
     r"published\.safetensors holds no threshold; a 'jumprelu' sae needs one"),
    ({**TENSORS, "threshold": THRESHOLD}, STATED,
     r"published\.safetensors holds a JumpReLU threshold; its activation is 'jumprelu', not 'relu'"),
    ({**TENSORS, "threshold": THRESHOLD}, {"activation": "topk", "k": 4, "centered": True, **WHERE},
     r"holds a JumpReLU threshold; its activation is 'jumprelu', not 'topk'"),
    (TENSORS, {**STATED, "latents": 64}, r"published\.safetensors has 32 latents; this document says 64"),
    ({**TENSORS, "threshold": THRESHOLD}, {},
     r"published\.safetensors is a published sae, which says neither how it encodes nor where it was trained; "
     r"state 'activation', 'centered', 'site'"),
    (TENSORS, {"activation": "relu", **WHERE}, r"says neither how it encodes nor where it was trained; state "
                                               r"'centered', as"),
    (TENSORS, {"activation": "relu", "centered": True}, r"state 'site', as its release names it$"),
    (TENSORS, {"activation": "topk", "k": 33, "centered": True, **WHERE}, r"has 32 latents; 'topk' keeps k = 33 of "
                                                                          r"them"),
])
def test_a_published_file_with_other_tensors_or_shapes_is_refused_at_compile(tmp_path, tensors, featurizer, message):
    with pytest.raises(PlanError, match=r"featurizers\.dictionary\.file_path: .*" + message):
        published(tmp_path, tensors, featurizer).compile()


ROTATION = {"kind": "subspace", "k": "8", "parametrization": "cayley", "document": "das"}


@pytest.mark.parametrize("tensors, header, message", [
    ({"basis": torch.eye(16)[:, :8].contiguous()}, None,
     r"has no nnspec header; a subspace bundle is one nnspec wrote"),
    ({"rotation": torch.eye(16)[:, :8].contiguous()}, bundle_header(**ROTATION),
     r"holds rotation; a subspace bundle holds basis"),
])
def test_a_subspace_from_a_published_file_or_a_bundle_of_other_tensors_is_refused_at_compile(tmp_path, tensors,
                                                                                               header, message):
    """das_apply's rotation loaded from a file with no nnspec header (of the kinds, only an sae reads a published
    file), or from an nnspec bundle whose tensors are not a subspace's."""
    os.makedirs(tmp_path / "documents" / "artifacts")
    os.symlink(os.path.join(ROOT, "data"), tmp_path / "data")
    save_file(tensors, str(tmp_path / "documents" / "artifacts" / "fit.rot.safetensors"), metadata=header)
    with open(os.path.join(DOCUMENTS, "das_apply.json")) as f:
        doc = json.load(f)
    plan = Plan(**doc, document=str(tmp_path / "documents" / "das_apply.json"))
    with pytest.raises(PlanError, match=r"^featurizers\.rot\.file_path: documents/artifacts/fit\.rot\.safetensors "
                                        + message + "$"):
        plan.compile()


@pytest.mark.parametrize("kind, fields, kept", [
    ("subspace", {"k": 3}, None), ("gate", {}, None), ("gate", {"over": ["layers", "num_heads"]},
                                                       {"layers": [0, 1], "num_heads": [0, 2, 3]}),
    ("pca", {"file_path": "x", "k": 3}, None), ("sae", {"latents": 7}, None), ("linear", {"classes": "d.c"}, None),
    ("lora", {"rank": 3, "alpha": 6.0}, None), ("affine", {}, None), ("affine", {"over": ["layers"]}, {"layers": [2, 0]}),
    ("constant", {"init": 0}, None)])
def test_each_tensor_a_featurizer_holds_lies_on_the_axes_its_kind_lays_it_on(kind, fields, kept):
    """Every tensor a kind's bundle holds has a layout (`laid`), one axis per dimension of a drawn instance's tensor:
    `width` the site's width, the kind's coordinate axis as many as it has coordinates, `rank` a lora's, and over axes
    as many as the indices it keeps on each, a stacked kind's first and a gate's in place of `width`."""
    cls = featurizers.KINDS[kind]
    assert set(cls.saves) <= set(cls.laid)
    drawn = featurizers.make(kind, fields, 5, None, 0, ["a", "b"] if kind == "linear" else None, kept=kept)
    over = fields.get("over") or []
    sizes = {"width": 5, "features": 3 if kind in ("subspace", "pca") else 7, "classes": 2, "rank": 3,
             **{a: len((kept or {})[a]) for a in over}}
    for tensor in cls.held(None):
        axes = [a for axis in cls.laid[tensor] for a in (over if over and axis == "width" and not cls.stacked else [axis])]
        axes = [*(over if cls.stacked else []), *axes]
        assert list(getattr(drawn, tensor).shape) == [sizes[a] for a in axes], tensor


def test_an_unknown_kind_is_refused_naming_the_kinds():
    with pytest.raises(PlanError, match=r"^featurizers\.rot: unknown featurizer kind 'rotation'; valid: subspace, "
                                        r"gate, pca, sae, linear, lora, affine, constant$"):
        edited("das_apply", lambda d: d["featurizers"]["rot"].update(kind="rotation"))


@pytest.mark.parametrize("featurizer, message", [
    ({"latents": 32, "activation": "topk"}, r"activation 'topk' needs 'k', how many latents it keeps$"),
    ({"latents": 32, "k": 4}, r"'k' is how many latents activation 'topk' keeps; activation 'relu' takes none$"),
    ({"latents": 32, "activation": "jumprelu"}, r"a 'jumprelu' sae's threshold is a published file's; load one with "
                                                r"'file_path'"),
    ({"latents": 32, "activation": "topk", "k": 40}, r"'topk' keeps k = 40 of 32 latents"),
    ({"latents": 32, "activation": "gelu"}, r"activation: 'gelu' is not one of 'relu', 'jumprelu', 'topk'"),
])
def test_an_activation_and_its_k_are_checked_on_the_declaration(featurizer, message):
    with pytest.raises(PlanError, match=message):
        edited("sae_train", lambda d: d["featurizers"].update(dictionary={"kind": "sae", **featurizer})).compile()


def test_a_published_file_as_wide_as_another_site_is_refused_before_any_forward(engines, tmp_path):
    plan = published(tmp_path, {**TENSORS, "threshold": THRESHOLD}, {"activation": "jumprelu", "centered": False,
                                                                       **WHERE},
                     {"component": "mlp_activation", "layers": 0}).compile()
    with pytest.raises(ValueError, match=r"base\.latents: documents/artifacts/published\.safetensors has no nnspec "
                                         r"header and is 16 wide; this document uses 'dictionary' at mlp_activation, "
                                         r"whose width is \d+"):
        engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)


def test_a_published_file_has_no_header_and_an_nnspec_bundle_keeps_its_own():
    plan = Plan(os.path.join(DOCUMENTS, "sae_published_topk.json")).compile()
    assert points(plan.featurizers["dictionary"])[0].header is None
    nnspec_written = Plan(os.path.join(DOCUMENTS, "sae_feature_ablation.json")).compile()
    assert (points(nnspec_written.featurizers["dictionary"])[0].header or {})["format"] == "nnspec/1"


def test_a_published_dictionary_is_the_same_remote_local_and_through_the_stand_in_server(engines, monkeypatch):
    def plan():
        return Plan(os.path.join(DOCUMENTS, "sae_published_jumprelu.json")).compile()
    local = engines("nnterp", plan().model).execute(plan(), batch_size=BATCH_SIZE).results
    there = engines("ndif", plan().model, remote="local").execute(plan(), batch_size=BATCH_SIZE)
    assert same(there.results, local)
    came, trip = stand_in(engines, plan(), monkeypatch)
    assert trip == TRIP and same(came.results, local)


def topk_train(d):
    d["featurizers"]["dictionary"].update(activation="topk", k=4, centered=False)
    d["steps"]["fit"]["epochs"] = 2


def test_a_trained_topk_dictionary_records_its_activation_and_loads_back(engines, tmp_path):
    """sae_train with `activation: topk`: the bundle it writes records the activation, `k` and `centered`; a
    document that loads it states the same and encodes as the trained one did, and one that states another
    activation is refused naming the header key."""
    from test_fit import corpus_copy
    root = corpus_copy(tmp_path)
    engine = engines("hooks", edited("sae_train", topk_train).compile().model)
    trained = engine.execute(edited("sae_train", topk_train).compile(), batch_size=BATCH_SIZE)
    trained.dump(os.path.join(root, "documents", "artifacts"))
    with safe_open(os.path.join(root, "documents", "artifacts", "fit.dictionary.safetensors"), "pt") as f:
        header = f.metadata()
    assert {k: header[k] for k in ("kind", "latents", "activation", "k", "centered")} == \
        {"kind": "sae", "latents": "32", "activation": "topk", "k": "4", "centered": "false"}
    with open(os.path.join(root, "documents", "sae_train_ablation.json")) as f:
        doc = json.load(f)
    doc["featurizers"]["dictionary"].update(activation="topk", k=4, centered=False)
    plan = Plan(**doc, document=os.path.join(root, "documents", "sae_train_ablation.json")).compile()
    loaded = featurizers.make("sae", points(plan.featurizers["dictionary"])[0].model_extra or {}, 16,
                              points(plan.featurizers["dictionary"])[0].bundle, None, None)
    drawn = featurizers.make("sae", {"file_path": "x", "activation": "topk", "k": 4, "centered": False}, 16,
                             trained.results["fit.dictionary"], None, None)
    latents = loaded.call("encode", H, {})
    assert torch.equal(latents, drawn.call("encode", H, {})) and ((latents != 0).sum(-1) <= 4).all()
    doc["featurizers"]["dictionary"]["activation"] = "relu"
    doc["featurizers"]["dictionary"].pop("k")
    with pytest.raises(PlanError, match=r"fit\.dictionary\.safetensors: header 'activation' is 'topk'; this "
                                        r"document expects 'relu'"):
        Plan(**doc, document=os.path.join(root, "documents", "sae_train_ablation.json")).compile()


def test_an_saes_signature_prints_no_default_a_published_file_lacks():
    """`activation` and `centered` are `null` in the signature, since a published file has no default for them
    (its document states them, `Sae.published`); unstated, a drawn dictionary is relu and centered (`unstated`),
    which its bundle's header records, and an nnspec bundle loaded with neither stated encodes as its header says."""
    from nnspec.docs import describe
    from nnspec.featurizers import bundles
    from nnspec.plan import Model

    listed = describe("featurizers")
    assert 'activation: one of "relu", "jumprelu", "topk" or null = null' in listed and \
        "centered: bool or null = null" in listed
    model = Model(key="k", revision="r", dtype="float32")
    header = bundles.header("sae", {"latents": 32}, model, None)
    assert {k: header[k] for k in ("latents", "activation", "centered")} == \
        {"latents": "32", "activation": "relu", "centered": "true"} and "k" not in header
    unstated = featurizers.make("sae", {"file_path": "x"}, 16, TENSORS, None, None)
    assert isinstance(unstated, Sae) and (unstated.activation, unstated.centered, unstated.k) == ("relu", True, None)
    stated = Sae(16, file_path="x", activation="relu", centered=True, bundle=TENSORS)
    assert torch.equal(unstated.call("encode", H, {}), stated.call("encode", H, {}))


def test_an_nnspec_bundles_activation_and_centering_are_its_headers():
    """The committed fixture's header says relu and centered; a document that loads it states those or nothing,
    and one that states otherwise is refused naming the key."""
    with safe_open(os.path.join(DOCUMENTS, "artifacts", "sae.safetensors"), "pt") as f:
        assert {k: (f.metadata() or {})[k] for k in ("activation", "centered")} == {"activation": "relu",
                                                                                    "centered": "true"}
    edited("sae_feature_ablation", lambda d: d["featurizers"]["dictionary"].update(activation="relu",
                                                                                   centered=True)).compile()
    with pytest.raises(PlanError, match=r"sae\.safetensors: header 'centered' is 'true'; this document expects "
                                        r"'false'"):
        edited("sae_feature_ablation", lambda d: d["featurizers"]["dictionary"].update(centered=False)).compile()


@pytest.mark.parametrize("metadata, featurizer, message", [
    # An nnspec format: an nnspec bundle, its header checked whole.
    ({"format": "nnspec/1"}, STATED, r"published\.safetensors: header 'kind' is None; this document expects 'sae'"),
    # nnspec's header keys under another format: neither an nnspec bundle nor a published file.
    ({"kind": "sae", "model.key": "gpt2", "layer": "0"}, STATED,
     r"published\.safetensors carries nnspec header keys \(kind, model\.key, layer\) but its format is None, not "
     r"'nnspec/1'; an nnspec bundle's header is whole, a published file carries none of them"),
    # Unrelated metadata: a published file, read as one.
    ({"format": "pt"}, {}, r"published\.safetensors is a published sae, which says neither how it encodes nor where"),
])
def test_a_files_format_decides_whether_it_is_nnspecs_or_published(tmp_path, metadata, featurizer, message):
    with pytest.raises(PlanError, match=r"featurizers\.dictionary\.file_path: .*" + message):
        published(tmp_path, TENSORS, featurizer, metadata=metadata).compile()
    if metadata == {"format": "pt"}:
        plan = published(tmp_path, TENSORS, STATED, metadata=metadata).compile()
        assert points(plan.featurizers["dictionary"])[0].header is None


@pytest.mark.parametrize("kind, fields, bundle", [
    ("sae", {}, {"W_enc": (16, 32), "b_enc": (32,), "W_dec": (32, 16), "b_dec": (16,)}),
    ("subspace", {}, {"basis": (16, 4)}),
    ("pca", {"file_path": "basis.safetensors"}, {"basis": (16, 4), "mean": (16,)}),
    ("linear", {"classes": "d.c"}, {"W": (16, 3), "b": (3,)})])
def test_counting_a_featurizers_coordinates_moves_none_of_its_tensors(kind, fields, bundle):
    """T10's P4: `size()` encodes a zero vector made where the instance's tensors are, so an SAE's encoder on the
    GPU is not copied to the host to count its latents (1.27 s of the clamp document). On the meta device, whose
    tensors hold no data to copy, the count is the encode's width; a zero vector made on the CPU there fails."""
    f = featurizers.make(kind, fields, 16, {k: torch.zeros(shape) for k, shape in bundle.items()}, None,
                         ["a", "b", "c"], device="meta")
    assert f.size() == next(iter(bundle.values()))[-1]
    with pytest.raises(NotImplementedError):
        f.method("encode")(f, torch.zeros(1, 16))
