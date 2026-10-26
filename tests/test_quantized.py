"""Quantized `dtype`s on a GPU: the tiny Llama under `nf4`, `fp4` and `int8` on both engines, which load the one
`quantization_config` (`standard.loading`) and agree to the bit; a generation and a fit under `nf4`; the logit
lens on Llama-3.2-1B under `nf4` against float32; and `fp8` refused below compute capability 8.9.

The CPU-only suite skips this file (its refusals off CUDA are in `test_refusals.py`). It needs `bitsandbytes` and
runs as `test_real.py` does, from a venv whose torch matches the driver:

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. python -m pytest tests/test_quantized.py -q
"""

import pytest
import torch
from conftest import BATCH_SIZE, edited, same
from huggingface_hub import try_to_load_from_cache

from clab.engines import load

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

KEY, REVISION = "meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"
CACHED = isinstance(try_to_load_from_cache(KEY, "model.safetensors", revision=REVISION), str)
# What each quantization's model holds its parameters in: the compute dtype, and the packed weights.
HELD = {"nf4": ["bfloat16", "uint8"], "fp4": ["bfloat16", "uint8"], "int8": ["float16", "int8"]}


def quantized(name: str, dtype: str, change=lambda d: None):
    """Document `name` compiled with `model.dtype` set to `dtype`, and `change` applied."""
    def edit(d):
        d["model"]["dtype"] = dtype
        change(d)
    return edited(name, edit).compile()


def on_both(name: str, dtype: str, change=lambda d: None) -> dict:
    return {kind: load(kind, quantized(name, dtype, change).model, "cuda")
            .execute(quantized(name, dtype, change), batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")}


@pytest.mark.parametrize("dtype", ["nf4", "fp4", "int8"])
@pytest.mark.parametrize("name", ["patching", "logit_lens", "neuron_patching"])
def test_the_engines_load_one_quantization_and_agree_to_the_bit(name, dtype):
    """Each engine holds the weights packed and computes in the quantization's dtype, and their numbers are
    equal; the run records the packed dtype beside the compute one, the plan the quantization's name."""
    filled = on_both(name, dtype)
    assert same(filled["nnterp"].results, filled["hooks"].results)
    for plan in filled.values():
        assert plan.model.dtype == dtype and (plan.run.dtype, plan.run.devices) == (HELD[dtype], ["cuda:0"])


def test_a_quantization_changes_the_numbers():
    """`nf4` is a different model from float32, not a relabelled one."""
    plan = quantized("patching", "float32")
    assert not same(on_both("patching", "nf4")["nnterp"].results,
                    load("nnterp", plan.model, "cuda").execute(plan, batch_size=BATCH_SIZE).results)


def test_a_generation_and_a_fit_under_nf4_agree_on_both_engines():
    """`generated_answer` generates and `das` trains its rotation for one epoch, through the `nf4` model: the ids,
    the scores and the trained basis are the same bits on both engines, the basis made and trained in float32."""
    generated = on_both("generated_answer", "nf4")
    assert same(generated["nnterp"].results, generated["hooks"].results)
    fitted = on_both("das", "nf4", lambda d: d["steps"]["fit"].update(epochs=1))
    assert same(fitted["nnterp"].results, fitted["hooks"].results)
    basis = fitted["nnterp"].results["fit.rot"]["basis"]
    assert basis.dtype == torch.float32 and basis.shape == (16, 8)


@pytest.mark.skipif(not CACHED, reason=f"needs {KEY} in the local Hugging Face cache")
def test_the_logit_lens_on_llama_under_nf4_is_close_to_float32():
    """`logit_lens` on Llama-3.2-1B over the 42 weekday prompts, the lens at every layer: under `nf4` its next-token
    distributions against float32's, as the KL from float32's and how often the two pick the same token. At the last
    layer (the model's own prediction) they agree on every prompt (KL about 0.09); in the middle layers, where the
    lens is flat, the top token moves more often than not while the KL stays below about 0.25."""
    def lens(dtype: str):
        def change(d):
            d["model"].update(key=KEY, revision=REVISION, dtype=dtype)
            d["data"] = {"prompts": {"path": "weekdays_full/data"}}
            d["sites"]["resid"]["layers"] = {"sweep": list(range(16))}
            d["saves"] = ["lens.logits"]
        plan = edited("logit_lens", change).compile()
        return load("nnterp", plan.model, "cuda").execute(plan, batch_size=16).results["lens.logits"]

    exact, packed = lens("float32"), lens("nf4")
    kl, agree = {}, {}
    for layer, logits in exact.items():
        p, q = logits[:, -1].float().log_softmax(-1), packed[layer][:, -1].float().log_softmax(-1)
        kl[layer] = (p.exp() * (p - q)).sum(-1).mean().item()
        agree[layer] = (p.argmax(-1) == q.argmax(-1)).float().mean().item()
    last = max(kl)
    said = (f"nf4 against float32 over 42 prompts: at the last layer KL {kl[last]:.4f}, top-1 agreement "
            f"{agree[last]:.3f}; over all 16 layers mean KL {sum(kl.values()) / 16:.4f} (at most "
            f"{max(kl.values()):.4f}), top-1 agreement {sum(agree.values()) / 16:.3f}")
    assert kl[last] < 0.2 and agree[last] >= 0.9 and max(kl.values()) < 0.5, said


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
                    reason="this GPU runs fp8")
@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_fp8_is_refused_below_compute_capability_8_9(kind, monkeypatch):
    for where in ("nnterp", "hooks"):
        monkeypatch.setattr(f"clab.engines.{where}.StandardizedTransformer", lambda *a, **k: pytest.fail("loaded"))
    major, minor = torch.cuda.get_device_capability()
    with pytest.raises(ValueError, match=rf"^model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; "
                                         rf"cuda:0 \(.*\) is {major}.{minor}$"):
        load(kind, quantized("patching", "fp8").model, "cuda")
