"""The real-model documents (`documents/real/`, Llama-3.2-1B in float32) on a GPU: each on nnterp and hooks on
`cuda` at batch size 16, its numbers pinned per engine, the engines agreeing to the bit, and the findings each
document's description states.

The CPU-only suite skips this file: it needs a CUDA device and the checkpoint in the local Hugging Face cache. It
runs from a venv whose torch matches your CUDA driver, with clab installed `--no-deps` and `PYTHONPATH` at the
checkout, from the repo root:

    uv venv --python <a python whose torch runs on your driver> --system-site-packages /tmp/clab-gpu
    uv pip install --python /tmp/clab-gpu/bin/python --no-deps -e .
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/clab-gpu/bin/python -m pytest tests/test_real.py -q

About two minutes on an idle RTX A6000. Each snapshot records the GPU and the torch and transformers it was made
with (`made_with`), which need not be the lock's.
"""

import statistics

import pytest
import torch
import transformers
from conftest import REAL, close, placed, plain, plan_of, same, snapshot
from huggingface_hub import try_to_load_from_cache

KEY, REVISION = "meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"
CACHED = isinstance(try_to_load_from_cache(KEY, "model.safetensors", revision=REVISION), str)
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CACHED),
                                reason=f"needs a CUDA device and {KEY} in the local Hugging Face cache")

BATCH_SIZE = 16
# What the hooks engine refuses by name: the pattern is read inside the attention's forward.
INSIDE = {"real/pattern_patching"}
SERVED = [(name, kind) for name in REAL for kind in ("nnterp", "hooks") if not (kind == "hooks" and name in INSIDE)]


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    out = {}
    for name, kind in SERVED:
        plan = plan_of(name)
        out[name, kind] = engines(kind, plan.model, device="cuda").execute(plan, batch_size=BATCH_SIZE)
    return out


def pinned(results: dict) -> dict:
    """What a snapshot holds: every result but das's trained bases (`[2048, k]` each, checked below instead)."""
    return plain({k: v for k, v in results.items() if k != "fit.rot"})


@pytest.mark.parametrize("name, kind", SERVED)
def test_numbers_match_the_snapshot(filled, name, kind):
    plan = filled[name, kind]
    assert plan.run.devices == ["cuda:0"] and plan.run.dtype == ["float32"]
    values = pinned(plan.results)
    made_with = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                 "transformers": transformers.__version__}
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "made_with": made_with, "values": values})
    assert expected["batch_size"] == BATCH_SIZE
    close(values, expected["values"])


@pytest.mark.parametrize("name, kind", SERVED)
def test_each_steps_memory_is_its_peak_on_the_gpu_weights_included(engines, filled, name, kind):
    plan = filled[name, kind]
    weights = sum(p.numel() * p.element_size() for p in engines(kind, plan.model, device="cuda").model.parameters())
    costs = [c for by in plan.run.costs.values() for c in (by.values() if "memory" not in by else [by])]
    assert all(list(c["memory"]) == ["cuda:0"] and c["memory"]["cuda:0"] >= weights for c in costs)


@pytest.mark.parametrize("name", [n for n in REAL if n not in INSIDE])
def test_the_engines_agree_to_the_bit(filled, name):
    assert same(filled[name, "nnterp"].results, filled[name, "hooks"].results)
    assert placed(filled[name, "nnterp"]) == placed(filled[name, "hooks"])


def test_hooks_refuses_the_attention_pattern(engines):
    plan = plan_of("real/pattern_patching")
    with pytest.raises(ValueError, match="the hooks engine does not serve attention_probabilities: nnterp reads it "
                                         "inside a module's forward"):
        engines("hooks", plan.model, device="cuda").execute(plan, batch_size=BATCH_SIZE)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_behavior_answers_every_prompt(filled, kind):
    results = filled["real/behavior", kind].results
    assert results["accuracy"].item() == 1.0
    assert abs(results["p_answer"].item() - 0.453) < 5e-4


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_layer_sweep_sees_the_day_move_to_the_last_token_at_layer_12(filled, kind):
    """Pairs out of 42 the patch flips, per layer: at the day token through layer 11, at the last from 12."""
    results = filled["real/layer_sweep", kind].results
    flipped = {name: [round(results[name][layer].item() * 42) for layer in range(16)] for name in ("iia_day", "iia_last")}
    assert flipped == {"iia_day": [42] * 11 + [41, 4, 4, 0, 0], "iia_last": [0] * 12 + [37, 37, 42, 42]}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_head_patching_finds_layer_12_head_28(filled, kind):
    results = filled["real/head_patching", kind].results
    diff = {point: v.item() for point, v in results["logit_diff"].items()}
    best = max(diff, key=lambda point: diff[point])
    assert best == (12, 28) and abs(diff[best] - 0.054) < 1e-3 and results["iia"][best].item() == 0.5
    assert abs(statistics.median(diff.values()) + 2.403) < 1e-3
    assert [point for point, v in results["iia"].items() if v.item() > 0] == [best]


def test_pattern_patching_moves_nothing(filled):
    results = filled["real/pattern_patching", "nnterp"].results
    diff = {point: v.item() for point, v in results["logit_diff"].items()}
    assert abs(diff[28] + 2.396) < 1e-3 and abs(statistics.median(diff.values()) + 2.403) < 1e-3
    assert all(v.item() == 0.0 for v in results["iia"].values())


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_das_accuracy_climbs_with_rank(filled, kind):
    """The held-out curve by rank, not the CPU's to the bit: twenty epochs of optimizer steps carry float32
    rounding forward, so a GPU run need not reproduce the CPU's 0, 0.75, 0.92, 1."""
    results = filled["real/das", kind].results
    curve = [results["held_iia"][k].mean().item() for k in (1, 4, 16, 64)]
    assert curve[0] == 0.0 and curve[-1] == 1.0 and curve == sorted(curve)
    for k, bundle in results["fit.rot"].items():
        basis = bundle["basis"]
        assert basis.shape == (2048, k)
        torch.testing.assert_close(basis.T @ basis, torch.eye(k), atol=1e-4, rtol=0)
