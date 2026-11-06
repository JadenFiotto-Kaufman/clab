"""The real-model documents (`documents/real/`, Llama-3.2-1B in float32) on a GPU: each on nnterp and hooks on
`cuda` at batch size 16, its numbers pinned per engine, the engines agreeing to the bit, and the findings each
document's description states; and behavior and layer_sweep in bfloat16 over prompts of four lengths under each
padding, `batch` and `sorted` feeding fewer tokens for numbers within bfloat16's rounding of the `dataset` run's.

The CPU-only suite skips this file: it needs a CUDA device and the checkpoint in the local Hugging Face cache. It
runs from a venv whose torch matches your CUDA driver, with nnspec installed `--no-deps` and `PYTHONPATH` at the
checkout, from the repo root:

    uv venv --python <a python whose torch runs on your driver> --system-site-packages /tmp/nnspec-gpu
    uv pip install --python /tmp/nnspec-gpu/bin/python --no-deps -e .
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=. /tmp/nnspec-gpu/bin/python -m pytest tests/test_real.py -q

Each snapshot records the GPU and the torch and transformers it was made
with (`made_with`), which need not be the lock's. The record is provenance, not a value: a re-pin that moves no
value leaves the file as it is, the record with it.
"""

import copy
import json
import os
import statistics

import pytest
import torch
import transformers
from conftest import REAL, ROOT, close, edited, placed, plain, plan_of, same, snapshot, within
from huggingface_hub import try_to_load_from_cache

from nnspec import Plan
from nnspec.format.data import PADDINGS

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
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "made_with": made_with, "values": values},
                        holds=lambda held: held["batch_size"] == BATCH_SIZE and within(held["values"], values))
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


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_head_patching_makes_291_model_calls(filled, kind):
    """The 32 counterfactual points read one head each from one forward, three batches of 16 over the 42 pairs, and
    each of the 96 patched points runs its own three: 291 model calls, where a forward per point would make 384."""
    costs = filled["real/head_patching", kind].run.costs
    assert sum(c["forwards"] for by in costs.values() for c in ([by] if "forwards" in by else by.values())) == 291


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


# How far `batch` and `sorted` padding may move a bfloat16 run's numbers from the `dataset` run's: measured, 0.031
# on one row's probability of its answer, 0.002 on the mean, and one pair of the 42 in a layer's interchange
# accuracy (1/42); pinned at 0.05, two pairs, on a probability or a share alike. In float32 the same runs differ by
# 3.4e-6 at most, and no pair flips.
BFLOAT16 = 0.05


def prefaced(name: str) -> Plan:
    """`name` in bfloat16 over the 42 weekday pairs, each pair's two prompts after a preface of `i // 11` sentences
    for row `i`, so that the rows grow in four steps and padding has something to change; a position counted from
    the end (the day, the last token) is where it was."""
    with open(os.path.join(ROOT, "data", "weekdays_full", "data.json")) as f:
        rows = json.load(f)
    preface = [{**r, "input": "Read the next line. " * (i // 11) + r["input"],
                "counterfactual_inputs": ["Read the next line. " * (i // 11) + r["counterfactual_inputs"][0]]}
               for i, r in enumerate(rows)]

    def change(d):
        d["data"] = {"days": {"rows": preface}}
        d["model"]["dtype"] = "bfloat16"
    return edited(name, change).compile()


PADDED = ["real/behavior", "real/layer_sweep"]


@pytest.fixture(scope="module")
def padded(engines) -> dict:
    out = {}
    for name in PADDED:
        for kind in ("nnterp", "hooks"):
            for padding in PADDINGS:
                plan = prefaced(name)
                out[name, kind, padding] = engines(kind, plan.model, device="cuda").execute(
                    plan, batch_size=BATCH_SIZE, padding=padding)
    return out


def fed(run) -> tuple[int, int]:
    """A run's tokens fed and the rows' own of them, over every step."""
    costs = [c for by in run.costs.values() for c in (by.values() if "forwards" not in by else [by])]
    return sum(c["tokens"] for c in costs), sum(c["real"] for c in costs)


@pytest.mark.parametrize("padding", ["batch", "sorted"])
@pytest.mark.parametrize("name", PADDED)
def test_padding_feeds_fewer_tokens_for_numbers_within_bfloat16s_rounding(padded, name, padding):
    for kind in ("nnterp", "hooks"):
        exact, got = padded[name, kind, "dataset"], padded[name, kind, padding]
        assert got.run.padding == padding and got.run.dtype == ["bfloat16"]
        (tokens, real), (exact_tokens, exact_real) = fed(got.run), fed(exact.run)
        assert real == exact_real and tokens < exact_tokens
        assert placed(got) == placed(exact)
        for key, value in exact.results.items():
            torch.testing.assert_close(got.results[key], value, rtol=0, atol=BFLOAT16, msg=lambda m: f"{key}: {m}")
    if name == "real/behavior":
        assert padded[name, "nnterp", padding].results["accuracy"].item() == 1.0


@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("name", PADDED)
def test_the_engines_agree_to_the_bit_under_each_padding(padded, name, padding):
    nnterp, hooks = padded[name, "nnterp", padding], padded[name, "hooks", padding]
    assert same(nnterp.results, hooks.results) and placed(nnterp) == placed(hooks)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_attribution_patching_ranks_layer_12_head_28_first(filled, kind):
    """Of the 512 heads, from two forwards and one backward, the one head_patching.json finds, at +2.20 where patching
    it measures +2.46; over the 96 heads both cover, the estimate follows the patched effect closely."""
    results = filled["real/attribution_patching", kind].results
    attribution, base = results["attribution"], results["logit_diff"].item()
    assert attribution.shape == (16, 32) and divmod(int(attribution.argmax()), 32) == (12, 28)
    assert abs(attribution[12, 28].item() - 2.204) < 1e-3 and abs(base + 2.4036) < 1e-3
    patched = filled["real/head_patching", kind].results["logit_diff"]
    effect = torch.tensor([[patched[layer, head].item() - base for head in range(32)] for layer in (10, 11, 12)])
    assert torch.corrcoef(torch.stack([effect.flatten(), attribution[10:13].flatten()]))[0, 1] > 0.99


def test_a_gradient_holds_less_than_a_fit_and_frozen_weights_less_than_thawed(engines):
    """On das.json's minibatches of 10 training pairs: a forward that takes one gradient at layer 8, where das trains,
    peaks below das's fit, whose minibatch forwards train a rotation there; and that fit peaks lower with the weights
    frozen, as every engine holds them, than with them requiring grad, its every result the same to the bit."""
    def one_rank(d):
        d["featurizers"]["rot"]["k"] = 16
        d["steps"]["fit"]["epochs"] = 2

    def gradient(d):
        d["steps"] = {"taken": {"kind": "forward", "input": "train.input", "interventions": {
            "g": {"site": "target", "pos": -4, "ops": [{"op": "grad", "of": "ce"}]},
            "ce": {"site": "head", "pos": -1, "ops": [{"op": "cross_entropy", "target": "train.cf_answer"}]}}}}
        d["saves"] = ["taken.g"]
    fit, taken = edited("real/das", one_rank).compile(), edited("real/das", gradient).compile()
    engine = engines("nnterp", fit.model, device="cuda")
    frozen = engine.execute(fit, batch_size=BATCH_SIZE)
    results, peak = copy.deepcopy(frozen.results), frozen.run.costs["fit"]["memory"]["cuda:0"]
    engine.model.requires_grad_(True)
    try:
        thawed = engine.execute(fit, batch_size=BATCH_SIZE)
    finally:
        engine.model.requires_grad_(False)
    assert same(results, thawed.results) and peak < thawed.run.costs["fit"]["memory"]["cuda:0"]
    gradient_peak = engine.execute(taken, batch_size=10).run.costs["taken"]["memory"]["cuda:0"]
    assert gradient_peak < peak


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_per_call_metric_holds_a_number_a_call_where_the_logits_hold_the_vocabulary(engines, kind):
    """64 greedy tokens after each of the 42 weekday prompts, with the answer's probability at every call: run at each
    call, `token_prob` holds one number a call and equals to the bit the same metric of the whole logits, which a
    read holds until the generation ends (`[16, 64, 128256]` in float32 a batch, 501 MiB, then arranged by position);
    the generation peaks at least one batch's logits lower, within half of one of what the device held before it."""
    def generation(interventions: dict, steps: dict, saves: list[str]):
        def change(d):
            d["steps"] = {"gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 64,
                                  "min_new_tokens": 64, "do_sample": False, "interventions": interventions}, **steps}
            d["saves"] = saves
        return edited("real/behavior", change).compile()
    each = generation({"p": {"site": "head", "pos": {"frame": ":"},
                             "ops": [{"op": "token_prob", "token": "days.base_answer"}]}}, {}, ["gen.p"])
    whole = generation({"l": {"site": "head", "pos": {"frame": ":"}}},
                       {"p": {"kind": "op", "op": "token_prob", "of": "gen.l", "token": "days.base_answer"}}, ["p"])
    engine = engines(kind, each.model, device="cuda")
    resident = torch.cuda.memory_allocated("cuda:0")        # the weights, and those of any engine loaded before
    a, b = (engine.execute(plan, batch_size=BATCH_SIZE) for plan in (each, whole))
    assert a.results["gen.p"].shape == (42, 64) and same(a.results["gen.p"], b.results["p"])
    logits = BATCH_SIZE * 64 * 128256 * 4
    peak, held = (run.run.costs["gen"]["memory"]["cuda:0"] for run in (a, b))
    assert held - peak > logits and peak - resident < logits / 2
