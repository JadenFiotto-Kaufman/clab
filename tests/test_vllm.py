"""The vllm engine, on a GPU with vLLM installed: on `HuggingFaceTB/SmolLM2-135M-Instruct` in float32, its reads and
writes are plain nnsight's on nnterp's `StandardizedVLLM` to the bit (the same requests in one trace), a null is
exact and a run repeats exactly at any batch size (the batch-invariant kernels), a generation's ids are vLLM's own,
`at-reads` is `full` to vLLM's last bits (the readout runs the final norm on the stream alone, where the forward fuses
it with the last residual add), what vLLM cannot do is refused by name, the run record says what served it, and the
transformers engine agrees within vLLM's kernels' rounding, not to the bit.

The CPU-only suite skips this file: it needs vLLM importable, a CUDA device and the checkpoint in the local Hugging
Face cache. It runs from a venv whose torch runs on your driver, with vLLM, nnsight, nnterp and clab installed, from
the repo root:

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 <venv>/bin/python -m pytest tests/test_vllm.py -q

One engine for the module (vLLM reserves `memory` of the card at start, 0.25 here, about 12 GB on a 48 GB card).
"""

import copy
import importlib.util
import os
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, document, edited, same
from huggingface_hub import try_to_load_from_cache

from clab import Plan
from clab.engines import load
from clab.plan import Model

KEY, REVISION = "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"
CACHED = isinstance(try_to_load_from_cache(KEY, "model.safetensors", revision=REVISION), str)
VLLM = importlib.util.find_spec("vllm") is not None
pytestmark = pytest.mark.skipif(not (VLLM and torch.cuda.is_available() and CACHED),
                                reason=f"needs vLLM, a CUDA device and {KEY} in the local Hugging Face cache")

MODEL: dict[str, Any] = {"key": KEY, "revision": REVISION, "dtype": "float32", "attn_implementation": "eager"}
OPTIONS = {"memory": 0.25, "max_len": 512}
LAYER = 12
PROMPTS = {"path": "weekdays/train"}


@pytest.fixture(scope="module")
def engines():
    """The vllm engine and the nnterp engine on the CPU, each once, for the one model block."""
    model = Model.model_validate(MODEL)
    return {"vllm": load("vllm", model, **OPTIONS), "nnterp": load("nnterp", model, "cpu")}


def plan(steps: dict, saves: list[str], data: dict | None = None, **top) -> Plan:
    """A compiled plan on the module's model over the weekday pairs (or `data`)."""
    return Plan(model=MODEL, data=data or {"pairs": PROMPTS}, steps=steps, saves=saves, document=document("patching"),
                **top).compile()


def run(engine, built: Plan, **options) -> Plan:
    """`built` executed on `engine` at the pinned batch size: a copy of the filled plan holding its own results."""
    filled = engine.execute(built, batch_size=BATCH_SIZE, **options)
    return filled.model_copy(update={"results": copy.deepcopy(filled.results)})


def patching_plan() -> Plan:
    """The smallest interchange, its reads saved raw: the counterfactual stream at the last token of layer `LAYER`,
    swapped into the base prompts, and the head read at the last position as `logits` and as `lm_head_output`."""
    return plan({
        "counterfactual": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]",
                           "interventions": {"v_cf": {"site": "target", "pos": -1}}},
        "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
            "logits": {"site": {"component": "logits"}, "pos": -1},
            "head": {"site": {"component": "lm_head_output"}, "pos": -1}}},
    }, ["counterfactual.v_cf", "patched.logits", "patched.head"],
        sites={"target": {"component": "layer_output", "layers": LAYER}})


def encoded(engine, built: Plan, column: str) -> list[list[int]]:
    """Each row's token ids for a column of the plan's dataset, as the run encodes them."""
    from clab.format import data

    rows = built.data["pairs"].rows
    assert rows is not None
    return [data.encode(engine.tokenizer, data.field(r, column), None)["ids"] for r in rows]


def plain_patching(engine, base: list[list[int]], counterfactual: list[list[int]]) -> dict:
    """What `patching_plan` reads, done in plain nnsight on the engine's `StandardizedVLLM`: the same requests in
    one trace each, the counterfactual read, the swap by assignment, the head at the last position."""
    import nnsight

    model, layer = engine.model, engine.model.layers[LAYER]
    v: Any = None
    with model.trace(temperature=0.0, max_tokens=1) as tracer:
        for ids in counterfactual:
            with tracer.invoke(ids):
                v = layer.layer_output[:, -1:].clone().save()
    donors = v if isinstance(v, list) else [v]
    out: Any = None
    with model.trace(temperature=0.0, max_tokens=1) as tracer:
        for ids, donor in zip(base, donors):
            with tracer.invoke(ids):
                out = nnsight.save({})
                stream = layer.layer_output.clone()
                stream[:, -1:] = donor
                layer.layer_output = stream
                out["logits"] = model.logits[:, -1:].clone()
    outs = out if isinstance(out, list) else [out]
    return {"v_cf": torch.cat(donors), "logits": torch.cat([o["logits"] for o in outs])}


def test_reads_and_writes_are_plain_nnsights_on_vllm_to_the_bit(engines):
    engine = engines["vllm"]
    built = patching_plan()
    filled = run(engine, built)
    plain = plain_patching(engine, encoded(engine, built, "pairs.input"),
                           encoded(engine, built, "pairs.counterfactual_inputs[0]"))
    assert torch.equal(filled.results["counterfactual.v_cf"], plain["v_cf"].cpu())
    assert torch.equal(filled.results["patched.logits"], plain["logits"].cpu())
    # lm_head_output is the logits where the model does nothing after the head
    assert torch.equal(filled.results["patched.head"], filled.results["patched.logits"])
    assert filled.results["patched.logits"].shape == (4, 1, 49152)


def test_a_null_write_is_exact_and_a_run_repeats_to_the_bit_at_any_batch_size(engines):
    engine = engines["vllm"]
    clean = plan({"base": {"kind": "forward", "input": "pairs.input", "interventions": {
        "v": {"site": "target", "pos": -1}, "logits": {"site": {"component": "logits"}, "pos": -1}}}},
        ["base.v", "base.logits"], sites={"target": {"component": "layer_output", "layers": LAYER}})
    null = plan({"base": {"kind": "forward", "input": "pairs.input", "interventions": {
        "v": {"site": "target", "pos": -1},
        "back": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "base.v"}]},
        "logits": {"site": {"component": "logits"}, "pos": -1}}}},
        ["base.v", "base.logits"], sites={"target": {"component": "layer_output", "layers": LAYER}})
    first, again, nulled = run(engine, clean), run(engine, clean), run(engine, null)
    assert same(first.results, again.results)
    assert same(first.results, nulled.results)
    whole = engine.execute(clean, batch_size=4)          # one batch of every row: the invariant kernels' promise
    assert same(first.results, whole.results)
    assert first.run is not None
    assert first.run.options == {"tp": 1, "memory": 0.25, "max_len": 512, "invariant": True}


def test_at_reads_is_full_to_vllms_last_bits(engines):
    """The readout runs the model's final norm on the stream alone and vLLM's logits processor (`project_on_vocab`
    in the worker), where vLLM's own forward fuses the norm with the last residual add: the two agree to the last
    bits, as `at-reads` promises, not to the bit; the top token agrees."""
    engine = engines["vllm"]
    built = patching_plan()
    full, at_reads = run(engine, built), run(engine, built, readout="at-reads")
    assert at_reads.run is not None and at_reads.run.readout == "at-reads"
    assert same(full.results["patched.head"], at_reads.results["patched.head"])      # stays at the head
    a, b = full.results["patched.logits"], at_reads.results["patched.logits"]
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert (a - b).abs().max() < 1e-4, (a - b).abs().max()


def test_a_generation_is_vllms_own(engines):
    """Greedy ids from `model.samples` step by step equal vLLM's `generate` on the same prompts, a row that stops
    at its EOS included, and a decode-call read of the head holds one value per call the batch made, blank past a
    row's own."""
    engine = engines["vllm"]
    rows = [{"input": "If today is Thursday, tomorrow is"}, {"input": "<|im_start|>user\nSay hi.<|im_end|>\n"
                                                             "<|im_start|>assistant\n"},
            {"input": "The capital of France is"}, {"input": "Two plus two is"}]
    built = plan({"answer": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 12, "do_sample": False,
                             "interventions": {"head": {"site": {"component": "lm_head_output"},
                                                        "pos": {"frame": ":"}}}}},
                 ["answer", "answer.head"], data={"prompts": {"rows": rows}})
    filled = run(engine, built)
    outputs = engine.model.generate([r["input"] for r in rows], max_tokens=12, temperature=0.0)
    ids = filled.results["answer"]
    for r, output in enumerate(outputs):
        said = list(output.outputs[0].token_ids)
        assert ids[r, :len(said)].tolist() == said, r
        assert (ids[r, len(said):] == engine.eos[0]).all(), r
    said = [len(o.outputs[0].token_ids) for o in outputs]
    assert min(said) < max(said) == ids.shape[1], said     # the chat row stops at its EOS before the others
    head = filled.results["answer.head"]
    assert head.shape[:2] == (4, max(said))                 # one call per new token: the prefill, then the decode calls
    short = min(range(4), key=said.__getitem__)
    assert head[short, said[short]:].isnan().all() and not head[short, :said[short]].isnan().any()


def test_a_decode_call_write_lands_and_its_null_is_exact(engines):
    """A steer added at every decode call changes what the rows say; the same write at scale 0 says what the
    unsteered run says, to the bit, ids and head reads alike."""
    engine = engines["vllm"]
    rows = [{"input": "If today is Thursday, tomorrow is"}, {"input": "The capital of France is"},
            {"input": "Two plus two is"}, {"input": "My favourite colour is"}]

    def built(scale: float | None) -> Plan:
        steer = {} if scale is None else {"steer": {"site": "target", "pos": {"frame": ":"}, "ops": [
            {"op": "add", "with": "direction", "scale": scale}]}}
        return plan({"read": {"kind": "forward", "input": "prompts.input",
                              "interventions": {"v": {"site": "target", "pos": -1}}},
                     "direction": {"kind": "op", "op": "mean", "of": "read.v"},
                     "answer": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 8, "do_sample": False,
                                "interventions": {**steer, "head": {"site": {"component": "logits"},
                                                                    "pos": {"frame": "1:"}}}}},
                    ["answer", "answer.head"], data={"prompts": {"rows": rows}},
                    sites={"target": {"component": "layer_output", "layers": LAYER}})
    clean, null, steered = run(engine, built(None)), run(engine, built(0.0)), run(engine, built(4.0))
    assert same(clean.results, null.results)
    assert not torch.equal(clean.results["answer"], steered.results["answer"])
    assert steered.results["answer.head"].shape[:2] == (4, 7) and not steered.results["answer.head"].isnan().any()


def test_transformers_agrees_within_vllms_kernels_rounding(engines):
    """vLLM computes the same model with other kernels: the top token agrees, the logits to vLLM's rounding."""
    built = patching_plan()
    ours, theirs = run(engines["vllm"], built), run(engines["nnterp"], built)
    a, b = ours.results["patched.logits"].float(), theirs.results["patched.logits"].float()
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert (a - b).abs().max() < 1e-3, (a - b).abs().max()


def test_what_vllm_cannot_do_is_refused_by_name(engines):
    engine = engines["vllm"]
    model = Model.model_validate(MODEL)
    with pytest.raises(ValueError, match=r"^fit: a fit trains through the model's gradient, which vLLM's kernels do "
                                         r"not take; run it on the nnterp or hooks engine$"):
        engine.execute(edited("das", lambda d: d.update(model=MODEL)).compile(), batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^patched.h: a grad entry takes the model's gradient"):
        engine.execute(edited("attribution_patching", lambda d: d.update(model=MODEL)).compile(),
                       batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^patched.zero: the vllm engine does not write attention_probabilities: "):
        engine.execute(edited("attention_knockout", lambda d: d.update(model=MODEL)).compile(), batch_size=BATCH_SIZE)
    first = plan({"base": {"kind": "forward", "input": "pairs.input",
                           "interventions": {"logits": {"site": {"component": "logits"}, "pos": 0}}}},
                 ["base.logits"])
    with pytest.raises(ValueError, match=r"^base.logits: the vllm engine serves logits at the prompt's last position "
                                         r"alone \(vLLM computes the head at the prompt's last position alone\); this "
                                         r"entry meets position \[0\] on a row of 7 tokens: name pos -1"):
        engine.execute(first, batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^device: the vllm engine cannot honor 'cpu': it keeps the run's values on "
                                         r"cuda:0"):
        load("vllm", model, "cpu")
    with pytest.raises(ValueError, match=r"^model.dtype 'nf4': the vllm engine cannot honor it: vLLM holds a quantized"):
        load("vllm", model.model_copy(update={"dtype": "nf4"}))
    with pytest.raises(ValueError, match=r"^memory 2: the fraction of each GPU's memory"):
        load("vllm", model, memory=2)
    with pytest.raises(ValueError, match=r"^answer: generate rejected 'penalty_alpha': the vllm engine's generate maps "):
        engine.execute(plan({"answer": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 2,
                                        "do_sample": False, "penalty_alpha": 0.5, "interventions": {}}},
                            ["answer"]), batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^answer: generate rejected 'max_new_tokens': the vllm engine needs the step"):
        engine.execute(plan({"answer": {"kind": "generate", "input": "pairs.input", "do_sample": False,
                                        "interventions": {}}}, ["answer"]), batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^answer: pairs.input's longest row has 7 tokens, and the step generates up "
                                         r"to 600 more, past the 512 the vllm engine serves \(its max_len; raise it "
                                         r"with -o max_len, up to the checkpoint's context\)$"):
        engine.execute(plan({"answer": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 600,
                                        "do_sample": False, "interventions": {}}}, ["answer"]), batch_size=BATCH_SIZE)


def test_the_run_record_says_what_served_it(engines):
    filled = run(engines["vllm"], patching_plan())
    record = filled.run
    assert record is not None
    assert record.engine == "vllm" and record.device == "cuda:0" and record.devices == ["cuda:0"]
    assert record.dtype == ["float32"] and record.threads is not None and record.versions["vllm"]
    assert all(c["memory"] is None for c in record.costs.values())
    # 4 rows at the pinned batch size of 2: two traces, 7 tokens a row
    assert record.costs["patched"]["forwards"] == 2 and record.costs["patched"]["real"] == 28
