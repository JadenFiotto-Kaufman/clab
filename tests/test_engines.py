"""The nnterp engine's `readout` is the model's own; the hooks engine's standard form (final norm, head, softcap)
agrees with it on every tiny model, Gemma-2's softcap included, and on a tiny Cohere, whose model scales its logits
after the head, differs, as it may.

The tiny model that softcaps its logits (`final_logit_softcapping` 30) is
`hf-internal-testing/tiny-random-Gemma2ForCausalLM` at commit de7c11b6. The suite reads it from the local Hugging Face
cache under HF_HUB_OFFLINE=1, as it does the other tiny checkpoints; to fetch it once, with hub access:

    env -u HF_HUB_OFFLINE uv run python -c "from huggingface_hub import snapshot_download as get; [get(key,
        revision=commit, ignore_patterns=['onnx/*']) for key, commit in [
        ('hf-internal-testing/tiny-random-Gemma2ForCausalLM', 'de7c11b6c25d26ddd1bf4324fcf479b61d18e440')]]"

The difference test uses `trl-internal-testing/tiny-CohereForCausalLM` at commit 90bc56bd (`logit_scale` 0.125)
where it is cached, and skips without.
"""

import pytest
import torch
from conftest import BATCH_SIZE, GEMMA2, document, plan_of, same
from huggingface_hub import try_to_load_from_cache

from clab import Plan

TINY = {"llama": plan_of("patching").model.model_dump(), "gpt2": plan_of("gpt2_reach").model.model_dump(),
        "qwen3": {"key": "trl-internal-testing/tiny-Qwen3ForCausalLM",
                  "revision": "25bb9b8963688237e94750f6f3e48f1947d3c7a3"},
        "gemma2": GEMMA2}
COHERE = {"key": "trl-internal-testing/tiny-CohereForCausalLM", "revision": "90bc56bd12c642f6bf5e710ebd7c626dd7c0201c"}
COHERE_CACHED = isinstance(try_to_load_from_cache(COHERE["key"], "model.safetensors", revision=COHERE["revision"]), str)


def reading(model: dict, entries: dict, **rows) -> Plan:
    """One forward over the weekday prompts (`rows` narrows them: `take`, ...) on `model`, saving each of
    `entries` (by name, `f.<name>`)."""
    return Plan(model=model, data={"prompts": {"path": "weekdays/train", **rows}},
                steps={"f": {"kind": "forward", "input": "prompts.input", "interventions": entries}},
                saves=[f"f.{name}" for name in entries], document=document("patching")).compile()


def ran(plan: Plan, engines) -> tuple[dict, dict]:
    """`plan`'s results on the nnterp engine and on the hooks engine, each its own: a run fills the plan's one dict."""
    def run(kind: str) -> dict:
        return dict(engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results)
    return run("nnterp"), run("hooks")


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("name", list(TINY))
def test_the_readout_of_the_last_blocks_output_is_the_models_logits(engines, name, kind):
    """At every token of a prompt, the last block's output read out is the logits the model returns, to the bit:
    on Gemma-2 after its softcap, which the readout applies as the model does. One prompt, so that the readout
    projects a matrix of the model's own shape: a narrower one (a batch's rows read at fewer positions than its
    padded width) can differ in the last bit, as a batch of another shape can."""
    plan = reading(TINY[name], {"lens": {"site": {"component": "layer_output", "layers": -1},
                                         "ops": [{"op": "readout"}]},
                                "logits": {"site": {"component": "logits"}}}, take=1)
    values = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results
    assert same(values["f.lens"], values["f.logits"])


@pytest.mark.skipif(not COHERE_CACHED, reason=f"needs {COHERE['key']} in the local Hugging Face cache")
def test_on_a_family_that_scales_its_logits_only_the_nnterp_engines_readout_is_the_models(engines):
    """Cohere multiplies the head's output by `logit_scale`: nnterp's readout does as the model does, so of the last
    block's output it is the model's logits to the bit; the hooks engine's standard form stops at the head."""
    plan = reading(COHERE, {"lens": {"site": {"component": "layer_output", "layers": -1}, "ops": [{"op": "readout"}]},
                            "logits": {"site": {"component": "logits"}}}, take=1)
    nnterp, hooks = ran(plan, engines)
    assert same(nnterp["f.lens"], nnterp["f.logits"]) and same(nnterp["f.logits"], hooks["f.logits"])
    assert not torch.equal(hooks["f.lens"], hooks["f.logits"])
