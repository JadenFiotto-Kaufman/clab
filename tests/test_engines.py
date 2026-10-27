"""A component means one thing on every engine, which only reaches it. The nnterp engine's `readout` is the model's
own; the hooks engine's standard form (final norm, head, softcap) agrees with it on every tiny model, Gemma-2's softcap
included, and on a tiny Cohere, whose model scales its logits after the head, differs, as it may. What an engine
refuses it refuses layer by layer, from the one source both read (`clab.engines.standard`), so that the engines differ
only where nnterp reads a value inside a module's forward, which the hooks engine refuses at every layer that has the
module.

Two checkpoints are this module's: `hf-internal-testing/tiny-random-Gemma2ForCausalLM` at commit de7c11b6, the one
tiny model that softcaps its logits (`final_logit_softcapping` 30), and `yujiepan/qwen3-next-moe-tiny-random` at
commit ec636a93, a hybrid whose blocks 0 to 2 mix by linear attention and have no `self_attn`, and whose block 3 is
full attention. The suite reads both from the local Hugging Face cache under HF_HUB_OFFLINE=1, as it does the other
tiny checkpoints; to fetch them once, with hub access:

    env -u HF_HUB_OFFLINE uv run python -c "from huggingface_hub import snapshot_download as get; [get(key,
        revision=commit, ignore_patterns=['onnx/*']) for key, commit in [
        ('hf-internal-testing/tiny-random-Gemma2ForCausalLM', 'de7c11b6c25d26ddd1bf4324fcf479b61d18e440'),
        ('yujiepan/qwen3-next-moe-tiny-random', 'ec636a93e953ed50d2ecebda99b04673b5f1e248')]]"

The difference test uses `trl-internal-testing/tiny-CohereForCausalLM` at commit 90bc56bd (`logit_scale` 0.125)
where it is cached, and skips without.
"""

import pytest
import torch
from conftest import BATCH_SIZE, GEMMA2, HYBRID, document, plan_of, same
from huggingface_hub import try_to_load_from_cache

from clab import Plan
from clab.format.address import COMPONENTS
from clab.plan import Model

TINY = {"llama": plan_of("patching").model.model_dump(), "gpt2": plan_of("gpt2_reach").model.model_dump(),
        "qwen3": {"key": "trl-internal-testing/tiny-Qwen3ForCausalLM",
                  "revision": "25bb9b8963688237e94750f6f3e48f1947d3c7a3"},
        "gemma2": GEMMA2, "hybrid": HYBRID}
EAGER = {"attn_implementation": "eager"}
# What nnterp reads inside the attention's forward, which the hooks engine refuses at the hybrid's block 3, the one
# with attention.
INTERIOR = {"attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities",
            "attention_head_outputs"}
LINEAR = "no self_attn module on this block"
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


def at(component: str, layers: int) -> dict:
    return {"site": {"component": component, "layers": layers}, "pos": -1}


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


def test_the_engines_differ_only_where_nnterp_reads_inside_a_forward(engines):
    """On the hybrid, block by block, the hooks engine's answer is nnterp's (`attention_output` at blocks 0 to 2,
    which have no `self_attn`) but where nnterp reads the value inside a module's forward at a block with the module:
    the attention interior at block 3, which the hooks engine refuses for that reason under sdpa and eager alike."""
    answers = {}
    for name, extra in (("sdpa", {}), ("eager", EAGER)):
        model = Model.model_validate({**HYBRID, **extra})
        nnterp, hooks = (engines(kind, model).unserved() for kind in ("nnterp", "hooks"))
        assert nnterp["attention_output"] == hooks["attention_output"] == {0: LINEAR, 1: LINEAR, 2: LINEAR}
        for component in COMPONENTS:
            own = {3: hooks[component][3]} if component in INTERIOR else {}
            assert hooks[component] == ({**(nnterp[component] or {}), **own} if own else nnterp[component]), component
            assert all(why.startswith("nnterp reads it inside a module's forward (source.") for why in own.values())
        answers[name] = hooks
    assert answers["sdpa"] == answers["eager"]


def test_attention_output_at_the_hybrids_attention_block_is_one_value_on_both_engines(engines):
    plan = reading(HYBRID, {"attn": at("attention_output", 3), "premix": at("attention_premix", 3)})
    nnterp, hooks = ran(plan, engines)
    assert same(nnterp, hooks)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_attention_output_at_a_linear_attention_block_is_refused_by_name(engines, kind):
    plan = reading(HYBRID, {"attn": at("attention_output", 1)})
    engine = engines(kind, plan.model)
    with pytest.raises(ValueError, match=rf"^f\.attn: the {kind} engine does not serve attention_output: {LINEAR}$"):
        engine.execute(plan, batch_size=BATCH_SIZE)


def test_the_attention_interior_of_the_hybrid_is_served_by_nnterp_alone(engines):
    """Under eager attention nnterp reads block 3's attention pattern; the hooks engine refuses it there, since
    nnterp reads it inside the attention's forward rather than at a module boundary."""
    plan = reading({**HYBRID, **EAGER}, {"pattern": at("attention_probabilities", 3)})
    pattern = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results["f.pattern"]
    assert pattern.shape[1:3] == (1, 16) and torch.allclose(pattern.nan_to_num().sum(-1), torch.ones(()))
    with pytest.raises(ValueError, match=r"^f\.pattern: the hooks engine does not serve attention_probabilities: "
                                         r"nnterp reads it inside a module's forward \(source\..*\), not at a module "
                                         r"boundary$"):
        engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
