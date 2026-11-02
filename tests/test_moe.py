"""A mixture of experts' six values as components, nnterp's names and layouts, on the tiny mixtures the cache holds:
`router_logits`, `expert_weights` and `expert_indices` (the routing), `expert_outputs` (each slot's weighted output),
`routed_output` (their sum) and `shared_expert_output`, with the model's `num_experts` and `top_k`. They are what their
names say (the indices are the top k of the logits, the weights their softmax as the family normalizes it, the slots
sum to the routed output and with the shared expert to the MLP's), an expert's usage is an `eq` summed, the null
knockout is the clean run to the bit while the knockout and the forced route move the logits, and the engines agree
to the bit on what the hooks engine serves at a module boundary, which refuses the rest by name. The tiny mixtures,
read from the local Hugging Face cache under HF_HUB_OFFLINE=1 at the revisions `tests/conftest.py` pins; to fetch
them once, with hub access:

    env -u HF_HUB_OFFLINE uv run python -c "from huggingface_hub import snapshot_download as get; [get(key,
        revision=commit) for key, commit in [
        ('hf-internal-testing/tiny-random-MixtralForCausalLM', 'ccb12fe2fc142cb752085506c3db22572290e90c'),
        ('hf-internal-testing/tiny-random-OlmoeForCausalLM', '99e4bc8eb1a024e46da438c410d7e707ed49d15a'),
        ('trl-internal-testing/tiny-Qwen3MoeForCausalLM', '6db57163fa56fa0e00d4b820169908ac0d3a219d'),
        ('trl-internal-testing/tiny-Glm4MoeForCausalLM', '19b382257595016963b60289ba295e33abad6834'),
        ('hf-tiny-v2/tiny-random-JetMoeForCausalLM', '236ca328ccfedeb09989bfb2188b6fbb0cae5afc')]]"
"""

import pytest
import torch
from conftest import BATCH_SIZE, GLM4_MOE, JETMOE, MIXTRAL, OLMOE, QWEN3_MOE, document, edited, plan_of, same

from clab import Plan
from clab.plan import Model

# Each tiny mixture by name, read at block 1 (GLM-4-MoE's block 0 is dense); every one has 4 experts and routes each
# token to 2.
MIXTURES = {"mixtral": MIXTRAL, "olmoe": OLMOE, "qwen3_moe": QWEN3_MOE, "glm4_moe": GLM4_MOE}
SOFTMAX = ("mixtral", "olmoe", "qwen3_moe")          # a softmax router; GLM-4-MoE's is a sigmoid with a bias
VALUES = ("router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output",
          "shared_expert_output")
NO_SHARED = "this mixture has no shared expert"


def reading(model: dict, entries: dict, steps: dict | None = None) -> Plan:
    """One forward `f` over the weekday prompts on `model`, saving each of `entries` (`f.<name>`), and `steps`."""
    return Plan(model=model, data={"prompts": {"path": "weekdays/train"}},
                steps={"f": {"kind": "forward", "input": "prompts.input", "interventions": entries}, **(steps or {})},
                saves=[f"f.{name}" for name in entries] + list(steps or {}), document=document("patching")).compile()


def at(component: str, layer: int = 1, pos=None) -> dict:
    return {"site": {"component": component, "layers": layer}, "pos": {"index": ":"} if pos is None else pos}


def served(name: str) -> list[str]:
    return [v for v in VALUES if v != "shared_expert_output" or name == "glm4_moe"]


@pytest.fixture(scope="module")
def reads(engines) -> dict:
    """Each mixture's values at block 1, read at every token on the nnterp engine, with `sum` over `top_k` of the
    slots as a step."""
    out = {}
    for name, model in MIXTURES.items():
        plan = reading(model, {v: at(v) for v in served(name)} | {"mlp_output": at("mlp_output")},
                       {"slots": {"kind": "op", "op": "sum", "of": "f.expert_outputs", "over": ["top_k"]}})
        out[name] = dict(engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results)
    return out


@pytest.mark.parametrize("name", list(MIXTURES))
def test_the_six_values_read_in_their_layouts_and_are_what_their_names_say(engines, reads, name):
    """`[rows, pos, num_experts]`, `[rows, pos, top_k]` twice (the indices int64), `[rows, pos, top_k, hidden_size]`
    and `[rows, pos, hidden_size]`, the model's sizes those of its `Moe`; on a softmax router the indices are the top
    2 of the logits and the weights their softmax's, renormalized where the family does (Mixtral, Qwen3-MoE; OLMoE
    leaves `norm_topk_prob` off), to float32 rounding; a mixture without a shared expert refuses its value by name."""
    engine, r = engines("nnterp", Model.model_validate(MIXTURES[name])), reads[name]
    sizes = engine.sizes()
    assert (sizes["num_experts"], sizes["top_k"]) == (4, 2)
    rows, width = r["f.routed_output"].shape[:2]
    assert r["f.router_logits"].shape == (rows, width, 4) and r["f.expert_outputs"].shape == (rows, width, 2,
                                                                                                sizes["hidden_size"])
    assert r["f.expert_weights"].shape == r["f.expert_indices"].shape == (rows, width, 2)
    assert r["f.expert_indices"].dtype == torch.int64 and r["f.expert_weights"].dtype == torch.float32
    held = r["f.expert_indices"][..., 0] != -1
    assert held.all()                                       # the prompts are one length on these tokenizers
    if name in SOFTMAX:
        logits, weights = r["f.router_logits"], r["f.expert_weights"]
        assert torch.equal(logits.topk(2, -1).indices.sort(-1).values, r["f.expert_indices"].sort(-1).values)
        top = logits.softmax(-1).topk(2, -1).values
        top = top if name == "olmoe" else top / top.sum(-1, keepdim=True)
        torch.testing.assert_close(top, weights, rtol=0, atol=1e-6)
    if name != "glm4_moe":
        assert engine.unserved()["shared_expert_output"] == {0: NO_SHARED, 1: NO_SHARED}


@pytest.mark.parametrize("name", list(MIXTURES))
def test_the_slots_sum_to_the_routed_output_and_with_the_shared_expert_to_the_mlps_output(reads, name):
    """`sum` over `top_k` of `expert_outputs` is `routed_output` (to the bit here, two slots in float32 on the CPU;
    in general to float32 rounding), which is `mlp_output` to the bit on a mixture without a shared expert, and
    `routed_output + shared_expert_output` is `mlp_output` on GLM-4-MoE's, which adds the two."""
    r = reads[name]
    torch.testing.assert_close(r["slots"], r["f.routed_output"], rtol=1e-6, atol=1e-7)
    assert torch.equal(r["slots"], r["f.routed_output"])
    shared = r.get("f.shared_expert_output")
    assert torch.equal(r["f.routed_output"] if shared is None else r["f.routed_output"] + shared, r["f.mlp_output"])


def test_routing_entropy_is_entropy_of_the_router_logits_in_an_entry_and_as_a_step(engines):
    """In an entry, over `num_experts` by default, it is the entropy of the logits' softmax that a step computes
    from the read, to the bit, under ln 4 at every token."""
    plan = reading(MIXTRAL, {"logits": at("router_logits"),
                             "h": {**at("router_logits"), "ops": [{"op": "entropy"}]}},
                   {"stepped": {"kind": "op", "op": "entropy", "of": "f.logits"}})
    r = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    p = r["f.logits"].softmax(-1)
    assert torch.equal(r["f.h"], r["stepped"]) and r["f.h"].shape == p.shape[:2]
    torch.testing.assert_close(r["f.h"], -(p * p.log()).sum(-1))
    assert (r["f.h"] < torch.log(torch.tensor(4.0))).all()


def counted(d: dict) -> None:
    """expert_knockout.json with the clean run's raw slots and logits saved, the knocked-out run's slots and routed
    output, the clean run's slots' outputs, the forced run's slots, and every run's logits."""
    clean, knockout = d["steps"]["clean"]["interventions"], d["steps"]["knockout"]["interventions"]
    clean.update(raw={"site": "slots", "pos": {"index": ":8"}},
                 outputs={"site": {"component": "expert_outputs", "layers": 0}, "pos": {"index": ":8"}},
                 routed={"site": {"component": "routed_output", "layers": 0}, "pos": {"index": ":8"}})
    knockout.update(routed={"site": {"component": "routed_output", "layers": 0}, "pos": {"index": ":8"}})
    d["steps"]["forced"]["interventions"]["slots"] = {"site": "slots", "pos": {"index": ":8"}}
    for step in ("knockout", "null", "forced"):
        d["steps"][step]["interventions"]["logits"] = {"site": "head", "pos": -1}
    d["saves"] += ["clean.raw", "clean.outputs", "clean.routed", "clean.logits", "knockout.routed", "knockout.logits",
                   "null.logits", "forced.logits", "forced.slots"]


@pytest.fixture(scope="module")
def knocked(engines) -> dict:
    plan = edited("expert_knockout", counted).compile()
    return dict(engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results)


def test_an_experts_usage_is_eq_summed_over_rows_and_positions(knocked):
    """`usage` (each expert's `eq` of the slots, summed over positions and slots per row, then over rows) is the count
    of the raw indices, and the four experts' counts are every slot: 4 rows, 8 tokens, 2 slots."""
    raw = knocked["clean.raw"]
    usage = knocked["usage"]
    assert {e: usage[e].item() for e in range(4)} == {e: float((raw == e).sum()) for e in range(4)}
    assert sum(usage[e].item() for e in range(4)) == 64 == raw.numel()


def test_the_null_knockout_is_the_clean_run_to_the_bit_and_the_knockout_and_the_forced_route_move_the_logits(knocked):
    """Scaled by the mask of an expert no slot holds (4), the weights are the clean ones and so is every logit, its
    KL exactly 0; scaled by expert e's, its slots add nothing to the routed output (the clean routed output less
    their outputs, to float32 rounding) and the logits move; forced away from e by a -1e9 logit, no slot chooses it
    and the logits move."""
    clean, raw = knocked["clean.logits"], knocked["clean.raw"]
    assert same(knocked["null.logits"], clean) and torch.equal(knocked["null.kl"], torch.zeros(4, 1))
    for e in range(4):
        kept = (raw != e).float()[..., None]
        expected = (knocked["clean.outputs"] * kept).sum(2)
        torch.testing.assert_close(knocked["knockout.routed"][e], expected, rtol=1e-5, atol=1e-7)
        assert not torch.equal(knocked["knockout.logits"][e], clean)
        assert not (knocked["forced.slots"][e] == e).any() and (raw == e).any()
        assert not torch.equal(knocked["forced.logits"][e], clean)


@pytest.mark.parametrize("name", list(MIXTURES))
def test_the_engines_agree_to_the_bit_on_what_hooks_serves_and_hooks_refuses_the_rest_by_name(engines, name):
    """The hooks engine serves `routed_output` (and GLM-4-MoE's `shared_expert_output`), module outputs nnterp holds
    flat over tokens or not, bit-equal to nnterp's, and a write there gives both engines' logits to the bit; it
    refuses the logits read inside the router's forward, the experts' arguments and the slots inside their forward,
    each by name."""
    entries = {"routed": at("routed_output"), "half": {**at("routed_output", 0), "ops": [{"op": "scale",
                                                                                          "factor": 0.5}]},
               "logits": {"site": {"component": "logits"}, "pos": -1}}
    if name == "glm4_moe":
        entries["shared"] = at("shared_expert_output")
        entries["half"] = {**at("shared_expert_output"), "ops": [{"op": "scale", "factor": 0.5}]}
    plan = reading(MIXTURES[name], entries)
    nnterp, hooks = (dict(engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results)
                     for kind in ("nnterp", "hooks"))
    assert same(nnterp, hooks)
    refused = engines("hooks", plan.model).unserved()
    inside = "nnterp reads it inside a module's forward ({}), not at a module boundary"
    assert refused["router_logits"][1] == inside.format("router.source.F_linear_0.output")
    assert refused["expert_outputs"][1] == inside.format(
        "experts.source.experts_forward_1.source.weighted_out_view_0.output")
    for value in ("expert_weights", "expert_indices"):
        assert refused[value][1] == ("nnterp reads it among a module's call arguments (experts.inputs), not at a "
                                     "module boundary")
        with pytest.raises(ValueError, match=rf"^f\.x: the hooks engine does not serve {value}: nnterp reads it "
                                             r"among a module's call arguments \(experts\.inputs\), not at a module "
                                             r"boundary$"):
            engines("hooks", plan.model).execute(reading(MIXTURES[name], {"x": at(value)}), batch_size=BATCH_SIZE)


def test_remote_local_gives_the_nnterp_engines_reads_to_the_bit(engines, reads):
    plan = reading(MIXTRAL, {v: at(v) for v in served("mixtral")} | {"mlp_output": at("mlp_output")},
                   {"slots": {"kind": "op", "op": "sum", "of": "f.expert_outputs", "over": ["top_k"]}})
    there = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    assert same(dict(there.results), reads["mixtral"])


def test_on_jetmoe_nnterp_ranks_the_indices_before_the_weights_and_hooks_serves_the_routers_projection(engines):
    """JetMoE's router takes the top 2 of its logits and then their softmax, computing the indices before the
    weights, which nnterp's rank orders so (at Mixtral's experts module they share a rank): one forward reads both,
    the weights the softmax of the logits at the indices, to float32 rounding. Its logits are its projection's
    output, which the hooks engine serves, to the bit; its routed sum is an operation inside the mixture's forward,
    which it refuses by name."""
    nnterp = engines("nnterp", Model.model_validate(JETMOE))
    assert nnterp.rank("expert_indices", 1) < nnterp.rank("expert_weights", 1)
    mixtral = engines("nnterp", Model.model_validate(MIXTRAL))
    assert mixtral.rank("expert_indices", 1) == mixtral.rank("expert_weights", 1)
    plan = reading(JETMOE, {"weights": at("expert_weights"), "indices": at("expert_indices")})
    r = nnterp.execute(plan, batch_size=BATCH_SIZE).results
    logits = engines("nnterp", plan.model).execute(reading(JETMOE, {"l": at("router_logits")}),
                                                   batch_size=BATCH_SIZE).results["f.l"]
    torch.testing.assert_close(logits.gather(-1, r["f.indices"]).softmax(-1), r["f.weights"], rtol=0, atol=1e-6)
    plan = reading(JETMOE, {"l": at("router_logits")})
    assert same(*(dict(engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results)
                  for kind in ("nnterp", "hooks")))
    assert engines("hooks", plan.model).unserved()["routed_output"][1] == \
        "nnterp reads it inside a module's forward (source.layer_output_1.output), not at a module boundary"


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_without_a_mixture_each_value_is_refused_by_name_and_its_sizes_are_absent(engines, kind):
    """On the tiny Llama no block has a mixture: `num_experts` and `top_k` are not among its sizes and each value is
    refused by name; on GLM-4-MoE block 0 is dense and refuses each by nnterp's reason, and block 1 serves them,
    whose `select` is held to the mixture's 4 experts."""
    llama = plan_of("patching").model
    engine = engines(kind, llama)
    assert "num_experts" not in engine.sizes() and "top_k" not in engine.sizes()
    plan = reading(llama.model_dump(), {"x": at("router_logits", 0)})
    with pytest.raises(ValueError, match=rf"^f\.x: the {kind} engine does not serve router_logits: no block of this "
                                         r"model has a mixture of experts$"):
        engine.execute(plan, batch_size=BATCH_SIZE)
    glm = engines(kind, Model.model_validate(GLM4_MOE))
    with pytest.raises(ValueError, match=rf"^f\.x: the {kind} engine does not serve routed_output: no routed_output "
                                         r"value on this block's mlp$"):
        glm.execute(reading(GLM4_MOE, {"x": at("routed_output", 0)}), batch_size=BATCH_SIZE)
    glm.execute(reading(GLM4_MOE, {"x": at("routed_output", 1)}), batch_size=BATCH_SIZE)
    if kind == "nnterp":
        select = {"site": {"component": "router_logits", "layers": 1, "select": {"num_experts": 4}}, "pos": -1}
        with pytest.raises(ValueError, match=r"^f\.x: select num_experts 4 outside the model's 4"):
            glm.execute(reading(GLM4_MOE, {"x": select}), batch_size=BATCH_SIZE)
