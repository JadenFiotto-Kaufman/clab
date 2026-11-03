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
from clab.format.sweeps import points
from clab.plan import Model, entries

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


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("ndif", {"remote": "local"})])
def test_on_a_fresh_engine_jetmoe_reads_its_router_logits_with_its_indices_in_one_forward(kind, options):
    """The router's logits are its child projection's output and its indices an operation inside its forward, which
    nnsight instruments lazily; on an engine no trace has warmed, one forward reading both raised `OutOfOrderError`
    until `ready` touched the router's `.source` at load. The indices are the top 2 of the logits, on nnterp and on
    `remote="local"` alike, to the bit."""
    from clab.engines import load

    plan = reading(JETMOE, {"logits": at("router_logits"), "indices": at("expert_indices")})
    r = dict(load(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE).results)
    assert torch.equal(r["f.logits"].topk(2, -1).indices.sort(-1).values, r["f.indices"].sort(-1).values)
    if kind == "ndif":
        assert same(r, dict(load("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results))


def test_under_nnterps_default_family_a_mixtures_value_is_refused_as_unaddressed():
    """A checkpoint nnterp has no family for loads under its default family, which says a block's MLP is a mixture
    (its reason for the activation) and names none of a mixture's values: each is refused as not addressed, with the
    MLPs' components served there, not as a model without a mixture."""
    from nnterp import StandardizedTransformer
    from nnterp.families import default

    from clab.engines import standard

    shell = StandardizedTransformer(MIXTRAL["key"], revision=MIXTRAL["revision"], family=default)
    answer = standard.own(standard.unserved(shell))
    for value in VALUES:
        assert answer[value] == ("nnterp has no family for this architecture (it reads the model under its default "
                                 "family), so its mixture of experts is not addressed; its MLPs are served as "
                                 "mlp_input")
    assert answer["mlp_activation"] == {0: default.MIXTURE, 1: default.MIXTURE}


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


# -- a write of ids: a swap of experts the model has, refused by name otherwise before the model reads it ------------


def rerouted(ops: list[dict], steps: dict | None = None, pos=-1) -> Plan:
    """A forward `f` over the weekday prompts on the tiny Mixtral writing slot 0 of `expert_indices` at block 1 with
    `ops`, reading every token's `routed_output` and `expert_indices` there, beside `steps`."""
    write = {"site": {"component": "expert_indices", "layers": 1, "select": {"top_k": 0}}, "pos": pos, "ops": ops}
    return reading(MIXTRAL, {"w": write, "routed": at("routed_output"), "idx": at("expert_indices")}, steps)


@pytest.mark.parametrize("op", [{"op": "add", "with": 1}, {"op": "scale", "factor": 0.5},
                                {"op": "noise", "scale": 1.0, "seed": 0}, {"op": "clamp", "max": 3}])
def test_an_edit_of_expert_indices_other_than_swap_is_refused_at_compile(op):
    """`add 1`, `scale 0.5` and `noise` would compute on ids as numbers and the write would truncate them back to
    int64 (3 * 0.5 lands as 1, 3 + 1 as 4, outside the 4 experts), so each, and every other edit, is refused by name
    before the model is loaded."""
    from clab.plan import PlanError

    with pytest.raises(PlanError, match=rf"^steps\.f\.interventions\.w\.ops\[0\]: '{op['op']}' would compute on "
                                        r"expert_indices, the ids the model indexes its num_experts with, as numbers, "
                                        r"and the write would cast the result back to ids, truncated; an index takes "
                                        r"a swap, \{'op': 'swap', 'with': <an id, or a named value of ids>\}$"):
        rerouted([op])


@pytest.mark.parametrize("given", [4, -1, 2.7])
def test_a_swap_of_expert_indices_with_a_number_that_is_no_expert_is_refused_before_any_forward(engines, given):
    """4 is past the tiny Mixtral's 4 experts and -1 before them, which transformers' grouped expert matmul drops,
    shifting every later group, and 2.7 would land as 2: each is refused by name against the model's `num_experts`
    before any forward."""
    plan = rerouted([{"op": "swap", "with": given}])
    with pytest.raises(ValueError, match=rf"^f\.w: swap with {given}: expert_indices holds ids, and an id is a whole "
                                         r"number from 0 to 3 \(the model's num_experts, 4\)$"):
        engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)


def test_a_swap_of_an_expert_reroutes_its_slot_and_moves_no_other_token_past_rounding(engines):
    """Every row's last token routes slot 0 to experts 2 and 3; a swap of 0, 1 or 3 there lands as written, moves
    that token's routed output at the routed output's own scale, and every other token's by the grouped matmul's
    rounding alone (below 1e-8 here, where an id outside the experts moved them by as much as the token's own), and a
    swap of the id already there changes nothing, to the bit."""
    clean = reading(MIXTRAL, {"routed": at("routed_output"), "idx": at("expert_indices")})
    base = dict(engines("nnterp", clean.model).execute(clean, batch_size=BATCH_SIZE).results)
    assert base["f.idx"][:, -1].tolist() == [[2, 3]] * 4
    for e in range(4):
        r = dict(engines("nnterp", clean.model).execute(rerouted([{"op": "swap", "with": e}]),
                                                         batch_size=BATCH_SIZE).results)
        assert r["f.idx"][:, -1].tolist() == [[e, 3]] * 4 and torch.equal(r["f.w"], torch.full((4, 1, 1), e))
        moved = (r["f.routed"] - base["f.routed"]).abs()
        assert moved[:, :-1].max() < 1e-8
        assert torch.equal(r["f.routed"], base["f.routed"]) if e == 2 else moved[:, -1].amax(-1).min() > 1e-3


def test_an_id_a_write_lands_that_is_no_expert_is_refused_by_name_before_the_model_reads_it(engines):
    """A generation's `expert_indices` read at decode calls 1 to 3 holds the hole, -1, on the rows that stopped
    first, and a swap of it into a forward's last 3 tokens is refused at its first hole, by entry, row and token;
    a step's ids moved past the 4 experts (`add 3`), and a float read (the router logits' `max`), are refused the same
    way at the first entry that is no id."""
    def swapped(steps: dict, operand: str, pos) -> Plan:
        write = {"site": {"component": "expert_indices", "layers": 1}, "pos": pos,
                 "ops": [{"op": "swap", "with": operand}]}
        return Plan(model=MIXTRAL, data={"prompts": {"path": "weekdays/train"}},
                    steps={**steps, "f": {"kind": "forward", "input": "prompts.input", "interventions": {"w": write}}},
                    saves=["f.w"], document=document("patching")).compile()
    generation = {"kind": "generate", "input": "prompts.input", "max_new_tokens": 4, "do_sample": False,
                  "eos_token_id": 5521,
                  "interventions": {"idx": at("expert_indices", 1, {"frame": "1:4", "index": ":3"})}}
    engine = engines("nnterp", Model.model_validate(MIXTRAL))
    with pytest.raises(ValueError, match=r"^f\.w: writes a hole, -1, into expert_indices at row 0, token 5, where its "
                                         r"operand holds no id \(a row that stopped before the call it was read at, "
                                         r"or a position past a row's own\); an id is a whole number from 0 to 3 \(the "
                                         r"model's num_experts, 4\), which the model indexes with$"):
        engine.execute(swapped({"g": generation}, "g.idx", {"index": "-3:"}), batch_size=BATCH_SIZE)
    clean = {"kind": "forward", "input": "prompts.input", "interventions": {
        "idx": at("expert_indices", 1, -1),
        "top": {**at("router_logits", 1, -1), "ops": [{"op": "max", "over": ["num_experts"]}]}}}
    past = {"clean": clean, "past": {"kind": "op", "op": "add", "of": "clean.idx", "with": 3}}
    with pytest.raises(ValueError, match=r"^f\.w: writes 5 into expert_indices at row 0, token 7, which is no id; "):
        engine.execute(swapped(past, "past", -1), batch_size=BATCH_SIZE)
    with pytest.raises(ValueError, match=r"^f\.w: writes 0\.\d+ into expert_indices at row 0, token 7, which is no "
                                         r"id; "):
        engine.execute(swapped({"clean": clean}, "clean.top", -1), batch_size=BATCH_SIZE)


# -- an id's hole is missing to a reduction ---------------------------------------------------------------------------


def test_a_reduction_of_a_ragged_expert_indices_read_leaves_its_hole_out():
    """On prompts of 11, 2, 11 and 3 tokens the read at every token holds -1 past each shorter row's own; `sum`,
    `mean`, `min`, `max`, `std`, `norm`, `dot` and `entropy` (over the slots) and `square` of it are each that op of
    the ids with the hole as NaN, to the bit, so a row's `min` is the least of its own experts, never -1, and its
    `sum` is its own ids' sum."""
    from clab.engines import load

    rows = [{"input": "If today is Monday, tomorrow is Tuesday and then"}, {"input": "Hi"},
            {"input": "One two three four five six seven eight nine ten"}, {"input": "Yes no"}]
    over = {"over": ["pos", "top_k"]}
    reduced = {"sum": over, "mean": over, "min": over, "max": over, "std": over, "norm": {"over": ["top_k"]},
               "dot": {"with": 2, "over": ["top_k"]}, "entropy": {"over": ["top_k"]}, "square": {}}
    entries = {"idx": at("expert_indices", 0)} | {op: {**at("expert_indices", 0), "ops": [{"op": op, **args}]}
                                                  for op, args in reduced.items()}
    plan = Plan(model=MIXTRAL, data={"prompts": {"rows": rows}},
                steps={"f": {"kind": "forward", "input": "prompts.input", "interventions": entries}},
                saves=[f"f.{name}" for name in entries], document=document("patching")).compile()
    r = load("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    idx = r["f.idx"]
    hole = idx == -1
    assert hole.any(-1).sum(-1).tolist() == [0, 9, 0, 8]
    floated = idx.float().masked_fill(hole, float("nan"))
    dims = ["rows", "pos", "top_k"]
    from clab import ops
    for op, args in reduced.items():
        assert same(r[f"f.{op}"], ops.call(op, floated, args, dims=dims)), op
    assert (r["f.min"] >= 0).all() and r["f.sum"].tolist() == [int(idx[i][~hole[i]].sum()) for i in range(4)]
    assert same(r["f.square"], floated.square())


def test_a_generations_per_call_ops_of_ids_agree_with_the_same_ops_over_the_whole_read(engines):
    """`square` and `min` over the slots run at each decode call, where a row that stopped is a missing position,
    NaN; as steps over the whole read they meet its hole, -1, which they now take as missing too, so the two give
    the same bits (where the steps gave 1 and -1 on the stopped rows)."""
    pos = {"frame": "1:4", "index": ":3"}
    generation = {"kind": "generate", "input": "prompts.input", "max_new_tokens": 4, "do_sample": False,
                  "eos_token_id": 5521, "interventions": {
                      "raw": at("expert_indices", 1, pos),
                      "square": {**at("expert_indices", 1, pos), "ops": [{"op": "square"}]},
                      "min": {**at("expert_indices", 1, pos), "ops": [{"op": "min", "over": ["top_k"]}]}}}
    steps = {"g": generation, "square": {"kind": "op", "op": "square", "of": "g.raw"},
             "min": {"kind": "op", "op": "min", "of": "g.raw", "over": ["top_k"]}}
    plan = Plan(model=MIXTRAL, data={"prompts": {"path": "weekdays/train"}}, steps=steps,
                saves=["g.raw", "g.square", "g.min", "square", "min"], document=document("patching")).compile()
    held = entries(points(plan.steps["g"])[0])
    assert held["square"].per_call == held["min"].per_call == 1
    r = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    stopped = (r["g.raw"] == -1).all(-1)
    assert stopped.any() and not stopped.all()
    assert same(r["g.square"], r["square"]) and same(r["g.min"], r["min"]) and r["min"][stopped].isnan().all()
