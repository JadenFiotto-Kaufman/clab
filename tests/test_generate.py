"""Generation: both generate documents' numbers pinned per engine at one batch size, the engines agreeing to
the bit and `remote="local"` too; and the numbers that pin what the generated frame means: index 0 is the
prompt call's last position, step k's logits chose token k, a scoped read resolves per row from what the row
said (missing, ambiguous, where it stopped), a step the run never made is missing, and `decode` stops at EOS.

The tiny Llama says the same six tokens after every weekday prompt ("Handler substr energcalc tossorter"),
and its generation stops on id 1 (its generation config's EOS, which is not its tokenizer's).
"""

import json
import os
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from clab import Plan, positions
from clab.plan import PlanError
from conftest import BATCH_SIZE, GENERATE, ROOT, close, edited, plain, plan_of, same, snapshot, stand_in

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM", "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    out = {}
    for name in GENERATE:
        for kind in ("nnterp", "hooks"):
            plan = plan_of(name)
            out[name, kind] = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    return out


def run(engines, plan: Plan, kind: str = "nnterp", **kwargs) -> Any:
    """`plan` executed on `kind` at the pinned batch size: the filled plan."""
    return engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE, **kwargs)


@pytest.mark.parametrize("name", GENERATE)
@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_numbers_match_the_snapshot(filled, name, kind):
    plan = filled[name, kind]
    values = {"results": plain(plan.results), "positions": plan.run.positions, "said": plan.run.tokens}
    expected = snapshot(f"{name}.{kind}.json", {"batch_size": BATCH_SIZE, "values": values})
    assert expected["batch_size"] == BATCH_SIZE
    close(values, expected["values"])


@pytest.mark.parametrize("name", GENERATE)
def test_the_engines_agree_to_the_bit(filled, name):
    a, b = filled[name, "nnterp"], filled[name, "hooks"]
    assert same(a.results, b.results)
    assert a.run.positions == b.run.positions and a.run.tokens == b.run.tokens


@pytest.mark.parametrize("name", GENERATE)
def test_remote_local_is_bit_equal_to_local(engines, filled, name):
    there = run(engines, plan_of(name), remote="local")
    assert there.run.remote == "local" and same(there.results, filled[name, "nnterp"].results)
    assert there.run.positions == filled[name, "nnterp"].run.positions


def test_generated_answer_scores_the_two_rows_that_said_their_word(filled):
    """The header: two rows score, two are `alignment_missing` under `at_said`, and `at_stop` is missing on
    every row, since `min_new_tokens` holds EOS off for all six steps."""
    plan = filled["generated_answer", "nnterp"]
    assert plan.run.positions["answer.at_said"] == [[1], [2], "alignment_missing", "alignment_missing"]
    assert plan.run.positions["answer.at_stop"] == ["alignment_missing"] * 4
    p = plan.results["p_answer"]
    assert p["row"].tolist() == [0, 1] and p["tok"].tolist() == [1, 2] and p["values"].shape == (2,)
    said = plan.run.tokens["answer"]
    assert [said[r][s] for r, s in ((0, 1), (1, 2))] == ["▁substr", "▁energ"]


def test_index_0_is_the_prompt_calls_last_position_and_each_step_chose_its_token(engines):
    """generate_probe.json's patched step over its three steps at once: step 0 is what a forward with the same
    patch reads at the prompt's last position, and every step's argmax is the token generated there. The
    final norm's output there is the forward's to the bit; the head's to an ulp, since a generation's prompt
    call projects the last position alone and a forward projects every position (a different matmul)."""
    def change(d):
        d["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "span": [0, 3]}
        d["steps"]["forward"] = {**d["steps"]["patched"], "kind": "forward",
                                 "reads": {"logits": {"site": "head", "pos": -1}}}
        for key in ("max_new_tokens", "min_new_tokens", "do_sample"):
            del d["steps"]["forward"][key]
        d["steps"]["patched"]["reads"]["resid"] = {"site": {"component": "final_norm_output"}, "pos": -1}
        d["steps"]["forward"]["reads"]["resid"] = {"site": {"component": "final_norm_output"}, "pos": -1}
        d["saves"] = ["patched", "patched.logits", "forward.logits", "patched.resid", "forward.resid"]
    plan = edited("generate_probe", change).compile()
    for kind in ("nnterp", "hooks"):
        values = run(engines, plan, kind).results
        logits = values["patched.logits"]
        assert logits.shape == (4, 3, 32000)
        assert torch.equal(values["patched.resid"], values["forward.resid"])
        torch.testing.assert_close(logits[:, :1], values["forward.logits"], atol=1e-6, rtol=0)
        assert torch.equal(logits.argmax(-1), values["patched"])


def test_a_row_that_never_said_its_word_is_missing_beside_its_text(filled, engines):
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "decode", "of": "answer"}
        d["saves"] = ["p_answer", "text"]
    plan = run(engines, edited("generated_answer", change).compile())
    text = plan.results["text"]
    assert text == ["Handler substr energcalc tossorter"] * 4
    rows = plan.table("p_answer", "text", columns=["said"])
    assert [(r["row"], r.get("step"), r.get("step_token"), r.get("alignment_missing")) for r in rows] == \
        [(0, 1, "▁substr", None), (1, 2, "▁energ", None), (2, None, None, True), (3, None, None, True)]
    assert all(r["text"] == text[0] for r in rows) and [r["said"] for r in rows][2:] == ["Friday", "Saturday"]
    assert "p_answer" not in rows[2] and rows[0]["p_answer"] == plan.results["p_answer"]["values"][0].item()


def test_a_word_said_twice_is_ambiguous(engines):
    """"er" occurs three times in what the model said (Handl-er, en-er-g, tossort-er); "subst" once."""
    def change(d):
        with open(os.path.join(ROOT, "data", "weekdays", "said.json")) as f:
            rows = json.load(f)
        d["data"]["prompts"] = {"rows": [{**rows[0], "said": "er"}, {**rows[1], "said": "subst"}]}
        d["saves"] = ["p_answer"]
    plan = run(engines, edited("generated_answer", change).compile())
    assert plan.run.positions["answer.at_said"] == ["alignment_ambiguous", [1]]
    assert [r.get("alignment_ambiguous") for r in plan.table("p_answer")] == [True, None]


def test_eos_is_where_each_row_stopped_and_decode_stops_before_it(engines):
    """With `forced_eos_token_id` (forwarded to `generate` as it is) every row's third token is EOS: `eos`
    resolves to step 2, and decode leaves it out. Without it (and without `min_new_tokens`) nothing changes."""
    def change(d):
        answer = d["steps"]["answer"]
        answer.update(max_new_tokens=3, forced_eos_token_id=1)
        del answer["min_new_tokens"]
        d["steps"]["text"] = {"kind": "op", "op": "decode", "of": "answer"}
        d["steps"]["pieces"] = {"kind": "op", "op": "decode", "of": "answer", "tokens": True}
        d["saves"] = ["answer", "text", "pieces"]
    plan = edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        values = run(engines, plan, kind)
        assert values.run.positions["answer.at_stop"] == [[2]] * 4
        assert values.results["answer"][:, 2].tolist() == [1] * 4
        assert values.results["text"] == ["Handler substr"] * 4
        assert values.results["pieces"] == [["Handler", "▁substr"]] * 4
        assert values.run.tokens["answer"][0] == ["Handler", "▁substr", "<s>"]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_step_the_run_never_made_is_missing(engines, kind):
    """A read at step 5 of a three-step generation: the loop asks for a step the run does not make, and every
    row is `alignment_missing`, its value NaN, the run going on."""
    def change(d):
        d["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 5}
        d["saves"] = ["p_answer", "patched"]
    plan = run(engines, edited("generate_probe", change).compile(), kind)
    assert plan.run.positions["patched.logits"] == ["alignment_missing"] * 4
    assert plan.results["p_answer"].isnan().all() and plan.results["patched"].shape == (4, 3)
    assert plan.table("p_answer")[0] == {"row": 0, "alignment_missing": True,
                                         "input": "If today is Thursday, tomorrow is", "cf_answer": " Sunday"}


def test_a_forwarded_option_sweeps(engines):
    def change(d):
        d["steps"]["patched"]["max_new_tokens"] = {"sweep": [2, 3]}
        d["steps"]["patched"]["min_new_tokens"] = 0
        d["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 0}
        d["saves"] = ["patched"]
    plan = run(engines, edited("generate_probe", change).compile())
    assert {k: v.shape for k, v in plan.results["patched"].items()} == {2: (4, 2), 3: (4, 3)}
    assert torch.equal(plan.results["patched"][3][:, :2], plan.results["patched"][2])


def test_char_spans_give_a_split_character_to_the_token_that_completes_it():
    """The pear is four byte tokens on this tokenizer: the first three decode, as prefixes, to `\ufffd`."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    ids = tokenizer("say 🍐 now", add_special_tokens=False)["input_ids"]
    assert tokenizer.convert_ids_to_tokens(ids) == ["▁say", "▁", "<0xF0>", "<0x9F>", "<0x8D>", "<0x90>", "▁now"]
    text, spans = positions.char_spans(tokenizer, ids)
    assert text == "say 🍐 now"
    assert [text[a:b] for a, b in spans] == ["say", " ", "", "", "", "🍐", " now"]
    every = {"frame": "generated", "scope": {"variable": "pear"}, "start": 0, "stop": None}
    assert positions.generated(every, ids, {1}, tokenizer, "🍐") == [2, 3, 4, 5]
    assert positions.generated({**every, "start": -1}, ids, {1}, tokenizer, "🍐") == [5]


def test_generated_positions_resolve_per_row():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    words = tokenizer(" alpha beta", add_special_tokens=False)["input_ids"]
    eos = {1}
    last = {"frame": "generated", "scope": None, "start": -1, "stop": None}
    stop = {**last, "scope": {"segment": "eos"}}
    beta = {**last, "scope": {"variable": "beta"}}
    assert positions.generated(last, [*words, 1, 0, 0], eos, tokenizer, None) == [len(words)]
    assert positions.generated(last, words, eos, tokenizer, None) == [len(words) - 1]
    assert positions.generated(stop, [*words, 1, 0], eos, tokenizer, None) == [len(words)]
    assert positions.generated(stop, words, eos, tokenizer, None) == "alignment_missing"
    assert positions.generated(beta, [*words, 1], eos, tokenizer, "beta") == [len(words) - 1]
    assert positions.generated(beta, [*words, 1], eos, tokenizer, "gamma") == "alignment_missing"
    assert positions.generated(beta, [*words, *words], eos, tokenizer, "beta") == "alignment_ambiguous"
    assert positions.generated(beta, [1, *words], eos, tokenizer, "beta") == "alignment_missing"   # after its EOS
    fixed = {"frame": "generated", "scope": None, "start": 2, "stop": 3}
    assert positions.generated(fixed, [5, 1, 0, 0], eos, tokenizer, None) == "alignment_missing"   # stopped first
    assert positions.generated(fixed, [5, 6, 7], eos, tokenizer, None) == [2]


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"frame": "generated", "index": 0}),
     r"steps\.patched\.writes\.patch\.pos: a generate step's writes land on the prompt"),
    (lambda d: d["steps"]["patched"]["reads"]["logits"].update(pos=-1),
     r"steps\.patched\.reads\.logits\.pos: a generation's prompt call computes lm_head_output at its last "
     r"position only"),
    (lambda d: d["steps"]["patched"]["reads"].update(pattern={
        "site": {"component": "attention_probabilities", "layers": 0}, "pos": {"frame": "generated", "index": 0}}),
     r"steps\.patched\.reads\.pattern\.site: attention_probabilities's key axis grows with every decode step"),
])
def test_compile_refuses_what_a_generation_cannot_mean(change, message):
    with pytest.raises(PlanError, match=message):
        edited("generate_probe", change).compile()


@pytest.mark.parametrize("option, message", [
    ({"num_return_sequences": 2}, "patched: num_return_sequences 2; it stays 1: more sequences than rows"),
    ({"num_beams": 2}, "patched: num_beams 2; it stays 1: beams change the rows each forward call sees"),
    ({"return_dict_in_generate": True}, "patched: return_dict_in_generate True; it stays False: the step's value"),
    ({"use_cache": False}, "patched: use_cache False; it stays True: without the cache a decode step recomputes"),
])
@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_options_a_step_cannot_forward_are_refused_before_any_forward(engines, kind, option, message):
    plan = edited("generate_probe", lambda d: d["steps"]["patched"].update(option)).compile()
    engine = engines(kind, plan.model)
    calls = []
    original = engine.generate
    engine.generate = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=message):
            engine.execute(plan)
    finally:
        del engine.generate
    assert calls == []


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_an_option_the_engine_rejects_fails_naming_the_step(engines, kind):
    """The engine's own `generate` validates the options it is forwarded, when the step runs."""
    plan = edited("generate_probe", lambda d: d["steps"]["patched"].update(max_new_toknes=3)).compile()
    with pytest.raises(ValueError, match=r"^patched: generate rejected 'max_new_toknes': The following "
                                         r"`model_kwargs` are not used by the model"):
        engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)


def test_a_steps_own_eos_token_id_is_where_its_rows_stop(engines):
    """`eos_token_id` is forwarded and is that step's EOS: 25148 (`▁substr`), the second token every row says,
    so every row stops at step 1, `eos` resolves there and decode stops before it."""
    def change(d):
        answer = d["steps"]["answer"]
        answer["eos_token_id"] = 25148
        del answer["min_new_tokens"]
        d["steps"]["text"] = {"kind": "op", "op": "decode", "of": "answer"}
        d["saves"] = ["answer", "text"]
    plan = edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        values = run(engines, plan, kind)
        assert values.run.positions["answer.at_stop"] == [[1]] * 4
        assert values.results["answer"].shape == (4, 2) and values.results["text"] == ["Handler"] * 4


def test_the_manifest_holds_each_rows_decode_step_and_what_it_said(filled, tmp_path):
    out = tmp_path / "probe"
    filled["generate_probe", "nnterp"].dump(str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["positions"]["patched.logits"] == {f"index={k}": [[k]] * 4 for k in range(3)}
    assert manifest["run"]["tokens"]["patched"]["index=0"][0] == ["Handler", "▁substr", "▁energ"]
    rows = Plan.load(str(out)).table("p_answer")
    assert [(r["at"], r["row"], r["step"], r["step_token"]) for r in rows][4] == ({"index": 1}, 0, 1, "▁substr")


@pytest.mark.parametrize("name", GENERATE)
def test_the_way_home_through_a_stand_in_server(engines, filled, monkeypatch, name):
    plan = plan_of(name)
    home, trip = stand_in(engines("nnterp", plan.model), plan, monkeypatch)
    assert trip == {"by_reference": True, "names": ["plan"], "types": ["Plan"]}
    assert home is not plan and same(home.results, filled[name, "nnterp"].results)
    assert home.run is not None and home.run.positions == filled[name, "nnterp"].run.positions


class Steer(LogitsProcessor):
    """Rows that say different things and stop at different steps, which the tiny Llama's own generation (the
    same six tokens after every prompt, no EOS) never does. Additive, so `min_new_tokens` still holds EOS off:
    the Thursday row is pushed to EOS from step 1, the Saturday row from step 3, and the Sunday row to
    `▁Friday` at step 0."""

    def __init__(self, tokenizer, width: int):
        self.tokenizer, self.width = tokenizer, width
        self.friday = tokenizer(" Friday", add_special_tokens=False)["input_ids"][0]

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.width
        for i, prompt in enumerate(self.tokenizer.batch_decode(input_ids[:, :self.width], skip_special_tokens=True)):
            if "Thursday" in prompt and step >= 1 or "Saturday" in prompt and step >= 3:
                scores[i, 1] += 1e4
            if "Sunday" in prompt and step == 0:
                scores[i, self.friday] += 1e4
        return scores


def steered(engines, kind: str, batch_size: int, held: bool) -> Any:
    """generated_answer.json steered by `Steer`, with EOS held off (`held`) or not, reading the head at steps 1
    and 2, layer 1 at step 0 and at the prompt's last token, and scoring step 2."""
    def change(d):
        answer = d["steps"]["answer"]
        if not held:
            del answer["min_new_tokens"]
        answer["reads"].update(
            at1={"site": "head", "pos": {"frame": "generated", "index": 1}},
            at2={"site": "head", "pos": {"frame": "generated", "index": 2}},
            resid0={"site": {"component": "layer_output", "layers": 1}, "pos": {"frame": "generated", "index": 0}},
            resid={"site": {"component": "layer_output", "layers": 1}, "pos": -1})
        d["steps"].update(iia={"kind": "op", "op": "match", "of": "answer.at2", "expected": "prompts.cf_answer"},
                          acc={"kind": "op", "op": "mean", "of": "iia"},
                          top={"kind": "op", "op": "argmax", "of": "answer.at2"},
                          text={"kind": "op", "op": "decode", "of": "answer"})
        d["saves"] = ["answer", "answer.at1", "answer.at2", "answer.resid0", "answer.resid", "iia", "acc", "top",
                      "text"]
    plan = edited("generated_answer", change).compile()
    engine = engines(kind, plan.model)
    width = max(len(engine.tokenizer(r["input"])["input_ids"]) for r in plan.data["prompts"].rows or [])
    step: Any = plan.steps["answer"]
    step.__pydantic_extra__["logits_processor"] = LogitsProcessorList([Steer(engine.tokenizer, width)])
    return engine.execute(plan, batch_size=batch_size)


def missing_as(value: Any) -> Any:
    """A value with NaN as 7, so two values with the same missing rows compare to the bit."""
    if isinstance(value, torch.Tensor):
        return value.nan_to_num(7.0) if value.is_floating_point() else value
    return {k: missing_as(v) for k, v in value.items()} if isinstance(value, dict) else value


def test_rows_that_stop_at_different_steps(engines):
    """Each row's ids are its own, a row that stopped at step 1 is missing at step 2 and not at step 1, and is
    missing all the way down (NaN in `match`, -1 from `argmax`, left out of the mean); the decode call is read
    at its last position; on both engines and at batch sizes 1, 2 and 4, to the bit."""
    runs = {(kind, size): steered(engines, kind, size, held=False) for kind in ("nnterp", "hooks") for size in (1, 2, 4)}
    plan = runs["nnterp", 2]
    v, positions = plan.results, plan.run.positions
    assert v["answer"].tolist() == [[4598, 1, 1, 1, 1, 1], [4598, 25148, 18190, 28667, 28189, 9555],
                                    [4598, 25148, 18190, 1, 1, 1], [28728, 5117, 1325, 13194, 21367, 28630]]
    assert v["text"] == ["Handler", "Handler substr energcalc tossorter", "Handler substr energ",
                         "FridayreichсиhovervelopeAlpha"]
    assert positions["answer.at_stop"] == [[1], "alignment_missing", [3], "alignment_missing"]
    assert positions["answer.at1"] == [[1]] * 4
    assert positions["answer.at2"] == ["alignment_missing", [2], [2], [2]]
    assert not v["answer.at1"].isnan().any() and v["answer.at2"][0].isnan().all() and not v["answer.at2"][1:].isnan().any()
    assert v["iia"][0].isnan().all() and v["top"][:, 0].tolist() == [-1, 18190, 18190, 1325]
    assert torch.equal(v["acc"], v["iia"][1:].mean())
    assert torch.equal(v["answer.resid0"], v["answer.resid"])          # step 0 is the prompt call's last token
    for kind, size in runs:
        other = runs[kind, size]
        assert other.run.positions == positions and other.run.tokens == plan.run.tokens, (kind, size)
        assert torch.equal(other.results["answer"], v["answer"]), (kind, size)
        assert same(missing_as(other.results), missing_as(runs["nnterp", size].results)), (kind, size)
    rows = plan.table("iia", "top")
    assert rows[0] == {"row": 0, "alignment_missing": True, "example_id": "substr", "input": rows[0]["input"],
                       "cf_answer": " Sunday"} and rows[1]["top"] == 18190


def test_min_new_tokens_holds_eos_off(engines):
    for kind in ("nnterp", "hooks"):
        plan = steered(engines, kind, 2, held=True)
        assert 1 not in plan.results["answer"].flatten().tolist()
        assert plan.run.positions["answer.at_stop"] == ["alignment_missing"] * 4
        assert plan.run.positions["answer.at2"] == [[2]] * 4
