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
from transformers import AutoTokenizer

from clab import Plan, positions
from clab.plan import PlanError
from conftest import BATCH_SIZE, GENERATE, ROOT, close, edited, plain, plan_of, same, snapshot

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
    ({"num_return_sequences": 2}, "patched: num_return_sequences 2; a generate step's value is one sequence"),
    ({"num_beams": 2}, "patched: num_beams 2;"),
    ({"return_dict_in_generate": True}, "patched: return_dict_in_generate True;"),
    ({"eos_token_id": 5}, r"patched: eos_token_id; a row stops at the engine's EOS \(\[1\]\)"),
])
def test_options_that_change_the_rows_or_the_value_are_refused_before_any_forward(engines, option, message):
    plan = edited("generate_probe", lambda d: d["steps"]["patched"].update(option)).compile()
    engine = engines("hooks", plan.model)
    with pytest.raises(ValueError, match=message):
        engine.execute(plan)


def test_the_manifest_holds_each_rows_decode_step_and_what_it_said(filled, tmp_path):
    out = tmp_path / "probe"
    filled["generate_probe", "nnterp"].dump(str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["positions"]["patched.logits"] == {f"index={k}": [[k]] * 4 for k in range(3)}
    assert manifest["run"]["tokens"]["patched"]["index=0"][0] == ["Handler", "▁substr", "▁energ"]
    rows = Plan.load(str(out)).table("p_answer")
    assert [(r["at"], r["row"], r["step"], r["step_token"]) for r in rows][4] == ({"index": 1}, 0, 1, "▁substr")
