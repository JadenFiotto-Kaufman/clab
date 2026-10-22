"""Generation: both generate documents' numbers pinned per engine at one batch size, the engines agreeing to
the bit and `remote="local"` too; and the numbers that pin what a position means across calls: the prefill's
last position is the forward's, decode call k holds generated token k-1 and its logits chose token k, a
scoped read resolves per row from what the row said (which occurrence, missing, where it stopped), a call the run
never made is missing, a write at a decode call changes the tokens from that call on, and `detokenize` stops at
EOS.

The tiny Llama says the same six tokens after every weekday prompt ("Handler substr energcalc tossorter"),
and its generation stops on id 1 (its generation config's EOS, which is not its tokenizer's). The weekday
prompts are 11, 9, 9 and 9 tokens long, BOS included, so generated token j of row r sits at position
`P_r + j`: 11 + j on row 0, 9 + j on the others.
"""

import json
import os
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from clab import Plan
from clab.plan import PlanError
from clab.runtime import positions
from conftest import BATCH_SIZE, GENERATE, ROOT, TRIP, close, edited, plain, plan_of, same, snapshot, stand_in

PROMPT = [11, 9, 9, 9]                  # each weekday prompt's tokens, BOS included
# Two texts the model says on every row, at tokens 0 and 2 (`Handler`, `▁energ`): a scope's two occurrences in
# what it said, each a whole token, since an occurrence starts and ends at token boundaries.
TWICE = ["Handler", "energ"]
LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM", "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


@pytest.fixture(scope="module")
def filled(engines) -> dict:
    out = {}
    for name in GENERATE:
        for kind in ("nnterp", "hooks"):
            plan = plan_of(name)
            out[name, kind] = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    return out


def run(engines, plan: Plan, kind: str = "nnterp", **options) -> Any:
    """`plan` executed on `kind` (loaded with `options`) at the pinned batch size: the filled plan."""
    return engines(kind, plan.model, **options).execute(plan, batch_size=BATCH_SIZE)


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
    there = run(engines, plan_of(name), "ndif", remote="local")
    assert there.run.options == {"remote": "local"} and same(there.results, filled[name, "nnterp"].results)
    assert there.run.positions == filled[name, "nnterp"].run.positions


def test_generated_answer_scores_the_two_rows_that_said_their_word(filled):
    """The description: two rows score, two are `alignment_missing` under `at_said`, and `at_last` is each
    row's sixth call, 5, since `min_new_tokens` holds EOS off for all six tokens. The read sits at the
    word's last token where it is an input: row 0 said `▁substr` as token 1 (position 12, call 2), row 1
    `▁energ` as token 2 (position 11, call 3)."""
    plan = filled["generated_answer", "nnterp"]
    assert plan.run.positions["answer.at_said"] == [[12], [11], "alignment_missing", "alignment_missing"]
    assert plan.run.positions["answer.at_last"] == [[p + 4] for p in PROMPT]
    p = plan.results["p_answer"]
    assert p["row"].tolist() == [0, 1] and p["tok"].tolist() == [12, 11] and p["values"].shape == (2,)
    said = plan.run.tokens["answer"]
    assert [said[0][1], said[1][2]] == ["▁substr", "▁energ"]


def test_call_k_holds_token_k_minus_1_and_its_logits_chose_token_k(engines):
    """generate_probe.json's patched step: the prefill's last position is what a forward with the same patch
    reads there (the final norm to the bit; the head to an ulp, since a generation's prefill projects the
    last position alone and a forward every position, a different matmul); the head at calls 0, 1, 2 chose
    tokens 0, 1, 2; and the embedding at calls 1 and 2 is that of tokens 0 and 1, their inputs."""
    def change(d):
        entries = d["steps"]["patched"]["interventions"]
        entries["logits"]["pos"] = {"frame": 0, "index": -1}
        entries["later"] = {"site": "head", "pos": {"frame": "1:3", "index": "0:2"}}
        entries["inputs"] = {"site": {"component": "token_embeddings"}, "pos": {"frame": "1:3", "index": "0:2"}}
        entries["resid"] = {"site": {"component": "final_norm_output"}, "pos": -1}
        d["steps"]["forward"] = {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": entries["patch"], "logits": {"site": "head", "pos": -1},
            "resid": {"site": {"component": "final_norm_output"}, "pos": -1}}}
        d["saves"] = ["patched", "patched.logits", "patched.later", "patched.inputs", "patched.resid",
                      "forward.logits", "forward.resid"]
    plan = edited("generate_probe", change).compile()
    for kind in ("nnterp", "hooks"):
        engine = engines(kind, plan.model)
        values = engine.execute(plan, batch_size=BATCH_SIZE).results
        assert torch.equal(values["patched.resid"], values["forward.resid"])
        torch.testing.assert_close(values["patched.logits"], values["forward.logits"], atol=1e-6, rtol=0)
        chose = torch.cat([values["patched.logits"], values["patched.later"]], 1).argmax(-1)
        assert torch.equal(chose, values["patched"])
        if kind == "hooks":
            table = engine.model.get_input_embeddings().weight
            assert torch.equal(values["patched.inputs"], table[values["patched"][:, :2]])


def test_a_row_that_never_said_its_word_is_missing_beside_its_text(filled, engines):
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "detokenize", "of": "answer"}
        d["saves"] = ["p_answer", "text"]
    plan = run(engines, edited("generated_answer", change).compile())
    text = plan.results["text"]
    assert text == ["Handler substr energcalc tossorter"] * 4
    rows = plan.table("p_answer", "text", columns=["said"])
    assert [(r["row"], r.get("pos"), r.get("pos_token"), r.get("call"), r.get("alignment_missing")) for r in rows] \
        == [(0, 12, "▁substr", 2, None), (1, 11, "▁energ", 3, None), (2, None, None, None, True),
            (3, None, None, None, True)]
    assert all(r["text"] == text[0] for r in rows) and [r["said"] for r in rows][2:] == ["Friday", "Saturday"]
    assert "p_answer" not in rows[2] and rows[0]["p_answer"] == plan.results["p_answer"]["values"][0].item()


def test_a_word_said_twice_takes_the_occurrence_asked_for(engines):
    """The set `["Handler", "energ"]` ends twice on an input in what the model said (`Handler`, token 0;
    `▁energ`, token 2); "substr" once. Occurrence 0 is the first, occurrence 1 the
    second, which a row that says it once does not have."""
    def change(occurrence):
        def changed(d):
            with open(os.path.join(ROOT, "data", "weekdays", "said.json")) as f:
                rows = json.load(f)
            d["data"]["prompts"] = {"rows": [{**rows[0], "said": TWICE}, {**rows[1], "said": "substr"}]}
            d["steps"]["answer"]["interventions"]["at_said"]["pos"]["scope"]["occurrence"] = occurrence
            d["saves"] = ["p_answer"]
        return changed
    first = run(engines, edited("generated_answer", change(0)).compile())
    assert first.run.positions["answer.at_said"] == [[11], [10]]
    second = run(engines, edited("generated_answer", change(1)).compile())
    assert second.run.positions["answer.at_said"] == [[13], "alignment_missing"]
    assert [r.get("alignment_missing") for r in second.table("p_answer")] == [None, True]


def test_the_last_call_is_the_one_that_chose_eos_and_detokenize_stops_before_it(engines):
    """With `forced_eos_token_id` (forwarded to `generate` as it is) every row's third token is EOS: frame -1 is
    call 2, the one that chose it, whose one position is token 1, and detokenize leaves the EOS out."""
    def change(d):
        answer = d["steps"]["answer"]
        answer.update(max_new_tokens=3, forced_eos_token_id=1)
        del answer["min_new_tokens"]
        d["steps"]["text"] = {"kind": "op", "op": "detokenize", "of": "answer"}
        d["steps"]["pieces"] = {"kind": "op", "op": "detokenize", "of": "answer", "tokens": True}
        d["saves"] = ["answer", "text", "pieces"]
    plan = edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        values = run(engines, plan, kind)
        assert values.run.positions["answer.at_last"] == [[p + 1] for p in PROMPT]
        assert values.results["answer"][:, 2].tolist() == [1] * 4
        assert values.results["text"] == ["Handler substr"] * 4
        assert values.results["pieces"] == [["Handler", "▁substr"]] * 4
        assert values.run.tokens["answer"][0] == ["Handler", "▁substr", "<s>"]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_call_the_run_never_made_is_missing(engines, kind):
    """A read at call 5 of a three-token generation: the loop asks for a call the run does not make, and
    every row is `alignment_missing`, its value NaN, the run going on."""
    def change(d):
        d["steps"]["patched"]["interventions"]["logits"]["pos"] = {"frame": 5, "index": -1}
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
        d["steps"]["patched"]["interventions"]["logits"]["pos"] = {"frame": 0, "index": -1}
        d["saves"] = ["patched"]
    plan = run(engines, edited("generate_probe", change).compile())
    assert {k: v.shape for k, v in plan.results["patched"].items()} == {2: (4, 2), 3: (4, 3)}
    assert torch.equal(plan.results["patched"][3][:, :2], plan.results["patched"][2])


def prompt_of(ids: list[int]) -> dict:
    """An encoded prompt of `ids` (`data.encode`'s shape), its first token BOS, for resolving by hand."""
    return {"ids": ids, "offsets": [(0, 0)] + [(0, 1)] * (len(ids) - 1), "text": "x", "turns": []}


def resolved(pos: dict, said: list[int], tokenizer, text: str | None = None, prompt: int = 2):
    """`pos` (defaults filled) resolved in a row of a `prompt`-token prompt that generated `said`, EOS 1."""
    pos = {"frame": 0, "scope": None, "index": ":", **pos}
    row, said = prompt_of([1] + [5] * (prompt - 1)), positions.continuation(said, {1})
    found = positions.occurrences(pos["scope"], row, said, tokenizer, [(text, False, False)] if text else [])
    return positions.resolve(pos, prompt, list(range(1, prompt)), said, found)


def test_char_spans_give_a_split_character_to_the_token_that_completes_it():
    """The pear is four byte tokens on this tokenizer: the first three decode, as prefixes, to `\ufffd`."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    ids = tokenizer("say 🍐 now", add_special_tokens=False)["input_ids"]
    assert tokenizer.convert_ids_to_tokens(ids) == ["▁say", "▁", "<0xF0>", "<0x9F>", "<0x8D>", "<0x90>", "▁now"]
    text, spans = positions.char_spans(tokenizer, ids)
    assert text == "say 🍐 now"
    assert [text[a:b] for a, b in spans] == ["say", " ", "", "", "", "🍐", " now"]
    pear = {"frame": "1:", "scope": {"text": "🍐"}}
    assert resolved(pear, ids, tokenizer, "🍐") == [4, 5, 6, 7]          # tokens 2..5, after a 2-token prompt
    assert resolved({**pear, "index": -1}, ids, tokenizer, "🍐") == [7]


def test_each_stage_narrows_the_one_before():
    """The three stages by hand, on a two-token prompt (BOS, then position 1) and what a row generated: the
    calls a frame picks (a row's calls end with the one that chose its last token, so that token is never an
    input), the text a scope finds in their tokens, the positions an index picks."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    words = tokenizer(" alpha beta", add_special_tokens=False)["input_ids"]
    n = len(words)
    assert resolved({"frame": 0, "index": -1}, [], tokenizer) == [1]                     # BOS is never a position
    assert resolved({"frame": 1}, words, tokenizer) == [2]                               # call 1 holds token 0
    assert resolved({"frame": -1}, words, tokenizer) == [n]                              # the row's last call
    assert resolved({"frame": "1:"}, [*words, 1], tokenizer) == list(range(2, 2 + n))
    assert resolved({"frame": ":"}, [*words, 1], tokenizer) == list(range(1, 2 + n))
    assert resolved({"frame": "1:3", "index": 1}, [*words, 1], tokenizer) == [3]
    assert resolved({"frame": -1}, [*words, 1, 0], tokenizer) == [1 + n]                 # the call that chose EOS
    assert resolved({"frame": -1}, [*words, 7], tokenizer) == [1 + n]                    # or ran out at its last
    assert resolved({"frame": n + 1, "index": -1}, words, tokenizer) == "alignment_missing"
    beta = {"frame": "1:", "scope": {"text": "beta"}, "index": -1}
    assert resolved(beta, [*words, 1], tokenizer, "beta") == [1 + n]
    assert resolved(beta, [*words, 1], tokenizer, "gamma") == "alignment_missing"
    twice = [*words, *words, 1]                                                          # beta at positions 3 and 5
    assert resolved(beta, twice, tokenizer, "beta") == [3]
    assert resolved({**beta, "scope": {"text": "beta", "occurrence": -1}}, twice, tokenizer, "beta") == [5]
    every = {"frame": "1:", "scope": {"text": "beta", "occurrence": ":"}}
    assert resolved(every, twice, tokenizer, "beta") == [3, 5]
    assert resolved({**every, "index": -1}, twice, tokenizer, "beta") == [5]
    assert resolved({**beta, "scope": {"text": "beta", "occurrence": 2}}, twice, tokenizer, "beta") == \
        "alignment_missing"
    assert resolved(beta, [1, *words], tokenizer, "beta") == "alignment_missing"         # said after its EOS
    assert resolved(beta, words, tokenizer, "beta") == "alignment_missing"    # the last token, never an input
    assert resolved({"frame": 3, "index": -1}, [5, 1, 0, 0], tokenizer) == "alignment_missing"   # stopped first
    assert resolved({"frame": 2, "index": -1}, [5, 6, 7], tokenizer) == [3]


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": -1, "index": -1}),
     r"steps\.patched\.interventions\.patch\.pos: frame -1 with index -1 names calls that depend on how far each row"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": 2, "scope": {"text": "is"}}),
     r"steps\.patched\.interventions\.patch\.pos\.index: index ':': a scoped write lands at the call fed its text's"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": "1:", "index": -1}),
     r"steps\.patched\.interventions\.patch\.pos: frame '1:' with index -1 names calls that depend on how far"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": "1:4", "index": -1}),
     r"steps\.patched\.interventions\.patch\.pos: frame '1:4' with index -1 names calls that depend on how far"),
    (lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": ":", "index": 0}),
     r"steps\.patched\.interventions\.patch\.pos: frame ':' with index 0 names calls that depend on how far"),
    (lambda d: d["steps"]["patched"]["interventions"].update(pattern={
        "site": {"component": "attention_probabilities", "layers": 0}, "pos": {"frame": 1, "index": -1}}),
     r"steps\.patched\.interventions\.pattern\.site: attention_probabilities's key axis grows with every decode call"),
    (lambda d: d["steps"]["patched"]["interventions"]["logits"].update(pos={"frame": 1, "scope": {"column": "pairs.said"}}),
     r"steps\.patched\.interventions\.logits\.pos\.scope\.column: dataset 'pairs' has no column 'said'"),
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
    so every row stops at call 1, whose one position is token 0, and detokenize stops before it."""
    def change(d):
        answer = d["steps"]["answer"]
        answer["eos_token_id"] = 25148
        del answer["min_new_tokens"]
        d["steps"]["text"] = {"kind": "op", "op": "detokenize", "of": "answer"}
        d["saves"] = ["answer", "text"]
    plan = edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        values = run(engines, plan, kind)
        assert values.run.positions["answer.at_last"] == [[p] for p in PROMPT]
        assert values.results["answer"].shape == (4, 2) and values.results["text"] == ["Handler"] * 4


def test_the_manifest_holds_each_rows_positions_and_what_it_said(filled, tmp_path):
    out = tmp_path / "probe"
    filled["generate_probe", "nnterp"].dump(str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run"]["positions"]["patched.logits"] == {f"frame={k}": [[p - 1 + k] for p in PROMPT]
                                                              for k in range(3)}
    assert manifest["run"]["tokens"]["patched"]["frame=0"][0] == ["Handler", "▁substr", "▁energ"]
    rows = Plan.load(str(out)).table("p_answer")
    assert [(r["at"], r["row"], r["pos"], r["pos_token"], r["call"]) for r in rows][4] == \
        ({"frame": 1}, 0, 11, "Handler", 1)


@pytest.mark.parametrize("name", GENERATE)
def test_the_way_home_through_a_stand_in_server(engines, filled, monkeypatch, name):
    plan = plan_of(name)
    home, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP
    assert home is plan and same(home.results, filled[name, "nnterp"].results)    # what came home, in the client's plan
    assert home.run is not None and home.run.positions == filled[name, "nnterp"].run.positions


class Steer(LogitsProcessor):
    """Rows that say different things and stop at different calls, which the tiny Llama's own generation (the
    same six tokens after every prompt, no EOS) never does. Additive, so `min_new_tokens` still holds EOS off:
    the Thursday row is pushed to EOS from token 1 on (it stops after two tokens, calls 0 and 1), the
    Saturday row from token 3, and the Sunday row to `▁Friday` as token 0."""

    def __init__(self, tokenizer, width: int):
        self.tokenizer, self.width = tokenizer, width
        self.friday = tokenizer(" Friday", add_special_tokens=False)["input_ids"][0]

    def __call__(self, input_ids, scores):
        token = input_ids.shape[1] - self.width
        for i, prompt in enumerate(self.tokenizer.batch_decode(input_ids[:, :self.width], skip_special_tokens=True)):
            if "Thursday" in prompt and token >= 1 or "Saturday" in prompt and token >= 3:
                scores[i, 1] += 1e4
            if "Sunday" in prompt and token == 0:
                scores[i, self.friday] += 1e4
        return scores


def steered(engines, kind: str, batch_size: int, held: bool, change=None, **options) -> Any:
    """generated_answer.json steered by `Steer`, with EOS held off (`held`) or not, reading the head at calls
    1 and 2, layer 1 at the prompt's last token from inside a read beyond the prefill and from the prefill,
    and scoring call 2; `change` edits the document further."""
    def changed(d):
        answer = d["steps"]["answer"]
        if not held:
            del answer["min_new_tokens"]
        answer["interventions"].update(
            at1={"site": "head", "pos": {"frame": 1, "index": -1}},
            at2={"site": "head", "pos": {"frame": 2, "index": -1}},
            resid0={"site": {"component": "layer_output", "layers": 1}, "pos": {"frame": "0:2", "index": -2}},
            resid={"site": {"component": "layer_output", "layers": 1}, "pos": -1})
        d["steps"].update(iia={"kind": "op", "op": "match", "of": "answer.at2", "expected": "prompts.cf_answer"},
                          acc={"kind": "op", "op": "mean", "of": "iia"},
                          top={"kind": "op", "op": "argmax", "of": "answer.at2"},
                          text={"kind": "op", "op": "detokenize", "of": "answer"})
        d["saves"] = ["answer", "answer.at1", "answer.at2", "answer.resid0", "answer.resid", "iia", "acc", "top",
                      "text"]
        if change:
            change(d)
    plan = edited("generated_answer", changed).compile()
    engine = engines(kind, plan.model, **options)
    width = max(len(engine.tokenizer(r["input"])["input_ids"]) for r in plan.data["prompts"].rows or [])
    for entry in plan.steps.values():
        step: Any = entry
        if getattr(step, "kind", None) == "generate":
            step.__pydantic_extra__["logits_processor"] = LogitsProcessorList([Steer(engine.tokenizer, width)])
    return engine.execute(plan, batch_size=batch_size)


def missing_as(value: Any) -> Any:
    """A value with NaN as 7, so two values with the same missing rows compare to the bit."""
    if isinstance(value, torch.Tensor):
        return value.nan_to_num(7.0) if value.is_floating_point() else value
    return {k: missing_as(v) for k, v in value.items()} if isinstance(value, dict) else value


def test_rows_that_stop_at_different_calls(engines):
    """Each row's ids are its own; a row that stopped after call 1 is missing at call 2 and not at call 1, and
    missing all the way down (NaN in `match`, -1 from `argmax`, left out of the mean); a read beyond the
    prefill takes a prompt position from the prefill's own; on both engines and at batch sizes 1, 2 and 4, to
    the bit."""
    runs = {(kind, size): steered(engines, kind, size, held=False)
            for kind in ("nnterp", "hooks") for size in (1, 2, 4)}
    plan = runs["nnterp", 2]
    v, positions = plan.results, plan.run.positions
    assert v["answer"].tolist() == [[4598, 1, 1, 1, 1, 1], [4598, 25148, 18190, 28667, 28189, 9555],
                                    [4598, 25148, 18190, 1, 1, 1], [28728, 5117, 1325, 13194, 21367, 28630]]
    assert v["text"] == ["Handler", "Handler substr energcalc tossorter", "Handler substr energ",
                         "FridayreichсиhovervelopeAlpha"]
    assert positions["answer.at_last"] == [[11], [13], [11], [13]]     # stopped at calls 1 and 3, or ran to 5
    assert positions["answer.at1"] == [[p] for p in PROMPT]
    assert positions["answer.at2"] == ["alignment_missing", [10], [10], [10]]
    assert not v["answer.at1"].isnan().any() and v["answer.at2"][0].isnan().all()
    assert not v["answer.at2"][1:].isnan().any()
    assert v["iia"][0].isnan().all() and v["top"][:, 0].tolist() == [-1, 18190, 18190, 1325]
    assert torch.equal(v["acc"], v["iia"][1:].mean())
    assert torch.equal(v["answer.resid0"], v["answer.resid"])          # the prompt's last token, either way
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
        assert plan.run.positions["answer.at_last"] == [[p + 4] for p in PROMPT]
        assert plan.run.positions["answer.at2"] == [[p + 1] for p in PROMPT]


WRITE = {"site": {"component": "layer_output", "layers": 0}, "pos": {"frame": 2, "index": -1}}


def written(d, held: bool = True):
    """generated_answer.json as a clean generation beside two that write at decode call 2, a swap of zeros
    and an add of the prompts' mean last-token vector (scaled up), each keeping its ids."""
    answer = {k: v for k, v in d["steps"]["answer"].items() if k != "interventions"}
    if not held:
        answer.pop("min_new_tokens", None)
    d["steps"] = {
        "harvest": {"kind": "forward", "input": "prompts.input", "interventions": {"acts": {**WRITE, "pos": -1}}},
        "mean": {"kind": "op", "op": "mean", "of": "harvest.acts"},
        "clean": answer,
        "swapped": {**answer, "interventions": {"w": {**WRITE, "ops": [{"op": "swap", "with": 0.0}]}}},
        "added": {**answer, "interventions": {"w": {**WRITE, "ops": [{"op": "add", "with": "mean", "scale": 50.0}]}}}}
    d["saves"] = ["clean", "swapped", "added"]


def test_a_write_at_a_decode_call_changes_the_tokens_from_there_on(engines):
    """A swap and an add at call 2 on layer 0's output edit generated token 1 where it is an input: tokens 0
    and 1 are as without the write and token 2 on are not; both engines at batch sizes 1, 2 and 4 and
    `remote="local"` give the same ids, to the bit, and the manifest records the written position."""
    plan = edited("generated_answer", written).compile()
    runs: dict = {(kind, size): engines(kind, plan.model).execute(edited("generated_answer", written).compile(),
                                                            batch_size=size)
            for kind in ("nnterp", "hooks") for size in (1, 2, 4)}
    runs["local"] = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    v = runs["nnterp", 2].results
    for key in ("swapped", "added"):
        assert torch.equal(v[key][:, :2], v["clean"][:, :2]) and (v[key][:, 2] != v["clean"][:, 2]).any(), key
        assert runs["nnterp", 2].run.positions[f"{key}.w"] == [[p + 1] for p in PROMPT]
    for where, other in runs.items():
        assert same(other.results, v), where


def test_a_write_at_a_call_a_row_never_reaches_skips_that_row(engines):
    """The Thursday row stops after call 1, so a write at call 2 has nothing of it to write: it is recorded
    missing there and its ids are those of the clean run; the other rows are written, on both engines."""
    for kind in ("nnterp", "hooks"):
        plan = steered(engines, kind, BATCH_SIZE, held=False, change=lambda d: written(d, held=False))
        v, where = plan.results, plan.run.positions
        assert where["swapped.w"] == ["alignment_missing", [10], [10], [10]], kind
        assert torch.equal(v["swapped"][0], v["clean"][0]) and v["clean"][0].tolist()[:2] == [4598, 1]
        assert (v["swapped"][1:, 2] != v["clean"][1:, 2]).any()


def test_a_decode_call_writes_operand_is_checked_before_any_forward(engines):
    """A decode call holds one position, so an operand read three tokens wide cannot be written there: refused
    by name before the generation, not as a size error inside it."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = {"index": "-4:-1"}
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = {"frame": 2, "index": -1}
    plan = edited("generate_probe", change).compile()
    engine = engines("hooks", plan.model)
    calls = []
    original = engine.generate
    engine.generate = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=r"^patched\.patch: operand 'counterfactual\.v_cf' is 3 token\(s\) wide"):
            engine.execute(plan)
    finally:
        del engine.generate
    assert calls == []


def test_a_text_scope_counts_the_prompt_only_where_the_frame_holds_it():
    """`beta` generated as token 1 sits at position 3 after a two-token prompt. A frame of every call finds it
    there and no prompt position; with `beta` in the prompt too, every call's tokens hold two occurrences,
    the prompt's first, while the generated calls alone hold one."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    words = tokenizer(" alpha beta", add_special_tokens=False)["input_ids"]
    every = {"frame": ":", "scope": {"text": "beta"}}
    assert resolved(every, [*words, 1], tokenizer, "beta") == [3]
    row, said = {"ids": [1, 5], "offsets": [(0, 0), (0, 4)], "text": "beta", "turns": []}, [*words, 1]

    def found(pos):
        pos = {"frame": 0, "scope": None, "index": ":", **pos}
        hits = positions.occurrences(pos["scope"], row, said, tokenizer, [("beta", False, False)])
        return positions.resolve(pos, 2, [1], said, hits)
    assert found(every) == [1]
    assert found({**every, "scope": {"text": "beta", "occurrence": -1}}) == [3]
    assert found({**every, "scope": {"text": "beta", "occurrence": ":"}}) == [1, 3]
    assert found({**every, "frame": "1:"}) == [3]


def test_a_span_of_calls_with_every_position_is_ragged(engines):
    """`{"frame": "1:4"}` reads calls 1 to 3 of each row: the Thursday row stopped after call 1, so it has
    one entry and the others three."""
    plan = steered(engines, "nnterp", BATCH_SIZE, held=False, change=lambda d: (
        d["steps"]["answer"]["interventions"].update(span={"site": "head", "pos": {"frame": "1:4"}}),
        d["saves"].append("answer.span")))
    assert plan.run.positions["answer.span"] == [[11], [9, 10, 11], [9, 10, 11], [9, 10, 11]]
    value = plan.results["answer.span"]
    assert value["row"].tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert value["tok"].tolist() == [11, 9, 10, 11, 9, 10, 11, 9, 10, 11] and value["values"].shape == (10, 32000)
    assert torch.equal(value["values"][[0, 1]], plan.results["answer.at1"][:2, 0])     # call 1 of rows 0 and 1


def test_picked_calls_and_picked_tokens_keep_the_order_written(engines):
    """`{"frame": [3, 1], "index": [1, 0]}` holds call 3's input then call 1's, and picks the second then the
    first: positions 9 then 11 (a 9-token prompt), rectangular, call 1 then call 3; and a slice with a step
    over the generated calls, `{"frame": "1:6:2"}`, is calls 1, 3 and 5."""
    plan = steered(engines, "nnterp", BATCH_SIZE, held=False, change=lambda d: (
        d["steps"]["answer"]["interventions"].update(picked={"site": "head", "pos": {"frame": [3, 1], "index": [1, 0]}},
                                             stepped={"site": "head", "pos": {"frame": "1:6:2", "index": [0, 1]}}),
        d["saves"].extend(["answer.picked", "answer.stepped"])))
    assert plan.run.positions["answer.picked"][1:] == [[9, 11]] * 3
    assert plan.run.positions["answer.stepped"][1:] == [[9, 11]] * 3
    value = plan.results["answer.picked"]
    assert value.shape == (4, 2, 32000) and value[0].isnan().all()      # the row that stopped after call 1
    assert torch.equal(value[1:], plan.results["answer.stepped"][1:])
    assert torch.equal(value[1:2, 0], plan.results["answer.at1"][1:2, 0])


@pytest.mark.parametrize("pos, calls", [
    ({"frame": "2:"}, [2, 3, 4, 5]), ({"frame": "3:"}, [3, 4, 5]), ({"frame": "1::2"}, [1, 3, 5]),
    ({"frame": "1:", "index": "2:"}, [3, 4, 5]), ({"frame": "1:", "index": [0, 2]}, [1, 3])])
def test_a_write_to_the_rows_last_call_lands_at_each_call_up_to_max_new_tokens(engines, pos, calls):
    """A frame not counted from the end names call `k` on every row that makes it; up to `max_new_tokens`
    (6: calls 0 to 5) its calls are listed before the run, and the write is the one at those calls, to the bit,
    on both engines."""
    def at(where):
        def change(d):
            written(d)
            d["steps"]["swapped"]["interventions"]["w"]["pos"] = where
        return edited("generated_answer", change).compile()
    runs = {}
    for kind in ("nnterp", "hooks"):
        open_ended = engines(kind, at(pos).model).execute(at(pos), batch_size=BATCH_SIZE)
        listed = engines(kind, at({"frame": calls}).model).execute(at({"frame": calls}), batch_size=BATCH_SIZE)
        assert torch.equal(open_ended.results["swapped"], listed.results["swapped"]), kind
        assert open_ended.run.positions["swapped.w"] == listed.run.positions["swapped.w"], kind
        assert not torch.equal(open_ended.results["swapped"], open_ended.results["clean"]), kind
        runs[kind] = open_ended.results
    assert same(runs["nnterp"], runs["hooks"])


def test_a_write_to_the_rows_last_call_without_max_new_tokens_is_refused():
    def change(d):
        written(d)
        d["steps"]["swapped"].pop("max_new_tokens")
        d["steps"]["swapped"]["interventions"]["w"]["pos"] = {"frame": "2:"}
    with pytest.raises(PlanError, match=r"steps\.swapped\.interventions\.w\.pos: frame '2:' with index ':' names calls up "
                                        r"to the row's last; a write lands at calls listed before the run, so the "
                                        r"step gives max_new_tokens"):
        edited("generated_answer", change).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_write_over_a_span_of_calls_takes_an_index_counted_from_its_start(engines, kind):
    """`{"frame": "2:3", "index": -1}` (one call) and `{"frame": "1:4", "index": 1}` (the second of calls 1
    to 3) both name call 2, so they write what `{"frame": 2, "index": -1}` does."""
    def at(pos):
        def change(d):
            written(d)
            d["steps"]["swapped"]["interventions"]["w"]["pos"] = pos
        return engines(kind, plan_of("generated_answer").model).execute(
            edited("generated_answer", change).compile(), batch_size=BATCH_SIZE)
    base = at({"frame": 2, "index": -1})
    for pos in ({"frame": "2:3", "index": -1}, {"frame": "1:4", "index": 1}):
        other = at(pos)
        assert torch.equal(other.results["swapped"], base.results["swapped"]), pos
        assert other.run.positions["swapped.w"] == base.run.positions["swapped.w"], pos


def test_a_scope_the_prefills_head_does_not_hold_says_why(engines):
    plan = edited("generate_probe", lambda d: d["steps"]["patched"]["interventions"]["logits"].update(
        pos={"scope": {"text": "today"}, "index": -1})).compile()
    with pytest.raises(ValueError, match=r"^patched\.logits: row 0: text 'today' is not in its input; a "
                                         r"generation's prefill computes lm_head_output at its last position only"):
        engines("hooks", plan.model).execute(plan)


def scoped_write(text: str | list[str], held: bool):
    """`written`, its swap replaced by a scale by 0 at the call fed `text`'s last token, found online."""
    def change(d):
        written(d, held=held)
        d["steps"]["swapped"]["interventions"]["w"] = {**WRITE, "pos": {"frame": "1:", "scope": {"text": text},
                                                                 "index": -1}, "ops": [{"op": "scale", "factor": 0.0}]}
    return change


def test_a_scoped_write_lands_online_at_the_call_fed_its_text(engines):
    """`substr`, said as token 1 by the rows the steering leaves alone, is fed to call 2: a scale by 0 there
    changes their tokens from 2 on and not before, and the rows that never say it (the Thursday row stops
    first, the Sunday row says something else) are untouched and recorded missing; both engines at batch
    sizes 1, 2 and 4 and `remote="local"` agree to the bit."""
    change = scoped_write("substr", held=False)
    runs: dict = {(kind, size): steered(engines, kind, size, held=False, change=change)
                  for kind in ("nnterp", "hooks") for size in (1, 2, 4)}
    runs["local"] = steered(engines, "ndif", BATCH_SIZE, held=False, change=change, remote="local")
    plan = runs["nnterp", 2]
    v = plan.results
    assert plan.run.positions["swapped.w"] == ["alignment_missing", [10], [10], "alignment_missing"]
    assert torch.equal(v["swapped"][[0, 3]], v["clean"][[0, 3]])
    assert torch.equal(v["swapped"][1:3, :2], v["clean"][1:3, :2])
    assert (v["swapped"][1:3, 2] != v["clean"][1:3, 2]).all()
    for where, other in runs.items():
        assert torch.equal(other.results["swapped"], v["swapped"]), where
        assert other.run.positions == plan.run.positions, where


def test_a_scoped_write_lands_at_the_occurrence_it_picks(engines):
    """`TWICE` first ends on token 0 (`Handler`), fed to call 1, and next on token 2 (`▁energ`), fed to call 3:
    occurrence 0 writes what `{"frame": 1, "index": -1}` does, once, and occurrence 1 what `{"frame": 3,
    "index": -1}` does, not at the first."""
    def at(pos):
        def change(d):
            scoped_write(TWICE, held=True)(d)
            d["steps"]["swapped"]["interventions"]["w"]["pos"] = pos
        return edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        for occurrence, call in ((0, 1), (1, 3)):
            plan = at({"frame": "1:", "scope": {"text": TWICE, "occurrence": occurrence}, "index": -1})
            scoped = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
            fixed = engines(kind, plan.model).execute(at({"frame": call, "index": -1}), batch_size=BATCH_SIZE)
            assert torch.equal(scoped.results["swapped"], fixed.results["swapped"]), (kind, occurrence)
            assert scoped.run.positions["swapped.w"] == fixed.run.positions["swapped.w"] == \
                [[p + call - 1] for p in PROMPT], (kind, occurrence)
            assert torch.equal(scoped.results["swapped"][:, :call], scoped.results["clean"][:, :call])


def test_a_scoped_write_in_a_later_frame_lands_where_a_read_of_it_resolves(engines):
    """`TWICE` ends on token 0 (fed to call 1) and token 2 (fed to call 3): at `{"frame": 3}` occurrence 0 is the
    one there, so the write lands at call 3, where a read of the same record resolves, and writes what
    `{"frame": 3, "index": -1}` does; both engines to the bit."""
    pos = {"frame": 3, "scope": {"text": TWICE}, "index": -1}

    def at(where, read=False):
        def change(d):
            scoped_write(TWICE, held=True)(d)
            d["steps"]["swapped"]["interventions"]["w"]["pos"] = where
            if read:
                d["steps"]["swapped"]["interventions"]["r"] = {"site": "head", "pos": pos}
        return edited("generated_answer", change).compile()
    runs = {kind: (engines(kind, at(pos).model).execute(at(pos, read=True), batch_size=BATCH_SIZE),
                   engines(kind, at(pos).model).execute(at({"frame": 3, "index": -1}), batch_size=BATCH_SIZE))
            for kind in ("nnterp", "hooks")}
    for kind, (scoped, fixed) in runs.items():
        where = scoped.run.positions
        assert where["swapped.w"] == where["swapped.r"] == fixed.run.positions["swapped.w"] == \
            [[p + 2] for p in PROMPT], kind
        assert torch.equal(scoped.results["swapped"], fixed.results["swapped"]), kind
    assert torch.equal(runs["nnterp"][0].results["swapped"], runs["hooks"][0].results["swapped"])


def test_an_occurrence_span_lands_at_each_occurrence_it_picks(engines):
    """`occurrence: "0:2"` of `Handler`, `energ` and `sier` lands at call 1 (`Handler`) and, where the row still
    says one, at the call fed the next: call 3 on rows 0, 1 and 3 (`▁energ`, `▁energ`, `▁sier`), as a write at
    each of calls 1 and 3 does there; row 2, which the write at call 1 sends elsewhere, has no second; both
    engines to the bit."""
    texts = [*TWICE, "sier"]

    def at(where):
        def change(d):
            scoped_write(texts, held=True)(d)
            w = d["steps"]["swapped"]["interventions"].pop("w")
            d["steps"]["swapped"]["interventions"] = {k: {**w, "pos": p} for k, p in where.items()}
        return edited("generated_answer", change).compile()
    span = {"w": {"frame": "1:", "scope": {"text": texts, "occurrence": "0:2"}, "index": -1}}
    calls = {"w1": {"frame": 1, "index": -1}, "w3": {"frame": 3, "index": -1}}
    results = {}
    for kind in ("nnterp", "hooks"):
        scoped = engines(kind, at(span).model).execute(at(span), batch_size=BATCH_SIZE)
        fixed = engines(kind, at(calls).model).execute(at(calls), batch_size=BATCH_SIZE)
        assert scoped.run.positions["swapped.w"] == [[11, 13], [9, 11], [9], [9, 11]], kind
        assert torch.equal(scoped.results["swapped"][[0, 1, 3]], fixed.results["swapped"][[0, 1, 3]]), kind
        assert torch.equal(scoped.results["swapped"][:, :1], scoped.results["clean"][:, :1]), kind
        results[kind] = scoped.results["swapped"]
    assert torch.equal(results["nnterp"], results["hooks"])


@pytest.mark.parametrize("pos, message", [
    ({"frame": ":", "scope": {"text": "is"}, "index": -1}, r"\.frame: frame ':': a scoped write beyond the"),
    ({"frame": -1, "scope": {"text": "is"}, "index": -1}, r"\.frame: frame -1: a scoped write beyond the"),
    ({"frame": "1:", "scope": {"text": "is"}}, r"\.index: index ':': a scoped write lands at the call fed"),
])
def test_a_scoped_write_needs_decode_calls_and_its_texts_last_token(pos, message):
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.patch\.pos" + message):
        edited("generate_probe", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=pos)).compile()


def doubled(d, scope: dict, index):
    """patching.json over inline rows that say `Friday` twice, reading layer 0 there and everywhere."""
    d["data"] = {"pairs": {"rows": [
        {"input": "Friday or Friday, tomorrow is", "counterfactual_inputs": ["x"], "cf_answer": " Sunday",
         "base_answer": " Friday"},
        {"input": "If it is Friday and it is Friday, then", "counterfactual_inputs": ["x"], "cf_answer": " Sunday",
         "base_answer": " Friday"}]}}
    d["steps"] = {"look": {"kind": "forward", "input": "pairs.input",
                           "interventions": {"there": {"site": "target", "pos": {"scope": scope, "index": index}},
                                             "every": {"site": "target", "pos": {"index": ":"}}}}}
    d["saves"] = ["look.there", "look.every"]


@pytest.mark.parametrize("occurrence, index, picks", [
    (0, -1, [0]), (1, -1, [1]), (-1, -1, [1]), (":", ":", [0, 1]), ("0:2", ":", [0, 1]), ([1, 0], ":", [0, 1])])
def test_a_doubled_prefill_text_takes_the_occurrence_asked_for(engines, occurrence, index, picks):
    """Each row says `Friday` twice: `occurrence` picks which, the same selector `frame` and `index` use; an
    int keeps the read rectangular, a slice or picks make it ragged (the picked occurrences' tokens, in
    order). Positions and values on both engines, to the bit, are those of the read of every position there."""
    scope = {"text": "Friday", "occurrence": occurrence}
    plan = edited("patching", lambda d: doubled(d, scope, index)).compile()
    runs = {kind: engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")}
    for kind, filled in runs.items():
        tokens = filled.run.tokens["pairs.input"]
        fridays = [[i for i, t in enumerate(row) if t == "▁Friday"] for row in tokens]
        assert all(len(f) == 2 for f in fridays)
        assert filled.run.positions["look.there"] == [[f[i] for i in picks] for f in fridays], kind
        there, every = filled.results["look.there"], filled.results["look.every"]
        entries = list(zip(every["row"].tolist(), every["tok"].tolist()))
        want = [every["values"][entries.index((r, f[i]))] for r, f in enumerate(fridays) for i in picks]
        got = there["values"] if isinstance(there, dict) else there[:, 0]
        assert torch.equal(got, torch.stack(want)), kind
    assert same(runs["nnterp"].results, runs["hooks"].results)


def test_a_generated_text_said_twice_reads_the_occurrence_asked_for(engines):
    """`TWICE` ends on inputs 0 and 2 of what every row said: occurrence 1 is the second, `":"` both."""
    def change(d):
        d["steps"]["answer"]["interventions"] = {
            "second": {"site": "head", "pos": {"frame": "1:", "scope": {"text": TWICE, "occurrence": 1},
                                               "index": -1}},
            "both": {"site": "head", "pos": {"frame": "1:", "scope": {"text": TWICE, "occurrence": ":"}}}}
        d["steps"] = {"answer": d["steps"]["answer"]}
        d["saves"] = ["answer.second", "answer.both"]
    plan = edited("generated_answer", change).compile()
    for kind in ("nnterp", "hooks"):
        filled = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
        assert filled.run.positions["answer.second"] == [[p + 2] for p in PROMPT]
        assert filled.run.positions["answer.both"] == [[p, p + 2] for p in PROMPT]    # the occurrences' tokens only
        assert filled.results["answer.both"]["values"].shape == (8, 32000)


@pytest.mark.parametrize("strip, found", [(False, False), (True, True)])
def test_strip_locates_the_text_without_its_surrounding_space(engines, strip, found):
    """` Friday` with its leading space is not in a prompt that starts with `Friday`; stripped, it is."""
    def change(d):
        doubled(d, {"text": " Friday", "strip": strip}, -1)
        d["data"]["pairs"]["rows"] = [{**d["data"]["pairs"]["rows"][0], "input": "Friday comes, then"}]
    plan = edited("patching", change).compile()
    engine = engines("hooks", plan.model)
    if not found:
        with pytest.raises(ValueError, match=r"^look\.there: row 0: text ' Friday' is not in its input"):
            engine.execute(plan, batch_size=BATCH_SIZE)
        return
    filled = engine.execute(plan, batch_size=BATCH_SIZE)
    assert [filled.run.tokens["pairs.input"][0][i] for (i,) in filled.run.positions["look.there"]] == ["▁Friday"]


@pytest.mark.parametrize("scope, message", [
    ({"text": "er", "occurrence": -1}, r"\.scope\.occurrence: occurrence -1 counts from the last occurrence, which"),
    ({"text": "er", "occurrence": "0:-1"}, r"\.scope\.occurrence: occurrence '0:-1' counts from the last"),
])
def test_a_scoped_write_counts_its_occurrences_from_the_first(scope, message):
    def change(d):
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = {"frame": "1:", "scope": scope, "index": -1}
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.patch\.pos" + message):
        edited("generate_probe", change).compile()


@pytest.mark.parametrize("scope, message", [
    ({"turn": "user", "strip": True}, r"a turn's messages are located as the chat template rendered them; it takes "
                                      r"no strip"),
    ({"turn": "user[1]"}, r"turn 'user\[1\]' is a role; its messages are its occurrences, so "),
])
def test_a_turn_is_a_role_and_takes_no_strip(scope, message):
    def change(d):
        d["steps"]["blanked"]["interventions"]["blank"]["pos"]["scope"] = scope
    with pytest.raises(PlanError, match=message):
        edited("chat_turn", change)


def test_a_roles_messages_are_its_occurrences(engines):
    """A user, assistant, user chat: `turn: user` occurrence 0 is the first user message's tokens, 1 and -1
    the second's, `":"` both, and `turn: assistant` the reply's; each is the tokens the template rendered
    the message's content as, on both engines to the bit."""
    chat = [{"role": "user", "content": "Name a weekday."}, {"role": "assistant", "content": "Thursday."},
            {"role": "user", "content": "If today is Thursday, tomorrow is"}]
    reads = {"first": {"turn": "user"}, "second": {"turn": "user", "occurrence": 1},
             "last": {"turn": "user", "occurrence": -1}, "both": {"turn": "user", "occurrence": ":"},
             "reply": {"turn": "assistant"}}

    def change(d):
        d["data"] = {"chats": {"rows": [{"chat": chat, "base_answer": " Friday"}]}}
        d["steps"] = {"look": {"kind": "forward", "input": "chats.chat",
                               "interventions": {k: {"site": {"component": "layer_output", "layers": 0},
                                                     "pos": {"scope": scope}} for k, scope in reads.items()}}}
        d["saves"] = [f"look.{k}" for k in reads]
    plan = edited("chat_turn", change).compile()
    runs = {kind: engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")}
    assert same(runs["nnterp"].results, runs["hooks"].results)
    assert runs["nnterp"].run.positions == runs["hooks"].run.positions
    where, tokens = runs["nnterp"].run.positions, runs["nnterp"].run.tokens["chats.chat"][0]
    said = {k: "".join(tokens[i] for i in where[f"look.{k}"][0]).replace("▁", " ").strip() for k in reads}
    assert said == {"first": "Name a weekday.", "second": "If today is Thursday, tomorrow is",
                    "last": "If today is Thursday, tomorrow is",
                    "both": "Name a weekday. If today is Thursday, tomorrow is", "reply": "Thursday."}


def listed(d, rows: list[dict], reads: dict):
    """patching.json over inline `rows`, one forward reading layer 0 at each of `reads` (name -> pos)."""
    d["data"] = {"pairs": {"rows": [{"counterfactual_inputs": ["x"], "cf_answer": " Sunday", "base_answer": " Friday",
                                     **row} for row in rows]}}
    d["steps"] = {"look": {"kind": "forward", "input": "pairs.input",
                           "interventions": {k: {"site": "target", "pos": pos} for k, pos in reads.items()}}}
    d["saves"] = [f"look.{k}" for k in reads]


def test_a_list_of_texts_takes_its_occurrences_in_order(engines):
    """`["Larry", "Bob"]` over rows that name Bob first and Larry first: occurrence 0 is whichever comes first
    in the row, 1 the other, `":"` both, on both engines to the bit."""
    rows = [{"input": "Bob met Larry, then"}, {"input": "Larry met Bob, then"}]
    both = ["Larry", "Bob"]
    reads = {"first": {"scope": {"text": both}}, "second": {"scope": {"text": both, "occurrence": 1}},
             "both": {"scope": {"text": both, "occurrence": ":"}},
             "bob": {"scope": {"text": "Bob"}}, "larry": {"scope": {"text": "Larry"}}}
    plan = edited("patching", lambda d: listed(d, rows, reads)).compile()
    runs = {kind: engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE) for kind in ("nnterp", "hooks")}
    where = runs["nnterp"].run.positions
    bob, larry = where["look.bob"], where["look.larry"]
    assert where["look.first"] == [bob[0], larry[1]] and where["look.second"] == [larry[0], bob[1]]
    assert where["look.both"] == [sorted(b + l) for b, l in zip(bob, larry)]
    assert same(runs["nnterp"].results, runs["hooks"].results)
    assert runs["nnterp"].run.positions == runs["hooks"].run.positions


def test_overlapping_texts_take_the_union_of_their_tokens(engines):
    """`is` occurs inside `this` and on its own: `":"` of `["this", "is"]` holds each token once."""
    reads = {"union": {"scope": {"text": ["this", "is"], "occurrence": ":"}}, "every": {"index": ":"}}
    plan = edited("patching", lambda d: listed(d, [{"input": "this is it, then"}], reads)).compile()
    filled = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    (got,), tokens = filled.run.positions["look.union"], filled.run.tokens["pairs.input"][0]
    assert got == sorted(set(got)) and [tokens[i] for i in got] == ["▁this", "▁is"]


def as_written(*texts: str) -> list[tuple[str, bool, bool]]:
    """Texts a scope searches for as written, neither stripped nor lowercased."""
    return [(t, False, False) for t in texts]


def test_occurrences_are_ordered_by_where_they_end():
    """`b` inside `abc` ends first, so it is occurrence 0 and `abc` occurrence 1, in the prompt, in the
    generated inputs and online alike; of two that end together, the one that starts first comes first."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    scope = {"text": ["abc", "b"], "occurrence": 0}
    letters = [(i, i + 1) for i in range(5)]           # a token per letter, so every letter is a boundary
    assert positions.located(as_written("abc", "b"), "xabcx", letters) == [(2, 3), (1, 4)]
    assert positions.located(as_written("bc", "abc"), "xabcx", letters) == [(1, 4), (2, 4)]
    ids = tokenizer("x a b c x", add_special_tokens=False)["input_ids"]
    words, chars = positions.char_spans(tokenizer, ids)
    spans = positions.located(as_written("a b c", "b"), words, chars)
    assert [words[a:b] for a, b in spans] == ["b", "a b c"]
    def lands_at(k):
        pos = {"frame": "1:", "scope": {**scope, "occurrence": k}, "index": -1}
        return [j for j in range(len(ids)) if positions.landing(pos, as_written("a b c", "b"), ids[:j + 1], set(), tokenizer)]
    first, second = lands_at(0), lands_at(1)
    assert len(first) == len(second) == 1 and first[0] < second[0]     # `b` is known before `a b c` is


def test_a_scoped_write_counts_its_occurrences_within_its_frame():
    """`Bob` is fed to calls 1 and 3: at frame 3, occurrence 0 is the one there, as a read of the same record
    counts it, and an occurrence span lands at each occurrence it picks, as it appears."""
    tokenizer = AutoTokenizer.from_pretrained(LLAMA[0], revision=LLAMA[1])
    ids = tokenizer("Bob met Bob met", add_special_tokens=False)["input_ids"]
    assert tokenizer.convert_ids_to_tokens(ids) == ["▁Bob", "▁met", "▁Bob", "▁met"]

    def calls(pos):
        return [k for k in range(1, len(ids) + 1) if positions.landing(pos, as_written("Bob"), ids[:k], set(), tokenizer)]
    at3 = {"frame": 3, "scope": {"text": "Bob", "occurrence": 0}, "index": -1}
    assert calls(at3) == [3]
    assert resolved(at3, ids, tokenizer, "Bob") == [2 + 2]                # call 3's input, after a 2-token prompt
    assert calls({**at3, "frame": "1:", "scope": {"text": "Bob", "occurrence": "0:2"}}) == [1, 3]


def test_a_list_of_columns_takes_the_first_of_either(engines):
    """`["pairs.number", "pairs.entity"]`: the number comes first on one row, the entity on the other."""
    rows = [{"input": "Friday has 3 meetings", "entity": "Friday", "number": "3"},
            {"input": "There are 2 on Monday", "entity": "Monday", "number": "2"}]
    reads = {"first": {"scope": {"column": ["pairs.number", "pairs.entity"]}},
             "entity": {"scope": {"column": "pairs.entity"}}, "number": {"scope": {"column": "pairs.number"}}}
    plan = edited("patching", lambda d: listed(d, rows, reads)).compile()
    filled = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE)
    where = filled.run.positions
    assert where["look.first"] == [where["look.entity"][0], where["look.number"][1]]


def test_a_list_of_generated_texts_lands_on_whichever_completes_first(engines):
    """Steered, rows say different things: `["energ", "Friday"]` is `▁energ` (token 2, fed to call 3) on the
    rows that say it and `▁Friday` (token 0, fed to call 1) on the Sunday row; the Thursday row stops before
    either and is missing. A read finds them there and a write lands there, on both engines."""
    texts = ["energ", "Friday"]
    pos = {"frame": "1:", "scope": {"text": texts}, "index": -1}

    def change(d):
        written(d, held=False)
        d["steps"]["swapped"]["interventions"]["w"] = {**WRITE, "pos": pos, "ops": [{"op": "scale", "factor": 0.0}]}
        d["steps"]["clean"]["interventions"] = {"found": {"site": "head", "pos": pos}}
        d["saves"] += ["clean.found"]
    for kind in ("nnterp", "hooks"):
        plan = steered(engines, kind, BATCH_SIZE, held=False, change=change)
        v, where = plan.results, plan.run.positions
        assert where["clean.found"] == ["alignment_missing", [9 + 2], [9 + 2], [9 + 0]], kind     # P + token
        assert where["swapped.w"] == where["clean.found"], kind
        assert torch.equal(v["swapped"][0], v["clean"][0])
        for row, call in ((1, 3), (3, 1)):
            assert torch.equal(v["swapped"][row, :call], v["clean"][row, :call]), (kind, row)
            assert v["swapped"][row, call] != v["clean"][row, call], (kind, row)
