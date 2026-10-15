"""Answers: a metric lines its token column up with the read it scores, a column reference compares as its flags
say, a set is any of its members, a scope that is `before` is the positions that predict its tokens, and a list
`input` joins columns per row."""

import pytest
import torch

from clab import Plan
from clab.plan.model import PlanError
from clab.runtime import data
from clab.runtime.steps.state import answered
from conftest import BATCH_SIZE, plan_of

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM", "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}
GPT2 = plan_of("gpt2_reach").model.model_dump()
DAYS = [{"input": "If today is Monday, tomorrow is", "answer": " Tuesday", "short": " Friday", "lowered": "monday",
         "chat": [{"role": "user", "content": "Hi"}]},
        {"input": "If today is Thursday, tomorrow is", "answer": " Friday", "short": " Sunday", "lowered": "thursday",
         "chat": [{"role": "user", "content": "Hi"}]}]
HEAD = {"component": "lm_head_output"}


def doc(steps: dict, saves: list, rows: list | None = None, model: dict | None = None) -> Plan:
    return Plan(model=model or TINY, data={"days": {"rows": rows or DAYS}}, steps=steps, saves=saves).compile()


def run(engines, plan: Plan, kind: str = "nnterp") -> Plan:
    return engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)


def record(plan: Plan):
    assert plan.run is not None
    return plan.run


def forced(scope: dict, extra: dict | None = None) -> dict:
    return {"forced": {"kind": "forward", "input": ["days.input", "days.answer"],
                       "reads": {"logits": {"site": HEAD, "pos": {"scope": scope}}}}, **(extra or {})}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_before_reads_the_positions_that_predict_the_scope_and_each_scores_its_token(engines, kind):
    """The joined input is the prompt and its answer; the answer's tokens sit at the end, `before` reads the
    position before each, and token i of the answer scores the distribution there."""
    steps = forced({"column": "days.answer", "occurrence": -1, "before": True}, {
        "at": {"kind": "forward", "input": ["days.input", "days.answer"],
               "reads": {"logits": {"site": HEAD, "pos": {"scope": {"column": "days.answer", "occurrence": -1}}}}},
        "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.answer"}})
    filled = run(engines, doc(steps, ["nll", "forced.logits"]), kind)
    at, before = record(filled).positions["at.logits"], record(filled).positions["forced.logits"]
    assert [len(p) for p in at] == [3, 1] and before == [[p - 1 for p in row] for row in at]
    tokens = record(filled).tokens["days.input + days.answer"]
    assert tokens[0][-3:] == ["▁T", "ues", "day"] and tokens[1][-1] == "▁Friday"
    nll, logits = filled.results["nll"], filled.results["forced.logits"]
    tokenizer = engines(kind, filled.model).tokenizer
    for r in range(2):
        ids = [i[0] for i in data.answers(tokenizer, [DAYS[r]["answer"]], "a")[0]]
        expected = -logits["values"][logits["row"] == r].log_softmax(-1)[torch.arange(len(ids)), ids]
        assert torch.equal(nll["values"][nll["row"] == r], expected)


def test_a_scope_token_at_position_0_has_no_position_before_it(engines):
    """GPT-2 adds no BOS, so a prompt's first word sits at position 0, and nothing comes before it."""
    plan = doc({"read": {"kind": "forward", "input": "days.input",
                         "reads": {"x": {"site": HEAD, "pos": {"scope": {"text": "If", "before": True}}}}}},
               ["read.x"], model=GPT2)
    with pytest.raises(ValueError, match=r"^read\.x: row 0: text 'If' begins the row, at position 0, which no "
                                         r"position comes before$"):
        run(engines, plan)


def test_lower_searches_the_input_lowercased_and_the_positions_are_the_inputs(engines):
    def read(scope):
        return {"kind": "forward", "input": "days.input", "reads": {"x": {"site": HEAD, "pos": {"scope": scope}}}}
    plan = doc({"exact": read({"text": ["Monday", "Thursday"]}),
                "lowered": read({"column": {"column": "days.lowered", "lower": True}}),
                "text": read({"text": [" MONDAY ", "THURSDAY"], "lower": True, "strip": True})}, ["exact.x"])
    positions = record(run(engines, plan)).positions
    assert positions["lowered.x"] == positions["exact.x"] == positions["text.x"]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_class_is_every_spelling_and_a_set_any_member(engines, kind):
    """Under `strip` and `lower` " Yes" stands for every token whose text is `yes` so compared; a list is a set:
    `token_prob` sums its members, `match` hits on any, `cross_entropy` is minus the log of the sum."""
    rows = [{"input": "Is it? Answer:", "answer": " Yes", "set": [" Yes", " True", " Yes"]}]
    plan = Plan(model=TINY, data={"qa": {"rows": rows}}, steps={
        "clean": {"kind": "forward", "input": "qa.input", "reads": {"logits": {"site": HEAD, "pos": -1}}},
        "p": {"kind": "op", "op": "token_prob", "of": "clean.logits",
              "token": {"column": "qa.answer", "strip": True, "lower": True}},
        "any": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "qa.set"},
        "nll": {"kind": "op", "op": "cross_entropy", "of": "clean.logits", "target": "qa.set"},
        "hit": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "qa.set"}},
        saves=["clean.logits", "p", "any", "nll", "hit"]).compile()
    filled = run(engines, plan, kind)
    tokenizer = engines(kind, plan.model).tokenizer
    spelled = data.spellings(tokenizer, True, True)["yes"]
    assert sorted(tokenizer.convert_ids_to_tokens(spelled)) == sorted(["▁Yes", "▁yes", "▁YES", "Yes", "yes", "YES"])
    probs = filled.results["clean.logits"][0, 0].softmax(-1)
    torch.testing.assert_close(filled.results["p"][0, 0], probs[spelled].sum())
    members = [tokenizer(" Yes", add_special_tokens=False)["input_ids"][0],
               tokenizer(" True", add_special_tokens=False)["input_ids"][0]]
    torch.testing.assert_close(filled.results["any"][0, 0], probs[members].sum())        # " Yes" once
    torch.testing.assert_close(filled.results["nll"][0, 0], -probs[members].sum().log())
    top = int(probs.argmax())
    assert filled.results["hit"][0, 0].item() == float(top in members)


def refused_before_any_forward(engines, plan: Plan, message: str) -> None:
    engine = engines("nnterp", plan.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=message):
            engine.execute(plan, batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    assert not calls


def test_an_answer_of_several_tokens_needs_a_position_per_token(engines):
    plan = doc({"read": {"kind": "forward", "input": "days.input", "reads": {"logits": {"site": HEAD, "pos": -1}}},
                "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": "days.answer"}}, ["p"])
    refused_before_any_forward(engines, plan, r"^p: days\.answer row 0: ' Tuesday' is 3 tokens; the read has 1 "
                                              r"position on that row; token i of an answer scores the read's position "
                                              r"i, so the read carries one per token \(a scope with \"before\": true "
                                              r"reads the positions that predict the answer's tokens\)$")


def test_an_answer_of_several_tokens_needs_a_pos_axis(engines):
    plan = doc(forced({"column": "days.answer", "before": True}, {
        "avg": {"kind": "op", "op": "mean", "of": "forced.logits", "over": ["pos"]},
        "p": {"kind": "op", "op": "token_prob", "of": "avg", "token": "days.answer"}}), ["p"])
    refused_before_any_forward(engines, plan, r"^p: days\.answer row 0: ' Tuesday' is 3 tokens, a position each, "
                                              r"and the value has no pos axis \(\['rows', 'vocab_size'\]\)$")


@pytest.mark.parametrize("op, args", [
    ("logit_diff", {"a": {"column": "days.word", "lower": True, "strip": True}, "b": "days.short"}),
    ("token_logit", {"token": "days.set"})])
def test_a_class_or_a_set_has_no_one_logit(engines, op, args):
    """" today" under strip and lower is ▁today, ▁Today and today; a set is its members."""
    rows = [{**r, "set": [" Friday", " Sunday"], "word": " today"} for r in DAYS]
    plan = doc({"read": {"kind": "forward", "input": "days.input", "reads": {"logits": {"site": HEAD, "pos": -1}}},
                "x": {"kind": "op", "op": op, "of": "read.logits", **args}}, ["x"], rows)
    refused_before_any_forward(engines, plan, rf"^x: days\.(word|set) row 0: .* stands for (3|2) ids \(a set, or a "
                                              rf"class under strip or lower\), which have no one logit for {op}; "
                                              r"token_prob scores them together$")


def test_a_sets_members_are_one_width(engines):
    rows = [{**r, "set": [" Monday", " Tuesday"]} for r in DAYS]
    plan = doc({"read": {"kind": "forward", "input": "days.input", "reads": {"logits": {"site": HEAD, "pos": -1}}},
                "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": "days.set"}}, ["p"], rows)
    refused_before_any_forward(engines, plan, r"^p: days\.set row 0: its members are ' Monday' 1, ' Tuesday' 3 "
                                              r"tokens; a set's members are one width, the positions its read aligns$")


def test_a_generated_rows_positions_that_do_not_line_up_with_its_answer_are_refused():
    """Beyond the prefill positions are found in the run; there a row with entries, but not one per token of
    its answer, is refused by name."""
    of = {"values": torch.zeros(2, 5), "row": torch.tensor([0, 0]), "tok": torch.tensor([3, 4])}
    with pytest.raises(ValueError, match=r"^days\.answer row 0: ' Tuesday' is 3 tokens; the read has 2 positions"):
        answered([" Tuesday"], [[[1], [2], [3]]], "days.answer", of, ["entries", "vocab_size"], None)
    missing = {"values": torch.zeros(0, 5), "row": torch.tensor([], dtype=torch.long), "tok": torch.tensor([])}
    assert answered([" Tuesday"], [[[1], [2], [3]]], "days.answer", missing, ["entries", "vocab_size"], None).shape == (0, 1)


@pytest.mark.parametrize("scope, message", [
    ({"column": "days.answer", "lower": True}, r"a column's lower goes on its reference"),
    ({"column": "days.answer", "strip": True}, r"a column's strip goes on its reference"),
    ({"turn": "user", "lower": True}, r"a turn's messages are located as the chat template rendered them; it takes "
                                      r"no lower"),
    ({"column": {"column": "days.answer", "upper": True}}, r"unknown field 'upper'; valid: column, strip, lower"),
])
def test_a_scopes_flags_are_a_texts_or_on_a_columns_reference(scope, message):
    with pytest.raises(PlanError, match=message):
        doc({"read": {"kind": "forward", "input": "days.input", "reads": {"x": {"site": HEAD, "pos": {"scope": scope}}}}},
            ["read.x"])


def test_a_write_beyond_the_prefill_takes_no_before():
    with pytest.raises(PlanError, match=r"writes\.w\.pos\.scope\.before: a write beyond the prefill lands as its "
                                        r"call runs, before the token it would predict is chosen"):
        doc({"gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 2, "writes": {"w": {
            "site": {"component": "layer_output", "layers": 0}, "pos": {"frame": "1:", "scope": {
                "text": "Friday", "before": True}, "index": -1}, "ops": [{"op": "scale", "factor": 0.0}]}}}}, ["gen"])


@pytest.mark.parametrize("input, message", [
    (["days.input", "days.chat"], r"input\[1\]: 'days\.chat' is not text on row 0 of 'days'; a list input joins text"),
    (["days.input", "other.input"], r"input\[1\]: 'other\.input' is a column of 'other', and the value it goes with "
                                    r"has a row per example of 'days'"),
    ([{"text": "only a literal"}], r"a list input names at least one column"),
])
def test_a_list_input_joins_text_columns_of_one_dataset(input, message):
    with pytest.raises(PlanError, match=message):
        Plan(model=TINY, data={"days": {"rows": DAYS}, "other": {"rows": DAYS}}, steps={
            "read": {"kind": "forward", "input": input, "reads": {"x": {"site": HEAD, "pos": -1}}}},
            saves=["read.x"]).compile()


def test_a_list_input_joins_its_parts_as_they_are(engines):
    plan = doc({"read": {"kind": "forward", "input": ["days.input", {"text": " and then"}, "days.answer"],
                         "reads": {"x": {"site": HEAD, "pos": -1}}}}, ["read.x"])
    tokens = record(run(engines, plan)).tokens['days.input + " and then" + days.answer']
    assert engines("nnterp", plan.model).tokenizer.convert_tokens_to_string(tokens[1][1:]) == \
        "If today is Thursday, tomorrow is and then Friday"


def test_contains_reads_a_decodes_text():
    with pytest.raises(PlanError, match=r"said\.of: contains reads text, a decode's value \(without tokens\); "
                                        r"'gen' is not one"):
        doc({"gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 2},
             "said": {"kind": "op", "op": "contains", "of": "gen", "text": "days.answer"}}, ["said"])
