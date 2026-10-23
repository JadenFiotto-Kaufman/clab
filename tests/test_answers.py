"""Answers: a metric lines its token column up with the read it scores, a column reference compares as its flags
say, a set is any of its members, a scope that is `before` is the positions that predict its tokens, and a list
`input` joins columns per row."""

import pytest
import torch
from conftest import BATCH_SIZE, plan_of

from clab import Plan, ops
from clab.compile import answers
from clab.compile.answers import Answer, answered
from clab.plan import PlanError

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}
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
                       "interventions": {"logits": {"site": HEAD, "pos": {"scope": scope}}}}, **(extra or {})}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_before_reads_the_positions_that_predict_the_scope_and_each_scores_its_token(engines, kind):
    """The joined input is the prompt and its answer; the answer's tokens sit at the end, `before` reads the
    position before each, and token i of the answer scores the distribution there."""
    steps = forced({"column": "days.answer", "occurrence": -1, "before": True}, {
        "at": {"kind": "forward", "input": ["days.input", "days.answer"],
               "interventions": {"logits": {"site": HEAD, "pos": {"scope": {"column": "days.answer", "occurrence": -1}}}}},
        "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "days.answer"}})
    filled = run(engines, doc(steps, ["nll", "forced.logits"]), kind)
    at, before = record(filled).positions["at.logits"], record(filled).positions["forced.logits"]
    assert [len(p) for p in at] == [3, 1] and before == [[p - 1 for p in row] for row in at]
    tokens = record(filled).tokens["days.input + days.answer"]
    assert tokens[0][-3:] == ["▁T", "ues", "day"] and tokens[1][-1] == "▁Friday"
    nll, logits = filled.results["nll"], filled.results["forced.logits"]
    tokenizer = engines(kind, filled.model).tokenizer
    for r in range(2):                  # the joined input's own next ids, which the answer scores
        joined = tokenizer.convert_tokens_to_ids(tokens[r])
        ids = [joined[p + 1] for p in before[r]]
        expected = -logits[r, :len(ids)].log_softmax(-1)[torch.arange(len(ids)), ids]
        assert torch.equal(nll[r, :len(ids)], expected) and nll[r, len(ids):].isnan().all()


def test_a_scope_token_at_position_0_has_no_position_before_it(engines):
    """GPT-2 adds no BOS, so a prompt's first word sits at position 0, and nothing comes before it."""
    plan = doc({"read": {"kind": "forward", "input": "days.input",
                         "interventions": {"x": {"site": HEAD, "pos": {"scope": {"text": "If", "before": True}}}}}},
               ["read.x"], model=GPT2)
    with pytest.raises(ValueError, match=r"^read\.x: row 0: text 'If' begins the row, at position 0, which no "
                                         r"position comes before$"):
        run(engines, plan)


def test_lower_searches_the_input_lowercased_and_the_positions_are_the_inputs(engines):
    def read(scope):
        return {"kind": "forward", "input": "days.input", "interventions": {"x": {"site": HEAD, "pos": {"scope": scope}}}}
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
        "clean": {"kind": "forward", "input": "qa.input", "interventions": {"logits": {"site": HEAD, "pos": -1}}},
        "p": {"kind": "op", "op": "token_prob", "of": "clean.logits",
              "token": {"column": "qa.answer", "strip": True, "lower": True}},
        "any": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "qa.set"},
        "nll": {"kind": "op", "op": "cross_entropy", "of": "clean.logits", "target": "qa.set"},
        "hit": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "qa.set"}},
        saves=["clean.logits", "p", "any", "nll", "hit"]).compile()
    filled = run(engines, plan, kind)
    tokenizer = engines(kind, plan.model).tokenizer
    spelled = answers.spellings(tokenizer, True, True)["yes"]
    assert sorted(tokenizer.convert_ids_to_tokens(spelled)) == \
        sorted(["▁Yes", "▁yes", "▁YES", "Yes", "yes", "YES"])
    probs = filled.results["clean.logits"][0, 0].softmax(-1)
    torch.testing.assert_close(filled.results["p"][0, 0], probs[spelled].sum())
    members = [tokenizer(" Yes", add_special_tokens=False)["input_ids"][0],
               tokenizer(" True", add_special_tokens=False)["input_ids"][0]]
    torch.testing.assert_close(filled.results["any"][0, 0], probs[members].sum())        # " Yes" once
    torch.testing.assert_close(filled.results["nll"][0, 0], -probs[members].sum().log())
    top = int(probs.argmax())
    assert filled.results["hit"][0, 0].item() == float(top in members)


def test_a_set_scores_nothing_past_a_rows_own_answer():
    """Token i of an answer scores position i, and a row's positions past its answer are NaN in the value. A
    set's row there holds only padding ids, and its probability and cross-entropy stay missing, not 0 and inf."""
    of = torch.randn(2, 3, 6)
    of[1, 1:] = float("nan")                            # row 1's answer is one token, row 0's three
    ids = torch.tensor([[[1, -1], [2, -1], [3, -1]], [[4, 5], [-1, -1], [-1, -1]]])
    dims = ["rows", "pos", "vocab_size"]
    p, nll = ops.metrics.token_prob(of, ids, dims), ops.metrics.cross_entropy(of, ids, dims)
    probs = of.softmax(-1)
    torch.testing.assert_close(p[0], probs[0, [0, 1, 2], [1, 2, 3]])
    torch.testing.assert_close(p[1, 0], probs[1, 0, [4, 5]].sum())
    torch.testing.assert_close(nll[1, 0], -probs[1, 0, [4, 5]].sum().log())
    assert p[1, 1:].isnan().all() and nll[1, 1:].isnan().all()


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
    plan = doc({"read": {"kind": "forward", "input": "days.input", "interventions": {"logits": {"site": HEAD, "pos": -1}}},
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
    plan = doc({"read": {"kind": "forward", "input": "days.input", "interventions": {"logits": {"site": HEAD, "pos": -1}}},
                "x": {"kind": "op", "op": op, "of": "read.logits", **args}}, ["x"], rows)
    refused_before_any_forward(engines, plan, rf"^x: days\.(word|set) row 0: .* stands for (3|2) ids \(a set, or a "
                                              rf"class under strip or lower\), which have no one logit for {op}; "
                                              r"token_prob scores them together$")


def test_a_sets_members_are_one_width(engines):
    rows = [{**r, "set": [" Monday", " Tuesday"]} for r in DAYS]
    plan = doc({"read": {"kind": "forward", "input": "days.input", "interventions": {"logits": {"site": HEAD, "pos": -1}}},
                "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": "days.set"}}, ["p"], rows)
    refused_before_any_forward(engines, plan, r"^p: days\.set row 0: its members are ' Monday' 1, ' Tuesday' 3 "
                                              r"tokens; a set's members are one width, the positions its read aligns$")


def test_a_generated_rows_positions_that_do_not_line_up_with_its_answer_are_refused():
    """Beyond the prefill positions are found in the run; there a row with positions, but not one per token of
    its answer, is refused by name, and a row with none is missing."""
    of = torch.zeros(1, 2, 5)
    with pytest.raises(ValueError, match=r"^days\.answer row 0: ' Tuesday' is 3 tokens; the read has 2 positions"):
        answered(Answer([" Tuesday"], [[[1], [2], [3]]], "days.answer", True), of, ["rows", "pos", "vocab_size"],
                 None)
    missing = torch.full((1, 3, 5), float("nan"))
    assert answered(Answer([" Tuesday"], [[[1], [2], [3]]], "days.answer", True), missing,
                    ["rows", "pos", "vocab_size"], None).shape == (1, 3, 1)


@pytest.mark.parametrize("scope, message", [
    ({"column": "days.answer", "lower": True}, r"a column's lower goes on its reference"),
    ({"column": "days.answer", "strip": True}, r"a column's strip goes on its reference"),
    ({"turn": "user", "lower": True}, r"a turn's messages are located as the chat template rendered them; it takes "
                                      r"no lower"),
    ({"column": {"column": "days.answer", "upper": True}}, r"unknown field 'upper'; valid: column, strip, lower"),
])
def test_a_scopes_flags_are_a_texts_or_on_a_columns_reference(scope, message):
    with pytest.raises(PlanError, match=message):
        doc({"read": {"kind": "forward", "input": "days.input",
                      "interventions": {"x": {"site": HEAD, "pos": {"scope": scope}}}}}, ["read.x"])


def test_a_write_beyond_the_prefill_takes_no_before():
    with pytest.raises(PlanError, match=r"interventions\.w\.pos\.scope\.before: a write beyond the prefill lands as its "
                                        r"call runs, before the token it would predict is chosen"):
        doc({"gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 2, "interventions": {"w": {
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
            "read": {"kind": "forward", "input": input, "interventions": {"x": {"site": HEAD, "pos": -1}}}},
            saves=["read.x"]).compile()


def test_a_list_input_joins_its_parts_as_they_are(engines):
    plan = doc({"read": {"kind": "forward", "input": ["days.input", {"text": " and then"}, "days.answer"],
                         "interventions": {"x": {"site": HEAD, "pos": -1}}}}, ["read.x"])
    tokens = record(run(engines, plan)).tokens['days.input + " and then" + days.answer']
    assert engines("nnterp", plan.model).tokenizer.convert_tokens_to_string(tokens[1][1:]) == \
        "If today is Thursday, tomorrow is and then Friday"


def test_contains_reads_a_detokenized_text():
    with pytest.raises(PlanError, match=r"said\.of: contains reads text, a detokenize's value \(without tokens\); "
                                        r"'gen' is not one"):
        doc({"gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 2},
             "said": {"kind": "op", "op": "contains", "of": "gen", "text": "days.answer"}}, ["said"])


def test_a_before_scope_in_a_generation_is_the_calls_that_chose_its_tokens(engines):
    """`frame: "1:"` picks the generated tokens, the last included: a word only the prompt says is missing, and
    one the generation says is read at the position before each of its tokens, its first chosen by the prefill
    (this checkpoint says `Handler substr ...` on every row)."""
    def read(text):
        return {"site": {"component": "layer_output", "layers": 0},
                "pos": {"frame": "1:", "scope": {"text": text, "before": True}}}
    plan = Plan(model=TINY, data={"days": {"rows": [{"input": r["input"]} for r in DAYS]}}, steps={
        "gen": {"kind": "generate", "input": "days.input", "max_new_tokens": 6, "min_new_tokens": 6,
                "do_sample": False, "interventions": {"prompt": read("today"), "first": read("Handler"),
                                              "said": read("substr")}}},
        saves=["gen.said"]).compile()
    filled = run(engines, plan)
    prompts = [len(t) for t in record(filled).tokens["days.input"]]
    assert record(filled).positions["gen.prompt"] == ["alignment_missing"] * 2
    assert record(filled).positions["gen.first"] == [[n - 1] for n in prompts]       # the prefill's last
    assert record(filled).positions["gen.said"] == [[n] for n in prompts]             # the call fed `Handler`


def test_a_teacher_forced_answer_is_the_inputs_own_next_token(engines):
    """`"day"` alone is ▁day; appended to `tomorrow is (` it is the token `day` after `▁(`, the token the position
    before predicts. (Appended to `Sun`, `"day"` would be inside the word `Sunday`, where no occurrence is.)"""
    rows = [{"prefix": "tomorrow is (", "bare": "day"}]
    plan = Plan(model=TINY, data={"days": {"rows": rows}}, steps={
        "forced": {"kind": "forward", "input": ["days.prefix", "days.bare"], "interventions": {"logits": {
            "site": HEAD, "pos": {"scope": {"column": "days.bare", "occurrence": -1, "before": True}}}}},
        "p": {"kind": "op", "op": "token_prob", "of": "forced.logits", "token": "days.bare"}}, saves=["p"]).compile()
    refused_before_any_forward(engines, plan, r"^p: days\.bare row 0: position 4 predicts 'day' in the input, "
                                              r"which is not token 0 of 'day' as it tokenizes alone \(\['▁day'\]\); "
                                              r"the answer is scored as it tokenizes, so the input says it the same "
                                              r"way$")


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_strip_and_lower_at_the_point_of_use(engines, kind):
    """The proposal's example E: `"  New York "` finds `New York` in the input under lower and strip, and is
    scored as ▁New ▁York (its whitespace-only end tokens dropped), each against its class."""
    rows = [{"input": "The Statue of Liberty stands in New York", "city": "  New York "}]
    city = {"column": "days.city", "strip": True, "lower": True}
    plan = Plan(model=TINY, data={"days": {"rows": rows}}, steps={
        "read": {"kind": "forward", "input": "days.input", "interventions": {"logits": {
            "site": HEAD, "pos": {"scope": {"column": city, "before": True}}}}},
        "hit": {"kind": "op", "op": "match", "of": "read.logits", "expected": city},
        "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": city}},
        saves=["hit", "p", "read.logits"]).compile()
    filled = run(engines, plan, kind)
    assert record(filled).positions["read.logits"] == [[8, 9]]
    tokenizer = engines(kind, plan.model).tokenizer
    classes = [answers.spellings(tokenizer, True, True)[w] for w in ("new", "york")]
    probs = filled.results["read.logits"][0].softmax(-1)
    torch.testing.assert_close(filled.results["p"][0], torch.stack([probs[i, c].sum() for i, c in
                                                                    enumerate(classes)]))


def test_a_column_with_an_answer_of_several_tokens_lines_up_every_row(engines):
    """One row of several tokens makes the whole column line up, a one-token row included; a column all of one
    token scores every position of its row."""
    def plan(rows):
        return doc({"read": {"kind": "forward", "input": "days.input",
                             "interventions": {"logits": {"site": HEAD, "pos": {"index": "-3:"}}}},
                    "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": "days.answer"}}, ["p"], rows)
    refused_before_any_forward(engines, plan(DAYS), r"^p: days\.answer row 1: ' Friday' is 1 token; the read has 3 "
                                                    r"positions on that row")
    one = run(engines, plan([{**r, "answer": " Friday"} for r in DAYS])).results["p"]
    assert one.shape == (2, 3)


def test_the_minimum_over_every_position_of_a_value_with_none_is_missing():
    empty = torch.zeros(2, 0)
    assert ops.call("min", empty, {"over": ["rows", "pos"]}, dims=["rows", "pos"]).isnan().all()


def test_a_token_that_is_part_of_a_character_is_its_own_class(engines):
    tokenizer = engines("nnterp", plan_of("gpt2_reach").model).tokenizer
    parts = [i for i, text in enumerate(answers.vocabulary(tokenizer)) if "�" in text]
    assert len(parts) > 1
    assert all(answers.spelled(tokenizer, i, True, True) == [i] for i in parts)
