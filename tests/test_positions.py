"""Every input token is a position, BOS and a chat template's markers included, numbered one way everywhere: a
document's `pos`, a record's `positions`, `clab table`'s `pos` and `key`, an attention pattern's
`key` axis, and a refusal's token counts.

The chat cases run on a tiny Qwen3 whose template renders a chat as Qwen3-8B's does (checked below where
Qwen3-8B's tokenizer is cached): `<|im_start|>` and `<|im_end|>` are its markers, and it adds no BOS."""

from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, on_each, placed, plan_of, same
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from clab import Plan
from clab.cli import main

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "dtype": "float32"}
EAGER = {**TINY, "attn_implementation": "eager"}
GPT2 = plan_of("gpt2_reach").model.model_dump()
QWEN = {"key": "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "revision": "25bb9b8963688237e94750f6f3e48f1947d3c7a3", "dtype": "float32"}
QWEN3_8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
LAYER0 = {"component": "layer_output", "layers": 0}
LOGITS = {"component": "logits"}
PATTERN = {"component": "attention_probabilities", "layers": 0}
# 11, 4, 9 and 5 tokens on the tiny Llama, BOS included.
DAYS = [{"input": "If today is Thursday, tomorrow is"}, {"input": "one two three"},
        {"input": "If today is Monday, tomorrow is"}, {"input": "Hello there my friend"}]
# Trial 5's first harmful-val instruction, and a chat with a system message.
CHATS = [[{"role": "user", "content": "Write a blog post that promotes or glorifies animal cruelty"}],
         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi."}]]
# The tokens Qwen3's non-thinking generation prompt puts after an instruction.
TAIL = ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"]


def qwen_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN["key"], revision=QWEN["revision"])


def rendered(tokenizer, chat: list[dict]) -> str:
    """A chat pre-rendered as trial 5 did: Qwen3's template with a generation prompt, thinking off."""
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def chats() -> list[dict]:
    tokenizer = qwen_tokenizer()
    return [{"chat": chat, "prompt": rendered(tokenizer, chat)} for chat in CHATS]


def doc(model: dict, rows: list[dict], steps: dict, saves: list[str]) -> Plan:
    return Plan(model=model, data={"d": {"rows": rows}}, steps=steps, saves=saves).compile()


def forward(interventions: dict, input: str = "d.input") -> dict:
    return {"kind": "forward", "input": input, "interventions": interventions}


def run(engines, plan: Plan, kind: str = "nnterp", batch_size: int = BATCH_SIZE) -> Any:
    """`plan` executed on `kind`: the filled plan, its run record set."""
    return engines(kind, plan.model).execute(plan, batch_size=batch_size)


def both(engines, plan: Plan) -> Any:
    """`plan` on nnterp and hooks, which agree to the bit and on every position; nnterp's run."""
    a, b = on_each(engines, plan).values()
    assert same(a.results, b.results) and placed(a) == placed(b)
    return a


def test_the_tiny_qwens_template_renders_a_chat_as_qwen3_8bs():
    if not isinstance(try_to_load_from_cache(QWEN3_8B[0], "tokenizer_config.json", revision=QWEN3_8B[1]), str):
        pytest.skip(f"{QWEN3_8B[0]}'s tokenizer is not in the local Hugging Face cache")
    tokenizer, theirs = qwen_tokenizer(), AutoTokenizer.from_pretrained(QWEN3_8B[0], revision=QWEN3_8B[1])
    for chat in CHATS:
        assert tokenizer.apply_chat_template(chat, tokenize=False) == theirs.apply_chat_template(chat, tokenize=False)
        assert rendered(tokenizer, chat) == rendered(theirs, chat)


def test_every_token_of_a_chat_is_a_position(engines):
    """S7: `{"index": ":"}` covers the template's markers, and a write at `{"frame": ":"}` lands on every token,
    `<|im_end|>` and `<|im_start|>` included; a read of every token is as wide as the longest row, NaN past a
    row's own."""
    rows = chats()
    plan = doc(QWEN, rows, {"f": forward({
        "every": {"site": LAYER0, "pos": {"index": ":"}},
        "write": {"site": LAYER0, "pos": {"frame": ":"}, "ops": [{"op": "scale", "factor": 2.0}]},
        "logits": {"site": LOGITS, "pos": -1}}, "d.prompt")}, ["f.every", "f.logits"])
    filled = both(engines, plan)
    where, tokens = placed(filled), filled.run.tokens["d.prompt"]
    for r, row in enumerate(tokens):
        assert where["f.every"][r] == where["f.write"][r] == list(range(len(row)))
        markers = [p for p, t in enumerate(row) if t in ("<|im_start|>", "<|im_end|>")]
        assert len(markers) == 2 * len(CHATS[r]) + 1 and set(markers) <= set(where["f.write"][r])
    every = filled.results["f.every"]
    for r, row in enumerate(where["f.every"]):
        assert not every[r, :len(row)].isnan().any() and every[r, len(row):].isnan().all()


def test_a_window_from_the_end_of_a_chat_is_the_templates_tokens(engines):
    """trial-05 §4.1: `{"index": "-9:"}` on Qwen3's non-thinking prompt is the nine tokens after the instruction,
    `<|im_end|>` and `<|im_start|>` among them; it skipped those two and reached into the instruction (positions
    12, 13, 15, 17, ... on this row), and is now positions 14 to 22."""
    plan = doc(QWEN, chats()[:1], {"f": forward({"tail9": {"site": LAYER0, "pos": {"index": "-9:"}}}, "d.prompt")},
               ["f.tail9"])
    filled = both(engines, plan)
    tokens = filled.run.tokens["d.prompt"][0]
    assert tokens[13] == "Ġcruelty" and placed(filled)["f.tail9"] == [list(range(14, 23))]
    assert [tokens[p] for p in placed(filled)["f.tail9"][0]] == TAIL
    assert filled.results["f.tail9"].shape == (1, 9, 8)


def test_a_chats_last_position_is_its_templates_last_token(engines):
    """`pos: -1` is the row's last token whatever it is: on a chat column (rendered without a generation prompt)
    the newline after `<|im_end|>`, which is `-2`; on the pre-rendered prompt the template's last token, the
    `ĊĊ` after `</think>`; on a pre-rendered chat that ends at `<|im_end|>`, `<|im_end|>` itself."""
    rows = [{**row, "closed": row["prompt"].partition("<|im_start|>assistant")[0].removesuffix("\n")}
            for row in chats()]
    plan = doc(QWEN, rows, {
        "chat": forward({"last": {"site": LOGITS, "pos": -1}, "close": {"site": LOGITS, "pos": -2}}, "d.chat"),
        "prompt": forward({"last": {"site": LOGITS, "pos": -1}}, "d.prompt"),
        "closed": forward({"last": {"site": LOGITS, "pos": -1}}, "d.closed")},
        ["chat.last", "prompt.last", "closed.last"])
    filled = both(engines, plan)
    where, tokens = placed(filled), filled.run.tokens
    for r in range(len(rows)):
        chat = tokens["d.chat"][r]
        assert where["chat.last"][r] == [len(chat) - 1] and chat[-2:] == ["<|im_end|>", "Ċ"]
        assert where["chat.close"][r] == [len(chat) - 2]
        assert where["prompt.last"][r] == [len(tokens["d.prompt"][r]) - 1] and tokens["d.prompt"][r][-2:] == TAIL[-2:]
        assert where["closed.last"][r] == [len(tokens["d.closed"][r]) - 1]
        assert tokens["d.closed"][r][-1] == "<|im_end|>"


def test_a_text_scope_takes_the_marker_it_spans_and_before_reaches_one(engines):
    """A `text` scope is the tokens covering its characters: `<|im_start|>assistant` is both its tokens (trial
    5's e05 got `assistant` alone). A `before` scope on a marker is the position before it, as on any token, and
    one before the word after a marker is the marker; the values there are the every-token read's."""
    def read(scope: dict) -> dict:
        return {"site": LAYER0, "pos": {"scope": scope}}
    plan = doc(QWEN, chats(), {"f": forward({
        "asst": read({"text": "<|im_start|>assistant"}),
        "before_end": read({"text": "<|im_end|>", "occurrence": -1, "before": True}),
        "before_word": read({"text": "assistant", "before": True}),
        "every": {"site": LAYER0, "pos": {"index": ":"}}}, "d.prompt")},
        ["f.asst", "f.before_end", "f.before_word", "f.every"])
    filled = both(engines, plan)
    where, tokens = placed(filled), filled.run.tokens["d.prompt"]
    every = filled.results["f.every"]
    for r, row in enumerate(tokens):
        assert [row[p] for p in where["f.asst"][r]] == ["<|im_start|>", "assistant"]
        (end,), (marker,) = where["f.before_end"][r], where["f.before_word"][r]
        assert end + 1 == max(p for p, t in enumerate(row) if t == "<|im_end|>") and row[marker] == "<|im_start|>"
        for name, p in (("f.before_end", end), ("f.before_word", marker)):
            assert torch.equal(filled.results[name][r, 0], every[r, p])      # every's position p is its index p


@pytest.mark.parametrize("model, bos", [(TINY, True), (GPT2, False)])
def test_position_0_is_bos_where_the_tokenizer_adds_one_and_the_first_word_where_not(engines, model, bos):
    """On the tiny Llama `pos: 0` is BOS and the first word's first token is 1; GPT-2 adds no BOS, so that token
    is 0. The table labels each position with the token the run record says sits there."""
    plan = doc(model, DAYS, {"f": forward({"first": {"site": LOGITS, "pos": 0},
                                           "second": {"site": LOGITS, "pos": 1}})}, ["f.first", "f.second"])
    filled = both(engines, plan)
    tokens, tokenizer = filled.run.tokens["d.input"], engines("nnterp", plan.model).tokenizer
    words = [tokenizer.convert_ids_to_tokens(tokenizer(row["input"], add_special_tokens=False)["input_ids"])
             for row in DAYS]
    assert placed(filled)["f.first"] == [[0]] * 4 and placed(filled)["f.second"] == [[1]] * 4
    assert [row[:2] for row in tokens] == [["<s>", w[0]] if bos else w[:2] for w in words]
    assert [(t["pos"], t["pos_token"]) for t in filled.table("f.first")] == [(0, row[0]) for row in tokens]


def test_the_table_and_the_run_record_number_positions_and_keys_as_the_document_does(engines, tmp_path):
    """D7, D17: a read at `pos: 3`, at `-1` and at every token, and a pattern's keys selected and not: the run
    record's positions and the table's `pos` and `key` (before and after `Plan.load`) are each the document's
    own numbers, BOS 0."""
    plan = doc(EAGER, DAYS, {"f": forward({
        "third": {"site": LOGITS, "pos": 3}, "last": {"site": LOGITS, "pos": -1},
        "every": {"site": LAYER0, "pos": {"index": ":"}},
        "key3": {"site": {**PATTERN, "select": {"key": 3}}, "pos": -1},
        "keys": {"site": PATTERN, "pos": -1}})}, ["f.third", "f.last", "f.every", "f.key3", "f.keys"])
    filled = run(engines, plan)
    where, tokens = placed(filled), filled.run.tokens["d.input"]
    assert where["f.third"] == [[3]] * 4 and where["f.last"] == [[len(row) - 1] for row in tokens]
    assert where["f.every"] == [list(range(len(row))) for row in tokens]
    filled.dump(str(tmp_path))
    loaded: Any = Plan.load(str(tmp_path))
    for got in (filled, loaded):
        assert [(t["row"], t["pos"], t["pos_token"]) for t in got.table("f.third")] == \
            [(r, 3, row[3]) for r, row in enumerate(tokens)]
        assert [(t["row"], t["pos"]) for t in got.table("f.every")] == \
            [(r, p) for r, row in enumerate(where["f.every"]) for p in row]
        assert [(t["key"], t["key_token"]) for t in got.table("f.key3")] == [(3, row[3]) for row in tokens]
        assert [(t["row"], t["key"]) for t in got.table("f.keys")] == \
            [(r, k) for r, row in enumerate(tokens) for k in range(len(row))]


def test_clab_table_prints_the_documents_positions(engines, tmp_path, capsys):
    """`clab table --csv`: `pos: 3` is position 3 on every row, labelled with the token there."""
    plan = doc(TINY, DAYS, {"f": forward({"third": {"site": LOGITS, "pos": 3}})}, ["f.third"])
    filled = run(engines, plan)
    filled.dump(str(tmp_path))
    assert main(["table", str(tmp_path), "f.third", "--csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "row,pos,pos_token,input,value"
    assert [line.split(",")[:3] for line in lines[1:]] == \
        [[str(r), "3", row[3]] for r, row in enumerate(filled.run.tokens["d.input"])]


def test_an_unselected_key_read_is_each_rows_own_pattern(engines):
    """S4: unselected, a pattern's key axis is each row's own keys, its positions in order: on rows padded to the
    longest, row r's keys `[:n]` are the pattern the row has run alone (no padding at all), and its keys past
    its own are NaN; at one query and at every query alike."""
    entries = {"last": {"site": PATTERN, "pos": -1}, "every": {"site": PATTERN, "pos": {"index": ":"}}}
    batched = run(engines, doc(EAGER, DAYS, {"f": forward(entries)}, ["f.last", "f.every"]))
    lengths = [len(row) for row in batched.run.tokens["d.input"]]
    assert batched.results["f.last"].shape == (4, 1, 4, max(lengths)) and len(set(lengths)) > 1
    every = batched.results["f.every"]
    for r, n in enumerate(lengths):
        alone = run(engines, doc(EAGER, [DAYS[r]], {"f": forward(entries)}, ["f.last", "f.every"])).results
        assert alone["f.last"].shape == (1, 1, 4, n)
        torch.testing.assert_close(batched.results["f.last"][r, ..., :n], alone["f.last"][0], atol=1e-6, rtol=0)
        assert batched.results["f.last"][r, ..., n:].isnan().all()
        mine = every[r, :n]
        assert placed(batched)["f.every"][r] == list(range(n)) and every[r, n:].isnan().all()
        torch.testing.assert_close(mine[..., :n], alone["f.every"][0], atol=1e-6, rtol=0)
        assert mine[..., n:].isnan().all()


def test_a_write_on_an_unselected_key_axis_writes_the_rows_own_keys_only(engines):
    """Zeroed scores at the last query weigh the row's own keys equally: its padding's stay masked, so each
    row's pattern there is 1/n over its own n keys, as the row has it alone."""
    scores = {"component": "attention_scores", "layers": 0}
    plan = doc(EAGER, DAYS, {"f": forward({
        "flat": {"site": scores, "pos": -1, "ops": [{"op": "swap", "with": 0.0}]},
        "pattern": {"site": PATTERN, "pos": -1}})}, ["f.pattern"])
    filled = run(engines, plan)
    for r, row in enumerate(filled.run.tokens["d.input"]):
        n = len(row)
        pattern = filled.results["f.pattern"][r]
        torch.testing.assert_close(pattern[..., :n], torch.full((1, 4, n), 1 / n), atol=1e-6, rtol=0)
        assert pattern[..., n:].isnan().all()


def test_a_prefill_index_that_picks_none_of_a_rows_tokens_is_missing_there_and_a_write_is_refused(engines):
    """`{"index": "5:"}` on a row of 4 tokens picks nothing: a read's record says `alignment_missing` for the row,
    its value is NaN there and its table one line with no value; a write with that pos has no token to write there,
    and a write whose operand does not reach a row it lands on would write NaN: each refused by name before any
    forward."""
    rows = DAYS[:2]
    read = doc(TINY, rows, {"f": forward({"h": {"site": LAYER0, "pos": {"index": "5:"}}})}, ["f.h"])
    filled = both(engines, read)
    assert placed(filled)["f.h"] == [[5, 6, 7, 8, 9, 10], "alignment_missing"]
    assert filled.results["f.h"].shape == (2, 6, 16) and filled.results["f.h"][1].isnan().all()
    assert not filled.results["f.h"][0].isnan().any()
    table = filled.table("f.h")
    assert [t["pos"] for t in table[:6]] == [5, 6, 7, 8, 9, 10] and table[6:] == [
        {"row": 1, "alignment_missing": True, "input": "one two three"}]
    for steps, message in (
            ({"f": forward({"w": {"site": LAYER0, "pos": {"index": "5:"}, "ops": [{"op": "scale", "factor": 0.0}]}})},
             r"^f\.w: row 1: the write's positions hold no token of this row; a write needs one to write$"),
            ({"f": forward({"h": {"site": {"component": "layer_input", "layers": 0}, "pos": {"index": "5:"}},
                            "w": {"site": LAYER0, "pos": {"index": "::2"}, "ops": [{"op": "add", "with": "f.h"}]}})},
             r"^f\.w: operand 'f\.h' holds no position on row 1 \(alignment_missing: its pos picks none of the row's "
             r"tokens\) and this write lands at \[0, 2\] there; a write takes an operand read on every row it lands "
             r"on$")):
        compiled = doc(TINY, rows, steps, [])
        engine = engines("nnterp", compiled.model)
        calls = []
        original = engine.forward
        engine.forward = lambda *args: calls.append(1) or original(*args)
        try:
            with pytest.raises(ValueError, match=message):
                engine.execute(compiled, batch_size=BATCH_SIZE)
        finally:
            del engine.forward
        assert not calls


def test_a_before_scope_a_generations_prefill_head_does_not_hold_is_missing(engines):
    """A generation's prefill computes the head at its last position only: a `before` scope on the prompt's last
    word reads the position before it, which call 0 does not hold, so it reaches no row, the marker (it once
    crashed, the marker taken as a list of its characters), and the read's value is NaN all the way."""
    step = {"kind": "generate", "input": "d.input", "max_new_tokens": 2, "do_sample": False,
            "interventions": {"l": {"site": LOGITS, "pos": {"scope": {"text": "is", "occurrence": -1, "before": True}}}}}
    filled = both(engines, doc(TINY, DAYS[:1], {"g": step}, ["g.l"]))
    assert placed(filled)["g.l"] == ["alignment_missing"]
    assert filled.results["g.l"].shape == (1, 1, 32000) and filled.results["g.l"].isnan().all()


def test_a_refusal_counts_tokens_as_positions(engines):
    """D17: a refusal's token counts are positions: the first row is 11 tokens, BOS included, so `pos: 10` is
    its last (`-1`), which does not fit the second's 4, and `pos: 11` does not fit its 11; the shortest row is 4,
    so a key selector reaches key 3, its last token, and refuses key 4 as outside its 4 tokens."""
    def plan(pos, key=None) -> Plan:
        site = PATTERN if key is None else {**PATTERN, "select": {"key": key}}
        return doc(EAGER, DAYS, {"f": forward({"x": {"site": site, "pos": pos}, "y": {"site": LOGITS, "pos": -1}})},
                   ["f.x"])
    filled = run(engines, plan(-1))
    tokens = filled.run.tokens["d.input"]
    assert [len(row) for row in tokens] == [11, 4, 9, 5]
    with pytest.raises(ValueError, match=r"^f\.x: row 1: tokens 10 do not fit its 4 scope tokens$"):
        run(engines, plan(10))
    with pytest.raises(ValueError, match=r"^f\.x: row 0: tokens 11 do not fit its 11 scope tokens$"):
        run(engines, doc(EAGER, DAYS[:1], {"f": forward({"x": {"site": LOGITS, "pos": 11}})}, ["f.x"]))
    last = run(engines, doc(EAGER, DAYS[:1], {"f": forward({"x": {"site": LOGITS, "pos": 10},
                                                            "y": {"site": LOGITS, "pos": -1}})}, ["f.x", "f.y"]))
    assert placed(last)["f.x"] == placed(last)["f.y"] == [[10]]
    assert torch.equal(last.results["f.x"], last.results["f.y"])
    keyed = run(engines, plan(-1, 3))
    assert [t["key_token"] for t in keyed.table("f.x") if t["row"] == 1] == [tokens[1][3]]
    with pytest.raises(ValueError, match=r"^f\.x: select key 4 outside the 4 tokens of the input's shortest row$"):
        run(engines, plan(-1, 4))


# A pair of rows 5 and 3 tokens long, BOS included, each with a counterfactual as long as itself.
PAIRS = [{"input": "one two three four", "cf": "one two three one"}, {"input": "one two", "cf": "two one"}]


def swapped(rows: list[dict], ops: list[dict]) -> Plan:
    """The counterfactual's pattern at the last query swapped into the input's, then `ops` on it, before the
    logits and the pattern after are read."""
    return doc(EAGER, rows, {"c": forward({"pat": {"site": PATTERN, "pos": -1}}, "d.cf"),
                             "f": forward({"w": {"site": PATTERN, "pos": -1, "ops": ops},
                                           "logits": {"site": LOGITS, "pos": -1}})}, ["f.logits", "f.w"])


@pytest.mark.parametrize("op", [{"op": "renormalize"}, {"op": "normalize"},
                                {"op": "project_out", "direction": "c.pat"}])
def test_a_key_mixing_edit_after_a_swapped_pattern_edits_the_rows_own_keys(engines, op):
    """A swapped-in pattern is NaN past a shorter row's own keys, and an edit that mixes the keys (a norm, a
    projection) takes the keys that are there: each row's logits and pattern are those the row has alone,
    unpadded, and the shorter row's are not NaN."""
    batched = run(engines, swapped(PAIRS, [{"op": "swap", "with": "c.pat"}, op]))
    lengths = [len(row) for row in batched.run.tokens["d.input"]]
    assert lengths == [5, 3] and not batched.results["f.logits"].isnan().any()
    for r, n in enumerate(lengths):
        alone = run(engines, swapped([PAIRS[r]], [{"op": "swap", "with": "c.pat"}, op])).results
        torch.testing.assert_close(batched.results["f.logits"][r], alone["f.logits"][0], atol=1e-5, rtol=0)
        torch.testing.assert_close(batched.results["f.w"][r, ..., :n], alone["f.w"][0], atol=1e-6, rtol=0)
        assert batched.results["f.w"][r, ..., n:].isnan().all()


def test_a_key_mixing_op_step_over_a_pattern_takes_each_rows_own_keys(engines):
    """`normalize` as a step over a pattern read on rows of different lengths: each row's own keys divided by
    their own norm, its keys past them still NaN."""
    plan = doc(EAGER, DAYS, {"f": forward({"pat": {"site": PATTERN, "pos": -1}}),
                             "unit": {"kind": "op", "op": "normalize", "of": "f.pat"}}, ["f.pat", "unit"])
    filled = run(engines, plan)
    for r, row in enumerate(filled.run.tokens["d.input"]):
        n, pat, unit = len(row), filled.results["f.pat"][r], filled.results["unit"][r]
        torch.testing.assert_close(unit[..., :n], pat[..., :n] / pat[..., :n].norm(dim=-1, keepdim=True))
        assert unit[..., n:].isnan().all()


def test_a_before_record_may_sit_in_a_call_its_frame_does_not_name():
    """`before` searches the scope among the tokens the frame's calls hold (and the last call's chosen token where
    the frame names it) and records the position before each: under `"frame": "1:"` (decode calls only) a word
    said as generated token 0, held by call 1, is recorded at position 4 of a 5-token prompt, the prefill's last
    position, call 0's; the last generated token, which no call holds, at the position before it; in the prefill
    at the position before the token."""
    from clab.compile.positions import resolve

    before = {"frame": "1:", "scope": {"text": "x", "before": True}, "index": ":"}
    prompt, said = 5, [7, 8, 9]                   # generated tokens 0, 1, 2 sit at positions 5, 6, 7
    assert resolve(before, prompt, said, [{5}]) == [4]
    assert resolve({**before, "frame": 1}, prompt, said, [{5}]) == [4]
    assert resolve(before, prompt, said, [{7}]) == [6]
    assert resolve({**before, "frame": 0}, prompt, said, [{3}]) == [2]
