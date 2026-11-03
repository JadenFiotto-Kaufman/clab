"""Padding as a run option: `dataset` (the default) pads every batch to the input's longest row and is the exact run
the rest of the suite pins; `batch` pads each batch to its own longest row; `sorted` batches a step's rows longest
first and puts every value back in dataset row order.

Every corpus document under `batch` and under `sorted` gives every saved value close to the `dataset` run's (`RTOL`,
`ATOL`), the same rows missing and the same NaN along `pos`, its entries landing where they did, and the same
generated tokens; on both engines, which agree to the bit under each padding, and the same bits on a second run. A
`sorted` run is a `batch` run of its rows in sorted order, put back row for row. A `key` axis holds each row's own
keys, as many as the input's longest row has, whatever the batch's width. The run record says how it padded, and
`batch` and `sorted` feed fewer tokens than `dataset` of the same real ones.
"""

import copy
import os
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, INSIDE, NAMES, edited, placed, same

from clab import Plan
from clab.cli import main
from clab.format.data import PADDINGS

# Measured on the corpus at the pinned 12 threads, against the `dataset` run: the largest move is 6.9e-7, in das's
# trained basis under `sorted` (its minibatches' gradients sum over their rows in another order, every epoch), and
# 1.8e-7 under `batch` (sae_train's dictionary); a value of a forward or a generation moves by 6e-8 at most
# (mixed_lengths' logits). The worst |Δ| / (ATOL + RTOL·|x|) is 0.36 under `sorted` and 0.04 under `batch`.
RTOL, ATOL = 1e-5, 1e-6
MODES = ("batch", "sorted")


def counts(run: Any, key: str) -> int:
    """A run's cost `key` (`tokens`, `real`) summed over its steps and their points."""
    return sum(c[key] for by in run.costs.values() for c in (by.values() if "forwards" not in by else [by]))


@pytest.fixture(scope="module")
def ran(engines, corpus) -> dict:
    """Every corpus document, from the corpus copy that holds the bundles some of them load, on each engine that
    serves it under each padding, and once more on nnterp under `batch` and `sorted`: each run's results, where its
    entries landed, its tokens and its costs."""
    out = {}
    for name in NAMES:
        for kind in ("nnterp", "hooks"):
            if kind == "hooks" and name in INSIDE:
                continue
            for padding in PADDINGS:
                for again in ((False, True) if kind == "nnterp" and padding != "dataset" else (False,)):
                    plan = Plan(os.path.join(corpus, "documents", f"{name}.json")).compile()
                    filled = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE, padding=padding)
                    assert filled.run is not None and filled.run.padding == padding
                    out[name, kind, padding, again] = (copy.deepcopy(filled.results), placed(filled),
                                                       filled.run.tokens, filled.run)
    return out


def tensors(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """A result's leaves by path: each tensor, and each other value (text, a list of tokens) whole."""
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in tensors(v, f"{path}.{k}")]
    if isinstance(value, list) and any(isinstance(v, (torch.Tensor, dict)) for v in value):
        return [leaf for i, v in enumerate(value) for leaf in tensors(v, f"{path}[{i}]")]
    return [(path, value)]


@pytest.mark.parametrize("padding", MODES)
@pytest.mark.parametrize("name", NAMES)
def test_every_value_is_close_to_the_exact_runs_its_missing_rows_and_positions_the_same(ran, name, padding):
    exact, got = ran[name, "nnterp", "dataset", False], ran[name, "nnterp", padding, False]
    assert got[0].keys() == exact[0].keys()
    for (path, x), (_, y) in zip(tensors(exact[0]), tensors(got[0]), strict=True):
        if isinstance(x, torch.Tensor):     # NaN where the exact run has NaN: a missing row, a position past a row's
            torch.testing.assert_close(y, x, rtol=RTOL, atol=ATOL, equal_nan=True, msg=lambda m: f"{path}: {m}")
        else:
            assert y == x, path
    assert got[1] == exact[1]               # every entry's positions per row, alignment_missing rows among them
    assert got[2] == exact[2]               # every input's tokens and what each generation's rows said


def test_the_corpus_compares_missing_rows_nan_positions_and_key_axes(ran):
    """The comparison above meets each thing padding could move: a row a read beyond the prefill never reached
    (generated_answer), NaN past a row's own positions (forced_answer's ragged read), and an attention pattern's key
    axis (attention_knockout's `clean.pattern`): over the weekday prompts, 11, 9, 9 and 9 tokens long, the second
    batch is 9 tokens wide under `batch` and `sorted`, and the pattern is still 11 keys wide, NaN past the 9-token
    rows' own."""
    missing = [n for n in NAMES if "alignment_missing" in str(ran[n, "nnterp", "dataset", False][1])]
    assert "generated_answer" in missing
    nan = {n for n in NAMES for _, x in tensors(ran[n, "nnterp", "dataset", False][0])
           if isinstance(x, torch.Tensor) and x.is_floating_point() and x.isnan().any()}
    assert {"generated_answer", "forced_answer", "attention_knockout"} <= nan
    exact = ran["attention_knockout", "nnterp", "dataset", False][3]
    for padding in MODES:
        assert counts(ran["attention_knockout", "nnterp", padding, False][3], "tokens") < counts(exact, "tokens")
        pattern = ran["attention_knockout", "nnterp", padding, False][0]["clean.pattern"]
        assert pattern.shape[-1] == 11 and pattern[1:, ..., 9:].isnan().all()
        assert not pattern[0].isnan().any() and not pattern[1:, ..., :9].isnan().any()


@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("name", [n for n in NAMES if n not in INSIDE])
def test_the_engines_agree_to_the_bit_under_each_padding(ran, name, padding):
    nnterp, hooks = ran[name, "nnterp", padding, False], ran[name, "hooks", padding, False]
    assert same(nnterp[0], hooks[0]) and nnterp[1] == hooks[1] and nnterp[2] == hooks[2]


@pytest.mark.parametrize("padding", MODES)
@pytest.mark.parametrize("name", NAMES)
def test_a_document_gives_the_same_bits_again_at_one_batch_size_and_padding(ran, name, padding):
    first, again = ran[name, "nnterp", padding, False], ran[name, "nnterp", padding, True]
    assert same(first[0], again[0]) and first[1] == again[1]


@pytest.mark.parametrize("name", NAMES)
def test_batch_and_sorted_feed_no_more_tokens_and_the_same_real_ones(ran, name):
    """Of the same real tokens, `batch` feeds no more than `dataset` and `sorted` no more than `batch`: sorted rows
    put rows of near lengths in one batch, the shortest in the last."""
    runs = {p: ran[name, "nnterp", p, False][3] for p in PADDINGS}
    assert len({counts(run, "real") for run in runs.values()}) == 1
    assert counts(runs["sorted"], "tokens") <= counts(runs["batch"], "tokens") <= counts(runs["dataset"], "tokens")


# GPT-2 prompts (no BOS) whose inputs and counterfactuals are both longer on every next row, 1, 4, 12 and 14 tokens
# and 2, 3, 6 and 17: `sorted` runs them last row first, in the batches a `batch` run of them reversed makes.
GROWING = [
    {"input": "one", "counterfactual_inputs": ["two"], "base_answer": " two", "cf_answer": " three"},
    {"input": "four, then", "counterfactual_inputs": ["one, two"], "base_answer": " one", "cf_answer": " two"},
    {"input": "If the count is one, next is", "counterfactual_inputs": ["The count is four"],
     "base_answer": " two", "cf_answer": " four"},
    {"input": "If the count is two, then the one after it is",
     "counterfactual_inputs": ["If the count is three and it goes on, next is"], "base_answer": " three",
     "cf_answer": " one"},
]


def reach(rows: list[dict]) -> Plan:
    """GPT-2 at layer 1, the last token of one prompt swapped into another's, over inline `rows`; it saves the
    scores, the swapped vectors, and the patched forward's every position at the layer, NaN past a shorter row's."""
    def change(d):
        d["data"] = {"pairs": {"rows": rows}}
        d["sites"]["target"]["layers"] = 1
        d["steps"]["patched"]["interventions"]["resid"] = {"site": "target", "pos": -1}
        d["steps"]["patched"]["interventions"]["every"] = {"site": "target"}
        d["saves"] = ["logit_diff", "patched.resid", "patched.every", "counterfactual.v_cf"]
    return edited("gpt2_reach", change).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_sorted_puts_every_row_back_in_dataset_order(engines, kind):
    """A value's row i is the dataset's row i under `sorted`: a run of rows that grow row by row equals, to the bit
    and row for row, a `batch` run of the same rows reversed, read back reversed; each row's positions are its own,
    and the ragged read is NaN past each row's own tokens, on the same rows."""
    engine = engines(kind, reach(GROWING).model)
    ordered = engine.execute(reach(GROWING), batch_size=BATCH_SIZE, padding="sorted")
    sorted_ = {k: v.clone() for k, v in ordered.results.items()}
    landed = placed(ordered)
    reversed_ = engine.execute(reach(GROWING[::-1]), batch_size=BATCH_SIZE, padding="batch")
    back = torch.arange(len(GROWING) - 1, -1, -1)
    for key, value in reversed_.results.items():
        assert same(sorted_[key], value[back]), key
    assert {k: v["pairs"][::-1] if isinstance(v, dict) else v[::-1] for k, v in placed(reversed_).items()} == landed
    every = sorted_["patched.every"]
    assert every.shape[1] == 14 and [int((~every[i, :, 0].isnan()).sum()) for i in range(4)] == [1, 4, 12, 14]
    assert ordered.run is not None and reversed_.run is not None
    assert counts(ordered.run, "tokens") == counts(reversed_.run, "tokens") < \
        counts(engine.execute(reach(GROWING), batch_size=BATCH_SIZE).run, "tokens")


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_sorted_puts_each_tensor_of_a_top_k_read_back_in_dataset_order(engines, kind):
    """A `top_k` read is a dict of tensors, `ids` and `probs`; under `sorted` each is put back row for row, as one
    tensor is."""
    def top(rows: list[dict]) -> Plan:
        def change(d):
            d["data"] = {"pairs": {"rows": rows}}
            d["steps"]["patched"]["interventions"]["top"] = {"site": {"component": "lm_head_output"}, "pos": -1,
                                                            "ops": [{"op": "top_k", "k": 2}]}
            d["saves"] = ["patched.top"]
        return edited("gpt2_reach", change).compile()
    engine = engines(kind, top(GROWING).model)
    ordered = engine.execute(top(GROWING), batch_size=BATCH_SIZE, padding="sorted").results["patched.top"]
    reversed_ = engine.execute(top(GROWING[::-1]), batch_size=BATCH_SIZE, padding="batch").results["patched.top"]
    back = torch.arange(len(GROWING) - 1, -1, -1)
    assert set(ordered) == {"ids", "probs"} and all(same(ordered[k], reversed_[k][back]) for k in ordered)


def two_widths() -> Plan:
    """attention_pattern_patching's head 0 over prompts of 11, 11, 9 and 9 tokens, each the same length as its
    counterfactual: at batch size 2 the second batch is 9 tokens wide under `batch` padding, 11 under `dataset`."""
    def row(day: str, cf: str) -> dict:
        return {"input": f"If today is {day}, tomorrow is", "counterfactual_inputs": [f"If today is {cf}, tomorrow is"],
                "base_answer": " Friday", "cf_answer": " Sunday"}

    def change(d):
        d["data"] = {"pairs": {"rows": [row("Thursday", "Thursday"), row("Thursday", "Thursday"),
                                        row("Monday", "Friday"), row("Friday", "Monday")]}}
        d["sites"]["one_head"]["select"] = {"num_heads": 0}
        d["saves"] = ["logit_diff", "counterfactual.their_pattern", "patched.look_there"]
    return edited("attention_pattern_patching", change).compile()


@pytest.mark.parametrize("padding", MODES)
def test_a_key_axis_is_each_rows_own_keys_as_wide_as_the_longest_row_whatever_the_batchs_width(engines, padding):
    """A pattern read in one forward and written in another: under per-batch widths the read is still 11 keys wide,
    each row's own keys then NaN, and the write over a 9-token batch takes its 11-key operand, as under `dataset`."""
    plan = two_widths()
    engine = engines("nnterp", plan.model)
    exact = {k: v.clone() for k, v in engine.execute(plan, batch_size=BATCH_SIZE).results.items()}
    got = engine.execute(two_widths(), batch_size=BATCH_SIZE, padding=padding)
    assert got.run is not None and counts(got.run, "tokens") < 4 * 11 * 2
    pattern = got.results["counterfactual.their_pattern"]
    assert pattern.shape == (4, 1, 1, 11) and pattern[2:, ..., 9:].isnan().all() and not pattern[:2].isnan().any()
    for key, value in exact.items():
        torch.testing.assert_close(got.results[key], value, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_the_run_records_its_padding_and_show_prints_it(engines, tmp_path, capsys):
    plan = reach(GROWING)
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE, padding="sorted").dump(str(tmp_path))
    back = Plan.load(str(tmp_path))
    assert back.run is not None and back.run.padding == "sorted"
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert ", batch size 2, padding sorted, started " in shown[0]
    assert shown[3].split()[-2:] == ["tokens", "real"]


def test_padding_is_refused_by_name_unless_it_is_one_of_the_three(engines, capsys):
    plan = reach(GROWING)
    with pytest.raises(ValueError, match=r"^padding 'rows': one of 'dataset', 'batch', 'sorted' \(--padding\)$"):
        engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE, padding="rows")
    document = os.path.join(os.path.dirname(__file__), "..", "documents", "patching.json")
    assert main(["run", document, "--padding", "rows"]) == 1
    assert capsys.readouterr().err == "clab: padding 'rows': one of 'dataset', 'batch', 'sorted' (--padding)\n"
    assert main(["run", document, "-o", "padding=batch"]) == 1
    assert capsys.readouterr().err == "clab: -o padding: it is given as --padding\n"
