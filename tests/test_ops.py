"""The implemented ops on plain tensors, and the site machinery that runs an entry's records at a site."""

import math
import warnings

import numpy
import pytest
import torch

from conftest import BATCH_SIZE, edited, placed, same

from clab import ops
from clab.ops import edits, metrics, reductions, transforms
from clab.steps import batches

torch.manual_seed(0)
X = torch.randn(3, 2, 5)                       # rows, pos, hidden
LOGITS = torch.randn(3, 1, 7)                  # rows, pos, vocab
IDS = torch.tensor([1, 4, 6])
ONE = IDS[:, None]                              # a token argument: per row, the ids it may be
DIMS = ["rows", "pos", "hidden_size"]


def test_every_op_is_listed_once_in_order_under_its_own_name():
    assert list(ops.OPS) == ["swap", "add", "sub", "scale", "renormalize", "normalize", "project_out", "decode",
                             "apply", "noise", "clamp", "readout", "encode", "grad", "mean", "sum", "min", "max", "std",
                             "median", "count", "norm", "dot", "entropy", "pca", "match", "logit_diff", "token_prob",
                             "token_logit", "cross_entropy", "contains", "kl", "js", "top_k", "argmax", "detokenize",
                             "permute", "abs", "exp", "log", "sign", "sqrt", "square", "neg", "mul", "div", "pow", "eq", "ne",
                             "gt", "ge", "lt", "le", "cosine"]
    assert all(fn.__name__ == name for name, fn in ops.OPS.items())
    assert set(ops.DECLARED) == set(ops.OPS)


def test_each_op_declares_its_group_and_bundle():
    groups = {name: ops.DECLARED[name]["group"] for name in ops.OPS}
    assert {n for n, g in groups.items() if g == "edits"} == {"swap", "add", "sub", "scale", "renormalize",
                                                               "normalize", "project_out", "decode", "apply", "noise",
                                                               "clamp"}
    assert {n for n, g in groups.items() if g == "reductions"} == {"mean", "sum", "min", "max", "std", "median",
                                                                    "count", "norm", "dot", "entropy", "pca"}
    assert {n for n, g in groups.items() if g == "transforms"} == {"readout", "encode", "grad", "top_k", "argmax",
                                                                    "detokenize", "permute", "abs", "exp", "log", "sign", "sqrt",
                                                                    "square", "neg", "mul", "div", "pow", "eq", "ne",
                                                                    "gt", "ge", "lt", "le"}
    assert {n for n, g in groups.items() if g == "metrics"} == {"match", "logit_diff", "token_prob", "token_logit",
                                                                 "cross_entropy", "contains", "kl", "js", "cosine"}
    assert {n: d["bundle"] for n, d in ops.DECLARED.items() if d["bundle"]} == {"pca": "pca"}


def test_an_ops_arguments_bind_to_its_signature():
    ops.check("match", {"expected": "pairs.cf_answer"})
    with pytest.raises(ValueError, match="'match' needs 'expected'"):
        ops.check("match", {})
    with pytest.raises(ValueError, match="'add' takes no 'factor'; its arguments: with, scale"):
        ops.check("add", {"with": "x", "factor": 2.0})
    with pytest.raises(ValueError, match="k: 'three' is not int"):
        ops.check("top_k", {"k": "three"})


def test_swap_broadcasts_a_vector_or_a_number_exactly():
    v = torch.randn(5)
    assert torch.equal(edits.swap(X, v), v.expand(3, 2, 5))
    assert torch.equal(edits.swap(X, 0.0), torch.zeros_like(X))
    assert torch.equal(edits.swap(X, X.flip(0)), X.flip(0))


def test_add_sub_scale():
    v = torch.randn(5)
    assert torch.equal(edits.add(X, v, scale=4.0), X + 4.0 * v)
    assert torch.equal(edits.sub(X, v), X - v)
    assert torch.equal(edits.scale(X, 0.5), X * 0.5)


def test_renormalize_restores_the_original_norm():
    out = edits.renormalize(X * 3 + 1, original=X)
    torch.testing.assert_close(out.norm(dim=-1), X.norm(dim=-1))
    edited = X * 3 + 1                  # norm "l2", the default, is the rescaling it always was, to the bit
    assert torch.equal(edits.renormalize(edited, original=X, norm="l2"), out)
    assert torch.equal(out, edited * (X.norm(dim=-1, keepdim=True) / edited.norm(dim=-1, keepdim=True)))


def test_renormalize_by_sum_restores_each_rows_sum_over_the_entries_that_are_there():
    """`norm: "sum"` divides so each vector sums to what `original`'s did: a pattern row with a key knocked out
    is a distribution again, and the keys past a row's own (NaN) stay missing."""
    pattern = torch.rand(2, 3, 6).softmax(-1)
    pattern[1, :, 4:] = float("nan")                    # row 1 has four keys
    pattern[1, :, :4] = pattern[1, :, :4].softmax(-1)
    knocked = pattern.clone()
    knocked[..., 0] = 0
    out = edits.renormalize(knocked, original=pattern, norm="sum")
    torch.testing.assert_close(out.nansum(-1), torch.ones(2, 3))
    assert out[1, :, 4:].isnan().all() and not out[:, :, :4].isnan().any() and (out[..., 0] == 0).all()
    torch.testing.assert_close(out[0], knocked[0] / knocked[0].sum(-1, keepdim=True))
    assert not torch.allclose(edits.renormalize(knocked, original=pattern).nansum(-1), torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"^norm: 'l1' is not one of 'l2', 'sum'$"):
        ops.check("renormalize", {"norm": "l1"})


def test_normalize_and_project_out():
    torch.testing.assert_close(edits.normalize(X).norm(dim=-1), torch.ones(3, 2))
    d = torch.randn(5)
    out = edits.project_out(X, d)
    torch.testing.assert_close(out @ d, torch.zeros(3, 2), atol=1e-6, rtol=0)
    # `direction` is normalised first: a thousand times longer, the same result to float rounding.
    torch.testing.assert_close(edits.project_out(X, 1000 * d), out, atol=1e-6, rtol=0)


def test_a_rows_pos_factor_scales_each_positions_vector_by_its_own_number(engines):
    """`scale` with a `[rows, pos]` factor, a probability read at three positions, broadcasts over `hidden_size`
    (Ops' one rule, `batches.argument_binding`): the written value is the stream times each position's own number."""
    from conftest import BATCH_SIZE, edited

    def change(d):
        d["sites"]["head"] = {"component": "lm_head_output"}
        d["steps"] = {
            "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
                "h": {"site": "target", "pos": {"index": "-3:"}},
                "p": {"site": "head", "pos": {"index": "-3:"},
                      "ops": [{"op": "token_prob", "token": "pairs.base_answer"}]}}},
            "scaled": {"kind": "forward", "input": "pairs.input", "interventions": {
                "patch": {"site": "target", "pos": {"index": "-3:"}, "ops": [{"op": "scale", "factor": "clean.p"}]}}}}
        d["saves"] = ["clean.h", "clean.p", "scaled.patch"]
    plan = edited("patching", change).compile()
    results = engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    h, p, patch = results["clean.h"], results["clean.p"], results["scaled.patch"]
    assert h.shape == patch.shape == (4, 3, 16) and p.shape == (4, 3)
    assert torch.equal(patch, h * p[..., None])


def test_a_zero_vector_normalizes_to_itself_and_a_zero_direction_projects_nothing_out():
    """As `renormalize` leaves a vector of norm zero: `normalize` divides by nothing (no NaN), and `project_out`
    along a zero direction removes no component; the other vectors are edited as before."""
    zeroed = X.clone()
    zeroed[1, 0] = 0
    unit = edits.normalize(zeroed)
    assert torch.equal(unit[1, 0], torch.zeros(5)) and not unit.isnan().any()
    torch.testing.assert_close(unit[0].norm(dim=-1), torch.ones(2))
    assert torch.equal(edits.project_out(X, torch.zeros(5)), X)


def test_mean_over_the_example_axes_by_default():
    assert torch.equal(reductions.mean(X, dims=["rows", "pos", "hidden_size"]), X.mean(dim=[0, 1]))
    assert torch.equal(reductions.mean(X, dims=["rows", "pos", "hidden_size"], over=["pos"]), X.mean(dim=1))


def test_a_mean_skips_the_missing_and_otherwise_has_the_plain_means_bits():
    """With none missing a mean is the plain mean to the bit, with some the mean of those there, the NaN-skipping
    mean's; a value nothing is there of is missing. The same with a gradient through it, which reaches each value
    that is there as the plain mean's or the NaN-skipping mean's does."""
    holed = X.clone()
    holed[1, 0] = math.nan
    for value in (X, holed, torch.full((2, 3), math.nan)):
        dims = ["rows", "pos", "hidden_size"][:value.dim()]
        want = value.nanmean(dim=[0, 1]) if value.isnan().any() else value.mean(dim=[0, 1])
        assert same(reductions.mean(value, dims=dims), want)
        leaf, theirs = value.clone().requires_grad_(), value.clone().requires_grad_()
        got = reductions.mean(leaf, dims=dims)
        assert same(got.detach(), want)
        (theirs.nanmean(dim=[0, 1]) if value.isnan().any() else theirs.mean(dim=[0, 1])).sum().backward()
        got.sum().backward()
        assert same(leaf.grad, theirs.grad)


def test_metrics_on_logits():
    d = ["rows", "pos", "vocab_size"]
    assert torch.equal(metrics.match(LOGITS, LOGITS.argmax(-1)[:, :1], d), torch.ones(3, 1))
    assert torch.equal(metrics.match(LOGITS, (LOGITS.argmax(-1)[:, :1] + 1) % 7, d), torch.zeros(3, 1))
    picked = LOGITS[torch.arange(3), 0, IDS][:, None]
    assert torch.equal(metrics.token_logit(LOGITS, ONE, d), picked)
    assert torch.equal(metrics.logit_diff(LOGITS, ONE, ONE, d), torch.zeros(3, 1))
    torch.testing.assert_close(metrics.token_prob(LOGITS, ONE, d), LOGITS.softmax(-1)[torch.arange(3), 0, IDS][:, None])
    torch.testing.assert_close(metrics.cross_entropy(LOGITS, ONE, d),
                               torch.nn.functional.cross_entropy(LOGITS[:, 0], IDS, reduction="none")[:, None])


def test_a_set_of_ids_is_one_answer_any_of_whose_ids_is_it():
    """A row's ids padded with -1: `match` hits on any, `token_prob` sums them, `cross_entropy` is minus the log
    of that sum."""
    d = ["rows", "pos", "vocab_size"]
    sets = torch.tensor([[1, 2], [4, -1], [6, 0]])
    p = LOGITS.softmax(-1)[:, 0]
    mass = torch.stack([p[0, 1] + p[0, 2], p[1, 4], p[2, 6] + p[2, 0]])[:, None]
    torch.testing.assert_close(metrics.token_prob(LOGITS, sets, d), mass)
    torch.testing.assert_close(metrics.cross_entropy(LOGITS, sets, d), -mass.log())
    top = LOGITS.argmax(-1)[:, 0]
    assert torch.equal(metrics.match(LOGITS, torch.stack([top, torch.full((3,), -1)], -1), d), torch.ones(3, 1))


def test_token_i_of_an_answer_scores_position_i():
    """`[rows, pos, m]` ids: each position its own token."""
    d = ["rows", "pos", "vocab_size"]
    logits = torch.randn(2, 3, 7)
    ids = torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])
    assert torch.equal(metrics.token_logit(logits, ids, d), logits.gather(-1, ids).squeeze(-1))


def test_sum_and_min_skip_what_is_missing_and_a_row_with_nothing_stays_missing():
    nan = float("nan")
    x = torch.tensor([[1.0, 2.0, nan], [nan, nan, nan], [3.0, -1.0, 0.5]])
    d = ["rows", "pos"]
    s, m = reductions.sum(x, d, over=["pos"]), reductions.min(x, d, over=["pos"])
    assert s[[0, 2]].tolist() == [3.0, 2.5] and s[1].isnan()
    assert m[[0, 2]].tolist() == [1.0, -1.0] and m[1].isnan()


def test_a_value_with_missing_positions_reduces_over_the_ones_that_are_there():
    """A row's positions past its own are NaN: over `pos` a row with none is missing, over rows and pos the
    mean is over every position there is."""
    nan = float("nan")
    value = torch.tensor([[1.0, 2.0], [nan, nan], [5.0, nan]])
    assert ops.out_dims("sum", ["rows", "pos"], {"over": ["pos"]}) == ["rows"]
    assert ops.out_dims("sum", ["rows", "pos"], {"over": ["rows", "pos"]}) == []
    with pytest.raises(ValueError, match="a reduction cannot remove 'layers'"):
        ops.out_dims("sum", ["rows", "pos"], {"over": ["layers"]})
    per_row = ops.call("sum", value, {"over": ["pos"]}, dims=["rows", "pos"])
    assert per_row[[0, 2]].tolist() == [3.0, 5.0] and per_row[1].isnan()
    assert ops.call("mean", value, {"over": ["rows", "pos"]}, dims=["rows", "pos"]).item() == pytest.approx(8 / 3)


def test_contains_finds_the_answer_in_each_rows_text():
    said = ["It is Friday.", "monday then", "nothing"]
    answers = [(["Friday"], False, False), (["Monday", " Tuesday"], False, True), ([" nothing "], True, False)]
    assert metrics.contains(said, answers, ["rows"]).tolist() == [1.0, 1.0, 1.0]
    assert metrics.contains(said, [(["friday"], False, False)] * 3, ["rows"]).tolist() == [0.0, 0.0, 0.0]


def test_metrics_on_a_value_without_positions_take_each_rows_own_token():
    rows = LOGITS[:, 0]                                   # rows, vocab
    assert torch.equal(metrics.token_logit(rows, ONE, ["rows", "vocab_size"]), rows[torch.arange(3), IDS])
    assert torch.equal(metrics.match(rows, rows.argmax(-1)[:, None], ["rows", "vocab_size"]), torch.ones(3))


def test_metrics_broadcast_over_a_leading_layer_axis():
    stacked = torch.stack([LOGITS, LOGITS * 2])          # layers, rows, pos, vocab
    out = metrics.token_logit(stacked, ONE, ["layers", "rows", "pos", "vocab_size"])
    assert out.shape == (2, 3, 1)
    assert torch.equal(out[1], 2 * metrics.token_logit(LOGITS, ONE, ["rows", "pos", "vocab_size"]))


def test_kl_and_js():
    other = torch.randn(3, 1, 7)
    p, q = LOGITS.log_softmax(-1), other.log_softmax(-1)
    torch.testing.assert_close(metrics.kl(LOGITS, other),
                               torch.nn.functional.kl_div(q, p, log_target=True, reduction="none").sum(-1))
    assert torch.equal(metrics.kl(LOGITS, LOGITS), torch.zeros(3, 1))
    js = metrics.js(LOGITS, other)
    torch.testing.assert_close(js, metrics.js(other, LOGITS))
    assert bool((js >= 0).all() and (js <= math.log(2)).all())


def test_kl_reversed_is_kl_with_of_and_against_swapped_to_the_bit():
    """`reverse` gives KL(against ‖ of): the bits of `kl` with the two swapped, on a value without a `layers` axis and
    on one with it against a distribution without it (laid out as an operand meets the value, a singleton where it
    leaves the axis out); the two directions differ."""
    other = torch.randn(3, 1, 7)
    assert torch.equal(metrics.kl(LOGITS, other, reverse=True), metrics.kl(other, LOGITS))
    assert not torch.equal(metrics.kl(LOGITS, other, reverse=True), metrics.kl(LOGITS, other))
    lens, final = torch.randn(2, 3, 1, 7), other[None]                 # layers, rows, pos, vocab
    assert torch.equal(metrics.kl(lens, final, reverse=True), metrics.kl(final, lens))
    assert metrics.kl(lens, final, reverse=True).shape == (2, 3, 1)
    assert torch.equal(ops.call("kl", lens, {"against": final, "reverse": True}), metrics.kl(final, lens))


def test_out_dims_follows_each_ops_declaration():
    assert ops.out_dims("match", ["rows", "pos", "vocab_size"], {}) == ["rows", "pos"]
    assert ops.out_dims("mean", ["rows", "pos", "hidden_size"], {"over": None}) == ["hidden_size"]
    assert ops.out_dims("readout", ["layers", "rows", "pos", "hidden_size"], {}) == ["layers", "rows", "pos", "vocab_size"]
    with pytest.raises(ValueError, match="needs a value with a 'vocab_size' or 'classes' axis"):
        ops.out_dims("match", ["rows", "pos", "hidden_size"], {})


def taken(value: torch.Tensor, component: str, rec: dict) -> torch.Tensor:
    """What a read record at a site takes of `value`."""
    got: dict = {}
    assert not batches.visit_site(value, component, [{**rec, "name": "r"}], got)[1]
    return got["r"]


def record(start: int, width: int, value, ops_=None) -> dict:
    positions = torch.arange(start, start + width).expand(3, width)
    return {"positions": positions, "select": {},
            "ops": ops_ or [{"op": "swap", "args": {"with": value}, "dims": DIMS}]}


def test_grad_gives_back_the_gradient_it_meets_in_its_layout():
    """The run puts the gradient in the value's place before the entry's ops run; `grad` hands it on as it is."""
    assert transforms.grad(X, of_="metric") is X
    assert ops.out_dims("grad", DIMS, {"of": "metric"}) == DIMS


def test_a_capture_hands_on_a_leaf_whose_gradient_differentiate_takes():
    """`visit_site` puts the value the forward goes on with in `got` whole, a leaf that requires grad, says to set it
    back, and does a later write on a copy; `differentiate` replaces the capture with the gradient of the sum of its
    `of` (a NaN row adding nothing), at the record's positions, and detaches every value: here d sum((2 x)^2) / dx
    at the written position, 8 x."""
    value = torch.randn(2, 3, 4)                    # batch, seq, hidden
    capture = {"name": "g@0", "capture": True, "positions": torch.tensor([[2], [2]]), "select": {}}
    double = {"positions": torch.tensor([[2], [2]]), "select": {},
              "ops": [{"op": "scale", "args": {"factor": 2.0}, "dims": DIMS}]}
    got: dict = {}
    with torch.enable_grad():
        out, wrote = batches.visit_site(value, "layer_output", [capture, double], got)
        assert wrote and got["g@0"].is_leaf and got["g@0"].requires_grad and torch.equal(got["g@0"], value)
        assert torch.equal(out[:, 2], 2 * value[:, 2]) and torch.equal(got["g@0"], value)    # the leaf unwritten
        got["m@None"] = (out[:, 2:3] ** 2).sum(-1)
    got["m@None"] = torch.where(torch.tensor([[True], [False]]), got["m@None"], torch.nan)
    batches.differentiate(got, [{"of": [("m@None", lambda v: v)], "wrt": [capture]}])
    torch.testing.assert_close(got["g@0"][0], 8 * value[0, 2:3])
    assert torch.equal(got["g@0"][1], torch.zeros(1, 4))
    assert not any(v.requires_grad for v in got.values())


def test_a_ragged_write_writes_only_the_positions_it_keeps():
    value = torch.randn(2, 4, 5)
    keep = torch.tensor([[True, True], [True, False]])
    ragged = {"positions": torch.tensor([[1, 2], [3, 3]]), "keep": keep, "kept": keep.nonzero().T,
              "select": {}, "ops": [{"op": "swap", "args": {"with": 0.0}, "dims": DIMS}]}
    out = batches.visit_site(value, "layer_output", [ragged], {})[0]
    assert torch.equal(out[0, 1:3], torch.zeros(2, 5)) and torch.equal(out[1, 3], torch.zeros(5))
    unwritten = torch.tensor([[1, 0, 0, 1], [1, 1, 1, 0]], dtype=torch.bool)
    assert torch.equal(out[unwritten], value[unwritten])
    empty = {**ragged, "keep": torch.zeros(2, 2, dtype=torch.bool), "kept": torch.zeros(2, 2).nonzero().T}
    assert torch.equal(batches.visit_site(value, "layer_output", [empty], {})[0], value)


def test_a_batch_of_equal_rows_binds_no_mask_and_writes_as_an_all_true_one():
    """`selection_in_batch` gives `keep` only to a batch whose rows hold different numbers of positions; one whose rows
    hold as many (equal-width reads and writes, the common case) indexes by its positions alone, with no mask the device
    must count before the host goes on, and writes what an all-true mask writes, to the bit."""
    from conftest import plan_of

    from clab.plan import entries
    from clab.format.sweeps import points

    record = entries(points(plan_of("patching").steps["patched"])[0])["patch"]
    even = batches.selection_in_batch(record, {}, [[1], [2], [0]], [0, 1, 2], 4, False)
    ragged = batches.selection_in_batch(record, {}, [[1, 2], [3], "missing"], [0, 1, 0], 4, False)
    assert "keep" not in even and ragged["keep"].tolist() == [[True, True], [True, False], [False, False]]
    assert "kept" not in even and ragged["kept"].tolist() == [[0, 0, 1], [0, 1, 0]]         # rows, then positions
    value, other = torch.randn(3, 4, 5), torch.randn(3, 1, 5)
    write = {**even, "ops": [{"op": "swap", "args": {"with": other}, "dims": DIMS}]}
    keep = torch.ones(3, 1, dtype=torch.bool)
    masked = batches.visit_site(value, "layer_output", [{**write, "keep": keep, "kept": keep.nonzero().T}], {})[0]
    assert torch.equal(batches.visit_site(value, "layer_output", [write], {})[0], masked)


def test_a_window_write_and_three_single_writes_are_one_tensor():
    value = torch.randn(3, 6, 5)                         # batch, seq, hidden
    other = torch.randn(3, 6, 5)
    window = batches.visit_site(value, "layer_output", [record(2, 3, other[:, 2:5])], {})[0]
    singles = batches.visit_site(value, "layer_output", [record(i, 1, other[:, i:i + 1]) for i in (2, 3, 4)], {})[0]
    assert torch.equal(window, singles)
    assert torch.equal(window[:, 2:5], other[:, 2:5]) and torch.equal(window[:, :2], value[:, :2])


def test_writes_compose_in_order_and_renormalize_sees_the_value_before_them():
    value = torch.randn(3, 4, 5)
    steer = record(3, 1, None, [{"op": "add", "args": {"with": torch.ones(5), "scale": 4.0}, "dims": DIMS}])
    restore = {**record(3, 1, None, [{"op": "renormalize", "args": {}, "dims": DIMS}]), "takes": {"original"}}
    out = batches.visit_site(value, "layer_output", [steer, restore], {})[0]
    torch.testing.assert_close(out[:, 3].norm(dim=-1), value[:, 3].norm(dim=-1))
    assert not torch.equal(out[:, 3], value[:, 3])
    assert torch.equal(out[:, :3], value[:, :3])


def test_select_narrows_one_axis_and_write_puts_it_back():
    value = torch.randn(2, 3, 4, 2)                      # batch, seq, num_heads, head_dim (attention_premix)
    head = {"positions": torch.tensor([[2], [2]]), "select": {2: torch.tensor([1])}}
    assert torch.equal(taken(value, "attention_premix", head), value[:, 2:3, 1:2])
    out = batches.visit_site(value, "attention_premix", [{**head, "ops": [
        {"op": "swap", "args": {"with": 0.0}, "dims": ["rows", "pos", "num_heads", "head_dim"]}]}], {})[0]
    assert torch.equal(out[:, 2, 1], torch.zeros(2, 2))
    out[:, 2, 1] = value[:, 2, 1]
    assert torch.equal(out, value)


def test_a_pattern_is_selected_along_its_query_axis():
    value = torch.randn(2, 4, 5, 5)                      # batch, num_heads, query, key
    rec = {"positions": torch.tensor([[4], [4]]), "select": {}}
    assert torch.equal(taken(value, "attention_probabilities", rec), value[:, :, 4:5].movedim(2, 1))


def test_pca_gives_an_orthonormal_basis_of_the_largest_variance_first():
    torch.manual_seed(1)
    x = torch.randn(40, 5) * torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1]) + 2.0
    got = reductions.pca(x, k=3, dims=["rows", "hidden_size"])
    basis, mean = got["basis"], got["mean"]
    assert basis.shape == (5, 3) and mean.shape == (5,)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(mean, x.mean(0))
    variance = ((x - mean) @ basis).var(0)
    assert bool((variance[:-1] >= variance[1:]).all())
    assert torch.equal(reductions.pca(x, k=3, dims=["rows", "hidden_size"])["basis"], basis)     # deterministic signs
    with pytest.raises(ValueError, match="pca of 3 vectors has no rank-3 basis"):
        reductions.pca(x[:3], k=3, dims=["rows", "hidden_size"])


def test_pca_of_a_bfloat16_value_is_a_float32_bundle_orthonormal_to_float32s_precision():
    """A bundle is float32 whatever the value's dtype, as a fit's trained featurizers are: the basis of a bfloat16
    value is its double-precision SVD rounded once to float32, not to bfloat16's few bits."""
    torch.manual_seed(1)
    x = (torch.randn(200, 64) * torch.linspace(4.0, 0.1, 64) + 1.0).bfloat16()
    got = reductions.pca(x, k=16, dims=["rows", "hidden_size"])
    assert {k: v.dtype for k, v in got.items()} == {"basis": torch.float32, "mean": torch.float32}
    assert (got["basis"].T @ got["basis"] - torch.eye(16)).abs().max() < 1e-6
    exact = reductions.pca(x.double(), k=16, dims=["rows", "hidden_size"])
    assert all(torch.equal(got[k], exact[k]) for k in got)


def test_top_k_and_argmax():
    d = ["rows", "pos", "vocab_size"]
    got = transforms.top_k(LOGITS, k=2, dims=d)
    assert got["ids"].shape == got["probs"].shape == (3, 1, 2)
    assert torch.equal(got["ids"][..., 0], transforms.argmax(LOGITS, d))
    torch.testing.assert_close(got["probs"][..., 0], LOGITS.softmax(-1).max(-1).values)


def test_top_k_and_argmax_rank_the_axis_over_names_wherever_it_is():
    """`over` names the axis ranked; the result keeps the others in order, `top_k`'s `k` last."""
    assert torch.equal(transforms.argmax(X, DIMS, over="pos"), X.argmax(1))
    got = transforms.top_k(X, 2, DIMS, over="rows")
    assert got["ids"].shape == (2, 5, 2) and torch.equal(got["ids"], X.movedim(0, -1).topk(2).indices)
    assert ops.out_dims("top_k", DIMS, {"k": 2, "over": "rows"}) == ["pos", "hidden_size", "k"]
    assert ops.out_dims("argmax", DIMS, {"over": "hidden_size"}) == ["rows", "pos"]


def test_top_k_over_a_list_of_axes_ranks_their_entries_as_one_axis():
    """Over `["rows", "pos"]` top_k ranks the (row, position) pairs as one axis laid out rows then positions: its
    values are a `top_k` of the flattened pair's, its `ids` per axis recover where each pick is, and both axes leave
    the layout. Over a vocabulary among the axes, each pick's `probs` is its softmax over that vocabulary."""
    x = torch.randn(4, 3, 5)                                           # features, rows, pos
    d = ["features", "rows", "pos"]
    got = transforms.top_k(x, 4, d, over=["rows", "pos"])
    flat = transforms.top_k(x.reshape(4, 15), 4, ["features", "pair"], over="pair")
    assert set(got) == {"ids", "values"} and list(got["ids"]) == ["rows", "pos"]
    assert torch.equal(got["values"], flat["values"])
    assert torch.equal(got["ids"]["rows"] * 5 + got["ids"]["pos"], flat["ids"])
    assert torch.equal(x[torch.arange(4)[:, None], got["ids"]["rows"], got["ids"]["pos"]], got["values"])
    assert ops.out_dims("top_k", d, {"k": 4, "over": ["rows", "pos"]}) == ["features", "k"]
    assert ops.out_dims("top_k", d, {"k": 4, "over": ["pos", "features"]}) == ["rows", "k"]
    assert torch.equal(transforms.top_k(x, 4, d, over=["pos", "rows"])["values"], got["values"])
    logits = torch.randn(3, 2, 7)                                       # rows, pos, vocab
    top = transforms.top_k(logits, 3, ["rows", "pos", "vocab_size"], over=["pos", "vocab_size"])
    rows = torch.arange(3)[:, None]
    assert torch.equal(top["probs"], logits.float().softmax(-1)[rows, top["ids"]["pos"], top["ids"]["vocab_size"]])
    assert torch.equal(logits[rows, top["ids"]["pos"], top["ids"]["vocab_size"]],
                       logits.reshape(3, 14).topk(3).values)
    with pytest.raises(ValueError, match=r"^over \['pos', 'pos'\]: a list of axes names each axis it ranks once$"):
        ops.out_dims("top_k", d, {"k": 1, "over": ["pos", "pos"]})
    with pytest.raises(ValueError, match=r"^over: \['pos'\] is not str$"):
        ops.check("argmax", {"over": ["pos"]})


def test_top_k_over_a_list_breaks_a_tie_by_the_first_axis_then_the_next_and_ranks_a_missing_entry_last():
    """An exact tie goes to the lower index along the first axis listed, then the next; a missing (NaN) entry is
    ranked below every other, and a pick past those there is -1 on every axis, NaN in `values`."""
    nan = float("nan")
    x = torch.tensor([[1.0, 2.0, nan], [2.0, nan, nan]])                # rows, pos
    got = transforms.top_k(x, 5, ["rows", "pos"], over=["rows", "pos"])
    assert got["ids"]["rows"].tolist() == [0, 1, 0, -1, -1] and got["ids"]["pos"].tolist() == [1, 0, 0, -1, -1]
    assert got["values"][:3].tolist() == [2.0, 2.0, 1.0] and got["values"][3:].isnan().all()
    swapped = transforms.top_k(x, 5, ["rows", "pos"], over=["pos", "rows"])
    assert swapped["ids"]["rows"].tolist() == [1, 0, 0, -1, -1] and swapped["ids"]["pos"].tolist() == [0, 1, 0, -1, -1]


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class Engine:
    tokenizer = Tokenizer()
    eos = [0]


def test_detokenize_stops_at_each_rows_first_eos():
    ids = torch.tensor([[5, 6, 0, 7], [8, 9, 9, 9]])
    assert transforms.detokenize(ids, engine=Engine(), eos=[0]) == ["<5><6>", "<8><9><9><9>"]
    assert transforms.detokenize(ids, engine=Engine(), eos=[0], tokens=True) == [["t5", "t6"], ["t8", "t9", "t9", "t9"]]
    assert transforms.detokenize(ids, engine=Engine(), eos=[6, 9]) == ["<5>", "<8>"]


def test_a_missing_row_is_missing_all_the_way_down():
    """Row 1 is NaN, as a rectangular generated-frame read holds a row that never reached its step: every
    metric of it is NaN, argmax and top_k give it id -1, and mean and pca skip it."""
    d = ["rows", "pos", "vocab_size"]
    logits = LOGITS.clone()
    logits[1] = float("nan")
    hit = metrics.match(logits, logits.nan_to_num().argmax(-1)[:, :1], d)
    assert hit[[0, 2]].eq(1).all() and hit[1].isnan().all()
    assert metrics.token_prob(logits, ONE, d)[1].isnan().all() and metrics.logit_diff(logits, ONE, ONE, d)[1].isnan().all()
    assert transforms.argmax(logits, d).tolist() == [[LOGITS[0, 0].argmax().item()], [-1],
                                                     [LOGITS[2, 0].argmax().item()]]
    top = transforms.top_k(logits, 2, d)
    assert top["ids"][1].eq(-1).all() and top["probs"][1].isnan().all() and top["ids"][0].ge(0).all()
    assert torch.equal(reductions.mean(hit, dims=["rows", "pos"]), torch.tensor(1.0))
    x = X.clone()
    x[1] = float("nan")
    torch.testing.assert_close(reductions.mean(x, dims=DIMS), X[[0, 2]].mean((0, 1)))
    assert torch.equal(reductions.mean(X, dims=DIMS), X.mean((0, 1)))                    # no missing row: the plain mean
    big = torch.randn(8, 2, 5)
    hole = torch.cat([big, torch.full((1, 2, 5), float("nan"))])
    assert all(torch.equal(a, b) for a, b in zip(reductions.pca(hole, 2, DIMS).values(), reductions.pca(big, 2, DIMS).values()))


def test_noise_is_drawn_per_row_and_position_from_the_seed_alone():
    """The draw at a row's position is a CPU generator's seeded by (seed, row, position): the same whichever
    rows share the value, the same for a value without a pos axis at its row's one position, never the same
    at two rows or two positions, cast to the value's dtype."""
    rows, places = torch.tensor([4, 5, 6]), torch.tensor([[7, 8], [7, 8], [2, 9]])
    drawn = torch.stack([torch.stack([torch.randn(5, generator=edits.seeded(3, int(rows[i]), int(places[i, j])))
                                      for j in range(2)]) for i in range(3)])
    got = edits.noise(X, 0.25, DIMS, seed=3, rows=rows, positions=places)
    assert torch.equal(got, X + 0.25 * drawn)
    assert torch.equal(edits.noise(X[1:], 0.25, DIMS, seed=3, rows=rows[1:], positions=places[1:]), got[1:])
    assert len({tuple(d.tolist()) for d in drawn.reshape(6, 5)}) == 6
    assert not torch.equal(edits.noise(X, 0.25, DIMS, seed=4, rows=rows, positions=places), got)
    half = X.to(torch.bfloat16)
    assert torch.equal(edits.noise(half, 0.25, DIMS, seed=3, rows=rows, positions=places),
                       half + 0.25 * drawn.to(torch.bfloat16))
    first = edits.noise(X[:, 0], 0.5, ["rows", "hidden_size"], seed=3, rows=rows, positions=places[:, 0])
    assert torch.equal(first, X[:, 0] + 0.5 * drawn[:, 0])
    assert torch.equal(edits.noise(X, 1.0, DIMS), X + torch.randn(X.shape, generator=edits.seeded(0)))  # no rows


def test_clamp_takes_either_bound_and_needs_one():
    assert torch.equal(edits.clamp(X, min_=-0.5), X.clamp(min=-0.5))
    assert torch.equal(edits.clamp(X, max_=0.5), X.clamp(max=0.5))
    assert torch.equal(edits.clamp(X, -0.1, 0.2), X.clamp(-0.1, 0.2))
    ops.check("clamp", {"max": 1})
    with pytest.raises(ValueError, match="'clamp' needs 'min' or 'max', or both"):
        ops.check("clamp", {})
    with pytest.raises(ValueError, match="min: 'low' is not float or null"):
        ops.check("clamp", {"min": "low"})
    with pytest.raises(ValueError, match="'clamp' min 1.0 is greater than its max 0.5"):
        ops.check("clamp", {"min": 1.0, "max": 0.5})
    ops.check("clamp", {"min": {"sweep": [0.0, 1.0]}, "max": 0.5})          # a sweep is checked per point


def test_cosine_is_over_the_hidden_axis_one_number_per_row_and_position():
    other = torch.randn(3, 2, 5)
    expected = (X * other).sum(-1) / (X.norm(dim=-1) * other.norm(dim=-1))
    torch.testing.assert_close(metrics.cosine(X, other), expected)
    torch.testing.assert_close(metrics.cosine(X, 2 * X), torch.ones(3, 2))
    assert ops.out_dims("cosine", ["layers", "rows", "pos", "hidden_size"], {}) == ["layers", "rows", "pos"]
    with pytest.raises(ValueError, match="'cosine' needs a value with a 'hidden_size' axis"):
        ops.out_dims("cosine", ["rows", "pos", "vocab_size"], {})


# -- arithmetic, comparisons and the reductions beside `mean` ---------------------------------------------------------

NAN = float("nan")


def test_the_elementwise_transforms_are_their_formulas_in_float32_and_keep_a_missing_entry_missing():
    """Each is torch's own function of the float32 value, its layout kept; a NaN stays NaN (`sign`'s too, which
    torch makes 0, and `pow`'s on either side, which torch makes 1 at `NaN ** 0` and `1 ** NaN`); a bfloat16 value
    comes back float32."""
    x = X.clone()
    x[1, 0] = NAN
    for name, expected in {"abs": x.abs(), "exp": x.exp(), "log": x.log(), "sqrt": x.sqrt(), "square": x * x,
                           "neg": -x, "sign": x.sign().masked_fill(x.isnan(), NAN)}.items():
        assert same(ops.call(name, x, {}), expected), name
        assert ops.out_dims(name, DIMS, {}) == DIMS
        assert ops.call(name, x.bfloat16(), {}).dtype == torch.float32
    assert same(transforms.pow(x, 3), x.pow(3)) and torch.equal(transforms.pow(X, 2), transforms.square(X))
    assert torch.equal(transforms.pow(X.abs(), 0.5), transforms.sqrt(X.abs()))
    assert same(transforms.pow(x, X.flip(0)), x.pow(X.flip(0)))           # a named value, entry by entry
    assert same(transforms.pow(torch.tensor([NAN, 1.0]), 0), torch.tensor([NAN, 1.0]))
    assert same(transforms.pow(torch.tensor([1.0, 1.0]), torch.tensor([NAN, 2.0])), torch.tensor([NAN, 1.0]))
    assert same(transforms.mul(x, X.flip(0)), x * X.flip(0)) and torch.equal(transforms.mul(X, 2), X * 2)
    assert transforms.mul(x.bfloat16(), 2).dtype == torch.float32 and ops.out_dims("mul", DIMS, {"with": 2}) == DIMS


def test_div_is_nan_wherever_its_divisor_is_zero_and_div_then_scale_is_the_value():
    y = torch.randn(3, 2, 5)
    y[0, 1, 2] = 0.0
    got = transforms.div(X, y)
    assert got[0, 1, 2].isnan() and got.isnan().sum() == 1
    assert torch.equal(got.masked_fill(got.isnan(), 0), (X / y).masked_fill(y == 0, 0))
    assert transforms.div(X, 0).isnan().all() and torch.equal(transforms.div(X, 2), X / 2)
    back = edits.scale(got, y)
    held = y != 0
    assert ((back - X).abs() <= 2 * torch.finfo(torch.float32).eps * X.abs())[held].all()


def test_comparisons_give_0_or_1_in_the_values_dtype_and_nan_where_either_side_is_missing():
    x = torch.tensor([[1.0, 2.0, NAN, 4.0]])
    other = torch.tensor([[2.0, 2.0, 0.0, NAN]])
    for name, holds in {"eq": torch.eq, "ne": torch.ne, "gt": torch.gt, "ge": torch.ge, "lt": torch.lt,
                        "le": torch.le}.items():
        got = ops.call(name, x, {"with": other})
        assert same(got, holds(x, other).float().masked_fill(x.isnan() | other.isnan(), NAN)), name
        assert ops.call(name, x.bfloat16(), {"with": 2}).dtype == torch.bfloat16
        assert ops.out_dims(name, DIMS, {"with": 0}) == DIMS
    assert torch.equal(transforms.eq(X, 0.5) + transforms.ne(X, 0.5), torch.ones_like(X))
    ids = torch.tensor([[3, -1, 7, 2]])                    # -1 is a hole, as `argmax` gives one
    agree = transforms.eq(ids, torch.tensor([[3, -1, 2, -1]]))      # ids as they are, float32, a hole on either side
    assert same(agree, torch.tensor([[1.0, NAN, 0.0, NAN]]))
    assert same(transforms.lt(ids, 2.5), torch.tensor([[0.0, NAN, 0.0, 1.0]]))
    assert same(transforms.ne(ids, torch.tensor([[NAN, 3.0, 3.0, 3.0]])), torch.tensor([[NAN, NAN, 1.0, 1.0]]))
    expected = torch.zeros_like(X)
    expected[..., 1] = NAN
    assert same(transforms.eq(X, torch.tensor([3, -1, 0, 1, 2])), expected)     # ids as the operand: its -1 a hole too
    d = ["rows", "pos"]                                    # a reduction and arithmetic leave the hole out too
    assert reductions.sum(ids, d).item() == 12 and reductions.min(ids, d).item() == 2 and reductions.mean(ids, d) == 4
    assert same(transforms.square(ids), torch.tensor([[9.0, NAN, 49.0, 4.0]]))
    assert same(transforms.mul(torch.ones(1, 4), ids), torch.tensor([[3.0, NAN, 7.0, 2.0]]))     # a named operand's
    assert reductions.dot(torch.ones(1, 4), ids, d, over=["pos"]).item() == 12 and transforms.mul(ids, -1)[0, 0] == -3
    half = torch.tensor([1.0039]).bfloat16()               # 1.0039 in bfloat16 is 1.0, which float32 compares exactly
    assert transforms.eq(half, 1.0).item() == 1 and transforms.gt(half, 0.999).item() == 1


def test_max_mirrors_min_each_keeping_an_infinity_and_std_is_the_sample_standard_deviation_of_what_is_there():
    x = torch.tensor([[1.0, 2.0, NAN], [NAN, NAN, NAN], [3.0, -1.0, 0.5], [7.0, NAN, NAN]])
    d = ["rows", "pos"]
    assert same(reductions.max(x, d, over=["pos"]), torch.tensor([2.0, NAN, 3.0, 7.0]))
    assert same(reductions.max(-x, d, over=["pos"]), -reductions.min(x, d, over=["pos"]))
    assert torch.equal(reductions.max(X, DIMS), X.amax((0, 1)))
    inf = torch.tensor([[1.0, float("inf")]])
    assert reductions.max(inf, d, over=["pos"]).item() == float("inf")
    infinite = torch.tensor([[-math.inf, 1.0, NAN], [math.inf, NAN, NAN], [math.inf, math.inf, NAN]])
    assert torch.equal(reductions.min(infinite, d, over=["pos"]), torch.tensor([-math.inf, math.inf, math.inf]))
    assert reductions.min(transforms.log(torch.tensor([[0.0, 2.0]])), d, over=["pos"]).item() == -math.inf
    std = reductions.std(x, d, over=["pos"])
    torch.testing.assert_close(std[[0, 2]], torch.stack([x[0, :2].std(), x[2].std()]))
    assert std[1].isnan() and std[3].isnan()              # none there, and one: no spread to estimate
    torch.testing.assert_close(reductions.std(X, DIMS), X.std((0, 1)))
    assert ops.out_dims("std", DIMS, {}) == ["hidden_size"] and ops.out_dims("max", DIMS, {"over": ["pos"]}) == \
        ["rows", "hidden_size"]


def test_median_is_numpys_over_the_values_there_and_the_mean_of_the_middle_two_at_an_even_count():
    """Over one axis and several, at odd and even counts of the values there, on random values with NaNs and signed
    zeros: the bits of numpy's `nanmedian` of the float32 values, a zero's sign included (+0.0, where the sort alone
    would keep a -0.0), a `layers` axis kept by default; NaN where none is, and on no values at all; ids with their
    hole -1 taken as missing; float32 from bfloat16, its two middle ones averaged in float32."""
    def numpy_median(x: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # an all-NaN slice: NaN, as the op gives
            return torch.from_numpy(numpy.asarray(numpy.nanmedian(x.numpy(), axis=axes), dtype=numpy.float32))

    def bits(x: torch.Tensor) -> torch.Tensor:                     # NaN as one pattern, every other value's bits
        return x.nan_to_num(nan=7.0, posinf=8.0, neginf=9.0).view(torch.int32)
    generator = torch.Generator().manual_seed(1)
    layered = ["layers", "rows", "pos", "hidden_size"]
    x = torch.randn(3, 6, 4, 5, generator=generator)
    x[torch.rand(x.shape, generator=generator) < 0.3] = NAN
    zeros = torch.rand(x.shape, generator=generator) < 0.15
    x[zeros] = torch.where(torch.rand(x.shape, generator=generator) < 0.5, -0.0, 0.0)[zeros]
    x[1, 2] = NAN                                                      # row 2 missing at layer 1
    x[2, 0, 1] = torch.tensor([-0.0, -0.0, NAN, 0.0, NAN])            # three there, the middle sorted -0.0
    for over, axes in ((None, (1, 2)), (["pos"], (2,)), (["hidden_size"], (3,)),
                       (["rows", "pos", "hidden_size"], (1, 2, 3)), (layered, (0, 1, 2, 3))):
        got = reductions.median(x, layered, over)
        assert torch.equal(bits(got), bits(numpy_median(x, axes))), over
        assert list(got.shape) == [x.shape[i] for i in range(4) if i not in axes], over
    assert (reductions.median(x, layered, ["hidden_size"]) == 0).sum() > 0     # zero medians were compared
    held = (~x.isnan()).sum(2)
    assert {int(n) % 2 for n in held.flatten()} == {0, 1}              # odd and even counts over pos alike
    assert reductions.median(x, layered, ["pos"])[1, 2].isnan().all()     # row 2 has none there at layer 1
    d = ["rows", "pos"]
    assert same(reductions.median(torch.tensor([[3.0, 1.0, 2.0], [4.0, 1.0, NAN], [NAN, NAN, NAN]]), d, ["pos"]),
                torch.tensor([2.0, 2.5, NAN]))                         # torch.median would give 1.0 on row 1
    assert reductions.median(torch.zeros(0, 3), d).isnan().item() and \
        reductions.median(torch.zeros(2, 0), d, ["pos"]).isnan().all()
    assert reductions.median(torch.tensor([[3, -1, 7, 2]]), d).item() == 3.0          # 2, 3, 7: the hole left out
    assert reductions.median(X.bfloat16(), DIMS).dtype == torch.float32
    pair = torch.tensor([[1.0, 1.0078125]]).bfloat16()               # two adjacent bfloat16 values
    assert reductions.median(pair, d).item() == 1.00390625             # their mean in float32; in bfloat16, 1.0
    assert ops.out_dims("median", layered, {}) == ["layers", "hidden_size"]
    assert ops.out_dims("median", DIMS, {"over": ["pos"]}) == ["rows", "hidden_size"]


def test_norm_and_dot_reduce_over_hidden_size_unless_over_names_other_axes():
    other = torch.randn(3, 2, 5)
    assert torch.equal(reductions.norm(X, DIMS), reductions.sum(X * X, DIMS, over=["hidden_size"]).sqrt())
    torch.testing.assert_close(reductions.norm(X, DIMS), X.norm(dim=-1))
    torch.testing.assert_close(reductions.norm(X, DIMS, p=1), X.abs().sum(-1))
    torch.testing.assert_close(reductions.norm(X, DIMS, over=["pos"], p=3), X.norm(p=3, dim=1))
    product = transforms.mul(X, other)
    assert torch.equal(reductions.dot(X, other, DIMS), reductions.sum(product, DIMS, over=["hidden_size"]))
    torch.testing.assert_close(reductions.dot(X, other, DIMS), (X * other).sum(-1))
    pattern = torch.rand(2, 6)
    pattern[1, 4:] = NAN                                     # a row's keys past its own: the entries there are taken
    torch.testing.assert_close(reductions.norm(pattern, ["rows", "key"], over=["key"])[1], pattern[1, :4].norm())
    assert reductions.norm(torch.full((1, 5), NAN), ["rows", "hidden_size"]).isnan().all()
    assert ops.out_dims("norm", DIMS, {}) == ["rows", "pos"] and ops.out_dims("dot", DIMS, {"over": ["pos"]}) == \
        ["rows", "hidden_size"]
    with pytest.raises(ValueError, match=r"it reduces over hidden_size unless 'over' names the axes, and the value is "
                                         r"laid out \['rows', 'pos', 'vocab_size'\]"):
        ops.out_dims("norm", ["rows", "pos", "vocab_size"], {})
    with pytest.raises(ValueError, match=r"'norm' p 0 is not greater than 0"):
        ops.check("norm", {"p": 0})
    for p in (math.inf, -math.inf, NAN):                     # the largest absolute value is a `max` of `abs`
        with pytest.raises(ValueError, match=rf"'norm' p {p} is not finite; an Lp norm takes a finite p > 0"):
            ops.check("norm", {"p": p})
    ops.check("norm", {"p": 1})


def test_count_counts_the_positions_that_hold_the_value_a_vector_once():
    """Over `pos` a row's own positions, the features removed with them; a row with none counts 0; over `rows`, how
    many rows hold each position; a pattern's ragged keys over `key`; in ids, -1 is no position."""
    x = X.clone()
    x[1, 1] = NAN
    x[2] = NAN
    assert torch.equal(reductions.count(x, DIMS), torch.tensor([2.0, 1.0, 0.0]))
    assert torch.equal(reductions.count(x, DIMS, over=["rows"]), torch.tensor([2.0, 1.0]))
    assert torch.equal(reductions.count(x, DIMS, over=["rows", "pos"]), torch.tensor(3.0))
    assert torch.equal(reductions.count(x, DIMS, over=["hidden_size"]), torch.tensor([[5.0, 5.0], [5.0, 0.0],
                                                                                       [0.0, 0.0]]))
    pattern = torch.rand(2, 3, 2, 4)                         # rows, pos, num_heads, key
    pattern[1, :, :, 2:] = NAN
    layout = ["rows", "pos", "num_heads", "key"]
    assert torch.equal(reductions.count(pattern, layout, over=["key"]), torch.tensor([[4.0] * 3, [2.0] * 3]))
    assert torch.equal(reductions.count(torch.tensor([[3, -1, 0]]), ["rows", "pos"]), torch.tensor([2.0]))
    assert ops.out_dims("count", ["layers", *DIMS], {}) == ["layers", "rows"]
    assert ops.out_dims("count", layout, {"over": ["key"]}) == ["rows", "pos"]
    with pytest.raises(ValueError, match=r"count counts along 'pos' unless 'over' names the axes it counts along, and "
                                         r"the value is laid out \['rows', 'hidden_size'\]"):
        ops.out_dims("count", ["rows", "hidden_size"], {})


def test_entropy_is_of_the_softmax_over_the_vocabulary_or_the_experts_unless_over_names_other_axes():
    """`-sum(p log p)` of the float32 softmax over `vocab_size` where the value has it, else `num_experts`, else the
    axes `over` names: ln n where the softmax is uniform, a missing entry with no share and a vector with none
    missing, an entry of -inf (a probability of 0) adding 0; a value with neither axis and no `over` is refused."""
    p = LOGITS.softmax(-1)
    torch.testing.assert_close(reductions.entropy(LOGITS, ["rows", "pos", "vocab_size"]), -(p * p.log()).sum(-1))
    routed = ["rows", "pos", "num_experts"]
    assert reductions.entropy(torch.zeros(2, 3, 4), routed).sub(math.log(4)).abs().max() < 1e-6
    x = X.clone()
    x[0, 0] = NAN
    x[1, 1, 2:] = NAN
    x[2, 0, 1] = -math.inf
    got = reductions.entropy(x, routed)
    q = X[1, 1, :2].softmax(-1)
    assert got[0, 0].isnan() and not got[1:].isnan().any()
    torch.testing.assert_close(got[1, 1], -(q * q.log()).sum())
    torch.testing.assert_close(got[2, 0], reductions.entropy(X[2, 0, [0, 2, 3, 4]][None], ["pos", "num_experts"])[0])
    pattern = torch.randn(2, 3, 4, 5)
    flat = pattern.flatten(2).softmax(-1)
    torch.testing.assert_close(reductions.entropy(pattern, ["rows", "pos", "num_heads", "key"],
                                                  over=["num_heads", "key"]), -(flat * flat.log()).sum(-1))
    assert reductions.entropy(LOGITS.bfloat16(), ["rows", "pos", "vocab_size"]).dtype == torch.float32
    assert ops.out_dims("entropy", ["layers", *routed], {}) == ["layers", "rows", "pos"]
    assert ops.out_dims("entropy", ["rows", "pos", "vocab_size", "num_experts"], {}) == ["rows", "pos", "num_experts"]
    assert ops.out_dims("entropy", ["rows", "pos", "num_heads", "key"], {"over": ["key"]}) == \
        ["rows", "pos", "num_heads"]
    with pytest.raises(ValueError, match=r"it takes the softmax over vocab_size or num_experts unless 'over' names the "
                                         r"axes, and the value is laid out \['rows', 'pos', 'hidden_size'\]"):
        ops.out_dims("entropy", DIMS, {})


def arithmetic(d: dict) -> None:
    """patching.json as the identities' document: three reads at layer 0 (the last token, every token and the whole
    entity, the last two ragged) and the counterfactual's last token, each identity a pair of values, and the ops in
    a read and in a write as well as as steps."""
    target, cf = "target", "counterfactual.v_cf"
    d["sites"]["head"] = {"component": "lm_head_output"}
    op = lambda name, of, **args: {"kind": "op", "op": name, "of": of, **args}       # noqa: E731
    d["steps"] = {
        "counterfactual": d["steps"]["counterfactual"],
        "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
            "h": {"site": target, "pos": -1}, "last3": {"site": target, "pos": {"index": "-3:"}},
            "every": {"site": target, "pos": {"index": ":"}},
            "entity": {"site": target, "pos": {"scope": {"column": "pairs.entity"}}},
            "positive": {"site": target, "pos": -1, "ops": [{"op": "gt", "with": 0}]},
            "n_within": {"site": target, "pos": -1, "ops": [{"op": "norm"}]},
            "dot_within": {"site": target, "pos": -1, "ops": [{"op": "dot", "with": cf}]},
            "count_within": {"site": target, "pos": {"scope": {"column": "pairs.entity"}}, "ops": [{"op": "count"}]},
            "abs_within": {"site": target, "pos": -1, "ops": [{"op": "abs"}]},
            "logits": {"site": "head", "pos": -1}}},
        "square": op("square", "clean.h"), "summed": op("sum", "square", over=["hidden_size"]),
        "root": op("sqrt", "summed"), "norm": op("norm", "clean.h"),
        "product": op("mul", "clean.h", **{"with": cf}), "product_sum": op("sum", "product", over=["hidden_size"]),
        "dot": op("dot", "clean.h", **{"with": cf}),
        "sign": op("sign", "clean.h"), "eq": op("eq", "sign", **{"with": 1}), "ne": op("ne", "sign", **{"with": 1}),
        "eq_ne": op("add", "eq", **{"with": "ne"}),
        "gt": op("gt", "clean.h", **{"with": cf}), "le": op("le", "clean.h", **{"with": cf}),
        "gt_le": op("add", "gt", **{"with": "le"}),
        "count3": op("count", "clean.last3"), "count_every": op("count", "clean.every"),
        "count_entity": op("count", "clean.entity"), "abs": op("abs", "clean.h"),
        "div": op("div", "clean.h", **{"with": cf}), "back": op("scale", "div", factor=cf),
        "masked_value": op("scale", "clean.h", factor="clean.positive"),
        "masked": {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": {"site": target, "pos": -1, "ops": [{"op": "scale", "factor": "clean.positive"}]},
            "logits": {"site": "head", "pos": -1}}},
        "swapped": {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": {"site": target, "pos": -1, "ops": [{"op": "swap", "with": "masked_value"}]},
            "logits": {"site": "head", "pos": -1}}},
        "live": {"kind": "forward", "input": "pairs.input", "interventions": {
            "patch": {"site": target, "pos": -1, "ops": [{"op": "sign"}, {"op": "scale", "factor": 0.1}]},
            "after": {"site": target, "pos": -1}}},
        **{f"{name}_read": {"kind": "forward", "input": "pairs.input", "interventions": {
            "p": {"site": target, "pos": -1, "ops": [product, {"op": "sum", "over": ["hidden_size"]}]},
            "logits": {"site": "head", "pos": -1}}}
           for name, product in (("mul", {"op": "mul", "with": cf}), ("scale", {"op": "scale", "factor": cf}))}}
    d["saves"] = [f"clean.{k}" for k in d["steps"]["clean"]["interventions"]] + [
        k for k, step in d["steps"].items() if step["kind"] == "op"] + [
        "masked.patch", "masked.logits", "swapped.logits", "live.patch", "live.after", "mul_read.p", "mul_read.logits",
        "scale_read.p", "scale_read.logits"]


@pytest.fixture(scope="module")
def identities(engines) -> dict:
    """The identities' document run on each engine, ndif under `remote="local"`: each run's results."""
    plan = edited("patching", arithmetic).compile()
    runs = {kind: engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE).results for kind in ("nnterp", "hooks")}
    there = engines("ndif", plan.model, remote="local").execute(plan, batch_size=BATCH_SIZE)
    return {**runs, "ndif": there.results, "positions": placed(there)}


def test_the_arithmetic_identities_hold_on_the_corpus_and_the_engines_agree_to_the_bit(identities):
    """On weekdays/train at layer 0: `norm` is `sqrt` of the `sum` of the `square` and `dot` the `sum` of `mul`, to
    the bit; `eq` and `ne`, and `gt` and `le`, sum to exactly 1; `count` of a static-width read is its width and of a
    ragged one each row's own number of positions; `div` then `scale` by the same value is the value to two float32
    roundings (at worst 0.91 of float32's epsilon, relative, and 86% of entries exact); each op inside a read is its step's value to the bit; and nnterp, hooks and ndif under
    `remote="local"` give the same bits."""
    v = identities["nnterp"]
    assert torch.equal(v["norm"], v["root"]) and v["norm"].shape == (4, 1)
    assert torch.equal(v["dot"], v["product_sum"])
    assert torch.equal(v["eq_ne"], torch.ones(4, 1, 16)) and 0 < v["eq"].sum() < 64
    assert torch.equal(v["gt_le"], torch.ones(4, 1, 16)) and 0 < v["gt"].sum() < 64
    assert torch.equal(v["count3"], torch.full((4,), 3.0))
    every, entity = (identities["positions"][f"clean.{k}"] for k in ("every", "entity"))
    assert v["count_every"].tolist() == [len(p) for p in every] == [11, 9, 9, 9]
    assert v["count_entity"].tolist() == [len(p) for p in entity] == [3, 1, 1, 1]
    h = v["clean.h"]
    assert ((v["back"] - h).abs() <= 2 * torch.finfo(torch.float32).eps * h.abs()).all()
    assert torch.equal(v["clean.n_within"], v["norm"]) and torch.equal(v["clean.dot_within"], v["dot"])
    assert torch.equal(v["clean.count_within"], v["count_entity"]) and torch.equal(v["clean.abs_within"], v["abs"])
    assert v["clean.positive"].dtype == torch.float32 and set(v["clean.positive"].unique().tolist()) == {0.0, 1.0}
    assert same(identities["hooks"], v) and same(identities["ndif"], v)


def test_a_comparison_masks_a_write_and_an_elementwise_transform_acts_on_the_live_value(identities):
    """`scale` by a comparison read in another forward writes the value where it holds and zero where not, the same
    logits as swapping in the masked value made as a step; `[sign, scale 0.1]` writes a tenth of each entry's sign,
    which a read after it sees; `abs` alone in a read writes nothing (the clean logits are the unpatched ones)."""
    v = identities["nnterp"]
    h, positive = v["clean.h"], v["clean.positive"]
    assert torch.equal(v["masked.patch"], h * positive) and torch.equal(v["masked.logits"], v["swapped.logits"])
    assert not torch.equal(v["masked.logits"], v["clean.logits"])
    assert torch.equal(v["live.patch"], 0.1 * h.sign()) and torch.equal(v["live.after"], v["live.patch"])


def test_a_product_in_a_read_is_mul_which_writes_nothing_where_scale_writes_it_back(identities):
    """`[mul v, sum over hidden_size]` in a read binds the `dot` with v and leaves the forward's logits the unpatched
    ones to the bit; `[scale v, sum]` binds the same number, but `scale` is an edit, so it writes the product back and
    the logits move."""
    v = identities["nnterp"]
    assert torch.equal(v["mul_read.p"], v["dot"]) and torch.equal(v["scale_read.p"], v["dot"])
    assert torch.equal(v["mul_read.logits"], v["clean.logits"])
    assert not torch.equal(v["scale_read.logits"], v["clean.logits"])


def agreement(d: dict) -> None:
    """patching.json's logits read at every token, whose rows hold 11, 9, 9 and 9, so a ragged read with six holes:
    the agreement of two `argmax` steps of it (the second of its `square`, since a ragged read pairs with no other
    read's), its `mean` and its `count`."""
    d["steps"] = {
        "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
            "logits": {"site": {"component": "lm_head_output"}, "pos": {"index": ":"}}}},
        "top": {"kind": "op", "op": "argmax", "of": "clean.logits"},
        "square": {"kind": "op", "op": "square", "of": "clean.logits"},
        "other_top": {"kind": "op", "op": "argmax", "of": "square"},
        "agree": {"kind": "op", "op": "eq", "of": "top", "with": "other_top"},
        "rate": {"kind": "op", "op": "mean", "of": "agree"},
        "n": {"kind": "op", "op": "count", "of": "agree"}}
    d["saves"] = ["top", "other_top", "agree", "rate", "n"]


def test_a_comparison_of_ids_is_missing_at_a_hole_so_an_agreement_is_over_the_positions_there(engines):
    """Both `argmax` steps give the six positions past the shorter rows' own id -1, and `eq` of them is NaN there,
    float32, so the `mean` of the agreement is 28 of the 38 positions there, 0.737, where counting the holes as
    agreeing gave 34 of 44, 0.773; its `count` is each row's own number of positions."""
    plan = edited("patching", agreement).compile()
    r = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    hole = r["top"] == -1
    assert int(hole.sum()) == 6 and torch.equal(hole, r["other_top"] == -1)
    agree = r["agree"]
    assert agree.dtype == torch.float32 and agree[hole].isnan().all() and not agree[~hole].isnan().any()
    assert agree[~hole].sum().item() == 28 and round(r["rate"].item(), 3) == 0.737
    assert r["rate"].item() == pytest.approx(28 / 38, rel=1e-6)
    assert r["n"].tolist() == [11.0, 9.0, 9.0, 9.0]


def test_a_reduction_or_an_elementwise_transform_of_ids_takes_their_hole_as_missing(engines):
    """The ragged read's `argmax` holds -1 at the six positions past the shorter rows' own; `min`, `sum`, `mean`,
    `std` and `norm` over `pos` of it, and its `square`, are each that op of the ids with the hole as NaN, to the bit,
    so a row's least id is one of its own tokens', never -1, where counting the hole gave -1 on the three short rows."""
    def change(d: dict) -> None:
        agreement(d)
        d["steps"] = {k: d["steps"][k] for k in ("clean", "top")}
        for op in ("min", "sum", "mean", "std", "norm"):
            d["steps"][op] = {"kind": "op", "op": op, "of": "top", "over": ["pos"]}
        d["steps"]["square"] = {"kind": "op", "op": "square", "of": "top"}
        d["saves"] = list(d["steps"])[1:]
    plan = edited("patching", change).compile()
    r = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results
    top = r["top"]
    floated = top.float().masked_fill(top == -1, NAN)
    for op in ("min", "sum", "mean", "std", "norm"):
        assert same(r[op], ops.call(op, floated, {"over": ["pos"]}, dims=["rows", "pos"])), op
    assert same(r["square"], floated.square()) and int((top == -1).sum()) == 6
    own = [top[i][top[i] != -1] for i in range(4)]
    assert r["min"].tolist() == [float(ids.min()) for ids in own] and (r["min"] >= 0).all()
    assert r["sum"].tolist() == [float(ids.sum()) for ids in own]


def test_argmax_and_top_k_rank_ids_with_their_hole_below_every_id():
    """Ids (a top_k step's `ids`, an `argmax`'s) rank as the integers they are, exactly, their hole -1 missing as NaN is:
    ranked below every id, a pick past the ids there -1, and a vector of holes -1; `top_k`'s `values` keep the ids'
    dtype, the hole -1 in them."""
    ids = torch.tensor([[[3, -1, 7, 5]], [[-1, -1, -1, -1]], [[2**40 + 1, 2**40, -1, 0]]])
    dims = ["rows", "pos", "k"]
    assert transforms.argmax(ids, dims, over="k").tolist() == [[2], [-1], [0]]
    top = transforms.top_k(ids, 4, dims, over="k")
    assert top["ids"].tolist() == [[[2, 3, 0, -1]], [[-1, -1, -1, -1]], [[0, 1, 3, -1]]]
    assert top["values"].dtype == torch.int64
    assert top["values"].tolist() == [[[7, 5, 3, -1]], [[-1, -1, -1, -1]], [[2**40 + 1, 2**40, 0, -1]]]


def test_the_new_reductions_are_refused_by_name_where_they_cannot_mean_anything():
    """At compile, with the step's path: `norm` of a metric with no `over` (it has no `hidden_size`), `count` of a
    value with no `pos`, a `count` over `rows` inside an entry (it sees one batch), a `p` of 0, and a `p` of
    `Infinity`, which a JSON document can spell and which gave 1.0 for every vector."""
    from clab.plan import PlanError

    def with_steps(**steps):
        def change(d):
            d["steps"].update(steps)
        return change
    cases = {
        r"steps\.n: it reduces over hidden_size unless 'over' names the axes, and the value is laid out \['rows', 'pos'\]":
            {"n": {"kind": "op", "op": "norm", "of": "logit_diff"}},
        r"steps\.c: count counts along 'pos' unless 'over' names the axes it counts along, and the value is laid out "
        r"\['hidden_size'\]": {"m": {"kind": "op", "op": "mean", "of": "counterfactual.v_cf"},
                               "c": {"kind": "op", "op": "count", "of": "m"}},
        r"ops\[0\]: 'count' aggregates over rows, and inside an entry it sees one batch":
            {"r": {"kind": "forward", "input": "pairs.input", "interventions": {
                "x": {"site": "target", "pos": -1, "ops": [{"op": "count", "over": ["rows"]}]}}}},
        r"steps\.p: 'norm' p 0 is not greater than 0": {"p": {"kind": "op", "op": "norm", "of": "logit_diff",
                                                                "over": ["pos"], "p": 0}},
        r"steps\.p: 'norm' p inf is not finite": {"p": {"kind": "op", "op": "norm", "of": "logit_diff",
                                                          "over": ["pos"], "p": math.inf}},
    }
    for message, steps in cases.items():
        with pytest.raises(PlanError, match=message):
            edited("patching", with_steps(**steps)).compile()


def test_permute_with_seed_null_is_the_identity_and_with_a_seed_the_permutation_it_draws():
    """Row i holds row π(i)'s, π the identity where `seed` is null and otherwise `permutation`'s draw from the seed
    alone, which a later draw repeats; applying the inverse gives `of` back, and the layout is kept, a leading
    `layers` axis included."""
    assert torch.equal(transforms.permute(X, None, DIMS), X)
    pi = transforms.permutation(7, 3)
    got = transforms.permute(X, 7, DIMS)
    assert torch.equal(got, X[pi]) and torch.equal(pi, transforms.permutation(7, 3))
    assert torch.equal(got[torch.argsort(pi)], X)
    assert torch.equal(transforms.permute(X, 7, DIMS, rows=torch.tensor([4, 5, 6])), got)
    stacked = torch.stack([X, 2 * X])
    assert torch.equal(transforms.permute(stacked, 7, ["layers", *DIMS]), stacked[:, pi])
    assert ops.out_dims("permute", DIMS, {"seed": 7}) == DIMS
    assert sorted(transforms.permutation(1, 50).tolist()) == list(range(50))


def test_a_derangement_leaves_no_row_where_it_was_and_over_one_row_is_refused():
    """Drawn again from the seed's generator until no row is fixed, so it is the same draw every time; seed null,
    the identity, moves no row and is refused beside it, and so is a value of one row."""
    for seed in range(20):
        pi = transforms.permutation(seed, 4, derangement=True)
        assert not (pi == torch.arange(4)).any() and torch.equal(pi, transforms.permutation(seed, 4, True))
    assert torch.equal(transforms.permutation(0, 2, True), torch.tensor([1, 0]))
    assert transforms.permutation(0, 4, True).tolist() == [2, 3, 0, 1]          # the corpus's draw, pinned
    with pytest.raises(ValueError, match="^a derangement moves every row to another, and over 1 row there is none"):
        transforms.permute(X[:1], 0, DIMS, derangement=True)
    with pytest.raises(ValueError, match="'permute' with seed null is the identity, which moves no row"):
        ops.check("permute", {"seed": None, "derangement": True})
    with pytest.raises(ValueError, match="'permute' needs 'seed'"):
        ops.check("permute", {"derangement": True})
    with pytest.raises(ValueError, match="seed: 'a' is not int or null"):
        ops.check("permute", {"seed": "a"})
