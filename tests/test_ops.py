"""The implemented ops on plain tensors, and the site machinery that runs an entry's records at a site."""

import math

import pytest
import torch

from clab import batches, ops
from clab.ops import edits, metrics, reductions, transforms

torch.manual_seed(0)
X = torch.randn(3, 2, 5)                       # rows, pos, hidden
LOGITS = torch.randn(3, 1, 7)                  # rows, pos, vocab
IDS = torch.tensor([1, 4, 6])
ONE = IDS[:, None]                              # a token argument: per row, the ids it may be
DIMS = ["rows", "pos", "hidden_size"]


def test_every_op_is_listed_once_in_order_under_its_own_name():
    assert list(ops.OPS) == ["swap", "add", "sub", "scale", "renormalize", "normalize", "project_out", "decode",
                             "apply", "noise", "clamp", "readout", "encode", "mean", "sum", "min", "pca", "match",
                             "logit_diff", "token_prob", "token_logit", "cross_entropy", "contains", "kl", "js", "top_k",
                             "argmax", "detokenize", "cosine"]
    assert all(fn.__name__ == name for name, fn in ops.OPS.items())
    assert set(ops.DECLARED) == set(ops.OPS)


def test_each_op_declares_its_group_and_bundle():
    groups = {name: ops.DECLARED[name]["group"] for name in ops.OPS}
    assert {n for n, g in groups.items() if g == "edits"} == {"swap", "add", "sub", "scale", "renormalize",
                                                               "normalize", "project_out", "decode", "apply", "noise",
                                                               "clamp"}
    assert {n for n, g in groups.items() if g == "reductions"} == {"mean", "sum", "min", "pca"}
    assert {n for n, g in groups.items() if g == "transforms"} == {"readout", "encode", "top_k", "argmax",
                                                                    "detokenize"}
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


def test_normalize_and_project_out():
    torch.testing.assert_close(edits.normalize(X).norm(dim=-1), torch.ones(3, 2))
    d = torch.randn(5)
    out = edits.project_out(X, d)
    torch.testing.assert_close(out @ d, torch.zeros(3, 2), atol=1e-6, rtol=0)


def test_mean_over_the_example_axes_by_default():
    assert torch.equal(reductions.mean(X, dims=["rows", "pos", "hidden_size"]), X.mean(dim=[0, 1]))
    assert torch.equal(reductions.mean(X, dims=["rows", "pos", "hidden_size"], over=["pos"]), X.mean(dim=1))


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


def test_out_dims_follows_each_ops_declaration():
    assert ops.out_dims("match", ["rows", "pos", "vocab_size"], {}) == ["rows", "pos"]
    assert ops.out_dims("mean", ["rows", "pos", "hidden_size"], {"over": None}) == ["hidden_size"]
    assert ops.out_dims("readout", ["layers", "rows", "pos", "hidden_size"], {}) == ["layers", "rows", "pos", "vocab_size"]
    with pytest.raises(ValueError, match="needs a value with a 'vocab_size' or 'classes' axis"):
        ops.out_dims("match", ["rows", "pos", "hidden_size"], {})


def taken(value: torch.Tensor, component: str, rec: dict) -> torch.Tensor:
    """What a read record at a site takes of `value`."""
    got: dict = {}
    assert not batches.visit(value, component, [{**rec, "name": "r"}], got)[1]
    return got["r"]


def record(start: int, width: int, value, ops_=None) -> dict:
    positions = torch.arange(start, start + width).expand(3, width)
    return {"positions": positions, "select": {},
            "ops": ops_ or [{"op": "swap", "args": {"with": value}, "dims": DIMS}]}


def test_a_ragged_write_writes_only_the_positions_it_keeps():
    value = torch.randn(2, 4, 5)
    ragged = {"positions": torch.tensor([[1, 2], [3, 3]]), "keep": torch.tensor([[True, True], [True, False]]),
              "select": {}, "ops": [{"op": "swap", "args": {"with": 0.0}, "dims": DIMS}]}
    out = batches.visit(value, "layer_output", [ragged], {})[0]
    assert torch.equal(out[0, 1:3], torch.zeros(2, 5)) and torch.equal(out[1, 3], torch.zeros(5))
    unwritten = torch.tensor([[1, 0, 0, 1], [1, 1, 1, 0]], dtype=torch.bool)
    assert torch.equal(out[unwritten], value[unwritten])
    empty = {**ragged, "keep": torch.tensor([[False, False], [False, False]])}
    assert torch.equal(batches.visit(value, "layer_output", [empty], {})[0], value)


def test_a_window_write_and_three_single_writes_are_one_tensor():
    value = torch.randn(3, 6, 5)                         # batch, seq, hidden
    other = torch.randn(3, 6, 5)
    window = batches.visit(value, "layer_output", [record(2, 3, other[:, 2:5])], {})[0]
    singles = batches.visit(value, "layer_output", [record(i, 1, other[:, i:i + 1]) for i in (2, 3, 4)], {})[0]
    assert torch.equal(window, singles)
    assert torch.equal(window[:, 2:5], other[:, 2:5]) and torch.equal(window[:, :2], value[:, :2])


def test_writes_compose_in_order_and_renormalize_sees_the_value_before_them():
    value = torch.randn(3, 4, 5)
    steer = record(3, 1, None, [{"op": "add", "args": {"with": torch.ones(5), "scale": 4.0}, "dims": DIMS}])
    restore = record(3, 1, None, [{"op": "renormalize", "args": {}, "dims": DIMS}])
    out = batches.visit(value, "layer_output", [steer, restore], {})[0]
    torch.testing.assert_close(out[:, 3].norm(dim=-1), value[:, 3].norm(dim=-1))
    assert not torch.equal(out[:, 3], value[:, 3])
    assert torch.equal(out[:, :3], value[:, :3])


def test_select_narrows_one_axis_and_write_puts_it_back():
    value = torch.randn(2, 3, 4, 2)                      # batch, seq, num_heads, head_dim (attention_premix)
    head = {"positions": torch.tensor([[2], [2]]), "select": {2: torch.tensor([1])}}
    assert torch.equal(taken(value, "attention_premix", head), value[:, 2:3, 1:2])
    out = batches.visit(value, "attention_premix", [{**head, "ops": [
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
    got = reductions.pca(x, k=3, dims=["entries", "hidden_size"])
    basis, mean = got["basis"], got["mean"]
    assert basis.shape == (5, 3) and mean.shape == (5,)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(mean, x.mean(0))
    variance = ((x - mean) @ basis).var(0)
    assert bool((variance[:-1] >= variance[1:]).all())
    assert torch.equal(reductions.pca(x, k=3, dims=["entries", "hidden_size"])["basis"], basis)     # deterministic signs
    with pytest.raises(ValueError, match="pca of 3 vectors has no rank-3 basis"):
        reductions.pca(x[:3], k=3, dims=["entries", "hidden_size"])


def test_top_k_and_argmax():
    got = transforms.top_k(LOGITS, k=2)
    assert got["ids"].shape == got["probs"].shape == (3, 1, 2)
    assert torch.equal(got["ids"][..., 0], transforms.argmax(LOGITS))
    torch.testing.assert_close(got["probs"][..., 0], LOGITS.softmax(-1).max(-1).values)


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
    assert transforms.argmax(logits).tolist() == [[LOGITS[0, 0].argmax().item()], [-1], [LOGITS[2, 0].argmax().item()]]
    top = transforms.top_k(logits, 2)
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
