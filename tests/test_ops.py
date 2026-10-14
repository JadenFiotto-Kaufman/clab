"""The implemented ops on plain tensors, and the site machinery that runs them inside a write."""

import math

import pytest
import torch

from clab import ops

torch.manual_seed(0)
X = torch.randn(3, 2, 5)                       # rows, pos, hidden
LOGITS = torch.randn(3, 1, 7)                  # rows, pos, vocab
IDS = torch.tensor([1, 4, 6])
DIMS = ["rows", "pos", "hidden_size"]


def test_every_op_registered_itself_under_its_own_name():
    assert set(ops.OPS) == {"swap", "add", "sub", "scale", "renormalize", "normalize", "project_out", "readout",
                            "encode", "mean", "pca", "match", "logit_diff", "token_prob", "token_logit",
                            "cross_entropy", "kl", "js", "top_k", "argmax", "decode"}
    assert all(fn.__name__ == name for name, fn in ops.OPS.items())
    assert set(ops.AXES) == set(ops.OPS)


def test_ops_group_by_what_they_do_to_the_rows():
    groups = {name: ops.group(name) for name in ops.OPS}
    assert {n for n, g in groups.items() if g == "edits"} == {"swap", "add", "sub", "scale", "renormalize",
                                                               "normalize", "project_out"}
    assert {n for n, g in groups.items() if g == "reductions"} == {"mean", "pca"}
    assert groups["match"] == groups["top_k"] == groups["readout"] == "metrics"


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
    assert torch.equal(ops.swap(X, v), v.expand(3, 2, 5))
    assert torch.equal(ops.swap(X, 0.0), torch.zeros_like(X))
    assert torch.equal(ops.swap(X, X.flip(0)), X.flip(0))


def test_add_sub_scale():
    v = torch.randn(5)
    assert torch.equal(ops.add(X, v, scale=4.0), X + 4.0 * v)
    assert torch.equal(ops.sub(X, v), X - v)
    assert torch.equal(ops.scale(X, 0.5), X * 0.5)


def test_renormalize_restores_the_original_norm():
    out = ops.renormalize(X * 3 + 1, original=X)
    torch.testing.assert_close(out.norm(dim=-1), X.norm(dim=-1))


def test_normalize_and_project_out():
    torch.testing.assert_close(ops.normalize(X).norm(dim=-1), torch.ones(3, 2))
    d = torch.randn(5)
    out = ops.project_out(X, d)
    torch.testing.assert_close(out @ d, torch.zeros(3, 2), atol=1e-6, rtol=0)


def test_mean_over_the_example_axes_by_default():
    assert torch.equal(ops.mean(X, dims=["rows", "pos", "hidden_size"]), X.mean(dim=[0, 1]))
    assert torch.equal(ops.mean(X, dims=["rows", "pos", "hidden_size"], over=["pos"]), X.mean(dim=1))


def test_metrics_on_logits():
    assert torch.equal(ops.match(LOGITS, LOGITS.argmax(-1)[:, 0]), torch.ones(3, 1))
    assert torch.equal(ops.match(LOGITS, (LOGITS.argmax(-1)[:, 0] + 1) % 7), torch.zeros(3, 1))
    picked = LOGITS[torch.arange(3), 0, IDS][:, None]
    assert torch.equal(ops.token_logit(LOGITS, IDS), picked)
    assert torch.equal(ops.logit_diff(LOGITS, IDS, IDS), torch.zeros(3, 1))
    torch.testing.assert_close(ops.token_prob(LOGITS, IDS), LOGITS.softmax(-1)[torch.arange(3), 0, IDS][:, None])
    torch.testing.assert_close(ops.cross_entropy(LOGITS, IDS),
                               torch.nn.functional.cross_entropy(LOGITS[:, 0], IDS, reduction="none")[:, None])


def test_metrics_broadcast_over_a_leading_layer_axis():
    stacked = torch.stack([LOGITS, LOGITS * 2])          # layers, rows, pos, vocab
    out = ops.token_logit(stacked, IDS)
    assert out.shape == (2, 3, 1)
    assert torch.equal(out[1], 2 * ops.token_logit(LOGITS, IDS))


def test_kl_and_js():
    other = torch.randn(3, 1, 7)
    p, q = LOGITS.log_softmax(-1), other.log_softmax(-1)
    torch.testing.assert_close(ops.kl(LOGITS, other),
                               torch.nn.functional.kl_div(q, p, log_target=True, reduction="none").sum(-1))
    assert torch.equal(ops.kl(LOGITS, LOGITS), torch.zeros(3, 1))
    js = ops.js(LOGITS, other)
    torch.testing.assert_close(js, ops.js(other, LOGITS))
    assert bool((js >= 0).all() and (js <= math.log(2)).all())


def test_out_dims_follows_each_ops_declaration():
    assert ops.out_dims("match", ["rows", "pos", "vocab_size"], {}) == ["rows", "pos"]
    assert ops.out_dims("mean", ["entries", "hidden_size"], {"over": None}) == ["hidden_size"]
    assert ops.out_dims("readout", ["layers", "rows", "pos", "hidden_size"], {}) == ["layers", "rows", "pos", "vocab_size"]
    with pytest.raises(ValueError, match="needs a value with a 'vocab_size' axis"):
        ops.out_dims("match", ["rows", "pos", "hidden_size"], {})


def record(start: int, width: int, value, ops_=None) -> dict:
    positions = torch.arange(start, start + width).expand(3, width)
    return {"positions": positions, "select": {},
            "ops": ops_ or [{"op": "swap", "args": {"with": value}, "featurizer": None, "dims": DIMS}]}


def test_a_window_write_and_three_single_writes_are_one_tensor():
    value = torch.randn(3, 6, 5)                         # batch, seq, hidden
    other = torch.randn(3, 6, 5)
    window = ops.write(value, "layer_output", [record(2, 3, other[:, 2:5])])
    singles = ops.write(value, "layer_output", [record(i, 1, other[:, i:i + 1]) for i in (2, 3, 4)])
    assert torch.equal(window, singles)
    assert torch.equal(window[:, 2:5], other[:, 2:5]) and torch.equal(window[:, :2], value[:, :2])


def test_writes_compose_in_order_and_renormalize_sees_the_value_before_them():
    value = torch.randn(3, 4, 5)
    steer = record(3, 1, None, [{"op": "add", "args": {"with": torch.ones(5), "scale": 4.0}, "featurizer": None, "dims": DIMS}])
    restore = record(3, 1, None, [{"op": "renormalize", "args": {}, "featurizer": None, "dims": DIMS}])
    out = ops.write(value, "layer_output", [steer, restore])
    torch.testing.assert_close(out[:, 3].norm(dim=-1), value[:, 3].norm(dim=-1))
    assert not torch.equal(out[:, 3], value[:, 3])
    assert torch.equal(out[:, :3], value[:, :3])


def test_select_narrows_one_axis_and_write_puts_it_back():
    value = torch.randn(2, 3, 4, 2)                      # batch, seq, num_heads, head_dim (attention_premix)
    head = {"positions": torch.tensor([[2], [2]]), "select": {2: torch.tensor([1])}}
    assert torch.equal(ops.select(value, "attention_premix", head), value[:, 2:3, 1:2])
    out = ops.write(value, "attention_premix", [{**head, "ops": [
        {"op": "swap", "args": {"with": 0.0}, "featurizer": None, "dims": ["rows", "pos", "num_heads", "head_dim"]}]}])
    assert torch.equal(out[:, 2, 1], torch.zeros(2, 2))
    out[:, 2, 1] = value[:, 2, 1]
    assert torch.equal(out, value)


def test_a_pattern_is_selected_along_its_query_axis():
    value = torch.randn(2, 4, 5, 5)                      # batch, num_heads, query, key
    rec = {"positions": torch.tensor([[4], [4]]), "select": {}}
    assert torch.equal(ops.select(value, "attention_probabilities", rec), value[:, :, 4:5].movedim(2, 1))


def test_pca_gives_an_orthonormal_basis_of_the_largest_variance_first():
    torch.manual_seed(1)
    x = torch.randn(40, 5) * torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1]) + 2.0
    got = ops.pca(x, k=3, dims=["entries", "hidden_size"])
    basis, mean = got["basis"], got["mean"]
    assert basis.shape == (5, 3) and mean.shape == (5,)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(mean, x.mean(0))
    variance = ((x - mean) @ basis).var(0)
    assert bool((variance[:-1] >= variance[1:]).all())
    assert torch.equal(ops.pca(x, k=3, dims=["entries", "hidden_size"])["basis"], basis)     # deterministic signs
    with pytest.raises(ValueError, match="pca of 3 vectors has no rank-3 basis"):
        ops.pca(x[:3], k=3, dims=["entries", "hidden_size"])


def test_top_k_and_argmax():
    got = ops.top_k(LOGITS, k=2)
    assert got["ids"].shape == got["probs"].shape == (3, 1, 2)
    assert torch.equal(got["ids"][..., 0], ops.argmax(LOGITS))
    torch.testing.assert_close(got["probs"][..., 0], LOGITS.softmax(-1).max(-1).values)


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class Engine:
    tokenizer = Tokenizer()


def test_decode_stops_at_each_rows_first_eos():
    ids = torch.tensor([[5, 6, 0, 7], [8, 9, 9, 9]])
    assert ops.decode(ids, engine=Engine()) == ["<5><6>", "<8><9><9><9>"]
    assert ops.decode(ids, engine=Engine(), tokens=True) == [["t5", "t6"], ["t8", "t9", "t9", "t9"]]
