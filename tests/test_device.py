"""On a GPU: every tensor clab builds where the plan runs (positions, masks, per-row indices, token ids,
operands) meets an activation on that activation's device, and the engines run a document on `device`.

The project's CPU-only suite skips this file. Where the venv's torch cannot use the driver, an env whose torch
can, with clab installed in it, runs it with one command from the repo root:

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m pytest tests/test_device.py -q
"""

import functools
import warnings
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, edited, plan_of, same
from test_ops import taken

from clab import featurizers, ops
from clab.engines import load, standard
from clab.format import data
from clab.ops import metrics
from clab.steps import batches

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DIMS = ["rows", "pos", "hidden_size"]
G = torch.Generator().manual_seed(0)
VALUE = torch.randn(3, 6, 8, generator=G)
OTHER = torch.randn(3, 2, 8, generator=G)


def writes() -> list[dict]:
    """Writes built as a batch binds them, on the CPU: a ragged swap with its `keep`, an add of a CPU operand
    on the features a gate's encode keeps, a narrowing to two hidden units, each row's own key indices, a
    renormalize, and a gate over the hidden units interpolating toward a CPU operand."""
    gate = featurizers.make("gate", {}, 8, {"theta": torch.linspace(-1, 1, 8)}, None, None)
    over = featurizers.make("gate", {"over": ["hidden_size"]}, 8, {"theta": torch.linspace(1, -1, 8)}, None, None,
                            kept={"hidden_size": list(range(8))})
    return [
        {"positions": torch.tensor([[1, 2], [3, 3], [4, 5]]), "keep": torch.tensor([[1, 1], [1, 0], [1, 1]]).bool(),
         "select": {}, "ops": [{"op": "swap", "args": {"with": OTHER}, "dims": DIMS}]},
        {"positions": torch.tensor([[0], [0], [0]]), "select": {}, "ops": [
            {"op": "encode", "args": {"featurizer": gate}, "features": torch.tensor([1, 6]), "dims": DIMS, "inner": [
                {"op": "add", "args": {"with": torch.ones(2), "scale": 2.0}, "dims": ["rows", "pos", "features"]}]}]},
        {"positions": torch.tensor([[5], [5], [5]]), "select": {2: torch.tensor([0, 7])},
         "ops": [{"op": "scale", "args": {"factor": 3.0}, "dims": DIMS}]},
        {"positions": torch.tensor([[2], [2], [2]]), "select": {},     # a key index as wide as the axis (`keys`)
         "keys": {"dim": 2, "index": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [3, 4, 5, 6, 7, 0, 1, 2], [5, 6, 7, 0, 1, 2, 3, 4]]),
                  "own": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]).bool()},
         "ops": [{"op": "sub", "args": {"with": 1.0}, "dims": DIMS}]},
        {"positions": torch.tensor([[3], [3], [3]]), "select": {}, "rows": torch.arange(3),
         "ops": [{"op": "renormalize", "args": {}, "dims": DIMS}]},
        {"positions": torch.tensor([[4], [4], [4]]), "select": {},
         "ops": [{"op": "apply", "args": {"featurizer": over, "with": OTHER[:, :1]}, "dims": DIMS, "layers": 0}]},
    ]


def asking(monkeypatch: pytest.MonkeyPatch, asked: list[int]) -> None:
    """`mean` run with torch's sync debug mode set to warn, the host-device syncs each call makes counted into
    `asked`: it asks the device whether any value is missing, one wait, which costs less than computing both the
    plain mean and the NaN-skipping one for the device to pick between."""
    mean = ops.OPS["mean"]

    @functools.wraps(mean)
    def warned(*args: Any, **kwargs: Any) -> Any:
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                return mean(*args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(mode)
                asked.append(sum("synchronizing CUDA operation" in str(w.message) for w in caught))
    monkeypatch.setitem(ops.OPS, "mean", warned)


def test_a_write_built_on_the_cpu_lands_on_the_values_device():
    on_cpu = batches.visit(VALUE, "layer_output", writes(), {})[0]
    on_gpu = batches.visit(VALUE.cuda(), "layer_output", writes(), {})[0]
    assert on_gpu.device.type == "cuda"
    torch.testing.assert_close(on_gpu.cpu(), on_cpu, atol=1e-6, rtol=1e-6)


def test_a_read_built_on_the_cpu_selects_on_the_values_device():
    rec = {"positions": torch.tensor([[1, 2], [3, 4], [0, 5]]), "select": {},
           "keys": {"dim": 2, "index": torch.tensor([[1, 2], [3, 4], [5, 6]]), "own": torch.ones(3, 2, dtype=torch.bool)}}
    got = taken(VALUE.cuda(), "layer_output", rec)
    assert got.device.type == "cuda" and torch.equal(got.cpu(), taken(VALUE, "layer_output", rec))


def test_metrics_and_edits_take_cpu_ids_and_operands():
    logits, d = torch.randn(3, 1, 11, generator=G), ["rows", "pos", "vocab_size"]
    ids = torch.tensor([[1, 2], [4, -1], [9, 0]])            # per row a set, padded with -1
    one = ids[:, :1]
    for fn, given in ((metrics.match, ids), (metrics.token_logit, one), (metrics.token_prob, ids),
                      (metrics.cross_entropy, ids)):
        torch.testing.assert_close(fn(logits.cuda(), given, d).cpu(), fn(logits, given, d), msg=fn.__name__)
    assert torch.equal(metrics.logit_diff(logits.cuda(), one, one.flip(0), d).cpu(),
                       metrics.logit_diff(logits, one, one.flip(0), d))
    against = torch.randn(3, 1, 11, generator=G)
    torch.testing.assert_close(metrics.kl(logits.cuda(), against).cpu(), metrics.kl(logits, against))
    direction = torch.randn(8, generator=G)
    torch.testing.assert_close(ops.call("project_out", VALUE.cuda(), {"direction": direction}).cpu(),
                               ops.call("project_out", VALUE, {"direction": direction}))
    noisy = ops.call("noise", VALUE.cuda(), {"scale": 0.5, "seed": 3}, dims=DIMS, rows=torch.arange(3),
                     positions=torch.arange(6).expand(3, 6))
    assert torch.equal(noisy.cpu(), ops.call("noise", VALUE, {"scale": 0.5, "seed": 3}, dims=DIMS, rows=torch.arange(3),
                                             positions=torch.arange(6).expand(3, 6)))


def test_a_batch_bound_on_the_gpu_reaches_its_values_without_the_host_waiting(monkeypatch):
    """What a batch bound on the GPU hands the engine (each record's positions, its selection and a key axis's
    indices, made there from pinned memory), the inputs fed to the model (one non-blocking copy) and an operand's
    rows looked up by id get there without the host waiting for the GPU, and an equal-width write and a read
    through `visit` make no host-device sync, nor do a gate's `apply` (which masks a missing vector out of its math
    whether or not one is) and a scale by a number as `edit` binds it (on the device, once per batch): under torch's
    sync debug mode set to raise, none of it raises, and each value is the CPU's. A mean taken off the model makes
    its one sync, asking whether any value is missing (`asking`)."""
    from clab.format.sweeps import points
    from clab.plan import entries

    plan = plan_of("patching")
    record = entries(points(plan.steps["patched"])[0])["patch"]
    pattern = entries(points(plan_of("attention_pattern_patching").steps["patched"])[0])["look_there"]
    value, other = VALUE.cuda(), OTHER[:, :1].cuda()
    batch = {"input_ids": torch.tensor([[0, 5, 6], [7, 8, 9]]), "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
             "position_ids": torch.tensor([[0, 0, 1], [0, 1, 2]])}
    swap = [{"op": "swap", "args": {"with": other}, "dims": DIMS}]
    rows = [[1], [2], [4]]
    probs = torch.randn(3, 2, 6, 6, generator=G)              # as the model serves it: [rows, num_heads, pos, key]

    def bound(device: str) -> tuple:
        picked = batches.site(record, {"hidden_size": [0, 7]}, rows, [0, 1, 0], 6, False, device)
        whole = batches.site(record, {}, rows, [0, 1, 0], 6, False, device)
        keyed = batches.site(pattern, {}, rows, [0, 1, 0], 6, False, device)
        return picked, whole, keyed

    narrow = [{**swap[0], "args": {"with": other[..., [0, 7]]}}]           # a write on a selection
    probs_there = probs.cuda()
    gate = featurizers.make("gate", {"over": ["hidden_size"]}, 8, {"theta": torch.linspace(1, -1, 8)}, None, None,
                            device="cuda:0", kept={"hidden_size": list(range(8))})
    gated = [{"op": "apply", "args": {"featurizer": gate, "with": other}, "dims": DIMS, "layers": 0}]
    asked: list[int] = []
    asking(monkeypatch, asked)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        picked, whole, keyed = bound("cuda:0")
        fed = standard.inputs(batch, ("input_ids", "attention_mask", "position_ids"), "cuda:0")
        looked = data.by_id(value, [2, 0])
        got: dict = {}
        out = batches.visit(value, "layer_output", [{**picked, "name": "r"}, {**whole, "ops": swap}], got)[0]
        chosen = batches.visit(value, "layer_output", [{**picked, "ops": narrow}], {})[0]
        seen: dict = {}
        batches.visit(probs_there, "attention_probabilities", [{**keyed, "name": "k"}], seen)   # a key axis read
        factor = [{"op": "scale", "args": {"factor": batches.placed(3.0, "cuda:0")}, "dims": DIMS}]
        changed = batches.visit(value, "layer_output", [{**whole, "ops": gated}, {**whole, "ops": factor}], {})[0]
        averaged = ops.call("mean", got["r"], {"over": ["pos"]}, dims=DIMS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    here = bound("cpu")
    assert all(t.device.type == "cuda" for rec in (picked, whole) for t in (rec["positions"], *rec["select"].values()))
    assert all(keyed["keys"][k].device.type == "cuda" for k in ("index", "own"))
    assert "keep" not in whole and whole["tok"].device.type == "cpu"
    assert all(torch.equal(keyed["keys"][k].cpu(), here[2]["keys"][k]) for k in ("index", "own"))
    assert all(torch.equal(fed[k].cpu(), batch[k]) for k in batch) and torch.equal(looked.cpu(), VALUE[[2, 0]])
    mine: dict = {}
    on_cpu = batches.visit(VALUE, "layer_output", [{**here[0], "name": "r"},
                                                   {**here[1], "ops": [{**swap[0], "args": {"with": OTHER[:, :1]}}]}],
                           mine)[0]
    assert torch.equal(got["r"].cpu(), mine["r"]) and torch.equal(out.cpu(), on_cpu)
    chosen_cpu = batches.visit(VALUE, "layer_output", [{**here[0], "ops": [{**narrow[0], "args": {"with": OTHER[:, :1, [0, 7]]}}]}],
                               {})[0]
    assert torch.equal(chosen.cpu(), chosen_cpu)
    keys_cpu: dict = {}
    batches.visit(probs, "attention_probabilities", [{**here[2], "name": "k"}], keys_cpu)
    assert torch.equal(seen["k"].cpu().nan_to_num(-9), keys_cpu["k"].nan_to_num(-9))      # NaN past a row's own keys
    on_host = featurizers.make("gate", {"over": ["hidden_size"]}, 8, {"theta": torch.linspace(1, -1, 8)}, None, None,
                               kept={"hidden_size": list(range(8))})
    changed_cpu = batches.visit(VALUE, "layer_output", [
        {**here[1], "ops": [{**gated[0], "args": {"featurizer": on_host, "with": OTHER[:, :1]}}]},
        {**here[1], "ops": [{**factor[0], "args": {"factor": torch.tensor(3.0)}}]}], {})[0]
    torch.testing.assert_close(changed.cpu(), changed_cpu, atol=1e-6, rtol=1e-6)
    assert asked == [1] and torch.equal(averaged.cpu(), ops.call("mean", mine["r"], {"over": ["pos"]}, dims=DIMS))


def test_an_op_step_runs_without_the_host_waiting(monkeypatch):
    """mean_ablation on the GPU, each op step (a mean of a harvest, then two logit differences) run with torch's
    sync debug mode set to raise and torch.cuda.synchronize refused: none of it raises, since a step that calls no
    model is timed as the host runs it and its ops wait for nothing but the mean's one question, whether any value
    is missing (`asking`: one sync, cheaper than computing both means), and the values are a plain run's to the
    bit."""
    from clab import steps

    engine = load("nnterp", plan_of("mean_ablation").model, "cuda")
    want = engine.execute(plan_of("mean_ablation"), batch_size=BATCH_SIZE).results
    measured, sync, strict = steps.measured, torch.cuda.synchronize, []

    def refused(*_: object) -> None:
        raise AssertionError("an op step synchronized the device")

    def timed(devices: list[str], cost: dict, call, synced: bool = True) -> None:
        if synced:
            return measured(devices, cost, call, synced)
        strict.append(cost)
        torch.cuda.set_sync_debug_mode("error")
        monkeypatch.setattr(torch.cuda, "synchronize", refused)
        try:
            measured(devices, cost, call, synced)
        finally:
            monkeypatch.setattr(torch.cuda, "synchronize", sync)
            torch.cuda.set_sync_debug_mode("default")
    monkeypatch.setattr(steps, "measured", timed)
    asked: list[int] = []
    asking(monkeypatch, asked)
    got = engine.execute(plan_of("mean_ablation"), batch_size=BATCH_SIZE).results
    assert len(strict) == 3 and all(set(c["memory"]) == {"cuda:0"} for c in strict) and asked == [1]
    assert same(got, want)


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
@pytest.mark.parametrize("name", ["patching", "generated_answer", "steer_renormalize", "noise_write",
                                  "sae_published_jumprelu", "pca_harvest"])
def test_an_engine_runs_a_document_on_the_device(engines, kind, options, name):
    """The document's numbers on `cuda` are the CPU's to float32 rounding, the run records the device and
    sees the residual stream there, and the results come home as CPU tensors."""
    plan = plan_of(name)
    there = load(kind, plan.model, "cuda", **options).execute(plan_of(name), batch_size=BATCH_SIZE)
    here = engines(kind, plan.model, **options).execute(plan_of(name), batch_size=BATCH_SIZE)
    assert there.run is not None and (there.run.device, there.run.devices) == ("cuda", ["cuda:0"])
    assert all(v.device.type == "cpu" for v in there.results.values() if isinstance(v, torch.Tensor))
    torch.testing.assert_close(there.results, here.results, atol=1e-5, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("device", ["auto", "cuda:0"])
@pytest.mark.parametrize("name", ["patching", "generated_answer"])
def test_a_device_places_the_model_and_its_inputs(engines, kind, device, name):
    """On nnterp and hooks "auto" (a map transformers computes; here one GPU, so all of the model on it) and one
    torch device both put the model there, each batch's inputs go where its first parameter is, and the
    numbers are the CPU's to float32 rounding; the run records the device asked for and where it ran."""
    plan = plan_of(name)
    there = load(kind, plan.model, device).execute(plan_of(name), batch_size=BATCH_SIZE)
    here = engines(kind, plan.model).execute(plan_of(name), batch_size=BATCH_SIZE)
    assert there.run is not None and (there.run.device, there.run.devices) == (device, ["cuda:0"])
    torch.testing.assert_close(there.results, here.results, atol=1e-5, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("device, where", [("cpu", "cpu"), ("cuda:0", "cuda:0")])
def test_one_torch_device_holds_the_model_beside_a_visible_gpu(capfd, kind, device, where):
    """One torch device is where the model is, `cpu` included while a GPU is visible (a transformers pipeline,
    which nnsight loads through, moves a model it is given no device for to the first GPU), and the loader
    is given `device` alone, so transformers does not warn of `device` beside `device_map`."""
    plan = plan_of("patching")
    run = load(kind, plan.model, device).execute(plan, batch_size=BATCH_SIZE).run
    assert run is not None and (run.device, run.devices) == (device, [where])
    assert "device_map" not in capfd.readouterr().err


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
def test_a_head_gate_is_a_null_on_the_device_and_comes_home_on_the_cpu(kind, options):
    """head_gates on `cuda`: its untrained gate over every head with the corrupt value is the clean run to the bit
    there too, and the trained gates come home on the CPU, laid out `[layers, num_heads]`."""
    def untrained(d):
        d["steps"]["fit"].update(epochs=0, objective={"sq": 1.0, "heads.mask": 0.0001})
        d["saves"] = ["clean.logits", "fit.train.run.logits", "clean_ld", "fit.train.ld"]
    null = edited("head_gates", untrained).compile()
    engine = load(kind, null.model, "cuda", **options)
    values = engine.execute(null, batch_size=BATCH_SIZE).results
    assert torch.equal(values["fit.train.run.logits"], values["clean.logits"])
    assert torch.equal(values["fit.train.ld"], values["clean_ld"])
    trained = engine.execute(plan_of("head_gates"), batch_size=BATCH_SIZE).results["fit.heads"]
    assert all(m["theta"].device.type == "cpu" and m["theta"].shape == (5, 4) for m in trained.values())


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_fit_trains_on_the_device(kind):
    """das trains its rotation through the model on `cuda`: its tensors come home on the CPU, shaped as on the
    CPU, still orthonormal. Ten epochs of optimizer steps apart, GPU and CPU rounding do not stay within float32
    of each other, so only the CPU run is pinned."""
    plan = plan_of("das")
    there = load(kind, plan.model, "cuda").execute(plan, batch_size=BATCH_SIZE)
    assert there.run is not None and there.run.devices == ["cuda:0"]
    basis = there.results["fit.rot"]["basis"]
    assert basis.device.type == "cpu" and basis.shape == (16, 8)
    torch.testing.assert_close(basis.T @ basis, torch.eye(8), atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind, device, options", [
    ("nnterp", "cuda", {}), ("hooks", "cuda", {}), ("ndif", "cuda", {"remote": "local"}),
    ("nnterp", "auto", {}), ("hooks", "auto", {})])
def test_a_fit_trains_its_featurizers_where_the_model_is(monkeypatch, kind, device, options):
    """das's rotation is made where the model is, on `cuda` whether one device or a device map ("auto") put it
    there: at every optimizer step its parameters, their gradients and the optimizer's state are on the model's
    device, and the trained bundle still comes home on the CPU."""
    from clab.steps import fit

    seen: set[str] = set()

    class Watched(torch.optim.AdamW):
        def step(self, closure=None):  # type: ignore[override]
            out = super().step(closure)
            for group in self.param_groups:
                for p in group["params"]:
                    seen.update({str(p.device), str(p.grad.device) if p.grad is not None else "none",
                                 *(str(v.device) for v in self.state[p].values() if isinstance(v, torch.Tensor)
                                   and v.dim() > 0)})
            return out

    monkeypatch.setitem(fit.OPTIMIZERS, "adamw", Watched)
    plan = plan_of("das")
    there = load(kind, plan.model, device, **options).execute(plan, batch_size=BATCH_SIZE)
    assert seen == {"cuda:0"}
    assert there.results["fit.rot"]["basis"].device.type == "cpu"
