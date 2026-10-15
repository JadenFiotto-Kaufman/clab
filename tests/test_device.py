"""On a GPU: every tensor clab builds where the plan runs (positions, masks, per-row indices, token ids,
operands) meets an activation on that activation's device, and the engines run a document on `device`.

The project's CPU-only suite skips this file. On this host the project venv's torch cannot use the driver;
an env whose torch can, with clab installed in it, runs it with one command from the repo root:

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m pytest tests/test_device.py -q
"""

import pytest
import torch

from clab.engines import load
from clab.runtime import featurizers, ops
from clab.runtime.ops import metrics
from conftest import BATCH_SIZE, plan_of

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DIMS = ["rows", "pos", "hidden_size"]
G = torch.Generator().manual_seed(0)
VALUE = torch.randn(3, 6, 8, generator=G)
OTHER = torch.randn(3, 2, 8, generator=G)


def writes() -> list[dict]:
    """Writes built as a batch binds them, on the CPU: a ragged swap with its `keep`, an add of a CPU operand
    on the features a gate keeps, a narrowing to two hidden units, each row's own key indices, and a
    renormalize."""
    gate = featurizers.make("gate", {}, 8, {"theta": torch.linspace(-1, 1, 8)}, None, None)
    return [
        {"positions": torch.tensor([[1, 2], [3, 3], [4, 5]]), "keep": torch.tensor([[1, 1], [1, 0], [1, 1]]).bool(),
         "select": {}, "ops": [{"op": "swap", "args": {"with": OTHER}, "featurizer": None, "dims": DIMS}]},
        {"positions": torch.tensor([[0], [0], [0]]), "select": {}, "ops": [
            {"op": "add", "args": {"with": torch.ones(2), "scale": 2.0}, "featurizer": gate,
             "features": torch.tensor([1, 6]), "dims": ["rows", "pos", "features"]}]},
        {"positions": torch.tensor([[5], [5], [5]]), "select": {2: torch.tensor([0, 7])},
         "ops": [{"op": "scale", "args": {"factor": 3.0}, "featurizer": None, "dims": DIMS}]},
        {"positions": torch.tensor([[2], [2], [2]]), "select": {2: torch.tensor([[1, 2], [3, 4], [5, 6]])},
         "ops": [{"op": "sub", "args": {"with": 1.0}, "featurizer": None, "dims": DIMS}]},
        {"positions": torch.tensor([[3], [3], [3]]), "select": {}, "rows": torch.arange(3),
         "ops": [{"op": "renormalize", "args": {}, "featurizer": None, "dims": DIMS}]},
    ]


def test_a_write_built_on_the_cpu_lands_on_the_values_device():
    on_cpu = ops.write(VALUE, "layer_output", writes())
    on_gpu = ops.write(VALUE.cuda(), "layer_output", writes())
    assert on_gpu.device.type == "cuda"
    torch.testing.assert_close(on_gpu.cpu(), on_cpu, atol=1e-6, rtol=1e-6)


def test_a_read_built_on_the_cpu_selects_on_the_values_device():
    rec = {"positions": torch.tensor([[1, 2], [3, 4], [0, 5]]), "select": {2: torch.tensor([[1, 2], [3, 4], [5, 6]])}}
    got = ops.select(VALUE.cuda(), "layer_output", rec)
    assert got.device.type == "cuda" and torch.equal(got.cpu(), ops.select(VALUE, "layer_output", rec))


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
    torch.testing.assert_close(there.results, here.results, atol=1e-5, rtol=1e-5)


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
    torch.testing.assert_close(there.results, here.results, atol=1e-5, rtol=1e-5)


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
    from clab.runtime.steps import fit

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
