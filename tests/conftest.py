import json
import os
import shutil
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from clab import Plan  # noqa: E402
from clab.featurizers import FORMAT  # noqa: E402
from clab.plan import Model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS = os.path.join(ROOT, "documents")
SNAPSHOTS = os.path.join(ROOT, "tests", "snapshots")
UPDATE = os.environ.get("CLAB_UPDATE_SNAPSHOTS") == "1"
BATCH_SIZE = 2      # results are reproducible to the bit only at a fixed batch size
NAMES = sorted(os.path.splitext(n)[0] for n in os.listdir(DOCUMENTS) if n.endswith(".json"))
# The real-model documents, by their path under documents/: compiled and schema-checked with the rest, run only by
# test_real.py, on a GPU.
REAL = sorted(f"real/{os.path.splitext(n)[0]}" for n in os.listdir(os.path.join(DOCUMENTS, "real"))
              if n.endswith(".json"))
# Documents that train or load a featurizer, pinned in test_fit.py.
FIT = ["das", "das_apply", "dbm", "lora_apply", "lora_train", "pca_control", "random_subspace",
       "sae_feature_ablation", "sae_published_jumprelu", "sae_published_topk", "sae_train", "sae_train_ablation"]
GENERATE = ["generate_probe", "generated_answer", "generated_confidence", "generated_contains"]
FORWARD = [n for n in NAMES if n not in FIT and n not in GENERATE]
# What the hooks engine refuses by name: values nnterp reads inside a module's forward.
INSIDE = {"attention_knockout", "attention_pattern_patching", "attention_query", "gaze_heads"}
# What the ndif engine refuses by name: a chat with an image part, a file beside the document the server cannot read.
IMAGES = {"gaze_heads"}


def document(name: str) -> str:
    return os.path.join(DOCUMENTS, f"{name}.json")


def plan_of(name: str) -> Plan:
    return Plan(document(name)).compile()


def edited(name: str, change) -> Plan:
    """Document `name` with `change` applied to its JSON, read as if from its own file."""
    with open(document(name)) as f:
        doc = json.load(f)
    change(doc)
    return Plan(**doc, document=document(name))


def snapshot(name: str, value) -> Any:
    """The checked-in snapshot `name`; with CLAB_UPDATE_SNAPSHOTS=1, `value` is written as it first."""
    path = os.path.join(SNAPSHOTS, name)
    if UPDATE:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=1, ensure_ascii=False)
            f.write("\n")
    with open(path) as f:
        return json.load(f)


def plain(value) -> Any:
    """A result as JSON: tensors as nested lists, coordinate keys as text."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def close(a, b, path: str = "") -> None:
    """`a` and `b` (plain values) equal, floats to 1e-6, NaN to NaN."""
    assert type(a) is type(b) or {type(a), type(b)} <= {int, float}, path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert abs(a - b) <= 1e-6 or a != a and b != b, (path, a, b)         # NaN, a missing row, on both
    else:
        assert a == b, (path, a, b)


def same(a, b) -> bool:
    """Bit-equal, all the way down, a missing value (NaN) where the other's is."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.isnan(), b.isnan()) and \
            torch.equal(a[~a.isnan()], b[~b.isnan()])
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


# The modules a run's request names by reference, and the marks of something shipped by value: a class
# rebuilt from its definition (cloudpickle) or a function from its source (nnsight).
REFERENCED = (b"clab.plan", b"clab.engines.ndif", b"clab.steps")
BY_VALUE = (b"_make_skeleton_class", b"_load_function")


def by_reference(blob: bytes) -> bool:
    """Whether a serialized request names clab's plan, engine and steps by reference and ships nothing by value."""
    return all(name in blob for name in REFERENCED) and not any(mark in blob for mark in BY_VALUE)


def stand_in(engines, plan: Plan, monkeypatch) -> tuple[Plan, dict]:
    """`plan` run on the ndif engine against a stand-in for a server:
    serialize the session the client's meta shell opens, run the *deserialized* tracer with the persistent
    objects of a loaded model (the nnterp engine's, which is what a server holds), take what its outermost
    boundary pushes, round-trip that through torch.save/torch.load as a response does, then mark it and push
    it into the caller's frame as RemoteBackend._push does. Returns what came home and what the trip saw:
    whether the request named clab's plan, engines and steps by reference with nothing shipped by value, and
    the names and types that came back, with the keys and types of the one dict that did."""
    import io

    from nnsight.schema.request import RequestModel
    from nnsight.tracing.backend import Backend
    from nnsight.tracing.tracer import _saves, mark
    from nnsight.tracing.util import push

    client, server = engines("ndif", plan.model, remote=True), engines("nnterp", plan.model)
    trip: dict = {}

    class StandIn(Backend):
        def __call__(self, tracer):
            blob = RequestModel.serialize(tracer, compress=False)
            trip["by_reference"] = by_reference(blob)
            deserialized = RequestModel.deserialize(blob, server.model._remoteable_persistent_objects(),
                                                    compress=False)
            assert deserialized.info is not None
            deserialized.execute(deserialized.info.code)
            result = {k: v for k, v in deserialized.info.frame.f_locals.items() if id(v) in _saves()}
            buffer = io.BytesIO()
            torch.save(result, buffer)
            buffer.seek(0)
            result = torch.load(buffer, weights_only=False)
            trip.update(names=sorted(result), types=[type(v).__name__ for v in result.values()],
                        carried={k: type(x).__name__ for v in result.values() if isinstance(v, dict)
                                 for k, x in v.items()})
            for value in result.values():
                mark(value)
            assert tracer.info is not None
            push(tracer.info.frame, result)

    monkeypatch.setattr(type(client.model), "_remote_backend", lambda self, *args: StandIn())
    return client.execute(plan, batch_size=BATCH_SIZE), trip


# What the stand-in's trip shows of every run: the request by reference, and the results and run record home.
TRIP = {"by_reference": True, "names": ["home"], "types": ["dict"], "carried": {"results": "dict", "run": "Run"}}


@pytest.fixture(scope="session")
def engines():
    """Each engine loaded once per model block and options, for the whole session."""
    from clab.engines import load

    loaded: dict = {}

    def get(kind: str, model: Model, **options):
        key = (kind, model.model_dump_json(), tuple(sorted(options.items())))
        if key not in loaded:
            loaded[key] = load(kind, model, **options)
        return loaded[key]

    return get


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> str:
    """A copy of the corpus whose `documents/artifacts` holds, beside the committed SAE fixture, stand-in bundles
    for the four documents that load what another writes, each with the header its document checks; the
    tensors are the first 8 unit vectors, a zero basis, a zero dictionary and a zero adapter."""
    root = tmp_path_factory.mktemp("corpus")
    shutil.copytree(DOCUMENTS, root / "documents")
    os.symlink(os.path.join(ROOT, "data"), root / "data")
    stand_ins(str(root))
    return str(root)


def stand_ins(root: str) -> None:
    """The stand-in bundles the `corpus` fixture's artifacts hold."""
    for file, header, tensors in (
            ("fit.rot.safetensors", {"kind": "subspace", "k": "8", "parametrization": "cayley", "document": "das"},
             {"basis": torch.eye(16)[:, :8].contiguous()}),
            ("basis.safetensors", {"kind": "pca", "k": "4", "document": "pca_harvest"},
             {"basis": torch.zeros(16, 4), "mean": torch.zeros(16)}),
            ("fit.dictionary.safetensors", {"kind": "sae", "activation": "relu", "centered": "true",
                                            "document": "sae_train"},
             {"W_enc": torch.zeros(16, 32), "b_enc": torch.zeros(32), "W_dec": torch.zeros(32, 16),
              "b_dec": torch.zeros(16)}),
            ("fit.lora.safetensors", {"kind": "lora", "rank": "4", "alpha": "8", "document": "lora_train"},
             {"A": torch.zeros(16, 4), "B": torch.zeros(4, 16)})):
        save_file(tensors, os.path.join(root, "documents", "artifacts", file), metadata={**bundle_header(), **header})


def bundle_header(**changes: str) -> dict:
    """The header of a bundle made on the tiny Llama at layer_output layer 0, with `changes`."""
    return {"format": FORMAT, "model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "model.revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "model.dtype": "float32",
            "component": "layer_output", "layer": "0", "width": "16", **changes}
