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
# Documents whose steps a later milestone serves: featurizers and fits (M4).
LATER = {"das", "das_apply", "dbm", "pca_control", "random_subspace", "sae_feature_ablation"}
GENERATE = ["generate_probe", "generated_answer"]
FORWARD = [n for n in NAMES if n not in LATER and n not in GENERATE]
# What the hooks engine refuses by name: values nnterp reads inside a module's forward.
INSIDE = {"attention_knockout", "attention_pattern_patching", "attention_query"}


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
    """`a` and `b` (plain values) equal, floats to 1e-6."""
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
        assert abs(a - b) <= 1e-6, (path, a, b)
    else:
        assert a == b, (path, a, b)


def same(a, b) -> bool:
    """Bit-equal, all the way down."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="session")
def engines():
    """Each engine loaded once per model block, for the whole session."""
    from clab.engines import ENGINES

    loaded: dict = {}

    def get(kind: str, model: Model):
        key = (kind, model.model_dump_json())
        if key not in loaded:
            loaded[key] = ENGINES[kind](model)
        return loaded[key]

    return get


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> str:
    """A copy of the corpus whose `documents/artifacts` holds stand-in bundles for the three documents that
    load one, each with the header its document checks; the tensors are zeros."""
    root = tmp_path_factory.mktemp("corpus")
    shutil.copytree(DOCUMENTS, root / "documents")
    os.symlink(os.path.join(ROOT, "data"), root / "data")
    (root / "documents" / "artifacts").mkdir(exist_ok=True)
    model = {"model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
             "model.revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}
    for file, header, tensors in (
            ("fit.rot.safetensors", {"kind": "subspace", "k": "8"}, {"basis": torch.zeros(16, 8)}),
            ("basis.safetensors", {"kind": "pca", "k": "4"}, {"basis": torch.zeros(16, 4), "mean": torch.zeros(16)}),
            ("sae.safetensors", {"kind": "sae"}, {"W_enc": torch.zeros(16, 32)})):
        save_file(tensors, str(root / "documents" / "artifacts" / file), metadata={"format": FORMAT, **model, **header})
    return str(root)
