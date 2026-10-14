import json
import os
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

from clab.compile import compile, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS = os.path.join(ROOT, "documents")
SNAPSHOTS = os.path.join(ROOT, "tests", "snapshots")
UPDATE = os.environ.get("CLAB_UPDATE_SNAPSHOTS") == "1"


def document(name: str) -> str:
    return os.path.join(DOCUMENTS, f"{name}.json")


def plan_of(name: str) -> dict:
    return compile(load(document(name)), document(name))


def snapshot(name: str, value) -> Any:
    """The checked-in snapshot `name`; with CLAB_UPDATE_SNAPSHOTS=1, `value` is written as it first."""
    path = os.path.join(SNAPSHOTS, name)
    if UPDATE:
        with open(path, "w") as f:
            json.dump(value, f, indent=1, ensure_ascii=False)
            f.write("\n")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def engines():
    """Each engine loaded once per model block, for the whole session."""
    from clab.engines import ENGINES

    loaded: dict = {}

    def get(kind: str, model: dict):
        key = (kind, json.dumps(model, sort_keys=True))
        if key not in loaded:
            loaded[key] = ENGINES[kind](model)
        return loaded[key]

    return get
