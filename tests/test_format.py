"""FORMAT.md and schema.json are generated from the code; the committed copies must be what the code generates now."""

import json
import os

from clab.cli import describe, markdown
from clab.plan import Plan
from conftest import ROOT


def test_format_md_is_what_the_code_generates():
    with open(os.path.join(ROOT, "FORMAT.md")) as f:
        committed = f.read()
    assert committed == markdown(), "FORMAT.md is stale: run `clab describe --markdown` from the repo root"


def test_schema_json_is_what_the_code_generates():
    with open(os.path.join(ROOT, "schema.json")) as f:
        committed = json.load(f)
    assert committed == Plan.model_json_schema(), "schema.json is stale: run `clab schema > schema.json`"


def test_describe_lists_every_registry():
    text = describe()
    for word in ("records:", "step kinds:", "metrics:", "reductions:", "edits:", "components:", "featurizer kinds:",
                 "pos:", "logit_diff(", "layer_output", "subspace(", "fit(", "decode("):
        assert word in text, word
