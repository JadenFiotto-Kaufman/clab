"""FORMAT.md is generated from the code; the committed copy must be what the code generates now."""

import os

from clab.cli import describe, markdown
from conftest import ROOT


def test_format_md_is_what_the_code_generates():
    with open(os.path.join(ROOT, "FORMAT.md")) as f:
        committed = f.read()
    assert committed == markdown(), "FORMAT.md is stale: run `clab describe --markdown` from the repo root"


def test_describe_lists_every_registry():
    text = describe()
    for word in ("step kinds:", "ops", "components:", "featurizer kinds:", "pos:", "logit_diff(", "layer_output",
                 "subspace(", "fit("):
        assert word in text, word
