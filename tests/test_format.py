"""FORMAT.md is generated from the code; the committed copy must be what the code generates now."""

import os

from conftest import ROOT

from clab.docs import describe, markdown


def test_format_md_is_what_the_code_generates():
    with open(os.path.join(ROOT, "FORMAT.md")) as f:
        committed = f.read()
    assert committed == markdown(), "FORMAT.md is stale: run `clab describe --markdown` from the repo root"


def test_describe_lists_every_registry():
    text = describe()
    for word in ("records:", "step kinds:", "metrics:", "reductions:", "edits:", "transforms:", "components:",
                 "featurizer kinds:",
                 "pos:", "logit_diff(", "layer_output", "subspace(", "fit(", "decode(", "detokenize("):
        assert word in text, word


def test_describe_lists_the_engines_and_spells_literals_as_json():
    lines = describe("engines").splitlines()
    assert lines[2] == ('      device: str = "cpu" — one torch device ("cpu", "cuda:0") as the loader\'s device, or '
                        '"auto", "balanced", ... as its device_map')
    assert lines[7].startswith('      remote: one of true, "local" = true — true for the NDIF server')
    assert "    match(expected: Tokens)  [" in describe("ops")


def test_every_engine_option_says_what_it_takes():
    """`describe engines` reads what each option takes off the docstring of the method that declares it."""
    from clab.docs import run_options
    from clab.engines import ENGINES

    for name in ENGINES:
        listed = run_options(name)
        assert listed and all(text for _, text in listed), (name, listed)
