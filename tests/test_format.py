"""FORMAT.md and schema.json are generated from the code; the committed copies must be what the code generates now."""

import json
import os

import copy

import jsonschema

from clab.cli import describe, markdown, schema
from conftest import NAMES, REAL, ROOT, document


def test_format_md_is_what_the_code_generates():
    with open(os.path.join(ROOT, "FORMAT.md")) as f:
        committed = f.read()
    assert committed == markdown(), "FORMAT.md is stale: run `clab describe --markdown` from the repo root"


def test_schema_json_is_what_the_code_generates():
    with open(os.path.join(ROOT, "schema.json")) as f:
        committed = json.load(f)
    assert committed == schema(), "schema.json is stale: run `clab schema > schema.json`"


def test_the_schema_accepts_the_corpus_and_refuses_what_the_plan_refuses():
    validator = jsonschema.Draft202012Validator(schema())
    for name in NAMES + REAL:
        with open(document(name)) as f:
            assert validator.is_valid(json.load(f)), name
    with open(document("patching")) as f:
        doc = json.load(f)
    for change in (lambda d: d["steps"]["iia"].update(op="matches"),
                   lambda d: d["sites"]["target"].update(component="nope"),
                   lambda d: d["steps"]["iia"].update(expect="pairs.cf_answer"),
                   lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"index": -1, "span": [0, 2]}),
                   lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"index": "all"}),
                   lambda d: d["steps"]["patched"]["writes"]["patch"].update(pos={"index": [1, 1]}),
                   lambda d: d["sites"]["target"].update(layers="1:2:3:4"),
                   lambda d: d["steps"]["patched"]["writes"]["patch"].update(
                       pos={"scope": {"column": "pairs.base_answer", "lower": True}}),
                   lambda d: d.update(compiled=True),
                   lambda d: d["steps"]["iia"].update(dims=["rows"])):
        bad = copy.deepcopy(doc)
        change(bad)
        assert not validator.is_valid(bad)


def test_describe_lists_every_registry():
    text = describe()
    for word in ("records:", "step kinds:", "metrics:", "reductions:", "edits:", "transforms:", "components:",
                 "featurizer kinds:",
                 "pos:", "logit_diff(", "layer_output", "subspace(", "fit(", "decode("):
        assert word in text, word


def test_describe_lists_the_engines_and_spells_literals_as_json():
    lines = describe("engines").splitlines()
    assert lines[2:6] == ["      loads with:",
                          '        device: str = "cpu" — one torch device ("cpu", "cuda:0") as the loader\'s device, or '
                          '"auto", "balanced", ... as its device_map',
                          "      runs with:",
                          "        batch_size: int = 8 — how many of a step's rows go through the model in one "
                          "forward or generation"]
    assert lines[14].startswith('        remote: one of true, "local" = true — true for the NDIF server')
    assert "    match(expected: Tokens)  [" in describe("ops")


def test_every_engine_option_says_what_it_takes():
    """`describe engines` reads what each option takes off the docstring of the method that declares it."""
    from clab.cli import run_options
    from clab.engines import ENGINES

    for name in ENGINES:
        for side, listed in run_options(name).items():
            assert listed and all(text for _, text in listed), (name, side, listed)
