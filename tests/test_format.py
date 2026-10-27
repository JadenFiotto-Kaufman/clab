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


def test_the_edits_group_in_the_prose_is_generated_from_the_declared_ops():
    """`ops.ABOUT` and the `Intervention` docstring name the edits group from `DECLARED`, in the tables' order,
    so a new edit joins both and FORMAT.md goes stale until regenerated (the first test here)."""
    import inspect

    from clab import ops, plan

    edits = [n for n in ops.OPS if ops.DECLARED[n]["group"] == "edits"]
    listed = ops.listed("edits")
    assert listed == ", ".join(f"`{n}`" for n in edits) and "`clamp`" in listed and "`swap`" in listed
    assert f"**edits** ({listed}) keep the value's layout" in ops.ABOUT
    assert f"(the edits group: {listed}) writes back" in inspect.getdoc(plan.Intervention)
    assert "<the edits group>" not in ops.ABOUT + inspect.getdoc(plan.Intervention) + markdown()


def test_the_components_table_shows_a_selections_layout_and_which_values_readout_projects():
    """The table lays a component out as a selection presents it, `rows, pos` then its own axes, and says whether
    it is a residual-stream value `readout` may project (`stream`), in `describe` and in FORMAT.md alike."""
    lines = describe("components").splitlines()
    assert "  layer_output             [rows, pos, hidden_size]  narrowed by layers, hidden_size  stream  — " \
           "the residual stream leaving block i" in lines
    assert "  attention_probabilities  [rows, pos, num_heads, key]  narrowed by layers, num_heads, key  — " \
           "the pattern the values are mixed with" in lines
    table = markdown()
    assert "| component | dims | narrowed by | stream | meaning |" in table
    assert "| `layer_output` | rows, pos, hidden_size | layers, hidden_size | yes | the residual stream leaving " \
           "block i |" in table
    assert "| `mlp_norm_output` | rows, pos, hidden_size | layers, hidden_size | no | the normed stream entering " \
           "the MLP |" in table
    rows = [line for line in table.partition("## Components")[2].partition("## Featurizer kinds")[0].splitlines()
            if line.startswith("| `")]
    assert len(rows) == 21 and not any("batch" in r or "seq" in r or "query" in r for r in rows)


def test_describe_lists_the_engines_and_spells_literals_as_json():
    lines = describe("engines").splitlines()
    assert lines[2] == ('      device: str = "cpu" — one torch device ("cpu", "cuda:0") as the loader\'s device, or '
                        '"auto", "balanced", ... as its device_map; under tp, "cuda" or "cpu"')
    assert lines[3].startswith("      tp: int = 1 — how many ranks tensor parallelism shards the model over")
    assert lines[8].startswith('      remote: one of true, "local" = true — true for the NDIF server')
    assert "    match(expected: Tokens)  [" in describe("ops")


def test_every_engine_option_says_what_it_takes():
    """`describe engines` reads what each option takes off the docstring of the method that declares it."""
    from clab.docs import run_options
    from clab.engines import ENGINES

    for name in ENGINES:
        listed = run_options(name)
        assert listed and all(text for _, text in listed), (name, listed)
