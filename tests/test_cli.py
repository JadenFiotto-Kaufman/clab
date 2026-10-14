"""`clab show` and `clab table`: `Plan.load` and one call each, printed."""

import pytest

from clab.cli import main
from conftest import BATCH_SIZE, ROOT, plan_of


@pytest.fixture(scope="module")
def out(engines, tmp_path_factory) -> str:
    plan = plan_of("generate_probe")
    where = tmp_path_factory.mktemp("probe")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(where))
    return str(where)


def test_show_prints_each_result_and_how_it_ran(out, capsys):
    assert main(["show", out]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith("generate_probe: run on nnterp, remote False, batch size 2, started ")
    assert shown[1].startswith("  versions: clab ")
    assert shown[2].split() == ["result", "step", "dims", "axes", "shape"]
    assert shown[3].split() == ["p_answer", "p_answer", "rows,", "pos", "index",
                                '{"index=0":[4,1],"index=1":[4,1],"index=2":[4,1]}']
    assert shown[4].split()[:5] == ["patched", "patched", "rows,", "new", "index"]


def test_table_prints_aligned_columns_or_csv(out, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)                                   # a loaded plan's rows come back from the data file
    assert main(["table", out, "p_answer", "patched"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["at", "row", "step", "step_token", "input", "cf_answer", "p_answer", "patched"]
    assert len(lines) == 13
    assert main(["table", out, "p_answer", "--csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "at,row,step,step_token,input,cf_answer,value" and len(lines) == 13
    assert lines[5].startswith('"{""index"":1}",0,1,▁substr,')


def test_table_refuses_a_name_that_was_not_saved(out, capsys):
    assert main(["table", out, "patched.logits"]) == 1
    assert "'patched.logits' was not saved; add it to saves" in capsys.readouterr().err
