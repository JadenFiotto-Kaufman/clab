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
    assert shown[0].startswith("generate_probe: run on nnterp, device cpu, device_map null, batch size 2, started ")
    assert shown[1] == "  served: float32 on cpu"
    assert shown[2].startswith("  versions: clab ")
    assert shown[3].split() == ["result", "step", "dims", "axes", "shape"]
    assert shown[4].split() == ["p_answer", "p_answer", "rows,", "pos", "frame",
                                '{"frame=0":[4,1],"frame=1":[4,1],"frame=2":[4,1]}']
    assert shown[5].split()[:5] == ["patched", "patched", "rows,", "new", "frame"]


def test_table_prints_aligned_columns_or_csv(out, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)                                   # a loaded plan's rows come back from the data file
    assert main(["table", out, "p_answer", "patched"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["at", "row", "pos", "pos_token", "call", "input", "cf_answer", "p_answer", "patched"]
    assert len(lines) == 13
    assert main(["table", out, "p_answer", "--csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "at,row,pos,pos_token,call,input,cf_answer,value" and len(lines) == 13
    assert lines[1].startswith('"{""frame"":0}",0,10,▁is,0,')                 # the prefill's last prompt token
    assert lines[5].startswith('"{""frame"":1}",0,11,Handler,1,')             # call 1 holds generated token 0


def test_table_refuses_a_name_that_was_not_saved(out, capsys):
    assert main(["table", out, "patched.logits"]) == 1
    assert "'patched.logits' was not saved; add it to saves" in capsys.readouterr().err
