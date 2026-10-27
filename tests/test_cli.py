"""`clab show` and `clab table`: `Plan.load` and one call each, printed; what `run` prints and refuses."""

import json
import os
import sys

import pytest
from conftest import BATCH_SIZE, ROOT, document, edited, plan_of

from clab.cli import main, shaped


@pytest.fixture(scope="module")
def out(engines, tmp_path_factory) -> str:
    plan = plan_of("generate_probe")
    where = tmp_path_factory.mktemp("probe")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(where))
    return str(where)


def test_show_prints_how_it_ran_what_each_step_cost_and_each_result(out, capsys):
    assert main(["show", out]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith("generate_probe: run on nnterp, device cpu, tp 1, batch size 2, started ")
    assert shown[1] == "  served: float32 on cpu"
    assert shown[2].startswith("  versions: clab ")
    assert shown[3].split() == ["step", "at", "seconds", "memory", "forwards", "tokens"]
    assert [line.split()[:2] for line in shown[4:11]] == [["counterfactual", "—"]] + \
        [[step, f"frame={k}"] for step in ("patched", "p_answer") for k in range(3)]
    assert shown[11].split() == ["result", "step", "dims", "axes", "shape"]
    assert shown[12].split() == ["p_answer", "p_answer", "rows,", "pos", "frame",
                                 '{"frame=0":[4,1],"frame=1":[4,1],"frame=2":[4,1]}']
    assert shown[13].split()[:5] == ["patched", "patched", "rows,", "new", "frame"]


@pytest.mark.parametrize("name", ["generated_contains", "das"])
def test_show_prints_a_value_with_no_axes_and_the_manifest_keeps_its_dims(engines, tmp_path, capsys, name):
    """A `mean` over everything (generated_contains's `acc`) and a trained featurizer (das's `fit.rot`) have no
    axes: `show` prints them with the rest, `[]`, and the manifest records their `dims` as it does every value's."""
    plan = plan_of(name)
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert all(any(line.split()[:1] == [saved] for line in shown.splitlines()) for saved in plan.saves)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert all("dims" in manifest["results"][saved] for saved in plan.saves)
    assert manifest["results"]["acc" if name == "generated_contains" else "fit.rot"]["dims"] == []


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


def test_a_shape_is_printed_once_per_value_and_a_token_list_as_its_full_shape(engines, tmp_path, capsys):
    """`run` and `show` print a swept value's shape counted over its points, the distinct shapes listed up to three
    (`shaped`), and a nested token list's (`top_k`'s `tokens`) as its full shape, which the manifest records."""
    assert shaped({"a": [4, 1], "b": [4, 2], "c": [4, 1]}, swept=True) == "3 points, each [4,1] or [4,2]"
    assert shaped({k: [4, k] for k in range(4)}, swept=True) == "4 points of 4 shapes"
    assert shaped({"ids": [4, 1, 3], "tokens": [4, 1, 3]}, swept=False) == '{"ids":[4,1,3],"tokens":[4,1,3]}'
    out = tmp_path / "readout"
    assert main(["run", document("readout"), "--batch-size", str(BATCH_SIZE), "--out", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[1].split() == ["top3.json", "top3", '{"ids":[4,1,3],"probs":[4,1,3],"tokens":[4,1,3]}']
    assert json.loads((out / "manifest.json").read_text())["results"]["top3"]["shape"]["tokens"] == [4, 1, 3]
    out = tmp_path / "pos_sweep"
    assert main(["run", document("pos_sweep"), "--batch-size", str(BATCH_SIZE), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[1].split() == ["iia.safetensors", "iia", "3", "points,", "each", "[4,1]"]
    assert main(["show", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].split()[-4:] == ["3", "points,", "each", "[4,1]"]

def test_a_reader_that_goes_away_ends_show_quietly(out, monkeypatch):
    """`clab show OUT | head -1`: the pipe's reader is gone before `show` has written, and the command returns
    0 with no traceback (stdout is sent to /dev/null for the interpreter's own flush at exit)."""
    reading, writing = os.pipe()
    os.close(reading)
    monkeypatch.setattr(sys, "stdout", os.fdopen(writing, "w"))
    assert main(["show", out]) == 0
def test_the_cli_refuses_what_run_cannot_do_in_one_line(tmp_path, capsys):
    """A missing document, a device that is no torch device, a CUDA device where none is visible (the suite runs
    with none) and a batch size below 1 are each one `clab:` line and exit status 1."""
    out = str(tmp_path / "out")
    assert main(["run", "nowhere.json", "--out", out]) == 1
    assert capsys.readouterr().err == "clab: [Errno 2] No such file or directory: 'nowhere.json'\n"
    assert main(["run", document("patching"), "--device", "gpu", "--out", out]) == 1
    assert capsys.readouterr().err == ("clab: device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map "
                                       "transformers computes ('auto', 'balanced', 'balanced_low_0', 'sequential')\n")
    for engine in ("nnterp", "hooks", "ndif"):
        assert main(["run", document("patching"), "--engine", engine, "--device", "cuda:0", "--out", out]) == 1
        assert capsys.readouterr().err == ("clab: device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() "
                                           "is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)\n")
    assert main(["run", document("patching"), "--batch-size", "0", "--out", out]) == 1
    assert capsys.readouterr().err == "clab: batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)\n"
    assert not os.path.exists(out)                       # and nothing loaded: each refusal came before the model
