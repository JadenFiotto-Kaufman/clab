"""`clab show` and `clab table`: `Plan.load` and one call each, printed; what `run` prints and refuses."""

import json
import os
import sys

import pytest
from conftest import BATCH_SIZE, ROOT, document, edited, plan_of

from clab import cli
from clab.cli import compact, main, shaped


@pytest.fixture(scope="module")
def out(engines, tmp_path_factory) -> str:
    plan = plan_of("generate_probe")
    where = tmp_path_factory.mktemp("probe")
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(where))
    return str(where)


def test_show_prints_how_it_ran_what_each_step_cost_where_each_entry_resolved_and_each_result(out, capsys):
    assert main(["show", out]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith("generate_probe: run on nnterp, device cpu, tp 1, batch size 2, padding dataset, "
                               "started ")
    assert shown[1] == "  served: float32 on cpu"
    assert shown[2].startswith("  versions: clab ")
    assert shown[3].split() == ["step", "at", "serves", "seconds", "memory", "forwards", "tokens", "real"]
    assert [line.split()[:2] for line in shown[4:9]] == [["counterfactual", "—"], ["patched", "frame=0"]] + \
        [["p_answer", f"frame={k}"] for k in range(3)]                  # one generation serves the three points
    assert shown[5].split(maxsplit=2)[2].startswith("all 3 points")
    assert shown[9].split() == ["entry", "at", "positions"]
    assert shown[10].split() == ["counterfactual.v_cf", "—", "[8],", "[8],", "[10],", "[8]"]
    assert [line.split(maxsplit=1) for line in shown[11:15]] == [       # the write lands alike at every frame
        ["patched.patch", "all 3 points  [10], [8], [8], [8]"],
        ["patched.logits", "frame=0       [10], [8], [8], [8]"],
        ["patched.logits", "frame=1       [11], [9], [9], [9]"],          # call 1 holds generated token 0
        ["patched.logits", "frame=2       [12], [10], [10], [10]"]]
    assert shown[15].split() == ["result", "step", "dims", "axes", "shape"]
    assert shown[16].split() == ["p_answer", "p_answer", "rows,", "pos", "frame", "3", "points,", "each", "[4,1]"]
    assert shown[17].split() == ["patched", "patched", "rows,", "new", "—", "[4,3]"]   # the ids depend on no axis


def test_show_prints_a_fits_evals_as_the_steps_after_it(engines, tmp_path, capsys):
    """A fit's evals are steps after it (`<fit>.<eval>.<step>`): `show` prints each one's cost after the fit's, where
    each of its entries resolved under its own dataset (lora_train's held-out rows are 9 tokens each), and each
    eval member beside the eval step that made it, as the manifest records it."""
    plan = plan_of("lora_train")
    engines("hooks", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    steps = shown.index(next(line for line in shown if line.split()[:2] == ["step", "at"]))
    assert [line.split()[0] for line in shown[steps + 1:steps + 6]] == [
        "fit", "fit.held_out.adapted", "fit.held_out.ce", "fit.train.adapted", "fit.train.ce"]
    assert [line.split()[5] for line in shown[steps + 2:steps + 6]] == ["1", "0", "1", "0"]    # forwards
    assert next(line for line in shown if line.split()[:1] == ["fit.held_out.adapted.logits"]).split()[1:] == \
        ["—", "[8]", "on", "all", "2", "rows"]
    results = shown.index(next(line for line in shown if line.split()[:2] == ["result", "step"]))
    assert {line.split()[0]: line.split()[1] for line in shown[results + 1:]} == {
        "fit.lora": "fit", "fit.history.ce": "fit", "fit.train.ce": "fit.train.ce",
        "fit.held_out.ce": "fit.held_out.ce"}
    with open(tmp_path / "manifest.json") as f:
        assert json.load(f)["results"]["fit.held_out.ce"]["step"] == "fit.held_out.ce"


def test_show_on_a_compiled_plan_that_has_not_run_prints_who_serves_whom():
    """A compiled plan has its schedule before any run: `show` prints which points each run of a step serves
    (generate_probe's one generation its three frames, `p_answer` one point each), and nothing it would need a
    run for."""
    from clab.cli import show

    shown = show(plan_of("generate_probe")).splitlines()
    assert shown[0] == "generate_probe: compiled, not run"
    assert shown[1].split() == ["step", "at", "serves"]
    assert [line.split(maxsplit=2) for line in shown[2:]] == [
        ["counterfactual", "—", "—"], ["patched", "frame=0", "all 3 points"],
        ["p_answer", "frame=0", "—"], ["p_answer", "frame=1", "—"], ["p_answer", "frame=2", "—"]]


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
    assert shaped({"pos=-1": [4, 1]}, swept=True) == "1 point, [4,1]"
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


@pytest.mark.parametrize("command", [["show"], ["table", "p_answer"]])
def test_a_reader_that_goes_away_mid_print_ends_the_command_quietly(out, monkeypatch, capsys, command):
    """`clab show OUT | head`, the reader gone while `show` prints more than the pipe holds: the write fails inside
    the command, which still returns 0 and says nothing on stderr, where it said `clab: [Errno 32] Broken pipe`
    (a broken pipe is an OSError, which the command refuses in one line). A line-buffered stdout makes the first
    line's write fail as a full pipe's would."""
    reading, writing = os.pipe()
    os.close(reading)
    monkeypatch.setattr(sys, "stdout", os.fdopen(writing, "w", buffering=1))
    assert main([command[0], out, *command[1:]]) == 0
    assert capsys.readouterr().err == ""

def test_the_cli_refuses_what_run_cannot_do_in_one_line(tmp_path, capsys):
    """A missing document, a device that is no torch device, a CUDA device where none is visible (the suite runs
    with none) and a batch size below 1 are each one `clab:` line and exit status 1."""
    out = str(tmp_path / "out")
    assert main(["run", "nowhere.json", "--out", out]) == 1
    assert capsys.readouterr().err == "clab: no document at 'nowhere.json'\n"
    assert main(["show", "nowhere"]) == 1
    assert capsys.readouterr().err == "clab: no run at 'nowhere' (no manifest.json); run the document first\n"
    assert main(["run", document("patching"), "--device", "gpu", "--out", out]) == 1
    assert capsys.readouterr().err == ("clab: device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map "
                                       "transformers computes ('auto', 'balanced', 'balanced_low_0', 'sequential')\n")
    for engine in ("nnterp", "hooks"):
        assert main(["run", document("patching"), "--engine", engine, "--device", "cuda:0", "--out", out]) == 1
        assert capsys.readouterr().err == ("clab: device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() "
                                           "is False); run on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)\n")
    assert main(["run", document("patching"), "--engine", "ndif", "--device", "cuda:0", "--out", out]) == 1
    assert capsys.readouterr().err == "clab: device: the ndif engine cannot honor 'cuda:0': the server places the model\n"
    assert main(["run", document("patching"), "--batch-size", "0", "--out", out]) == 1
    assert capsys.readouterr().err == "clab: batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)\n"
    assert not os.path.exists(out)                       # and nothing loaded: each refusal came before the model

def test_show_compacts_positions_every_row_shares(engines, tmp_path, capsys):
    """An entry every row resolved alike (patching's read and write moved to `pos: 0`, BOS on every row) is one
    list and the row count; one whose rows differ (its logits at the last token) is each row's."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = 0
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = 0
    plan = edited("patching", change).compile()
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[8].split() == ["entry", "at", "positions"]                 # after the four steps' costs
    assert [line.split(maxsplit=2) for line in shown[9:12]] == [
        ["counterfactual.v_cf", "—", "[0] on all 4 rows"], ["patched.patch", "—", "[0] on all 4 rows"],
        ["patched.logits", "—", "[10], [8], [8], [8]"]]


def test_show_prints_one_row_per_entry_and_distinct_positions_on_a_grid(engines, tmp_path, capsys):
    """On a 3×2 grid (the write's `pos` over `dst`, the read's over `src`) each entry's positions vary along one
    axis alone, so `show` prints one row per entry and distinct positions with how many points share it, 6 rows,
    not one per entry and point, 13."""
    def change(d):
        d["steps"]["patched"]["interventions"]["patch"]["pos"] = {"sweep": [-1, -2, -3], "label": "dst"}
        d["steps"]["patched"]["interventions"]["logits"]["pos"] = {"sweep": [-1, -2], "label": "src"}
    plan = edited("pos_sweep", change).compile()
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.splitlines()
    first = next(i for i, line in enumerate(shown) if line.split() == ["entry", "at", "positions"])
    last = next(i for i, line in enumerate(shown) if line.split() == ["result", "step", "dims", "axes", "shape"])
    assert [line.split(maxsplit=1) for line in shown[first + 1:last]] == [     # in the order the points meet them
        ["counterfactual.v_cf", "—              [8], [8], [10], [8]"],
        ["patched.patch", "2 of 6 points  [10], [8], [8], [8]"],
        ["patched.logits", "3 of 6 points  [10], [8], [8], [8]"],
        ["patched.logits", "3 of 6 points  [9], [7], [7], [7]"],
        ["patched.patch", "2 of 6 points  [9], [7], [7], [7]"],
        ["patched.patch", "2 of 6 points  [8], [6], [6], [6]"]]


def test_show_prints_an_entry_of_more_than_twelve_rows_as_its_first_three_and_last():
    """Up to twelve rows `show` prints each row's positions; past twelve, where rows differ, the first three rows',
    `…`, the last row's and the row count, and a row of more than twelve positions its first three and last, so
    one entry stays within a line or two at any number of rows."""
    assert compact([[i] for i in range(12)]) == ", ".join(f"[{i}]" for i in range(12))
    assert compact([[i] for i in range(50_000)]) == "[0], [1], [2], …, [49999] (50,000 rows; clab table shows each)"
    assert compact([[3]] * 50_000) == "[3] on all 50,000 rows"
    assert compact([list(range(12)), "alignment_missing"]) == "[0,1,2,3,4,5,6,7,8,9,10,11], —"
    assert compact([list(range(512)), list(range(3, 9))]) == "[0,1,2,…,511], [3,4,5,6,7,8]"


def test_show_keeps_apart_points_whose_positions_differ_only_in_rows_it_does_not_print(monkeypatch):
    """Two points of an entry are one row of `show` only where every row's positions agree, not where the cells
    printed for them do: pos_sweep's write at three points, whose 50 rows differ only at row 25, is three rows."""
    plan = plan_of("pos_sweep")
    at = {"pos=-1": 0, "pos=-2": 1, "pos=-3": 2}

    def placed(step, p):
        return [(f"{step}.patch", [[7]] * 25 + [[at[cli.coordinate_text(plan, p.at)]]] + [[7]] * 24)] \
            if step == "patched" else []
    monkeypatch.setattr(cli, "placed", placed)
    rows = cli.resolved(plan)
    assert [r[:2] for r in rows[1:]] == [["patched.patch", "pos=-1"], ["patched.patch", "pos=-2"],
                                         ["patched.patch", "pos=-3"]]
    assert {r[2] for r in rows[1:]} == {"[7], [7], [7], …, [7] (50 rows; clab table shows each)"}
