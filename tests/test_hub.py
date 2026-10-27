"""Datasets on the Hugging Face hub, held by reference: compiled with no rows in the plan, run on both engines bit-equal
to the same rows inline, through `remote="local"` and the stand-in server (which is sent no rows and loads them
itself), dumped, loaded and tabled, and every refusal by name.

The fixture is `nyu-mll/glue` at commit bcdcba79: config `wnli` (sentence1, sentence2, label, idx; 635 train, 71
validation and 146 test rows) and config `ax` (premise, hypothesis, label, idx; 1104 test rows). The suite reads it
from the local Hugging Face cache under HF_HUB_OFFLINE=1, as it does the tiny checkpoints; to fetch it once, with hub
access:

    env -u HF_HUB_OFFLINE uv run python -c "import datasets; [datasets.load_dataset('nyu-mll/glue', c,
        revision='bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c') for c in ('wnli', 'ax')]"
"""

import functools
import json

import pytest
from conftest import BATCH_SIZE, TRIP, same, stand_in

from clab import Plan
from clab.cli import main
from clab.compile import materialized
from clab.compile.preflight import preflight
from clab.format import data
from clab.plan import PlanError
from clab.steps import begin

REV = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
GLUE = {"hf": "nyu-mll/glue", "revision": REV, "config": "wnli", "split": "validation"}
LLAMA = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
         "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}


@functools.cache
def glue(config: str, split: str) -> list[dict]:
    """The fixture's rows as the `datasets` library itself gives them."""
    import datasets

    return datasets.load_dataset("nyu-mll/glue", config, split=split, revision=REV).to_list()


def texts(rows: list[dict]) -> list[bytes]:
    """Every string value of every row, as the bytes a pickle holds it in."""
    return [v.encode() for row in rows for v in row.values() if isinstance(v, str)]


def plan(d: dict | None = None, **more) -> Plan:
    """A forward over dataset `glue` (`d`; by default the first 6 of wnli's validation rows) reading each row's
    `sentence1` at its last token: the residual stream after layer 0, and the logits."""
    read = {"kind": "forward", "input": "glue.sentence1", "interventions": {
        "h": {"site": {"component": "layer_output", "layers": 0}, "pos": -1},
        "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}
    return Plan(**{"model": LLAMA, "data": {"glue": {**GLUE, "take": 6} if d is None else d},
                   "steps": {"read": read}, "saves": ["read.h", "read.logits"], **more})


@pytest.fixture(scope="module")
def ran(engines) -> dict[str, Plan]:
    compiled = plan().compile()
    return {kind: engines(kind, compiled.model).execute(plan().compile(), batch_size=BATCH_SIZE)
            for kind in ("nnterp", "hooks")}


def test_a_hub_dataset_compiles_with_its_digest_and_without_its_rows():
    compiled, rows = plan().compile(), glue("wnli", "validation")[:6]
    assert compiled.data["glue"].rows is None and compiled.data["glue"].n_rows == 6
    assert compiled.model_dump(mode="json", exclude_defaults=True)["data"]["glue"] == {
        **GLUE, "take": 6, "sha256": data.digest(rows), "n_rows": 6}
    assert materialized(compiled, "glue") == rows


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_forward_over_a_hub_dataset_is_the_forward_over_its_rows_inline(engines, ran, kind):
    inline = plan({"rows": glue("wnli", "validation")[:6]}).compile()
    assert inline.data["glue"].sha256 == ran[kind].data["glue"].sha256
    assert same(ran[kind].results, engines(kind, inline.model).execute(inline, batch_size=BATCH_SIZE).results)
    assert ran[kind].data["glue"].rows is None                 # the run loaded them; the plan never held them
    assert same(ran["nnterp"].results, ran["hooks"].results)


def rows_of(**d) -> list[dict]:
    """Dataset `glue` (the fixture's wnli validation rows, with `d`) as a compiled plan's run materializes it."""
    compiled = plan({**GLUE, **d}, steps={}, saves=[]).compile()
    rows = materialized(compiled, "glue")
    assert compiled.data["glue"].sha256 == data.digest(rows) and compiled.data["glue"].n_rows == len(rows)
    return rows


def test_split_config_shuffle_take_and_columns_apply_to_a_hub_dataset():
    validation = glue("wnli", "validation")
    assert rows_of() == validation and rows_of(split="train") == glue("wnli", "train")
    assert rows_of(config="ax", split="test") == rows_of(config="ax", split=None) == glue("ax", "test")
    order = [r["i"] for r in data.shuffled([{"i": i} for i in range(len(validation))])]
    sample = rows_of(shuffle=True, take=5)                      # take after shuffle: a random sample of 5
    assert sample == [validation[i] for i in order[:5]] and sample != validation[:5]
    assert rows_of(columns={"input": "sentence1"}) == data.renamed(validation, {"input": "sentence1"})
    assert rows_of(revision="main") == validation             # offline, a branch reads as the cache holds it
    with pytest.raises(PlanError, match=r"^data\.glue\.template: no step reads a chat column of 'glue', and a "
                                        r"template renders a chat"):
        rows_of(template={"add_generation_prompt": True})


AT = rf"at revision {REV}"
WNLI = rf"nyu-mll/glue \(config 'wnli', split 'validation'\) {AT}"


@pytest.mark.parametrize("d, message", [
    ({**GLUE, "path": "weekdays/data"},
     r"^data\.glue: a dataset's rows come from one of 'path', inline 'rows' and a hub dataset 'hf'; it gives 'path' "
     r"and 'hf'$"),
    ({**GLUE, "rows": [{"sentence1": "a"}]}, r"^data\.glue: a dataset's rows come from one of .*; it gives 'rows' and "
                                             r"'hf'$"),
    ({"path": "weekdays/data", "rows": [{"sentence1": "a"}]},
     r"^data\.glue: a dataset's rows come from one of .*; it gives 'path' and 'rows'$"),
    ({**GLUE, "hf": ""}, r"^data\.glue: hf '': a hub dataset is named '<owner>/<name>', as 'nyu-mll/glue' is$"),
    ({**GLUE, "hf": "glue"}, r"^data\.glue: hf 'glue': a hub dataset is named '<owner>/<name>'"),
    ({**GLUE, "hf": "nyu-mll/glue/wnli"}, r"^data\.glue: hf 'nyu-mll/glue/wnli': a hub dataset is named "),
    ({"hf": "nyu-mll/glue", "config": "wnli", "split": "validation"},
     r"^data\.glue: 'hf' needs 'revision', the commit of the hub dataset to load, as 'model' pins its checkpoint's$"),
    ({**GLUE, "revision": ""}, r"^data\.glue: 'hf' needs 'revision', the commit of the hub dataset to load"),
    ({"path": "weekdays/data", "revision": REV},
     r"^data\.glue: 'revision' pins a hub dataset \('hf'\); a file's rows and inline ones have none$"),
    ({"rows": [{"sentence1": "a"}], "config": "wnli"},
     r"^data\.glue: 'config' is the subset of a hub dataset \('hf'\); a file's rows and inline ones have none$"),
    ({**GLUE, "config": {"sweep": ["wnli", "ax"]}},
     r"^data\.glue\.config: a sweep cannot stand for 'config'; a dataset's rows are compiled once, and a step reads "
     r"them by name: write one dataset per value$"),
    ({**GLUE, "split": "nope"},
     rf"^data\.glue\.hf: nyu-mll/glue \(config 'wnli', split 'nope'\) {AT} has no such split; its splits: train, "
     r"validation, test$"),
    ({**GLUE, "split": None},
     rf"^data\.glue\.hf: nyu-mll/glue \(config 'wnli'\) {AT} has splits train, validation, test; 'split' names the one "
     r"to read$"),
    ({**GLUE, "config": "nope"}, rf"^data\.glue\.hf: nyu-mll/glue \(config 'nope', split 'validation'\) {AT}: "),
    ({**GLUE, "hf": "nobody-here/no-such-dataset", "config": None},
     rf"^data\.glue\.hf: nobody-here/no-such-dataset \(split 'validation'\) {AT}: "),
    ({**GLUE, "revision": "0" * 40},      # offline, the library would read the cached commit whatever the revision
     r"^data\.glue\.hf: nyu-mll/glue \(config 'wnli', split 'validation'\) at revision 0{40} is not in the cache, "
     rf"which holds it at {REV}$"),
    ({**GLUE, "take": 72}, r"^data\.glue\.take: take 72, and the dataset has 71 rows in split 'validation'; take keeps "
                           r"at most as many as there are$"),
])
def test_a_hub_dataset_that_cannot_be_read_is_refused_by_name(d, message):
    with pytest.raises(PlanError, match=message):
        plan(d).compile()


def test_a_compiled_file_dataset_holds_its_rows_beside_its_path():
    """Compile alone sets `sha256`, so the compiled form may hold a file's rows beside its path; a document may not."""
    compiled = plan({"path": "weekdays/data"}, steps={}, saves=[]).compile()
    again = Plan(**compiled.model_dump(exclude={"run"}))
    assert again.data["glue"].path == "weekdays/data" and again.data["glue"].rows == compiled.data["glue"].rows


def test_a_column_that_is_not_json_is_refused_naming_it(monkeypatch):
    import datasets

    held = datasets.DatasetDict({"validation": datasets.Dataset.from_dict({"sentence1": ["a", "b"],
                                                                           "image": [b"\x89PNG", b"\x89PNG"]})})
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: held)
    with pytest.raises(PlanError, match=rf"^data\.glue\.hf: {WNLI}: column 'image' holds values that are not JSON "
                                        r"\(images, audio, bytes\); a row is JSON"):
        plan().compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_rows_the_revision_no_longer_gives_are_refused_before_any_forward(engines, kind, monkeypatch):
    compiled = plan().compile()
    engine = engines(kind, compiled.model)
    monkeypatch.setattr(data, "hub", lambda d: glue("wnli", "validation")[::-1])
    monkeypatch.setattr(type(engine), "forward", lambda *args: pytest.fail("a forward ran"))
    with pytest.raises(PlanError, match=rf"^data\.glue: {WNLI} gives other rows here than compile read \(their "
                                        r"digest is not the one it recorded\)$"):
        engine.execute(compiled, batch_size=BATCH_SIZE)


def test_a_run_read_of_rows_never_loaded_is_refused_by_name(engines):
    """The run reads the plan `steps.held` loaded the rows into; a caller that skips it reads none, by name."""
    compiled = plan().compile()
    engine = engines("hooks", compiled.model)
    begin(engine, compiled, BATCH_SIZE)
    with pytest.raises(PlanError, match=r"^data\.glue: the plan holds none of its rows; `compile\.materialized` loads "
                                        r"them, as the run does first \(`steps\.held`\)$"):
        preflight(engine, compiled)


def test_remote_local_is_bit_equal_to_local(engines, ran):
    compiled = plan().compile()
    there = engines("ndif", compiled.model, remote="local").execute(compiled, batch_size=BATCH_SIZE)
    assert same(there.results, ran["nnterp"].results) and there.data["glue"].rows is None


def test_the_server_is_sent_no_rows_and_loads_them_itself(engines, ran, monkeypatch):
    """Through the stand-in server: the request holds no string of any row of the hub dataset's split (it holds
    every one of the rows inline, the control) and its plan's dataset no rows, the rows are loaded once, from the
    deserialized plan's dataset (the server's copy, not the client's), and the results that come home are the local
    run's."""
    from nnsight.schema.request import RequestModel

    sent, loaded = [], []
    serialize, hub = RequestModel.serialize, data.hub

    def kept(cls, tracer, compress=False):
        sent.append(serialize(tracer, compress))
        return sent[-1]

    def fetched(d):
        loaded.append(d)
        return hub(d)

    monkeypatch.setattr(RequestModel, "serialize", classmethod(kept))
    compiled, inline = plan().compile(), plan({"rows": glue("wnli", "validation")[:6]}).compile()
    monkeypatch.setattr(data, "hub", fetched)
    home, trip = stand_in(engines, compiled, monkeypatch)
    assert trip == TRIP and same(home.results, ran["nnterp"].results) and home.data["glue"].rows is None
    assert len(loaded) == 1 and loaded[0] is not compiled.data["glue"] and loaded[0] == compiled.data["glue"]
    stand_in(engines, inline, monkeypatch)
    assert len(sent) == 2 and not [t for t in texts(glue("wnli", "validation")) if t in sent[0]]
    assert all(t in sent[1] for t in texts(inline.data["glue"].rows or []))
    server = engines("nnterp", compiled.model).model
    shipped = RequestModel.deserialize(sent[0], server._remoteable_persistent_objects()).info.frame.f_locals["plan"]
    assert shipped.data["glue"].rows is None and "rows" not in shipped.model_dump(exclude_defaults=True)["data"]["glue"]


def test_the_output_holds_no_rows_and_a_loaded_plan_tables_them_again(ran, tmp_path, capsys, monkeypatch):
    rows = glue("wnli", "validation")[:6]
    ran["hooks"].dump(str(tmp_path))
    text = (tmp_path / "manifest.json").read_bytes()
    assert json.loads(text)["data"]["glue"] == {**GLUE, "take": 6, "sha256": data.digest(rows), "n_rows": 6}
    assert not [t for t in texts(rows) if t in text]
    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out.encode()
    assert not [t for t in texts(rows) if t in shown]
    back = Plan.load(str(tmp_path))
    assert back.data["glue"].rows is None
    assert [r["sentence1"] for r in back.table("read.h")] == [r["sentence1"] for r in rows]
    assert main(["table", str(tmp_path), "read.h", "--csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "row,pos,pos_token,sentence1,value" and len(lines) == 7
    monkeypatch.setattr(data, "hub", lambda d: glue("wnli", "validation")[::-1])
    with pytest.raises(PlanError, match=rf"^data\.glue: {WNLI} gives other rows here than compile read"):
        back.table("read.h")
