"""What the trials found (`FINDINGS.md`), one test per finding on the tiny models: each bug fixed or refused by
name, each silent behaviour a refusal or a check, and each refusal that did not say what to do saying it. A test
is named after its finding's tag."""

import json
import os
import shutil

import numpy
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from clab import Plan
from clab.plan import PlanError
from clab.plan.model import dims_of, points
from clab.runtime import data, ops, positions
from clab.runtime.ops import metrics, reductions, transforms
from conftest import BATCH_SIZE, DOCUMENTS, ROOT, bundle_header, edited, same

TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}
LAYER0 = {"component": "layer_output", "layers": 0}
HEAD = {"component": "logits"}
DAYS = {"rows": [{"input": "If today is Monday, tomorrow is"}, {"input": "If today is Friday, tomorrow is"},
                 {"input": "one two three"}, {"input": "Hello there my friend"}]}


def plan(root: str = ROOT, **fields) -> Plan:
    """A document on the tiny Llama, read as if from `documents/finding.json` under `root`."""
    return Plan(**{"model": TINY, **fields}, document=os.path.join(root, "documents", "finding.json"))


def corpus(tmp_path) -> str:
    """A copy of the corpus to write bundles into."""
    shutil.copytree(DOCUMENTS, tmp_path / "documents")
    os.symlink(os.path.join(ROOT, "data"), tmp_path / "data")
    return str(tmp_path)


def run(engines, compiled: Plan, kind: str = "nnterp", **options) -> Plan:
    return engines(kind, compiled.model, **options).execute(compiled, batch_size=BATCH_SIZE)


def refused_before_any_forward(engines, compiled: Plan, message: str) -> None:
    engine = engines("nnterp", compiled.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises(ValueError, match=message):
            engine.execute(compiled, batch_size=BATCH_SIZE)
    finally:
        del engine.forward
    assert not calls


def forward(interventions: dict, **more) -> dict:
    return {"kind": "forward", "input": "d.input", "interventions": interventions, **more}


# -- bugs ---------------------------------------------------------------------------------------------------


def test_b1_readout_of_a_narrowed_hidden_size_is_refused_at_compile():
    """The final norm takes the whole residual stream: a readout of a site that keeps some of `hidden_size`, in
    its entry or as a step over its read, is refused by name before the model loads."""
    half = {**LAYER0, "select": {"hidden_size": "0:8"}}
    with pytest.raises(PlanError, match=r"f\.interventions\.lens\.ops\[0\]: readout projects the whole residual "
                                        r"stream through the model's final norm and head, and layer_output select "
                                        r"hidden_size '0:8' keeps only some of hidden_size"):
        plan(data={"d": DAYS}, steps={"f": forward({"lens": {"site": half, "pos": -1, "ops": [{"op": "readout"}]}})}
             ).compile()
    with pytest.raises(PlanError, match=r"steps\.lens: readout projects the whole residual stream .* and 'f\.h' "
                                        r"keeps only some of hidden_size, which the norm cannot take; read the site "
                                        r"whole, readout, and narrow"):
        plan(data={"d": DAYS}, steps={"f": forward({"h": {"site": half, "pos": -1}}),
                                      "lens": {"kind": "op", "op": "readout", "of": "f.h"}}).compile()


def test_b2_kl_against_a_number_is_refused():
    with pytest.raises(PlanError, match=r"steps\.h: against 0 is a number, and a number is not a distribution; "
                                        r"'against' names a value with the same vocabulary axis"):
        plan(data={"d": DAYS}, steps={"f": forward({"logits": {"site": HEAD, "pos": -1}}),
                                      "h": {"kind": "op", "op": "kl", "of": "f.logits", "against": 0}})
    with pytest.raises(PlanError, match=r"against 1\.5 is a number"):
        plan(data={"d": DAYS}, steps={"f": forward({"logits": {"site": HEAD, "pos": -1}}),
                                      "h": {"kind": "op", "op": "js", "of": "f.logits", "against": 1.5}})


def test_b3_a_file_that_is_not_safetensors_is_refused_by_name_before_the_model_loads(tmp_path):
    root = corpus(tmp_path)
    numpy.savez(os.path.join(root, "documents", "artifacts", "params.npz"), W_enc=numpy.zeros((16, 32)))
    with pytest.raises(PlanError, match=r"featurizers\.dictionary\.file_path: documents/artifacts/params\.npz is not "
                                        r"a safetensors file \(.*\); a featurizer's file_path is read as safetensors.*"
                                        r"Gemma Scope's params\.npz needs converting first: load it with numpy\.load "
                                        r"and write its arrays with safetensors\.torch\.save_file"):
        plan(root, data={"d": DAYS}, featurizers={"dictionary": {
            "kind": "sae", "file_path": "documents/artifacts/params.npz", "activation": "jumprelu",
            "centered": False, "layer": 0, "latents": 32}},
            steps={"f": forward({"z": {"site": LAYER0, "pos": -1, "ops": [{"op": "encode",
                                                                         "featurizer": "dictionary"}]}})}).compile()


def test_b4_argmax_top_k_and_match_break_an_exact_tie_the_same_way():
    """Ids 1 and 3 tie exactly: every op picks id 1, the lower. In bfloat16 two logits one step apart round to
    the same probability; `top_k` ranks by the logits, so its first is still `argmax`'s."""
    tied = torch.tensor([[[0.0, 2.0, -1.0, 2.0, 1.0]]])
    assert transforms.argmax(tied).item() == 1
    assert transforms.top_k(tied, 2)["ids"].tolist() == [[[1, 3]]]
    assert metrics.match(tied, torch.tensor([[1]]), ["rows", "pos", "vocab_size"]).item() == 1.0
    assert metrics.match(tied, torch.tensor([[3]]), ["rows", "pos", "vocab_size"]).item() == 0.0
    close = torch.zeros(1, 1, 10, dtype=torch.bfloat16)
    close[0, 0, 5], close[0, 0, 2] = 0.25, 0.251953125          # one bfloat16 step apart
    probs = close.softmax(-1)
    assert probs[0, 0, 5] == probs[0, 0, 2]                     # the probabilities tie; the logits do not
    assert probs.topk(1).indices.item() == 5                    # which a top-k of the probabilities ranks first
    assert transforms.argmax(close).item() == 2 == transforms.top_k(close, 1)["ids"].item()
    for name in ("argmax", "top_k"):
        assert "an exact tie goes to the lower id" in " ".join((ops.OPS[name].__doc__ or "").split()), name


def test_b5_a_sweep_of_objects_is_refused_at_compile():
    """trial-06 e10: the points of a sweep are keyed by its values, so a value is a scalar or a list of them."""
    with pytest.raises(PlanError, match=r"a sweep's values are numbers, strings, booleans or lists of them, each "
                                        r"point keyed by its value; \{'index': '-2:'\} is an object: sweep a scalar "
                                        r"field inside it, \{'index': \{'sweep': \['-2:', \.\.\.\]\}\}"):
        plan(data={"d": DAYS}, steps={
            "f": forward({"w": {"site": LAYER0, "pos": {"sweep": [{"index": "-2:"}, {"index": ":"}]},
                                "ops": [{"op": "scale", "factor": 2.0}]}, "logits": {"site": HEAD, "pos": -1}}),
            "top": {"kind": "op", "op": "argmax", "of": "f.logits"}}, saves=["top"])
    with pytest.raises(PlanError, match=r"\[\[0, 1\], \[2\]\] holds a list or an object"):
        plan(data={"d": DAYS}, sites={"s": {**LAYER0, "select": {"hidden_size": {"sweep": [[[0, 1], [2]]]}}}},
             steps={"f": forward({"h": {"site": "s", "pos": -1}})})


def generate_at(frame, eos: int | None = 4598) -> Plan:
    """Generation from DAYS, reading the head at call `frame`; the tiny Llama says `Handler` (4598) first on
    the two weekday prompts, so with that as EOS the first batch stops in the prefill."""
    return plan(data={"d": DAYS}, steps={
        "g": {"kind": "generate", "input": "d.input", "max_new_tokens": 5, "do_sample": False,
              **({"eos_token_id": eos} if eos is not None else {}),
              "interventions": {"at": {"site": HEAD, "pos": {"frame": frame, "index": -1}}}},
        "top": {"kind": "op", "op": "argmax", "of": "g.at"}}, saves=["g", "g.at", "top"]).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_b6_a_call_a_whole_batch_never_reaches_is_missing(engines, kind):
    """trial-06 e09g: both rows of the first batch stop at once, so it makes no call 3; its rows are
    `alignment_missing`, NaN in the read and -1 from `argmax`, and the second batch's are read."""
    filled = run(engines, generate_at(3), kind)
    assert filled.run.positions["g.at"] == ["alignment_missing", "alignment_missing", [6], [7]]
    assert filled.results["g.at"][:2].isnan().all() and not filled.results["g.at"][2:].isnan().any()
    assert filled.results["top"][:, 0].tolist()[:2] == [-1, -1]


def test_b6_a_batch_with_no_call_makes_its_blank_where_the_model_is(engines, monkeypatch):
    """The blank of a batch that made no call the read names is made where the model is (`home`), so it joins
    the other batches' values on their device; a CPU blank beside CUDA values was the crash."""
    from clab.runtime.steps import generate

    compiled = generate_at(3)
    engine = engines("nnterp", compiled.model)
    step = compiled.steps["g"]
    record = step.interventions["at"]
    monkeypatch.setattr(generate, "home", lambda engine: "meta")
    value = generate.at_calls(engine, compiled, record, {}, None, [], ["alignment_missing"] * 2,
                              [{"ids": [0] * 9}] * 2, [0, 0], slice(0, 2), {}, {}, False, 5, {})
    assert value.device.type == "meta" and value.shape == (2, 1, 32000)


def test_b7_a_gate_inside_a_pca_encode_trains_on_the_pca_coordinates(engines, tmp_path):
    """trial-08 x07: brackets nest, so a gate encoded inside a pca's coordinates is as wide as they are and
    trains there; the fit runs on both engines alike."""
    root = corpus(tmp_path)
    basis = torch.linalg.qr(torch.randn(16, 4, generator=torch.Generator().manual_seed(0)))[0].contiguous()
    save_file({"basis": basis, "mean": torch.zeros(16)}, os.path.join(root, "documents", "artifacts", "pc.safetensors"),
              metadata=bundle_header(kind="pca", k="4", document="pca_harvest"))

    def change(d):
        d["featurizers"]["pc"] = {"kind": "pca", "k": 4, "file_path": "documents/artifacts/pc.safetensors"}
        d["interventions"]["cf_read"]["v_cf"]["ops"] = [{"op": "encode", "featurizer": "pc"},
                                                        {"op": "encode", "featurizer": "fit.mask"}]
        d["interventions"]["dbm"]["patch"]["ops"] = [
            {"op": "encode", "featurizer": "pc"}, {"op": "encode", "featurizer": "fit.mask"},
            {"op": "swap", "with": "counterfactual.v_cf"}, {"op": "decode", "featurizer": "fit.mask"},
            {"op": "decode", "featurizer": "pc"}]
        d["steps"]["fit"].update(epochs=2, objective={"ce": 1.0, "mask.mask": 0.01})
    one = edited("dbm", change)
    compiled = Plan(**one.model_dump(exclude_defaults=True, exclude={"document"}),
                    document=os.path.join(root, "documents", "dbm.json")).compile()
    got = [run(engines, compiled, kind).results for kind in ("nnterp", "hooks")]
    assert same(*got)
    assert got[0]["fit.mask"]["theta"].shape == (4,)


def test_b8_zero_epochs_scores_the_untrained_draw_and_saves_an_empty_history(engines, tmp_path):
    """trial-08 x09, trial-12 x03: an untrained lora is the identity, so its eval is the clean model's to the
    bit, a free null control; the history has no epochs. A negative count is refused."""
    def change(d):
        d["steps"]["fit"]["epochs"] = 0
        d["steps"]["clean"] = {"kind": "forward", "input": "train.input",
                               "interventions": {"logits": {"site": "head", "pos": -1}}}
        d["steps"]["ce"] = {"kind": "op", "op": "cross_entropy", "of": "clean.logits", "target": "train.cf_answer"}
        d["saves"] += ["ce"]
    compiled = edited("lora_train", change).compile()
    filled = run(engines, compiled)
    assert filled.results["fit.history.ce"].shape == (0,)
    assert torch.equal(filled.results["fit.train.ce"], filled.results["ce"])
    assert not filled.results["fit.lora"]["B"].any()
    filled.dump(str(tmp_path / "out"))
    assert Plan.load(str(tmp_path / "out")).results["fit.history.ce"].shape == (0,)
    with pytest.raises(PlanError, match=r"epochs -1: a fit runs 0 or more epochs \(0 scores the untrained draw\)"):
        edited("lora_train", lambda d: d["steps"]["fit"].update(epochs=-1))


def test_b9_a_swept_fits_bundle_file_holds_a_header_per_point(engines, tmp_path):
    """trial-08 §4.4: the file holds one bundle per point, and each point's header beside its tensors, after its
    coordinate, so the file says each point's own rank."""
    def change(d):
        d["featurizers"]["lora"]["rank"] = {"sweep": [2, 4]}
        d["steps"]["fit"]["epochs"] = 1
        d["saves"] = ["fit.lora"]
    filled = run(engines, edited("lora_train", change).compile())
    filled.dump(str(tmp_path))
    with safe_open(str(tmp_path / "fit.lora.safetensors"), "pt") as f:
        header, keys = f.metadata(), set(f.keys())
    assert keys == {"rank=2/A", "rank=2/B", "rank=4/A", "rank=4/B"}
    assert header["format"] == "clab/1"
    assert (header["rank=2/rank"], header["rank=4/rank"]) == ("2", "4")
    assert header["rank=2/alpha"] == header["rank=4/alpha"] == "8"
    assert header["rank=4/component"] == "layer_output" and header["rank=4/width"] == "16"


def probe_fit(change=None) -> dict:
    """A linear probe of layer 1 at the last token over weekdays/data's training rows (a file, so a manifest
    holds none of its rows), classes the counterfactual answers."""
    doc = {"data": {"train": {"path": "weekdays/data", "split": "train"}},
           "featurizers": {"probe": {"kind": "linear", "classes": "train.cf_answer"}},
           "steps": {"fit": {
               "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 2, "batch_size": 2, "seed": 0,
               "optimizer": {"name": "adam", "lr": 0.5}, "eval": {"train": {}},
               "steps": {"read": {"kind": "forward", "input": "train.input", "interventions": {"scores": {
                   "site": {"component": "layer_output", "layers": 1}, "pos": -1,
                   "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
                   "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "train.cf_answer"}}}},
           "saves": ["fit.probe"]}
    if change:
        change(doc)
    return doc


def test_b10_a_run_that_loads_a_probe_reads_back_and_tables(engines, tmp_path):
    """trial-09 §4.4: a run whose document loads a `linear` bundle by `file_path` is read back by `Plan.load`,
    `clab show` and `clab table`, though its manifest holds no rows to take the classes from."""
    from clab.cli import main

    root = corpus(tmp_path)
    trained = run(engines, plan(root, **probe_fit()).compile())
    trained.dump(os.path.join(root, "documents", "artifacts"))

    def loading(doc):
        doc["featurizers"]["probe"]["file_path"] = "documents/artifacts/fit.probe.safetensors"
        doc["steps"] = {"read": {"kind": "forward", "input": "train.input", "interventions": {"scores": {
            "site": {"component": "layer_output", "layers": 1}, "pos": -1,
            "ops": [{"op": "encode", "featurizer": "probe"}]}}},
            "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "train.cf_answer"}}
        doc["saves"] = ["hit"]
    filled = run(engines, plan(root, **probe_fit(loading)).compile())
    out = os.path.join(root, "out")
    filled.dump(out)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        loaded = Plan.load(out)
        assert loaded.table("hit") == filled.table("hit")
        assert main(["show", out]) == 0 and main(["table", out, "hit"]) == 0
    finally:
        os.chdir(cwd)


def test_b11_a_class_column_of_mixed_types_is_refused_by_name():
    def change(doc):
        doc["data"] = {"train": {"rows": [{"input": "If today is Monday", "cf_answer": " Tuesday"},
                                          {"input": "If today is Friday", "cf_answer": 1}]}}
    with pytest.raises(PlanError, match=r"featurizers\.probe\.classes: classes column 'train\.cf_answer' holds str "
                                        r"\(row 0 is ' Tuesday'\), int \(row 1 is 1\); a probe's classes are one "
                                        r"type, all strings, all ints, all floats or all booleans"):
        plan(**probe_fit(change)).compile()


def test_b12_a_message_that_says_its_role_header_is_found_in_its_own_text(engines):
    """trial-11 e34 on the tiny Llama's `[INST]` template: a user message `INST`, and an assistant one after a
    `[/INST]`, are their own tokens, not the header's."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY["key"], revision=TINY["revision"])
    for chat, role in (([{"role": "user", "content": "INST"}], "user"),
                       ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "INST"},
                         {"role": "user", "content": "ok"}], "assistant")):
        encoded = data.encode(tokenizer, chat)
        (_, a, b), = [t for t in encoded["turns"] if t[0] == role]
        assert encoded["text"][a:b] == "INST" and encoded["text"][a - 1] == " " and encoded["text"][b] == " "
    compiled = plan(data={"c": {"rows": [{"chat": [{"role": "user", "content": "INST"}]}]}}, steps={
        "f": {"kind": "forward", "input": "c.chat", "interventions": {"u": {
            "site": LAYER0, "pos": {"scope": {"turn": "user"}}}}}}, saves=["f.u"]).compile()
    filled = run(engines, compiled)
    tokens = filled.run.tokens["c.chat"][0]
    assert [tokens[p] for p in filled.run.positions["f.u"][0]] == ["▁IN", "ST"]


def test_b13_top_k_over_a_ragged_read_keeps_its_entries(engines, tmp_path):
    """trial-11 e33: the value is the entries' ids, probabilities and tokens beside their rows and positions,
    and it saves, reads back and tables."""
    compiled = plan(data={"d": {"rows": [{"input": "If today is Monday, tomorrow is"}, {"input": "Hello"}]}}, steps={
        "f": forward({"l": {"site": HEAD, "pos": {"index": ":"}}}),
        "k": {"kind": "op", "op": "top_k", "of": "f.l", "k": 3}}, saves=["k", "f.l"]).compile()
    filled = run(engines, compiled)
    k, read = filled.results["k"], filled.results["f.l"]
    assert set(k) == {"ids", "probs", "tokens", "row", "tok"}
    assert torch.equal(k["row"], read["row"]) and torch.equal(k["tok"], read["tok"])
    assert torch.equal(k["ids"][:, 0], read["values"].argmax(-1)) and len(k["tokens"]) == len(k["row"])
    rows = filled.table("k")
    assert len(rows) == len(k["row"]) and set(rows[0]["value"]) == {"ids", "probs", "tokens"}
    filled.dump(str(tmp_path))
    assert Plan.load(str(tmp_path)).table("k") == rows
