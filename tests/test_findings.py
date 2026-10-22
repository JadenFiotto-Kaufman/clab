"""What the trials found (`FINDINGS.md`), one test per finding on the tiny models: each bug fixed or refused by
name, each silent behaviour a refusal or a check, and each refusal that did not say what to do saying it. A test
is named after its finding's tag."""

import os
import json
import shutil
from typing import Any

import numpy
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from clab import Plan
from clab.plan import PlanError
from clab.plan.model import dims_of
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


def run(engines, compiled: Plan, kind: str = "nnterp", **options) -> Any:
    """`compiled` executed on `kind` at the pinned batch size: the filled plan."""
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


def test_b2_kl_against_anything_but_a_name_is_refused():
    """A number, or a flag, is not a distribution."""
    for op, against in (("kl", 0), ("js", 1.5), ("kl", True)):
        with pytest.raises(PlanError, match=rf"steps\.h: against {against!r} is not a name, and only a named value is "
                                            r"a distribution"):
            plan(data={"d": DAYS}, steps={"f": forward({"logits": {"site": HEAD, "pos": -1}}),
                                          "h": {"kind": "op", "op": op, "of": "f.logits", "against": against}})


def test_b3_a_file_that_is_not_safetensors_is_refused_by_name_before_the_model_loads(tmp_path):
    root = corpus(tmp_path)
    numpy.savez(os.path.join(root, "documents", "artifacts", "params.npz"), W_enc=numpy.zeros((16, 32)))
    with pytest.raises(PlanError, match=r"featurizers\.dictionary\.file_path: documents/artifacts/params\.npz is not "
                                        r"a safetensors file, which a file_path is read as; convert it first \(Gemma "
                                        r"Scope's params\.npz: numpy\.load, then safetensors\.torch\.save_file\)$"):
        plan(root, data={"d": DAYS}, featurizers={"dictionary": {
            "kind": "sae", "file_path": "documents/artifacts/params.npz", "activation": "jumprelu",
            "centered": False, "site": "target"}}, sites={"target": LAYER0},
            steps={"f": forward({"z": {"site": LAYER0, "pos": -1, "ops": [{"op": "encode",
                                                                         "featurizer": "dictionary"}]}})}).compile()


def test_b4_argmax_top_k_and_match_break_an_exact_tie_the_same_way():
    """Ids 1 and 3 tie exactly: every op picks id 1, the lower. In bfloat16 two logits one step apart round to
    the same probability; `top_k` ranks by the logits, so its first is still `argmax`'s. A tie at the k-th place
    with an id the top-k left out keeps the lower ids, and ties inside the top-k are in id order."""
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
    many = torch.tensor([[9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [3.0, 3.0, 3.0, 5.0, 0.0, 0.0, 0.0]])
    assert metrics.ranked(many, 3).tolist() == [[0, 1, 2], [3, 0, 1]]
    assert metrics.ranked(many, 1).tolist() == [[0], [3]]
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


def test_b6_batches_join_on_the_first_ones_device():
    """A batch that made no call a read names makes its blank where no value says otherwise; batches join on
    the first one's device, as a `layers` read's layers do, so a CPU blank beside CUDA values (the crash) joins.
    Two devices without a GPU: a meta batch first, then a CPU one."""
    from clab.runtime.steps.batches import joined

    compiled = generate_at(3)
    step: Any = compiled.steps["g"]
    state: dict = {}
    joined(state, "g", step, [{"at": torch.zeros(2, 1, 4, device="meta")}, {"at": torch.zeros(2, 1, 4)}])
    value = state["g.at"][()]
    assert value.device.type == "meta" and value.shape == (4, 1, 4)


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
    `[/INST]`, are their own tokens, not the header's, and so is one after a message that holds the marker the
    search places."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY["key"], revision=TINY["revision"])
    for chat, role in (([{"role": "user", "content": "INST"}], "user"),
                       ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "INST"},
                         {"role": "user", "content": "ok"}], "assistant")):
        encoded = data.encode(tokenizer, chat)
        (_, a, b), = [t for t in encoded["turns"] if t[0] == role]
        assert encoded["text"][a:b] == "INST" and encoded["text"][a - 1] == " " and encoded["text"][b] == " "
    marked = [{"role": "user", "content": data.MARKER}, {"role": "assistant", "content": "ok"},
              {"role": "user", "content": "INST"}]          # an earlier message holds the marker itself
    encoded = data.encode(tokenizer, marked)
    (_, a, b) = encoded["turns"][2]
    assert encoded["text"][a:b] == "INST" and encoded["text"][a - 7:a] == "[INST] "
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


# -- silent behaviours, now refusals or checks ----------------------------------------------------------------

SARA = {"rows": [{"input": "Then, Sara and Sarah went. Sara gave", "s": " Sara"}]}


def test_s3_an_occurrence_lies_on_token_and_word_boundaries(engines):
    """trial-02 07: ` Sara` is not found inside ` Sarah`, so its occurrence 1 is the second ` Sara` (▁S ara)
    and `":"` takes two occurrences, not three; ` Ann` is not in ` Annabel` though GPT-2 splits it `ĠAnn abel`;
    `Paris. ` ends where the next ` Paris` starts; `:` inside `():` is no occurrence of `:`. Online, a write sees
    no further than its token, and takes an end there as a word's."""
    from transformers import AutoTokenizer

    def at(occurrence):
        return {"site": LAYER0, "pos": {"scope": {"column": "d.s", "occurrence": occurrence}}}
    compiled = plan(data={"d": SARA}, steps={"f": forward({"first": at(0), "second": at(1), "every": at(":")})},
                    saves=["f.first", "f.second", "f.every"]).compile()
    filled = run(engines, compiled)
    tokens = filled.run.tokens["d.input"][0]
    assert tokens == ["<s>", "▁Then", ",", "▁S", "ara", "▁and", "▁Sarah", "▁went", ".", "▁S", "ara", "▁gave"]
    assert filled.run.positions["f.first"] == [[3, 4]] and filled.run.positions["f.second"] == [[9, 10]]
    assert filled.run.positions["f.every"] == [[3, 4, 9, 10]]
    gpt2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2",
                                         revision=plan_gpt2_revision())
    for tokenizer in (gpt2, engines("nnterp", compiled.model).tokenizer):
        for text, find, count in (("Then Ann and Annabel met. Ann left", " Ann", 2),
                                  ("I love Paris. Paris is big", "Paris", 2)):
            row = data.encode(tokenizer, text)
            assert len(positions.located([(find, False, False)], row["text"], row["offsets"])) == count
        row = data.encode(tokenizer, "I love Paris. Paris is big")
        (a, b), = positions.located([("Paris. ", False, False)], row["text"], row["offsets"])
        covered = [t for t, (x, y) in enumerate(row["offsets"]) if x < b and y > a]
        assert "Paris" not in tokenizer.decode([row["ids"][covered[-1]]])
    words, letters = "f(): a: b", [(0, 1), (1, 4), (4, 6), (6, 7), (7, 9)]       # `():` is one token
    assert positions.located([(":", False, False)], words, letters) == [(6, 7)]
    assert positions.located([("a", False, False)], words, letters) == [(5, 6)]     # ` a` without its space
    assert positions.located([("Ann", False, False)], "Annabel", [(0, 3), (3, 7)]) == []
    assert positions.located([("Ann", False, False)], "Annabel", [(0, 3), (3, 7)], ahead=False) == [(0, 3)]
    from clab.plan.model import Pos, Scope
    assert "starts where a token starts and ends where a token" in " ".join((Pos.__doc__ or "").split())
    assert "lies on token and word boundaries" in " ".join((Scope.__doc__ or "").split())


def plan_gpt2_revision() -> str:
    with open(os.path.join(DOCUMENTS, "gpt2_reach.json")) as f:
        return json.load(f)["model"]["revision"]


PAIR = {"rows": [{"input": "Then, John and Mary went to the store. John gave a ring to",
                  "abc": "Then, Quentavious and Kate went to the store. Lucy gave a ring to"}]}


def window(index, read=None) -> Plan:
    return plan(data={"d": PAIR}, steps={
        "abc": {"kind": "forward", "input": "d.abc", "interventions": {"v": {"site": LAYER0, "pos": {
            "index": index if read is None else read}}}},
        "patched": forward({"w": {"site": LAYER0, "pos": {"index": index}, "ops": [{"op": "swap", "with": "abc.v"}]},
                            "logits": {"site": HEAD, "pos": -1}})}, saves=["patched.logits"]).compile()


def test_s5_positions_past_where_two_prompts_differ_in_length_are_refused(engines):
    """trial-02: `1:15`, and the one position 4, pair the two prompts' tokens by index from the start, which
    after the names (` John` one token, ` Quentavious` several) are different tokens: refused by name before any
    forward. Positions within the tokens both share at the same distance from their side run (`-4:-1`, ` gave
    a ring`; `1:4`, up to the names' first tokens), and so do positions written differently on the two sides,
    which pair what they say. (`window_patch` and `multi_position_patch` stay bit-equal: test_documents.)"""
    for index in ("1:15", 4):
        refused_before_any_forward(engines, window(index), r"patched\.w: operand 'abc\.v' was read on d\.abc, row 0 "
                                   r"of which is \d+ tokens; d\.input has \d+, so position \d+ pairs '.*' with '.*'; "
                                   r"use a scope to name the token in each row$")
    filled = run(engines, window("-4:-1"))
    assert filled.run.positions["patched.w"] != filled.run.positions["abc.v"]      # the same tokens, from the end
    run(engines, window("1:4"))
    run(engines, window(4, read=-1))


def test_s8_a_metric_refuses_an_operand_of_another_pos_width():
    """trial-05 §4.5: `kl` of a one-position read against a two-position one is refused, as a layers mismatch is."""
    for op in ("kl", "js"):
        with pytest.raises(PlanError, match=r"steps\.k\.against: 'f\.b' holds 2 position\(s\) per row and the value "
                                            rf"'{op}' scores holds 1; position i scores against position i"):
            plan(data={"d": DAYS}, steps={"f": forward({"a": {"site": HEAD, "pos": -1},
                                                        "b": {"site": HEAD, "pos": {"index": "-2:"}}}),
                                          "k": {"kind": "op", "op": op, "of": "f.a", "against": "f.b"}}).compile()


def heads(read, written) -> Plan:
    """A mean of the premix at heads `read`, swapped into a write at heads `written`."""
    site = {"component": "attention_premix", "layers": 1}
    return plan(data={"d": DAYS}, steps={
        "harvest": forward({"a": {"site": {**site, "select": {"num_heads": read}}, "pos": -1}}),
        "m": {"kind": "op", "op": "mean", "of": "harvest.a"},
        "patched": forward({"w": {"site": {**site, "select": {"num_heads": written}}, "pos": -1,
                                  "ops": [{"op": "swap", "with": "m"}]}, "back": {"site": site, "pos": -1}})},
        saves=["m", "patched.back"]).compile()


def test_s10_an_operands_heads_in_another_order_are_refused(engines):
    """trial-07 E3: a mean read at heads `[1, 3]` swapped into a write at `[3, 1]` would trade them, and so would
    an op step or a metric over two reads in different orders; read in one order it lands head for head."""
    refused_before_any_forward(engines, heads([1, 3], [3, 1]), r"patched\.w: operand 'm' holds num_heads \[1, 3\], "
                               r"and this write's selection holds \[3, 1\]; a select list is positional, so they "
                               r"would trade places: select num_heads \[3, 1\] at the read$")
    filled = run(engines, heads([1, 3], [1, 3]))
    assert torch.equal(filled.results["patched.back"][:, :, [1, 3]], filled.results["m"].expand(4, 1, 2, 4))
    for site, axis, op in (({"component": "attention_premix", "layers": 1}, "num_heads", "sub"),
                           (LAYER0, "hidden_size", "cosine")):
        two = forward({"a": {"site": {**site, "select": {axis: [1, 3]}}, "pos": -1},
                       "b": {"site": {**site, "select": {axis: [3, 1]}}, "pos": -1}})
        refused_before_any_forward(engines, plan(data={"d": DAYS}, steps={
            "f": two, "d2": {"kind": "op", "op": op, "of": "f.a", "with": "f.b"}}).compile(),
            rf"d2: operand 'f\.b' holds {axis} \[3, 1\], and the value it acts on holds \[1, 3\]; a select list "
            rf"is positional, so they would trade places: select one {axis} list at both reads$")


def test_s11_one_heads_operand_into_several_heads_is_refused(engines):
    """trial-07 E5: an operand broadcasts over rows and positions only, never over heads."""
    refused_before_any_forward(engines, heads([0], [0, 1, 3]), r"patched\.w: operand 'm' holds num_heads \[0\], and "
                               r"this write's selection holds \[0, 1, 3\]; an operand broadcasts over rows and "
                               r"positions only: select num_heads \[0, 1, 3\] at the read$")
    refused_before_any_forward(engines, heads(":", [2]), r"operand 'm' holds num_heads \[0, 1, 2, 3\], and this "
                               r"write's selection holds \[2\]; an operand holds the same ones: select num_heads "
                               r"\[2\] at the read$")


def test_s12_a_mean_without_over_keeps_a_layers_axis(engines):
    """trial-07 E6: `mean` alone removes rows and positions; a `layers` axis stays unless `over` names it, and
    a value that keeps it is refused as a one-layer write's operand, naming the axis."""
    stacked = {"component": "layer_output", "layers": [0, 1]}
    compiled = plan(data={"d": DAYS}, steps={
        "f": forward({"h": {"site": stacked, "pos": -1}}), "m": {"kind": "op", "op": "mean", "of": "f.h"},
        "all": {"kind": "op", "op": "mean", "of": "f.h", "over": ["layers", "rows", "pos"]}},
        saves=["f.h", "m", "all"]).compile()
    assert dims_of(compiled, "m") == ["layers", "hidden_size"] and dims_of(compiled, "all") == ["hidden_size"]
    assert reductions.reduced(["layers", "entries", "hidden_size"], {}) == ["layers", "hidden_size"]
    filled = run(engines, compiled)
    assert torch.equal(filled.results["m"], filled.results["f.h"].mean(dim=[1, 2]))
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.w\.ops\[0\]\.with: 'm' is laid out "
                                        r"\['layers', 'hidden_size'\]: it keeps a layers axis, and this write lands "
                                        r"at one layer"):
        plan(data={"d": DAYS}, steps={
            "f": forward({"h": {"site": stacked, "pos": -1}}), "m": {"kind": "op", "op": "mean", "of": "f.h"},
            "patched": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "swap", "with": "m"}]}})}).compile()


def test_s13_a_subspace_wider_than_its_site_is_refused_before_any_forward(engines):
    """trial-08 x10: `k` counts directions of the site's value, so it is at most the site's width."""
    def change(d):
        d["featurizers"]["rot"]["k"] = 32
        d["steps"]["fit"]["epochs"] = 1
    refused_before_any_forward(engines, edited("das", change).compile(),
                               r"featurizer 'rot' has k 32, and it is used at width 16 \(layer_output\), which has no "
                               r"more than 16 directions; k is at most 16")
    refused_before_any_forward(engines, edited("pca_harvest", lambda d: d["steps"]["basis"].update(k=17)).compile(),
                               r"basis: pca k 17 counts directions of a value 16 wide, which has no more than 16")


def cross_site(root: str, steps: dict, featurizer: dict, saves: list[str]) -> Plan:
    return plan(root, data={"train": {"path": "weekdays/data", "split": "train"}},
                featurizers={"oproj": featurizer}, steps=steps, saves=saves).compile()


def adapted(name: str, featurizer: str, layer: int | None = 1) -> dict:
    """A forward adapting layer 1's attention output with `featurizer`, read with layer `layer`'s premix (none:
    without `with`)."""
    out = {"component": "attention_output", "layers": 1}
    entries: dict = {"premix": {"site": {"component": "attention_premix", "layers": layer}, "pos": -1}} \
        if layer is not None else {}
    apply = {"op": "apply", "featurizer": featurizer, **({"with": f"{name}.premix"} if layer is not None else {})}
    return {"kind": "forward", "input": "train.input", "interventions": {
        **entries, "adapt": {"site": out, "pos": -1, "ops": [apply]},
        "logits": {"site": {"component": "lm_head_output"}, "pos": -1}}}


def test_s20_a_cross_site_adapter_records_its_input_site_and_loads_only_reading_there(engines, tmp_path):
    """trial-12 x09: an o_proj adapter trained reading the premix records it (`with.component`, `with.layer`);
    loaded without `with`, or reading another layer's premix, it is refused before any forward."""
    root = corpus(tmp_path)
    fit = {"kind": "fit", "train": ["oproj"], "objective": {"ce": 1.0}, "epochs": 1, "batch_size": 2, "seed": 0,
           "optimizer": {"name": "adam", "lr": 0.05}, "eval": {},
           "steps": {"patched": adapted("patched", "fit.oproj"),
                     "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"}}}
    trained = run(engines, cross_site(root, {"fit": fit}, {"kind": "lora", "rank": 2, "alpha": 4}, ["fit.oproj"]))
    trained.dump(os.path.join(root, "documents", "artifacts"))
    file = os.path.join(root, "documents", "artifacts", "fit.oproj.safetensors")
    with safe_open(file, "pt") as f:
        header = f.metadata()
    assert (header["component"], header["with.component"], header["with.layer"]) == \
        ("attention_output", "attention_premix", "1")
    loaded = {"kind": "lora", "file_path": "documents/artifacts/fit.oproj.safetensors"}
    run(engines, cross_site(root, {"a": adapted("a", "oproj")}, loaded, ["a.logits"]))
    refused_before_any_forward(engines, cross_site(root, {"a": adapted("a", "oproj", None)}, loaded, ["a.logits"]),
                               r"a\.adapt: documents/artifacts/fit\.oproj\.safetensors was trained reading its input "
                               r"at attention_premix layer 1 \(with\), and this use gives no 'with'")
    refused_before_any_forward(engines, cross_site(root, {"a": adapted("a", "oproj", 0)}, loaded, ["a.logits"]),
                               r"a\.adapt: .* trained reading its input at attention_premix layer 1 \(with\), and "
                               r"this use's with 'a\.premix' is attention_premix layer 0")
    with safe_open(file, "pt") as f:                    # a bundle written before headers recorded a `with`
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    save_file(tensors, file, metadata={k: v for k, v in header.items() if not k.startswith("with.")})
    refused_before_any_forward(engines, cross_site(root, {"a": adapted("a", "oproj")}, loaded, ["a.logits"]),
                               r"a\.adapt: documents/artifacts/fit\.oproj\.safetensors's header records no input "
                               r"site \(with\.component\): it was trained on its own site's value, or before bundles "
                               r"recorded a 'with'; retrain it with this 'with'$")


def published_sae(sites: dict | None = None, **stated) -> Plan:
    def change(d):
        d["featurizers"]["dictionary"] = {"kind": "sae", "file_path": "documents/artifacts/sae_gemma_scope.safetensors",
                                          "activation": "jumprelu", "centered": False, **stated}
        d["sites"].update(sites or {})
    return edited("sae_published_jumprelu", change)


def test_s1_s2_a_published_sae_states_its_site_and_is_checked_against_it(engines):
    """trial-04 r02, r05: nothing in a published file says where it was trained, so the document says (`site`,
    a declared site or `{component, layers}` at one layer, beside `activation` and `centered`); a use at another
    layer, or at another component of the same width (an MLP's output beside the residual stream), is refused
    before any forward. Its latents are its tensors' to say."""
    with pytest.raises(PlanError, match=r"is a published sae, which says neither how it encodes nor where it was "
                                        r"trained; state 'site', as its release names them$"):
        published_sae().compile()
    with pytest.raises(PlanError, match=r"featurizers\.dictionary\.site: no site 'nowhere'; declared: target, head"):
        published_sae(site="nowhere").compile()
    with pytest.raises(PlanError, match=r"has 32 latents; this document says 16"):
        published_sae(site="target", latents=16).compile()
    elsewhere = {"mlp": {"component": "mlp_output", "layers": 0}, "one": {"component": "layer_output", "layers": 1}}
    for site, use in (("one", "layer_output layer 0"), ({"component": "mlp_output", "layers": 0}, "layer_output "
                                                                                                    "layer 0")):
        refused_before_any_forward(engines, published_sae(elsewhere, site=site).compile(),
                                   rf"base\.latents: featurizer 'dictionary' was trained at .* \(its site\); this use "
                                   rf"is {use}$")
    compiled = published_sae(elsewhere, site="target")
    compiled.sites["target"] = compiled.sites["target"].model_copy(update={"layers": -2})
    run(engines, compiled.compile())                    # layer -2 of two is layer 0, the one it was trained at


# -- refusals that say what to do ---------------------------------------------------------------------------------


def test_m1_two_sweeps_of_one_field_say_how_to_vary_both():
    sites = {"a": {**LAYER0, "layers": {"sweep": [0, 1]}}, "b": {"component": "mlp_output", "layers": {"sweep": [0, 1]}}}
    with pytest.raises(PlanError, match=r"sites\.b\.layers: label 'layers' is also the sweep at sites\.a\.layers; a "
                                        r"sweep's label is the name of the field it stands for, so two sweeps of one "
                                        r"field name cannot cross: sweep one of them and put each value of the "
                                        r"other in a step of its own"):
        plan(data={"d": DAYS}, sites=sites, steps={"f": forward({"x": {"site": "a", "pos": -1},
                                                                 "y": {"site": "b", "pos": -1}})}).compile()


def test_m2_a_metric_over_two_ragged_reads_says_to_read_at_a_static_index():
    with pytest.raises(PlanError, match=r"steps\.kl\.against: 'f\.logits' is laid out \['entries', 'vocab_size'\]; "
                                        r"'kl' scores a value laid out \['entries', 'vocab_size'\], and a ragged "
                                        r"value's entries are its rows' positions, which pair with no other value's; "
                                        r"read both at a static index"):
        plan(data={"d": DAYS}, steps={
            "f": forward({"lens": {"site": {"component": "layer_output", "layers": -1}, "pos": {"index": ":"},
                                   "ops": [{"op": "readout"}]}, "logits": {"site": HEAD, "pos": {"index": ":"}}}),
            "kl": {"kind": "op", "op": "kl", "of": "f.lens", "against": "f.logits"}}).compile()


def test_m3_one_latent_is_read_with_encodes_select_and_the_featurizer_op_is_gone(engines):
    """trial-04 r01's `select` on the `featurizer` op is now `encode`'s own, and reads one latent."""
    def read(*ops_):
        return plan(data={"d": DAYS}, featurizers={"rot": {"kind": "subspace", "k": 4, "seed": 0}},
                    steps={"f": forward({"z": {"site": LAYER0, "pos": -1, "ops": list(ops_)}})}, saves=["f.z"])
    with pytest.raises(PlanError, match=r"there is no 'featurizer' op: 'encode' gives a featurizer's coordinates of "
                                        r"the value and 'apply' its version of it"):
        read({"op": "featurizer", "featurizer": "rot", "mode": "encode", "select": {"features": [2]}})
    one = run(engines, read({"op": "encode", "featurizer": "rot", "select": {"features": [2]}}).compile())
    every = run(engines, read({"op": "encode", "featurizer": "rot"}).compile())
    assert torch.equal(one.results["f.z"], every.results["f.z"][..., [2]])


def test_m4_a_ragged_operand_into_a_ragged_write_says_what_differs():
    """trial-01, trial-02: the two layouts print the same, so the message says the difference is where the
    entries come from."""
    with pytest.raises(PlanError, match=r"with: 'abc\.v' and this write are both ragged, \['entries', "
                                        r"'hidden_size'\]; what differs is that its entries are another forward's "
                                        r"positions, which need not pair with this one's: read both at a static "
                                        r"index"):
        plan(data={"d": PAIR}, steps={
            "abc": {"kind": "forward", "input": "d.abc", "interventions": {"v": {"site": LAYER0, "pos": {"index": ":"}}}},
            "patched": forward({"w": {"site": LAYER0, "pos": {"index": ":"},
                                      "ops": [{"op": "swap", "with": "abc.v"}]}})}).compile()


def test_m5_argmax_over_features_says_it_ranks_a_vocabulary_only():
    with pytest.raises(PlanError, match=r"steps\.top: 'argmax' needs a value with a 'vocab_size' axis; this one is "
                                        r"\['rows', 'pos', 'features'\]; it ranks a vocabulary's tokens, and over "
                                        r"another axis \(an sae's features, a probe's classes\) it is not an op yet: "
                                        r"save the value and rank it after the run"):
        plan(data={"d": DAYS}, featurizers={"rot": {"kind": "subspace", "k": 4}}, steps={
            "f": forward({"z": {"site": LAYER0, "pos": -1, "ops": [{"op": "encode", "featurizer": "rot"}]}}),
            "top": {"kind": "op", "op": "argmax", "of": "f.z"}}).compile()


def test_m6_take_on_a_file_is_an_unknown_field():
    with pytest.raises(PlanError, match=r"data\.d: unknown field 'take'; valid: path, rows, split"):
        plan(data={"d": {"path": "weekdays/data", "take": 2}}, steps={})


def test_m7_a_featurizer_named_as_an_operand_says_it_is_one_and_how_to_reach_it():
    with pytest.raises(PlanError, match=r"w\.ops\[0\]\.direction: 'pc' is a featurizer, not a value; an op reaches it "
                                        r"\(\{'op': 'encode', 'featurizer': 'pc'\} gives a value's coordinates, "
                                        r"'apply' its version of the value\), and the value that entry binds is an "
                                        r"operand$"):
        plan(data={"d": DAYS}, featurizers={"pc": {"kind": "subspace", "k": 1, "seed": 0}}, steps={
            "abl": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "project_out", "direction": "pc"}]}})}
             ).compile()


def test_m8_a_turn_scope_on_text_says_the_input_is_text():
    with pytest.raises(PlanError, match=r"h\.interventions\.a\.pos\.scope\.turn: the input d\.input is text, not a "
                                        r"chat: a turn scope names a chat's messages .* locate the message's words "
                                        r"with a \{'text': \.\.\.\} or \{'column': \.\.\.\} scope"):
        plan(data={"d": DAYS}, steps={"h": forward({"a": {"site": LAYER0, "pos": {"scope": {"turn": "user"},
                                                                                  "index": -1}}})}).compile()


def test_m9_a_rows_free_operand_with_a_layers_axis_names_the_axis():
    """trial-07 E6b: a mean that keeps a layers axis has no rows, so it crosses datasets; what a one-layer write
    refuses is the axis, and the message says so (not "a row per example")."""
    stacked = {"component": "attention_premix", "layers": [0, 1]}
    with pytest.raises(PlanError, match=r"zs\.interventions\.w\.ops\[0\]\.with: 'm' is laid out \['layers', "
                                        r"'num_heads', 'head_dim'\]: it keeps a layers axis, and this write lands at "
                                        r"one layer, so its operand has none; read the operand at the write's own "
                                        r"layer$"):
        plan(data={"d": DAYS, "t": {"rows": DAYS["rows"][:2]}}, steps={
            "icl": forward({"a": {"site": stacked, "pos": -1}}),
            "m": {"kind": "op", "op": "mean", "of": "icl.a", "over": ["rows", "pos"]},
            "zs": {"kind": "forward", "input": "t.input", "interventions": {"w": {
                "site": {"component": "attention_premix", "layers": 1}, "pos": -1,
                "ops": [{"op": "swap", "with": "m"}]}}}}).compile()


def test_m10_a_generate_write_at_the_last_token_of_every_call_names_the_two_writes():
    with pytest.raises(PlanError, match=r"g\.interventions\.w\.pos: frame ':' with index -1 counts back from the "
                                        r"row's last position, known only when the row stops, and a write lands as "
                                        r"each call runs; for the prompt's last positions and every decode call, "
                                        r"write two entries: \{'frame': 0, 'index': -1\} and \{'frame': '1:'\}$"):
        plan(data={"d": DAYS}, steps={"g": {"kind": "generate", "input": "d.input", "max_new_tokens": 3,
                                            "interventions": {"w": {"site": LAYER0, "pos": {"frame": ":", "index": -1},
                                                                    "ops": [{"op": "scale", "factor": 2.0}]}}}}
             ).compile()


def test_m11_a_fit_over_datasets_of_unequal_rows_says_how_to_pair_them():
    def change(d):
        d["data"]["pin"] = {"rows": DAYS["rows"]}
        d["steps"]["fit"]["steps"]["pinned"] = {"kind": "forward", "input": "pin.input",
                                                "interventions": {"logits": {"site": "head", "pos": -1}}}
    with pytest.raises(PlanError, match=r"fit\.steps: the body's datasets have \{.*\} rows; it runs over "
                                        r"minibatches of the same rows of each, paired row by row, so they have as "
                                        r"many rows: repeat the smaller dataset's rows \(or cut the larger's\) to the "
                                        r"same count, or fit on one"):
        edited("lora_train", change).compile()


def test_m12_a_refusal_in_an_eval_names_the_eval_and_its_dataset(engines):
    """trial-12 x02: the training rows hold the scope's text and the held-out rows do not; the refusal names the
    fit, the eval and the dataset its rows are."""
    def change(d):
        d["steps"]["fit"]["steps"]["adapted"]["interventions"]["adapt"]["pos"] = {
            "scope": {"text": ["Thursday", "Friday"]}}
    refused_before_any_forward(engines, edited("lora_train", change).compile(),
                               r"^fit: eval 'held_out', with dataset 'train' read as 'test' \(a row is the eval's "
                               r"dataset's\): adapted\.adapt: row 0: text \['Thursday', 'Friday'\] is not in its "
                               r"input$")


def test_m13_a_sweep_on_the_optimizers_name_says_a_sweep_cannot_stand_on_a_name():
    with pytest.raises(PlanError, match=r"steps\.fit\.optimizer\.name: a sweep cannot stand on a name, and the "
                                        r"optimizer's name is one: a fit makes one optimizer; write one fit per "
                                        r"optimizer"):
        edited("lora_train", lambda d: d["steps"]["fit"]["optimizer"].update(name={"sweep": ["sgd", "adamw"]}))
