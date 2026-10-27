"""What the trials found, one test per finding on the tiny models: each bug fixed or refused by name, each silent
behaviour a refusal or a check, and each refusal that did not say what to do saying it. A test is named after its
finding's tag."""

import json
import os
import shutil
from collections import ChainMap
from typing import Any

import numpy
import pytest
import torch
from conftest import BATCH_SIZE, DOCUMENTS, ROOT, bundle_header, dims, edited, on_each, placed, same
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from clab import Plan, ops
from clab.compile import answers, positions
from clab.format import data
from clab.format.sweeps import points
from clab.ops import metrics, reductions, transforms
from clab.plan import PlanError

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
    """A number is not a distribution, and a flag is not even a value."""
    for op, against, why in (("kl", 0, "is not a name, and only a named value is a distribution"),
                             ("js", 1.5, "is not a name, and only a named value is a distribution"),
                             ("kl", True, "is not a named value or a number")):
        with pytest.raises(PlanError, match=rf"steps\.h: against:? {against!r} {why}"):
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
    dims = ["rows", "pos", "vocab_size"]
    assert transforms.argmax(tied, dims).item() == 1
    assert transforms.top_k(tied, 2, dims)["ids"].tolist() == [[[1, 3]]]
    assert metrics.match(tied, torch.tensor([[1]]), ["rows", "pos", "vocab_size"]).item() == 1.0
    assert metrics.match(tied, torch.tensor([[3]]), ["rows", "pos", "vocab_size"]).item() == 0.0
    close = torch.zeros(1, 1, 10, dtype=torch.bfloat16)
    close[0, 0, 5], close[0, 0, 2] = 0.25, 0.251953125          # one bfloat16 step apart
    probs = close.softmax(-1)
    assert probs[0, 0, 5] == probs[0, 0, 2]                     # the probabilities tie; the logits do not
    assert probs.topk(1).indices.item() == 5                    # which a top-k of the probabilities ranks first
    assert transforms.argmax(close, dims).item() == 2 == transforms.top_k(close, 1, dims)["ids"].item()
    many = torch.tensor([[9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [3.0, 3.0, 3.0, 5.0, 0.0, 0.0, 0.0]])
    assert metrics.ranked(many, 3).tolist() == [[0, 1, 2], [3, 0, 1]]
    assert metrics.ranked(many, 1).tolist() == [[0], [3]]
    for name in ("argmax", "top_k"):
        assert "an exact tie goes to the lower index" in " ".join((ops.OPS[name].__doc__ or "").split()).lower(), name


def test_b5_a_sweep_of_objects_is_refused_at_compile():
    """trial-06 e10: the points of a sweep are keyed by its values, so a value is a scalar or a list of them."""
    with pytest.raises(PlanError, match=r"a sweep's values are numbers, strings, booleans or lists of them, each "
                                        r"point keyed by its value; \{'index': '-2:'\} is an object: sweep a scalar "
                                        r"inside the pos, which takes one on frame, index and a scope's occurrence"):
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
    assert placed(filled)["g.at"] == ["alignment_missing", "alignment_missing", [6], [7]]
    assert filled.results["g.at"][:2].isnan().all() and not filled.results["g.at"][2:].isnan().any()
    assert filled.results["top"][:, 0].tolist()[:2] == [-1, -1]


def test_b6_batches_join_on_the_first_ones_device():
    """A batch that made no call a read names makes its blank where no value says otherwise; batches join on
    the first one's device, as a `layers` read's layers do, so a CPU blank beside CUDA values (the crash) joins.
    Two devices without a GPU: a meta batch first, then a CPU one."""
    from clab.steps.batches import joined

    compiled = generate_at(3)
    step: Any = compiled.steps["g"]
    state: ChainMap = ChainMap()
    joined(state, "g", step, [{"at": torch.zeros(2, 1, 4, device="meta")}, {"at": torch.zeros(2, 1, 4)}])
    value = state["g.at"][()]
    assert value.device.type == "meta" and value.shape == (4, 1, 4)


def test_b7_a_gate_inside_a_pca_encode_trains_on_the_pca_coordinates(engines, tmp_path):
    """trial-08 x07: brackets nest, so a gate encoded inside a pca's coordinates is as wide as they are and
    trains there, an encode step of the gate over a harvest read in them included; the fit runs on both engines
    alike."""
    root = corpus(tmp_path)
    basis = torch.linalg.qr(torch.randn(16, 4, generator=torch.Generator().manual_seed(0)))[0].contiguous()
    save_file({"basis": basis, "mean": torch.zeros(16)}, os.path.join(root, "documents", "artifacts", "pc.safetensors"),
              metadata=bundle_header(kind="pca", k="4", document="pca_harvest"))

    def change(d):
        d["featurizers"]["pc"] = {"kind": "pca", "k": 4, "file_path": "documents/artifacts/pc.safetensors"}
        d["steps"]["counterfactual"]["interventions"]["h"]["ops"] = [{"op": "encode", "featurizer": "pc"}]
        d["steps"]["fit"]["steps"]["patched"]["interventions"]["patch"]["ops"] = [
            {"op": "encode", "featurizer": "pc"}, {"op": "encode", "featurizer": "fit.mask"},
            {"op": "swap", "with": "v_cf"}, {"op": "decode", "featurizer": "fit.mask"},
            {"op": "decode", "featurizer": "pc"}]
        d["steps"]["fit"].update(epochs=2, objective={"ce": 1.0, "mask.mask": 0.01})
        for name in ("held_counterfactual", "held_patched", "held_iia"):
            del d["steps"][name]
        d["saves"] = ["fit.train.iia", "fit.mask"]
    one = edited("dbm", change)
    compiled = Plan(**one.model_dump(exclude_defaults=True, exclude={"document"}),
                    document=os.path.join(root, "documents", "dbm.json")).compile()
    got = [filled.results for filled in on_each(engines, compiled).values()]
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
        encoded = data.encode(tokenizer, chat, None)
        (_, a, b), = [t for t in encoded["turns"] if t[0] == role]
        assert encoded["text"][a:b] == "INST" and encoded["text"][a - 1] == " " and encoded["text"][b] == " "
    marked = [{"role": "user", "content": data.MARKER}, {"role": "assistant", "content": "ok"},
              {"role": "user", "content": "INST"}]          # an earlier message holds the marker itself
    encoded = data.encode(tokenizer, marked, None)
    (_, a, b) = encoded["turns"][2]
    assert encoded["text"][a:b] == "INST" and encoded["text"][a - 7:a] == "[INST] "
    compiled = plan(data={"c": {"rows": [{"chat": [{"role": "user", "content": "INST"}]}]}}, steps={
        "f": {"kind": "forward", "input": "c.chat", "interventions": {"u": {
            "site": LAYER0, "pos": {"scope": {"turn": "user"}}}}}}, saves=["f.u"]).compile()
    filled = run(engines, compiled)
    tokens = filled.run.tokens["c.chat"][0]
    assert [tokens[p] for p in placed(filled)["f.u"][0]] == ["▁IN", "ST"]


def test_b13_top_k_over_a_read_of_every_token_keeps_each_rows_own(engines, tmp_path):
    """trial-11 e33: the value is each row's ids, probabilities and tokens at each of its positions, -1, NaN
    and no token past its own, and it saves, reads back and tables."""
    compiled = plan(data={"d": {"rows": [{"input": "If today is Monday, tomorrow is"}, {"input": "Hello"}]}}, steps={
        "f": forward({"l": {"site": HEAD, "pos": {"index": ":"}}}),
        "k": {"kind": "op", "op": "top_k", "of": "f.l", "k": 3}}, saves=["k", "f.l"]).compile()
    filled = run(engines, compiled)
    k, read, where = filled.results["k"], filled.results["f.l"], placed(filled)["f.l"]
    assert set(k) == {"ids", "probs", "tokens"} and k["ids"].shape == (2, len(where[0]), 3)
    for r, p in enumerate(where):
        assert torch.equal(k["ids"][r, :len(p), 0], read[r, :len(p)].argmax(-1))
        assert (k["ids"][r, len(p):] == -1).all() and k["probs"][r, len(p):].isnan().all()
    assert [len(t) for t in k["tokens"]] == [len(where[0])] * 2 and k["tokens"][1][-1] == [None] * 3
    rows = filled.table("k")
    assert len(rows) == sum(map(len, where)) and set(rows[0]["value"]) == {"ids", "probs", "tokens"}
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
    assert placed(filled)["f.first"] == [[3, 4]] and placed(filled)["f.second"] == [[9, 10]]
    assert placed(filled)["f.every"] == [[3, 4, 9, 10]]
    gpt2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2",
                                         revision=plan_gpt2_revision())
    for tokenizer in (gpt2, engines("nnterp", compiled.model).tokenizer):
        for text, find, count in (("Then Ann and Annabel met. Ann left", " Ann", 2),
                                  ("I love Paris. Paris is big", "Paris", 2)):
            row = data.encode(tokenizer, text, None)
            assert len(positions.located([(find, False, False)], row["text"], row["offsets"])) == count
        row = data.encode(tokenizer, "I love Paris. Paris is big", None)
        (a, b), = positions.located([("Paris. ", False, False)], row["text"], row["offsets"])
        covered = [t for t, (x, y) in enumerate(row["offsets"]) if x < b and y > a]
        assert "Paris" not in tokenizer.decode([row["ids"][covered[-1]]])
    words, letters = "f(): a: b", [(0, 1), (1, 4), (4, 6), (6, 7), (7, 9)]       # `():` is one token
    assert positions.located([(":", False, False)], words, letters) == [(6, 7)]
    assert positions.located([("a", False, False)], words, letters) == [(5, 6)]     # ` a` without its space
    assert positions.located([("Ann", False, False)], "Annabel", [(0, 3), (3, 7)]) == []
    assert positions.located([("Ann", False, False)], "Annabel", [(0, 3), (3, 7)], ahead=False) == [(0, 3)]
    from clab.plan import Pos, Scope
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
    assert placed(filled)["patched.w"] != placed(filled)["abc.v"]      # the same tokens, from the end
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
    an op step or a metric over two reads in different orders, refused when the plan compiles (picks are the
    document's to resolve); read in one order it lands head for head."""
    with pytest.raises(PlanError, match=r"patched\.interventions\.w\.ops\[0\]\.with: 'm' holds num_heads \[1, 3\], "
                                        r"and this write's selection holds \[3, 1\]; a select list is positional, so "
                                        r"they would trade places: select num_heads \[3, 1\] at the read$"):
        heads([1, 3], [3, 1])
    filled = run(engines, heads([1, 3], [1, 3]))
    assert torch.equal(filled.results["patched.back"][:, :, [1, 3]], filled.results["m"].expand(4, 1, 2, 4))
    for site, axis, op in (({"component": "attention_premix", "layers": 1}, "num_heads", "sub"),
                           (LAYER0, "hidden_size", "cosine")):
        two = forward({"a": {"site": {**site, "select": {axis: [1, 3]}}, "pos": -1},
                       "b": {"site": {**site, "select": {axis: [3, 1]}}, "pos": -1}})
        with pytest.raises(PlanError, match=rf"steps\.d2\.with: 'f\.b' holds {axis} \[3, 1\], and the value it acts on "
                                            rf"holds \[1, 3\]; a select list is positional, so they would trade places: "
                                            rf"select one {axis} list at both reads$"):
            plan(data={"d": DAYS}, steps={"f": two, "d2": {"kind": "op", "op": op, "of": "f.a", "with": "f.b"}}).compile()


def test_s11_one_heads_operand_into_several_heads_is_refused(engines):
    """trial-07 E5: an operand broadcasts over rows and positions only, never over heads: picks are refused when
    the plan compiles, a `":"` against picks once the model says how many heads it has."""
    with pytest.raises(PlanError, match=r"patched\.interventions\.w\.ops\[0\]\.with: 'm' holds num_heads \[0\], and "
                                        r"this write's selection holds \[0, 1, 3\]; an operand broadcasts over rows "
                                        r"and positions only: select num_heads \[0, 1, 3\] at the read$"):
        heads([0], [0, 1, 3])
    refused_before_any_forward(engines, heads(":", [2]), r"operand 'm' holds num_heads \[0, 1, 2, 3\], and this "
                               r"write's selection holds \[2\]; an operand holds the same ones: select num_heads "
                               r"\[2\] at the read$")


def test_s12_a_mean_without_over_keeps_a_layers_axis(engines):
    """trial-07 E6: `mean` alone removes rows and positions; a `layers` axis stays unless `over` names it, and
    a one-layer write given a value that keeps it takes its slice labelled with the write's layer."""
    stacked = {"component": "layer_output", "layers": [1, 0]}
    compiled = plan(data={"d": DAYS}, steps={
        "f": forward({"h": {"site": stacked, "pos": -1}}), "m": {"kind": "op", "op": "mean", "of": "f.h"},
        "all": {"kind": "op", "op": "mean", "of": "f.h", "over": ["layers", "rows", "pos"]}},
        saves=["f.h", "m", "all"]).compile()
    assert dims(compiled, "m") == ["layers", "hidden_size"] and dims(compiled, "all") == ["hidden_size"]
    assert reductions.reduced(["layers", "rows", "pos", "hidden_size"], {}) == ["layers", "hidden_size"]
    filled = run(engines, compiled)
    assert torch.equal(filled.results["m"], filled.results["f.h"].mean(dim=[1, 2]))
    written = run(engines, plan(data={"d": DAYS}, steps={
        "f": forward({"h": {"site": stacked, "pos": -1}}), "m": {"kind": "op", "op": "mean", "of": "f.h"},
        "patched": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "swap", "with": "m"}]}})},
        saves=["m", "patched.w"]).compile())
    assert torch.equal(written.results["patched.w"], written.results["m"][1].expand(4, 1, 16))     # labelled 0


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
                                        r"trained; state 'site', as its release names it$"):
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


BF16 = {**TINY, "dtype": "bfloat16"}
# 64 rows whose last token is one word, so each row's residual there is near the rows' mean; and an answer that is
# the most likely token of every row's logits scaled by 200, at a probability near 1.
MANY = {"rows": [{"input": f"Row {i} says {i * 7 % 13} things", "answer": "Switch", "other": " the"}
                 for i in range(64)]}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_d8_o13_s17_a_bf16_models_mean_cosine_and_probability_are_float32(engines, kind):
    """trial-02's 64-row mean moved in bfloat16 steps, trial-05's cosine rounded to 2⁻⁸ near 1, trial-11's
    probabilities near 1 to 1.0: each is now the float32 computation, by hand, of the bfloat16 values the model
    gave, and differs from the same computation in bfloat16."""
    filled = run(engines, plan(model=BF16, data={"d": MANY}, steps={
        "f": forward({"resid": {"site": LAYER0, "pos": -1},
                      "hot": {"site": HEAD, "pos": -1, "ops": [{"op": "scale", "factor": 200.0}]}}),
        "avg": {"kind": "op", "op": "mean", "of": "f.resid"},
        "cos": {"kind": "op", "op": "cosine", "of": "f.resid", "with": "avg"},
        "p": {"kind": "op", "op": "token_prob", "of": "f.hot", "token": "d.answer"}},
        saves=["f.resid", "f.hot", "avg", "cos", "p"]).compile(), kind)
    resid, hot, avg, cos, p = (filled.results[k] for k in ("f.resid", "f.hot", "avg", "cos", "p"))
    assert resid.dtype == hot.dtype == torch.bfloat16 and resid.shape == (64, 1, 16)
    assert torch.equal(avg, resid.float().mean(dim=[0, 1]))
    assert not torch.equal(avg, resid.mean(dim=[0, 1]).float())
    assert torch.equal(cos, torch.nn.functional.cosine_similarity(resid.float(), avg, dim=-1))
    rounded = torch.nn.functional.cosine_similarity(resid, avg.to(torch.bfloat16), dim=-1).float()
    assert (cos > 0.99).float().mean() > 0.5 and (rounded * 256 % 1 == 0).all() and (rounded != cos).all()
    answer = engines(kind, filled.model).tokenizer("Switch", add_special_tokens=False)["input_ids"]
    assert len(answer) == 1
    assert torch.equal(p, hot.float().softmax(-1)[..., answer[0]])
    near = (p > 0.998) & (p < 1)
    assert near.any() and (hot.softmax(-1)[..., answer[0]][near] == 1).all()       # bfloat16 rounds them to 1


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_every_metric_and_reduction_is_float32_on_a_bf16_model_and_an_edit_stays_bf16(engines, kind):
    """Each metric and `mean`, `sum`, `min` returns float32 on a bfloat16 model; a value read from the model, and
    an edit's value, which it writes back, keep the model's dtype, even with a float32 operand: the edit runs in
    float32 and rounds once (`add` of the float32 mean, `project_out` of it as a step)."""
    few = {"rows": MANY["rows"][:4]}
    metric = {"match": {"expected": "d.answer"}, "logit_diff": {"a": "d.answer", "b": "d.other"},
              "token_prob": {"token": "d.answer"}, "token_logit": {"token": "d.answer"},
              "cross_entropy": {"target": "d.answer"}, "kl": {"against": "base.logits"},
              "js": {"against": "base.logits"}}
    steps = {"base": forward({"resid": {"site": LAYER0, "pos": -1}, "logits": {"site": HEAD, "pos": -1}}),
             "avg": {"kind": "op", "op": "mean", "of": "base.resid"},
             "f": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "add", "with": "avg"}]},
                           "logits": {"site": HEAD, "pos": -1}}),
             **{name: {"kind": "op", "op": name, "of": "f.logits", **args} for name, args in metric.items()},
             "cosine": {"kind": "op", "op": "cosine", "of": "f.w", "with": "avg"},
             "sum": {"kind": "op", "op": "sum", "of": "f.w"}, "min": {"kind": "op", "op": "min", "of": "f.w"},
             "gen": {"kind": "generate", "input": "d.input", "max_new_tokens": 2},
             "text": {"kind": "op", "op": "detokenize", "of": "gen"},
             "contains": {"kind": "op", "op": "contains", "of": "text", "text": "d.answer"},
             "out": {"kind": "op", "op": "project_out", "of": "base.resid", "direction": "avg"},
             "top": {"kind": "op", "op": "top_k", "of": "f.logits", "k": 3}}
    names = [*metric, "avg", "cosine", "sum", "min", "contains"]
    kept = ["base.resid", "f.w", "f.logits", "out"]
    filled = run(engines, plan(model=BF16, data={"d": few}, steps=steps, saves=[*kept, *names, "top"]).compile(),
                 kind)
    assert {k: filled.results[k].dtype for k in names} == {k: torch.float32 for k in names}
    top = filled.results["top"]
    assert top["probs"].dtype == torch.float32 and torch.equal(
        top["probs"], filled.results["f.logits"].float().softmax(-1).gather(-1, top["ids"]))
    assert {k: filled.results[k].dtype for k in kept} == {k: torch.bfloat16 for k in kept}
    resid, avg = filled.results["base.resid"], filled.results["avg"]
    assert torch.equal(filled.results["f.w"], (resid.float() + avg).to(torch.bfloat16))
    assert torch.equal(filled.results["out"], ops.edits.project_out(resid.float(), avg).to(torch.bfloat16))


def test_a_reduction_that_removes_no_axis_is_refused_at_compile(engines):
    """PR 18's bug 1: a `sum` of a `[hidden_size]` mean removed no axis and wrote the grand total into every
    entry (torch reads no dims as every one). It is refused by name before any forward, with the value's
    layout; naming the axis in `over` reduces it."""
    def summed(**over):
        return plan(data={"d": DAYS}, steps={
            "f": forward({"h": {"site": LAYER0, "pos": -1}}), "avg": {"kind": "op", "op": "mean", "of": "f.h"},
            "total": {"kind": "op", "op": "sum", "of": "avg", **over}}, saves=["avg", "total"])
    with pytest.raises(PlanError, match=r"^steps\.total: a reduction of a value laid out \['hidden_size'\] removes no "
                                        r"axis of it: by default it removes the rows and positions; name the axes it "
                                        r"removes in 'over' \(\"over\": \[\"hidden_size\"\]\)$"):
        summed().compile()
    with pytest.raises(PlanError, match=r"^steps\.total: a reduction of a value laid out \['hidden_size'\] removes no "
                                        r"axis of it: 'over' names none; name the axes"):
        summed(over=[]).compile()
    with pytest.raises(PlanError, match=r"^steps\.f\.interventions\.h\.ops\[1\]: a reduction of a value laid out "
                                        r"\['rows', 'hidden_size'\] removes no axis of it"):
        plan(data={"d": DAYS}, steps={"f": forward({"h": {"site": LAYER0, "pos": -1, "ops": [
            {"op": "mean", "over": ["pos"]}, {"op": "min", "over": []}]}})}).compile()
    filled = run(engines, summed(over=["hidden_size"]).compile())
    assert filled.results["total"].shape == () and torch.equal(filled.results["total"], filled.results["avg"].sum())


def test_s16_renormalize_by_sum_keeps_a_knocked_out_pattern_row_a_distribution(engines):
    """trial-10 §4.1: after a key knockout `renormalize` rescaled each row to its L2 norm, and rows summed 0.73 to
    2.31. With `norm: "sum"` a `scale 0` of key 0 then a renormalize of the whole row leaves each row summing to
    what it did before, 1, over its own keys, its tail past them NaN as the read has it."""
    eager = {**TINY, "attn_implementation": "eager"}
    pattern = {"component": "attention_probabilities", "layers": 0}

    def knocked(norm: str) -> dict:
        return forward({"knock": {"site": {**pattern, "select": {"key": 0}}, "pos": -1,
                                  "ops": [{"op": "scale", "factor": 0.0}]},
                        "renorm": {"site": pattern, "pos": -1, "ops": [{"op": "renormalize", "norm": norm}]}})
    filled = run(engines, plan(model=eager, data={"d": DAYS}, steps={
        "clean": forward({"p": {"site": pattern, "pos": -1}}), "sum": knocked("sum"), "l2": knocked("l2")},
        saves=["clean.p", "sum.renorm", "l2.renorm"]).compile())
    clean, by_sum, by_l2 = (filled.results[k] for k in ("clean.p", "sum.renorm", "l2.renorm"))
    assert clean[2:].isnan().any() and torch.equal(by_sum.isnan(), clean.isnan())  # rows 2 and 3 are shorter
    torch.testing.assert_close(clean.nansum(-1), torch.ones_like(clean[..., 0]))
    torch.testing.assert_close(by_sum.nansum(-1), clean.nansum(-1))
    out = clean.clone()
    out[..., 0] = 0
    torch.testing.assert_close(by_sum, out / out.nansum(-1, keepdim=True), equal_nan=True)
    assert not torch.allclose(by_l2.nansum(-1), clean.nansum(-1))


@pytest.mark.parametrize("norm", ["sum", "l2"])
def test_s16_knocking_out_the_sink_at_every_position_spreads_no_nan(engines, norm):
    """Key 0 knocked out at every position: the first position's row holds key 0 alone, so its sum and norm are
    zero, and `renormalize` leaves it as it is, zero, rather than dividing by zero; no row's logits are NaN."""
    eager = {**TINY, "attn_implementation": "eager"}
    pattern, every = {"component": "attention_probabilities", "layers": 0}, {"index": ":"}
    filled = run(engines, plan(model=eager, data={"d": DAYS}, steps={
        "clean": forward({"p": {"site": pattern, "pos": every}}),
        "sink": forward({"knock": {"site": {**pattern, "select": {"key": 0}}, "pos": every,
                                   "ops": [{"op": "scale", "factor": 0.0}]},
                         "renorm": {"site": pattern, "pos": every, "ops": [{"op": "renormalize", "norm": norm}]},
                         "logits": {"site": HEAD, "pos": every}})},
        saves=["clean.p", "sink.renorm", "sink.logits"]).compile())
    clean, renorm, logits = (filled.results[k] for k in ("clean.p", "sink.renorm", "sink.logits"))
    assert torch.equal(renorm.isnan(), clean.isnan())          # [rows, pos, num_heads, key]: only the ragged tail
    assert (renorm[:, 0].nan_to_num() == 0).all()             # position 0 attends to key 0 alone: left zero
    own = (~logits.isnan().all(-1)).sum(-1)                     # each row's positions, NaN past them
    assert own.tolist() == [9, 9, 4, 5] and not any(logits[r, :n].isnan().any() for r, n in enumerate(own))


QWEN = {"key": "trl-internal-testing/tiny-Qwen3ForCausalLM", "revision": "25bb9b8963688237e94750f6f3e48f1947d3c7a3"}


def test_s18_an_answer_under_strip_keeps_its_leading_space_and_widens_only_its_ends():
    """trial-11 §4.3: under `strip` an interior whitespace token stood for all 441 whitespace-only ids, and a
    number's leading ` ` was dropped. An answer keeps every token but its whitespace-only ones at the end, which
    what follows it would be: its first stands for its spellings without leading whitespace, its last for those
    without trailing, a token between them and a whitespace-only one for itself. A one-token answer's class is
    the one it always had: every spelling stripped both ways."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN["key"], revision=QWEN["revision"])
    texts = answers.vocabulary(tokenizer)

    def classes(value: str, lower: bool = False) -> list[list[str]]:
        return [[texts[i] for i in ids] for ids in answers.answers(tokenizer, [value], "d.a", True, lower)[0]]
    blank = [i for i, t in enumerate(texts) if not t.strip() and "\ufffd" not in t]
    assert len(blank) == 441
    assert classes(" 42") == [[" "], ["4"], ["2"]]                  # this tokenizer splits digits: ` 42` is three
    first, york, dot = classes(" New York.")
    assert first == [" New", "New"] and york == [" York"] and {".", ".\n", ".\n\n"} <= set(dot) and " ." not in dot
    assert all(t.rstrip() == "." for t in dot)
    assert classes(" New York ") == classes(" New York") == [[" New", "New"], [" York"]]
    assert classes("  New York ") == [[" "], [" New"], [" York"]]
    assert classes(" New  York") == [[" New", "New"], [" "], [" York"]]
    with pytest.raises(ValueError, match=r"^d\.a row 0: ' ' is no tokens once strip drops the whitespace-only tokens "
                                         r"at its end$"):
        classes(" ")
    for word in (" Yes", "Yes", " Paris", "YES", " the"):
        for lower in (False, True):
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1
            key = data.compared(texts[ids[0]], True, lower)              # the rule for one token before this PR
            before = [i for i, t in enumerate(texts) if "\ufffd" not in t and data.compared(t, True, lower) == key]
            assert answers.answers(tokenizer, [word], "d.a", True, lower)[0] == [before]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_s18_a_strip_answer_and_scope_that_do_not_line_up_name_the_whitespace_only_token(engines, kind):
    """` 1776` keeps its leading ` ` token as an answer, and a scope under strip covers `1776` alone: refused,
    naming the answer's token; a scope on ` 1776` with an answer `1776` is refused naming the read's position.
    The answer under strip with the scope on its text as written lines up, the space its first token's own."""
    rows = [{"input": "The year was 1776", "year": " 1776", "plain": "1776"}]
    year = {"column": "d.year", "strip": True}

    def scored(answer: Any, scope: dict) -> Plan:
        return plan(model=QWEN, data={"d": {"rows": rows}}, steps={
            "read": forward({"logits": {"site": HEAD, "pos": {"scope": {**scope, "before": True}}}}),
            "p": {"kind": "op", "op": "token_prob", "of": "read.logits", "token": answer}},
            saves=["p", "read.logits"]).compile()
    refused_before_any_forward(engines, scored(year, {"column": year}),
                               r"^p: d\.year row 0: ' 1776' is 5 tokens; the read has 4 positions on that row; .*; its "
                               r"token 0, ' ', is whitespace only, which an answer keeps and a scope under strip does "
                               r"not cover: scope the column without strip$")
    refused_before_any_forward(engines, scored("d.plain", {"text": " 1776"}),
                               r"^p: d\.plain row 0: '1776' is 4 tokens; the read has 5 positions on that row; .*; the "
                               r"read's position 2 predicts ' ', whitespace only, which the answer has no token for: "
                               r"write the answer with its space$")
    filled = run(engines, scored(year, {"column": "d.year"}), kind)
    assert placed(filled)["read.logits"] == [[2, 3, 4, 5, 6]]
    space = engines(kind, filled.model).tokenizer(" ", add_special_tokens=False)["input_ids"]
    probs = filled.results["read.logits"][0, 0].softmax(-1)
    assert len(space) == 1 and filled.results["p"][0, 0] == probs[space[0]]


CHAT = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi."}]
# The tokens Qwen3's non-thinking generation prompt puts after the last message.
HEADER = ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"]


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_d6_a_chats_template_carries_the_generation_prompt_and_keeps_its_turns(engines, kind):
    """trial-01, trial-05: a chat rendered without a generation prompt, so a generation from it was pre-rendered
    by hand and lost its `turn` scopes. A dataset's `template` goes to `apply_chat_template`: the chat ends in
    the assistant's header, as the pre-rendered prompt does, token for token; the header's tokens are positions
    (`pos: -1` its last) and no message's (no `assistant` turn); `turn` scopes still find their messages; and a
    generate from it says what one from the pre-rendered prompt says."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN["key"], revision=QWEN["revision"])
    template = {"add_generation_prompt": True, "enable_thinking": False}
    rows = [{"chat": CHAT, "prompt": tokenizer.apply_chat_template(CHAT, tokenize=False, **template)}]

    def chat(**more) -> Plan:
        return plan(model=QWEN, data={"d": {"rows": rows, "template": template}}, steps={
            "f": forward({"user": {"site": LAYER0, "pos": {"scope": {"turn": "user"}}},
                          "system": {"site": LAYER0, "pos": {"scope": {"turn": "system"}}},
                          "last": {"site": LAYER0, "pos": -1}, **more}, input="d.chat"),
            "chat": {"kind": "generate", "input": "d.chat", "max_new_tokens": 3, "do_sample": False},
            "text": {"kind": "generate", "input": "d.prompt", "max_new_tokens": 3, "do_sample": False}},
            saves=["chat", "text", "f.last"]).compile()
    filled = run(engines, chat(), kind)
    tokens = filled.run.tokens
    assert tokens["d.chat"] == tokens["d.prompt"] and tokens["d.chat"][0][-9:] == HEADER
    where = placed(filled)
    assert [tokens["d.chat"][0][p] for p in where["f.user"][0]] == ["Say", "Ġhi", "."]
    assert [tokens["d.chat"][0][p] for p in where["f.system"][0]] == ["Be", "Ġbrief", "."]
    assert where["f.last"] == [[len(tokens["d.chat"][0]) - 1]]
    assert torch.equal(filled.results["chat"], filled.results["text"]) and tokens["chat"] == tokens["text"]
    refused_before_any_forward(engines, chat(header={"site": LAYER0, "pos": {"scope": {"turn": "assistant"}}}),
                               r"^f\.header: .*the row has no 'assistant' message")
    plain = plan(model=QWEN, data={"d": {"rows": rows}}, steps={
        "f": forward({"last": {"site": LAYER0, "pos": -1}}, input="d.chat")}, saves=["f.last"]).compile()
    assert run(engines, plain, kind).run.tokens["d.chat"][0][-2:] == ["<|im_end|>", "Ċ"]


SAE = {"kind": "sae", "file_path": "documents/artifacts/sae.safetensors"}


def test_o2_argmax_and_top_k_over_a_patterns_key_axis_rank_each_rows_own_keys(engines):
    """A pattern's keys past a shorter row's own are NaN; ranking over `key` takes the row's own (they were all
    -1 when a NaN anywhere made the vector missing), and a pick past them is id -1 and NaN."""
    eager = {**TINY, "attn_implementation": "eager"}
    pattern = {"component": "attention_probabilities", "layers": 0}
    filled = run(engines, plan(model=eager, data={"d": DAYS}, steps={
        "f": forward({"p": {"site": pattern, "pos": -1}}),
        "most": {"kind": "op", "op": "argmax", "of": "f.p", "over": "key"},
        "top": {"kind": "op", "op": "top_k", "of": "f.p", "over": "key", "k": 6}},
        saves=["f.p", "most", "top"]).compile())
    p, most, top = filled.results["f.p"], filled.results["most"], filled.results["top"]
    assert dims(filled, "most") == ["rows", "pos", "num_heads"]
    assert (~p[:, 0, 0].isnan()).sum(-1).tolist() == [9, 9, 4, 5]            # each row's own keys
    assert most[:, 0, 0].tolist() == [1, 1, 1, 0]
    assert torch.equal(most, p.nan_to_num(nan=-1).argmax(-1))
    assert torch.equal(top["ids"][..., 0], most) and (top["ids"][2, ..., 4:] == -1).all()  # row 2 has four keys
    assert set(top) == {"ids", "values"} and torch.equal(top["values"][:2], p[:2].topk(6).values)
    assert top["values"][2, ..., 4:].isnan().all() and not top["values"][2, ..., :4].isnan().any()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_o2_top_k_ranks_an_sae_s_features_and_argmax_a_probe_s_classes(engines, kind):
    """trial-04: ranking an sae's latents went to Python. `top_k` with `over: "features"` after an encode of the
    fixture sae is its k largest latents, by index, an exact tie (the zeros a ReLU leaves) to the lower one, with
    no tokens to decode; `argmax` with `over: "classes"` is a probe's most likely class."""
    filled = run(engines, plan(data={"d": DAYS}, featurizers={"sae": SAE}, steps={
        "f": forward({"z": {"site": LAYER0, "pos": -1, "ops": [{"op": "encode", "featurizer": "sae"}]}}),
        "top": {"kind": "op", "op": "top_k", "of": "f.z", "k": 20, "over": "features"},
        "best": {"kind": "op", "op": "argmax", "of": "f.z", "over": "features"}},
        saves=["f.z", "top", "best"]).compile(), kind)
    z, top, best = filled.results["f.z"], filled.results["top"], filled.results["best"]
    assert set(top) == {"ids", "values"} and top["ids"].shape == (4, 1, 20) and best.shape == (4, 1)
    assert (z == 0).sum(-1).min() > 12                     # each row's top 20 runs into its zeros
    assert torch.equal(top["ids"], metrics.ranked(z, 20)) and torch.equal(best, top["ids"][..., 0])
    assert torch.equal(top["values"], z.gather(-1, top["ids"]))           # the latents, not a softmax of them
    for ids, values in zip(top["ids"][:, 0], z[:, 0].gather(-1, top["ids"][:, 0])):
        assert ids[values == 0].tolist() == sorted(ids[values == 0].tolist())    # the tied zeros in index order

    from test_fit import probe
    trained = probe(lambda d: (d["steps"]["fit"]["steps"].update(which={"kind": "op", "op": "argmax",
                                                                       "of": "scores", "over": "classes"}),
                               d["saves"].append("fit.train.which")))
    values = engines(kind, trained.model).execute(trained, batch_size=BATCH_SIZE).results
    scores = values["fit.train.scores"]
    assert scores.shape[-1] == 2 and torch.equal(values["fit.train.which"], metrics.ranked(scores, 1)[..., 0])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_o6_a_dose_is_a_named_value_or_a_number_and_the_two_are_the_same_bits(engines, kind, dtype):
    """trial-01: `scale`'s factor and `noise`'s scale took a literal only, so a measured dose was copied by hand.
    Each takes a `Value`, as `add`'s `with` does: a dose named from a `mean` step writes the bits its number
    written as a literal does, on a bfloat16 model too, where the edit runs in float32 and rounds once. Anything
    else is refused by name."""
    def doses(dose: Any) -> Plan:
        return plan(model={**TINY, "dtype": dtype}, data={"d": DAYS}, steps={
            "base": forward({"h": {"site": LAYER0, "pos": -1}}),
            "dose": {"kind": "op", "op": "mean", "of": "base.h", "over": ["rows", "pos", "hidden_size"]},
            "scaled": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "scale", "factor": dose}]},
                               "logits": {"site": HEAD, "pos": -1}}),
            "noisy": forward({"w": {"site": LAYER0, "pos": -1, "ops": [{"op": "noise", "scale": dose, "seed": 3}]},
                              "logits": {"site": HEAD, "pos": -1}})},
            saves=["dose", "scaled.w", "scaled.logits", "noisy.w", "noisy.logits"]).compile()
    named = run(engines, doses("dose"), kind).results
    assert named["dose"].shape == () and named["dose"].dtype == torch.float32 and named["dose"].item() != 0
    literal = run(engines, doses(named["dose"].item()), kind).results
    assert same(named, literal) and named["scaled.w"].dtype == named["noisy.w"].dtype == getattr(torch, dtype)
    h = run(engines, plan(model={**TINY, "dtype": dtype}, data={"d": DAYS}, steps={
        "base": forward({"h": {"site": LAYER0, "pos": -1}})}, saves=["base.h"]).compile(), kind).results["base.h"]
    assert torch.equal(named["scaled.w"], (h.float() * named["dose"]).to(h.dtype))
    for dose, message in ((True, r"factor: True is not a named value or a number"),
                          ({"x": 1}, r"factor: \{'x': 1\} is not a named value or a number"),
                          ("nothing", r"factor: 'nothing' names no value produced before this step")):
        with pytest.raises(PlanError, match=message):
            doses(dose)


def test_d6_an_eval_reads_a_dataset_as_another_only_where_their_templates_agree():
    """A dataset read as another under a fit's `eval` renders its chats with its own `template`, so an eval
    between datasets whose templates differ would score chats rendered another way than the body trained on."""
    template = {"add_generation_prompt": True, "enable_thinking": False}
    rows = [{"chat": [{"role": "user", "content": f"Is {d} a weekday?"}], "label": d[0]} for d in ("Monday", "Sunday")]

    def fit(other: dict) -> Plan:
        return plan(model=QWEN, data={"days": {"rows": rows, "template": template}, "other": {"rows": rows, **other}},
                    featurizers={"probe": {"kind": "linear", "classes": "days.label"}}, steps={"fit": {
                        "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 1, "batch_size": 2,
                        "seed": 0, "optimizer": {"name": "adam", "lr": 0.5}, "eval": {"plain": {"days": "other"}},
                        "steps": {"read": forward({"scores": {"site": LAYER0, "pos": -1, "ops": [
                            {"op": "encode", "featurizer": "fit.probe"}]}}, input="days.chat"),
                            "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores",
                                   "target": "days.label"}}}}, saves=["fit.plain.ce"])
    with pytest.raises(PlanError, match=r"^steps\.fit\.eval\.plain: dataset 'days' is read as 'other', and their "
                                        r"templates differ \('days': \{'add_generation_prompt': True, "
                                        r"'enable_thinking': False\}, 'other': None\); a chat renders with its own "
                                        r"dataset's template, so give the two the same one$"):
        fit({}).compile()
    assert fit({"template": template}).compile().data["other"].template == template
    # Where the body reads no chat column of the substituted dataset, the templates are not compared: a template on
    # a dataset no step reads a chat of is what is refused.
    text = [{"text": f"Is {d} a weekday?", "label": d[0]} for d in ("Monday", "Sunday")]
    with pytest.raises(PlanError, match=r"^data\.other\.template: no step reads a chat column of 'other', and a template "
                                        r"renders a chat"):
        plan(model=QWEN, data={"days": {"rows": text}, "other": {"rows": text, "template": template}},
             featurizers={"probe": {"kind": "linear", "classes": "days.label"}}, steps={"fit": {
                 "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0}, "epochs": 1, "batch_size": 2,
                 "seed": 0, "optimizer": {"name": "adam", "lr": 0.5}, "eval": {"plain": {"days": "other"}},
                 "steps": {"read": forward({"scores": {"site": LAYER0, "pos": -1, "ops": [
                     {"op": "encode", "featurizer": "fit.probe"}]}}, input="days.text"),
                     "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores",
                            "target": "days.label"}}}}, saves=["fit.plain.ce"]).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_o6_an_op_steps_dose_of_one_number_broadcasts_and_one_per_row_broadcasts_by_row(engines, kind):
    """As a step too, a dose of one number is the bits its literal is; one laid out `[rows]` (a mean over the
    positions and features) broadcasts by axis name, each row's dose over that row's positions and features, never
    against the features as a bare tensor would; one with an axis the value lacks is refused by name."""
    def stepped(dose: Any, over: list[str]) -> Plan:
        return plan(data={"d": DAYS}, steps={
            "base": forward({"h": {"site": LAYER0, "pos": -1}}),
            "dose": {"kind": "op", "op": "mean", "of": "base.h", "over": over},
            "scaled": {"kind": "op", "op": "scale", "of": "base.h", "factor": dose},
            "noisy": {"kind": "op", "op": "noise", "of": "base.h", "scale": dose}},
                    saves=["base.h", "dose", "scaled", "noisy"])
    every = ["rows", "pos", "hidden_size"]
    named = run(engines, stepped("dose", every).compile(), kind).results
    assert named["dose"].shape == () and same(named, run(engines, stepped(named["dose"].item(), every).compile(),
                                                          kind).results)
    per_row = run(engines, stepped("dose", ["pos", "hidden_size"]).compile(), kind).results
    assert per_row["dose"].shape == (4,)
    assert torch.equal(per_row["scaled"], per_row["base.h"] * per_row["dose"][:, None, None])
    torch.testing.assert_close(per_row["noisy"] - per_row["base.h"],       # the same draw, each row's dose
                               (named["noisy"] - named["base.h"]) * (per_row["dose"] / named["dose"])[:, None, None],
                               rtol=1e-5, atol=1e-7)
    with pytest.raises(PlanError, match=r"^steps\.scaled\.factor: 'lens' is laid out \['rows', 'pos', 'vocab_size'\]; "
                                        r"'scale' acts on a value laid out \['rows', 'pos', 'hidden_size'\], and an "
                                        r"operand's axes are among the value's"):
        plan(data={"d": DAYS}, steps={
            "base": forward({"h": {"site": LAYER0, "pos": -1}}),
            "lens": {"kind": "op", "op": "readout", "of": "base.h"},
            "scaled": {"kind": "op", "op": "scale", "of": "base.h", "factor": "lens"}}).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_o6_a_dose_per_layer_scales_each_layer_a_write_across_layers_is_made_at(engines, kind):
    """A dose with only a `layers` axis is one number at each layer: a write across layers takes its slice of
    each, the bits of one write per layer with that layer's number as a literal."""
    every = {"component": "layer_output", "layers": ":"}
    measured = {"base": forward({"h": {"site": every, "pos": -1}}),
                "dose": {"kind": "op", "op": "mean", "of": "base.h", "over": ["rows", "pos", "hidden_size"]}}
    head = {"logits": {"site": HEAD, "pos": -1}}
    across = run(engines, plan(data={"d": DAYS}, steps={**measured, "w": forward({
        "w": {"site": every, "pos": -1, "ops": [{"op": "scale", "factor": "dose"}]}, **head})},
        saves=["dose", "w.w", "w.logits"]).compile(), kind).results
    dose = across["dose"]
    assert dose.shape == (2,) and dose[0] != dose[1]
    one = run(engines, plan(data={"d": DAYS}, steps={"w": forward({
        **{f"w{layer}": {"site": {"component": "layer_output", "layers": layer}, "pos": -1,
                         "ops": [{"op": "scale", "factor": dose[layer].item()}]} for layer in (0, 1)}, **head})},
        saves=["w.w0", "w.w1", "w.logits"]).compile(), kind).results
    assert torch.equal(across["w.w"], torch.stack([one["w.w0"], one["w.w1"]]))
    assert torch.equal(across["w.logits"], one["w.logits"])


# -- refusals that say what to do ---------------------------------------------------------------------------------


def test_m1_two_sweeps_of_one_label_zip_and_the_refusal_names_label():
    """trial 1, 4: two sweeps of one field name are one label, so one axis: over the same values their points
    pair up; over different ones they are refused, and the message says what crosses them, a `label` of its
    own."""
    def two(b: dict) -> Plan:
        sites = {"a": {**LAYER0, "layers": {"sweep": [0, 1]}}, "b": {"component": "mlp_output", "layers": b}}
        return plan(data={"d": DAYS}, sites=sites, steps={"f": forward({"x": {"site": "a", "pos": -1},
                                                                         "y": {"site": "b", "pos": -1}})}).compile()
    zipped = two({"sweep": [0, 1]})
    assert [p.at for p in points(zipped.steps["f"])] == [{"layers": 0}, {"layers": 1}]
    assert list(zipped.sweeps) == ["layers"] and zipped.sweeps["layers"].values == [0, 1]
    with pytest.raises(PlanError, match=r"^sites\.b\.layers: label 'layers' is also the sweep at sites\.a\.layers, "
                                        r"over \[0, 1\]; one label is one axis, whose sweeps pair their points: give "
                                        r"one of them its own label to cross them \(\{'sweep': \[\.\.\.\], 'label': "
                                        r"'<name>'\}\), or the same values to pair them$"):
        two({"sweep": [1, 0]})
    assert len(points(two({"sweep": [1, 0], "label": "mlp"}).steps["f"])) == 4


def test_m2_a_metric_over_two_reads_without_a_static_width_says_to_read_at_a_static_index():
    with pytest.raises(PlanError, match=r"steps\.kl\.against: 'f\.logits' holds a different number of positions on each "
                                        r"row, which pair with no other value's; read both at a static index"):
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


def test_m4_an_operand_without_a_static_width_into_such_a_write_says_what_differs():
    """trial-01, trial-02: the two layouts print the same, so the message says the difference is where the
    positions come from."""
    with pytest.raises(PlanError, match=r"with: 'abc\.v' holds a different number of positions on each row, which "
                                        r"pair with no other value's; read both at a static index"):
        plan(data={"d": PAIR}, steps={
            "abc": {"kind": "forward", "input": "d.abc", "interventions": {"v": {"site": LAYER0, "pos": {"index": ":"}}}},
            "patched": forward({"w": {"site": LAYER0, "pos": {"index": ":"},
                                      "ops": [{"op": "swap", "with": "abc.v"}]}})}).compile()


def test_m5_o2_argmax_ranks_the_axis_over_names_and_one_the_value_lacks_is_refused():
    """trial-04 r01: `argmax` over an encode's features said it ranks a vocabulary only; `over` names the axis it
    ranks, `vocab_size` by default, and one the value lacks is refused by name."""
    def top(**over) -> Plan:
        return plan(data={"d": DAYS}, featurizers={"rot": {"kind": "subspace", "k": 4}}, steps={
            "f": forward({"z": {"site": LAYER0, "pos": -1, "ops": [{"op": "encode", "featurizer": "rot"}]}}),
            "top": {"kind": "op", "op": "argmax", "of": "f.z", **over}})
    with pytest.raises(PlanError, match=r"^steps\.top: it ranks the value's 'vocab_size' axis, and the value is laid "
                                        r"out \['rows', 'pos', 'features'\]; 'over' names the axis it ranks, one the "
                                        r"value has \(\"over\": \"features\"\)$"):
        top().compile()
    with pytest.raises(PlanError, match=r"^steps\.top: it ranks the value's 'classes' axis"):
        top(over="classes").compile()
    assert dims(top(over="features").compile(), "top") == ["rows", "pos"]


def test_m6_take_on_a_file_keeps_its_first_rows():
    """trial-03: `take` on a path dataset said it belonged to hub datasets; it is an option of any dataset."""
    compiled = plan(data={"d": {"path": "weekdays/data", "take": 2}}, steps={}).compile()
    assert compiled.data["d"].n_rows == 2
    with pytest.raises(PlanError, match=r"data\.d: unknown field 'repeat'; valid: path, rows, hf, revision, config, "
                                        r"split, shuffle, take, columns, template$"):
        plan(data={"d": {"path": "weekdays/data", "repeat": 2}}, steps={})


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


def test_m9_a_rows_free_operand_with_a_layers_axis_names_the_layer_it_lacks(engines):
    """trial-07 E6b: a mean that keeps a layers axis has no rows, so it crosses datasets; a one-layer write takes
    its slice labelled with the write's layer, and one at a layer the axis does not hold is refused when the plan
    compiles, naming that layer (not "a row per example")."""
    def function_vector(read_at):
        return plan(data={"d": DAYS, "t": {"rows": DAYS["rows"][:2]}}, steps={
            "icl": forward({"a": {"site": {"component": "attention_premix", "layers": read_at}, "pos": -1}}),
            "m": {"kind": "op", "op": "mean", "of": "icl.a", "over": ["rows", "pos"]},
            "zs": {"kind": "forward", "input": "t.input", "interventions": {"w": {
                "site": {"component": "attention_premix", "layers": 1}, "pos": -1,
                "ops": [{"op": "swap", "with": "m"}]}}}}, saves=["m", "zs.w"]).compile()
    written = run(engines, function_vector([0, 1]))
    assert torch.equal(written.results["zs.w"], written.results["m"][1].expand(2, 1, 4, 4))
    with pytest.raises(PlanError, match=r"^steps\.zs\.interventions\.w\.ops\[0\]\.with: 'm' holds layers \[0\], and "
                                        r"this op acts at layers \[1\]; at each layer it takes the operand's slice of "
                                        r"that layer, and it holds none of \[1\]: read it at those layers too$"):
        function_vector([0])


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
