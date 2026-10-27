"""A vision-language checkpoint: a chat with image parts, the image's tokens as positions, a region of the image
as keys, a write on heads beside keys a scope names through a generation, and what is refused by name.

On two tiny checkpoints loaded with their processor (`task: image-text-to-text`): a Qwen3-VL (the Qwen ViT,
`image_grid_thw`, M-RoPE, so the model lays its own position ids out) and a Llava (a fixed-resolution CLIP
tower, 576 tokens in a 24x24 grid, 1-d positions). The images are two 64x32 halves, red|blue and blue|red."""

import json
import os
from typing import Any

import pytest
import torch
from conftest import same

from clab import Plan
from clab.engines import load
from clab.plan import PlanError

QWEN = {"key": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
        "revision": "ba190cb38011bcda2f70f7781f2b8290627aef00", "dtype": "float32", "task": "image-text-to-text",
        "attn_implementation": "eager"}
LLAVA = {"key": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
         "revision": "cd71a22c800134d15563aba02c2ab4d6368fbb9b", "dtype": "float32", "task": "image-text-to-text",
         "attn_implementation": "eager"}
MODELS = {"qwen": QWEN, "llava": LLAVA}
# Image tokens per image, and the grid they tile it in, on each: Qwen's 64x32 image makes a 6x12 merged grid.
TOKENS = {"qwen": (72, (6, 12)), "llava": (576, (24, 24))}


def chat(image: str, text: str, first: str = "image") -> list[dict]:
    parts = [{"type": "image", "image": image}, {"type": "text", "text": text}]
    return [{"role": "user", "content": parts if first == "image" else parts[::-1]}]


ROWS = [
    {"chat": chat("images/rb.png", "What colour is on the left? Answer briefly."), "left": [0, 0, 0.5, 1], "answer": "red"},
    {"chat": chat("images/br.png", "What colour is on the left side of this image?"), "left": [0, 0, 0.5, 1], "answer": "blue"},
    {"chat": chat("images/rb.png", "Describe", "text"), "left": [0, 0, 0.5, 1], "answer": "red"},
]
LEFT = {"scope": {"image": 0, "region": "d.left"}}
RIGHT = {"scope": {"image": 0, "region": [0, 0, 0.5, 1], "outside": True}}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> str:
    """A corpus with the two images under data/images and the rows beside them."""
    from PIL import Image, ImageDraw

    root = tmp_path_factory.mktemp("vision")
    os.makedirs(root / "data" / "images")
    os.makedirs(root / "documents")
    for name, colours in (("rb", ("red", "blue")), ("br", ("blue", "red"))):
        image = Image.new("RGB", (64, 32), colours[0])
        ImageDraw.Draw(image).rectangle([32, 0, 63, 31], fill=colours[1])
        image.save(root / "data" / "images" / f"{name}.png")
    with open(root / "data" / "images" / "rows.json", "w") as f:
        json.dump(ROWS, f)
    return str(root)


def document(model: dict, interior: bool) -> dict:
    """The test document: image and turn scopes at a block's output, the next-token probability of the
    answer, a plain generation; with the attention interior (which the hooks engine does not serve), the
    pattern at the last token whole, over the left half's keys and over the right half's, and a generation
    with heads 0 and 2 of block 1 biased by +10000 toward the left half at every call, beside its null."""
    sites: dict[str, Any] = {"resid": {"component": "layer_output", "layers": 0}, "head": {"component": "logits"}}
    clean = {"image": {"site": "resid", "pos": {"scope": {"image": 0}}},
             "turn": {"site": "resid", "pos": {"scope": {"turn": "user"}}},
             "logits": {"site": "head", "pos": -1}}
    steps: dict[str, Any] = {"clean": {"kind": "forward", "input": "d.chat", "interventions": clean},
                             "plain": {"kind": "generate", "input": "d.chat", "max_new_tokens": 3, "do_sample": False},
                             "p": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "d.answer"}}
    saves = ["clean.image", "clean.turn", "plain", "p"]
    if interior:
        sites.update({
            "pattern": {"component": "attention_probabilities", "layers": 1},
            "left_keys": {"component": "attention_probabilities", "layers": 1, "select": {"key": LEFT}},
            "right_keys": {"component": "attention_probabilities", "layers": 1, "select": {"key": RIGHT}},
            "scores": {"component": "attention_scores", "layers": 1, "select": {"num_heads": [0, 2], "key": LEFT}}})
        clean.update({"pattern": {"site": "pattern", "pos": -1},
                      "left": {"site": "left_keys", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]},
                      "right": {"site": "right_keys", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]}})
        steps.update({
            "steered": {"kind": "generate", "input": "d.chat", "max_new_tokens": 3, "do_sample": False,
                        "interventions": {
                            "bias": {"site": "scores", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": 10000.0}]},
                            "pattern": {"site": "pattern", "pos": -1}}},
            "null": {"kind": "generate", "input": "d.chat", "max_new_tokens": 3, "do_sample": False,
                     "interventions": {
                         "bias": {"site": "scores", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": 0.0}]}}}})
        saves += ["clean.pattern", "clean.left", "clean.right", "steered", "steered.pattern", "null"]
    return {"description": "vision", "model": model,
            "data": {"d": {"path": "images/rows", "template": {"add_generation_prompt": True}}},
            "sites": sites, "steps": steps, "saves": saves}


def written(corpus: str, name: str, doc: dict) -> str:
    path = os.path.join(corpus, "documents", f"{name}.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


@pytest.fixture(scope="module")
def filled(engines, corpus) -> dict:
    """Each model's document run on nnterp at batch sizes 2 and 1 and, without the attention interior, on hooks."""
    out = {}
    for name, model in MODELS.items():
        for kind, interior in (("nnterp", True), ("hooks", False)):
            path = written(corpus, f"{name}_{kind}", document(model, interior))
            for bs in (2, 1):
                out[name, kind, bs] = engines(kind, Plan(path).model).execute(Plan(path).compile(), batch_size=bs)
    return out


@pytest.mark.parametrize("name", MODELS)
def test_an_images_tokens_are_positions_and_a_turn_holds_its_image(filled, name):
    """An image scope is the image's tokens, where the processor put them (after the template's header on
    the first two rows, after the word on the third); the user turn holds them with its words."""
    plan = filled[name, "nnterp", 2]
    n, _ = TOKENS[name]
    image = plan.run.positions["clean.image"]
    assert [len(p) for p in image] == [n, n, n]
    assert image[0] == image[1]
    # Row 2 writes the text part first: Qwen's template keeps the order, Llava's puts the image first regardless.
    assert image[2][0] > image[0][0] if name == "qwen" else image[2] == image[0]
    assert plan.results["clean.image"].shape == (3, n, 16)
    tokens = plan.run.tokens["d.chat"]
    assert all(tokens[r][p] == tokens[0][image[0][0]] for r in range(3) for p in image[r])
    for r, turn in enumerate(plan.run.positions["clean.turn"]):
        assert set(image[r]) < set(turn) and len(turn) > n


@pytest.mark.parametrize("name", MODELS)
def test_a_region_names_the_keys_of_half_the_grid(filled, name):
    """The left half's keys and the right half's (the same box, `outside`) split the image's tokens in two, in
    raster order over the grid, and their masses sum to the pattern's mass on the whole image."""
    plan = filled[name, "nnterp", 2]
    n, (rows, cols) = TOKENS[name]
    left, right, image = plan.run.keys["clean.left"], plan.run.keys["clean.right"], plan.run.positions["clean.image"]
    for r in range(3):
        start = image[r][0]
        assert left[r] == [start + i * cols + c for i in range(rows) for c in range(cols // 2)]
        assert right[r] == [start + i * cols + c for i in range(rows) for c in range(cols // 2, cols)]
        whole = plan.results["clean.pattern"][r, 0, :, image[r]].nansum(-1)
        torch.testing.assert_close(plan.results["clean.left"][r, 0] + plan.results["clean.right"][r, 0], whole,
                                   atol=1e-6, rtol=0)
    assert plan.results["clean.left"].shape == (3, 1, 4)


@pytest.mark.parametrize("name", MODELS)
def test_a_bias_on_heads_beside_scoped_keys_moves_their_whole_mass(filled, name):
    """+10000 on heads 0 and 2 of block 1 toward the left half's keys puts all of their mass there (the other
    heads keep theirs), at the prefill as at every decode call; by 0, the generation is the plain one."""
    plan = filled[name, "nnterp", 2]
    pattern, left = plan.results["steered.pattern"], plan.run.keys["clean.left"]
    for r in range(3):
        mass = pattern[r, 0, :, left[r]].sum(-1)
        torch.testing.assert_close(mass[[0, 2]], torch.ones(2), atol=1e-6, rtol=0)
        assert (mass[[1, 3]] < 0.95).all()
    assert torch.equal(plan.results["null"], plan.results["plain"])
    assert not torch.equal(plan.results["steered"], plan.results["plain"])


@pytest.mark.parametrize("name", MODELS)
def test_the_numbers_do_not_depend_on_the_batch_and_the_engines_agree(filled, name):
    """Left-padded rows carry the processor's per-token tensors and, on a model that lays its own position
    ids out over the image (Qwen), none of clab's: batch sizes 2 and 1 agree to the bit on each engine, and the
    engines on what both serve."""
    for kind in ("nnterp", "hooks"):
        assert same(filled[name, kind, 2].results, filled[name, kind, 1].results)
    shared = filled[name, "hooks", 2].results
    assert set(shared) == {"clean.image", "clean.turn", "plain", "p"}
    assert same({k: filled[name, "nnterp", 2].results[k] for k in shared}, shared)
    assert filled[name, "nnterp", 2].run.positions["clean.image"] == filled[name, "hooks", 2].run.positions["clean.image"]


def test_the_table_labels_scoped_keys_with_their_tokens(filled, corpus, monkeypatch):
    """`clab table` reads the rows from `data/` under the working directory: the corpus's."""
    plan = filled["qwen", "nnterp", 2]
    plan.dump(os.path.join(corpus, "out"))
    monkeypatch.chdir(corpus)
    back = Plan.load("out")
    rows = back.table("clean.pattern")
    keys = {(row["row"], row["key"]) for row in rows}
    assert all(row["key_token"] == plan.run.tokens["d.chat"][row["row"]][row["key"]] for row in rows)
    assert len(keys) == sum(len(t) for t in plan.run.tokens["d.chat"])        # the rows' own keys, each once


# -- refusals -------------------------------------------------------------------------------------------


def doc_with(corpus: str, change) -> Plan:
    doc = document(QWEN, True)
    change(doc)
    return Plan(written(corpus, "refused", doc))


@pytest.mark.parametrize("scope, message", [
    ({"image": 0, "region": [0, 0, 2, 1]}, r"region \[0, 0, 2, 1\]: a box is \[x0, y0, x1, y1\] with 0 <= x0 < x1 <= 1"),
    ({"image": 0, "outside": True}, r"'outside' flips a region to the image's tokens outside it, and the scope gives no region"),
    ({"text": "a", "region": [0, 0, 1, 1]}, r"'region' and 'outside' narrow an image scope"),
    ({"image": 0, "occurrence": 1}, r"an image scope's occurrences are the row's images, which 'image' picks among"),
    ({"image": 0, "strip": True}, r"an image's tokens are no text; an image scope takes no strip"),
])
def test_the_scope_grammar_is_refused_by_name(corpus, scope, message):
    def change(d):
        d["steps"]["clean"]["interventions"]["image"]["pos"] = {"scope": scope}
    with pytest.raises(PlanError, match=message):
        doc_with(corpus, change)


def test_an_image_scope_beyond_the_prefill_is_refused(corpus):
    def change(d):
        d["steps"]["steered"]["interventions"]["bias"]["pos"] = {"frame": 1, "scope": {"image": 0}}
    with pytest.raises(PlanError, match=r"scope image 0 names the prompt's image tokens; it takes frame 0"):
        doc_with(corpus, change)


def test_a_scoped_key_write_takes_a_number(corpus):
    def change(d):
        d["steps"]["clean"]["interventions"]["bias"] = {"site": "scores", "pos": -1,
                                                        "ops": [{"op": "add", "with": "clean.logits"}]}
    with pytest.raises(PlanError, match=r"this write's keys are named by a scope, a different number of them on "
                                        r"each row; such a write takes a number as its operand"):
        doc_with(corpus, change).compile()


def test_an_image_scope_on_a_text_input_is_refused(corpus):
    def change(d):
        d["data"]["d"] = {"rows": [{"text": "one two"}]}
        d["steps"] = {"clean": {"kind": "forward", "input": "d.text",
                                "interventions": {"image": {"site": "resid", "pos": {"scope": {"image": 0}}}}}}
        d["saves"] = ["clean.image"]
    with pytest.raises(PlanError, match=r"the input d\.text is text, not a chat: an image comes as a part of a chat "
                                        r"message's content"):
        doc_with(corpus, change).compile()


def test_a_region_column_holds_boxes_on_every_row(corpus):
    def change(d):
        d["sites"]["left_keys"]["select"]["key"] = {"scope": {"image": 0, "region": "d.answer"}}
    with pytest.raises(PlanError, match=r"select\.key\.scope\.region: 'd\.answer' on row 0 of 'd': 'red': a region "
                                        r"is a box"):
        doc_with(corpus, change).compile()


def test_an_image_part_needs_the_processor(engines, corpus):
    """The text-only load of the wrapper (no `task`) has no processor to put the image in."""
    def change(d):
        d["model"] = {k: v for k, v in QWEN.items() if k != "task"}
    plan = doc_with(corpus, change).compile()
    with pytest.raises(ValueError, match=r"clean: d\.chat row 0: the chat has an image part, and this model loaded "
                                         r"no processor; load the checkpoint with 'task': 'image-text-to-text'"):
        engines("nnterp", plan.model).execute(plan, batch_size=2)


def test_a_region_that_keeps_no_token_is_refused(engines, corpus):
    def change(d):
        d["sites"]["left_keys"]["select"]["key"] = {"scope": {"image": 0, "region": [0, 0, 0.01, 0.01]}}
    plan = doc_with(corpus, change).compile()
    with pytest.raises(ValueError, match=r"clean\.left: select key: row 0: region \[\[0\.0, 0\.0, 0\.01, 0\.01\]\] "
                                         r"keeps no token of image 0, whose 72 tokens tile it 6 rows by 12 columns"):
        engines("nnterp", plan.model).execute(plan, batch_size=2)


def test_a_missing_image_file_is_refused(engines, corpus):
    def change(d):
        d["data"]["d"] = {"rows": [{"chat": chat("images/none.png", "Hi"), "left": [0, 0, 0.5, 1], "answer": "red"}],
                          "template": {"add_generation_prompt": True}}
    plan = doc_with(corpus, change).compile()
    with pytest.raises(ValueError, match=r"clean: d\.chat row 0: image part 0: no file data/images/none\.png under "
                                         r"the corpus root"):
        engines("nnterp", plan.model).execute(plan, batch_size=2)


def test_the_ndif_engine_refuses_an_image_part(engines, corpus):
    plan = doc_with(corpus, lambda d: None).compile()
    with pytest.raises(ValueError, match=r"clean: d\.chat row 0 has an image part, a file under the corpus root's "
                                         r"data/, which the ndif engine's server cannot read"):
        engines("ndif", plan.model, remote="local").execute(plan, batch_size=2)


# -- what the review found ------------------------------------------------------------------------------


def test_aligned_merges_an_inserted_run_and_keeps_a_repeated_text_run():
    """An image's tokens (one id, put in by the processor) all take the placeholder's span whichever of them the
    alignment matched; repeated digits or newlines the plain tokenizer has too keep their own spans."""
    from clab.format.data import aligned

    plain, offsets = [7, 9, 9, 9, 5], [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8)]
    assert aligned(plain, offsets, plain) == offsets
    # The processor expanded the placeholder 9 at index 1 into four 9s, and the alignment kept the last of them.
    assert aligned([7, 9, 5], [(0, 2), (2, 9), (9, 12)], [7, 9, 9, 9, 9, 5]) == \
        [(0, 2), (2, 9), (2, 9), (2, 9), (2, 9), (9, 12)]
    # A BOS the processor added, before anything, covers nothing.
    assert aligned([7, 5], [(0, 2), (2, 4)], [1, 7, 5]) == [(0, 0), (0, 2), (2, 4)]


def test_a_region_may_be_a_list_of_boxes(engines, corpus):
    """Two boxes are their union: the left half and the top-right quarter of the 6 x 12 grid name 36 + 9 keys."""
    from clab.plan import Scope

    assert Scope(image=0, region=[[0, 0, 0.5, 1], [0.5, 0, 1, 0.5]]).region == [[0, 0, 0.5, 1], [0.5, 0, 1, 0.5]]

    def change(d):
        d["sites"]["left_keys"]["select"]["key"] = {"scope": {"image": 0, "region": [[0, 0, 0.5, 1], [0.5, 0, 1, 0.5]]}}
        d["steps"] = {"clean": d["steps"]["clean"]}
        d["saves"] = ["clean.left"]
    plan = doc_with(corpus, change).compile()
    filled = engines("nnterp", plan.model).execute(plan, batch_size=2)
    rows, cols = TOKENS["qwen"][1]
    start = filled.run.positions["clean.image"][0][0]
    union = sorted({start + i * cols + c for i in range(rows) for c in range(cols // 2)}
                   | {start + i * cols + c for i in range(rows // 2) for c in range(cols // 2, cols)})
    assert filled.run.keys["clean.left"][0] == union and len(union) == 36 + 18


def test_scoped_keys_of_different_widths_join_across_batches(engines, corpus):
    """A region column that names 36 keys on one row and 18 on the next: read whole (no sum), at batch size 1
    each batch is as wide as its own row, and the joined value is as wide as the widest, NaN past a row's own."""
    def change(d):
        d["data"]["d"] = {"rows": [{**ROWS[0], "box": [0, 0, 0.5, 1]}, {**ROWS[1], "box": [0, 0, 0.25, 1]},
                                   {**ROWS[2], "box": [0, 0, 0.5, 1]}], "template": {"add_generation_prompt": True}}
        d["sites"]["box_keys"] = {"component": "attention_probabilities", "layers": 1,
                                  "select": {"key": {"scope": {"image": 0, "region": "d.box"}}}}
        d["steps"] = {"clean": {"kind": "forward", "input": "d.chat",
                                "interventions": {"boxed": {"site": "box_keys", "pos": -1}}}}
        d["saves"] = ["clean.boxed"]
    plan = doc_with(corpus, change).compile()
    one = engines("nnterp", plan.model).execute(plan, batch_size=1)
    two = engines("nnterp", plan.model).execute(plan, batch_size=2)
    assert [len(k) for k in one.run.keys["clean.boxed"]] == [36, 18, 36]
    value = one.results["clean.boxed"]
    assert value.shape == (3, 1, 4, 36) and value[1, ..., 18:].isnan().all() and not value[1, ..., :18].isnan().any()
    assert same(one.results, two.results)


def test_a_fit_bodys_minibatches_keep_the_scoped_keys():
    """`narrowed` hands a training minibatch the keys preflight resolved, like its positions."""
    from clab.compile.state import Found
    from clab.steps.fit import narrowed

    found = Found(inputs={"d.chat": {"rows": [{"ids": [1]}, {"ids": [2]}, {"ids": [3]}], "width": 1}},
                  positions={"f.x": {(): [[0], [1], [2]]}}, keys={"f.x": {(): [[4, 5], [6], [7, 8, 9]]}})
    plan = Plan(model=QWEN, data={"d": {"rows": [{"chat": "a"}, {"chat": "b"}, {"chat": "c"}]}}, steps={})
    _, part = narrowed(plan, found, [2, 0], ["d"])
    assert part.keys == {"f.x": {(): [[7, 8, 9], [4, 5]]}} and part.positions == {"f.x": {(): [[2], [0]]}}
