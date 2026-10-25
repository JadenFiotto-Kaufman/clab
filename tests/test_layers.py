"""A write across layers and a sweep's label, on tiny GPT-2 (five layers) and the tiny Llama: a write's `layers` is a
selector, one write per picked layer in one forward, exactly the hand-unrolled entries; an operand's `layers` axis
aligns to the layer an op acts at by label; one label is one sweep axis. Every run is on both engines, which agree
to the bit (`both`)."""

import json
import os
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, DOCUMENTS, TRIP, edited, same, stand_in
from safetensors import safe_open

from clab import Plan
from clab.format.sweeps import points
from clab.plan import PlanError, Sweep, dims_of

GPT2 = {"key": "hf-internal-testing/tiny-random-gpt2", "revision": "71034c5d8bde858ff824298bdedc65515b97d2b9"}
TINY = {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998"}
PAIRS = {"pairs": {"path": "counting/train"}}
CF = "pairs.counterfactual_inputs[0]"
LOGITS = {"logits": {"site": {"component": "logits"}, "pos": -1}}
KINDS = ("nnterp", "hooks")


def plan(steps: dict, saves: list[str], model: dict = GPT2, **more) -> Plan:
    """A document over the counting pairs, on tiny GPT-2 unless `model` says, read as if from documents/."""
    return Plan(model=model, data=more.pop("data", PAIRS), steps=steps, saves=saves,
                document=os.path.join(DOCUMENTS, "layers.json"), **more).compile()


def stream(layers) -> dict:
    return {"component": "layer_output", "layers": layers}


def forward(entries: dict, input: str = "pairs.input") -> dict:
    return {"kind": "forward", "input": input, "interventions": entries}


def swap(layers, operand: str, pos: Any = -1) -> dict:
    return {"site": stream(layers), "pos": pos, "ops": [{"op": "swap", "with": operand}]}


def both(engines, compiled: Plan) -> Any:
    """`compiled` run on each engine at the pinned batch size, the two agreeing to the bit in their results and
    in the positions they record: the nnterp run."""
    a, b = (engines(kind, compiled.model).execute(compiled, batch_size=BATCH_SIZE) for kind in KINDS)
    assert same(a.results, b.results) and a.run.positions == b.run.positions
    return a


def refused(engines, compiled: Plan, message: str) -> None:
    """On each engine, refused before any forward with `message`."""
    for kind in KINDS:
        engine = engines(kind, compiled.model)
        calls = []
        original = engine.forward
        engine.forward = lambda *args: calls.append(1) or original(*args)
        try:
            with pytest.raises(ValueError, match=message):
                engine.execute(compiled, batch_size=BATCH_SIZE)
        finally:
            del engine.forward
        assert not calls, kind


def clean(engines) -> torch.Tensor:
    return both(engines, plan({"run": forward(LOGITS)}, ["run.logits"])).results["run.logits"]


# -- a write across layers -----------------------------------------------------------------------------


def test_a_window_swapped_with_its_own_read_and_nothing_added_at_every_layer_are_clean(engines):
    """`swap` at layers 1 to 3 with the same input's read there in an earlier step, and `add` of 0 times a
    value at every layer and position, change nothing, to the bit."""
    window = {"index": "-3:"}
    got = both(engines, plan({"read": forward({"h": {"site": stream("1:4"), "pos": window}}),
                              "run": forward({"w": swap("1:4", "read.h", window), **LOGITS})},
                             ["run.logits", "run.w", "read.h"])).results
    assert torch.equal(got["run.logits"], clean(engines)) and torch.equal(got["run.w"], got["read.h"])
    nothing = plan({"run": forward({"w": {"site": stream(":"), "pos": {"index": ":"},
                                          "ops": [{"op": "add", "with": 1.0, "scale": 0.0}]}, **LOGITS})}, ["run.logits"])
    assert torch.equal(both(engines, nothing).results["run.logits"], clean(engines))


@pytest.mark.parametrize("ops", [[{"op": "scale", "factor": 3.0}], [{"op": "noise", "scale": 0.5, "seed": 7}]],
                         ids=["scale", "noise"])
def test_one_entry_at_two_layers_is_two_entries_of_one_layer_each(engines, ops):
    """At `[1, 3]` the entry's ops run at layer 1, then at layer 3 on what layer 1's write made of it, exactly as
    an entry at 1 then one at 3 do; `noise` draws the same bits at each layer, as two entries with one seed do."""
    def written(*layers) -> Plan:
        return plan({"run": forward({**{f"w{i}": {"site": stream(layer), "pos": {"index": ":"}, "ops": ops}
                                        for i, layer in enumerate(layers)}, **LOGITS})}, ["run.logits"])
    one, two = (both(engines, written(*layers)).results["run.logits"] for layers in (([1, 3],), (1, 3)))
    assert torch.equal(one, two) and not torch.equal(one, clean(engines))
    if ops[0]["op"] == "noise":
        drawn = both(engines, plan({"run": forward({"n": {"site": stream([0, 3]), "pos": -1, "ops": [
            {"op": "swap", "with": 0.0}, *ops]}})}, ["run.n"])).results["run.n"]
        assert torch.equal(drawn[0], drawn[1]) and drawn.abs().sum() > 0


def unrolled(layers: list[int], read_input: str = CF, **more) -> Plan:
    """The hand-unrolled form of a swap at `layers` from the read of `read_input`: one read and one write per
    layer, each write given its own layer's read."""
    return plan({"cf": forward({f"h{layer}": {"site": stream(layer), "pos": -1} for layer in layers}, read_input),
                 "run": forward({**{f"w{layer}": swap(layer, f"cf.h{layer}") for layer in layers}, **LOGITS})},
                ["run.logits", *[f"run.w{layer}" for layer in layers]], **more)


def test_a_stacked_operand_gives_each_layer_its_own_slice(engines):
    """A swap at `[1, 3]` from one read of the counterfactual at every layer (`"0:5"`) is two writes each given
    its own layer's read, to the bit; at `[4, 1]`, reversed and a subset, layer 4 still gets slice 4."""
    for layers in ([1, 3], [4, 1]):
        ranged = both(engines, plan({"cf": forward({"h": {"site": stream("0:5"), "pos": -1}}, CF),
                                     "run": forward({"w": swap(layers, "cf.h"), **LOGITS})},
                                    ["run.logits", "run.w", "cf.h"])).results
        hand = both(engines, unrolled(layers)).results
        assert torch.equal(ranged["run.logits"], hand["run.logits"]), layers
        assert torch.equal(ranged["run.w"], torch.stack([hand[f"run.w{layer}"] for layer in layers])), layers
        assert torch.equal(ranged["run.w"][0], ranged["cf.h"][layers[0]]), layers      # by label, not by index
        assert not torch.equal(ranged["cf.h"][layers[0]], ranged["cf.h"][layers[1]])
    one = both(engines, plan({"cf": forward({"h": {"site": stream("0:5"), "pos": -1}}, CF),
                              "run": forward({"w": swap(3, "cf.h"), **LOGITS})}, ["run.logits"])).results
    assert torch.equal(one["run.logits"], both(engines, unrolled([3])).results["run.logits"])


def test_an_operand_of_the_same_forward_is_its_read_at_each_layer(engines):
    """Every block skipped: at each layer the output is swapped for that layer's input, read by an earlier entry
    of the same forward at every layer, exactly as the ten one-layer entries do; read where the forward reaches it
    only after the write (the output's read into the input's write), it is refused naming the first such layer."""
    every = plan({"run": forward({"h": {"site": {"component": "layer_input", "layers": ":"}, "pos": -1},
                                  "w": swap(":", "run.h"), **LOGITS})}, ["run.logits", "run.w", "run.h"])
    one_each = {}
    for layer in range(5):
        one_each[f"h{layer}"] = {"site": {"component": "layer_input", "layers": layer}, "pos": -1}
        one_each[f"w{layer}"] = swap(layer, f"run.h{layer}")
    hand = both(engines, plan({"run": forward({**one_each, **LOGITS})}, ["run.logits"])).results
    got = both(engines, every).results
    assert torch.equal(got["run.logits"], hand["run.logits"]) and torch.equal(got["run.w"], got["run.h"])
    assert not torch.equal(got["run.logits"], clean(engines))
    late = plan({"run": forward({"h": {"site": stream(":"), "pos": -1},
                                 "w": {**swap(":", "run.h"), "site": {"component": "layer_input", "layers": ":"}}})},
                ["run.w"])
    refused(engines, late, r"^run\.w: operand 'run\.h' is read at layer_output layer 0, which the \w+ engine's forward "
                           r"reaches no earlier than this write's layer_input layer 0; an operand from the write's own "
                           r"forward is an earlier entry at this site, or at a site the forward reaches first$")


def test_a_layer_the_operand_does_not_hold_is_refused_by_name_before_any_forward(engines):
    for operand, entries in ((["cf"], {"w": swap("0:3", "cf.h")}), ([], {"h": {"site": stream([0, 2]), "pos": -1},
                                                                         "w": swap("0:3", "run.h")})):
        steps = {"cf": forward({"h": {"site": stream([0, 2]), "pos": -1}}, CF)} if operand else {}
        name = "cf.h" if operand else "run.h"
        refused(engines, plan({**steps, "run": forward(entries)}, ["run.w"]),
                rf"^run\.w: operand '{name.replace('.', chr(92) + '.')}' holds layers \[0, 2\], and this entry is made "
                r"at layers \[0, 1, 2\]; at each layer an op takes the operand's slice of that layer, and it holds none "
                r"of \[1\]: read it at those layers too$")
    with pytest.raises(PlanError, match=r"^steps\.run\.interventions\.w\.site\.layers: '0:0' picks nothing, however "
                                        r"many things there are$"):
        plan({"run": forward({"w": {"site": stream("0:0"), "pos": -1, "ops": [{"op": "scale", "factor": 2.0}]}})}, [])
    with pytest.raises(PlanError, match=r"^steps\.run\.interventions\.w\.ops\[0\]\.with: 'cf\.h' is laid out "
                                        r"\['layers', 'rows', 'pos', 'hidden_size'\]: at each layer an op takes an "
                                        r"operand's slice of that layer, and logits is the model's own value, at no "
                                        r"layer"):
        plan({"cf": forward({"h": {"site": stream(":"), "pos": -1}}, CF),
              "run": forward({"w": {"site": {"component": "logits"}, "pos": -1,
                                    "ops": [{"op": "add", "with": "cf.h"}]}})}, [])


def test_a_write_across_layers_binds_a_layers_axis_that_its_table_and_positions_say(engines, tmp_path):
    """The value `<step>.<name>` has a `layers` axis first, labelled with the layers picked, in their order;
    `table` says each row's `layer`, after `Plan.load` too; `run.positions` records the write's landing once, as
    a read's; and the ops after its last edit see the axis: `[add 0, readout, token_prob]` at every layer is
    the logit lens, a probability per layer."""
    compiled = plan({"run": forward({"w": {"site": stream([4, 1]), "pos": {"index": "-2:"},
                                           "ops": [{"op": "scale", "factor": 2.0}]},
                                     "r": {"site": stream(0), "pos": {"index": "-2:"}}})}, ["run.w", "run.r"])
    got = both(engines, compiled)
    assert dims_of(compiled, "run.w") == ["layers", "rows", "pos", "hidden_size"]
    assert got.results["run.w"].shape == (2, 4, 2, 32)
    rows = got.table("run.w")
    assert [r["layer"] for r in rows] == [4] * 8 + [1] * 8 and rows[0]["pos"] == 10
    assert got.run.positions["run.w"] == got.run.positions["run.r"] == [[10, 11]] * 4
    got.dump(str(tmp_path))
    assert Plan.load(str(tmp_path)).table("run.w") == rows
    lens = [{"op": "readout"}, {"op": "token_prob", "token": "pairs.cf_answer"}]
    written = plan({"run": forward({"p": {"site": stream(":"), "pos": -1,
                                          "ops": [{"op": "add", "with": 1.0, "scale": 0.0}, *lens]}})}, ["run.p"])
    read = plan({"run": forward({"p": {"site": stream(":"), "pos": -1, "ops": lens}})}, ["run.p"])
    p = both(engines, written).results["run.p"]
    assert p.shape == (5, 4, 1) and torch.equal(p, both(engines, read).results["run.p"])


def test_a_window_sweep_on_a_write_is_one_window_per_point(engines):
    """`{"sweep": ["0:2", "2:5"]}` on a write's layers: two points, each the unrolled document of its window."""
    swept = both(engines, plan({"cf": forward({"h": {"site": stream(":"), "pos": -1}}, CF),
                                "run": forward({"w": swap({"sweep": ["0:2", "2:5"]}, "cf.h"), **LOGITS})},
                               ["run.logits"])).results["run.logits"]
    assert list(swept) == ["0:2", "2:5"]
    for window, layers in (("0:2", [0, 1]), ("2:5", [2, 3, 4])):
        assert torch.equal(swept[window], both(engines, unrolled(layers)).results["run.logits"]), window


def test_a_generation_writes_at_each_layer_in_the_prefill_and_every_decode_call(engines):
    """In a generation a write across layers lands at each layer in the prefill and at each decode call, as one
    entry per layer does."""
    days = {"d": {"rows": [{"input": "If today is Monday, tomorrow is"}, {"input": "one two three"}]}}

    def generated(*layers) -> Plan:
        entries = {f"w{i}": {"site": stream(layer), "pos": {"frame": ":"}, "ops": [{"op": "scale", "factor": 4.0}]}
                   for i, layer in enumerate(layers)}
        return plan({"g": {"kind": "generate", "input": "d.input", "max_new_tokens": 4, "do_sample": False,
                           "interventions": {**entries, "lens": {"site": {"component": "logits"},
                                                                 "pos": {"frame": "1:", "index": ":"}}}}},
                    ["g", "g.lens"], model=TINY, data=days)
    one, two = (both(engines, generated(*layers)).results for layers in (([1, 0],), (1, 0)))
    assert same(one, two)
    plain = both(engines, generated()).results
    assert not torch.equal(one["g.lens"].nan_to_num(), plain["g.lens"].nan_to_num())


def test_a_bundle_made_at_one_layer_is_refused_at_a_site_that_picks_another(engines, corpus):
    """pca_control's basis says layer 0: a site that picks layers 0 and 1 is refused before any forward, naming
    layer 1; one that picks layer 0 alone runs as `layers: 0` does."""
    path = os.path.join(corpus, "documents", "pca_control.json")

    def at(layers) -> Plan:
        with open(path) as f:
            doc = json.load(f)
        doc["sites"]["target"]["layers"] = layers
        return Plan(**doc, document=path).compile()
    refused(engines, at(":"), r"^counterfactual\.v_cf: documents/artifacts/basis\.safetensors: header 'layer' is '0'; "
                              r"this document uses 'rot' at layer_output layer ':', which picks layer 1$")
    assert same(both(engines, at([0])).results, both(engines, at(0)).results)


def test_direction_ablation_is_its_five_one_layer_entries_to_the_bit(engines, monkeypatch):
    """documents/direction_ablation.json's one entry at every layer, written out as one entry per layer, each
    with the same ops: the same numbers, its value the five stacked; and through the stand-in server too."""
    def one_layer_each(d):
        whole = d["steps"]["ablated"]["interventions"]
        entry = whole.pop("ablate")
        each = {f"ablate{layer}": {**entry, "site": stream(layer)} for layer in range(5)}
        d["steps"]["ablated"]["interventions"] = {**each, **whole}
        d["saves"] = ["clean_ld", "ablated_ld", *[f"ablated.{key}" for key in each]]
    document = Plan(os.path.join(DOCUMENTS, "direction_ablation.json")).compile()
    got = both(engines, document).results
    hand = both(engines, edited("direction_ablation", one_layer_each).compile()).results
    assert same(got["ablated_ld"], hand["ablated_ld"]) and not same(got["ablated_ld"], got["clean_ld"])
    assert same(got["ablated.ablate"], torch.stack([hand[f"ablated.ablate{layer}"] for layer in range(5)]))
    assert got["ablated.ablate"].abs().max() < 1e-5
    came, trip = stand_in(engines, document, monkeypatch)
    assert trip == TRIP and same(came.results, got)


# -- sweep labels --------------------------------------------------------------------------------------


def test_two_labelled_sweeps_cross_into_a_grid_keyed_by_label(engines, tmp_path):
    """A read's layers labelled `src` and a write's labelled `dst`: a 3×3 grid in one step, keyed `src=…,dst=…`
    in the output and `at` in a table; each point the unswept document."""
    def grid(src, dst) -> Plan:
        return plan({"cf": forward({"h": {"site": stream(src), "pos": -1}}, CF),
                     "run": forward({"w": swap(dst, "cf.h"), **LOGITS}),
                     "ld": {"kind": "op", "op": "logit_diff", "of": "run.logits", "a": "pairs.cf_answer",
                            "b": "pairs.base_answer"}}, ["ld"])
    swept = grid({"sweep": [0, 1, 2], "label": "src"}, {"sweep": [0, 1, 2], "label": "dst"})
    assert list(swept.sweeps) == ["src", "dst"]
    assert swept.sweeps["dst"].paths == ["steps.run.interventions.w.site.layers"]
    got = both(engines, swept)
    assert list(got.results["ld"]) == [(s, d) for s in range(3) for d in range(3)]
    assert torch.equal(got.results["ld"][2, 0], both(engines, grid(2, 0)).results["ld"])
    got.dump(str(tmp_path))
    with safe_open(str(tmp_path / "ld.safetensors"), "pt") as f:
        assert sorted(f.keys()) == sorted(f"src={s},dst={d}" for s in range(3) for d in range(3))
    assert [r["at"] for r in Plan.load(str(tmp_path)).table("ld")] == \
        [{"src": s, "dst": d} for s in range(3) for d in range(3) for _ in range(4)]


def test_two_sweeps_of_one_label_and_values_are_one_axis(engines):
    """A read's and a write's `layers`, each `{"sweep": [0, 1, 2]}`: one label, `layers`, so one axis whose
    points pair up, the write at layer L given the read at L; the axis lists both paths."""
    zipped = plan({"cf": forward({"h": {"site": stream({"sweep": [0, 1, 2]}), "pos": -1}}, CF),
                   "run": forward({"w": swap({"sweep": [0, 1, 2]}, "cf.h"), **LOGITS})}, ["run.logits"])
    assert zipped.sweeps["layers"].paths == ["steps.cf.interventions.h.site.layers",
                                             "steps.run.interventions.w.site.layers"]
    got = both(engines, zipped).results["run.logits"]
    assert list(got) == [0, 1, 2]
    for layer in range(3):
        assert torch.equal(got[layer], both(engines, unrolled([layer])).results["run.logits"]), layer


def test_one_label_over_other_values_is_refused_naming_label():
    def two(a: Any, b: Any) -> Plan:
        return plan({"cf": forward({"h": {"site": stream(a), "pos": -1}}, CF),
                     "run": forward({"w": swap(b, "cf.h")})}, [])
    with pytest.raises(PlanError, match=r"^steps\.run\.interventions\.w\.site\.layers: label 'at' is also the sweep at "
                                        r"steps\.cf\.interventions\.h\.site\.layers, over \[0, 1\]; one label is one "
                                        r"axis, whose sweeps pair their points: give one of them its own label to "
                                        r"cross them"):
        two({"sweep": [0, 1], "label": "at"}, {"sweep": [1, 2], "label": "at"})
    with pytest.raises(PlanError, match=r"^steps\.cf\.interventions\.h\.site\.layers\.label: 'a\.b': a sweep's label is "
                                        r"a name, with no dots"):
        two({"sweep": [0, 1], "label": "a.b"}, 0)
    assert len(points(two({"sweep": [0, 1], "label": "at"}, {"sweep": [0, 1], "label": "at"}).steps["run"])) == 2


def test_a_sweep_over_the_string_sweep_compiles():
    """A sweep's values may hold the string "sweep" (here, the name of a step); a value that is itself a sweep is
    refused."""
    assert Sweep(sweep=["sweep", "other"]).sweep == ["sweep", "other"]
    steps = {"f": forward({"h": {"site": stream(0), "pos": -1}}),
             "sweep": {"kind": "op", "op": "mean", "of": "f.h"}, "other": {"kind": "op", "op": "sum", "of": "f.h"},
             "sim": {"kind": "op", "op": "cosine", "of": "f.h", "with": {"sweep": ["sweep", "other"]}}}
    assert plan(steps, ["sim"]).sweeps["with"].values == ["sweep", "other"]
    with pytest.raises(PlanError, match=r"^steps\.f\.interventions\.h\.site\.layers\.sweep: a sweep's values hold "
                                        r"no sweep$"):
        plan({"f": forward({"h": {"site": stream({"sweep": [{"sweep": [0, 1]}]}), "pos": -1}})}, [])


def test_a_step_named_as_a_plans_derived_field_is_swept():
    """`run`, `results` and `sweeps` are fields of a plan's root, which hold no sweep; a step so named is swept
    like any other."""
    for name in ("run", "results", "sweeps"):
        swept = plan({name: forward({"h": {"site": stream({"sweep": [0, 1]}), "pos": -1}})}, [])
        assert len(points(swept.steps[name])) == 2
