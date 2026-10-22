"""What cannot be served, or would change a document's meaning, is refused by name before any forward runs."""

import re
from typing import Callable

import pytest

from clab.plan import PlanError
from conftest import BATCH_SIZE, edited, plan_of, same

# -- the plan and compile ---------------------------------------------------------------------------------


def test_select_and_featurizer_belong_to_encode():
    def change(key, value):
        def apply(d):
            d["steps"]["patched"]["interventions"]["patch"]["ops"][0][key] = value
        return apply
    with pytest.raises(PlanError, match=r"ops\[0\]: 'swap' takes no 'select': an encode keeps some of a featurizer's "
                                        r"coordinates, \{'op': 'encode', 'featurizer': \.\.\., 'select'"):
        edited("patching", change("select", {"features": [0]}))
    with pytest.raises(PlanError, match=r"ops\[0\]: 'swap' takes no 'featurizer': an op runs in a featurizer's space "
                                        r"between an encode and a decode of it"):
        edited("patching", change("featurizer", "rot"))
    with pytest.raises(PlanError, match=r"ops\[0\]: there is no 'featurizer' op: 'encode' gives a featurizer's "
                                        r"coordinates of the value and 'apply' its version of it"):
        edited("patching", change("op", "featurizer"))


@pytest.mark.parametrize("where", ["step", "bundle", "generate"])
@pytest.mark.parametrize("key", ["reads", "writes"])
def test_reads_and_writes_are_refused_by_name(where, key):
    """`reads` and `writes` are one `interventions` value, on a step and in a root bundle alike."""
    def change(d):
        entry = {"site": "target", "pos": -1}
        if where == "bundle":
            d["interventions"] = {"b": {key: {"x": entry}}}
        elif where == "step":
            d["steps"]["patched"][key] = {"x": entry}
        else:
            d["steps"]["gen"] = {"kind": "generate", "input": "pairs.input", "max_new_tokens": 1, key: {"x": entry}}
    with pytest.raises(PlanError, match=rf"{key!r}: a step's reads and writes are one 'interventions' value, and a "
                                        r"bundle is one dict of entries"):
        edited("patching", change)


def bracketed(*listed, featurizers=None, site="target") -> Callable:
    """random_subspace.json's patched step with the patch's ops `listed`, and further `featurizers`."""
    def change(d):
        d["featurizers"].update(featurizers or {})
        d["steps"]["patched"]["interventions"]["patch"].update(ops=list(listed), site=site)
    return change


ENC, DEC = {"op": "encode", "featurizer": "rot"}, {"op": "decode", "featurizer": "rot"}
SWAP = {"op": "swap", "with": "counterfactual.v_cf"}


@pytest.mark.parametrize("ops, message", [
    ([DEC], r"patch\.ops\[0\]: decode of 'rot' closes no encode: none is open in this entry here"),
    ([ENC, SWAP, DEC, DEC], r"patch\.ops\[3\]: decode of 'rot' closes no encode"),
    ([ENC, SWAP], r"patch\.ops\[1\]: the entry writes back here with the encode of 'rot' \(ops\[0\]\) still open; "
                  r"close it with \{'op': 'decode', 'featurizer': 'rot'\}"),
    ([ENC, {"op": "encode", "featurizer": "other"}, SWAP, DEC, {"op": "decode", "featurizer": "other"}],
     r"patch\.ops\[3\]: decode of 'rot', and the open encode is of 'other' \(ops\[1\]\); a decode closes the most "
     r"recent encode not yet closed"),
    ([{"op": "readout"}, {"op": "scale", "factor": 2.0}],
     r"patch\.ops\[1\]: 'scale' is an edit, which writes back, and it follows 'readout', which turned the value into "
     r"\['rows', 'pos', 'vocab_size'\]: there is no live value to write"),
    ([ENC, {"op": "mean", "over": ["features"]}, DEC],
     r"patch\.ops\[2\]: 'decode' is an edit, which writes back, and it follows 'mean'"),
    ([{"op": "swap", "with": 0.0}, ENC, {"op": "scale", "factor": 2.0}, DEC], None),
    ([ENC, {"op": "swap", "with": "counterfactual.raw"}, DEC],
     r"patch\.ops\[1\]\.with: 'counterfactual\.raw' is laid out \['rows', 'pos', 'hidden_size'\]; this write's value "
     r"is \['rows', 'pos', 'features'\] here"),
    ([SWAP], r"patch\.ops\[0\]\.with: 'counterfactual\.v_cf' is laid out \['rows', 'pos', 'features'\]; this write's "
             r"value is \['rows', 'pos', 'hidden_size'\] here"),
    ([{"op": "encode", "featurizer": "probe"}, {"op": "swap", "with": 0.0}, {"op": "decode", "featurizer": "probe"}],
     r"patch\.ops\[2\]\.featurizer: 'probe' is a 'linear' featurizer, which has no decode: it serves encode; it has "
     r"no way back into the value"),
    ([{"op": "encode", "featurizer": "adapter"}],
     r"patch\.ops\[0\]\.featurizer: 'adapter' is a 'lora' featurizer, which has no encode: it serves apply"),
    ([{"op": "apply", "featurizer": "probe"}],
     r"patch\.ops\[0\]\.featurizer: 'probe' is a 'linear' featurizer, which has no apply: it serves encode$"),
])
def test_a_bracket_is_refused_at_compile_by_name(ops, message):
    """Within an entry each decode closes the most recent encode not yet closed, of the same featurizer, and an
    entry writes back with none open; an edit follows no op that changed the layout but an open encode; between
    the brackets an operand is laid out on the coordinates; and a featurizer serves only the ops its kind has. An
    edit before an encode, and one inside a bracket after it, are fine."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["raw"] = {"site": "target", "pos": -1}
    more = {"other": {"kind": "subspace", "k": 4, "seed": 1},
            "probe": {"kind": "linear", "classes": "pairs.cf_answer"}, "adapter": {"kind": "lora", "rank": 2, "alpha": 4}}
    edit = bracketed(*ops, featurizers=more)

    def both(d):
        change(d)
        edit(d)
    if message is None:
        edited("random_subspace", both).compile()
        return
    with pytest.raises(PlanError, match=message):
        edited("random_subspace", both).compile()


def test_decode_as_a_step_decodes_a_generations_ids_only():
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "decode", "of": "counterfactual.v_cf", "featurizer": "rot"}
    with pytest.raises(PlanError, match=r"steps\.text\.featurizer: decode of a featurizer's coordinates closes an encode "
                                        r"inside the same entry; as a step it decodes a generation's ids"):
        edited("random_subspace", change).compile()


def test_an_operand_names_an_earlier_entry_whose_ops_all_write():
    """An operand from the same forward is taken off the model as the forward reaches it, so it names an
    earlier entry, with no op after its last edit, at one layer."""
    def operand(entries):
        def change(d):
            step = d["steps"]["patched"]
            step["interventions"] = {**entries, **step["interventions"]}
            step["interventions"]["patch"]["ops"][0]["with"] = "patched.early"
        return change
    early = {"site": {"component": "layer_input", "layers": 0}, "pos": -1}
    edited("patching", operand({"early": early})).compile()
    with pytest.raises(PlanError, match=r"'patched\.early' runs 'readout' once it is off the model, after this forward"):
        edited("patching", operand({"early": {**early, "ops": [{"op": "readout"}]}})).compile()
    with pytest.raises(PlanError, match=r"'patched\.later' is an entry this forward makes after this one; an operand "
                                        r"names an earlier entry"):
        def later(d):
            d["steps"]["patched"]["interventions"]["later"] = early
            d["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "patched.later"
        edited("patching", later).compile()
    with pytest.raises(PlanError, match=r"two entries named 'patch'; a step binds each name once"):
        def twice(d):
            d["interventions"] = {"again": {"patch": early}}
            d["steps"]["patched"]["interventions"] = ["again", d["steps"]["patched"]["interventions"]]
        edited("patching", twice).compile()


@pytest.mark.parametrize("change, message", [
    (lambda d: d["sites"]["target"].update(select={"heads": [0]}),
     r"sites\.target: layer_output has no axis 'heads' to select; its axes: hidden_size$"),
    (lambda d: d["sites"]["target"].update(component="attention_premix", select={"units": [0]}),
     r"sites\.target: attention_premix has no axis 'units' to select; its axes: num_heads, head_dim$"),
    (lambda d: d["sites"]["target"].update(heads=[0]),
     r"sites\.target: unknown field 'heads'; valid: component, layers, select$"),
])
def test_select_names_an_axis_the_component_publishes(change, message):
    with pytest.raises(PlanError, match=message):
        edited("patching", change)


@pytest.mark.parametrize("where", ["site", "op"])
@pytest.mark.parametrize("selector, why", [([], r"\[\] is not a selector: .*; a list of picks names at least one int"),
                                           ("3:3", r"'3:3' picks nothing, however many things there are")])
def test_a_select_that_picks_nothing_is_refused(where, selector, why):
    """A `select` is a selector, refused as `layers` and `pos` refuse one that can pick nothing."""
    def change(d):
        if where == "site":
            d["sites"]["target"]["select"] = {"hidden_size": selector}
        else:
            d["steps"]["patched"]["interventions"]["patch"]["ops"].insert(
                0, {"op": "encode", "featurizer": "rot", "select": {"features": selector}})
    with pytest.raises(PlanError, match=r"select\.(hidden_size|features): " + why):
        edited("patching", change)


def test_a_select_on_key_counts_a_fixed_number_from_the_start():
    for selector in (-1, "2:", [0, -1]):
        def change(d, selector=selector):
            d["sites"]["target"].update(component="attention_probabilities", select={"key": selector})
        with pytest.raises(PlanError, match=f"select key {re.escape(repr(selector))}: the key axis is as long as each "
                                            "row, so a selector there picks a fixed number of keys counted from the "
                                            "start"):
            edited("patching", change)


def test_a_write_selects_on_one_axis():
    def change(d):
        d["sites"]["target"].update(component="attention_premix", select={"num_heads": [0], "head_dim": [0]})
    with pytest.raises(PlanError, match=r"interventions\.patch\.site\.select: an entry that writes selects on one axis"):
        edited("patching", change).compile()


def test_an_operand_must_have_the_writes_layout():
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["logits"] = {"site": {"component": "lm_head_output"}, "pos": -1}
        d["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "counterfactual.logits"
    with pytest.raises(PlanError, match=r"'counterfactual\.logits' is laid out \['rows', 'pos', 'vocab_size'\]; "
                                        r"this write's value is \['rows', 'pos', 'hidden_size'\] here"):
        edited("patching", change).compile()


def test_renormalize_runs_only_inside_an_entry():
    def change(d):
        d["steps"]["iia"] = {"kind": "op", "op": "renormalize", "of": "counterfactual.v_cf"}
    with pytest.raises(PlanError, match="'renormalize' restores the value a site had before its writes; it runs only "
                                        "inside an entry"):
        edited("patching", change).compile()


def test_a_forward_makes_one_call():
    def change(d):
        d["steps"]["patched"]["interventions"]["logits"]["pos"] = {"frame": 1, "index": -1}
    with pytest.raises(PlanError, match=r"steps\.patched\.interventions\.logits\.pos\.frame: frame 1: a forward makes one "
                                        r"call, the prefill; its frame picks that call"):
        edited("patching", change).compile()


def test_readout_projects_only_the_residual_stream():
    def change(d):
        d["sites"]["resid"]["component"] = "mlp_activation"
    with pytest.raises(PlanError, match="readout projects a residual-stream value; mlp_activation is not one"):
        edited("logit_lens", change).compile()


# -- preflight, before any forward ----------------------------------------------------------------------------


def refused(engines, plan, kind: str = "hooks"):
    """Run `plan` on an engine that counts its forwards; return the refusal and the count."""
    engine = engines(kind, plan.model)
    calls = []
    original = engine.forward
    engine.forward = lambda *args: calls.append(1) or original(*args)
    try:
        with pytest.raises((ValueError, NotImplementedError)) as caught:
            engine.execute(plan, batch_size=2)
    finally:
        del engine.forward
    return str(caught.value), len(calls)


@pytest.mark.parametrize("name, change, message", [
    ("patching", lambda d: d["sites"]["target"].update(layers=99), "counterfactual.v_cf: layer 99; the model has 2 layers"),
    ("patching", lambda d: d["sites"]["target"].update(select={"hidden_size": [999]}),
     r"counterfactual.v_cf: select hidden_size \[999\] outside the model's 16"),
    ("patching", lambda d: d["sites"]["target"].update(select={"hidden_size": -17}),
     r"counterfactual.v_cf: select hidden_size -17 outside the model's 16"),
    ("patching", lambda d: d["sites"]["target"].update(select={"hidden_size": "20:"}),
     r"counterfactual.v_cf: select hidden_size '20:' keeps none of the model's 16"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"].update(scored={
        "site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "match", "expected": "pairs.input"}]}),
     "patched.scored: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens"),
    ("pca_harvest", lambda d: d["steps"]["harvest"]["interventions"]["acts"].update(pos=-1),
     "basis: pca of 4 vectors has no rank-4 basis"),
    ("entity_patch", lambda d: d["data"].update(pairs={"rows": [{
        "input": "If today is Monday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
     r"patched.patch: row 0: column 'pairs.entity' is not in its input"),
    ("entity_patch", lambda d: (d["data"].update(pairs={"rows": [{
        "input": "Friday or Friday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
        d["steps"]["patched"]["interventions"]["patch"]["pos"]["scope"].update(occurrence=2)),
     r"patched.patch: row 0: column 'pairs.entity' occurs 2 times in the row's input; it has no occurrence 2"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["interventions"]["blank"]["pos"]["scope"].update(turn="system"),
     r"blanked.blank: row 0: the row has no 'system' message"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["interventions"]["blank"]["pos"]["scope"].update(occurrence=2),
     r"blanked.blank: row 0: turn 'user' occurs 2 times in the row's input; it has no occurrence 2"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"index": "9:-1"},
                                                                           ops=[{"op": "swap", "with": 0.0}]),
     r"patched.patch: row 0: the write's positions hold no token of this row; a write needs one to write"),
    ("chat_turn", lambda d: d["data"].update(chats={"rows": [{"chat": [{"role": "user", "content": "  padded  "}],
                                                             "base_answer": " Friday"}]}),
     r"blanked: chats.chat row 0: message 0 \(user\) '  padded  ' is not where the chat template put the messages"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=20),
     r"patched.patch: row 0: tokens 20 do not fit its 10 scope tokens"),
])
def test_what_needs_the_model_is_checked_before_any_forward(engines, name, change, message):
    text, forwards = refused(engines, edited(name, change).compile())
    assert re.match(message, text) and forwards == 0, text


@pytest.mark.parametrize("selector", [[9], 9, "7:10"])
def test_a_select_on_key_outside_the_shortest_row_is_refused(engines, selector):
    plan = edited("attention_knockout", lambda d: d["sites"]["pattern"].update(select={"key": selector})).compile()
    text, forwards = refused(engines, plan, kind="nnterp")
    assert text == f"clean.pattern: select key {selector!r} outside the 9 tokens of the input's shortest row" and \
        forwards == 0


def test_an_operand_of_another_width_is_refused(engines):
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = -1
    text, forwards = refused(engines, edited("window_patch", change).compile())
    assert text.startswith("patched.patch: operand 'counterfactual.v_cf' is 1 token(s) wide") and forwards == 0, text


def test_a_pattern_from_an_input_of_other_lengths_is_refused(engines):
    def change(d):
        d["data"]["pairs"] = {"path": "weekdays/train"}
    text, forwards = refused(engines, edited("attention_pattern_patching", change).compile(), kind="nnterp")
    assert forwards == 0
    assert text.startswith("patched.look_there: operand 'counterfactual.their_pattern' was read on "
                           "pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there"), text


def test_the_aligned_pattern_dataset_passes_the_check(engines):
    plan = plan_of("attention_pattern_patching")
    engines("nnterp", plan.model).execute(plan, batch_size=2)


@pytest.mark.parametrize("kind", ["nnterp", "hooks", "ndif"])
@pytest.mark.parametrize("key, why", [("device", r"where the model runs is a run option: --device"),
                                      ("device_map", r"where the model runs is a run option: --device"),
                                      ("dispatch", "the engine places its model itself"),
                                      ("torch_dtype", "the dtype is spelled `dtype`")])
def test_a_loader_refuses_a_model_key_it_cannot_honor(kind, key, why):
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update({key: "cpu"}))
    with pytest.raises(ValueError, match=f"^model.{key}: the {kind} engine cannot honor it: {why}$"):
        ENGINES[kind](plan.model)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_key_the_loader_rejects_is_reported_as_that_key(kind):
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update(attn_implementation="nonsense"))
    with pytest.raises(ValueError, match=f"^model.attn_implementation: the {kind} engine's loader rejects it: "):
        ENGINES[kind](plan.model)


def test_the_model_block_passes_its_other_keys_through_and_spells_dtype_as_torch():
    plan = plan_of("attention_knockout")
    assert plan.model.dtype == "float32" and plan.model.model_extra == {"attn_implementation": "eager"}
    with pytest.raises(PlanError, match=r"model\.dtype: Input should be 'float32', 'float16' or 'bfloat16'"):
        edited("patching", lambda d: d["model"].update(dtype="fp32"))


def test_a_forwards_frame_is_any_selector_that_picks_its_one_call(engines):
    """`[0]`, `"0:1"` and `-1` pick a forward's one call as `0` does, and the record stays rectangular; `":"`,
    `"0:"` and `"::"` pick it too, but name calls to the row's last, so the record is ragged (one entry per
    token, whose operand is one vector). Each writes what its `frame: 0` twin does, to the bit."""
    engine = engines("hooks", plan_of("patching").model)

    def run(frame, operand):
        def change(d):
            d["steps"]["patched"]["interventions"]["patch"].update(pos={"frame": frame, "index": -1},
                                                            ops=[{"op": "swap", "with": operand}])
        return engine.execute(edited("patching", change).compile(), batch_size=BATCH_SIZE).results
    for frames, operand in (([[0], "0:1", -1], "counterfactual.v_cf"), ([":", "0:", "::"], 0.0)):
        base = run(0, operand)
        for frame in frames:
            assert same(run(frame, operand), base), frame
