"""What cannot be served, or would change a document's meaning, is refused by name before any forward runs."""

import re
from typing import Callable

import pytest
from conftest import BATCH_SIZE, dims, edited, plan_of, same

from clab.engines import load
from clab.format.sweeps import points
from clab.plan import PlanError

# -- the plan and compile ---------------------------------------------------------------------------------


def test_select_and_featurizer_belong_to_encode():
    def change(key, value):
        def apply(d):
            d["steps"]["patched"]["interventions"]["patch"]["ops"][0][key] = value
        return apply
    with pytest.raises(PlanError, match=r"ops\[0\]: 'swap' takes no 'select': an encode keeps some of a featurizer's "
                                        r"coordinates, \{'op': 'encode', 'featurizer': \.\.\., 'select'"):
        edited("patching", change("select", {"features": [0]}))
    with pytest.raises(PlanError, match=r"ops\[0\]: 'swap' takes no 'featurizer': an op runs on a featurizer's coordinates "
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
    ([DEC], r"patch\.ops\[0\] at seed=0: decode of 'rot' closes no encode: none is open in this entry here"),
    ([ENC, SWAP, DEC, DEC], r"patch\.ops\[3\] at seed=0: decode of 'rot' closes no encode"),
    ([ENC, SWAP], r"patch\.ops\[1\] at seed=0: the entry writes back here with the encode of 'rot' \(ops\[0\]\) still open; "
                  r"close it with \{'op': 'decode', 'featurizer': 'rot'\}"),
    ([ENC, {"op": "encode", "featurizer": "other"}, SWAP, DEC, {"op": "decode", "featurizer": "other"}],
     r"patch\.ops\[3\] at seed=0: decode of 'rot', and the open encode is of 'other' \(ops\[1\]\); close 'other' first: a "
     r"decode closes the most recent encode not yet closed"),
    ([{"op": "readout"}, {"op": "scale", "factor": 2.0}],
     r"patch\.ops\[1\]: 'scale' is an edit, which writes back, and it follows 'readout', which turned the value into "
     r"\['rows', 'pos', 'vocab_size'\]: there is no live value to write"),
    ([ENC, {"op": "mean", "over": ["features"]}, DEC],
     r"patch\.ops\[2\] at seed=0: 'decode' is an edit, which writes back, and it follows 'mean'"),
    ([{"op": "swap", "with": 0.0}, ENC, {"op": "scale", "factor": 2.0}, DEC], None),
    ([ENC, {"op": "swap", "with": "counterfactual.raw"}, DEC],
     r"patch\.ops\[1\]\.with at seed=0: 'counterfactual\.raw' is in the site's own layout; this bracket is in the coordinates "
     r"of 'rot', so an operand is encoded as it is"),
    ([SWAP], r"patch\.ops\[0\]\.with at seed=0: 'counterfactual\.v_cf' is in the coordinates of 'rot'; this write is in the "
             r"site's own layout"),
    ([ENC, {"op": "apply", "featurizer": "rot"}, DEC],
     r"patch\.ops\[1\] at seed=0: 'apply' of 'rot' at ops\[1\] is inside the encode of 'rot' \(ops\[0\]\), so it would act on "
     r"'rot''s own coordinates; put it before that encode or after the decode that closes it"),
    ([ENC, {"op": "encode", "featurizer": "rot"}, DEC, DEC], r"patch\.ops\[1\] at seed=0: 'encode' of 'rot' at ops\[1\] is inside"),
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
    edit before an encode, and one inside a bracket after it, are fine. random_subspace sweeps `rot`'s seed, so a
    refusal in an entry that names `rot` names the point it is at (the first, `seed=0`); one in an entry that
    does not is at no point."""
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


@pytest.mark.parametrize("k, select", [(4, None), (8, None), (4, {"features": [0, 1]})])
def test_an_operand_in_other_coordinates_is_refused_by_name(k, select):
    """Inside a bracket an operand is in its coordinates: read through the same featurizer with the same
    `select`. Another subspace of the same `k` would run silently and mean nothing; one of another `k` would
    fail only when it ran; another `select` keeps other coordinates. Each is refused at compile by name."""
    def change(d):
        d["featurizers"]["rot3"] = {"kind": "subspace", "k": k, "seed": 3}
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["ops"] = [
            {"op": "encode", "featurizer": "rot" if select else "rot3", **({"select": select} if select else {})}]
    theirs = r"'rot' \(select \{'features': \[0, 1\]\}\)" if select else r"'rot3'"
    with pytest.raises(PlanError, match=r"patched\.interventions\.patch\.ops\[1\]\.with at seed=0: 'counterfactual\.v_cf' is "
                                        rf"in the coordinates of {theirs}; this bracket is in the coordinates of 'rot'"):
        edited("random_subspace", change).compile()


def test_decode_runs_only_inside_the_entry_that_encodes():
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "decode", "of": "counterfactual.v_cf", "featurizer": "rot"}
    with pytest.raises(PlanError, match=r"steps\.text at seed=0: decode closes an encode of its own entry, so it runs inside the "
                                        r"entry that encodes.*\(detokenize turns a generation's ids into text\)"):
        edited("random_subspace", change).compile()


def test_an_operand_names_an_earlier_entry_whose_ops_all_write():
    """An operand from the same forward is taken off the model as the forward reaches it, so it names an
    earlier entry, with no op after its last edit, and is a write's: an op after an entry's last edit runs once
    the value is off the model, where the forward's entries are not yet values."""
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
    with pytest.raises(PlanError, match=r"'patched\.later' is this entry or one this forward makes after it; an "
                                        r"operand names an earlier entry"):
        def later(d):
            d["steps"]["patched"]["interventions"]["later"] = early
            d["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "patched.later"
        edited("patching", later).compile()
    with pytest.raises(PlanError, match=r"logits\.ops\[0\]\.with: 'patched\.early' is an entry of this forward, and "
                                        r"'cosine' here runs once the value is off the model"):
        def after(d):
            d["steps"]["patched"]["interventions"] = {"early": early, **d["steps"]["patched"]["interventions"]}
            d["steps"]["patched"]["interventions"]["logits"] = {
                "site": {"component": "layer_input", "layers": 1}, "pos": -1,
                "ops": [{"op": "cosine", "with": "patched.early"}]}
        edited("patching", after).compile()
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
    """The one rule's sentence covers an operand with no axes too: it leaves out every axis and broadcasts, as a
    mean over every axis (`dims: []`) does into a write."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["logits"] = {"site": {"component": "lm_head_output"}, "pos": -1}
        d["steps"]["patched"]["interventions"]["patch"]["ops"][0]["with"] = "counterfactual.logits"
    with pytest.raises(PlanError, match=r"'counterfactual\.logits' is laid out \['rows', 'pos', 'vocab_size'\]; "
                                        r"'swap' acts on a value laid out \['rows', 'pos', 'hidden_size'\], and an "
                                        r"operand's axes are among the value's, in its order \(those it leaves out it "
                                        r"broadcasts over\)$"):
        edited("patching", change).compile()

    def scalar(d):
        d["steps"]["m"] = {"kind": "op", "op": "mean", "of": "iia", "over": ["rows", "pos"]}
        d["steps"]["again"] = {"kind": "forward", "input": "pairs.input", "interventions": {
            "w": {"site": "target", "pos": -1, "ops": [{"op": "scale", "factor": "m"}]}}}
    assert dims(edited("patching", scalar).compile(), "m") == []


def test_a_select_mismatch_the_document_decides_is_refused_at_compile():
    """An operand narrowed by `select` to other indices than the write's selection is refused when the plan
    compiles, before any model loads, whether the two are written as picks or as front-bounded slices, which name
    their indices as picks do (`kept_of`)."""
    def change(theirs, ours):
        def apply(d):
            d["steps"]["counterfactual"]["interventions"]["v_cf"]["site"] = {
                "component": "layer_output", "layers": 0, "select": {"hidden_size": theirs}}
            d["steps"]["patched"]["interventions"]["patch"]["site"] = {
                "component": "layer_output", "layers": 0, "select": {"hidden_size": ours}}
        return apply
    for theirs, ours in (([0, 1], [0, 1, 2]), ("0:2", "0:3"), (":2", [0, 1, 2])):
        with pytest.raises(PlanError, match=r"^steps\.patched\.interventions\.patch\.ops\[0\]\.with: 'counterfactual\.v_cf' "
                                            r"holds hidden_size \[0, 1\], and this write's selection holds \[0, 1, 2\]; an "
                                            r"operand holds the same ones: select hidden_size \[0, 1, 2\] at the read$"):
            edited("patching", change(theirs, ours)).compile()
    edited("patching", change("0:2", ":2")).compile()


def test_an_op_that_aggregates_over_rows_says_where_it_goes():
    """After an entry's last edit the aggregate is a reduction of a read, a step after the forward; up to it the
    entry would write it back, which a step after the forward cannot feed: an op step before the forward, swapped
    in."""
    def ops_(*listed):
        return lambda d: d["steps"]["patched"]["interventions"]["patch"].update(ops=list(listed))
    swap = {"op": "swap", "with": 0.0}
    with pytest.raises(PlanError, match=r"patch\.ops\[1\]: 'mean' aggregates over rows, and inside an entry it sees one "
                                        r"batch; make it a step after the forward$"):
        edited("patching", ops_(swap, {"op": "mean"})).compile()
    with pytest.raises(PlanError, match=r"patch\.ops\[0\]: 'mean' aggregates over rows, and inside an entry it sees one "
                                        r"batch; this entry writes it back, so make it an op step before the forward, "
                                        r"over a read of the rows it aggregates, and write that in \(\{'op': 'swap', "
                                        r"'with': '<that step>'\}\)$"):
        edited("patching", ops_({"op": "mean"}, swap)).compile()


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
    """Inside an entry and as a step over a read alike (one check, `transforms.unstreamed`): the final norm takes
    the residual stream, so a readout of the MLP's normed input would norm it twice."""
    streams = "token_embeddings, layer_input, attention_output, layer_mid, mlp_output, layer_output"

    def change(d):
        d["sites"]["resid"]["component"] = "mlp_norm_output"
    with pytest.raises(PlanError, match=r"steps\.lens\.interventions\.logits\.ops\[0\] at layers=0: readout projects a "
                                        rf"residual-stream value; mlp_norm_output is not one: read one of {streams}$"):
        edited("logit_lens", change).compile()

    def as_step(d):
        change(d)
        d["steps"]["lens"]["interventions"]["logits"]["ops"] = []
        d["steps"] = {"lens": d["steps"]["lens"], "readout": {"kind": "op", "op": "readout", "of": "lens.logits"},
                      **{k: {**s, "of": "readout"} for k, s in d["steps"].items() if k != "lens"}}
    with pytest.raises(PlanError, match=r"steps\.readout\.of at layers=0: 'lens\.logits' is read at mlp_norm_output: "
                                        rf"readout projects a residual-stream value; mlp_norm_output is not one: read one of "
                                        rf"{streams}$"):
        edited("logit_lens", as_step).compile()


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
    ("patching", lambda d: d["sites"]["target"].update(select={"hidden_size": "14:20"}),
     r"counterfactual.v_cf: select hidden_size '14:20' outside the model's 16"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"].update(scored={
        "site": {"component": "lm_head_output"}, "pos": -1, "ops": [{"op": "match", "expected": "pairs.input"}]}),
     "patched.scored: pairs.input row 0: 'If today is Thursday, tomorrow is' is 10 tokens"),
    ("pca_harvest", lambda d: d["steps"]["harvest"]["interventions"]["acts"].update(pos=-1),
     "basis: pca of 4 vectors has no rank-4 basis"),
    ("entity_patch", lambda d: d["data"].update(pairs={"rows": [{
        "input": "If today is Monday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
     r"patched.patch: row 0: column 'pairs.entity' \('Friday'\) is not in its input$"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"text": "tom"}, "index": -1}),
     r"patched.patch: row 0: text 'tom' is not in its input; it occurs only inside 'tomorrow', cutting a word, and an "
     r"occurrence lies on word boundaries$"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"text": "day is"}, "index": -1}),
     r"patched.patch: row 0: text 'day is' is not in its input; it occurs only inside 'today is', cutting a word, and "
     r"an occurrence lies on word boundaries$"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"scope": {"text": "zebra"}, "index": -1}),
     r"patched.patch: row 0: text 'zebra' is not in its input$"),
    ("entity_patch", lambda d: d["steps"]["patched"]["interventions"]["patch"]["pos"]["scope"].update(occurrence=1),
     r"patched.patch: row 0: column 'pairs.entity' \('Thursday'\) occurs 1 time in the row's input; it has no "
     r"occurrence 1$"),
    ("entity_patch", lambda d: (d["data"].update(pairs={"rows": [{
        "input": "Friday or Friday, tomorrow is", "entity": "Friday", "cf_answer": " Sunday", "base_answer": " Friday",
        "counterfactual_inputs": ["If today is Friday, tomorrow is"],
        "counterfactual_inputs_variables": [{"entity": "Friday"}]}]}),
        d["steps"]["patched"]["interventions"]["patch"]["pos"]["scope"].update(occurrence=2)),
     r"patched.patch: row 0: column 'pairs.entity' \('Friday'\) occurs 2 times in the row's input; it has no occurrence 2$"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["interventions"]["blank"]["pos"]["scope"].update(turn="system"),
     r"blanked.blank: row 0: the row has no 'system' message"),
    ("chat_turn", lambda d: d["steps"]["blanked"]["interventions"]["blank"]["pos"]["scope"].update(occurrence=2),
     r"blanked.blank: row 0: turn 'user' occurs 2 times in the row's input; it has no occurrence 2"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"index": "10:-1"},
                                                                           ops=[{"op": "swap", "with": 0.0}]),
     r"patched.patch: row 0: the write's positions hold no token of this row; a write needs one to write"),
    ("chat_turn", lambda d: d["data"].update(chats={"rows": [{"chat": [{"role": "user", "content": "  padded  "}],
                                                             "base_answer": " Friday"}]}),
     r"blanked: chats.chat row 0: message 0 \(user\) '  padded  ' is not where the chat template put the messages"),
    ("patching", lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos=20),
     r"patched.patch: row 0: index 20 does not fit the row's 11 positions at this frame"),
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


def test_an_operand_of_another_width_is_refused():
    """window_patch's write is three positions wide; an operand read at one is refused when the plan compiles,
    both widths being static."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = -1
    with pytest.raises(PlanError, match=r"patch\.ops\[0\]\.with: 'counterfactual\.v_cf' holds 1 position\(s\) per row "
                                        r"and the value 'swap' acts on holds 3; position i meets position i"):
        edited("window_patch", change).compile()


def test_a_write_with_a_static_width_refuses_an_operand_whose_rows_disagree(engines):
    """entity_mean_ablation's harvest is a whole-entity scope, three positions on the Thursday row and one on the
    others; swapped into a window of three it would write NaN where a row has fewer, so preflight refuses it once
    the rows are known, naming them, as into a whole-entity write on another forward; the mean over it is one vector
    and lands anywhere."""
    def change(pos, d):
        d["steps"]["ablated"]["interventions"]["ablate"]["pos"] = pos
        d["steps"]["ablated"]["interventions"]["ablate"]["ops"] = [{"op": "swap", "with": "harvest.acts"}]
    for pos in ({"index": "-3:"}, {"scope": {"column": "prompts.entity"}, "index": ":"}):
        text, forwards = refused(engines, edited("entity_mean_ablation", lambda d: change(pos, d)).compile())
        assert text == "ablated.ablate: operand 'harvest.acts' holds 3 positions on row 0 and 1 position on rows 1-3; " \
            "a scoped read pairs with another forward's value when every row's scope holds the same number of " \
            "tokens: pick one with 'index' (-1, or '-2:'), or reduce it" and forwards == 0, text


def test_a_whole_scope_operand_is_refused_by_preflight_naming_the_rows_that_differ(engines):
    """T9's repro: entity_patch with both entity reads over their whole scope (its `index` dropped) compiles, since
    only the rows say how many tokens each row's entity is; preflight refuses it before any forward, naming the
    counterfactual's rows and their counts (' Thursday' three tokens on row 2) and the `'index'` that keeps the
    scope. On rows whose entities are each one token the same reads run (`documents/entity_patch_scope.json`)."""
    def whole(d):
        for step, entry in (("counterfactual", "v_cf"), ("patched", "patch")):
            del d["steps"][step]["interventions"][entry]["pos"]["index"]
    plan = edited("entity_patch", whole).compile()
    assert points(plan.steps["patched"])[0].interventions["patch"].width is None
    for kind in ("nnterp", "hooks"):
        text, forwards = refused(engines, plan, kind)
        assert text == "patched.patch: operand 'counterfactual.v_cf' holds 1 position on rows 0-1, 3 and 3 positions " \
            "on row 2; a scoped read pairs with another forward's value when every row's scope holds the same " \
            "number of tokens: pick one with 'index' (-1, or '-2:'), or reduce it" and forwards == 0, text


def test_a_whole_scope_write_is_named_as_what_its_operand_meets(engines):
    """entity_patch with the write over its whole entity (three tokens on the Thursday row) and its operand at the
    counterfactual's last entity token: preflight names the operand, and the write's own value as what it meets,
    with the write's rows; a `cosine` step whose scored value is such a scope says the same of what it scores."""
    def whole_write(d):
        del d["steps"]["patched"]["interventions"]["patch"]["pos"]["index"]
    text, forwards = refused(engines, edited("entity_patch", whole_write).compile())
    assert text == "patched.patch: operand 'counterfactual.v_cf' meets the value 'swap' acts on, which holds 3 " \
        "positions on row 0 and 1 position on rows 1-3; a scoped read pairs with another forward's value when every " \
        "row's scope holds the same number of tokens: pick one with 'index' (-1, or '-2:'), or reduce it" and \
        forwards == 0, text

    def scored(d):
        whole_write(d)
        d["steps"]["patched"]["interventions"]["patch"]["ops"] = []
        d["steps"]["counterfactual"]["interventions"]["v_cf"]["pos"] = -1
        d["steps"]["similarity"] = {"kind": "op", "op": "cosine", "of": "patched.patch", "with": "counterfactual.v_cf"}
        d["saves"] = ["similarity"]
    text, forwards = refused(engines, edited("entity_patch", scored).compile())
    assert text.startswith("similarity: operand 'counterfactual.v_cf' meets the value 'cosine' scores, which holds 3 "
                           "positions on row 0 and 1 position on rows 1-3; ") and forwards == 0, text


def test_the_rows_of_a_scope_that_disagree_are_named_by_count():
    """Each count with its rows, in the order the rows first hold it, a run of rows as its ends; a row reached at no
    position holds 0; past four runs a count's rows are its first three runs and how many there are."""
    from clab.compile.arguments import row_counts
    assert row_counts([[4]] * 8 + [[4, 5]]) == "1 position on rows 0-7 and 2 positions on row 8"
    assert row_counts([[3], [4, 5, 6], "alignment_missing", [3]]) == \
        "1 position on rows 0, 3, 3 positions on row 1 and 0 positions on row 2"
    assert row_counts([[1, 2] if r % 2 else [1] for r in range(12)]) == \
        "1 position on rows 0, 2, 4, … (6 rows) and 2 positions on rows 1, 3, 5, … (6 rows)"


def test_a_whole_scope_operand_of_another_confirmed_width_is_refused_by_preflight(engines):
    """entity_patch_scope's counterfactual entity is one token on every row, which preflight confirms; into a
    write three positions wide it is refused there, as two static widths are at compile."""
    plan = edited("entity_patch_scope",
                  lambda d: d["steps"]["patched"]["interventions"]["patch"].update(pos={"index": "-3:"})).compile()
    text, forwards = refused(engines, plan)
    assert text == "patched.patch: operand 'counterfactual.v_cf' holds 1 position(s) per row and the value 'swap' " \
        "acts on holds 3; position i meets position i, so both are read at as many (a pos axis does not broadcast)" \
        and forwards == 0, text


def test_a_pattern_from_an_input_of_other_lengths_is_refused(engines):
    def change(d):
        d["data"]["pairs"] = {"path": "weekdays/train"}
    text, forwards = refused(engines, edited("attention_pattern_patching", change).compile(), kind="nnterp")
    assert forwards == 0
    assert text.startswith("patched.look_there at num_heads=0: operand 'counterfactual.their_pattern' was read on "
                           "pairs.counterfactual_inputs[0], row 0 of which is 9 tokens; pairs.input has 11 there"), text


def test_the_aligned_pattern_dataset_passes_the_check(engines):
    plan = plan_of("attention_pattern_patching")
    engines("nnterp", plan.model).execute(plan, batch_size=2)


def test_a_grad_entry_is_refused_by_name_where_the_engine_takes_no_gradient(engines, monkeypatch):
    """`start_run` refuses a step's `grad` entry on the client, before the run starts, where the engine declares that
    it takes no gradient (`gradless`): the nnterp engine's under `tp`, given here by its options alone, so that the
    refusal runs without torchrun."""
    plan = plan_of("attribution_patching")
    monkeypatch.setattr(engines("nnterp", plan.model), "options", {"tp": 2})
    text, forwards = refused(engines, plan, kind="nnterp")
    assert text == ("patched.h: the nnterp engine takes no gradient under tp 2, whose ranks hold a sharded value's "
                    "gradient in pieces; run it at tp 1") and forwards == 0 and plan.run is None


@pytest.mark.parametrize("kind", ["nnterp", "hooks", "ndif"])
@pytest.mark.parametrize("key, why", [("device", r"where the model runs is a run option: --device"),
                                      ("device_map", r"where the model runs is a run option: --device"),
                                      ("dispatch", "the engine places its model itself"),
                                      ("torch_dtype", "the dtype is spelled `dtype`"),
                                      ("quantization_config",
                                       r"a quantization is spelled in `dtype` \(`nf4`, `fp4`, `int8`, `fp8`\)"),
                                      ("tp_plan", "tensor parallelism is a run option: -o tp=N"),
                                      ("distributed_config", "tensor parallelism is a run option: -o tp=N")])
def test_a_loader_refuses_a_model_key_it_cannot_honor(kind, key, why):
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update({key: "cpu"}))
    with pytest.raises(ValueError, match=f"^model.{key}: the {kind} engine cannot honor it: {why}$"):
        ENGINES[kind](plan.model)


@pytest.mark.parametrize("kind", ["nnterp", "hooks", "ndif"])
def test_a_checkpoint_the_loader_cannot_get_is_refused_as_the_model_key(kind):
    """Offline (the suite sets HF_HUB_OFFLINE), a checkpoint the cache lacks is refused naming the key, the
    revision and the cache, not as the loader's OSError about the files it looked for."""
    plan = edited("patching", lambda d: d["model"].update(key="nobody/nothing-here", revision="main"))
    with pytest.raises(ValueError, match=f"^model.key 'nobody/nothing-here' at revision 'main': the {kind} engine cannot "
                                         r"load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part "
                                         r"of it; fetch it online first \(the loader says: .+\)$"):
        load(kind, plan.model, **({"remote": "local"} if kind == "ndif" else {}))


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
def test_a_device_no_engine_can_place_a_model_on_is_refused_before_anything_loads(monkeypatch, kind, options):
    """Every engine's constructor (`standard.loading`, before its loader; `load` too) refuses a device that is no
    torch device or device map (the ndif engine by its own message, which test_patching pins), a CUDA device where
    none is visible, and one past those that are (a stand-in says two are)."""
    from clab.engines import ENGINES

    model = plan_of("patching").model
    if kind != "ndif":
        with pytest.raises(ValueError, match=r"^device 'gpu': not a torch device \('cpu', 'cuda:0'\) or a device map "
                                             r"transformers computes \('auto', 'balanced', 'balanced_low_0', 'sequential'\)$"):
            load(kind, model, "gpu")
    none = (r"^device 'cuda': no CUDA device is visible \(torch.cuda.is_available\(\) is False\); run on 'cpu', or "
            r"make a GPU visible \(CUDA_VISIBLE_DEVICES\)$")
    with pytest.raises(ValueError, match=none):
        load(kind, model, "cuda", **options)
    with pytest.raises(ValueError, match=none):
        ENGINES[kind](model, device="cuda", **options)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    with pytest.raises(ValueError, match="^device 'cuda:2': 2 CUDA devices visible, cuda:0 to cuda:1$"):
        load(kind, model, "cuda:2", **options)
    with pytest.raises(ValueError, match="^device 'cuda:2': 2 CUDA devices visible, cuda:0 to cuda:1$"):
        ENGINES[kind](model, device="cuda:2", **options)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_key_the_loader_rejects_is_reported_as_that_key(kind):
    """The reason is the loader's one line that names the key, without the report of every class nnsight's loader
    tried, so a misspelled key is refused by name on both engines alike: as hooks loads, and on nnterp, whose meta
    shell takes the misspelling and loads its weights before the first step of a run that calls the model, there."""
    from clab.engines import ENGINES
    plan = edited("patching", lambda d: d["model"].update(attn_implementation="nonsense"))
    with pytest.raises(ValueError, match=f"^model.attn_implementation: the {kind} engine's loader rejects it: "):
        ENGINES[kind](plan.model)
    plan = edited("patching", lambda d: d["model"].update(attn_implementaton="eager"))
    with pytest.raises(ValueError, match=f"^model.attn_implementaton: the {kind} engine's loader rejects it: "
                                         r"LlamaForCausalLM.__init__\(\) got an unexpected keyword argument "
                                         r"'attn_implementaton'$"):
        ENGINES[kind](plan.model).execute(plan.compile(), batch_size=BATCH_SIZE)


def test_a_bundle_entrys_refusal_names_the_step_that_pulls_it_in():
    """A bundle's entry has one document path and may be pulled into several steps; its operand resolves in the
    step that pulls it in, so a refusal names that step after the path."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["logits"] = {"site": {"component": "lm_head_output"}, "pos": -1}
        d["interventions"] = {"b": {"patch": {"site": "target", "pos": -1,
                                               "ops": [{"op": "swap", "with": "counterfactual.logits"}]}}}
        d["steps"]["patched"]["interventions"] = ["b", {"logits": d["steps"]["patched"]["interventions"]["logits"]}]
    with pytest.raises(PlanError, match=r"^interventions\.b\.patch\.ops\[0\]\.with in steps\.patched: "
                                        r"'counterfactual\.logits' is laid out \['rows', 'pos', 'vocab_size'\]"):
        edited("patching", change).compile()


def test_the_model_block_passes_its_other_keys_through_and_spells_dtype_as_torch():
    plan = plan_of("attention_knockout")
    assert plan.model.dtype == "float32" and plan.model.model_extra == {"attn_implementation": "eager"}
    for wrong in ("fp32", "int4"):                  # one spelling per thing: nnsight's `int4` is `nf4` here
        with pytest.raises(PlanError, match=r"model\.dtype: Input should be 'float32', 'float16', 'bfloat16', 'nf4', "
                                            r"'fp4', 'int8' or 'fp8'$"):
            edited("patching", lambda d: d["model"].update(dtype=wrong))


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


def test_a_metric_over_decode_frame_reads_is_told_a_frame_a_decode_read_can_satisfy():
    """Two reads over every decode call have no static width; the advice names a static index in the prefill and,
    for a decode read, a frame of a static number of calls with an index of static width, which lines up."""
    def change(pos):
        def apply(d):
            d["steps"]["patched"]["interventions"]["a"] = {"site": {"component": "logits"}, "pos": pos}
            d["steps"]["counterfactual"]["interventions"]["b"] = {"site": {"component": "logits"}, "pos": pos}
            d["steps"]["kl"] = {"kind": "op", "op": "kl", "of": "patched.a", "against": "counterfactual.b"}
            d["saves"] = ["kl"]
        return apply
    with pytest.raises(PlanError, match=r"steps\.kl\.against at frame=0: 'counterfactual\.b' has no width compile can "
                                        r"fix \(an open index, an `until` scope, several occurrences\), and two values "
                                        r"of different reads pair only at a fixed width, whether one forward or two "
                                        r"made them; read both at a static "
                                        r"index without a scope \('index': -1, or '-3:'\), at one occurrence of a scope "
                                        r"in the prefill, not until, whose rows give its width, or over decode calls at "
                                        r"a frame of a static number of them with an index of static width "
                                        r"\(\{'frame': '1:4', 'index': ':3'\}\)$"):
        edited("generate_probe", change({"frame": "1:", "index": ":"})).compile()
    compiled = edited("generate_probe", change({"frame": "1:4", "index": ":3"})).compile()
    assert points(compiled.steps["kl"])[0].width == 3


def test_a_quantized_dtype_names_what_it_computes_in():
    """A quantization's values are in its compute dtype, `bfloat16` (`int8`'s `float16`), which the compiled plan
    does not replace: its `model.dtype` is the quantization's name."""
    computes = {d: edited("patching", lambda doc: doc["model"].update(dtype=d)).compile().model
                for d in ("float32", "bfloat16", "nf4", "fp4", "int8", "fp8")}
    assert {d: m.computes for d, m in computes.items()} == {"float32": "float32", "bfloat16": "bfloat16",
                                                             "nf4": "bfloat16", "fp4": "bfloat16", "int8": "float16",
                                                             "fp8": "bfloat16"}
    assert computes["nf4"].dtype == "nf4"


@pytest.fixture
def unloadable(monkeypatch):
    """No engine can load a model: a refusal must come before anything loads (a meta shell included)."""
    for where in ("nnterp", "hooks", "ndif"):
        monkeypatch.setattr(f"clab.engines.{where}.StandardizedTransformer", lambda *a, **k: pytest.fail("loaded"))
    return monkeypatch


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
@pytest.mark.parametrize("dtype", ["nf4", "fp4", "int8"])
def test_a_bitsandbytes_dtype_without_bitsandbytes_is_refused_before_anything_loads(unloadable, kind, options, dtype):
    """Named, not an ImportError from inside transformers."""
    import importlib.util
    found = importlib.util.find_spec
    unloadable.setattr(importlib.util, "find_spec",
                       lambda name, *a: None if name == "bitsandbytes" else found(name, *a))
    plan = edited("patching", lambda d: d["model"].update(dtype=dtype))
    with pytest.raises(ValueError, match=f"^model.dtype '{dtype}' holds the weights in bitsandbytes, which is not "
                                         "installed: install bitsandbytes$"):
        load(kind, plan.model, **options)


@pytest.mark.parametrize("kind, options", [("nnterp", {}), ("hooks", {}), ("ndif", {"remote": "local"})])
def test_fp8_off_a_gpu_is_refused_before_anything_loads(unloadable, kind, options):
    plan = edited("patching", lambda d: d["model"].update(dtype="fp8"))
    with pytest.raises(ValueError, match=f"^model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; the "
                                         f"{kind} engine's device is 'cpu'"):
        load(kind, plan.model, **options)


def test_a_quantized_dtype_on_an_ndif_server_is_refused():
    """A server holds a model's weights as its deployment does, so a document cannot ask it for a quantization."""
    plan = edited("patching", lambda d: d["model"].update(dtype="nf4"))
    with pytest.raises(ValueError, match="^model.dtype 'nf4': the ndif engine cannot honor it: the server decides how "
                                         "a model's weights are held$"):
        load("ndif", plan.model)


def permuting(**steps) -> Callable:
    """mean_ablation with `steps` added after its harvest, read at the last three positions."""
    def change(d):
        d["steps"]["harvest"]["interventions"]["acts"]["pos"] = {"index": "-3:"}
        d["steps"] = {"harvest": d["steps"]["harvest"], **steps,
                      **{k: v for k, v in d["steps"].items() if k != "harvest"}}
    return change


def one_row(d):
    d["data"]["prompts"]["take"] = 1


def minibatch_of_one(d):
    d["featurizers"] = {"lora": {"kind": "lora", "rank": 1, "alpha": 1}}
    d["steps"]["fit"] = {"kind": "fit", "train": ["lora"], "objective": {"m": 1.0}, "epochs": 1, "batch_size": 3,
                         "seed": 0, "optimizer": {"name": "sgd", "lr": 0.1}, "steps": {
                             "run": {"kind": "forward", "input": "prompts.input", "interventions": {"x": {
                                 "site": "target", "pos": {"index": "-3:"},
                                 "ops": [{"op": "apply", "featurizer": "fit.lora"}]}}},
                             "p": {"kind": "op", "op": "permute", "of": "run.x", "seed": 0, "derangement": True},
                             "m": {"kind": "op", "op": "mean", "of": "p", "over": ["rows", "pos", "hidden_size"]}}}


def seeded_in_fit(seed: int, batch_size: int, epochs: int = 1) -> Callable:
    """minibatch_of_one's fit at `batch_size` and `epochs`, its permute by `seed` without derangement."""
    def change(d):
        minibatch_of_one(d)
        d["steps"]["fit"].update(batch_size=batch_size, epochs=epochs)
        d["steps"]["fit"]["steps"]["p"] = {"kind": "op", "op": "permute", "of": "run.x", "seed": seed}
    return change


@pytest.mark.parametrize("change, message", [
    (lambda d: d["steps"]["ablated"]["interventions"]["ablate"]["ops"].insert(0, {"op": "permute", "seed": 0}),
     r"^steps\.ablated\.interventions\.ablate\.ops\[0\]: permute is no op of an entry, whose ops act on each row's "
     r"live value: it is a step over a read"),
    (permuting(top={"kind": "op", "op": "top_k", "of": "harvest.acts", "k": 2, "over": "hidden_size"},
               p={"kind": "op", "op": "permute", "of": "top", "seed": 0}),
     r"^steps\.p\.of: 'top' is a top_k's value, which permute does not reorder; a top_k acts row by row, so permute "
     r"the value it is over \('harvest\.acts'\) and top_k that$"),
    (permuting(p={"kind": "op", "op": "permute", "of": "harvest.acts", "seed": 0},
               b={"kind": "op", "op": "pca", "of": "p", "k": 2}),
     r"^steps\.b\.of: 'p' holds other rows' values \(a permute's\).*which a pca's bundle records: pca the read itself"),
    (lambda d: (permuting(p={"kind": "op", "op": "permute", "of": "harvest.acts", "seed": 0, "derangement": True})(d),
                one_row(d)),
     r"^steps\.p: a derangement moves every row to another, and 'prompts' has 1 row: permute over two rows or more"),
    (minibatch_of_one,
     r"^steps\.fit\.steps\.p: a derangement moves every row to another, and over 4 rows in minibatches of 3 the last "
     r"of each epoch holds one row"),
    (seeded_in_fit(0, 2),
     r"^steps\.fit\.steps\.p\.seed: seed 0 draws the identity over a minibatch of 2 rows, which moves no row; give "
     r"derangement true, or a seed that moves them$"),
    (seeded_in_fit(1, 3),
     r"^steps\.fit\.steps\.p\.seed: seed 1 draws the identity over a minibatch of 1 row, which moves no row; give the "
     r"fit a batch_size that leaves every minibatch two rows or more$"),
])
def test_a_permute_is_refused_by_name_where_it_cannot_mean_anything(change, message):
    """As an entry's op, over a value that is no tensor of rows, before a pca, whose bundle records one read's site,
    as a derangement over a dataset or a fit's minibatch of one row, and in a fit's body by a seed that draws the
    identity over a minibatch's rows, which `permutation` decides as the run draws it."""
    with pytest.raises(PlanError, match=message):
        edited("mean_ablation", change).compile()


@pytest.mark.parametrize("change", [seeded_in_fit(1, 2), seeded_in_fit(0, 2, epochs=0), seeded_in_fit(0, 16)])
def test_a_seeded_permute_in_a_fit_compiles_where_no_minibatch_it_runs_draws_the_identity(change):
    """Seed 1 swaps two rows; a fit of 0 epochs runs no minibatch; at batch_size 16 the one minibatch is the 4 rows,
    which seed 0 moves."""
    edited("mean_ablation", change).compile()


def ranked_whole(steps: dict, ops: list | None = None):
    """readout.json with `steps` after its forward, and its head read with `ops` in place of none."""
    def change(d):
        if ops is not None:
            d["steps"]["clean"]["interventions"]["logits"]["ops"] = ops
            del d["steps"]["top3"], d["steps"]["p_answer"]
        d["steps"].update(steps)
        d["saves"] = list(steps) or ["clean.logits"]
    return change


TOP3 = r"'top3' is a top_k's value, several tensors \(its ids, their probs and their tokens\)"
RANK = r"rank what it needs with an op over the value the top_k ranks"


@pytest.mark.parametrize("change, message", [
    (ranked_whole({"m": {"kind": "op", "op": "mean", "of": "top3"}}),
     rf"^steps\.m\.of: {TOP3}, and 'mean' takes one: name the member it needs \('top3\.ids', 'top3\.probs'\), or "
     rf"{RANK}, 'clean\.logits'$"),
    (ranked_whole({"m": {"kind": "op", "op": "argmax", "of": "top3", "over": "k"}}), rf"^steps\.m\.of: {TOP3}, and 'argmax'"),
    (ranked_whole({"best": {"kind": "op", "op": "argmax", "of": "clean.logits"},
                   "m": {"kind": "op", "op": "eq", "of": "best", "with": "top3"}}),
     rf"^steps\.m\.with: {TOP3}, and an operand takes one: name the member it needs \('top3\.ids', 'top3\.probs'\), "
     rf"or {RANK}, 'clean\.logits'$"),
    (ranked_whole({"t": {"kind": "op", "op": "top_k", "of": "clean.logits", "k": 2, "over": ["rows", "vocab_size"]},
                   "m": {"kind": "op", "op": "mean", "of": "t", "over": ["k"]}}),
     r"^steps\.m\.of: 't' is a top_k's value, several tensors \(its ids \(one tensor per axis it ranks\), their probs and "
     r"their tokens\), and 'mean' takes one: name the member it needs \('t\.ids\.rows', 't\.ids\.vocab_size', "
     r"'t\.probs'\), or "),
    (ranked_whole({"m": {"kind": "op", "op": "max", "of": "clean.logits", "over": ["k"]}}, [{"op": "top_k", "k": 2}]),
     r"^steps\.m\.of: 'clean\.logits' is a top_k's value, several tensors \(its ids, their probs\), and 'max' takes one: "
     r"rank what it needs in that entry in place of the top_k, or make the top_k a step after the forward, whose "
     r"members a later step names \(`<step>\.ids`\)$"),
    (ranked_whole({}, [{"op": "top_k", "k": 2}, {"op": "mean", "over": ["pos"]}]),
     r"^steps\.clean\.interventions\.logits\.ops\[1\]: 'mean' follows top_k, whose value is several tensors \(its ids, "
     r"their probs\), and 'mean' takes one: rank what it needs in this entry in place of the top_k, or make the top_k a "
     r"step after the forward and 'mean' a step over the member it needs \(`<step>\.ids`\)$"),
    (ranked_whole({"t": {"kind": "op", "op": "top_k", "of": "clean.logits", "k": 2, "over": ["rows", "vocab_size"]},
                   "m": {"kind": "op", "op": "eq", "of": "t.ids", "with": 0}}),
     r"^steps\.m\.of: 't\.ids' is a top_k's ids over a list of axes, one tensor per axis it ranks, and 'eq' takes one: "
     r"name the axis's \('t\.ids\.rows', 't\.ids\.vocab_size'\)$"),
    (ranked_whole({"m": {"kind": "op", "op": "eq", "of": "top3.tokens", "with": 0}}),
     r"^steps\.m\.of: 'top3\.tokens' is a top_k's tokens, its picks' token strings, which are text, and 'eq' takes a "
     r"tensor: name their ids \('top3\.ids'\)$"),
], ids=["mean", "argmax", "operand", "over-a-list", "an-entry's", "after-it-in-an-entry", "ids-over-a-list", "tokens"])
def test_an_op_over_a_whole_top_k_value_is_refused_at_compile(change, message):
    """A top_k's value is several tensors, which every op met at the run with a raw error: an op step over one, of
    either form (`ids` one tensor, or one per axis over a list of them), an operand naming one, an op over an entry
    whose last op is one, and an op after one in its entry are refused by name at compile, naming the members a later
    step takes in its place; and so are its `ids` over a list of axes (one tensor per axis) and its `tokens` (text)."""
    with pytest.raises(PlanError, match=message):
        edited("readout", change).compile()


def ranked_past(read: dict, steps: dict, ops: list | None = None, featurizers: dict | None = None):
    """readout.json with its forward reading `read` (running `ops` in the entry), then `steps`."""
    def change(d):
        d["steps"] = {"clean": {"kind": "forward", "input": "prompts.input", "interventions": {
            "r": {**read, **({"ops": ops} if ops else {})}}}, **steps}
        d["saves"] = list(steps) or ["clean.r"]
        d["featurizers"] = featurizers or {}
    return change


TOP = {"kind": "op", "op": "top_k", "of": "clean.r"}
LAYER0 = {"component": "layer_output", "layers": 0}
NARROW = {"site": {"component": "logits", "select": {"vocab_size": [1, 2, 3]}}, "pos": -1}
PICKS = r", each of which it picks once: k is at most "


@pytest.mark.parametrize("change, message", [
    (ranked_past(NARROW, {"t": {**TOP, "k": 5}}),
     rf"^steps\.t\.k: k 5 is more than the 3 entries it ranks, 3 along vocab_size \(those its select keeps\){PICKS}3$"),
    (ranked_past(NARROW, {}, [{"op": "top_k", "k": 4}]),
     r"^steps\.clean\.interventions\.r\.ops\[0\]\.k: k 4 is more than the 3 entries it ranks, 3 along vocab_size \(those"),
    (ranked_past({"site": {"component": "layer_output", "layers": [0, 1]}, "pos": -1},
                 {"t": {**TOP, "k": 3, "over": "layers"}}),
     rf"^steps\.t\.k: k 3 is more than the 2 entries it ranks, 2 along layers \(the value's\){PICKS}2$"),
    (ranked_past({"site": LAYER0, "pos": {"index": "-3:"}}, {"t": {**TOP, "k": 4, "over": "pos"}}),
     rf"^steps\.t\.k: k 4 is more than the 3 entries it ranks, 3 along pos \(each row's\){PICKS}3$"),
    (ranked_past({"site": LAYER0, "pos": -1}, {}, [{"op": "encode", "featurizer": "probe"},
                                                    {"op": "top_k", "k": 5, "over": "classes"}],
                 {"probe": {"kind": "linear", "classes": "prompts.cf_answer"}}),
     rf"^steps\.clean\.interventions\.r\.ops\[1\]\.k: k 5 is more than the 4 entries it ranks, 4 along classes \(the "
     rf"probe's\){PICKS}4$"),
    (ranked_past(NARROW, {"t": {**TOP, "k": 0}}), r"^steps\.t\.k: k 0: a top_k picks at least one entry, so k is 1 or "
                                                  r"more$"),
    (ranked_past(NARROW, {}, [{"op": "top_k", "k": -1}]),
     r"^steps\.clean\.interventions\.r\.ops\[0\]\.k: k -1: a top_k picks at least one entry, so k is 1 or more$"),
    (ranked_past({"site": LAYER0, "pos": -1}, {"t": {**TOP, "k": 5, "over": ["rows"]}}),
     rf"^steps\.t\.k: k 5 is more than the 4 entries it ranks, 4 along rows \(the dataset's\){PICKS}4$"),
    (ranked_past({"site": LAYER0, "pos": -1}, {"t": {**TOP, "k": 5, "over": "features"}},
                 [{"op": "encode", "featurizer": "rot"}], {"rot": {"kind": "subspace", "k": 4}}),
     rf"^steps\.t\.k: k 5 is more than the 4 entries it ranks, 4 along features \(the subspace's coordinates\)"
     rf"{PICKS}4$"),
    (ranked_past({"site": LAYER0, "pos": -1}, {}, [{"op": "encode", "featurizer": "s", "select": {"features": [0, 1]}},
                                                    {"op": "top_k", "k": 3, "over": "features"}],
                 {"s": {"kind": "sae", "latents": 8}}),
     rf"^steps\.clean\.interventions\.r\.ops\[1\]\.k: k 3 is more than the 2 entries it ranks, 2 along features "
     rf"\(those its encode's select keeps\){PICKS}2$"),
    (ranked_past({"site": {"component": "logits"}, "pos": -1},
                 {"t": {**TOP, "k": 3}, "a": {"kind": "op", "op": "abs", "of": "t.probs"},
                  "u": {"kind": "op", "op": "top_k", "of": "a", "k": 5, "over": "k"}}),
     rf"^steps\.u\.k: k 5 is more than the 3 entries it ranks, 3 along k \(the picks of the top_k it ranks\){PICKS}3$"),
], ids=["select", "select-in-an-entry", "layers", "static-width", "classes", "zero", "negative", "rows", "features",
        "selected-features", "a-top_k's-k"])
def test_a_top_k_past_a_static_axis_the_document_sizes_is_refused_at_compile(change, message):
    """A `k` past the entries of an axis whose size the document fixes, a `select`'s indices, the layers it picks, a
    static width, a probe's classes, failed at the run with torch's own error: refused at compile, naming the axis
    and the most `k` can be: also the dataset's rows, a featurizer's coordinates (as many as an encode's `select`
    keeps), an earlier `top_k`'s `k` through an op step over its member; and so is a `k` of 0 or below, which picks
    nothing, on a step and in an entry."""
    with pytest.raises(PlanError, match=message):
        edited("readout", change).compile()


@pytest.mark.parametrize("batch_size, refused", [(1, True), (2, False), (3, False)])
def test_a_top_k_over_rows_in_a_fits_body_is_held_to_the_rows_its_fewest_row_run_holds(batch_size, refused):
    """das.json's body runs over minibatches of its 2 training rows and once over both: a `top_k` over rows there
    ranks each run's rows, so `k` 2 fits minibatches of 2 or more and is refused at compile, naming the bound, where
    every minibatch holds one."""
    def change(d):
        d["steps"]["fit"]["batch_size"] = batch_size
        d["steps"]["fit"]["steps"]["t"] = {"kind": "op", "op": "top_k", "of": "patched.logits", "k": 2, "over": "rows"}
    plan = edited("das", change)
    if not refused:
        plan.compile()
        return
    with pytest.raises(PlanError, match=r"^steps\.fit\.steps\.t\.k: k 2 is more than the 1 entry it ranks, 1 along rows "
                                        r"\(those the fewest-row run of the body holds, of 2 rows in minibatches of 1\), "
                                        r"each of which it picks once: k is at most 1$"):
        plan.compile()


@pytest.mark.parametrize("change, message", [
    (ranked_past({"site": {"component": "logits"}, "pos": -1}, {"t": {**TOP, "k": 40000}}),
     rf"^t: k 40000 is more than the 32000 entries it ranks, 32000 along vocab_size \(the model's\){PICKS}32000$"),
    (ranked_past({"site": LAYER0, "pos": -1}, {}, [{"op": "top_k", "k": 17, "over": "hidden_size"}]),
     rf"^clean\.r: k 17 is more than the 16 entries it ranks, 16 along hidden_size \(the model's\){PICKS}16$"),
    (ranked_past({"site": {"component": "layer_output", "layers": ":"}, "pos": -1},
                 {"t": {**TOP, "k": 3, "over": "layers"}}),
     rf"^t: k 3 is more than the 2 entries it ranks, 2 along layers \(the value's\){PICKS}2$"),
    (ranked_past({"site": LAYER0, "pos": {"scope": {"text": " tomorrow"}}}, {"t": {**TOP, "k": 3, "over": "pos"}}),
     rf"^t: k 3 is more than the 2 entries it ranks, 2 along pos \(each row's\){PICKS}2$"),
    (ranked_past({"site": LAYER0, "pos": {"index": "-2:"}}, {"t": {**TOP, "k": 33, "over": ["pos", "hidden_size"]}}),
     rf"^t: k 33 is more than the 32 entries it ranks, 2 along pos \(each row's\) by 16 along "
     rf"hidden_size \(the model's\){PICKS}32$"),
], ids=["vocab_size", "hidden_size-in-an-entry", "a-slice-of-layers", "a-scope's-width", "a-list-of-axes"])
def test_a_top_k_past_a_static_axis_the_model_or_the_rows_size_is_refused_before_any_forward(engines, change,
                                                                                            message):
    """A `k` past the entries of an axis the model sizes (`vocab_size`, `hidden_size`, the layers a slice picks) or
    the rows do (a scope whose rows confirm one width), alone or among a list of axes, is refused in preflight, before
    any forward, where it failed at the run with torch's own error."""
    text, forwards = refused(engines, edited("readout", change).compile())
    assert re.match(message, text) and forwards == 0, text


def test_a_top_k_past_the_models_size_at_one_point_of_a_sweep_is_refused_naming_the_point(engines):
    """Swept over `k`, the point whose `k` is past the model's vocabulary is refused before any forward, naming it."""
    plan = edited("readout", ranked_past({"site": {"component": "logits"}, "pos": -1},
                                         {"t": {**TOP, "k": {"sweep": [3, 40000]}}})).compile()
    text, forwards = refused(engines, plan)
    assert re.match(r"^t at k=40000: k 40000 is more than the 32000 entries it ranks", text) and forwards == 0, text


def test_a_top_k_over_a_list_of_axes_is_a_step_and_refused_inside_an_entry():
    """Its `ids` are an index per ranked axis of each pick, which an entry's batches do not join."""
    def change(d):
        d["steps"]["patched"]["interventions"]["logits"]["ops"] = [{"op": "top_k", "k": 2, "over": ["pos", "vocab_size"]}]
    with pytest.raises(PlanError, match=r"^steps\.patched\.interventions\.logits\.ops\[0\]: a top_k over a list of axes "
                                        r"gives an index per axis of each pick, which an entry's batches do not join; "
                                        r"make it a step after the forward$"):
        edited("patching", change).compile()
