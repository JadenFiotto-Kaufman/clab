"""One forward shared by the points of a step that differ only in what they read, and the schedule the engines
execute.

A value depends on the sweep axes its record says (`axes`: the dependency rule, `sweeps.find_sweeps`), and a
forward on its writes' and options' alone (`schedule["axes"]`), so compile has the first of the points that agree on
those serve the rest (each other's `schedule["serving"]`), and the run makes that forward once with every served
point's reads,
each bound at its own point. Each point's values are then those a document with that point alone would give, to
the bit (`alone`: the same document with each sweep replaced by the point's value, run on its own), on every
engine, under `remote="local"` and through the stand-in server; `run.costs` records each forward once, at the
point that ran it, with the points it served (`serves`). A sweep on a write (its operand, positions or layer)
runs a forward per write, as does one reached by a write and a read (zip); a grid of a swept write and a swept
read shares within each write; a write whose operand is an earlier entry of its forward shares while that entry
does not vary; a generation shares as a forward does, its ids one value, a sampled one's points reading one
sample; a bundle pulled into two steps shares within each step, never across them; and a later step or a fit's
body reads each point's own value. On every corpus document the forwards the engine makes are the schedule's,
and the sites each visits, in order, are those its schedule lists.
"""

import itertools
import json
import os
import types
from typing import Any, Callable, cast

import pytest
import torch
from conftest import BATCH_SIZE, INSIDE, NAMES, TRIP, document, plan_of, same, stand_in
from test_efficiency import at, counting

from clab import Plan
from clab.compile import schedule
from clab.format.sweeps import coordinate, find_sweeps, labelled, points, projected
from clab.plan import Fit, Forward, entries, horizon, record_of, serving
from clab.steps import batches, fit


def doc_of(name: str, change: Callable[[dict], None] | None = None) -> dict:
    """Document `name`'s JSON with `change` applied."""
    with open(document(name)) as f:
        doc = json.load(f)
    if change is not None:
        change(doc)
    return doc


def compiled(name: str, doc: dict) -> Plan:
    return Plan(**doc, document=document(name)).compile()


def alone(engine, name: str, doc: dict, batch_size: int = BATCH_SIZE) -> dict[tuple, Plan]:
    """The document run once per point of its sweeps, each with every sweep replaced by that point's value: what
    each point's own forward gives, keyed by the point."""
    sweeps = find_sweeps(doc)
    values: dict[str, list] = {}
    for path, label in sweeps.items():
        value: Any = doc
        for key in path:
            value = value[key]
        values.setdefault(label, value["sweep"])

    def substituted(value: Any, path: tuple, at: dict[str, int]) -> Any:
        if path in sweeps:
            return value["sweep"][at[sweeps[path]]]
        if isinstance(value, dict):
            return {k: substituted(v, path + (k,), at) for k, v in value.items()}
        if isinstance(value, list):
            return [substituted(v, path + (i,), at) for i, v in enumerate(value)]
        return value
    out = {}
    for index in itertools.product(*(range(len(v)) for v in values.values())):
        at = dict(zip(values, index))
        out[tuple(at.items())] = engine.execute(compiled(name, substituted(doc, (), at)), batch_size=batch_size)
    return out


def each_its_own(shared: Plan, alone: dict[tuple, Plan]) -> bool:
    """Whether every saved value of `shared`, at every point of its step, and where every read and write resolved,
    is what the document with that point `alone` gives, to the bit."""
    for name in shared.saves:
        entry = shared.steps[name.partition(".")[0]]
        for p in points(entry):
            own = projected(p.at, record_of(shared.steps, name, p.at).axes)
            mine = shared.results[name][coordinate(shared, own)] if own else shared.results[name]
            theirs = next(v for k, v in alone.items() if all(dict(k).get(a) == i for a, i in p.at.items()))
            if not same(mine, theirs.results[name]):
                return False
    for step, entry in shared.steps.items():
        for p in points(entry):
            theirs = next(v for k, v in alone.items() if all(dict(k).get(a) == i for a, i in p.at.items()))
            for key, record in entries(p).items():
                if record.positions != entries(theirs.steps[step])[key].positions:
                    return False
    return True


def costs(plan: Plan, name: str) -> dict[str, tuple[int, list[str]]]:
    """Each run of step `name`, by the point that ran it: its model calls and the points it served."""
    assert plan.run is not None
    return {at: (c["forwards"], c["serves"]) for at, c in plan.run.costs[name].items()}


def forwards(plan: Plan, name: str) -> list[int]:
    """The model calls of each run of step `name`, in point order."""
    assert plan.run is not None
    by = plan.run.costs[name]
    return [by["forwards"]] if "forwards" in by else [c["forwards"] for c in by.values()]


def total(plan: Plan) -> int:
    """Every model call the run made."""
    assert plan.run is not None
    return sum(c["forwards"] for by in plan.run.costs.values() for c in ([by] if "forwards" in by else by.values()))


def served(plan: Plan, name: str) -> list[list[str]]:
    """Which points each forward of step `name` serves, in point order, by coordinate."""
    return [[labelled(plan, q.at) for q in s] for p, s in serving(plan, name) if isinstance(p, Forward)]


def all_points(plan: Plan, name: str) -> list[str]:
    return [labelled(plan, p.at) for p in points(plan.steps[name])]


def sharing(plan: Plan) -> bool:
    """Whether a point of `plan` is served by another's forward."""
    return any("serving" in p.schedule for e in plan.steps.values() for p in points(e) if isinstance(p, Forward))


def axes_of(plan: Plan, name: str) -> list[str]:
    return record_of(plan.steps, name, points(plan.steps[name.partition(".")[0]])[0].at).axes


def counted(engine, plan: Plan, batch_size: int) -> tuple[Plan, list[int]]:
    """`plan` run on `engine`, and each block that ran (`counting`'s blocks alone), once it has run uncounted."""
    engine.execute(plan, batch_size=batch_size)
    with counting(engine) as ran:
        filled = engine.execute(plan, batch_size=batch_size)
    return filled, blocks(ran)


def blocks(ran: list) -> list[int]:
    """The blocks among the modules `counting` saw run."""
    return [tag for tag in ran if isinstance(tag, int)]


ENGINES = {"nnterp": {}, "hooks": {}, "ndif": {"remote": "local"}}


def gpt2_doc(entries: dict[str, Any], **root: Any) -> dict:
    """Tiny GPT-2 (five blocks) over gpt2_reach's four pairs, one forward `lens` with `entries`, each saved;
    `root` adds root keys."""
    def change(d):
        d.update(root)
        d["steps"] = {"lens": {"kind": "forward", "input": "pairs.input", "interventions": entries}}
        d["saves"] = [f"lens.{k}" for k in entries]
    return doc_of("gpt2_reach", change)


def gpt2(entries: dict[str, Any], **root: Any) -> Plan:
    return compiled("gpt2_reach", gpt2_doc(entries, **root))


ZERO = [{"op": "swap", "with": 0.0}]
READS = {
    "layers": {"read": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4]}}, "pos": -1,
                        "ops": [{"op": "readout"}]}},
    "pos": {"read": {"site": {"component": "layer_output", "layers": 2}, "pos": {"sweep": [-1, -2, -3, -4, -5]}}},
}


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("axis", READS)
def test_a_read_only_sweep_runs_one_forward_and_each_point_reads_what_its_own_would(engines, kind, axis):
    """Five points that read at five layers (a readout of each), or at five positions: the forward depends on no
    axis and the read on its own (`axes`), so one forward of the four rows in one batch, at the first point, serves
    all five, and the run records it once; the blocks run once, to the deepest point's (each point alone runs to
    its own; the ndif engine runs every forward whole); each point's value is its own forward's to the bit."""
    doc = gpt2_doc(READS[axis])
    plan = compiled("gpt2_reach", doc)
    lens = points(plan.steps["lens"])
    assert lens[0].schedule == {"axes": []} and [p.schedule for p in lens[1:]] == [{"serving": lens[0].at}] * 4
    assert served(plan, "lens") == [all_points(plan, "lens")]
    assert axes_of(plan, "lens.read") == [axis]
    engine = engines(kind, plan.model, **ENGINES[kind])
    shared, ran = counted(engine, plan, 4)
    deepest = [4] * 5 if kind == "ndif" else [0, 1, 2, 3, 4] if axis == "layers" else [2] * 5
    assert ran == [*range(max(deepest) + 1)]
    first = labelled(plan, lens[0].at)
    assert costs(shared, "lens") == {first: (1, all_points(plan, "lens"))}
    assert shared.run is not None and shared.run.costs["lens"][first]["memory"] is None
    theirs = {}
    for key, point in alone(engine, "gpt2_reach", doc, 4).items():
        with counting(engine) as ran_alone:
            engine.execute(point, batch_size=4)
        theirs[key] = point
        assert blocks(ran_alone) == [*range(deepest[dict(key)[axis]] + 1)]
    assert each_its_own(shared, theirs)
    values = list(shared.results["lens.read"].values())
    assert not any(same(values[0], v) for v in values[1:])


def test_the_stand_in_server_shares_the_forward_as_the_client_would(engines, monkeypatch):
    """The serialized plan carries each point's schedule and the server completes and runs it: on its model the
    five points' two batches run the blocks once each, the server's cost comes home at the serving point, and
    what comes home is the local run's."""
    plan = gpt2(READS["layers"])
    local = engines("nnterp", plan.model).execute(gpt2(READS["layers"]), batch_size=BATCH_SIZE)
    with counting(engines("nnterp", plan.model)) as ran:
        came, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP and blocks(ran) == [*range(5)] * 2
    assert costs(came, "lens") == {"layers=0": (2, all_points(plan, "lens"))}
    assert same(came.results, local.results)


WRITES = {
    "operand": {"scaled": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                           "ops": [{"op": "scale", "factor": {"sweep": [0.0, 0.5, 2.0]}}]}},
    "position": {"zero": {"site": {"component": "layer_output", "layers": 1}, "pos": {"sweep": [-1, -2, -3]},
                          "ops": ZERO}},
    "layer": {"zero": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}, "pos": -1, "ops": ZERO}},
}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
@pytest.mark.parametrize("swept", WRITES)
def test_a_sweep_on_a_write_runs_a_forward_per_point(engines, swept, kind):
    """A write's operand, positions or layer swept: the forward depends on that axis, so every point writes
    otherwise, each runs its own forward, and the read after the writes, which depends on it too, sees each: every
    value is its own document's, to the bit."""
    doc = gpt2_doc({**WRITES[swept], "read": at("layer_output", 3)})
    plan = compiled("gpt2_reach", doc)
    axis = {"operand": "factor", "position": "pos", "layer": "layers"}[swept]
    assert [p.schedule for p in points(plan.steps["lens"])] == [{"axes": [axis]}] * 3
    assert axes_of(plan, "lens.read") == [axis]
    engine = engines(kind, plan.model)
    filled = engine.execute(plan, batch_size=4)
    assert costs(filled, "lens") == {c: (1, [c]) for c in all_points(plan, "lens")}
    assert each_its_own(filled, alone(engine, "gpt2_reach", doc, 4))
    values = list(filled.results["lens.read"].values())
    assert not same(values[0], values[1]) and not same(values[1], values[2])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_write_whose_operand_is_a_swept_value_runs_a_forward_per_point(engines, kind):
    """A read-only sweep over layers in `cf` runs one forward; a write in `patched` swapping `cf.read` in at one
    layer depends on the operand's axis, so `patched` runs one forward per point and its read inherits the axis:
    every value is its own document's, to the bit."""
    def change(d):
        d["steps"] = {
            "cf": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]", "interventions": {
                "read": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}, "pos": -1}}},
            "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
                "patch": {"site": {"component": "layer_output", "layers": 3}, "pos": -1,
                          "ops": [{"op": "swap", "with": "cf.read"}]},
                "logits": at("lm_head_output")}}}
        d["saves"] = ["cf.read", "patched.logits"]
    doc = doc_of("gpt2_reach", change)
    plan = compiled("gpt2_reach", doc)
    assert axes_of(plan, "cf.read") == ["layers"] and axes_of(plan, "patched.logits") == ["layers"]
    assert [p.schedule.get("axes") for p in points(plan.steps["patched"])] == [["layers"]] * 3
    assert points(plan.steps["cf"])[0].schedule["axes"] == []
    engine = engines(kind, plan.model)
    filled = engine.execute(plan, batch_size=4)
    assert costs(filled, "cf") == {"layers=0": (1, all_points(plan, "cf"))}
    assert costs(filled, "patched") == {c: (1, [c]) for c in all_points(plan, "patched")}
    assert each_its_own(filled, alone(engine, "gpt2_reach", doc, 4))


def fewer_doc(stacked: bool) -> dict:
    """Tiny GPT-2: `cf` reads `v` unswept beside a read `w` swept over three blocks (or, `stacked`, `v` at blocks 1
    and 2 beside `w` swept over two positions), and `patched` swaps `cf.v` in at block 2 (`stacked`: at block 1 or
    2, the slice of `v` read there), then reads the head."""
    v = {"site": {"component": "layer_output", "layers": [1, 2] if stacked else 2}, "pos": -1}
    w = {"site": {"component": "layer_output", "layers": 3}, "pos": {"sweep": [-1, -2]}} if stacked else \
        {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}, "pos": -1}
    layers = {"sweep": [1, 2]} if stacked else 2

    def change(d):
        d["steps"] = {
            "cf": {"kind": "forward", "input": "pairs.counterfactual_inputs[0]", "interventions": {"v": v, "w": w}},
            "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
                "patch": {"site": {"component": "layer_output", "layers": layers}, "pos": -1,
                          "ops": [{"op": "swap", "with": "cf.v"}]},
                "logits": at("lm_head_output")}}}
        d["saves"] = ["cf.w", "patched.logits"]
    return doc_of("gpt2_reach", change)


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("stacked", [False, True], ids=["one", "stacked"])
def test_a_write_takes_an_operand_that_depends_on_fewer_axes_than_its_producers_step(engines, kind, stacked):
    """`cf.v` depends on no axis though `cf` is swept (its points share a forward, `w` alone varying), so the write
    that swaps it in finds its record at the write's own point and its value at its own coordinates, none; the
    stacked one takes the slice of `v` read at its own swept layer. Every value is what each point's own document
    gives, to the bit, on each engine, and the engines agree."""
    doc = fewer_doc(stacked)
    plan = compiled("gpt2_reach", doc)
    assert axes_of(plan, "cf.v") == [] and axes_of(plan, "cf.w") == ["pos" if stacked else "layers"]
    assert axes_of(plan, "patched.logits") == (["layers"] if stacked else [])
    engine = engines(kind, plan.model, **ENGINES[kind])
    filled = engine.execute(plan, batch_size=4)
    assert each_its_own(filled, alone(engine, "gpt2_reach", doc, 4))
    local = engines("nnterp", plan.model).execute(compiled("gpt2_reach", doc), batch_size=4)
    assert same(filled.results, local.results)


GRID = {"read": {"site": {"component": "layer_output", "layers": {"sweep": [2, 3, 4], "label": "src"}}, "pos": -1},
        "zero": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "dst"}}, "pos": -1,
                 "ops": ZERO}}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_grid_of_a_swept_write_and_a_swept_read_shares_within_each_write(engines, kind):
    """A read at layers 2 to 4 (`src`, the first axis) crossed with a write at layer 0 or 1 (`dst`): the forward
    depends on `dst` alone, so the points with one write share its first point's forward, the six make two, and
    the run records the two in point order; each value is its own forward's."""
    doc = gpt2_doc(GRID)
    plan = compiled("gpt2_reach", doc)
    assert list(plan.sweeps) == ["src", "dst"]
    lens = points(plan.steps["lens"])
    assert [p.schedule.get("serving", p.schedule.get("axes")) for p in lens] == \
        [["dst"], ["dst"], lens[0].at, lens[1].at, lens[0].at, lens[1].at]
    assert axes_of(plan, "lens.read") == ["src", "dst"] and axes_of(plan, "lens.zero") == ["dst"]
    engine = engines(kind, plan.model)
    shared = engine.execute(plan, batch_size=4)
    assert costs(shared, "lens") == {"src=2,dst=0": (1, ["src=2,dst=0", "src=3,dst=0", "src=4,dst=0"]),
                                     "src=2,dst=1": (1, ["src=2,dst=1", "src=3,dst=1", "src=4,dst=1"])}
    assert each_its_own(shared, alone(engine, "gpt2_reach", doc, 4))


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_one_sweep_reached_by_a_read_and_a_write_shares_nothing(engines, kind):
    """A swept site that a write and a read both use is one axis (zip): each point writes at its own layer, and
    every value is its own document's, to the bit."""
    doc = gpt2_doc({"zero": {"site": "swept", "pos": -1, "ops": ZERO}, "read": {"site": "swept", "pos": -2}},
                   sites={"swept": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}})
    plan = compiled("gpt2_reach", doc)
    assert list(plan.sweeps) == ["layers"]
    assert [p.schedule.get("axes") for p in points(plan.steps["lens"])] == [["layers"]] * 3
    engine = engines(kind, plan.model)
    filled = engine.execute(plan, batch_size=4)
    assert costs(filled, "lens") == {c: (1, [c]) for c in all_points(plan, "lens")}
    assert each_its_own(filled, alone(engine, "gpt2_reach", doc, 4))


LIVE = {"premix": {"site": {"component": "layer_output", "layers": 0}, "pos": -1},
        "added": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                  "ops": [{"op": "add", "with": "lens.premix"}]}}


def live_doc() -> dict:
    """Tiny GPT-2: a write at block 1 that adds what an earlier entry of its forward read at block 0, beside a read
    swept over blocks 2 to 4."""
    return gpt2_doc({**LIVE, "read": {"site": {"component": "layer_output", "layers": {"sweep": [2, 3, 4]}}, "pos": -1}})


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_write_from_an_earlier_entry_of_its_forward_shares_while_that_entry_does_not_vary(engines, kind):
    """The write's operand is an earlier entry of the forward that depends on no axis, so the three points share
    the forward, whose write takes the first point's read of that entry (the schedule names it); with that entry
    swept over blocks 0 and 1 instead, the write depends on its axis, and each point runs its own forward. Either
    way every value is its own document's, to the bit."""
    doc = live_doc()
    plan = compiled("gpt2_reach", doc)
    assert points(plan.steps["lens"])[0].schedule["axes"] == [] and served(plan, "lens") == [all_points(plan, "lens")]
    engine = engines(kind, plan.model)
    shared = engine.execute(plan, batch_size=4)
    assert costs(shared, "lens") == {"layers=2": (1, all_points(plan, "lens"))}
    sites = points(plan.steps["lens"])[0].schedule["sites"]            # `premix` and `added`, one value each, read once
    assert [(s["component"], s["layer"], [r.get("write") or r.get("read") for r in s["records"]]) for s in sites] == [
        ("layer_output", 0, ["premix"]), ("layer_output", 1, ["added", "added"]),       # the write, then its read
        ("layer_output", 2, ["read"]), ("layer_output", 3, ["read"]), ("layer_output", 4, ["read"])]
    assert sites[1]["records"][0]["ops"] == [{"op": "add", "with": {"name": "lens.premix", "live": "premix@0"}}]
    assert [r["at"] for s in sites for r in s["records"] if "read" in r] == \
        [{}, {}, {"layers": 0}, {"layers": 1}, {"layers": 2}]
    assert each_its_own(shared, alone(engine, "gpt2_reach", doc, 4))
    doc = gpt2_doc({**LIVE, "premix": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1]}}, "pos": -1},
                    "read": at("layer_output", 3)})
    swept = compiled("gpt2_reach", doc)
    assert [p.schedule.get("axes") for p in points(swept.steps["lens"])] == [["layers"]] * 2
    filled = engine.execute(swept, batch_size=4)
    assert costs(filled, "lens") == {c: (1, [c]) for c in all_points(swept, "lens")}
    assert each_its_own(filled, alone(engine, "gpt2_reach", doc, 4))


def ranked(rank: Callable[[str, int | None], Any]) -> Any:
    """A stand-in engine for `schedule.scheduled`: its name, and the rank it gives each `(component, layer)`."""
    return types.SimpleNamespace(name="stand-in", rank=rank)


def test_an_operand_read_at_a_site_the_engine_ranks_alike_with_the_writes_is_refused():
    """The write at block 1 adds what the forward read at block 0: taken where the engine ranks block 0 strictly
    first, refused where it ranks the two alike (an order that ties them may list either first), and taken from the
    write's own site whatever the ranks, read there first, in entry order."""
    plan = gpt2(LIVE)
    step = points(plan.steps["lens"])[0]
    schedule.scheduled(ranked(lambda component, layer: layer), plan, "lens", step, [step], {})
    assert [s["layer"] for s in step.schedule["sites"]] == [0, 1]
    with pytest.raises(ValueError, match=r"^lens\.added: operand 'lens\.premix' is read at layer_output layer 0, which "
                                         r"the stand-in engine's forward reaches no earlier than this write's "
                                         r"layer_output layer 1; "):
        schedule.scheduled(ranked(lambda component, layer: 0), plan, "lens", step, [step], {})
    here = gpt2({**LIVE, "premix": at("layer_output", 1)})
    step = points(here.steps["lens"])[0]
    schedule.scheduled(ranked(lambda component, layer: 0), here, "lens", step, [step], {})
    assert [[r.get("write") or r.get("read") for r in s["records"]] for s in step.schedule["sites"]] == \
        [["premix", "added", "added"]]


def probe_doc() -> dict:
    """generate_probe, its patched generation's ids also as text."""
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "detokenize", "of": "patched"}
        d["saves"].append("text")
    return doc_of("generate_probe", change)


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_generations_read_only_sweep_shares_the_generation(engines, kind):
    """generate_probe's patched generation reads the head at frame 0, 1 or 2 beside one write: the generation
    depends on no axis, so its three points share one (two batches of three calls), its ids are one value and so is
    their text (an op over them depends on no axis either), while the read and the metric over it vary by frame;
    the ids, the text, each row's tokens and where each read landed are each point's own generation's."""
    doc = probe_doc()
    plan = compiled("generate_probe", doc)
    patched = points(plan.steps["patched"])
    assert served(plan, "patched") == [all_points(plan, "patched")] and patched[0].axes == []
    assert axes_of(plan, "patched.logits") == ["frame"] and axes_of(plan, "p_answer") == ["frame"]
    assert axes_of(plan, "text") == [] and isinstance(plan.steps["text"], list)
    engine = engines(kind, plan.model)
    shared = engine.execute(plan, batch_size=BATCH_SIZE)
    assert costs(shared, "patched") == {"frame=0": (6, all_points(plan, "patched"))}
    assert costs(shared, "text") == {"frame=0": (0, all_points(plan, "text"))}
    assert shared.results["patched"].shape == (4, 3) and isinstance(shared.results["text"], list)
    assert shared.run is not None and isinstance(shared.run.tokens["patched"], list)
    theirs = alone(engine, "generate_probe", doc)
    assert each_its_own(shared, theirs)
    assert all(p.run is not None and shared.run.tokens["patched"] == p.run.tokens["patched"] for p in theirs.values())


def test_a_sampled_generations_points_read_one_sample(engines):
    """generate_probe's patched generation sampled (`do_sample`): its ids depend on no axis, so its points share
    one generation and each reads the one continuation, where each point's own generation would draw another."""
    def change(d):
        d["steps"]["patched"]["do_sample"] = True
    plan = compiled("generate_probe", doc_of("generate_probe", change))
    assert points(plan.steps["patched"])[0].axes == [] and axes_of(plan, "p_answer") == ["frame"]
    filled = engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE)
    assert costs(filled, "patched") == {"frame=0": (6, all_points(plan, "patched"))}
    assert filled.results["patched"].shape == (4, 3) and len(filled.results["p_answer"]) == 3


def bundled_doc() -> dict:
    """Tiny GPT-2 with a bundle whose read is swept over three layers, pulled into two alike steps."""
    def change(d):
        d["interventions"] = {"lens": {"read": {"site": {"component": "layer_output",
                                                         "layers": {"sweep": [0, 1, 2]}}, "pos": -1}}}
        d["steps"] = {name: {"kind": "forward", "input": "pairs.input", "interventions": "lens"}
                      for name in ("first", "second")}
        d["saves"] = ["first.read", "second.read"]
    return doc_of("gpt2_reach", change)


def test_a_bundle_pulled_into_two_steps_shares_within_each_step_alone(engines):
    """Each step's three points share that step's own forward, so the run makes two forwards, one per step,
    though the two are alike."""
    doc = bundled_doc()
    plan = compiled("gpt2_reach", doc)
    engine = engines("nnterp", plan.model)
    shared = engine.execute(plan, batch_size=4)
    assert costs(shared, "first") == costs(shared, "second") == {"layers=0": (1, all_points(plan, "first"))}
    assert each_its_own(shared, alone(engine, "gpt2_reach", doc, 4))
    assert same(shared.results["first.read"], shared.results["second.read"])


def centered_doc() -> dict:
    """das, with a forward before the fit that only reads, at layer 0 or 1 (`src`), whose mean over rows and
    position (`center`) the body subtracts at layer 0 before the patch: the fit is swept on `src`, one epoch."""
    def change(d):
        d["steps"] = {
            "base": {"kind": "forward", "input": "train.input", "interventions": {
                "read": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "src"}},
                         "pos": -1}}},
            "center": {"kind": "op", "op": "mean", "of": "base.read", "over": ["rows", "pos"]},
            **d["steps"]}
        fit = d["steps"]["fit"]
        fit.update(epochs=1, batch_size=2)
        del fit["early_stop"]
        fit["steps"]["patched"]["interventions"] = [
            {"shift": {"site": "target", "pos": -1, "ops": [{"op": "sub", "with": "center"}]}},
            fit["steps"]["patched"]["interventions"]]
    return doc_of("das", change)


def test_a_later_step_and_a_fit_body_read_each_points_own_value_of_a_shared_forward(engines):
    """`base`'s two points share a forward; `center`, a step over its value, depends on `src`, as does the fit's
    body, which uses `center`, and so the fit's members: every result is what each point's own forward gives, and
    the two points' fits differ."""
    doc = centered_doc()
    plan = compiled("das", doc)
    assert served(plan, "base") == [["src=0", "src=1"]]
    assert axes_of(plan, "center") == ["src"] and axes_of(plan, "fit.train.iia") == ["src"]
    assert [p.at for p in points(plan.steps["fit"])] == [{"src": 0}, {"src": 1}]
    engine = engines("nnterp", plan.model)
    shared = engine.execute(plan, batch_size=BATCH_SIZE)
    assert costs(shared, "base") == {"src=0": (1, ["src=0", "src=1"])}
    assert [c for c, _ in costs(shared, "fit").values()] == [3, 3]        # one minibatch, its scoring, the eval
    assert each_its_own(shared, alone(engine, "das", doc))
    scored = list(shared.results["fit.train.iia"].values())
    assert not same(scored[0], scored[1])


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_swept_read_with_its_own_select_takes_its_own_slice_of_the_shared_forward(engines, kind):
    """head_patching's counterfactual reads one head per point from one forward: each point's value is its own
    head's slice of a read of every head."""
    def change(d):
        d["steps"]["counterfactual"]["interventions"]["every"] = {
            "site": {"component": "attention_premix", "layers": 0}, "pos": -1}
        d["saves"] = ["counterfactual.v_cf", "counterfactual.every"]
    plan = compiled("head_patching", doc_of("head_patching", change))
    assert axes_of(plan, "counterfactual.v_cf") == ["num_heads"] and axes_of(plan, "counterfactual.every") == []
    filled = engines(kind, plan.model).execute(plan, batch_size=BATCH_SIZE)
    assert costs(filled, "counterfactual") == {"num_heads=0": (2, all_points(plan, "counterfactual"))}
    every = filled.results["counterfactual.every"]
    assert every.shape == (4, 1, 4, 4)
    for head, value in filled.results["counterfactual.v_cf"].items():
        assert value.shape == (4, 1, 1, 4) and torch.equal(value[:, :, 0], every[:, :, head])


def corpus_plan(corpus: str, name: str) -> tuple[Plan, dict]:
    path = os.path.join(corpus, "documents", f"{name}.json")
    with open(path) as f:
        doc = json.load(f)
    return Plan(path).compile(), doc


# The corpus documents a point of which is served by another's forward: each a step that reads at swept layers,
# heads, neurons, frames, featurizer seeds or op arguments (expert_knockout's expert) beside writes that do not vary,
# or, in dbm, that reads through a featurizer the swept fit trained, an encode off the model.
SHARE = ["attention_pattern_patching", "dbm", "expert_knockout", "generate_probe", "head_patching", "logit_lens",
         "neuron_patching", "random_subspace"]


def test_the_corpus_documents_whose_points_share_a_forward(corpus):
    assert [n for n in NAMES if sharing(corpus_plan(corpus, n)[0])] == SHARE


@pytest.mark.parametrize("name, kind", [(n, k) for n in SHARE for k in ("nnterp", "hooks")
                                        if not (k == "hooks" and n in INSIDE)])
def test_a_corpus_document_that_shares_gives_each_point_its_own_forwards_values(engines, corpus, name, kind):
    """Its results, positions and tokens are those each point's own document gives, to the bit, with fewer model
    calls than points."""
    plan, doc = corpus_plan(corpus, name)
    engine = engines(kind, plan.model)
    shared = engine.execute(plan, batch_size=BATCH_SIZE)
    theirs = {}
    for key, point in alone(engine, name, doc).items():      # the corpus's own artifacts: the document resolves there
        theirs[key] = point
    assert each_its_own(shared, theirs)
    assert total(shared) < sum(total(p) for p in theirs.values())
    assert sharing(shared) and all(not sharing(p) for p in theirs.values())


def test_an_op_over_a_swept_value_depends_on_the_values_axes():
    """head_patching's `iia` is a mean over `patched.hit`, which depends on the head patched: so does `iia`; and a
    detokenize over generate_probe's ids, which depend on no axis, depends on none, one value for its points."""
    plan = plan_of("head_patching")
    assert axes_of(plan, "patched.logits") == ["num_heads"] and axes_of(plan, "iia") == ["num_heads"]
    probe = compiled("generate_probe", probe_doc())
    assert axes_of(plan_of("generate_probe"), "patched") == [] and axes_of(probe, "text") == []
    assert [p.at for p in points(probe.steps["text"])] == [{"frame": i} for i in range(3)]
    assert [(p.at, [q.at for q in s]) for p, s in serving(probe, "text")] == \
        [({"frame": 0}, [{"frame": k} for k in range(3)])]


def test_an_op_whose_swept_operand_names_values_on_other_axes_gives_each_point_its_own(engines):
    """An op whose operand is a swept name (`other`) picking a read on no axis or one swept over `pos`: its points
    depend on different axes, its runs, and the mean's after it, serve the points that agree with each on its own
    (the first operand's two `pos` share one), and each point's value is what its own document gives, to the bit."""
    def change(d):
        d["steps"] = {"read": {"kind": "forward", "input": "pairs.input", "interventions": {
                          "one": {"site": "target", "pos": -1},
                          "many": {"site": "target", "pos": {"sweep": [-1, -2]}}}},
                      "cos": {"kind": "op", "op": "cosine", "of": "read.one",
                              "with": {"sweep": ["read.one", "read.many"], "label": "other"}},
                      "avg": {"kind": "op", "op": "mean", "of": "cos"}}
        d["saves"] = ["cos", "avg"]
    doc = doc_of("pos_sweep", change)
    plan = compiled("pos_sweep", doc)
    assert [p.axes for p in points(plan.steps["cos"])] == [["other"], ["pos", "other"]] * 2
    runs = [['pos=-1,other="read.one"', 'pos=-2,other="read.one"'], ['pos=-1,other="read.many"'],
            ['pos=-2,other="read.many"']]
    assert [[labelled(plan, q.at) for q in s] for _, s in serving(plan, "cos")] == runs
    assert [[labelled(plan, q.at) for q in s] for _, s in serving(plan, "avg")] == runs
    engine = engines("nnterp", plan.model)
    assert each_its_own(engine.execute(plan, batch_size=BATCH_SIZE), alone(engine, "pos_sweep", doc))


def test_the_manifest_keys_each_result_by_its_own_axes_and_each_cost_by_the_forward_that_ran(engines, tmp_path):
    plan = compiled("generate_probe", probe_doc())
    engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).dump(str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    results = manifest["results"]
    assert results["patched"]["axes"] == [] and results["patched"]["shape"] == [4, 3]
    assert results["p_answer"]["axes"] == ["frame"] and list(results["p_answer"]["shape"]) == ["frame=0", "frame=1", "frame=2"]
    assert results["text"]["axes"] == [] and results["text"]["shape"] == [4]
    assert manifest["run"]["costs"]["patched"] == {"frame=0": {
        **{k: manifest["run"]["costs"]["patched"]["frame=0"][k] for k in ("seconds", "memory")},
        "forwards": 6, **{k: manifest["run"]["costs"]["patched"]["frame=0"][k] for k in ("tokens", "real")},
        "serves": ["frame=0", "frame=1", "frame=2"]}}
    assert "serves" not in manifest["run"]["costs"]["counterfactual"]
    steps = manifest["steps"]["patched"]
    assert set(steps[0]["schedule"]) == {"axes", "sites", "stop", "loop", "online"}
    assert [s["schedule"] for s in steps[1:]] == [{"serving": {"frame": 0}}] * 2
    assert "schedule" not in manifest["steps"]["p_answer"][0]
    back = Plan.load(str(tmp_path))
    assert same(back.results, plan.results)


def called(step: Forward, rows: int, batch_size: int) -> list:
    """The model calls one run of a serving point makes over `rows` rows in batches of `batch_size`, as its
    schedule says: per batch a forward's one call, a generation's `max_new_tokens`, each `"call"` then the sites it
    visits in order (a forward's every site, a generation's call the sites with a record at that call,
    `batches.now`)."""
    sites = step.schedule["sites"]
    calls = range(cast(int, horizon(cast(Any, step)))) if step.kind == "generate" else [0]
    one = [tag for k in calls for tag in ["call", *((s["component"], s["layer"]) for s in sites
                                                  if batches.now(s["records"], k))]]
    return one * -(-rows // batch_size)


def ran(plan: Plan, bodies: list[tuple]) -> list:
    """Every model call the schedules of `plan` make, in order, as `called` lists them: each serving point of a
    root forward or generate step over its dataset's rows, at the suite's batch size; at each fit point, each run
    of its body the fit made (`bodies`: its name, point, rows and the plan it ran over, in order, as many as its
    epochs and evals made, which its early stop decides), marked `"body"`, each body forward over those rows at
    the fit's batch size."""
    out: list = []
    for name, entry in plan.steps.items():
        for p, _ in serving(plan, name):
            if isinstance(p, Forward):
                out += called(p, cast(int, plan.data[cast(str, p.dataset)].n_rows), BATCH_SIZE)
            if not isinstance(p, Fit):
                continue                    # a forward's are listed, an op calls no model
            for _, _, rows, source in (b for b in bodies if b[:2] == (name, p.at)):
                out.append("body")
                for s in p.steps.values():
                    if isinstance(s, Forward):
                        n = rows if rows is not None else cast(int, source.data[cast(str, s.dataset)].n_rows)
                        out += called(s, n, cast(int, p.batch_size))
    return out


@pytest.mark.parametrize("name, kind", [(n, k) for n in NAMES for k in ("nnterp", "hooks")
                                        if not (k == "hooks" and n in INSIDE)])
def test_the_schedule_is_what_ran(engines, corpus, name, kind, monkeypatch):
    """On every corpus document and both local engines, the model calls (a pre-hook on its root module) and the
    sites each visited (`batches.visit`), in order, are exactly those the schedules give (`ran`): one forward per
    batch of each serving point, a generation's `max_new_tokens` calls per batch, each visiting its schedule's
    sites in order, and a fit's body the same in each run the fit made of it."""
    plan, _ = corpus_plan(corpus, name)
    engine: Any = engines(kind, plan.model)
    engine.execute(plan, batch_size=BATCH_SIZE)                 # what the model does once is done
    plan, _ = corpus_plan(corpus, name)                         # a run leaves a generation's landings on the records
    seen: list[Any] = []
    bodies: list[tuple] = []
    visit, body_run = batches.visit, fit.body_run

    def noted(value, component, records, got):
        if records:                 # the hooks engine's hook of a site with no record at this call does nothing
            seen.append((component, records[0]["layer"]))
        return visit(value, component, records, got)

    def body(engine, name, step, state, trained, source, ids, cost, reading):
        bodies.append((name, step.at, None if ids is None else len(ids), source[0]))
        seen.append("body")
        return body_run(engine, name, step, state, trained, source, ids, cost, reading)
    monkeypatch.setattr(batches, "visit", noted)
    monkeypatch.setattr(fit, "body_run", body)
    root = getattr(engine.model, "_module", engine.model)
    handle = root.register_forward_pre_hook(lambda module, args: seen.append("call"))
    try:
        filled = engine.execute(plan, batch_size=BATCH_SIZE)
    finally:
        handle.remove()
    assert seen == ran(filled, bodies)
