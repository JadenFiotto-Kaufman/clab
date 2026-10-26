"""One forward shared by the points of a step that differ only in what they read.

Compile decides it from the points alone (`Forward.shares`): points of one forward or generate step with the same
input, options and writes share the first one's forward, which carries every point's reads, each bound at its own
point. Each point's values are then those its own forward gives, to the bit (`unshared` is the run as it was
before points shared: every `shares` cleared), on every engine, under `remote="local"` and through the stand-in
server; the point that runs the forward records its calls and tokens and every other point none. A sweep on a
write (its operand, positions or layer) runs a forward per write, as does one reached by a write and a read (zip);
a grid of a swept write and a swept read shares within each write; a write whose operand is an earlier entry of its
forward shares while that entry does not vary; a generation shares as a forward does; a bundle pulled into two steps
shares within each step, never across them; and a later step or a fit's body reads each point's value as before.
"""

import os
from typing import Any, Callable

import pytest
from conftest import BATCH_SIZE, INSIDE, NAMES, TRIP, edited, same, stand_in
from test_efficiency import at, counting

from clab import Plan
from clab.format.sweeps import labelled, points
from clab.plan import Forward


def unshared(plan: Plan) -> Plan:
    """`plan` with every point running its own forward, as before points shared one."""
    for entry in plan.steps.values():
        for step in points(entry):
            if isinstance(step, Forward):
                step.shares = None
    return plan


def shares(plan: Plan, name: str) -> list[str | None]:
    return [p.shares for p in points(plan.steps[name])]


def sharing(plan: Plan) -> bool:
    """Whether a point of `plan` shares another's forward."""
    return any(isinstance(p, Forward) and p.shares is not None for e in plan.steps.values() for p in points(e))


def calls(plan: Plan, name: str) -> list[int]:
    """Each point's model calls, in point order."""
    assert plan.run is not None
    return [c["forwards"] for c in plan.run.costs[name].values()]


def total(plan: Plan) -> int:
    """Every model call the run made."""
    assert plan.run is not None
    return sum(c["forwards"] for costs in plan.run.costs.values()
               for c in ([costs] if "forwards" in costs else costs.values()))


def twice(engine, make: Callable[[], Plan], batch_size: int = BATCH_SIZE) -> tuple[Plan, Plan]:
    """The plan `make` builds, run as it compiled and `unshared`."""
    return (engine.execute(make(), batch_size=batch_size),
            engine.execute(unshared(make()), batch_size=batch_size))


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


def gpt2(entries: dict[str, Any], **root: Any) -> Plan:
    """Tiny GPT-2 (five blocks) over gpt2_reach's four pairs, one forward `lens` with `entries`, each saved;
    `root` adds root keys."""
    def change(d):
        d.update(root)
        d["steps"] = {"lens": {"kind": "forward", "input": "pairs.input", "interventions": entries}}
        d["saves"] = [f"lens.{k}" for k in entries]
    return edited("gpt2_reach", change).compile()


ZERO = [{"op": "swap", "with": 0.0}]
READS = {
    "layers": {"read": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4]}}, "pos": -1,
                        "ops": [{"op": "readout"}]}},
    "pos": {"read": {"site": {"component": "layer_output", "layers": 2}, "pos": {"sweep": [-1, -2, -3, -4, -5]}}},
}


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("axis", READS)
def test_a_read_only_sweep_runs_one_forward_and_each_point_reads_what_its_own_would(engines, kind, axis):
    """Five points that read at five layers (a readout of each), or at five positions: one forward of the four
    rows in one batch, which the first point records and the others do not; the blocks run once, to the deepest
    point's (each point alone runs to its own; the ndif engine runs every forward whole); each point's value is
    its own forward's to the bit."""
    plan = gpt2(READS[axis])
    first = labelled(plan, points(plan.steps["lens"])[0].at)
    assert first == f"{axis}={0 if axis == 'layers' else -1}" and shares(plan, "lens") == [None] + [first] * 4
    engine = engines(kind, plan.model, **ENGINES[kind])
    shared, ran = counted(engine, plan, 4)
    alone, ran_alone = counted(engine, unshared(gpt2(READS[axis])), 4)
    deepest = [4] * 5 if kind == "ndif" else [0, 1, 2, 3, 4] if axis == "layers" else [2] * 5
    assert ran == [*range(max(deepest) + 1)] and ran_alone == [b for layer in deepest for b in range(layer + 1)]
    assert shared.run is not None and alone.run is not None
    costs = list(shared.run.costs["lens"].values())
    assert calls(shared, "lens") == [1, 0, 0, 0, 0] and calls(alone, "lens") == [1] * 5
    assert [c["tokens"] for c in costs][1:] == [0] * 4 and all(c["memory"] is None for c in costs)
    assert same(shared.results, alone.results) and shared.run.positions == alone.run.positions
    values = list(shared.results["lens.read"].values())
    assert not any(same(values[0], v) for v in values[1:])


def test_the_stand_in_server_shares_the_forward_as_the_client_would(engines, monkeypatch):
    """The serialized plan carries `shares` and the server runs the same `steps.run`: on its model the five
    points' two batches run the blocks once each, and what comes home is the local run's."""
    plan = gpt2(READS["layers"])
    local = engines("nnterp", plan.model).execute(gpt2(READS["layers"]), batch_size=BATCH_SIZE)
    with counting(engines("nnterp", plan.model)) as ran:
        came, trip = stand_in(engines, plan, monkeypatch)
    assert trip == TRIP and blocks(ran) == [*range(5)] * 2
    assert calls(came, "lens") == [2, 0, 0, 0, 0]
    assert same(came.results, local.results)


WRITES = {
    "operand": {"scaled": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                           "ops": [{"op": "scale", "factor": {"sweep": [0.0, 0.5, 2.0]}}]}},
    "position": {"zero": {"site": {"component": "layer_output", "layers": 1}, "pos": {"sweep": [-1, -2, -3]},
                          "ops": ZERO}},
    "layer": {"zero": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}, "pos": -1, "ops": ZERO}},
}


@pytest.mark.parametrize("swept", WRITES)
def test_a_sweep_on_a_write_runs_a_forward_per_point(engines, swept):
    """A write's operand, positions or layer swept: every point writes otherwise, so each runs its own forward,
    and the read after the writes sees each."""
    plan = gpt2({**WRITES[swept], "read": at("layer_output", 3)})
    assert shares(plan, "lens") == [None] * 3
    filled = engines("nnterp", plan.model).execute(plan, batch_size=4)
    assert calls(filled, "lens") == [1, 1, 1]
    values = list(filled.results["lens.read"].values())
    assert not same(values[0], values[1]) and not same(values[1], values[2])


GRID = {"read": {"site": {"component": "layer_output", "layers": {"sweep": [2, 3, 4], "label": "src"}}, "pos": -1},
        "zero": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1], "label": "dst"}}, "pos": -1,
                 "ops": ZERO}}


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_grid_of_a_swept_write_and_a_swept_read_shares_within_each_write(engines, kind):
    """A read at layers 2 to 4 (`src`, the first axis) crossed with a write at layer 0 or 1 (`dst`): the points
    with one write share its first point's forward, so the six make two; the run record keeps point order; each
    value is its own forward's."""
    plan = gpt2(GRID)
    assert list(plan.sweeps) == ["src", "dst"]
    assert shares(plan, "lens") == [None, None, "src=2,dst=0", "src=2,dst=1", "src=2,dst=0", "src=2,dst=1"]
    shared, alone = twice(engines(kind, plan.model), lambda: gpt2(GRID), batch_size=4)
    assert shared.run is not None
    assert list(shared.run.costs["lens"]) == [labelled(plan, p.at) for p in points(plan.steps["lens"])]
    assert calls(shared, "lens") == [1, 1, 0, 0, 0, 0] and calls(alone, "lens") == [1] * 6
    assert same(shared.results, alone.results)


def test_one_sweep_reached_by_a_read_and_a_write_shares_nothing(engines):
    """A swept site that a write and a read both use is one axis (zip): each point writes at its own layer."""
    plan = gpt2({"zero": {"site": "swept", "pos": -1, "ops": ZERO}, "read": {"site": "swept", "pos": -2}},
                sites={"swept": {"component": "layer_output", "layers": {"sweep": [0, 1, 2]}}})
    assert list(plan.sweeps) == ["layers"] and shares(plan, "lens") == [None] * 3
    filled = engines("nnterp", plan.model).execute(plan, batch_size=4)
    assert calls(filled, "lens") == [1, 1, 1]


LIVE = {"premix": {"site": {"component": "layer_output", "layers": 0}, "pos": -1},
        "added": {"site": {"component": "layer_output", "layers": 1}, "pos": -1,
                  "ops": [{"op": "add", "with": "lens.premix"}]}}


def live() -> Plan:
    """Tiny GPT-2: a write at block 1 that adds what an earlier entry of its forward read at block 0, beside a read
    swept over blocks 2 to 4."""
    return gpt2({**LIVE, "read": {"site": {"component": "layer_output", "layers": {"sweep": [2, 3, 4]}}, "pos": -1}})


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_write_from_an_earlier_entry_of_its_forward_shares_while_that_entry_does_not_vary(engines, kind):
    """The three points share the forward, whose write takes the first point's read of the earlier entry, which
    is every point's; with that entry swept over blocks 0 and 1 instead, the write varies, and each point runs its
    own forward."""
    plan = live()
    assert shares(plan, "lens") == [None, "layers=2", "layers=2"]
    shared, alone = twice(engines(kind, plan.model), live, batch_size=4)
    assert calls(shared, "lens") == [1, 0, 0]
    assert same(shared.results, alone.results)
    swept = gpt2({**LIVE, "premix": {"site": {"component": "layer_output", "layers": {"sweep": [0, 1]}}, "pos": -1},
                  "read": at("layer_output", 3)})
    assert shares(swept, "lens") == [None, None]


def probe() -> Plan:
    """generate_probe, its patched generation's ids also as text."""
    def change(d):
        d["steps"]["text"] = {"kind": "op", "op": "detokenize", "of": "patched"}
        d["saves"].append("text")
    return edited("generate_probe", change).compile()


@pytest.mark.parametrize("kind", ["nnterp", "hooks"])
def test_a_generations_read_only_sweep_shares_the_generation(engines, kind):
    """generate_probe's patched generation reads the head at frame 0, 1 or 2 beside one write: its three points
    share one generation (two batches of three calls), and the ids, the text, each row's tokens and where each
    read landed are each point's own generation's."""
    plan = probe()
    assert shares(plan, "patched") == [None, "frame=0", "frame=0"]
    shared, alone = twice(engines(kind, plan.model), probe)
    assert shared.run is not None and alone.run is not None
    assert calls(shared, "patched") == [6, 0, 0] and calls(alone, "patched") == [6, 6, 6]
    assert same(shared.results, alone.results)
    assert shared.run.tokens == alone.run.tokens and shared.run.positions == alone.run.positions


def bundled() -> Plan:
    """Tiny GPT-2 with a bundle whose read is swept over three layers, pulled into two alike steps."""
    def change(d):
        d["interventions"] = {"lens": {"read": {"site": {"component": "layer_output",
                                                         "layers": {"sweep": [0, 1, 2]}}, "pos": -1}}}
        d["steps"] = {name: {"kind": "forward", "input": "pairs.input", "interventions": "lens"}
                      for name in ("first", "second")}
        d["saves"] = ["first.read", "second.read"]
    return edited("gpt2_reach", change).compile()


def test_a_bundle_pulled_into_two_steps_shares_within_each_step_alone(engines):
    """Each step's three points share that step's own forward, so the run makes two forwards, one per step,
    though the two are alike."""
    plan = bundled()
    assert shares(plan, "first") == shares(plan, "second") == [None, "layers=0", "layers=0"]
    shared, alone = twice(engines("nnterp", plan.model), bundled, batch_size=4)
    assert calls(shared, "first") == calls(shared, "second") == [1, 0, 0]
    assert same(shared.results, alone.results)
    assert same(shared.results["first.read"], shared.results["second.read"])


def centered() -> Plan:
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
            {"shift": {"site": "target", "pos": -1, "ops": [{"op": "sub", "with": "center"}]}}, "das"]
    return edited("das", change).compile()


def test_a_later_step_and_a_fit_body_read_each_points_own_value_of_a_shared_forward(engines):
    """`base`'s two points share a forward; `center`, a step over its value, and the fit's body, which uses
    `center`, each read it at their own point: every result is what each point's own forward gives, and the
    two points' fits differ."""
    plan = centered()
    assert shares(plan, "base") == [None, "src=0"]
    assert [p.at for p in points(plan.steps["fit"])] == [{"src": 0}, {"src": 1}]
    shared, alone = twice(engines("nnterp", plan.model), centered)
    assert calls(shared, "base") == [1, 0] and calls(alone, "base") == [1, 1]
    assert same(shared.results, alone.results)
    held_out = list(shared.results["fit.held_out.iia"].values())
    assert not same(held_out[0], held_out[1])


def corpus_plan(corpus: str, name: str) -> Plan:
    return Plan(os.path.join(corpus, "documents", f"{name}.json")).compile()


# The corpus documents a point of which shares a forward: each a step that reads at swept layers, heads, neurons,
# frames or featurizer seeds beside writes that do not vary.
SHARE = ["attention_pattern_patching", "generate_probe", "head_patching", "logit_lens", "neuron_patching",
         "random_subspace"]


def test_the_corpus_documents_whose_points_share_a_forward(corpus):
    assert [n for n in NAMES if sharing(corpus_plan(corpus, n))] == SHARE


@pytest.mark.parametrize("name, kind", [(n, k) for n in SHARE for k in ("nnterp", "hooks")
                                        if not (k == "hooks" and n in INSIDE)])
def test_a_corpus_document_that_shares_gives_each_point_its_own_forwards_values(engines, corpus, name, kind):
    """Its results, positions and tokens are those of the run where every point runs its own forward, to the
    bit, with fewer model calls."""
    plan = corpus_plan(corpus, name)
    shared, alone = twice(engines(kind, plan.model), lambda: corpus_plan(corpus, name))
    assert shared.run is not None and alone.run is not None
    assert same(shared.results, alone.results)
    assert shared.run.positions == alone.run.positions and shared.run.tokens == alone.run.tokens
    assert total(shared) < total(alone)
