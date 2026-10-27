"""Tensor parallelism, the nnterp engine's `tp` (`clab run DOC -o tp=N` under `torchrun --nproc-per-node=N`): what
is refused before anything loads, which components every rank holds whole (`split`, read off the model's plan),
rank 0 alone writing and printing, and the tiny Llama (and a tiny hybrid, where it is cached) at tp 2 with its two
ranks run by torchrun on the CPU (gloo) and, given two GPUs, on CUDA (NCCL): each rank's results the other's to the
bit, tp 1's to rounding, its generated text tp 1's, and a site the ranks hold in pieces refused on both."""

import os
import subprocess
import sys
from typing import Any

import pytest
import torch
from conftest import BATCH_SIZE, ROOT, document, edited, plan_of, same
from huggingface_hub import try_to_load_from_cache

import clab
from clab.cli import main
from clab.engines import load
from clab.engines.nnterp import split
from clab.format.address import COMPONENTS
from tp_ranks import HYBRID, planned

# What the ranks hold in pieces on a Llama at every layer: the attention's interior but its output projection's input
# (which nnsight gathers), and the MLP's activation. The embedding's output is in pieces too where the plan shards
# the embedding by vocabulary, which each test reads off the plan the checkpoint ships.
PIECES = {"attention_queries", "attention_keys", "attention_values", "attention_scores", "attention_probabilities",
          "attention_head_outputs", "mlp_activation"}
HYBRID_CACHED = isinstance(try_to_load_from_cache(HYBRID["key"], "config.json", revision=HYBRID["revision"]), str)
# The runs torchrun makes at tp 2 (`tp_ranks.CHANGES`: `das` for one epoch, `sampled` sampling its generation, the
# hybrid's where it is cached), and those it refuses, each with the site it names first and where it is computed.
SERVED = ["patching", "head_patching", "hydra_effect", "logit_lens", "generated_contains", "generate_probe", "das",
          *(["hybrid_patching"] if HYBRID_CACHED else [])]
REFUSED = {"attention_query": ("counterfactual.v_cf", "attention_queries", "model.layers.0.self_attn", "q_proj"),
           "neuron_patching": ("counterfactual.v_cf", "mlp_activation", "model.layers.0.mlp", "gate_proj"),
           "attention_knockout": ("clean.pattern", "attention_probabilities", "model.layers.0.self_attn", "q_proj"),
           **({"hybrid_queries": ("counterfactual.v_cf", "attention_queries", "model.layers.3.self_attn", "q_proj")}
              if HYBRID_CACHED else {})}
GPUS = torch.cuda.device_count()


@pytest.fixture
def unloadable(monkeypatch):
    """No engine can load a model: a refusal must come before anything loads."""
    for where in ("nnterp", "hooks", "ndif"):
        monkeypatch.setattr(f"clab.engines.{where}.StandardizedTransformer", lambda *a, **k: pytest.fail("loaded"))
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_tp_outside_torchrun_is_refused_with_the_command(unloadable):
    with pytest.raises(ValueError, match=r"^tp 2 shards the model over 2 processes, one per rank, that torchrun "
                                         r"starts: `torchrun --nproc-per-node=2 --no-python clab run <document> "
                                         r"-o tp=2`; this process is not one of them \(WORLD_SIZE is unset\)$"):
        load("nnterp", plan_of("patching").model, tp=2)
    with pytest.raises(ValueError, match="^tp 0: tensor parallelism shards the model over at least one rank$"):
        load("nnterp", plan_of("patching").model, tp=0)


@pytest.mark.parametrize("world, tp", [("2", 1), ("4", 2)])
def test_a_world_size_other_than_tp_is_refused(unloadable, world, tp):
    """Each process torchrun starts runs the whole plan, so their number is the degree the model is sharded to."""
    unloadable.setenv("WORLD_SIZE", world)
    with pytest.raises(ValueError, match=rf"^tp {tp}: torchrun started {world} processes \(WORLD_SIZE\), each of which "
                                         rf"runs the whole plan, and tp shards the model over {tp}; give both the one "
                                         rf"number: `torchrun --nproc-per-node={world} --no-python clab run "
                                         rf"<document> -o tp={world}`$"):
        load("nnterp", plan_of("patching").model, tp=tp)


def test_under_tp_the_device_is_where_transformers_puts_each_shard(unloadable):
    unloadable.setenv("WORLD_SIZE", "2")
    kind, where = ("cuda", "on its own GPU, cuda:<LOCAL_RANK>") if torch.cuda.is_available() else \
        ("cpu", "on the CPU, no GPU being visible")
    with pytest.raises(ValueError, match=rf"^device 'cuda:0': under tp 2 transformers puts each rank's shard {where}, "
                                         rf"so the device is '{kind}'$"):
        load("nnterp", plan_of("patching").model, "cuda:0", tp=2)


def test_under_tp_a_node_needs_a_gpu_per_rank(unloadable):
    for key, value in (("WORLD_SIZE", "2"), ("LOCAL_WORLD_SIZE", "2")):
        unloadable.setenv(key, value)
    unloadable.setattr("torch.cuda.is_available", lambda: True)
    unloadable.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match=r"^tp 2: each rank holds its shard on its own GPU, and torchrun started 2 "
                                         r"ranks on this node, which shows 1 \(CUDA_VISIBLE_DEVICES\)$"):
        load("nnterp", plan_of("patching").model, "cuda", tp=2)


def test_a_bitsandbytes_dtype_is_refused_under_tp(unloadable):
    """Its packed weights do not shard: refused by name, not inside bitsandbytes after the load."""
    unloadable.setenv("WORLD_SIZE", "2")
    plan = edited("patching", lambda d: d["model"].update(dtype="nf4"))
    with pytest.raises(ValueError, match="^model.dtype 'nf4': bitsandbytes' packed weights do not shard across ranks, "
                                         "so it runs at tp 1 alone, not tp 2$"):
        load("nnterp", plan.model, "cuda" if torch.cuda.is_available() else "cpu", tp=2)


def test_a_checkpoint_nnsight_cannot_split_is_refused_naming_tp_and_the_checkpoint(unloadable):
    """GPT-2 publishes no tensor-parallel plan."""
    unloadable.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match=r"^tp 2: hf-internal-testing/tiny-random-gpt2 cannot be sharded 2 ways: its "
                                         r"config gives no tensor-parallel plan nnsight can gather, or its sizes "
                                         r"divide no degree above 1; run it at tp 1, on one GPU or spread over several "
                                         r"with --device auto$"):
        load("nnterp", plan_of("gpt2_reach").model, "cuda" if torch.cuda.is_available() else "cpu", tp=2)


def test_the_other_engines_refuse_tp_and_torchrun_by_name(unloadable, capsys):
    """The ndif engine's docstring says why it takes no `tp`: the server shards the model as its deployment does."""
    model = plan_of("patching").model
    for kind, theirs in (("hooks", "device"), ("ndif", "device, remote")):
        with pytest.raises(ValueError, match=f"^the {kind} engine takes no 'tp'; its options: {theirs}$"):
            load(kind, model, tp=2)
    unloadable.setenv("WORLD_SIZE", "2")
    for kind in ("hooks", "ndif"):              # each rank would run the whole plan, under ndif each its own job
        with pytest.raises(ValueError, match=f"^WORLD_SIZE 2: torchrun started 2 processes, and the {kind} engine "
                                             "takes no `tp` to shard the model over them, so each would run the whole "
                                             "plan; run it without torchrun$"):
            load(kind, model)
    unloadable.delenv("WORLD_SIZE")
    assert main(["run", document("patching"), "-o", "tp=2"]) == 1
    assert capsys.readouterr().err.startswith("clab: tp 2 shards the model over 2 processes")


@pytest.mark.parametrize("rank, printed", [("0", True), ("1", False)])
def test_a_refusal_prints_once_under_torchrun(unloadable, capsys, rank, printed):
    unloadable.setenv("RANK", rank)
    assert main(["run", document("patching"), "-o", "tp=2"]) == 1
    assert bool(capsys.readouterr().err) == printed


def shell(key: str, revision: str):
    from nnterp import StandardizedTransformer
    return StandardizedTransformer(key, revision=revision, dispatch=False, device_map=None,
                                   attn_implementation="eager")


def test_split_reads_which_components_are_whole_off_the_models_plan():
    """On the tiny Llama's plan (q, k, v, gate and up `colwise`, o and down `rowwise`, the head
    `colwise_gather_output`) the attention's interior and the MLP's activation are in pieces, at every layer, and
    everything else is whole: the residual stream, the norms, the attention's premix (the output projection's input,
    which nnsight gathers) and output, the MLP's neurons (the down projection's) and output, the head, the logits,
    and the embedding's output where the plan leaves the embedding replicated; whether it does differs between
    transformers versions (an untied embedding is replicated by some, sharded by vocabulary as `embedding_rowwise` by
    others), so the plan the model holds says which."""
    tiny = shell("hf-internal-testing/tiny-random-LlamaForCausalLM", "9fb191250dd56d0ba7ec9785a025ed29c03d5998")
    plan: Any = tiny._module.tp_plan              # transformers' plan, one style per module
    assert {plan[f"model.layers.*.{m}"] for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                   "mlp.gate_proj", "mlp.up_proj")} == {"colwise"}
    assert {plan["model.layers.*.self_attn.o_proj"], plan["model.layers.*.mlp.down_proj"]} == {"rowwise"}
    assert plan["lm_head"] == "colwise_gather_output"
    embedding = {"token_embeddings"} if plan.get("model.embed_tokens") == "embedding_rowwise" else set()
    for layer in (0, 1):
        assert {c for c in COMPONENTS if split(tiny, c, layer if COMPONENTS[c]["per_layer"] else None)} == \
            PIECES | embedding
    assert split(tiny, "attention_scores", 1) == ("it is computed inside model.layers.1.self_attn, whose forward runs "
                                                  "on each rank's piece (model.layers.1.self_attn.q_proj is 'colwise')")


@pytest.mark.skipif(not HYBRID_CACHED, reason=f"needs {HYBRID['key']} in the local Hugging Face cache")
def test_on_a_hybrid_tp_refuses_a_value_in_pieces_at_the_blocks_that_have_it():
    """The tiny hybrid's blocks 0 to 2 have no `self_attn`: there nnterp's own reason stands, and only block 3's
    queries are asked of `split`, which refuses them; the residual stream is served."""
    from clab.engines.nnterp import NnterpEngine

    engine: Any = object.__new__(NnterpEngine)              # the engine's `unserved` over the hybrid's meta shell
    engine.model, engine.options = shell(HYBRID["key"], HYBRID["revision"]), {"tp": 2}
    reasons = engine.unserved()
    assert reasons["attention_queries"] == {
        **{layer: "no self_attn module on this block" for layer in (0, 1, 2)},
        3: "under tp 2 it is computed inside model.layers.3.self_attn, whose forward runs on each rank's piece "
           "(model.layers.3.self_attn.q_proj is 'colwise')"}
    assert reasons["layer_output"] is None and reasons["attention_output"] == {
        layer: "no self_attn module on this block" for layer in (0, 1, 2)}


@pytest.mark.skipif(not isinstance(try_to_load_from_cache("meta-llama/Llama-3.2-1B", "config.json",
                                                          revision="4e20de362430cd3b72f300e6b0f18e50e7166e08"), str),
                    reason="needs meta-llama/Llama-3.2-1B's config in the local Hugging Face cache")
def test_split_refuses_a_vocab_parallel_embeddings_output():
    """Llama-3.2-1B's plan shards its tied embedding by vocabulary (`embedding_rowwise`), whose output nnsight
    cannot reassemble: `token_embeddings` is refused too, and `layer_input` at layer 0, the same tensor, is whole."""
    llama = shell("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08")
    assert {c for c in COMPONENTS if split(llama, c, 0 if COMPONENTS[c]["per_layer"] else None)} == \
        PIECES | {"token_embeddings"}
    assert split(llama, "token_embeddings", None) == ("model.embed_tokens is a vocab-parallel embedding, whose "
                                                      "output nnsight cannot reassemble")


def test_tp_1_in_one_torchrun_process_is_the_plain_run(engines, monkeypatch):
    """`torchrun --nproc-per-node=1 ... -o tp=1` loads as a plain run does, and runs it to the bit."""
    for key, value in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(key, value)
    plan = plan_of("patching")
    filled = load("nnterp", plan.model, tp=1).execute(plan_of("patching"), batch_size=BATCH_SIZE)
    assert same(filled.results, engines("nnterp", plan.model).execute(plan, batch_size=BATCH_SIZE).results)
    assert filled.run is not None and filled.run.options == {"tp": 1}


def test_rank_0_alone_writes_the_output_and_prints(engines, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RANK", "1")
    plan = plan_of("patching")
    filled = clab.run(plan, engine=engines("nnterp", plan.model), out=str(tmp_path / "a"), batch_size=BATCH_SIZE)
    assert filled.results and not (tmp_path / "a").exists()
    assert main(["run", document("patching"), "--batch-size", str(BATCH_SIZE), "--out", str(tmp_path / "b")]) == 0
    assert capsys.readouterr().out == "" and not (tmp_path / "b").exists()


def torchrun(out: str, device: str, names: list[str], env: dict) -> dict:
    """Every run `names` lists at tp 2 on `device`, as torchrun's two ranks got it."""
    done = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                           os.path.join(ROOT, "tests", "tp_ranks.py"), "2", device, out, *names],
                          env={**os.environ, "HF_HUB_OFFLINE": "1", **env}, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return {n: [torch.load(os.path.join(out, f"{n}.rank{r}.pt"), weights_only=False) for r in (0, 1)] for n in names}


@pytest.fixture(scope="module")
def ranks(tmp_path_factory) -> tuple[dict, str]:
    """Every run of `SERVED`, the sampled ones and `REFUSED` at tp 2 on the CPU (gloo), and the directory rank 0
    wrote them to."""
    out = str(tmp_path_factory.mktemp("tp"))
    return torchrun(out, "cpu", [*SERVED, "sampled", "sampled_again", *REFUSED], {"CUDA_VISIBLE_DEVICES": ""}), out


@pytest.fixture(scope="module")
def gpu_ranks(tmp_path_factory) -> tuple[dict, str]:
    """`SERVED` and the sampled runs at tp 2 on two GPUs (NCCL)."""
    out = str(tmp_path_factory.mktemp("tp_gpu"))
    return torchrun(out, "cuda", [*SERVED, "sampled", "sampled_again"], {}), out


def agrees(runs: dict, out: str, name: str, device: str, devices: list[str], engines) -> None:
    """Both ranks hold the same bits, and rank 0's run record says tp 2 on every rank's device; tp 1's numbers to
    rounding (a sharded matmul sums its parts in another order), its generated ids and text exactly. A fit's trained
    rotation is tp 1's to 1e-2 only: after an epoch of Adam, whose step divides by the gradient's own scale, a
    coordinate whose gradient is near zero moves by the sign of that rounding."""
    first, second = runs[name]
    assert same(first["results"], second["results"])
    assert first["run"].options == {"tp": 2} and first["run"].devices == devices
    assert os.path.exists(os.path.join(out, name, "manifest.json"))         # rank 0 wrote it
    plan = planned(name).compile()
    one = dict(engines("nnterp", plan.model, device=device).execute(plan, batch_size=BATCH_SIZE).results)
    sharded = dict(first["results"])
    if name == "das":
        torch.testing.assert_close(sharded.pop("fit.rot"), one.pop("fit.rot"), atol=1e-2, rtol=0)
    for key, value in one.items():
        if isinstance(value, list) or isinstance(value, torch.Tensor) and not value.is_floating_point():
            assert same(sharded[key], value), key                           # ids and text
        else:
            torch.testing.assert_close(sharded[key], value, atol=1e-5, rtol=1e-5, equal_nan=True, msg=key)
    # Not bit-equal, which says the model was sharded: generated_contains saves its text and an accuracy alone.
    assert not same(sharded, one) or name == "generated_contains"


@pytest.mark.parametrize("name", SERVED)
def test_at_tp_2_the_ranks_agree_and_are_tp_1_to_rounding(ranks, engines, name):
    runs, out = ranks
    agrees(runs, out, name, "cpu", ["cpu"], engines)


@pytest.mark.skipif(GPUS < 2, reason="needs two visible GPUs")
@pytest.mark.parametrize("name", SERVED)
def test_at_tp_2_on_two_gpus_the_ranks_agree_and_are_tp_1_to_rounding(gpu_ranks, engines, name):
    """The NCCL path: each rank on its own GPU, the seed and the devices passed between them."""
    runs, out = gpu_ranks
    agrees(runs, out, name, "cuda", ["cuda:0", "cuda:1"], engines)


@pytest.mark.parametrize("on", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(GPUS < 2,
                                                                                  reason="needs two visible GPUs"))])
def test_at_tp_2_a_sampled_generation_draws_the_same_tokens_on_every_rank(request, on):
    """Every rank samples from a seed rank 0 draws afresh for each run: without it each rank would draw its own
    tokens and sum the others' activations of other sequences into its own, and with one drawn once a second run in
    the process would repeat the first's draws."""
    runs = request.getfixturevalue("ranks" if on == "cpu" else "gpu_ranks")[0]
    for name in ("sampled", "sampled_again"):
        first, second = runs[name]
        assert same(first["results"], second["results"])
    assert not same(runs["sampled"][0]["results"]["text"], runs["sampled_again"][0]["results"]["text"])


@pytest.mark.parametrize("name", REFUSED)
def test_at_tp_2_a_site_the_ranks_hold_in_pieces_is_refused_on_both(ranks, name):
    where, component, host, child = REFUSED[name]
    first, second = ranks[0][name]
    assert first == second == {"refused": f"{where}: the nnterp engine does not serve {component}: under tp 2 it is "
                                          f"computed inside {host}, whose forward runs on each rank's piece "
                                          f"({host}.{child} is 'colwise')"}
