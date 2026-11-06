"""The vLLM engine: nnterp's StandardizedVLLM, every row of a batch its own request inside one trace per batch.

vLLM runs the model in a worker process of its own, where a request's activations are its own tokens, `[tokens,
...]`, with no batch axis and no padding, and where a traced block runs once per request. So this engine takes the
batches `steps.run` hands it as every engine does and executes each batch's bound schedule row by row: one trace per
batch, one invoke per row, carrying the row's own token ids and the half of the bound schedule that is the row's
(`batches.schedule_for_row`). In the worker, `visit_request` visits each site as the nnterp engine's `visited` does,
on the value nnterp serves there (`[1, tokens, ...]`, a private copy) padded on the left to the batch's width, so that
the bound records' indices into the padded batch hold as they are, `batches.visit_site` doing the site's records, and
the value handed back unpadded where a record wrote. Each row's reads come home and are joined along rows (`joined`)
into what the nnterp engine's one forward returns, NaN (-1 in ids) at a decode call a row never made. Nothing is
decided here that the schedule does not say; the model never sees a padding token, so under the batch-invariant
kernels (`invariant`) a row's numbers depend on neither its batch companions nor the run's `padding`.

What vLLM makes different, each refused by name before anything runs, or recorded:

- The head is computed at the last position alone: `logits` (and `lm_head_output`, which nnterp serves as the logits
  where the model does nothing after the head, and refuses naming the softcap or scale where it does) is read or
  written at the prompt's last position only (`last_only`; preflight refuses any other, once the schedule says where
  each record is done).
- The attention pattern and the scores are recomputed from the queries and keys: a read is served, a write is
  refused (`execute`); a read of them beyond the prefill, which compile refuses on every engine, a decode call could
  not serve here either.
- No gradient: a fit is refused in `execute`, a `grad` entry by `steps.start_run` (`gradless`). A quantized `dtype`
  too: vLLM's quantizations are not bitsandbytes' weights. `attn_implementation` in the model block is dropped: vLLM's
  attention backend is its own.
- The engine serves sequences up to its `max_len` (`longest`, vLLM's `max_model_len`): preflight refuses a step whose
  longest row, plus what a generation adds, does not fit, since vLLM would cut the generation short with no EOS.
- A forward runs whole (`stops`): nothing ends a request's forward inside the worker.
- `readout` runs the model's final norm, head and logits processor in the worker (`project_on_vocab`), in a trace of
  its own: the model's own, as the nnterp engine's is. Under `--readout at-reads` it meets the stream alone where
  the forward's final norm fuses it with the last residual add, another kernel, so the two agree to the last bits,
  as `at-reads` promises, not to the bit.
- vLLM's default kernels are not run-to-run stable: two identical request sets differ in the last bits (in bfloat16
  by whole ulps). The engine runs vLLM's batch-invariant kernels (`VLLM_BATCH_INVARIANT`) unless `invariant` is
  false, so that the model's values reproduce to the bit and a null is exact (an entry's own ops after its last edit
  still run per batch on the client's GPU, as on every engine), and has Triton run a float32 matmul in float32
  (`TRITON_F32_DEFAULT=ieee`, where nothing set it), not in TF32 as it would by default. Both are set as the worker
  starts (`ready`), which is when it reads them.
- A generation's options are mapped onto vLLM's sampling (`sampling`), what the step leaves out taken from the
  generation config as transformers takes it, and a row stops on clab's stop ids alone (`ignore_eos`, so that vLLM's
  own EOS does not widen the set); a row's new ids are what vLLM sampled, padded past the row's end with the
  generation config's pad token, as transformers pads.
- The worker's memory is vLLM's: a step's `cost.memory` is None (`nvidia-smi` says), and `cost.tokens` counts the
  batch's padding the model never saw (`real` is what it fed). The run record's `devices` and `dtype` are the
  worker's (the client holds a meta tree), and its `versions` name vLLM's.
- The run's traces are not inside a session (`execute` says why): each batch is one request to vLLM, in plain
  Python, as the hooks engine's calls are.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import time
from typing import Any

import nnsight
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from clab import steps
from clab.engines import engine, standard
from clab.engines.standard import build, envoy, loading, nnterp_name, stops
from clab.format.address import COMPONENTS
from clab.plan import QUANTIZED, Model, Plan, calls_model, entries, walk, writes
from clab.steps import batches
from clab.steps.batches import acting_at_call, stamped_at_call

# The attention values nnterp recomputes from the queries and keys on vLLM: read-only there.
RECOMPUTED = ("attention_scores", "attention_probabilities")
# The head, which vLLM computes at the last position alone.
HEAD = ("logits", "lm_head_output")
# The axes of a component's layout that run over tokens, which a request's value holds unpadded.
TOKEN_AXES = ("seq", "query", "key")
# The HuggingFace generate options the engine maps onto vLLM's sampling, each with vLLM's name for it, or None where
# the engine maps it itself (`sampling`).
GENERATE = {"max_new_tokens": "max_tokens", "min_new_tokens": "min_tokens", "temperature": "temperature",
            "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "repetition_penalty": "repetition_penalty",
            "do_sample": None, "eos_token_id": None, "pad_token_id": None}
# The generation config's fields transformers turns into a logits processor whatever the step says, which vLLM's
# sampling has no name for, each with the value that turns it off: a config that sets one is refused by name.
UNMAPPED = {"no_repeat_ngram_size": 0, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "bad_words_ids": None,
            "suppress_tokens": None, "begin_suppress_tokens": None, "forced_bos_token_id": None,
            "forced_eos_token_id": None, "sequence_bias": None, "encoder_repetition_penalty": 1.0,
            "exponential_decay_length_penalty": None}
# clab's reasons for a mixture's values on vLLM: at a block whose MLP is a mixture, which vLLM's fused kernel serves no
# routing value of; at a dense block.
FUSED = ("vLLM runs the mixture of experts inside its fused MoE kernel, which serves no routing value; the mixture's "
         "output is mlp_output")
DENSE_BLOCK = "this block's MLP is dense, not a mixture of experts"


@engine("vllm")
class VllmEngine:
    name: str
    remote: bool | str = False      # its traces run in this process
    stops: bool = False             # a request's forward runs whole in the worker
    last_only = {c: "vLLM computes the head at the prompt's last position alone" for c in HEAD}
    gradless = ("a grad entry takes the model's gradient, which vLLM's kernels do not take; run it on the nnterp or "
                "hooks engine")
    config: dict[str, Any]          # what it read off the model: its `sizes`, and once the worker is up, `longest`

    def __init__(self, model: Model, *, device: str = "cuda:0", tp: int = 1, memory: float = 0.9,
                 max_len: int | None = None, invariant: bool = True) -> None:
        """The checkpoint `model` pins on nnsight's vLLM engine, through nnterp's StandardizedVLLM, in its dtype,
        sharded over `tp` GPUs with `memory` of each reserved: a meta tree until a run calls the model (`ready`),
        when vLLM starts its worker and loads the weights. The block's other keys go to vLLM's engine arguments as
        they are, but `attn_implementation`, which means nothing there. A quantized `dtype` is refused: vLLM holds a
        quantized model in its own formats, not bitsandbytes'.

        device: "cuda:0", the first of the GPUs vLLM runs the model on (CUDA_VISIBLE_DEVICES names them), where the run keeps its values; no other
        tp: how many GPUs vLLM shards the model over (its tensor_parallel_size)
        memory: the fraction of each GPU's memory vLLM reserves for the weights and its cache (its gpu_memory_utilization); the run's own values live outside it
        max_len: the longest sequence, prompt and new tokens, the engine serves (its max_model_len); null for the checkpoint's
        invariant: true runs vLLM's batch-invariant kernels, so that a run reproduces to the bit; false its fastest, whose last bits move between runs
        """
        from nnterp import StandardizedVLLM

        if model.dtype in QUANTIZED:
            raise ValueError(f"model.dtype {model.dtype!r}: the vllm engine cannot honor it: vLLM holds a quantized "
                             "model's weights in its own formats, not bitsandbytes', so they would not be the weights "
                             "the other engines hold; run it on the nnterp or hooks engine, or in a float dtype here")
        loading(model, "vllm", device)
        where = torch.device(device)
        if (where.type, where.index or 0) != ("cuda", 0):
            raise ValueError(f"device: the vllm engine cannot honor {device!r}: it keeps the run's values on cuda:0, "
                             "the first of the GPUs vLLM runs the model on (CUDA_VISIBLE_DEVICES names them, tp how "
                             "many)")
        if tp < 1:
            raise ValueError(f"tp {tp}: vLLM shards the model over one GPU or more")
        if not 0 < memory <= 1:
            raise ValueError(f"memory {memory}: the fraction of each GPU's memory vLLM reserves is above 0 and at most 1")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len {max_len}: the longest sequence the engine serves is one token or more")
        self.spec, self.device, self.options = model, device, {"tp": tp, "memory": memory, "max_len": max_len,
                                                               "invariant": invariant}
        arguments: dict[str, Any] = {"dtype": model.dtype, "tensor_parallel_size": tp, "gpu_memory_utilization": memory,
                                     "enable_prefix_caching": False, "async_scheduling": False}
        if max_len is not None:
            arguments["max_model_len"] = max_len
        self.model = build(model, "vllm", lambda extra: StandardizedVLLM(
            model.key, revision=model.revision, dispatch=False, **arguments,
            **{k: v for k, v in extra.items() if k != "attn_implementation"}))
        self.config = {"sizes": standard.sizes(self.model)}

    def execute(self, plan: Plan, *, batch_size: int = 8, padding: str = "dataset",
                readout: str = "full") -> Plan:
        """The whole plan in this process, as the hooks engine runs it: what vLLM cannot do refused by name first (a
        fit, which trains through the model's gradient; a write at the recomputed pattern or scores; a generate step
        whose options vLLM's sampling cannot take, `sampling`), then a `grad` entry by `steps.start_run` (`gradless`),
        the model `ready` before the clock where the run calls it, then `steps.run` with each batch its own trace,
        outside any session. Not inside one, as the nnterp engine's is: nnsight marks what a trace saved by object id
        in a set a session keeps across its traces, and reads that set when it serializes a request, so a row's dict
        of a later batch that reuses the address of an earlier batch's freed, saved dict would travel home as saved,
        and the root envoy in it cannot (a run of many batches met that); outside a session each trace clears the
        set. The run record's `devices`, `dtype` and vLLM's version are then the worker's, which the client's meta
        tree does not say."""
        for name, step in walk(plan.steps):
            if step.kind == "fit":
                raise ValueError(f"{name}: a fit trains through the model's gradient, which vLLM's kernels do not "
                                 "take; run it on the nnterp or hooks engine")
            for key, e in entries(step).items():
                if writes(e) and e.site.component in RECOMPUTED:
                    raise ValueError(f"{name}.{key}: the vllm engine does not write {e.site.component}: vLLM's "
                                     "attention kernel serves nothing between its inputs and its output, so nnterp "
                                     "recomputes it from the queries and keys, read-only; write attention_queries, "
                                     "attention_keys or attention_head_outputs instead")
            if step.kind == "generate":
                try:
                    self.sampling(dict(step.model_extra or {}))
                except ValueError as e:
                    raise ValueError(f"{name}: generate rejected {e}") from None
        started = steps.start_run(self, plan, batch_size, padding, readout)
        if calls_model(plan.steps):          # before the clock: `seconds` is the run, not vLLM's start
            self.ready()
            started = steps.start_run(self, plan, batch_size, padding, readout)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        if self.model.dispatched:
            dtype = str(self.model.vllm_entrypoint.llm_engine.model_config.dtype).removeprefix("torch.")
            plan.run.dtype, plan.run.devices = [dtype], [f"cuda:{i}" for i in range(self.options["tp"])]
            plan.run.versions["vllm"] = importlib.metadata.version("vllm")
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    def ready(self) -> None:
        """Starts vLLM's worker and loads the weights (`dispatch`), once, before the first step of a run that calls
        the model, a key the loader rejects refused by name as `build` refuses one; the kernel switches the worker
        reads as it starts are set just before (the batch-invariant kernels per `invariant`, and IEEE float32 for
        Triton's matmul where nothing set it, since TF32, its default, moves float32 logits well past their last
        bits). Then the longest sequence the worker
        serves is known (`longest`)."""
        if self.model.dispatched:
            return
        os.environ["VLLM_BATCH_INVARIANT"] = "1" if self.options["invariant"] else "0"
        os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
        build(self.spec, self.name, lambda extra: self.model.dispatch())
        self.config["longest"] = int(self.model.vllm_entrypoint.llm_engine.model_config.max_model_len)

    @property
    def longest(self) -> int | None:
        """The longest sequence, prompt and new tokens, the worker serves (vLLM's `max_model_len`), which preflight
        checks each step's rows against; None until the worker is up, which it is before any step that calls the
        model."""
        return self.config.get("longest")

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    @property
    def eos(self) -> list[int]:
        return stops(self.model.generation_config, self.tokenizer)

    @property
    def cuda(self) -> list[str]:
        """None: the model is in vLLM's worker, whose memory is its own to account."""
        return []

    def sizes(self) -> dict[str, int]:
        return self.config["sizes"]

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """nnterp's answer for the vLLM engine (`standard.unserved` over StandardizedVLLM's `support()`), but a
        mixture's values, which vLLM's fused kernel serves none of and nnterp's vLLM answer does not name: refused
        with that reason (`FUSED`) at the blocks whose MLP nnterp calls routed (the reason it gives `mlp_activation`
        there, `ROUTED`) and as dense at the others, since clab's reading of nnterp's silence would call the whole
        model dense."""
        from nnterp.components.vllm.mlp import ROUTED

        reasons, n = standard.unserved(self.model), self.model.num_layers
        activation = reasons.get("mlp_activation")
        routed = ([layer for layer, why in activation.items() if why == ROUTED] if isinstance(activation, dict)
                  else list(range(n)) if activation == ROUTED else [])
        for component in standard.mixed():
            if routed and isinstance(reasons.get(component), str):
                reasons[component] = {layer: FUSED if layer in routed else DENSE_BLOCK for layer in range(n)}
        return standard.own(reasons)

    def rank(self, component: str, layer: int | None) -> tuple[int, int]:
        """Where the component falls in the forward: nnterp's order, which it measures on a weightless transformers
        model of the checkpoint for the values the engine serves."""
        return self.model.rank(nnterp_name(component), layer)

    def requests(self, batch: dict[str, Tensor], bound: dict[str, Any]) -> tuple[list[list[int]], list[dict]]:
        """The batch as vLLM takes it: each row's own token ids, the padding left off, and each row's half of the
        bound schedule (`batches.schedule_for_row`) with the site's envoy and the batch's width, which the worker pads
        the row's values back to (`visit_request`); a record at the head, which preflight holds to the row's last
        position, indexes the one position vLLM serves there (-1), so the head is never padded."""
        ids, mask = batch["input_ids"], batch["attention_mask"]
        width = ids.shape[1]
        prompts = [int(m.sum()) for m in mask]
        rows = []
        for r in range(len(prompts)):
            own = batches.schedule_for_row(bound, r)
            sites = []
            for site in own["sites"]:
                records = site["records"]
                if site["component"] in HEAD:
                    records = [{**record, "positions": torch.full_like(record["positions"], -1)}
                               for record in records]
                sites.append({**site, "records": records,
                              "host": envoy(self.model, site["component"], site["layer"])})
            rows.append({"width": width, "sites": sites})
        return [ids[r, width - p:].tolist() for r, p in enumerate(prompts)], rows

    def forward(self, batch: dict[str, Tensor], bound: dict[str, Any]) -> dict[str, Tensor]:
        """One trace over the batch, one request per row (`requests`), each visiting the row's sites in the worker
        (`visit_request`) and saving its reads, joined along rows once home (`joined`)."""
        tokens, rows = self.requests(batch, bound)
        try:
            with self.model.trace(temperature=0.0, max_tokens=1) as tracer:
                for ids, row in zip(tokens, rows):
                    with tracer.invoke(ids):
                        mine = nnsight.save({})
                        visit_request(row, mine)
        except RuntimeError as e:
            raise refusal(e) from None
        return joined(listed(mine))         # pyright: ignore[reportPossiblyUnboundVariable]  (bound by the trace)

    def generate(self, batch: dict[str, Tensor], bound: dict[str, Any],
                 options: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        """One trace over the batch with the step's options as vLLM's sampling (`sampling`), one request per row:
        every step of each request (`tracer.all()`, the prefill then one per new token) visits the row's sites
        where the schedule's `loop` names the call (`visit_request`, from the inputs the row has been fed so far
        where a write is found `online`) and takes the token the step sampled, the row's new ids; a row's steps end
        at its EOS, and its ids are padded past it to the batch's longest with the pad token (`sampling`), as
        transformers pads. The new ids `[rows, new]`, and each read joined along rows."""
        tokens, rows = self.requests(batch, bound)
        sampling, loop, online = self.sampling(options), bound["loop"], bound["online"]
        pad = sampling.pop("pad_token_id")
        model, spec = self.model, (self.spec.key, self.spec.revision,
                                   bool((self.spec.model_extra or {}).get("trust_remote_code", False)))
        try:
            with self.model.trace(**sampling) as tracer:
                for ids, row in zip(tokens, rows):
                    with tracer.invoke(ids):
                        mine = nnsight.save({"got": {}, "ids": []})
                        seen: list[Tensor] = []
                        tokenizer = worker_tokenizer(*spec) if online else None
                        for step in tracer.all():
                            if online and step > 0:
                                seen.append(torch.tensor([[mine["ids"][-1]]]))
                            if loop is None or step in loop:
                                visit_request(row, mine["got"], step, seen, tokenizer)
                            mine["ids"].append(model.samples.item())
        except RuntimeError as e:
            raise refusal(e) from None
        per = listed(mine)                  # pyright: ignore[reportPossiblyUnboundVariable]  (bound by the trace)
        new = max(len(p["ids"]) for p in per)
        return torch.tensor([p["ids"] + [pad] * (new - len(p["ids"])) for p in per]), joined([p["got"] for p in per])

    def sampling(self, options: dict[str, Any]) -> dict[str, Any]:
        """A generate step's options as vLLM's sampling parameters (`GENERATE`), what the step leaves out taken from
        the generation config as transformers takes it: `max_new_tokens`, which the step must give (vLLM's own
        default length, 16 tokens, is not the config's), and `min_new_tokens` as vLLM spells them; `do_sample` false
        (or the config's) as temperature 0, true as the step's temperature, top_p, top_k and min_p, each the config's
        where the step leaves it out; `repetition_penalty`, which transformers applies under greedy decoding too; the
        step's `eos_token_id` where it gives one, else the engine's stop ids (`eos`), as the only ids a row stops on
        (`ignore_eos`: vLLM would add its own EOS to the set); and `pad_token_id`, the step's, else the config's, else
        the first stop id, which `generate` pads the rows' ids with and pops. Refused by name: an option vLLM's
        sampling has no name for, and a config that sets a field transformers turns into a logits processor and
        vLLM's sampling has no name for (`UNMAPPED`), which the other engines would apply."""
        config = self.model.generation_config
        for key in options:
            if key not in GENERATE:
                raise ValueError(f"{key!r}: the vllm engine's generate maps {', '.join(GENERATE)} onto vLLM's "
                                 f"sampling, and has nothing to map {key!r} to")
        if "max_new_tokens" not in options:
            raise ValueError("'max_new_tokens': the vllm engine needs the step to give it: vLLM's own default length "
                             "(16 tokens) is not the generation config's")
        for key, default in UNMAPPED.items():
            value = getattr(config, key, default)
            if value is not None and value != default:
                raise ValueError(f"'{key}': the checkpoint's generation config sets {key} {value!r}, which the nnterp "
                                 "and hooks engines apply as transformers does and vLLM's sampling has no name for; "
                                 "run the step on them")
        out = {GENERATE[k]: v for k, v in options.items() if GENERATE.get(k)}
        if not options.get("do_sample", bool(getattr(config, "do_sample", False))):
            out = {k: v for k, v in out.items() if k not in ("temperature", "top_p", "top_k", "min_p")}
            out["temperature"] = 0.0
        else:
            for key in ("temperature", "top_p", "top_k", "min_p"):
                if key not in out and getattr(config, key, None) is not None:
                    out[key] = getattr(config, key)
        penalty = options.get("repetition_penalty", getattr(config, "repetition_penalty", None))
        if penalty not in (None, 1, 1.0):
            out["repetition_penalty"] = penalty
        given = options.get("eos_token_id")
        out["stop_token_ids"] = self.eos if given is None else [given] if isinstance(given, int) else list(given)
        out["ignore_eos"] = True
        pad = options.get("pad_token_id", getattr(config, "pad_token_id", None))
        out["pad_token_id"] = out["stop_token_ids"][0] if pad is None else pad
        return out

    def readout(self, hidden: Tensor) -> Tensor:
        """nnterp's `project_on_vocab` (the final norm, the head and vLLM's logits processor: the model's own, as
        the nnterp engine's readout is), run in the worker inside a trace of its own over a one-token prompt,
        `hidden` travelling with the block; the logits come home on cuda:0."""
        model = self.model
        try:
            with self.model.trace([self.eos[0]], temperature=0.0, max_tokens=1):
                out = model.project_on_vocab(hidden.to("cuda")).save()
        except RuntimeError as e:
            raise refusal(e) from None
        return out


def visit_request(row: dict, got: dict, call: int = 0, seen: list[Tensor] | None = None,
                  tokenizer: Any = None) -> None:
    """In the worker, forward call `call` of one request (0 is a forward's one call, or a generation's prefill): each
    site of the row's schedule in its order, doing there what `acting_at_call` says and, at a decode call,
    `stamped_at_call` from the inputs `seen` so far (`tokenizer` decodes them where a write is found online), as
    `NnterpEngine.visited` does, on nnterp's value for the request padded on the left to the batch's width in the
    prefill (`padded`), so that the records' indices hold, and set back unpadded where `visit_site` says to; the
    head's one position is indexed as it is. `lm_head_output` is read and written as `logits`: nnterp serves it there
    where the two are one value, and refuses it before any forward where they are not."""
    for site in row["sites"]:
        here = acting_at_call(site["records"], call)
        here = stamped_at_call(here, seen or [], tokenizer, call) if call > 0 else here
        if not here:
            continue
        component, host = site["component"], site["host"]
        name = "logits" if component == "lm_head_output" else component
        value = getattr(host, name)
        shape, whole = value.shape, call == 0 and component not in HEAD
        value, wrote = batches.visit_site(padded(value, component, row["width"]) if whole else value, component, here,
                                          got)
        if wrote:
            setattr(host, name, unpadded(value, component, shape) if whole else value)


def padded(value: Tensor, component: str, width: int) -> Tensor:
    """A request's value (`[1, tokens, ...]` in the component's layout) padded on the left along each of its token
    axes (`TOKEN_AXES`) to the batch's `width`, with zeros, which no record's index reaches: the row's own positions
    sit where the padded batch holds them."""
    for axis, name in enumerate(COMPONENTS[component]["dims"]):
        if name in TOKEN_AXES and value.shape[axis] < width:
            size = list(value.shape)
            size[axis] = width - value.shape[axis]
            value = torch.cat([value.new_zeros(size), value], axis)
    return value


def unpadded(value: Tensor, component: str, shape: torch.Size) -> Tensor:
    """What `padded` padded, back to the request's own `shape` along each token axis: what the model is handed."""
    for axis, name in enumerate(COMPONENTS[component]["dims"]):
        if name in TOKEN_AXES and value.shape[axis] != shape[axis]:
            value = value.narrow(axis, value.shape[axis] - shape[axis], shape[axis])
    return value


@functools.lru_cache(maxsize=None)
def worker_tokenizer(key: str, revision: str, trust_remote_code: bool) -> Any:
    """The checkpoint's tokenizer, loaded once in the worker, for a write found online (`stamped_at_call` decodes the
    inputs fed so far with it); the engine's own tokenizer would travel with every request."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(key, revision=revision, trust_remote_code=trust_remote_code)


def listed(saved: Any) -> list:
    """What a name saved in every invoke of a trace comes home as: a list in invoke order, or the one value where
    the trace had one invoke."""
    return saved if isinstance(saved, list) else [saved]


def joined(parts: list[dict]) -> dict[str, Any]:
    """The rows' reads (`parts`, one dict per row as `visit_site` filled it) joined along rows under each name: a
    prefill read's `[1, ...]` values stacked into `[rows, ...]`, a decode-call read's lists into one list of
    `[rows, 1, ...]` per call the batch made, a row that made fewer calls blank at the rest (NaN, -1 in ids: what
    marks a missing value, which `generate.value_beyond_prefill` never reads)."""
    got: dict[str, Any] = {}
    for name in dict.fromkeys(k for p in parts for k in p):
        values = [p.get(name) for p in parts]
        if any(isinstance(v, list) for v in values):
            most = max(len(v) for v in values if v is not None)
            got[name] = [stacked([v[k] if v is not None and k < len(v) else None for v in values])
                         for k in range(most)]
        else:
            got[name] = stacked(values)
    return got


def stacked(values: list) -> Any:
    """The rows' values of one read at one call stacked along rows: each `[1, ...]` (a `top_k`'s a dict of them),
    a row that holds none (None) blank."""
    some = next(v for v in values if v is not None)
    if isinstance(some, dict):
        return {k: stacked([None if v is None else v[k] for v in values]) for k in some}
    blank = torch.full_like(some, float("nan") if some.is_floating_point() else -1)
    return torch.cat([blank if v is None else v.to(some.device) for v in values], 0)


def refusal(error: RuntimeError) -> BaseException:
    """An error raised in the worker, which nnsight re-raises as a RuntimeError naming the original type and
    carrying its traceback: a refusal (a ValueError, `visit_site`'s and the ops') as the ValueError it was, its
    message alone, so that it is reported as every refusal is; a CUDA out-of-memory as torch's, so that the run
    stops as it does on every engine (`steps.run`); anything else as it came."""
    head = str(error).split("\n\nIntervention traceback", 1)[0]
    kind, _, message = head.partition(": ")
    if kind == "ValueError":
        return ValueError(message)
    if kind.rpartition(".")[2] == "OutOfMemoryError":
        return torch.OutOfMemoryError(message)
    return error
