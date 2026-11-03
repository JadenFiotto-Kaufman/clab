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
decided here that the schedule does not say; the model never sees a padding token, so a row's numbers depend on
neither its batch companions nor the run's `padding`.

What vLLM makes different, each refused by name before anything runs, or recorded:

- The head is computed at the last position alone: `logits` (and `lm_head_output`, which this engine serves as the
  logits where the model does nothing after the head, no `final_logit_softcapping`, `logit_scale` or `logits_scaling`
  in its config, the two being one value then) is read or written at the prompt's last position only (`last_only`;
  preflight refuses any other).
- The attention pattern and the scores are recomputed from the queries and keys: read-only (`unwritable`); a read
  of them beyond the prefill, which compile refuses on every engine, a decode call could not serve here either.
- No gradient: a fit and a `grad` entry are refused in `execute`. A quantized `dtype` too: vLLM's quantizations are
  not bitsandbytes' weights. `attn_implementation` in the model block is dropped: vLLM's attention backend is its own.
- A forward runs whole (`stops`): nothing ends a request's forward inside the worker.
- `readout` runs the model's final norm, head and logits processor in the worker (`project_on_vocab`), in a trace of
  its own: the model's own, as the nnterp engine's is. Under `--readout at-reads` it meets the stream alone where
  the forward's final norm fuses it with the last residual add, another kernel, so the two agree to the last bits
  (1e-5 on SmolLM2 in float32), as `at-reads` promises, not to the bit.
- vLLM's default kernels are not run-to-run stable: two identical request sets differ in the last bits (in bfloat16
  by whole ulps). The engine runs vLLM's batch-invariant kernels (`VLLM_BATCH_INVARIANT`) unless `invariant` is
  false, so that a run reproduces to the bit and a null is exact, and has Triton run a float32 matmul in float32
  (`TRITON_F32_DEFAULT=ieee`, where nothing set it), not in TF32 as it would by default.
- The worker's memory is vLLM's: a step's `cost.memory` is None (`nvidia-smi` says), and `cost.tokens` counts the
  batch's padding the model never saw (`real` is what it fed). The run record's `devices` and `dtype` are the
  worker's (the client holds a meta tree).
- The run's traces are not inside a session (`execute` says why): each batch is one request to vLLM, in plain
  Python, as the hooks engine's calls are.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any

import nnsight
import torch
from torch import Tensor

from clab import steps
from clab.engines import engine, standard
from clab.engines.nnterp import NnterpEngine
from clab.engines.standard import build, envoy, loading, nnterp_name, stops
from clab.format.address import COMPONENTS
from clab.plan import QUANTIZED, Model, Plan, calls_model, entries, gradient_of, walk
from clab.steps import batches
from clab.steps.batches import acting_at_call, stamped_at_call

# The attention values nnterp recomputes from the queries and keys on vLLM.
RECOMPUTED = ("attention_scores", "attention_probabilities")
# The head, which vLLM computes at the last position alone.
HEAD = ("logits", "lm_head_output")
# The axes of a component's layout that run over tokens, which a request's value holds unpadded.
TOKEN_AXES = ("seq", "query", "key")
# The HuggingFace generate options the engine maps onto vLLM's sampling, each with vLLM's name for it, or None where
# the engine maps it itself (`sampling`).
GENERATE = {"max_new_tokens": "max_tokens", "min_new_tokens": "min_tokens", "temperature": "temperature",
            "top_p": "top_p", "top_k": "top_k", "repetition_penalty": "repetition_penalty", "do_sample": None,
            "eos_token_id": None, "pad_token_id": None}


@engine("vllm")
class VllmEngine(NnterpEngine):
    stops = False                   # a request's forward runs whole in the worker
    unwritable = {c: "vLLM's attention kernel serves nothing between its inputs and its output, so nnterp recomputes "
                     "it from the queries and keys, read-only; write attention_queries, attention_keys or "
                     "attention_head_outputs instead" for c in RECOMPUTED}
    last_only = {c: "vLLM computes the head at the prompt's last position alone" for c in HEAD}

    def __init__(self, model: Model, *, device: str = "cuda:0", tp: int = 1, memory: float = 0.9,
                 max_len: int | None = None, invariant: bool = True) -> None:
        """The checkpoint `model` pins on nnsight's vLLM engine, through nnterp's StandardizedVLLM, in its dtype,
        sharded over `tp` GPUs with `memory` of each reserved: a meta tree until a run calls the model (`ready`),
        when vLLM starts its worker and loads the weights. The block's other keys go to vLLM's engine arguments as
        they are, but `attn_implementation`, which means nothing there. A quantized `dtype` is refused: vLLM holds a
        quantized model in its own formats, not bitsandbytes'.

        device: "cuda:0", the first of the GPUs vLLM runs the model on (CUDA_VISIBLE_DEVICES names them), where the run keeps its values; no other
        tp: how many GPUs vLLM shards the model over (its tensor_parallel_size)
        memory: the fraction of each GPU's memory vLLM reserves for the weights and its cache (its gpu_memory_utilization)
        max_len: the longest sequence, prompt and new tokens, the engine serves (its max_model_len); null for the checkpoint's
        invariant: true runs vLLM's batch-invariant kernels, so that a run reproduces to the bit; false its fastest, whose last bits move between runs
        """
        from nnterp import StandardizedTransformer, StandardizedVLLM

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
        # Both read when the worker starts. Triton runs a float32 `tl.dot` in TF32 unless told otherwise, and the
        # batch-invariant matmul is one: in float32 it moved SmolLM2's logits by 6e-3 against transformers', and by
        # 8e-6 once IEEE; a value set outside stands.
        os.environ["VLLM_BATCH_INVARIANT"] = "1" if invariant else "0"
        os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
        self.spec, self.device, self.options = model, device, {"tp": tp, "memory": memory, "max_len": max_len,
                                                               "invariant": invariant}
        # The forward order and the generation config are a transformers meta shell's: StandardizedVLLM has neither,
        # and the standard values sit at the same module boundaries on both engines.
        self.shell = build(model, "vllm", lambda extra: StandardizedTransformer(
            model.key, revision=model.revision, dispatch=False, device_map=None,
            **{**extra, "attn_implementation": "eager"}))
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
        fit, which trains through the model's gradient, and a `grad` entry), the model `ready` before the clock where
        the run calls it, then `steps.run` with each batch its own trace, outside any session. Not inside one, as the
        nnterp engine's is: nnsight marks what a trace saved by object id in a set a session keeps across its traces,
        and reads that set when it serializes a request, so a row's dict of a later batch that reuses the address of
        an earlier batch's freed, saved dict would travel home as saved, and the root envoy in it cannot (a run of
        many batches met that); outside a session each trace clears the set. The run record's `devices` and `dtype`
        are the worker's, which the client's meta tree does not say."""
        for name, step in walk(plan.steps):
            if step.kind == "fit":
                raise ValueError(f"{name}: a fit trains through the model's gradient, which vLLM's kernels do not "
                                 "take; run it on the nnterp or hooks engine")
            for key in [k for k, e in entries(step).items() if gradient_of(e) is not None][:1]:
                raise ValueError(f"{name}.{key}: a grad entry takes the model's gradient, which vLLM's kernels do not "
                                 "take; run it on the nnterp or hooks engine")
        started = steps.start_run(self, plan, batch_size, padding, readout)
        if calls_model(plan.steps):          # before the clock: `seconds` is the run, not vLLM's start
            self.ready()
            started = steps.start_run(self, plan, batch_size, padding, readout)
        steps.run(self, plan, batch_size)
        assert plan.run is not None
        if self.model.dispatched:
            dtype = str(self.model.vllm_entrypoint.llm_engine.model_config.dtype).removeprefix("torch.")
            plan.run.dtype, plan.run.devices = [dtype], [f"cuda:{i}" for i in range(self.options["tp"])]
        plan.run.seconds = round(time.time() - started, 3)
        return plan

    def ready(self) -> None:
        """Starts vLLM's worker and loads the weights (`dispatch`), once, before the first step of a run that calls
        the model, a key the loader rejects refused by name as `build` refuses one. The order and the mixture
        interiors the nnterp engine's `ready` prepares are not this engine's: the shell answers the order, and no
        mixture interior is served here."""
        if not self.model.dispatched:
            build(self.spec, self.name, lambda extra: self.model.dispatch())

    @property
    def eos(self) -> list[int]:
        return stops(self.shell.generation_config, self.tokenizer)

    @property
    def cuda(self) -> list[str]:
        """None: the model is in vLLM's worker, whose memory is its own to account."""
        return []

    def unserved(self) -> dict[str, str | dict[int, str] | None]:
        """nnterp's answer for the vLLM engine (`standard.unserved` over StandardizedVLLM's `support()`), but
        `lm_head_output`, served as the logits where the model does nothing after the head (`after_head`), the two
        being one value then, and refused naming what it does otherwise."""
        reasons = standard.unserved(self.model)
        if isinstance(reasons.get("lm_head_output"), str):
            after = self.after_head()
            reasons["lm_head_output"] = None if after is None else (
                f"vLLM never calls lm_head, and this model's {after} after the head keeps its logits from standing in "
                "for the head's own projection; read logits")
        return standard.own(reasons)

    def after_head(self) -> str | None:
        """What the model does after its head, from its text config, as `<key> <value>` (a softcap, a scale); None
        where it does nothing."""
        config = self.shell.config.get_text_config()
        for key in ("final_logit_softcapping", "logit_scale", "logits_scaling"):
            value = getattr(config, key, None)
            if value not in (None, 0, 1):
                return f"{key} {value}"
        return None

    def rank(self, component: str, layer: int | None) -> tuple[int, int]:
        """Where the component falls in the forward: nnterp's measured order on the transformers meta shell of the
        checkpoint (StandardizedVLLM measures none)."""
        return self.shell.rank(nnterp_name(component), layer)

    def requests(self, batch: dict[str, Tensor], bound: dict[str, Any]) -> tuple[list[list[int]], list[dict]]:
        """The batch as vLLM takes it: each row's own token ids, the padding left off, and each row's half of the
        bound schedule (`batches.schedule_for_row`) with the site's envoy, the batch's width and the row's padding,
        which the worker pads the row's values back to (`visit_request`). A record at the head that meets another
        prefill position than the row's last is refused here, naming it, where preflight let it through (under
        `--readout at-reads` a read the readout could not move stays at the head)."""
        ids, mask = batch["input_ids"], batch["attention_mask"]
        width = ids.shape[1]
        prompts = [int(m.sum()) for m in mask]
        for site in bound["sites"]:
            for record in site["records"] if site["component"] in HEAD else []:
                if "calls" in record:
                    continue                    # a decode call's one position, the head's
                for r, p in enumerate(prompts):
                    met = sorted({int(t) for t in record["tok"][r].tolist() if t not in (-1, p - 1)})
                    if met:
                        what = f"read {record['name'].partition('@')[0]!r}" if "name" in record else "write"
                        raise ValueError(f"{site['component']} {what}: the vllm engine serves {site['component']} at "
                                         f"the prompt's last position alone, and this {what.split()[0]} meets "
                                         f"position{'s' if len(met) > 1 else ''} {met} on a row of {p} tokens; name "
                                         "pos -1, or read layer_output at the last block and apply the readout op")
        rows = []
        for r, p in enumerate(prompts):
            own = batches.schedule_for_row(bound, r)
            rows.append({"width": width, "pad": width - p, "sites": [
                {**site, "host": envoy(self.model, site["component"], site["layer"])} for site in own["sites"]]})
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
        at its EOS, and its ids are padded past it to the batch's longest with the first stop id, as the hooks
        engine's are. The new ids `[rows, new]`, and each read joined along rows."""
        tokens, rows = self.requests(batch, bound)
        sampling, loop, online = self.sampling(options), bound["loop"], bound["online"]
        model, stop = self.model, (self.spec.key, self.spec.revision,
                                   bool((self.spec.model_extra or {}).get("trust_remote_code", False)))
        try:
            with self.model.trace(**sampling) as tracer:
                for ids, row in zip(tokens, rows):
                    with tracer.invoke(ids):
                        mine = nnsight.save({"got": {}, "ids": []})
                        seen: list[Tensor] = []
                        tokenizer = worker_tokenizer(*stop) if online else None
                        for step in tracer.all():
                            if online and step > 0:
                                seen.append(torch.tensor([[mine["ids"][-1]]]))
                            if loop is None or step in loop:
                                visit_request(row, mine["got"], step, seen, tokenizer)
                            mine["ids"].append(model.samples.item())
        except RuntimeError as e:
            raise refusal(e) from None
        per = listed(mine)                  # pyright: ignore[reportPossiblyUnboundVariable]  (bound by the trace)
        new, pad = max(len(p["ids"]) for p in per), sampling["stop_token_ids"][0]
        return torch.tensor([p["ids"] + [pad] * (new - len(p["ids"])) for p in per]), joined([p["got"] for p in per])

    def sampling(self, options: dict[str, Any]) -> dict[str, Any]:
        """A generate step's options as vLLM's sampling parameters (`GENERATE`): `max_new_tokens`, which the step
        must give (vLLM's own default, 16 tokens, is not the generation config's), and `min_new_tokens` as vLLM
        spells them; `do_sample` false (or the generation config's, where the step says nothing) as temperature 0,
        true as the step's temperature, top_p and top_k, each the generation config's where the step leaves it out;
        `repetition_penalty` as it is; the step's `eos_token_id` where it gives one, else the engine's stop ids
        (`eos`), as the ids a request stops on. Any other option is refused by name: nothing maps it."""
        for key in options:
            if key not in GENERATE:
                raise ValueError(f"{key!r}: the vllm engine's generate maps {', '.join(GENERATE)} onto vLLM's "
                                 f"sampling, and has nothing to map {key!r} to")
        if "max_new_tokens" not in options:
            raise ValueError("'max_new_tokens': the vllm engine needs the step to give it: vLLM's own default length "
                             "(16 tokens) is not the generation config's")
        config = self.shell.generation_config
        out = {GENERATE[k]: v for k, v in options.items() if GENERATE.get(k)}
        if not options.get("do_sample", bool(getattr(config, "do_sample", False))):
            out = {k: v for k, v in out.items() if k not in ("temperature", "top_p", "top_k")}
            out["temperature"] = 0.0
        else:
            for key in ("temperature", "top_p", "top_k"):
                if key not in out and getattr(config, key, None) is not None:
                    out[key] = getattr(config, key)
        given = options.get("eos_token_id")
        out["stop_token_ids"] = self.eos if given is None else [given] if isinstance(given, int) else list(given)
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
    prefill (`padded`), so that the records' indices hold, and set back unpadded where `visit_site` says to.
    `lm_head_output` is the logits (the engine serves it where the two are one value). An error other than a
    refusal (a ValueError, which travels as it is) or nnsight's own (a read the run never served, which cuts the
    block short as nnsight means it to) is raised again as text: vLLM pickles a request's error home, and one that
    holds a torch object cannot be pickled, which would hide it."""
    try:
        visit_sites(row, got, call, seen, tokenizer)
    except ValueError:
        raise
    except Exception as e:
        if type(e).__module__.partition(".")[0] == "nnsight":
            raise               # nnsight's own, and its to handle: a read the run never served cuts the block short
        raise RuntimeError(f"{type(e).__name__}: {e}") from None


def visit_sites(row: dict, got: dict, call: int, seen: list[Tensor] | None, tokenizer: Any) -> None:
    """`visit_request`'s visits."""
    for site in row["sites"]:
        here = acting_at_call(site["records"], call)
        here = stamped_at_call(here, seen or [], tokenizer, call) if call > 0 else here
        if not here:
            continue
        component, host = site["component"], site["host"]
        name = "logits" if component == "lm_head_output" else component
        value = getattr(host, name)
        shape = value.shape
        value, wrote = batches.visit_site(padded(value, component, row["width"]) if call == 0 else value, component,
                                          here, got)
        if wrote:
            setattr(host, name, unpadded(value, component, shape) if call == 0 else value)


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
    message alone, so that it is reported as every refusal is; anything else as it came."""
    head = str(error).split("\n\nIntervention traceback", 1)[0]
    kind, _, message = head.partition(": ")
    return ValueError(message) if kind == "ValueError" else error
