"""What the engines share: how nnterp reaches clab's components (the envoy that serves each, its dotted name
there, the sizes), how a model block becomes a loader call (its dtype as the loader's arguments), where a batch's
inputs go, where a HuggingFace generation stops, and a forward's or a generation's records grouped by site."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from clab.format.address import COMPONENTS
from clab.plan import QUANTIZED

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy
    from nnterp import StandardizedTransformer
    from torch import Tensor
    from transformers import GenerationConfig, PreTrainedTokenizerBase

    from clab.plan import Model

SIZES = ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "qk_head_dim", "intermediate_size",
         "vocab_size")

# The per-layer components nnterp serves below the block, on its attention or its MLP.
BELOW = {
    "self_attn": ("attention_queries", "attention_keys", "attention_values", "attention_scores",
                  "attention_probabilities", "attention_head_outputs", "attention_premix", "attention_output"),
    "mlp": ("mlp_activation", "mlp_neurons", "mlp_output"),
}


def host(component: str) -> str:
    """`model`, `layer`, `self_attn` or `mlp`: the envoy nnterp serves the component on."""
    if not COMPONENTS[component]["per_layer"]:
        return "model"
    return next((h for h, names in BELOW.items() if component in names), "layer")


def nnterp_name(component: str) -> str:
    """The dotted name nnterp's `support()` and `rank()` know the component by."""
    return f"{host(component)}.{component}" if host(component) in BELOW else component


def envoy(model: StandardizedTransformer, component: str, layer: int | None) -> Envoy:
    """The envoy on `model` (a loaded model or a meta shell) that serves the component."""
    where = host(component)
    if where == "model":
        return model
    block = model.layers[cast(int, layer)]
    return block if where == "layer" else getattr(block, where)


# The model-block keys an engine cannot honor, and why: where the model runs is a run option, and `dtype` is
# the block's one spelling of the dtype and of a quantization.
OWNED = {"device": "where the model runs is a run option: --device", "device_map": "where the model runs is a run "
         "option: --device", "dispatch": "the engine places its model itself",
         "torch_dtype": "the dtype is spelled `dtype`", "quantization_config": "a quantization is spelled in `dtype` "
         "(`nf4`, `fp4`, `int8`, `fp8`)"}


# The device maps transformers computes, which may spread a model over several devices; any other `device` is one
# torch device.
COMPUTED = ("auto", "balanced", "balanced_low_0", "sequential")


def loading(model: Model, engine: str, device: str) -> dict[str, Any]:
    """The loader's arguments for the model's `dtype`, which every engine takes from here, so that a quantization
    holds the same weights and computes in the same dtype on each by construction: `dtype`, torch's (the compute
    dtype, `Model.computes`, under a quantization), and for a quantization the `quantization_config` nnsight builds
    for its name: bitsandbytes 4-bit in that quant type computing in that dtype (without double quantization),
    LLM.int8(), or transformers' FP8. A quantization is refused by name before anything loads where `device` puts
    the model off CUDA, and `fp8` on a GPU below compute capability 8.9 (where transformers would load bfloat16)."""
    import torch
    from transformers import BitsAndBytesConfig, FineGrainedFP8Config

    dtype = getattr(torch, model.computes)
    if model.dtype not in QUANTIZED:
        return {"dtype": dtype}
    asked = device in COMPUTED or torch.device(device).type == "cuda"
    if not (asked and torch.cuda.is_available()):
        raise ValueError(f"model.dtype {model.dtype!r} needs a CUDA device; the {engine} engine's device is {device!r}"
                         + (", and no GPU is visible" if asked else ""))
    gpus = range(torch.cuda.device_count()) if device in COMPUTED else [torch.device(device).index or 0]
    for i in gpus if model.dtype == "fp8" else []:
        capability = torch.cuda.get_device_capability(i)
        if capability < (8, 9):
            raise ValueError(f"model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; cuda:{i} "
                             f"({torch.cuda.get_device_name(i)}) is {capability[0]}.{capability[1]}")
    config = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=model.dtype, bnb_4bit_compute_dtype=dtype)
              if model.dtype in ("nf4", "fp4") else
              BitsAndBytesConfig(load_in_8bit=True) if model.dtype == "int8" else FineGrainedFP8Config())
    return {"dtype": dtype, "quantization_config": config}


def inputs(batch: dict[str, Tensor], keys: tuple[str, ...], device: str) -> dict[str, Tensor]:
    """The batch's `keys` on `device`: where the model's first parameter is (`batches.home`), which is where a
    computed device map puts the model's input."""
    return {k: batch[k].to(device) for k in keys}


def build(model: Model, engine: str, call: Callable[[dict[str, Any]], Any]) -> Any:
    """`call` with the model block's keys other than `key`, `revision` and `dtype` (which `loading` reads), as
    they are. A key in
    `OWNED` is refused by name, and one the loader rejects is reported as that key."""
    extra = dict(model.model_extra or {})
    for key in extra:
        if key in OWNED:
            raise ValueError(f"model.{key}: the {engine} engine cannot honor it: {OWNED[key]}")
    try:
        return call(extra)
    except (TypeError, ValueError) as e:
        key = next((k for k in extra if repr(k) in str(e) or f"{k}=" in str(e) or f" {k} " in str(e)), None)
        if key is None:
            raise
        raise ValueError(f"model.{key}: the {engine} engine's loader rejects it: {e}") from None


def stops(config: GenerationConfig, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """The ids a generation stops on: the generation config's EOS, else the tokenizer's."""
    found = getattr(config, "eos_token_id", None)
    found = cast("int | list[int]", tokenizer.eos_token_id if found is None else found)
    return [found] if isinstance(found, int) else list(found)


def by_site(records: list[dict]) -> dict[tuple[str, int | None], list[dict]]:
    """A forward's records by `(component, layer)`, each site's in listed order."""
    sites: dict[tuple[str, int | None], list[dict]] = {}
    for record in records:
        sites.setdefault((record["component"], record["layer"]), []).append(record)
    return sites
