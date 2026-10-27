"""What the engines share: how nnterp reaches clab's components (the envoy, its dotted name there, the sizes) and where
not, whether its key reads inside a forward, how a model block becomes a loader call (its dtype as the loader's
arguments), where a batch's inputs go, where a HuggingFace generation stops, and a forward's or a generation's
records grouped by site."""

from __future__ import annotations

import importlib.util
import logging
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


def inside(value: Any, host: Envoy) -> bool:
    """Whether nnterp's key for `value` on `host` reads inside a module's forward: a `source` segment on its path."""
    return "source" in value.path(host).lstrip("./").split(".")


def unserved(shell: StandardizedTransformer) -> dict[str, str | dict[int, str] | None]:
    """nnterp's `support()` of each component: None, a reason for the whole component, or `{layer: reason}`. A
    component nnterp's answer does not name at all is refused by name, since nothing says it is served: a
    component is one nnterp serves, under nnterp's name for it (`nnterp_name`)."""
    support = shell.support()
    unnamed = [c for c in COMPONENTS if nnterp_name(c) not in support]
    if unnamed:
        raise ValueError(f"nnterp's support() does not name {', '.join(nnterp_name(c) for c in unnamed)}, so nothing "
                         f"says whether it serves {', '.join(unnamed)}; a component of address.COMPONENTS is one "
                         "nnterp serves under that name")
    return {c: support[nnterp_name(c)] for c in COMPONENTS}


def served(component: str, refused: str | dict[int, str] | None, num_layers: int) -> list[int | None]:
    """The layers nnterp serves the component at, by its answer `refused` (`unserved`'s); [None] for a model value."""
    if isinstance(refused, str):
        return []
    return [i for i in range(num_layers) if i not in (refused or {})] if COMPONENTS[component]["per_layer"] else [None]


# The model-block keys an engine cannot honor, and why: where the model runs and how it is sharded are run options,
# and `dtype` is the block's one spelling of the dtype and of a quantization.
OWNED = {"device": "where the model runs is a run option: --device", "device_map": "where the model runs is a run "
         "option: --device", "dispatch": "the engine places its model itself",
         "torch_dtype": "the dtype is spelled `dtype`", "quantization_config": "a quantization is spelled in `dtype` "
         "(`nf4`, `fp4`, `int8`, `fp8`)", "tp_plan": "tensor parallelism is a run option: -o tp=N",
         "distributed_config": "tensor parallelism is a run option: -o tp=N"}


# The device maps transformers computes, which may spread a model over several devices; any other `device` is one
# torch device.
COMPUTED = ("auto", "balanced", "balanced_low_0", "sequential")


def loading(model: Model, engine: str, device: str) -> dict[str, Any]:
    """The loader's arguments for the model's `dtype`, which every engine takes from here, so that a quantization
    holds the same weights and computes in the same dtype on each by construction: `dtype`, torch's (the compute
    dtype, `Model.computes`, under a quantization), and for a quantization the `quantization_config` nnsight builds
    for its name: bitsandbytes 4-bit in that quant type computing in that dtype (without double quantization),
    LLM.int8(), or transformers' FP8. Refused by name before anything loads: a `device` no engine can place a model
    on (`placeable`: every engine's constructor comes through here before its loader), a bitsandbytes format without
    bitsandbytes installed, and `fp8` off a GPU of compute capability 8.9 or later (below it transformers would load
    bfloat16)."""
    import torch
    from transformers import BitsAndBytesConfig, FineGrainedFP8Config

    placeable(device)
    dtype = getattr(torch, model.computes)
    if model.dtype not in QUANTIZED:
        return {"dtype": dtype}
    if model.dtype == "fp8":
        if device not in COMPUTED and torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError(f"model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; the {engine} "
                             f"engine's device is {device!r}" + ("" if torch.cuda.is_available() else
                                                                 ", and no GPU is visible"))
        for i in range(torch.cuda.device_count()) if device in COMPUTED else [torch.device(device).index or 0]:
            capability = torch.cuda.get_device_capability(i)
            if capability < (8, 9):
                raise ValueError(f"model.dtype 'fp8' needs a GPU of compute capability 8.9 or later; cuda:{i} "
                                 f"({torch.cuda.get_device_name(i)}) is {capability[0]}.{capability[1]}")
        return {"dtype": dtype, "quantization_config": FineGrainedFP8Config()}
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ValueError(f"model.dtype {model.dtype!r} holds the weights in bitsandbytes, which is not installed: "
                         "install bitsandbytes")
    config = (BitsAndBytesConfig(load_in_8bit=True) if model.dtype == "int8" else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=model.dtype, bnb_4bit_compute_dtype=dtype))
    return {"dtype": dtype, "quantization_config": config}


def placeable(device: str) -> None:
    """`device` as every engine takes it: a device map transformers computes (`COMPUTED`), or one torch device.
    Refused by name before anything loads: a string that is neither, a CUDA device where none is visible, and
    one beyond those that are."""
    import torch

    if device in COMPUTED:
        return
    try:
        where = torch.device(device)
    except RuntimeError:
        raise ValueError(f"device {device!r}: not a torch device ('cpu', 'cuda:0') or a device map transformers "
                         f"computes ({', '.join(map(repr, COMPUTED))})") from None
    if where.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise ValueError(f"device {device!r}: no CUDA device is visible (torch.cuda.is_available() is False); run "
                         "on 'cpu', or make a GPU visible (CUDA_VISIBLE_DEVICES)")
    if where.index is not None and where.index >= torch.cuda.device_count():
        n = torch.cuda.device_count()
        raise ValueError(f"device {device!r}: {n} CUDA device{'s' * (n != 1)} visible, cuda:0 to cuda:{n - 1}")


def placing(model: Model, device: str) -> dict[str, str]:
    """Where nnsight's loader puts the model: as its `device_map` a map transformers computes, else as its `device`
    one torch device, and a quantized model's as its `device_map` too, since a model is quantized where
    `device_map` says (the first GPU when it says nothing) and a quantized one cannot move after."""
    if device in COMPUTED:
        return {"device_map": device}
    return {"device": device, **({"device_map": device} if model.dtype in QUANTIZED else {})}


def inputs(batch: dict[str, Tensor], keys: tuple[str, ...], device: str) -> dict[str, Tensor]:
    """The batch's `keys` on `device`: where the model's first parameter is (`batches.home`), which is where a
    computed device map puts the model's input."""
    return {k: batch[k].to(device) for k in keys}


def build(model: Model, engine: str, call: Callable[[dict[str, Any]], Any]) -> Any:
    """`call` with the model block's keys other than `key`, `revision` and `dtype` (which `loading` reads), as
    they are. A key in
    `OWNED` is refused by name, and one the loader rejects is reported as that key. A checkpoint the loader cannot
    get (its `OSError`) is refused as `model.key`, saying whether the run is offline and the cache lacks it."""
    from huggingface_hub.constants import HF_HUB_OFFLINE

    extra = dict(model.model_extra or {})
    for key in extra:
        if key in OWNED:
            raise ValueError(f"model.{key}: the {engine} engine cannot honor it: {OWNED[key]}")
    quiet()
    try:
        return call(extra)
    except OSError as e:                   # before ValueError: the hub's not-found error is both
        why = ("HF_HUB_OFFLINE is set and the Hugging Face cache lacks it, or part of it; fetch it online first"
               if HF_HUB_OFFLINE else "the hub does not give it and the cache lacks it")
        said = (str(e).strip().splitlines() or [""])[0]
        raise ValueError(f"model.key {model.key!r} at revision {model.revision!r}: the {engine} engine cannot load "
                         f"it: {why} (the loader says: {said})") from None
    except (TypeError, ValueError) as e:
        def names(key: str, text: str) -> bool:
            return repr(key) in text or f"{key}=" in text or f" {key} " in text
        key = next((k for k in extra if names(k, str(e))), None)
        if key is None:
            raise
        # The last line that names the key, without its exception's name: nnsight's loader wraps the error in a
        # report of every class it tried, with each one's traceback.
        why = [line.strip() for line in str(e).splitlines() if names(key, line)][-1]
        raise ValueError(f"model.{key}: the {engine} engine's loader rejects it: "
                         f"{why.partition(': ')[2] if why.split(': ')[0].endswith('Error') else why}") from None


# The one transformers warning every load of a checkpoint whose config carries `pad_token_id: -1` prints (the tiny
# test models do), which says nothing about the run: clab pads with the tokenizer's own id (`data.pad`).
PADDED = "Model config: pad_token_id must be `None` or an integer within the vocabulary"


def unpadded(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith(PADDED)


def quiet() -> None:
    """The `pad_token_id` warning (`PADDED`) filtered off the transformers logger that emits it, once, before a
    model loads; every other message of that logger passes."""
    logger = logging.getLogger("transformers.configuration_utils")
    if unpadded not in logger.filters:
        logger.addFilter(unpadded)


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
