"""What the engines share: how nnterp reaches clab's components (the envoy that serves each, its dotted name
there, the sizes), how a model block becomes a loader call, where a batch's inputs go, where a HuggingFace generation
stops, and a forward's or a generation's records grouped by site."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from clab.format.address import COMPONENTS

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy
    from nnterp import StandardizedTransformer
    from torch import Tensor
    from transformers import GenerationConfig, PreTrainedTokenizerBase

    from clab.format.data import Vision
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
# the block's one spelling of the dtype.
OWNED = {"device": "where the model runs is a run option: --device", "device_map": "where the model runs is a run "
         "option: --device", "dispatch": "the engine places its model itself",
         "torch_dtype": "the dtype is spelled `dtype`"}


# The device maps transformers computes, which may spread a model over several devices; any other `device` is one
# torch device.
COMPUTED = ("auto", "balanced", "balanced_low_0", "sequential")


def inputs(batch: dict[str, Tensor], device: str, mrope: bool = False) -> dict[str, Tensor]:
    """The batch on `device`: where the model's first parameter is (`batches.home`), which is where a
    computed device map puts the model's input. Every tensor the batch carries goes: the ids, the mask, the
    position ids, a processor's per-token tensors and the image tensors; the position ids stay behind under
    `mrope` (a generation's always, which lays its own out)."""
    return {k: v.to(device) for k, v in batch.items() if not (mrope and k == "position_ids")}


def mrope(module: Any) -> bool:
    """Whether the model lays its own position ids out over a prompt with images (multimodal rotary positions:
    the Qwen-VL family's `get_rope_index`, from the attention mask and the images' grids), so that the 1-d ones
    clab counts would be wrong there and it is given none."""
    return any(hasattr(m, "get_rope_index") for m in (module, getattr(module, "model", None)) if m is not None)


def vision(processor: Any, config: Any) -> Vision | None:
    """What a chat with image parts needs of the loaded checkpoint (`data.Vision`): its processor, and the id
    of the token the processor puts where an image's tokens go (the config's `image_token_id`, or
    `image_token_index`); None on a checkpoint loaded without a processor (the default `text-generation` task)."""
    if processor is None:
        return None
    found = next((getattr(config, n) for n in ("image_token_id", "image_token_index")
                  if isinstance(getattr(config, n, None), int)), None)
    return {"processor": processor, "image_token_id": found}


def loader_extra(model: Model) -> dict[str, Any]:
    """The model block's keys for a plain transformers loader: every extra but `task`, which is the pipeline's
    (nnsight's loader takes it; `from_pretrained` does not)."""
    return {k: v for k, v in (model.model_extra or {}).items() if k != "task"}


def build(model: Model, engine: str, call: Callable[[dict[str, Any]], Any]) -> Any:
    """`call` with the model block's keys other than `key`, `revision` and `dtype`, as they are. A key in
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
