"""What the engines share: how nnterp reaches clab's components (the envoy that serves each, its dotted name
there, the sizes), how a model block becomes a loader call, where a batch's inputs go, where a HuggingFace generation
stops, a forward's or a generation's records grouped by site and by call, and where a write with an online
scope lands."""

from typing import Any, Callable, cast

from ..address import COMPONENTS


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


def envoy(model, component: str, layer: int | None):
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


def inputs(batch: dict, keys: tuple, device: str) -> dict:
    """The batch's `keys` on `device`: where the model's first parameter is (`batches.home`), which is where a
    computed device map puts the model's input."""
    return {k: batch[k].to(device) for k in keys}


def build(model, engine: str, call: Callable[[dict], Any]) -> Any:
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



def stops(config, tokenizer) -> list[int]:
    """The ids a generation stops on: the generation config's EOS, else the tokenizer's."""
    found = getattr(config, "eos_token_id", None)
    found = tokenizer.eos_token_id if found is None else found
    return [found] if isinstance(found, int) else list(found)


def by_site(records: list[dict]) -> dict[tuple, list]:
    """A forward's records by `(component, layer)`, each site's in listed order."""
    sites: dict[tuple, list] = {}
    for record in records:
        sites.setdefault((record["component"], record["layer"]), []).append(record)
    return sites
