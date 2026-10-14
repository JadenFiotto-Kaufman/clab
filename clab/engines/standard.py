"""How nnterp reaches clab's components, which both engines read: the envoy that serves each, its dotted name
there, the sizes, and a forward's records grouped by site."""

from typing import cast

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


def loader(model, engine: str, owned: tuple) -> dict:
    """The keys of a model block an engine's loader takes as they are: every key but `key`, `revision` and
    `dtype`. A key the engine sets itself (`owned`) is one it cannot honor, and is refused by name."""
    extra = dict(model.model_extra or {})
    for key in extra:
        if key in owned:
            raise ValueError(f"model.{key}: the {engine} engine sets {key} itself")
    return extra


def by_site(writes: list[dict], reads: list[dict]) -> dict[tuple, tuple[list, list]]:
    """A forward's writes and reads by `(component, layer)`, each site's in listed order."""
    sites: dict[tuple, tuple[list, list]] = {}
    for side, records in ((0, writes), (1, reads)):
        for record in records:
            sites.setdefault((record["component"], record["layer"]), ([], []))[side].append(record)
    return sites

