"""How nnterp reaches clab's components, which both engines read: the envoy that serves each, its dotted name
there, the sizes, a forward's records grouped by site, and the refusal of what an engine cannot serve."""

from typing import cast

from ..address import COMPONENTS
from ..plan import Plan
from ..steps import walk

SIZES = ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "intermediate_size", "vocab_size")

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


def by_site(writes: list[dict], reads: list[dict]) -> dict[tuple, tuple[list, list]]:
    """A forward's writes and reads by `(component, layer)`, each site's in listed order."""
    sites: dict[tuple, tuple[list, list]] = {}
    for side, records in ((0, writes), (1, reads)):
        for record in records:
            sites.setdefault((record["component"], record["layer"]), ([], []))[side].append(record)
    return sites


def refuse(plan: Plan, reasons: dict, engine: str) -> None:
    """Every component the plan reads or writes that the engine cannot serve, by name, before anything runs.
    `reasons` maps a component to None (served), why not, or why not per layer (nnterp's `support()` shape)."""
    for name, step in walk(plan.steps):
        for key, record in {**getattr(step, "writes", {}), **getattr(step, "reads", {})}.items():
            reason, layers = reasons.get(record.site.component), record.site.layers
            if isinstance(reason, dict):
                reason = next((r for l, r in reason.items() if r and layers in (l, "all")), None)
            if reason:
                raise ValueError(f"{name}.{key}: the {engine} engine does not serve {record.site.component}: {reason}")
