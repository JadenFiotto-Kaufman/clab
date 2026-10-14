"""What both engines share: how nnterp reaches clab's components (the envoy that serves each, its dotted name
there, the sizes), how a model block becomes a loader call, where a HuggingFace generation stops, a forward's
or a generation's records grouped by site and by call, and where a write with an online scope lands."""

from typing import Any, Callable, cast

import torch

from .. import positions
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


# The model-block keys an engine cannot honor, and why: each engine places its own model, and `dtype` is the
# block's one spelling of the dtype.
OWNED = {"device": "the engine places its model itself", "device_map": "the engine places its model itself",
         "dispatch": "the engine places its model itself", "torch_dtype": "the dtype is spelled `dtype`"}


def load(model, engine: str, call: Callable[[dict], Any]) -> Any:
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


def now(here: list, there: list, call: int) -> tuple[list, list]:
    """What a generation does at a site on forward call `call` (0 is the prefill): a record with no `calls`
    acts on the prefill only, one with `calls` on those decode calls (None: every one)."""
    def wanted(record: dict) -> bool:
        if "calls" not in record:
            return call == 0
        return call > 0 and (record["calls"] is None or call in record["calls"])
    return [w for w in here if wanted(w)], [r for r in there if wanted(r)]


def landed(writes: list[dict], seen: list, tokenizer, call: int) -> list[dict]:
    """The writes at a decode call, a write with an `online` scope narrowed to the rows it lands on there
    (`keep`): those whose scope's text first occurs ending on this call's input, given the inputs `seen` so
    far (one `[rows, 1]` per decode call); one that lands on no row is left out."""
    out = []
    for w in writes:
        online = w.get("online")
        if online is None:
            out.append(w)
            continue
        if online["within"] is not None and call not in online["within"]:
            continue
        rows = [positions.lands(online["scope"], text, [int(t[r, 0]) for t in seen], set(online["eos"]), tokenizer)
                for r, text in enumerate(online["texts"])]
        if any(rows):
            out.append({**w, "keep": torch.tensor(rows)[:, None]})
    return out


def keep(got: dict, record: dict, value) -> None:
    """A read's value into `got`: a prefill read's as itself, a decode-call read's appended to its list, one
    per call read."""
    if "calls" in record:
        got[record["name"]].append(value)
    else:
        got[record["name"]] = value


def by_site(writes: list[dict], reads: list[dict]) -> dict[tuple, tuple[list, list]]:
    """A forward's writes and reads by `(component, layer)`, each site's in listed order."""
    sites: dict[tuple, tuple[list, list]] = {}
    for side, records in ((0, writes), (1, reads)):
        for record in records:
            sites.setdefault((record["component"], record["layer"]), ([], []))[side].append(record)
    return sites

