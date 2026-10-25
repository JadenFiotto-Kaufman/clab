"""The locations clab allows: what each component means, its layout, and what narrows it.

A component's name is nnterp's name for the same value, and its `dims` are
nnterp's layout of it (`batch` first, the token axis named `seq`, or `query` on
an attention pattern). `per_layer` says it is a block's value, so a site names
its `layers`. A site's `select` (`{<axis>: <selector>}`) narrows any axis the table lists as narrowing the
component, as a selector: among that axis's size (the model's `num_heads`, ...) it keeps the indices picked,
in their order, and the axis stays in the layout, so that a site has one layout whatever its selector: an int
keeps one index, an axis of size 1 (`{"num_heads": 3}` is head 3 alone, the same value as `[3]`), where an
int `layers` drops its axis. `key` is each row's own keys, its positions as `pos` counts them (key 0 is BOS
where there is one), so a selector there picks a fixed number of keys counted from the start (`0`, `[0, 2]`,
`"0:3"`); unselected, the axis is as wide as the input's longest row, key j the row's token j, and past a
shorter row's own keys a read is NaN and a write changes nothing (its padding's keys stay as they are).
`stream` marks the residual-stream values `readout` may project.

A selection of a component (what an entry takes, and what its ops see) is
laid out `(rows, pos, *the remaining axes in order)`: a `layers` axis comes
first under `layers` that are a slice or a list of picks. A read whose rows hold different numbers of
positions is as wide as its widest row, NaN past a row's own. How an engine reaches a component is that
engine's business; the name says only what the value is.
"""

R = ["batch", "seq", "hidden_size"]

COMPONENTS: dict[str, dict] = {
    "token_embeddings": {"meaning": "the embedding module's output", "per_layer": False,
                         "dims": R, "stream": True},
    "layer_input": {"meaning": "the residual stream entering block i", "per_layer": True,
                    "dims": R, "stream": True},
    "attention_norm_output": {"meaning": "the normed stream the attention is called with", "per_layer": True,
                              "dims": R, "stream": False},
    "attention_queries": {"meaning": "queries after projection, head split and RoPE", "per_layer": True,
                          "dims": ["batch", "num_heads", "seq", "qk_head_dim"],
                          "stream": False},
    "attention_keys": {"meaning": "keys, before repeat_kv", "per_layer": True,
                       "dims": ["batch", "num_kv_heads", "seq", "qk_head_dim"],
                       "stream": False},
    "attention_values": {"meaning": "values, before repeat_kv", "per_layer": True,
                         "dims": ["batch", "num_kv_heads", "seq", "head_dim"],
                         "stream": False},
    "attention_scores": {"meaning": "scaled, masked scores entering the softmax", "per_layer": True,
                         "dims": ["batch", "num_heads", "query", "key"],
                         "stream": False},
    "attention_probabilities": {"meaning": "the pattern the values are mixed with", "per_layer": True,
                                "dims": ["batch", "num_heads", "query", "key"],
                                "stream": False},
    "attention_head_outputs": {"meaning": "each head's output, before the output projection", "per_layer": True,
                               "dims": ["batch", "seq", "num_heads", "head_dim"],
                               "stream": False},
    "attention_premix": {"meaning": "every head's result as the output projection takes it", "per_layer": True,
                         "dims": ["batch", "seq", "num_heads", "head_dim"],
                         "stream": False},
    "attention_output": {"meaning": "what attention adds to the stream", "per_layer": True,
                         "dims": R, "stream": True},
    "layer_mid": {"meaning": "the stream between the sublayers", "per_layer": True,
                  "dims": R, "stream": True},
    "mlp_norm_output": {"meaning": "the normed stream entering the MLP", "per_layer": True,
                        "dims": R, "stream": False},
    "mlp_input": {"meaning": "what the MLP is called with", "per_layer": True,
                  "dims": R, "stream": False},
    "mlp_activation": {"meaning": "act(gate_proj x) on a gated MLP, act(up_proj x) on a plain one", "per_layer": True,
                       "dims": ["batch", "seq", "intermediate_size"],
                       "stream": False},
    "mlp_neurons": {"meaning": "what the down projection sums", "per_layer": True,
                    "dims": ["batch", "seq", "intermediate_size"],
                    "stream": False},
    "mlp_output": {"meaning": "what the MLP adds to the stream", "per_layer": True,
                   "dims": R, "stream": True},
    "layer_output": {"meaning": "the residual stream leaving block i", "per_layer": True,
                     "dims": R, "stream": True},
    "final_norm_output": {"meaning": "the final norm's output, what the head projects", "per_layer": False,
                          "dims": R, "stream": False},
    "lm_head_output": {"meaning": "the head's own projection", "per_layer": False,
                       "dims": ["batch", "seq", "vocab_size"], "stream": False},
    "logits": {"meaning": "what the model returns, after any softcap", "per_layer": False,
               "dims": ["batch", "seq", "vocab_size"], "stream": False},
}


# The axes that index examples and positions rather than features.
EXAMPLE_AXES = ("layers", "rows", "pos")


def selectable(component: str) -> list[str]:
    """The axes a site's `select` may narrow: every axis of the component's layout but its rows and its tokens."""
    return [a for a in COMPONENTS[component]["dims"] if a not in ("batch", "seq", "query")]


def pos_axis(component: str) -> int:
    """Where the token axis sits in the component's own layout."""
    dims = COMPONENTS[component]["dims"]
    return dims.index("query") if "query" in dims else dims.index("seq")


def dims(component: str, layers: object = None) -> list[str]:
    """The axes a selection of this component presents: `(rows, pos, *the remaining axes)`, a `layers` axis
    first under `layers` that `stacks`."""
    return (["layers"] if stacks(layers) else []) + ["rows", "pos"] + selectable(component)


def unstacked(dims: list[str]) -> list[str]:
    """A layout at one of its layers: every axis but `layers`."""
    return [a for a in dims if a != "layers"]


def stacks(layers: object) -> bool:
    """Whether a site's `layers` make one value with a `layers` axis: a slice or a list of picks, which name
    several layers (or may), not an int, which names one."""
    return layers is not None and not isinstance(layers, int)
