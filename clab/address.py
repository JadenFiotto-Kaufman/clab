"""The locations clab allows: what each component means, its layout, and what narrows it.

A component's name is nnterp's name for the same value, and its `dims` are
nnterp's layout of it (`batch` first, the token axis named `seq`, or `query` on
an attention pattern). `host` says where nnterp serves it: on the model
(`model`), on block *i* (`layer`), or on block *i*'s `self_attn` or `mlp`.
`narrow` maps each narrowing key a site may carry to the axis it indexes.
`stream` marks the residual-stream values `readout` may project.

A selection of a component (what a read binds, and what a write's ops see) is
laid out `(rows, pos, *the remaining axes in order)`: a `layers` axis comes
first under `layers: "all"`, and ragged positions give `entries` (one per
token) in place of `rows` and `pos`. How an engine reaches a component is that
engine's business; the name says only what the value is.
"""

R = ["batch", "seq", "hidden_size"]

COMPONENTS: dict[str, dict] = {
    "token_embeddings": {"meaning": "the embedding module's output", "host": "model",
                         "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "layer_input": {"meaning": "the residual stream entering block i", "host": "layer",
                    "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "attention_norm_output": {"meaning": "the normed stream the attention is called with", "host": "layer",
                              "dims": R, "narrow": {"units": "hidden_size"}, "stream": False},
    "attention_queries": {"meaning": "queries after projection, head split and RoPE", "host": "self_attn",
                          "dims": ["batch", "num_heads", "seq", "qk_head_dim"], "narrow": {"heads": "num_heads"},
                          "stream": False},
    "attention_keys": {"meaning": "keys, before repeat_kv", "host": "self_attn",
                       "dims": ["batch", "num_kv_heads", "seq", "qk_head_dim"], "narrow": {"heads": "num_kv_heads"},
                       "stream": False},
    "attention_values": {"meaning": "values, before repeat_kv", "host": "self_attn",
                         "dims": ["batch", "num_kv_heads", "seq", "head_dim"], "narrow": {"heads": "num_kv_heads"},
                         "stream": False},
    "attention_scores": {"meaning": "scaled, masked scores entering the softmax", "host": "self_attn",
                         "dims": ["batch", "num_heads", "query", "key"], "narrow": {"heads": "num_heads"},
                         "stream": False},
    "attention_probabilities": {"meaning": "the pattern the values are mixed with", "host": "self_attn",
                                "dims": ["batch", "num_heads", "query", "key"], "narrow": {"heads": "num_heads"},
                                "stream": False},
    "attention_head_outputs": {"meaning": "each head's output, before the output projection", "host": "self_attn",
                               "dims": ["batch", "seq", "num_heads", "head_dim"], "narrow": {"heads": "num_heads"},
                               "stream": False},
    "attention_premix": {"meaning": "every head's result as the output projection takes it", "host": "self_attn",
                         "dims": ["batch", "seq", "num_heads", "head_dim"], "narrow": {"heads": "num_heads"},
                         "stream": False},
    "attention_output": {"meaning": "what attention adds to the stream", "host": "self_attn",
                         "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "layer_mid": {"meaning": "the stream between the sublayers", "host": "layer",
                  "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "mlp_norm_output": {"meaning": "the normed stream entering the MLP", "host": "layer",
                        "dims": R, "narrow": {"units": "hidden_size"}, "stream": False},
    "mlp_input": {"meaning": "what the MLP is called with", "host": "layer",
                  "dims": R, "narrow": {"units": "hidden_size"}, "stream": False},
    "mlp_activation": {"meaning": "act(gate_proj x) on a gated MLP, act(up_proj x) on a plain one", "host": "mlp",
                       "dims": ["batch", "seq", "intermediate_size"], "narrow": {"units": "intermediate_size"},
                       "stream": False},
    "mlp_neurons": {"meaning": "what the down projection sums", "host": "mlp",
                    "dims": ["batch", "seq", "intermediate_size"], "narrow": {"units": "intermediate_size"},
                    "stream": False},
    "mlp_output": {"meaning": "what the MLP adds to the stream", "host": "mlp",
                   "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "layer_output": {"meaning": "the residual stream leaving block i", "host": "layer",
                     "dims": R, "narrow": {"units": "hidden_size"}, "stream": True},
    "final_norm_output": {"meaning": "the final norm's output, what the head projects", "host": "model",
                          "dims": R, "narrow": {"units": "hidden_size"}, "stream": False},
    "lm_head_output": {"meaning": "the head's own projection", "host": "model",
                       "dims": ["batch", "seq", "vocab_size"], "narrow": {"units": "vocab_size"}, "stream": False},
    "logits": {"meaning": "what the model returns, after any softcap", "host": "model",
               "dims": ["batch", "seq", "vocab_size"], "narrow": {"units": "vocab_size"}, "stream": False},
}

# The axes that index examples and positions rather than features.
EXAMPLE_AXES = ("layers", "rows", "entries", "pos")


def pos_axis(component: str) -> int:
    """Where the token axis sits in the component's own layout."""
    dims = COMPONENTS[component]["dims"]
    return dims.index("query") if "query" in dims else dims.index("seq")


def dims(component: str, layers: str | int | None, ragged: bool) -> list[str]:
    """The axes a selection of this component presents: `(rows, pos, *the remaining axes)`.

    `layers: "all"` puts a `layers` axis first. A ragged selection holds one
    entry per token, so `entries` stands for `rows` and `pos` together.
    """
    rest = [a for i, a in enumerate(COMPONENTS[component]["dims"]) if a != "batch" and i != pos_axis(component)]
    lead = ["layers"] if layers == "all" else []
    return lead + (["entries"] if ragged else ["rows", "pos"]) + rest


def nnterp_name(component: str) -> str:
    """The dotted name nnterp's `support()` and `rank()` know the component by."""
    host = COMPONENTS[component]["host"]
    return f"{host}.{component}" if host in ("self_attn", "mlp") else component
