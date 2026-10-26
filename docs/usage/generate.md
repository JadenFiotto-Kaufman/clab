---
title: Generation
one_liner: "Generation steps: options passed to the model's generate, deterministic decoding, frames for the prefill and decode calls, writes that stay on through generation, scopes over generated text, EOS and rows that stop early."
tags: [usage, generation]
related: []
sources: []
---

# Generation

Owner: group D. Owns: `generate` in use; `do_sample`, `max_new_tokens`, `min_new_tokens`, `eos_token_id`; frames `0`, `"1:"`, `":"`, `-1`; online scoped writes; `detokenize`/`contains` by link to ops.
