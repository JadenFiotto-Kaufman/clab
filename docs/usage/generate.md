---
title: Generation
one_liner: "Generation steps: options passed to the model's generate, deterministic decoding, frames for the prefill and decode calls, writes kept on through generation, writes found online at generated text, EOS and rows that stop early."
tags: [usage, generation, decoding, eos]
related: [docs/concepts/positions.md, docs/usage/positions.md, docs/usage/ops.md, docs/errors/alignment-missing.md, docs/patterns/generation-experiments.md, docs/patterns/steering.md, docs/usage/sweeps.md]
sources: [clab/steps/generate.py, clab/steps/batches.py, clab/plan.py, clab/compile/positions.py]
---

# Generation

A `generate` step runs the model's `generate` on every row of its input. Its value is the new token ids, `[rows, new]`. Its entries read and write at the forward calls that `pos`'s `frame` names: call 0 is the prefill, and call k ≥ 1 is decode call k. The step's fields are in [FORMAT.md#generate](../../FORMAT.md#generate) and the frame stage is in [FORMAT.md#pos](../../FORMAT.md#pos). Why a generation is a timeline of calls is explained in [concepts/positions.md](../concepts/positions.md).

## Generate and score the text

From `documents/generated_contains.json`:

```json
"answer": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "min_new_tokens": 6, "do_sample": false},
"text": {"kind": "op", "op": "detokenize", "of": "answer"},
"said": {"kind": "op", "op": "contains", "of": "text", "text": "prompts.said"},
"acc": {"kind": "op", "op": "mean", "of": "said"}
```

`detokenize` turns each row's ids into text and stops before the row's first EOS. `contains` scores 1 where that text contains the row's answer. Both are covered in [usage/ops.md](ops.md).

### The first generated token's leading space

On a SentencePiece tokenizer such as the corpus Llama's, decoding drops the leading space of a row's first generated token: a row that generates `▁Finland ony` reads `Finlandony`. `detokenize` decodes this way, and so does the search for a scope over generated text. So an answer written with its leading space, as [data.md](data.md#answer-columns) asks, is not found when it is generated token 0, the token a write at the prompt's last position changes first: on a row that says ` Finland` there and never again, `contains` scores 0, and a scope over the decode calls finds a later occurrence or none. Give the column `strip: true` wherever it searches generated text (a `text` scope takes `"strip": true` beside its text):

```json
"said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "prompts.said", "strip": true}},
"at_said": {"site": "head", "pos": {"frame": "1:", "scope": {"column": {"column": "prompts.said", "strip": true}}, "index": -1}}
```

Under `strip` that row scores 1, and a scope finds a word said at token 0. A byte-level BPE tokenizer such as GPT-2's keeps the space when it decodes.

## Options

Every key other than `kind`, `input` and `interventions` goes to the engine's `generate` unchanged.

- **`do_sample`.** An option you leave out comes from the checkpoint's generation config, and some checkpoints sample by default. Nothing seeds sampling, so two runs with `do_sample: true` give different ids. `do_sample: false` decodes greedily and gives the same ids on every run. Write it on every generate step.
- **`max_new_tokens`** is the most new tokens a row gets. A write whose frame runs to the end of the row (`"2:"`, `"1::2"`) needs it, because it fixes how many calls the write lands at.
- **`min_new_tokens`** holds EOS off until a row has that many tokens. Setting it equal to `max_new_tokens` makes every row make the same calls.
- **`eos_token_id`** is the id, or list of ids, that a row stops on. Without it, rows stop on the id from the engine's generation config.

The step refuses four options before the model loads: `num_return_sequences`, `num_beams`, `return_dict_in_generate` and `use_cache`. Any other option is left for the engine to accept, and one it rejects fails when the step runs, with the step's name; an option of the wrong type (`"max_new_tokens": "6"`) can instead fail inside the model's `generate` as a Python `TypeError`. There is no `seed` option:

```text
clab: steps.g: num_beams 2; it stays 1: beams change the rows each forward call sees, so a read's row is no longer the input's
clab: g: generate rejected 'seed': The following `model_kwargs` are not used by the model: ['seed'] (note: typos in the generate arguments will also show up in this list)
```

Like a forward, a generation reproduces to the bit only at a fixed batch size ([usage/running.md](running.md)). Options can be swept; `max_new_tokens` and `min_new_tokens` crossed with each other is the trap described in [usage/sweeps.md](sweeps.md#labels-zip-or-cross).

## Frames: which forward call

Row 0 of `documents/generate_probe.json` has an 11-token prompt at positions 0 to 10, so generated token k sits at position 11 + k. A read at the head (`lm_head_output`) gives:

| `pos` | calls | what it holds on row 0 |
|---|---|---|
| `-1`, the same as `{"frame": 0, "index": -1}` | the prefill | position 10, the last prompt token; its logits are the distribution over generated token 0 |
| `{"frame": k}`, k ≥ 1 | decode call k | one position, 10 + k: generated token k−1 as input; its logits are the distribution over token k |
| `{"frame": "1:"}` | every decode call | one position per call, ragged across rows that stop at different lengths |
| `{"frame": ":"}` | the prefill and every decode call | at the head, one distribution per generated token (their `argmax` is the generated ids, EOS included); on a residual site, every prompt token followed by each generated input |
| `{"frame": -1}` | the row's last call | the call that chose the row's last token, whether that was EOS or token `max_new_tokens`−1; on a row whose first token was EOS that is the prefill, every prompt token on a residual site, so write `{"frame": -1, "index": -1}` for one position |

In the prefill, the head is computed only at the last position. So `{"frame": 0, "index": -2}` at the head is refused, while the same `pos` on a residual site reads the second-to-last prompt token. A row's last generated token is never an input to any call. In a generation, `clab table` adds a `call` column beside `pos` and `pos_token`.

## Writes during generation

A write lands as its call runs.

- A write at frame `0` changes the prompt's activations. At the last prompt position it changes generated token 0, which that position chooses. Later tokens see the write only through the cache.
- A write at decode call k changes generated tokens from k on. Zeroing layer 0 at `{"frame": 2}` leaves tokens 0 and 1 as they were and changes token 2 onward.
- To keep a write on for the whole generation, write two entries: the prompt's last token and every decode call. The complete document below compiles and runs:

```json
{
  "description": "Damp layer 0 of the residual stream at the prompt's last token and at every generated token, and compare the text with an undamped generation.",
  "model": {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "float32"
  },
  "data": {"prompts": {"path": "weekdays/said"}},
  "sites": {"resid": {"component": "layer_output", "layers": 0}},
  "steps": {
    "plain": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "min_new_tokens": 6, "do_sample": false},
    "damped": {
      "kind": "generate",
      "input": "prompts.input",
      "max_new_tokens": 6,
      "min_new_tokens": 6,
      "do_sample": false,
      "interventions": {
        "prompt": {"site": "resid", "pos": {"frame": 0, "index": -1}, "ops": [{"op": "scale", "factor": 0.5}]},
        "decode": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "scale", "factor": 0.5}]}
      }
    },
    "plain_text": {"kind": "op", "op": "detokenize", "of": "plain"},
    "damped_text": {"kind": "op", "op": "detokenize", "of": "damped"}
  },
  "saves": ["plain_text", "damped_text"]
}
```

`plan.run.positions` records where each write landed. `damped.prompt` is `[[10], [8], [8], [8]]` and `damped.decode` is `[[11, 12, 13, 14, 15], [9, ...], ...]`: every generated input except the last token, which is never fed.

To write at every position, the whole prompt included, use `{"frame": ":"}`. A root bundle written at `{"frame": ":"}` covers every call of whichever step pulls it in: every token of a forward, and every prompt token and generated input of a generation.

Because a write lands as its call runs, everything it selects must be known by then. It cannot count back from a row's last call, and its index must name the same calls on a row that stops early as on one that runs on:

```text
clab: steps.g.interventions.zero.pos: frame -1 with index ':' names calls that depend on how far each row gets; a write lands as its call runs, so its frame is not counted from the row's last call, and its index names the same calls on a row that stops early as on one that runs on: ':' over the prompt, and not counted from the end over several decode calls; or it takes a text scope
clab: steps.g.interventions.zero.pos: frame ':' with index -1 counts back from the row's last position, known only when the row stops, and a write lands as each call runs; for the prompt's last positions and every decode call, write two entries: {'frame': 0, 'index': -1} and {'frame': '1:'}
clab: steps.g.interventions.zero.pos: frame '2:' with index ':' names calls up to the row's last; a write lands at calls listed before the run, so the step gives max_new_tokens, which bounds them
```

Steering with a harvested direction through a generation is in [patterns/steering.md](../patterns/steering.md).

## Writes at generated text, found online

A write with a `column` or `text` scope over the decode calls is resolved while the row generates. At each call, clab decodes what the row has generated so far, and the write lands at the call that is fed the last token of the occurrence as it appears. A variation of `documents/generated_answer.json` zeroes layer 0 where each row says its `said` word:

```json
"zero": {"site": "l0", "pos": {"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}, "ops": [{"op": "scale", "factor": 0.0}]}
```

Row 0 says `substr` as generated token 1, so the write lands at position 12, the call fed `▁substr`, and the row's text changes from token 2 on. Rows 2 and 3 never say their word. They are `alignment_missing` in `plan.run.positions`, and their text is unchanged. Such a write takes index `-1`, counts `occurrence` from the first, and takes no `before`. Each of these is refused by name:

```text
clab: steps.answer.interventions.zero.pos.index: index 0: a scoped write lands at the call fed its text's last token, the first call that can know the text is there; earlier tokens were already fed, so its index is -1
clab: steps.answer.interventions.zero.pos.scope.before: a write beyond the prefill lands as its call runs, before the token it would predict is chosen; a scope that is before is a read's there, or a write's in the prefill
```

A read is resolved after the row has finished generating, so these limits do not apply to reads: a read over generated text can take `before`, any index and any occurrence. The recipes are in [usage/positions.md](positions.md).

## Rows that stop early

A row stops at its first EOS. On the first four rows of the `test` split of `data/weekdays_full/data.json`, with `"eos_token_id": 4827` (`▁Alex`) and `max_new_tokens: 8`, row 1 stops after 3 tokens and rows 0, 2 and 3 run to 8:

```json
"days": {"path": "weekdays_full/data", "split": "test", "take": 4},
"g": {"kind": "generate", "input": "days.input", "max_new_tokens": 8, "do_sample": false, "eos_token_id": 4827,
      "interventions": {"at3": {"site": "head", "pos": {"frame": 3}}}}
```

- The step's ids keep the shape `[4, 8]`. Every position after a row's EOS is filler, so read the text through `detokenize`, which stops before the EOS. `plan.run.tokens["g"]` lists each row's tokens up to and including its EOS: `['Handler', '▁tijd', '▁Alex']` on row 1.
- A read at a call a row never made is missing on that row. `{"frame": 3}` gives row 1 one table row, with `alignment_missing` true and no value.
- A ragged read (`"1:"`, `":"`) holds each row's own calls. Positions past a row's last call are NaN in the read, so an `argmax` of them is -1, and the table leaves them out.
- `{"frame": -1}` on row 1 is call 2, the call that chose EOS. Its `argmax` is the EOS id.
- With `min_new_tokens: 8`, every row makes all eight calls and `{"frame": 3}` has a value on every row. A row held past its EOS chooses another token where it would have stopped: row 1 says `▁note` at token 2 and goes on from there, a different generation, not the first one extended.

How missing rows pass through ops and tables is covered in [errors/alignment-missing.md](../errors/alignment-missing.md).

## Pitfalls

- **Leaving out `do_sample`.** The checkpoint decides whether to sample, and a sampled run does not repeat.
- **An answer with its leading space, searched in generated text.** It is not found at token 0; give the column `strip: true` ([above](#the-first-generated-tokens-leading-space)).
- **Sweeping a read's frame.** Each point re-runs the whole generation. Read `{"frame": "0:3"}` once instead ([usage/sweeps.md](sweeps.md#widen-the-read-instead)).
- **A word that ends on the row's last generated token.** No call is fed that token, so a read scoped to the word without `before` is `alignment_missing`. With `max_new_tokens: 2`, a word said as token 1 is missing at the word itself, and found with `before`. Leave room after the answer.
