---
title: Generation
one_liner: "How do I generate, and read or write as it generates? Options, frames (0, 1:, :, -1) for the prefill and decode calls, a read at every call, writes through every generated token, and rows that stop early."
tags: [usage, generation, decoding, eos]
related: [docs/concepts/positions.md, docs/usage/positions.md, docs/usage/ops.md, docs/errors/alignment-missing.md, docs/patterns/generation-experiments.md, docs/patterns/steering.md, docs/usage/sweeps.md]
sources: [clab/steps/generate.py, clab/steps/batches.py, clab/plan.py, clab/compile/positions.py]
---

# Generation

A `generate` step runs the model's `generate` on every row of its input. Its value is the new token ids, `[rows, new]`; a reduction of them names `new` in `over`, since the default removes only `rows` and `pos` ([ops.md](ops.md#reductions-and-over)). Its entries read and write at the forward calls that `pos`'s `frame` names: call 0 is the prefill, and call k ≥ 1 is decode call k. The step's fields are in [FORMAT.md#generate](../../FORMAT.md#generate) and the frame stage is in [FORMAT.md#pos](../../FORMAT.md#pos). Why a generation is a timeline of calls is explained in [concepts/positions.md](../concepts/positions.md).

`detokenize` turns each row's ids into text, stopped before the row's first EOS ([ops.md](ops.md)). Scoring that text (`contains`, a whole-word count, the text after a marker, a reasoning trace's length, teacher forcing) is [patterns/generation-experiments.md](../patterns/generation-experiments.md).

## The first generated token's leading space

On a SentencePiece tokenizer such as the corpus Llama's, decoding drops the leading space of a row's first generated token. A row whose tokens are `▁triple`, `▁Snow` reads `triple Snow`. `detokenize` decodes this way, and so does the search for a scope over generated text. A byte-level BPE tokenizer such as GPT-2's keeps the space when it decodes.

- An answer written with its leading space, as [data.md](data.md#answer-columns) asks, is not found when it is generated token 0.
- Token 0 is the token a write at the prompt's last position changes first.
- On a row that says ` triple` at token 0 and never again, `contains` scores 0, and a scope over the decode calls finds nothing.

Give the column `strip: true` wherever it searches generated text:

```json
"said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "prompts.said", "strip": true}},
"at_said": {"site": "head", "pos": {"frame": "1:", "scope": {"column": {"column": "prompts.said", "strip": true}}, "index": -1}}
```

Under `strip` that row scores 1, and the scope finds the word said at token 0.

## Options

Every key other than `kind`, `input` and `interventions` goes to the engine's `generate` unchanged (on the vllm engine, mapped onto vLLM's sampling, where `max_new_tokens` is required: [engines.md](engines.md#running-on-vllm)).

- **`do_sample`.** An option you leave out comes from the checkpoint's generation config, and some checkpoints sample by default. Nothing seeds sampling, so two runs with `do_sample: true` give different ids. `do_sample: false` decodes greedily and gives the same ids on every run. Write it on every generate step. Within one run, points of a step that differ only in what they read share one generation, so they read one sample ([sweeps.md](sweeps.md#what-a-sweep-costs)).
- **`max_new_tokens`** is the most new tokens a row gets. A write whose frame runs to the end of the row (`"2:"`, `"1::2"`) needs it, because it fixes how many calls the write lands at.
- **`min_new_tokens`** holds EOS off until a row has that many tokens. Setting it equal to `max_new_tokens` makes every row make the same calls.
- **`eos_token_id`** is the id, or list of ids, that a row stops on. Without it, rows stop on the id from the engine's generation config.

The step refuses `num_return_sequences`, `num_beams`, `return_dict_in_generate`, `use_cache` and `prefill_chunk_size` before the model loads, since each would change the rows, the value or the calls. Compile also refuses a parameter of `generate` itself (`generation_config`, `logits_processor`), which takes what a document cannot write. It refuses a value of a type transformers' `GenerationConfig` refuses (`"max_new_tokens": "6"`). The engine accepts or rejects any other option, and one it rejects fails when the step runs, naming the step. There is no `seed` option:

```text
clab: steps.answer: num_beams 2; it stays 1: beams change the rows each forward call sees, …
clab: steps.answer.generation_config: 'generation_config' is a parameter of generate itself, …
clab: steps.answer.max_new_tokens: '6' is not a value generate takes: …
clab: answer: generate rejected 'seed': The following `model_kwargs` are not used by the model: ['seed'] …
```

Like a forward, a generation gives the same numbers only at a fixed batch size ([running.md](running.md#when-two-runs-give-the-same-numbers)). Options can be swept; `max_new_tokens` crossed with `min_new_tokens` is the trap in [usage/sweeps.md](sweeps.md#labels-zip-or-cross).

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

### A read at every call

At the head, `{"frame": ":"}` is the per-call read: one position per call, the prefill's last prompt token and then each generated input, `[rows, calls, vocab_size]`. `{"frame": ":", "index": -1}` is one position, the last call's, since `index` counts over the tokens of every call the frame picks. On a residual site `{"frame": ":"}` is every prompt token and then each generated input.

The ops after an entry's last edit run in one of two places:

- **At each call, on that call's `[rows, 1, ...]` value:** the leading ops that are positionwise ([FORMAT.md#generate](../../FORMAT.md#generate) lists them). They stop at the first op that removes `pos`, takes an operand with a `pos` axis, or scores an answer of several tokens.
- **After the batch's generation, on the calls arranged by position:** the rest, from the first op that waits (`entropy` and `mean` wait, for example).
- **The entry's record counts the ops it runs at each call** (`per_call`, in the manifest's `steps`). Check it before relying on a per-call metric.

So a metric of the head inside its entry holds one number a call, `[rows, calls]`, and each call's logits are freed once the metric has been taken. Of `[{"op": "token_prob", ...}, {"op": "mean", "over": ["pos"]}]`, `token_prob` runs at each call and `mean` after the generation.

- **A read that waits is held whole until the generation ends.** That is a read with no ops, or whose first op waits. At the head it holds `calls × vocab_size` values a row in the model's dtype, as much again while the calls are arranged, and a float32 copy for a metric.
- **To hold less of a read that waits,** read only the ids you score (`select: {"vocab_size": [...]}`, raw logits, which no metric scores, since a metric needs the whole vocabulary), or lower `--batch-size`.

An op waits where its bits depend on the shape of the value it runs on: torch orders a matmul, or a sum over a long axis, by that shape. A call's value is `[rows, 1, ...]` where the whole read is `[rows, calls, ...]`, so running such an op at each call would move its last bits. On the CPU a per-call metric equals the same metric over the whole read. On CUDA a per-call softmax metric can differ from it in the last bits.

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

Each entry's record holds where it landed (`positions`, by dataset). `damped.prompt` is `[[10], [8], [8], [8]]` and `damped.decode` is `[[11, 12, 13, 14, 15], [9, ...], ...]`: every generated input except the last token, which is never fed. `clab show` prints both, one line per entry.

To write at every position, the whole prompt included, use `{"frame": ":"}`. A root bundle written at `{"frame": ":"}` covers every call of whichever step pulls it in: every token of a forward, and every prompt token and generated input of a generation.

Because a write lands as its call runs, everything it selects must be known by then. It cannot count back from a row's last call. Its index must name the same calls on a row that stops early as on one that runs on. For the prompt's last token and every decode call, write the two entries above:

```text
clab: steps.damped.interventions.zero.pos: frame -1 with index ':' names calls that depend on how far each row gets; …
clab: steps.damped.interventions.zero.pos: frame ':' with index -1 counts back from the row's last position, known only when the row stops, …
clab: steps.damped.interventions.zero.pos: frame '2:' with index ':' names calls up to the row's last; …
```

The last of these needs `max_new_tokens` on the step, which bounds the calls. Steering with a harvested direction through a generation is in [patterns/steering.md](../patterns/steering.md).

## Writes at generated text, found online

A write with a `column` or `text` scope over the decode calls is resolved while the row generates. At each call, clab decodes what the row has generated so far.

- The write lands at the call fed the last token of the occurrence, so the text changes from the next token on.
- A row that never says the text is `alignment_missing` in the write's `positions`, and it generates as if unwritten.

A variation of `documents/generated_answer.json` zeroes layer 0 where each row says its `said` word:

```json
"zero": {"site": "l0", "pos": {"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}, "ops": [{"op": "scale", "factor": 0.0}]}
```

It lands at `[[12], [12], 'alignment_missing', 'alignment_missing']`: row 0 at the call fed `▁substr`, and rows 2 and 3 never say their word. Such a write takes index `-1`, counts `occurrence` from the first, and takes no `before`. Each of these is refused by name:

```text
clab: steps.answer.interventions.zero.pos.index: index 0: a scoped write lands at the call fed its text's last token, …
clab: steps.answer.interventions.zero.pos.scope.occurrence: occurrence -1 counts from the last occurrence, which is known only when the row ends; …
clab: steps.answer.interventions.zero.pos.scope.before: a write beyond the prefill lands as its call runs, before the token it would predict is chosen; …
```

A read is resolved after the row has finished generating, so these limits do not apply to reads. A read over generated text can take `before`, any index and any occurrence; the recipes are in [usage/positions.md](positions.md).

## Rows that stop early

A row stops at its first EOS. On the first four rows of the `test` split of `data/weekdays_full/data.json`, with `"eos_token_id": 4827` (`▁Alex`) and `max_new_tokens: 8`, row 1 stops after 3 tokens and rows 0, 2 and 3 run to 8:

```json
"days": {"path": "weekdays_full/data", "split": "test", "take": 4},
"g": {"kind": "generate", "input": "days.input", "max_new_tokens": 8, "do_sample": false, "eos_token_id": 4827,
      "interventions": {"at3": {"site": "head", "pos": {"frame": 3}}}}
```

- **The step's ids keep the shape `[4, 8]`.** Every position after a row's EOS is filler, so read the text through `detokenize`, which stops before the EOS. `contains` therefore never finds the EOS token's text.
- **To see where a row stopped,** read `plan.run.tokens["g"]`, each row's tokens up to and including its EOS: `['Handler', '▁tijd', '▁Alex']` on row 1. A `count` of a read at `{"frame": "1:"}` is the decode calls each row made: 2 on row 1, 7 on the others.
- **A read at a call a row never made is missing on that row.** `{"frame": 3}` gives row 1 one table row, with `alignment_missing` true and no value. Such a read has no value on that row as a later write's operand either; hold EOS off with `min_new_tokens` before using it.
- **A ragged read (`"1:"`, `":"`) holds each row's own calls.** Positions past a row's last call are NaN in the read, so an `argmax` of them is -1, and the table leaves them out.
- **`{"frame": -1}` on row 1 is call 2,** the call that chose EOS. Its `argmax` is the EOS id.
- **With `min_new_tokens: 8`, every row makes all eight calls,** and `{"frame": 3}` has a value on every row. A row held past its EOS chooses another token where it would have stopped: row 1 says `▁note` at token 2 and goes on from there. That is a different generation, not the first one extended.

How missing rows pass through ops and tables is covered in [errors/alignment-missing.md](../errors/alignment-missing.md).

## Pitfalls

- **Sweeping a write's frame or a generate option.** Each point runs its own generation. A sweep of a read's frame shares one, and `{"frame": "0:3"}` reads the same calls as one value ([usage/sweeps.md](sweeps.md#widen-the-read-instead)).
- **A word that ends on the row's last generated token.** No call is fed that token, so a read scoped to the word without `before` is `alignment_missing`. With `max_new_tokens: 2`, a word said as token 1 is missing at the word itself, and found with `before`. Leave room after the answer.
