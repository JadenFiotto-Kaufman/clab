---
title: Vision-Language Models
one_liner: "Experiments on a vision-language checkpoint: an image in a chat, its tokens as positions, a region of it as keys, a head's attention onto a part of the image, gaze heads found by a score and redirected by a bias on their scores toward a chosen region, scored without a judge."
tags: [patterns, vision, image, gaze heads, attention, region, steering]
related: [docs/usage/data.md, docs/usage/positions.md, docs/usage/sites.md, docs/usage/interventions.md, docs/usage/generate.md, docs/usage/engines.md, docs/patterns/attention.md, docs/patterns/head-and-neuron-patching.md, docs/patterns/steering.md]
sources: [clab/format/data.py, clab/compile/positions.py, clab/steps/batches.py, clab/engines/standard.py, documents/gaze_heads.json, documents/gaze/gaze_discovery.json, documents/gaze/comics.py]
---

# Vision-Language Models

A vision-language checkpoint is a text model with a tower that turns an image into tokens, which sit in the prompt like any other: positions, keys, a turn's content. clab reaches them through three things ([data.md](../usage/data.md#image-parts), [positions.md](../usage/positions.md#image-tokens), [sites.md](../usage/sites.md#select)): the model block's `"task": "image-text-to-text"`, which loads the checkpoint with its processor; an image part in a chat message's content, a file under `data/`; and the `image` scope, with a `region` of the image, on an entry's `pos` or on a site's `select` of `key`. Everything else is the text model's: the components, the ops, the sweeps, the engines (not `ndif`, which cannot read the file).

The recipes are the gaze-head experiments of Gandikota and Bau, 2026: a few heads of a vision-language model attend to the part of the image the model is describing, they are found by one score over a few forward passes, and a bias on their attention scores toward a chosen region makes the model describe that region instead. `documents/gaze_heads.json` is the whole pattern on the tiny Qwen3-VL, on CPU; `documents/gaze/gaze_discovery.json` and `documents/gaze/gaze_steer.json` run it on Qwen3-VL-8B-Instruct over 500 comic strips, on a GPU.

## The image in the document

`documents/gaze_heads.json` reads `data/images/halves.json`, four rows over two 64 x 32 images, red|blue and blue|red; each row asks which colour is on one side and carries that side's box in `left` and the other side's in `right`:

```json
{"example_id": "rb",
 "chat": [{"role": "user", "content": [{"type": "image", "image": "images/rb.png"},
                                        {"type": "text", "text": "Which colour is on the left? Answer with the colour only."}]}],
 "left": [0, 0, 0.5, 1], "right": [0.5, 0, 1, 1], "answer": "red"}
```

```json
"model": {"key": "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", "revision": "ba190cb38011bcda2f70f7781f2b8290627aef00",
          "dtype": "float32", "task": "image-text-to-text", "attn_implementation": "eager"},
"data": {"halves": {"path": "images/halves", "template": {"add_generation_prompt": true}}}
```

The processor renders the chat and expands the image part into the image's tokens: 72 on the tiny Qwen ViT, a grid of 6 rows by 12 columns, between `<|vision_start|>` and `<|vision_end|>`, so the prompt is 95 tokens and `run.tokens` shows 72 `<|image_pad|>`. `attn_implementation: eager` serves the attention interior, as on a text model ([attention.md](attention.md)).

## A head's attention onto a part of the image

The gaze score is the attention a head puts, from the last prompt token, on the part of the image the prompt asks about. That is the pattern at `pos: -1`, its keys narrowed to the region, summed over `key`:

```json
"sites": {
  "asked": {"component": "attention_probabilities", "layers": ":",
            "select": {"key": {"scope": {"image": 0, "region": "halves.left"}}}},
  "other": {"component": "attention_probabilities", "layers": ":",
            "select": {"key": {"scope": {"image": 0, "region": "halves.left", "outside": true}}}},
  "image": {"component": "attention_probabilities", "layers": ":", "select": {"key": {"scope": {"image": 0}}}}
},
"steps": {
  "gaze": {"kind": "forward", "input": "halves.chat", "interventions": {
    "asked": {"site": "asked", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]},
    "other": {"site": "other", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]},
    "image": {"site": "image", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]}}},
  "score": {"kind": "op", "op": "mean", "of": "gaze.asked"},
  "whole": {"kind": "op", "op": "mean", "of": "gaze.image"}
}
```

- **`region` is each row's own box**, `"halves.left"`: the asked side, whichever it is on the row. A literal box names the same part on every row; `outside` is the rest of the image.
- **The keys a region names are the tokens whose cell's centre lies in the box**, on the grid the processor gives (`image_grid_thw`, merged two ways) or a square count; `run.keys["gaze.asked"]` lists them per row, 36 of the 72 here, columns 0 to 5 of each grid row.
- **`sum` over `key` inside the entry** gives one number per row, layer and head: `gaze.asked` is `[layers, rows, pos, heads]`, `score` its mean over rows, `[2, 4]`. The three sums tie out: `score` plus the mean of `gaze.other` equals `whole` to float rounding, since the two regions partition the image.

On the tiny model the scores are 0.31 to 0.45 per head: random weights spread their mass. On Qwen3-VL-8B-Instruct the same read separates the heads sharply ([below](#gaze-heads-on-qwen3-vl-8b)).

## Redirecting the gaze

The paper's intervention adds +δ to the attention scores of the gaze heads on the target region's image tokens and −δ on the other panels', δ large enough to saturate, in the prefill and at every decode call, and leaves text keys and other heads alone. In clab that is two writes on `attention_scores`, each selecting the heads and the keys:

```json
"sites": {
  "toward_other": {"component": "attention_scores", "layers": 1,
                   "select": {"num_heads": [0, 2], "key": {"scope": {"image": 0, "region": "halves.right"}}}},
  "away_from_asked": {"component": "attention_scores", "layers": 1,
                      "select": {"num_heads": [0, 2], "key": {"scope": {"image": 0, "region": "halves.left"}}}},
  "pattern": {"component": "attention_probabilities", "layers": 1}
},
"steered": {"kind": "generate", "input": "halves.chat", "max_new_tokens": 2, "do_sample": false, "interventions": {
  "toward": {"site": "toward_other", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": 10000.0}]},
  "away": {"site": "away_from_asked", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": -10000.0}]},
  "pattern": {"site": "pattern", "pos": -1}}},
"null": {"kind": "generate", "input": "halves.chat", "max_new_tokens": 2, "do_sample": false, "interventions": {
  "toward": {"site": "toward_other", "pos": {"frame": ":"}, "ops": [{"op": "add", "with": 0.0}]}}}
```

- **A write may select heads beside keys a scope names.** The one-axis rule counts feature axes; the keys are each row's own ([interventions.md](../usage/interventions.md#pitfalls)).
- **`"pos": {"frame": ":"}` is the prefill and every decode call.** A read of the key axis beyond the prefill is refused, since the axis grows; a write to keys a scope names is not, since the prompt's keys keep their places as the generation appends to them.
- **The operand is a number.** Rows name different numbers of keys, so a value from another read would not line up; `add` by a literal is the paper's bias. 10000 saturates the softmax in float32 as in bfloat16: in `steered.pattern`, heads 0 and 2 put exactly 1.0 of their mass on the other side on every row, heads 1 and 3 what they put before.
- **`null` equals `plain` to the bit**: the same writes with `0.0` change nothing, which checks that the keys and heads are the ones meant. `steered` generates other tokens than `plain` on every row.

## Gaze heads on Qwen3-VL-8B

`documents/gaze/comics.py` builds the comics data under the corpus's data directory from the paper's 500 six-panel strips (360 MB of images, which the repository leaves out): each strip stitched as the paper does, a lettered copy with a random letter on each panel, and the rows ([the script's docstring](../../documents/gaze/comics.py) lists their columns). `documents/gaze/gaze_discovery.json` is the discovery over the 3,000 (strip, panel) queries, the read above with `region` the queried panel's box:

```json
"data": {"q": {"path": "comics/queries", "template": {"add_generation_prompt": true}}},
"sites": {"own": {"component": "attention_probabilities", "layers": ":",
                  "select": {"key": {"scope": {"image": 0, "region": "q.panel"}}}}},
"steps": {"gaze": {"kind": "forward", "input": "q.chat", "interventions": {
            "own": {"site": "own", "pos": -1, "ops": [{"op": "sum", "over": ["key"]}]}}},
          "score": {"kind": "op", "op": "mean", "of": "gaze.own"}}
```

`score` is `[36, 32]`, each head's mean mass on the queried panel over the 3,000 queries: the mean over strips and panels of the diagonal of the paper's 6 x 6 queried-by-attended matrix, in one forward step. One row is one query, so no grouping is needed: the mean over all rows is the diagonal mean. The strip is 1536 x 256 pixels, 384 image tokens in 8 rows by 48 columns, a panel 8 by 8; `others_mean` and `image_mean` are the same read outside the panel and over the whole image.

On an RTX A6000 in bfloat16 the 3,000 queries take about 570 s at batch size 8. What comes back:

- **The gaze heads sit in a band of mid-late layers.** The top 100 heads by `score` (of 1,152; the cutoff is 0.152) fall in layers 18 to 34, 86 of them in layers 20 to 28, the band the paper reports; the top three are head 4 of layer 22 (0.71), head 18 of layer 21 (0.68) and head 11 of layer 21 (0.68). The gaze heads' mean score is 0.293 against 0.036 for the other heads of layers 20 to 35 and 0.046 over all heads.
- **A gaze head looks at whichever panel is asked about**: head 4 of layer 22 puts 0.62 to 0.80 of its mass on the queried panel, panel by panel, and 0.82 on the strip as a whole, so almost all of its image mass is on the right panel.
- **The set is stable.** Resampling the 500 strips with replacement 200 times and recomputing the top 100 keeps 99% of the set on average and 98% at worst; the per-row values are in `gaze.own`, `[36, 3000, 1, 32]`, which is what a bootstrap needs.
- **The same heads lead on Qwen3-VL-4B**: its top five by the same read are all among the 8B's top seven.
- **`score` plus `others_mean` equals `image_mean`** to float rounding, as on the tiny checkpoint.

### Redirecting the answer, scored without a judge

The paper judges redirection with an LLM. `documents/gaze/gaze_steer.json` scores it with a letter instead: the lettered strips carry a random letter on each panel, the prompt asks for the letter on the k-th panel (`chat`) or on "this panel" (`ambiguous`), and the gaze heads are biased toward another panel, the row's `target`; `match` against `target_answer` is 1 where the most likely next token is that panel's letter, and against `answer` where it is the asked panel's. A one-letter answer is chosen in the prefill, so the writes are forwards at `pos: -1`, not generations. The document was written from the discovery's `score` by a few lines of Python: the top 100 heads by layer, each layer's heads one entry toward the target and one away from it, and the same for 100 random heads of layers 20 to 35 outside the set (the paper's control) and for the gaze heads biased by 0 (the null):

```json
"g_in_22": {"site": {"component": "attention_scores", "layers": 22,
                     "select": {"num_heads": [0, 2, 4, 5, 11], "key": {"scope": {"image": 0, "region": "lq.target"}}}},
            "ops": [{"op": "add", "with": 10000.0}]},
"g_out_22": {"site": {"component": "attention_scores", "layers": 22,
                      "select": {"num_heads": [0, 2, 4, 5, 11], "key": {"scope": {"image": 0, "region": "lq.target", "outside": true}}}},
             "ops": [{"op": "add", "with": -10000.0}]}
```

Over the 3,000 (strip, target) pairs, about 83 minutes on the A6000 (eight forwards of 3,000 rows, most of it the processor's work on the images):

| question | condition | answer is the target panel's letter | answer is the asked panel's letter |
|---|---|---|---|
| "the k-th panel" | clean | 0.000 | 1.000 |
| | gaze heads | 0.664 (95% CI 0.644 to 0.683) | 0.073 |
| | 100 random heads, layers 20 to 35 | 0.033 (0.027 to 0.040) | 0.959 |
| | null (gaze heads, bias 0) | 0.000 | 1.000 |
| "this panel" | clean | 0.171 | 0.167 |
| | gaze heads | 0.660 (0.636 to 0.684) | 0.040 |
| | 100 random heads, layers 20 to 35 | 0.669 (0.644 to 0.693) | 0.068 |
| | null | 0.171 | 0.167 |

- **The model reads the letters**: asked for the k-th panel's letter, it answers it on every row; asked about "this panel", it answers at chance, with no preferred panel.
- **The gaze heads overrule the instruction.** Biased toward another panel, the model answers that panel's letter on two thirds of the rows against a prompt that names the panel; the same bias on 100 random heads of the same layers leaves the instruction intact (3%). The confidence intervals are bootstraps over strips (2,000 resamples), as the paper reports them.
- **An ambiguous question is redirected by either set.** With no panel named, 100 random mid-late heads forced onto a panel resolve the question to it as often as the gaze heads do. The paper's non-gaze control fails on its description task (14.6%); reading a letter asks less of the model than describing a scene, and any hundred heads whose image attention is pinned to a panel bring that panel's letter forward. The instruction-overriding case is the one that separates the sets here.
- **The null is the clean run to the bit**: `kl` between the null's and the clean logits is exactly 0 on both questions, so the 68 entries touch the keys and heads they name and nothing else.
- Redirection is weaker toward the first panels (0.52 to 0.61 for panels 1 to 3, 0.69 to 0.82 for panels 4 to 6) and from the first and last asked panels (0.53 and 0.47): the per-row `gaze_kth_hit` is what such a breakdown reads from.

## Pitfalls

- **A region's grid is the processor's.** On the Qwen ViTs it is `image_grid_thw` merged two ways on each side; on a fixed-resolution tower (Llava's CLIP, Gemma 3's SigLIP) a square count. A tower clab knows no grid for refuses `region` by name; `{"scope": {"image": 0}}` still names the whole image.
- **A batch's image tensors are the processor's, concatenated row after row.** That is the layout of the Qwen ViTs, CLIP and SigLIP; a processor whose image tensors carry a per-row axis (Idefics 3's `pixel_attention_mask`) cannot batch rows with different numbers of images, and the error is torch's, not a refusal. Run such rows at `--batch-size 1`.
- **A head's mass on the image is not its mass on a panel.** `image_mean` is large on early blocks of most models, where the text reads the image out; the gaze score asks for the queried panel in particular, which is why the paper keeps the attention raw rather than normalized over panels: a head that ignores the image scores zero either way.
- **Compare the bias to a null, not to clean alone.** `add` by 0 at the same heads and keys must equal the plain run to the bit; if it does not, the keys or the heads are not the ones meant ([interventions.md](../usage/interventions.md#the-null-control)).
- **The hooks engine does not serve the attention interior**, so the pattern reads and the score writes run on nnterp; what both engines serve (the residual stream at the image's positions, the logits, a generation) agrees to the bit on the tiny checkpoints.
- **Batch size and padding.** Rows are padded on the left as text rows are; a model that lays its own position ids out over the image (the Qwen-VL family) is given none and does so from the mask, and the numbers do not depend on the batch's companions ([running.md](../usage/running.md#batch-size-and-reproducibility)).
