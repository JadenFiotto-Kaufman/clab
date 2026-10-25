# clab — roadmap

What is agreed but not built, in no order. Bring an item up when the owner asks for open items.

- **Writes across layers.** A write's `layers` may be a slice or a list of picks, meaning one write per
  layer in one spelling. An operand with a `layers` axis is aligned to the write's layers by label (a read at
  `layers: ":"` into a write at `layers: ":"` pairs layer by layer); an operand without one is broadcast, as a
  direction already is. Refused: a `layers` selection the operand's axis does not cover. Replaces the
  hand-unrolled one-write-per-layer documents (trial 1's 3,300-write window grid, trial 5's 108-write
  all-layer ablation).
- **Sweep labels.** A sweep object may carry `"label": "<name>"`; the default label stays the field name.
  Same label zips, different labels cross, exactly today's rule with one more way to name a sweep, so a
  source-layer × target-layer grid is one step. The "rename one" refusal names `label`.
- **Float32 metrics and reductions.** `mean`, `sum`, `min`, `cosine`, `kl`, `js`, `token_prob`,
  `token_logit`, `logit_diff`, `cross_entropy`, `match` compute in float32 whatever the model's dtype and
  return float32; values read from the model keep the model's dtype. bf16 `mean` moved in steps of 0.0625,
  `cosine` rounded to 2⁻⁸ near 1, `token_prob` to 1.0 above ~0.998 (trials 2, 5, 11).
- **`renormalize`** gains `norm: "l2" | "sum"`, default `l2` (today's: the written value rescaled to the
  original's L2 norm); `sum` divides so the row sums to what it summed to before, the attention-pattern
  meaning (trial 10).
- **`strip` on an answer** trims whitespace at the two ends of the whole answer only: an interior
  whitespace token stays itself (no 441-id class), and a number's leading space token is kept (trial 11).
- **Dataset options**, all describing where rows come from and how text renders, never computing on
  values: `template` (a dict passed through to `apply_chat_template`, so a generate from a chat can carry
  the generation prompt and `enable_thinking` and keep `turn` scopes), `take` (first N rows, any source),
  `shuffle: true` (seed 0, recorded in the run), `columns` (a rename map so one file serves a fit's `eval`
  substitution). No `repeat`.
- **Op arguments.** `argmax` and `top_k` take `over` (an axis name, default `vocab_size`) so they rank a
  featurizer's coordinates or a probe's classes; `scale`'s `factor` and `noise`'s `scale` accept a named
  value as well as a literal (the `Value` kind `with` already has), so a dose can be a measured norm.
- **Hub datasets** (`hf`, pinned by revision; rows materialize where first touched).
- **Provenance in `manifest.json`**: the document's digest, a digest per loaded bundle, the device, thread count,
  GPU and driver, a run seed for sampled generation, the command line; on remote runs the server's job id. And
  `clab rerun <out> [--check]`, comparing every result to the original to the bit.
- **Efficiency**: per-step seconds and peak memory in `plan.run`; stop the forward after the deepest site; one
  forward shared by read-only sweep points; a fit-constant cache under `no_grad`; quantized `dtype` names;
  tensor parallelism through nnsight's `distributed_config`.
- **A cross-site adapter inside a generation** (an operand from an entry of its own step is served in a forward
  only).
- **A sequence metric** beyond what `before` and `sum` give; featurizer composition beyond nesting; `subspace`'s
  parametrization and its learning-rate scale; a dependency rule so that a step referencing a swept step
  inherits only the axes its value can depend on.
- **A clab skill for agents** (`~/wd/clab-trials/SKILL-NOTES.md` is the raw material).
