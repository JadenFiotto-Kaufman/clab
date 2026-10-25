# clab — roadmap

What is agreed but not built, in no order. Bring an item up when the owner asks for open items.

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
