# clab — roadmap

What is agreed but not built, in no order. Bring an item up when the owner asks for open items.

- **Provenance in `manifest.json`**: the document's digest, a digest per loaded bundle, the device, thread count,
  GPU and driver, a run seed for sampled generation, the command line; on remote runs the server's job id. And
  `clab rerun <out> [--check]`, comparing every result to the original to the bit.
- **Efficiency**: per-step seconds and peak memory in `plan.run`; stop the forward after the deepest site; one
  forward shared by read-only sweep points; quantized `dtype` names; tensor parallelism through nnsight's
  `distributed_config`.
- **A cross-site adapter inside a generation** (an operand from an entry of its own step is served in a forward
  only).
- **A sequence metric** beyond what `before` and `sum` give; featurizer composition beyond nesting; `subspace`'s
  parametrization and its learning-rate scale; a dependency rule so that a step referencing a swept step
  inherits only the axes its value can depend on.
- **A clab skill for agents** (`~/wd/clab-trials/SKILL-NOTES.md` is the raw material).
