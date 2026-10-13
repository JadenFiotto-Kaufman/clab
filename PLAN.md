# clab — what I believe, and the plan

For ratification before any code is written. Everything here is either a
decision you have already made (marked **fixed**), my reading of the corpus
(marked **reading**), or a proposal (marked **proposal**). Where you disagree,
say so and I change the line; silence is a yes.

## 1. What clab is

**fixed.** A JSON format for mechanistic-interpretability experiments and the
engines that run it. The format is engine-agnostic. Two engines ship: nnterp
(primary, remote-capable through nnsight sessions) and plain HuggingFace +
forward hooks (the seam-prover; may use an nnterp meta shell for names, sizes
and `support()`, but executes only with `AutoModelForCausalLM` and hooks).
`engine.execute(plan, **kwargs)` is the single entry point. The 31 documents
under `documents/` run as-is forever; the format may grow and may be tightened,
but only after you ratify each change.

## 2. What the corpus already says (reading)

The 31 documents use one small vocabulary. I take it as the format's base.

**Root keys:** `header`, `model`, `data`, `sites`, `interventions`,
`featurizers`, `steps`. The last four are optional. `steps` is an ordered
object; its final entry `saves` is not a step but the list of what comes home.

**Step kinds (five):** `forward`, `generate`, `metric`, `reduce`, `fit`.
Counts in the corpus: 68 / 3 / 66 / 5 / 2.

**Components (the address vocabulary):** `block_output`, `attention_output`,
`attention_probs`, `attention_query`, `attention_premix`, `mlp_activation`,
`lm_head`, `logits`. A site narrows a component with `layers` (an int, `"all"`,
or a sweep), `heads`, `units`, `features`. `view: logits` on a residual-stream
read applies the model's own readout (the logit lens).

**Positions (`pos`):** an int counted from the end; `{"span": [a, b]}`;
`{"all": true}`; `{"index": k | "all": true, "scope": {"variable": <column>}}`
(the tokens of that row's own text, located in its prompt);
`{"scope": {"segment": "user[1]" | "eos"}}` (a chat turn, or where the row
stopped); `{"frame": "generated", "index": k}` (a decode step).

**Mechanisms:** `swap`, `add_scaled` (with `params.scale`), `renormalize`.
Operands: a number, or a name. Several writes on one site compose in listed
order (`steer` then `restore`).

**Metrics:** `match`, `logit_diff`, `token_prob`, `token_logit`,
`cross_entropy`, `kl`, `js`, `top_k`; `token_form: bare` picks the unspaced
spelling. **Reduces:** `mean`, `pca` (with `k`).

**Featurizers:** `subspace` (cayley-parametrized rotation, `k`, optional
`seed`, optional `file_path`), `gate` (a soft mask, annealed), `pca` (loaded
basis), `sae` (loaded dictionary with an error term). A read or write with
`featurizer` happens in that featurizer's space; `features: [j]` narrows a
write to latents.

**`fit`:** trains named featurizers over a nested `steps` body; `objective` is
a weighted sum of the body's metric steps (weights may sweep); `optimizer`,
`epochs`, `batch_size`, `seed`, `early_stop` (on a body metric), `anneal`,
and `eval.data` (re-run the body with one dataset substituted for another).
`fit.rot` names the featurizer: the one being trained inside the body, the
trained one after.

**Sweeps:** `{"sweep": [...]}` may appear on a site's `layers`/`heads`/
`units`, a write's `features` or `pos`, a generated `index`, a featurizer's
`seed`, or an objective weight. At most one per document today.

**Names:** one flat namespace. `counterfactual.v_cf` is step `counterfactual`'s
read `v_cf`; `mean` is a reduce step's value; `pairs.cf_answer` is a dataset
column; `fit.rot` is a featurizer owned by a fit; `mask.mask` is a featurizer's
own scalar. Resolution is always "thing, then member".

**Data:** `path: "weekdays/train"` is `data/weekdays/train.json`;
`"weekdays/data#test"` filters its rows by the `split` column;
`field: "counterfactual_inputs[0]"` indexes into a column. A `chat` column is
a list of role/content messages rendered through the checkpoint's template.

**Semantics the headers spell out, which I adopt as given:** a row whose
variable text is not in its prompt is refused by name on a write, contributes
nothing on a ragged read; "did the model say it" under `generated` is a
reported result (`alignment_missing`), not an exception; attention-pattern
patching checks the two prompts tokenize to equal lengths before any forward;
a four-row single-position read has no rank-4 PCA and the compiler says so; a
window write of `[-4, -1]` and three single-position writes produce identical
logits; `layers: "all"` yields one value with the layer axis first; a sweep
point is a whole experiment and the root is a list of them.

## 3. Decisions taken today (fixed)

- The engine instance may be referenced inside the session: `steps.run(engine,
  plan, ...)`, `engine.model`, `engine.forward`. It holds nothing else
  client-side.
- Metrics, reduces (pca included) and the entire `fit` loop run inside the
  session. Nothing reaches the client before the document ends.
- The result is the filled plan. `nnsight.save(plan)` at the end of the session
  is all that comes home. With sweeps, a swept step becomes a list of points.
- `saves` is the whitelist of named values that come back in the plan. Every
  other value is ephemeral: it lives in the cross-step state during the run and
  is gone after.
- Sweeps expand at compile time; the engine never sees `{"sweep": ...}`.
- The output is a directory: the plan written out as its own hierarchy, with a
  manifest at the root.
- `address.py` carries the corpus's component names plus the obvious
  residual-stream and attention-interior siblings nnterp serves.
- `execute`, `forward`, `generate` take a `batch_size`. Left padding is the
  convention.
- Process: I plan and decide; Opus agents do every code task; one PR per
  milestone; I open the PR and stop for your review.

## 4. Charter amendments (proposal — exact wording)

Replace law 3 with:

> 3. **Inside a session, the engine and its model are the only client-side
>    objects.** Code that runs inside the session may reference the engine
>    instance (`engine`, `engine.model`, `engine.forward`) and the plan; it
>    may not close over anything else client-side: no loaded datasets, no
>    separately-constructed tokenizer, no open files, no helper objects. The
>    engine holds nothing but its model and its configuration. Everything else
>    is module-level functions over plain data. (nnsight ships every name a
>    block closes over, by value, to wherever the block runs.) The model's own
>    tokenizer is reached through the model inside the session.

Add to "Structure the owner has fixed":

> - The result of a run is the filled plan: every step's result slots filled
>   in place, swept steps as lists of points. `nnsight.save(plan)` at the end
>   of the session is the only thing that comes home. `saves` names which
>   values are kept in the plan; the rest are ephemeral cross-step state.
> - The output of a run is a directory: the plan written out as its own
>   hierarchy, with a manifest at the root.

Add to "Process":

> - Design and code are done by agents briefed from this charter and
>   `PLAN.md`. An agent proposes format changes in its PR description with
>   before/after JSON; none lands without the owner's ratification.

## 5. Architecture (proposal)

The step-0 design agent will flesh this out and may push back; this is the
shape I expect.

```
clab/
  document.py    load + validate a JSON document (names resolve, kinds known)
  compile.py     document -> plan: pure data; sweeps expanded; sites/
                 interventions/featurizers inlined by reference; result slots
  plan.py        the plan's data shape (dicts/lists/tuples + slots); walk/fill
  address.py     the component vocabulary: name -> meaning, axis names, what
                 may narrow it (layers/heads/units/features), value kind
  positions.py   pos specs -> per-row token index tensors, resolved on the
                 serving side from ids/offsets (law 4). Note: offset mappings
                 exist only on fast tokenizers, and some checkpoints ship only
                 slow ones — in the future this may need the char->token
                 alignment tracked manually (decoding growing prefixes of the
                 ids), or dual paths: offsets when the tokenizer is fast,
                 manual alignment when it is not.
  data.py        load data/<path>.json, #split filter, field access, chat
                 rendering, batching with left padding
  state.py       the cross-step namespace: name -> value; "thing.member"
  steps.py       run(engine, plan, batch_size): loop + dispatch on kind;
                 forward/generate/metric/reduce/fit as functions taking engine
  metrics.py     the eight metrics, module-level, over plain tensors
  reduces.py     mean, pca
  featurizers.py subspace/gate/pca/sae modules; load/save; the fit loop's needs
  output.py      plan -> directory hierarchy + manifest; safetensors/json
  cli.py         clab run <doc> [--engine] [--out] [--remote] [--batch-size]
  engines/
    base.py      the protocol: execute(plan, **kw); forward(batch, writes,
                 reads); generate(...); resolve(address) -> engine-side handle;
                 readout(hidden) for view: logits
    nnterp.py    StandardizedTransformer; execute opens one session, calls
                 steps.run(self, plan, ...), nnsight.save(plan)
    hooks.py     AutoModelForCausalLM + register_forward_hook; meta shell for
                 names; refuses source-located components by name
```

Key seams:

- **`compile` is the only place the document's spellings are interpreted.**
  Engines receive a plan whose every site is already `{component, layer,
  slice}` and every intervention is already a list of `{address, pos,
  mechanism, operand}` records. The hooks engine proving the seam means: it
  reads the same plan, resolves the same `address.py` names, and never looks
  at a document.
- **`steps.run(engine, plan, batch_size)` is engine-agnostic**: it loads and
  batches data through `data.py`, calls `engine.forward`/`engine.generate`
  per batch with the step's writes and reads, puts reads into the state,
  evaluates metrics/reduces from the state, drives `fit`, and fills plan
  slots for whatever `saves` names. It runs identically inside an nnsight
  session (nnterp) and in plain Python (hooks). A future engine may ignore it.
- **The plan is pure data** (law 2): nested dicts/lists with a `Slot`
  placeholder for results; picklable, printable, diffable.
- **Positions resolve where the model is** (law 4): the plan carries the pos
  spec; `positions.py` turns it into indices from the batch's ids and offsets
  inside the run.
- **Remote is one flag**: `execute(plan, remote=True)` changes only the
  `model.session(remote=...)` argument. `remote="local"` (nnsight's in-process
  serialize/deserialize/execute) is in the test suite from M1. Scope it
  honestly: it proves the OUTBOUND path — no closed-over client objects, no
  by-value shipping (laws 3 and 5) — but it does not serialize the way home,
  so the `nnsight.save(plan)` return trip is only truly exercised against a
  real server (M5). Until then the way home is a design believed, not proven.

Output directory (proposal): `out/<doc>/manifest.json` (document, model,
engine, versions, sweep axes, where each saved value landed) and one
subdirectory per step that has saved values — `out/<doc>/patched/logits.safetensors`,
`out/<doc>/iia/table.json`; a swept step is `patched/0/`, `patched/1/`, …,
each point's directory holding its coordinates. A filename given in the
corpus's `saves` (`"compare/patching/iia.json"`) is honored as the path within
the step's directory, so `two_observes.json` groups its tables as its header
describes; a bare list form (`"saves": ["kl", "js"]`) uses the default names.

## 6. Format tightenings to expect (proposal, for the design agent to argue)

The design agent is told the format is open to change if it makes the
abstractions tighter or more flexible, and that every change needs
before/after JSON and your ratification. Candidates I already see:

- Whether `saves` stays inside `steps` (as the corpus has it) or moves to the
  root; the corpus form must keep working either way.
- Whether `component`+`layers`+`heads`/`units`/`features` becomes one
  address string (`"layers[0].attention_premix[heads=0]"`) or stays a dict.
- Whether `interventions` (name | list | inline dict) and `site` (name |
  inline dict) keep three spellings each.
- Whether the five pos forms share one shape (`frame`, `index`/`span`/`all`,
  `scope`) explicitly.

## 7. Milestones (proposal — one PR each; I open it and stop)

- **M0 — design.** `DESIGN.md`: the plan's data shape with a worked example
  (patching.json compiled), the module layout, the engine protocol, the
  address vocabulary, the output hierarchy, and the format tightenings with
  before/after JSON. No code. You ratify.
- **M1 — one document end to end.** `patching.json` on both engines, bit-equal
  between them; CLI; `uv sync` environment; tests: semantics snapshot,
  engine agreement, `remote="local"` round trip; `uvx pyright` clean.
- **M2 — every forward-only document** (20): positions and windows, scopes
  (variable, segment, chat template), ragged reads, sweeps, `layers: "all"`,
  `view: logits`, reduces, all eight metrics, GPT-2, the composed writes. The
  hooks engine serves everything that is not source-located and refuses
  `attention_probs` / `attention_query` by name.
- **M3 — generation** (3): `generate` steps, the `generated` frame, `eos`
  segment, `alignment_missing` as a result.
- **M4 — featurizers and fit** (7): subspace/gate/pca/sae, the in-session
  training loop with early stop, anneal, eval substitution, bundle save/load
  with header checks (`das.json` then `das_apply.json`).
- **M5 — real remote** against a server, when you want it.

Acceptance at every milestone: the documents it claims run from the CLI on
CPU with `HF_HUB_OFFLINE=1`; `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`
green; both engines agree numerically where both serve; snapshots checked in.

## 8. How the agents are run (proposal)

- Model: Opus 5. One agent per task, in its own worktree on branch
  `m<N>-<name>`, briefed with: the charter, this file, `NNTERP.md`, the
  nnterp checkout's docs, the clean-room rule, and the specific milestone's
  acceptance list. Agents commit (one declarative sentence per commit) and
  report; I open the PR with the summary and the format proposals, and stop.
- A separate Opus reviewer agent reads each milestone branch cold (code, not
  my conclusions) before I open the PR, so what you review has had one
  independent pass.
- I do not read or edit the code myself; I read the agents' reports and the
  PR descriptions, and I decide.

## 9. Still open (say yes, or change it)

1. Charter amendments in §4: I apply them verbatim once you say yes.
2. The `saves` filename convention in §5 (honored within the step's
   directory).
3. The output hierarchy in §5.
4. The M0 design PR as a no-code PR.

## 10. Ratified after the M0 review (2026-10-13)

The owner read DESIGN.md and ratified it with the changes below. Where this
section and DESIGN.md or §5 above disagree, this section wins. M1 is built
from DESIGN.md as amended here.

### DESIGN.md §12 — all ten defaults accepted

1. Un-swept steps run once and are shared across points.
2. Bundle tensors a document names by `file_path` travel inside the plan
   (input files only).
3. M4 commits a seeded generator for `documents/artifacts/sae.safetensors`
   and the file it writes.
4. Gate: θ₀ = +1 on every unit; temperature linear over epochs; the best
   epoch's featurizer is restored.
5. Two documents run into one `--out` share it; the second manifest replaces
   the first.
6. M5 runs against a self-hosted NDIF with clab and nnterp installed.
7. Early stopping is judged on the training data in scoring mode; the eval
   substitution runs once, after the fit.
8. Sweeps: none under `description` or `model`, none on a data `path` or a
   `file_path`, none on a name or a kind; everything else may sweep.
9. In the generated frame an ambiguous variable match is reported per row as
   `alignment_ambiguous`; in the prompt frame it stays refused.
10. Output layout: files at the paths `saves` gives, the plan's hierarchy in
    `manifest.json` (charter reworded accordingly).

### One spelling per thing; the corpus is migrated

The format has exactly one spelling for each thing. No alias table exists in
`compile`. The 31 documents are rewritten once, in M1, into the canonical
spelling (a mechanical migration, reviewed as a diff in the M1 PR); from then
on the corpus is fixed. The one deliberate two-way pattern is *reuse vs
inline*: a thing declared at the root (`sites`, `interventions`,
`featurizers`) is referenced by name, or the same thing is written inline
where it is used once.

### The canonical format

- **Root:** `description`, `model`, `data`, `sites`, `interventions`,
  `featurizers`, `steps`. (`header` is gone.)
- **Step kinds:** `forward`, `generate`, `fit`, `op` — plus the reserved
  `saves` entry of `steps`, unchanged (dict `name -> file` or list of names).
  `metric` and `reduce` are gone: both are `op`. `generate` stays its own
  kind (nnsight's `generate` is its own method); it may share helpers with
  `forward`.
- **Ops are the one computation vocabulary**, in `ops.py`, each a plain
  function with a declared signature and declared axis behaviour. The same
  ops run in three places: as a step (`{"kind": "op", "op": "mean", "of":
  "harvest.acts"}`), inside a read (`"ops": [...]`, applied to the gathered
  value before the name binds), inside a write (`"ops": [...]`, applied to the
  live value at the site; the result is written). "Mechanism" no longer
  exists: `swap`, `add` (with `scale`), `renormalize`, `scale`, `project_out`
  are ops. `readout` (the logit lens; the one op that takes the engine) and
  `encode` (into a featurizer's space) are ops. Metrics (`match`,
  `logit_diff`, `token_prob`, `token_logit`, `cross_entropy`, `kl`, `js`,
  `top_k`) and reductions (`mean`, `pca`, `sub`, `add`, `normalize`) are ops.
  `mean` takes `over` (default: every axis but the feature axes).
- **The one in-forward rule:** an op inside a read or write sees one batch,
  so it must act per example; an op that aggregates over `rows` is refused
  there by name ("make it a step after the forward"). Each op declares its
  axis behaviour so compile can check this from the plan alone.
- **`featurizer` (and `features`) are arguments on any op** in a read or
  write: the op runs on the value's coordinates in that space and the delta is
  written back (DESIGN.md §9's delta rule, now for every op). `view: logits`
  is gone (`[{"op": "readout"}]`); `featurizer:` on a read is gone
  (`[{"op": "encode", "featurizer": F}]`).
- **A forward's input:** `"input": "pairs.counterfactual_inputs[0]"` (the
  thing.member syntax). `data` + `field` are gone.
- **Writes and reads sit directly on the step** (`"writes": {...}`,
  `"reads": {...}`). `"interventions": "das"` or `["steer", "restore"]` pulls
  a named root bundle into the step; the inline `interventions: {writes,
  reads}` form is gone.
- **Component names are nnterp's**: `layer_output`, `layer_input`,
  `attention_output`, `attention_probabilities`, `attention_queries`,
  `attention_keys`, `attention_values`, `attention_scores`,
  `attention_head_outputs`, `attention_premix`, `mlp_activation`,
  `mlp_neurons`, `mlp_output`, `lm_head_output`, `logits`, `token_embeddings`,
  `final_norm_output`, and the norm boundaries; DESIGN.md §4's table is
  re-keyed on them. The corpus's `block_output`, `attention_probs`,
  `attention_query`, `attention_z`, `lm_head` are migrated.
- **`fit`:** `train`, `objective` as a dict `{"ce": 1.0, "mask.mask":
  {"sweep": [...]}}`, `epochs`, `batch_size`, `seed`, `optimizer`,
  `early_stop`, `eval`, `steps`. `anneal` is gone: a gate declares its own
  schedule, `{"kind": "gate", "temperature": {"start": 1.0, "end": 0.05}}`.
- **Names and scoping (DESIGN.md §6, confirmed):** values are produced in
  document order and referenced only after they exist; a fit's body sees every
  root value produced before the fit; nothing leaves the body except through
  `fit.<member>`: `fit.<featurizer>`, `fit.<body metric>` (eval run, after
  training), `fit.<body step>.<read>`, and `fit.history.<metric>` (the
  per-epoch scoring values, `[epochs]`), which is added.
- **`saves`** is declared only at the root of `steps`, never inside a body;
  inner values are named through the owning step.
- **`data`** may hold inline rows beside `path`: `{"rows": [...]}`.
- **`pos`:** an int, or the one dict grammar of DESIGN.md §5. The int
  shorthand is the one deliberate shorthand.

### API

- `clab.run(path, engine="nnterp", out=None, batch_size=8, remote=False) -> plan`
- `clab.load(out_dir) -> plan` — the inverse of `output.write`: manifest +
  files back into the filled plan, tensors loaded.
- `clab.table(plan, name)` — one flat table over every point of a saved op.
- The CLI (`clab run`) is a thin wrapper over `clab.run`.

### Scope

- Gradients as a readable quantity (attribution patching and the like): out
  of scope.
- Probes: in scope, inside the training abstraction. A `linear` featurizer
  (`{"kind": "linear", "classes": "pairs.label"}`: `encode(h) = h @ W + b`,
  the output axis being the column's distinct values in sorted order) trained
  by `fit`; metrics resolve a column against the value's own vocabulary (the
  tokenizer's for logits, the class list for probe scores), so `cross_entropy`
  and `match` work unchanged against a label column.
- M1 includes the corpus migration and `patching.json` end to end on both
  engines in the canonical format.
