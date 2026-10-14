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

### Discoverability

The code is the spec. Every extensible thing is a dict of plain functions
whose signature is its contract: ops, step kinds, components, featurizer
kinds. A function registers itself under its own name through its decorator
(`@op` adds the axis declaration and the registration in one line; there is
no hand-maintained `OPS = {...}` dict). From those dicts and signatures:
compile binds arguments with `inspect.signature` and names the valid choices
in every error; `clab describe [ops|steps|components|featurizers|pos]`
prints the tables by introspection; `clab describe --markdown` generates
`FORMAT.md`, committed, with a test that fails when the committed copy
differs from the generated one; `clab schema` (M2) emits a JSON Schema from
the same source. No pydantic.

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

## 11. Ratified after M1 (2026-10-13)

Decided while reviewing PR #2; M2 implements them first, before the remaining
forward-only documents.

- **Results live at the root of the plan**, keyed by the names in `saves`:
  `plan.results["iia"]`, `plan.results["fit.rot"]`. Step records carry no
  result slots. `steps.run` fills one dict; `output.write` iterates one dict.
- **The document and the plan are pydantic models** (`Document`, `Plan`), the
  record models (reads, writes, ops, pos, sites, featurizers) shared between
  them where the shape is the same. Validation and error messages come from
  the models; `clab schema` is `Document.model_json_schema()`; `FORMAT.md` is
  generated from the models plus the op registry. Ops keep validating their
  arguments by signature. The hand-rolled validation in `compiler.py` goes.
- **The plan's methods are views and serialization, never execution**
  (charter law 2 amended): `plan.results[name].value` (unswept),
  `plan.results[name][coord]` and `.points` (swept, keyed by coordinate
  values in axis order), `.axes`; `plan.table(name)`; `plan.write(out)`;
  `Plan.load(out)`; `plan.run` (engine, remote, batch_size, versions,
  started, seconds, filled by `execute`). The manifest is
  `plan.model_dump()` with rows dropped and tensors replaced by
  `{file, dims, shape, dtype}`.
- **Engines load in `__init__`**: `NnterpEngine(plan.model)`; the registry
  maps a name to the class and engines register by decorator like ops;
  `clab.run(path, engine=<name or instance>)`, so one loaded model serves
  many documents. `clab.write` is exported beside `run` and `load`.
- **Descriptions are ours now**: the corpus prose is updated to the canonical
  names (`head_patching`'s premix/z sentence corrected to the served layout).
- Verify first in M2 (**belief**): a pydantic plan survives `nnsight.save`
  under `remote="local"` and the stand-in-server test, pickled by class
  reference with nothing by value.

### Tightened the same day

- **One model, `Plan`.** There is no `Document` model: the plan *is* the
  document, resolved. `compiled: bool` says which state it is in (rows loaded,
  references inlined, sweeps expanded); `execute` refuses an uncompiled plan
  and `dump` refuses one without results, each in one line; `repr` shows the
  state. Swept fields are typed `T | Sweep` and `steps` is
  `dict[str, Step | list[Step]]`; after compile no `Sweep` remains.
- **Results are plain values**, no `Result` type: `plan.results[name]` is the
  value for an unswept save, and for a swept save a dict keyed by coordinate
  value (bare for one axis, a tuple in axis order for more). `save` is in
  `plan.saves`; the axes are on the step.
- **The `Plan` API, complete:**
  `Plan(path_or_fields)` — a document file, or the fields themselves;
  `plan.compile() -> Plan`;
  `plan.dump(out)` — a run's output directory (result files + `manifest.json`);
  `Plan.load(out)` — the inverse of `dump`, the filled plan back;
  `plan.write(path)` — the plan itself as a JSON document, for editing in
  Python and saving; drops `results` and `run`, keeps the plan's state;
  `plan.table(name)`; `plan.run` (engine, remote, batch_size, versions,
  started, seconds, set by `execute`). `clab.run(plan_or_path, engine=name
  or instance, ...)` is `Plan(...)` → `compile` → `execute` → `dump`.
- Nothing is added to `Plan` without a stated reason.
- **`split` is a key**, not a path suffix: `{"path": "weekdays/data", "split":
  "train"}`. The `#split` form is gone; the corpus's four uses are migrated in
  M2.
- **An op step holds exactly one op.** Chaining is the next step
  (`"accuracy": {"kind": "op", "op": "mean", "of": "iia"}`). `ops` lists exist
  only inside a read or a write, where they act on the live value within one
  forward. Metrics stay per-example (`match` returns one 0/1 per row and
  position); the number is a `mean` step after it.
- **A result file is the value, nothing else.** A tensor (or a dict/bundle of
  tensors) is written as `.safetensors`; anything else as `.json` (plain
  `json.dumps`, tensors inside it as nested lists). A swept save is a dict
  keyed by coordinate in both formats (`pos=-1` as a safetensors key, `"-1"`
  as a JSON key); a ragged value is its own `{values, row}` shape. One file per
  save, named `<name>.<ext>`, no coordinate directories. Everything that
  describes a value — the step it came from, `dims`, `axes`, `shape`, `dtype`,
  example ids, rows that were `alignment_missing` — is in `manifest.json`
  once; that is how a result links back to its step. The labelled per-row
  view with the mean up front is `plan.table(name)` in memory, not a file.
- **`saves` is a list of names.** The filename is `<name>` and the extension
  follows the type, so a filename in the document carries no information. The
  corpus's dict form and its paths (`compare/patching/iia.json`) are migrated
  in M2; `das_apply.json`'s `file_path` becomes
  `documents/artifacts/fit.rot.safetensors`.
- **No `metric` step kind.** Per-example scoring is a property an op declares
  (it keeps the rows axis); `FORMAT.md` groups ops as metrics, reductions and
  edits so the word is there for readers.
- **Generated text is a `decode` op**, not a run attribute: a generate step's
  value is its ids; `{"kind": "op", "op": "decode", "of": "answer"}` gives one
  string per row (stopped at the row's first EOS; `"tokens": true` for the
  token strings), saved by name like any value. `decode` takes the engine,
  like `readout`. There is no `run.generated`.
- **The manifest carries descriptive metadata only**: how each input
  tokenized and which token (or decode step) every positioned read and write
  resolved to per row — the coordinates of a value, filled in preflight and
  the run, used by `plan.table` to label `pos_token`, `key_token`,
  `step_token` and `alignment_missing`. Values are never recorded there.
- **Every value with a rows axis knows its dataset**: compile annotates each
  step with the dataset its rows are aligned to (a forward's or generate's
  `input`; an op inherits it through `of`/`with`; two datasets in one op are
  refused). `plan.table(name)` joins on that annotation.
- **`plan.table(*names)`** joins several saved values wide on their shared
  axes (`at`, `row`, `pos`), one column per name, plus the dataset columns
  the step used (`columns=` to choose). A name not in `saves` is refused:
  "`answer` was not saved; add it to saves". A value with no rows axis tables
  one dict per element with its axis indices.
- **A dataset may be a pinned HuggingFace spec** (`{"hf": "org/name",
  "revision": ..., "split": ..., "take": N}`; an unpinned one is refused):
  compile keeps the spec; rows materialize wherever `plan.data[name].rows` is
  first touched — in preflight inside the session on a run, on the client
  during analysis — from the same spec. The manifest records each dataset's
  `n_rows` and a hash of its example ids so a re-materialization mismatch is
  an error. Naming a dataset in `saves` brings its rows home. M2 declares the
  field; loading from the hub lands when a document needs it.

### Ratified after M2 (2026-10-14)

- **`model` knows three keys** — `key`, `revision`, `dtype` (spelled torch's
  way: `float32`, `bfloat16`; the document and the plan use the same
  spelling) — and every other key is passed to the engine's loader untouched
  (`attn_implementation`, `device_map`, ...). An engine's loader refuses what
  it cannot honor. `schema.json` allows additional properties on `model`.
- **`generate` owns `input`, `writes`, `reads`, `interventions`**; every other
  key is forwarded to the engine's `generate` as-is (`max_new_tokens`,
  `do_sample`, `temperature`, ...), sweepable like any value. The generated
  frame's bound is what the engine actually generated: a read at a step a row
  never reached is `alignment_missing`.
- **One narrowing primitive, `select`**: `{"select": {<axis>: [indices]}}` on a
  site or on an op in a featurizer's space, keyed by the axis names every
  component publishes (`num_heads`, `intermediate_size`, `hidden_size`,
  `key`, `features`, ...). `heads`, `units` and `features` are gone; compile
  checks the axis exists on that component. `layers` stays (an address, not
  an axis: int, list, or `"all"`), and so does `pos`. Six corpus documents
  migrate (head_patching, neuron_patching, sae_feature_ablation, the three
  attention-interior ones).
- **Op groups stay** as the one-word label on `@op` that `describe` and
  `FORMAT.md` print under; nothing branches on it.
- **CLI gains `clab show <out>`** (each result with its step, dims, axes,
  shape; the run's engine, batch size, versions) **and `clab table <out>
  <name>... [--csv]`** (`plan.table(...)` printed). Both are `Plan.load` plus
  one method.
- `table` omits `at` on an unswept value (stated in FORMAT.md).

## 12. Roadmap (deferred, not scheduled)

Raised and agreed as worth doing; not assigned to a milestone. Bring them up
when the owner asks for open items.

- **Provenance in `manifest.json`** — the source document's `sha256`; a
  `sha256` per loaded bundle; `device`, CPU thread count, GPU name/driver and
  deterministic-algorithms flag; a run `seed` set at session start
  (`execute(seed=0)`) for sampled generation; the command line; on remote
  runs the server's versions and job id (M5). And **`clab rerun <out>
  [--check]`**: a dump is runnable as-is, and `--check` compares every result
  to the original bit-for-bit, naming any difference and the manifest fields
  that differ.
- **Hub datasets** (`hf` spec; rows materialize where first touched; naming a
  dataset in `saves`).
- **`fit` eval as named substitutions** (`fit.held_out.iia`, `fit.train.iia`)
  to remove the root repetition in das/dbm — after M4 has run one fit.
- **Probes** via the `linear` featurizer and label-vocabulary metrics (M4+).
- **Named sweeps** to zip two different fields, if a document ever needs it.
- Readability: the `cx` dict threaded through `compiler.py`;
  `output.elements`.

### Ratified after M3 (2026-10-14)

- **`pos` is three stages of narrowing: frame → scope → index.** Each stage
  narrows the one before; the same selector shape serves frame and index.

  ```
  pos := int                                  # shorthand: {"frame": 0, "index": int}
       | {"frame"?: frame, "scope"?: scope, "index"?: index}
  frame := int | [start, stop] | "generated" | "all" | "eos"      # default 0
  scope := {"column": "<dataset>.<column>"} | {"text": "<literal>"} | {"turn": "<role>"}   # each with occurrence; column/text with strip
  index := int | [start, stop] | "all"                            # default "all"
  ```

  **frame** picks forward calls: 0 is the prefill (the only call a `forward`
  step has), k ≥ 1 is decode call k, negative counts from the row's last
  call; a span is half-open; `"generated"` is every decode call; `"all"` is
  the whole timeline; `"eos"` is the call where the row stopped. Under the
  cache a decode call holds one position — generated token k−1, its input —
  and its logits are the distribution over token k; the prefill holds every
  prompt position and its last one is the distribution over generated token
  0. Positions mean the same thing in a forward and in a generation: where
  the input token sits.
  **scope** picks text within the selected calls' tokens: the row's own text
  from a column (named with the one dataset.column syntax — an indexed
  input's variables are named explicitly, e.g.
  `pairs.counterfactual_inputs_variables[0].entity`; the hidden
  `<column>_variables` convention is gone), a literal text (same mechanism),
  or a chat turn (call 0 only). Default: everything the frame selected.
  **index** picks positions within that: an int (negative from the end), a
  half-open span, or `"all"`. Default `"all"`.
  Rectangular vs ragged is decided from the record alone (one call or a span
  of calls with a fixed index is rectangular; `"all"`, `"generated"`,
  `"eos"` and text scopes are per-row). In the prefill a text that is absent
  or doubled is refused by name (the data decided); in the generated calls
  it is a result (`alignment_missing` / `alignment_ambiguous`), as is a call
  the row never reached. Any value may be a sweep.
- **Writes work in every frame.** A write at decode call k edits that call's
  position and takes effect from token k on through the cache; a span or
  `"generated"` applies it at each call independently; the operand is
  `[rows, 1, ...]` or a broadcast vector; a row that already stopped is
  skipped there and recorded as missing. Both engines agree to the bit.
  A write beyond the prefill may take a `column`/`text` scope: it is
  resolved online, call by call, from the growing continuation, and lands
  at the call fed the first occurrence's last token (`index: -1` is the
  only index a scoped generated write can have: a word is known only once
  its last token has been fed); a later second
  occurrence is reported as `alignment_ambiguous`, not re-applied; a row
  that never says the text is recorded missing. Refused for writes, because
  unknown until the row ends: a negative index over several calls, `"eos"`,
  and negative frames.
- **Text scopes take `occurrence` and `strip`.** `occurrence` is the same
  selector shape as `frame` and `index` (`int | [start, stop] | "all"`),
  default `0`: the scope stage locates every occurrence of the text in the
  selected calls' tokens, picks those, and `index` narrows within them. The
  concept of ambiguity is gone (`alignment_ambiguous` no longer exists; a
  doubled text takes occurrence 0). In the prefill an occurrence the row
  lacks is refused by name; in generated text occurrence k lands at the call
  fed the k-th occurrence's last token, `"all"` at each as it appears, and a
  row with too few occurrences is recorded missing; a negative occurrence is
  for reads only. `strip: true` strips the text's surrounding whitespace
  before locating it (default `false`). `turn` takes `occurrence` but not `strip`:
  `{"turn": "user", "occurrence": 1}` names the role, and the role's
  messages in order are its occurrences (`0` the first, `"all"` every one,
  ragged; a role the row lacks is refused by name). The `role[n]` syntax is
  gone; `chat_turn.json` migrates to `{"turn": "user", "occurrence": 1}`.
  `column` and `text` may be a list: every occurrence of every listed text is located,
  ordered by position, and `occurrence` picks among them by the same rule
  (default 0 = the first of any; `"all"` = the union of their tokens where
  occurrences overlap).
- `variable`, `segment`, `span`, `all` (as keys), `prompt` and the old
  "generated index k = the call that chose token k" convention are gone.
  Migration round four touches every document with a dict `pos` and the
  two generate documents (`generate_probe`'s three distributions become
  `{"frame": {"sweep": [0, 1, 2]}, "index": -1}`).
- **Milestone order from here:** M4 = positions (this grammar, migration
  round four, generated-frame writes); M5 = featurizers and `fit`; M6 =
  real remote.
- **One resolution rule for reads and writes.** A scoped generated write
  lands at a decode call iff, among the occurrences the selector picks on
  the tokens fed so far — counted only within the frame's calls, exactly as
  a read counts them — some picked occurrence's last token is that call's
  input. A read and a write with the same record always agree; each picked
  occurrence lands at its own last token.
