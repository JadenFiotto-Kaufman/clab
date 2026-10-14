# clab — charter

You are building, from scratch, a small library that runs interpretability
experiments on transformer language models from declarative JSON documents.
You have: this charter, `PLAN.md` (the owner's ratified plan: decisions,
the reading of the corpus, the expected architecture and the milestones), the
documents, their datasets, an environment, a cheat-sheet for the model-access
library you will build on, and the owner's instructions. Nothing else exists. Precedence: the owner's instructions win
over this charter wherever they disagree; the charter wins over your own
judgment; where both are silent, design it yourself.

## What it does

`documents/` holds the corpus of JSON documents (31 at the start; it grows by
example documents the owner adds). Each one declares a complete experiment:
which model (pinned checkpoint and revision), which data (`data/`, referenced
by name), what to run (forwards, generation, interventions on the model's
internals, metrics, optimization of small trainable modules), and what to
save. Your library reads a document, runs it end to end, and writes the
results to disk in a form a person can read and a script can consume.

**The format is general-purpose by intent; the 31 documents are its fixed
corpus.** The goal is a JSON format expressive enough to specify the vast
majority of interpretability experiments without becoming verbose — leave off
any flexibility that would cost real complexity. Formalize and generalize the
format however you judge best. The corpus is migrated once, in M1, to the
format's canonical spelling (PLAN.md §10 records it); from then on it is
fixed and every document runs unchanged. The format has exactly one spelling
for each thing — no aliases, no legacy forms. Where a document's meaning is ambiguous, you decide what
it means, implement that consistently, and write your reading down; your
implementation is the semantics. Study the corpus carefully before designing
— the documents reference each other's ideas (shared datasets, recurring
field shapes) and the format is smaller than it first looks.

## Laws

These are correctness constraints, learned at cost. They are not up for
redesign.

1. **One request, one session.** Running a document opens exactly one
   `model.session(...)` and everything happens inside it. The session's
   `remote=` flag is the ONLY difference between running locally and running
   on a remote nnsight server — there is no second code path, no
   remote-specific branch in what gets executed.
2. **What you compile a document into is pure data.** Strings, numbers,
   tuples, dicts — plus result slots that fill in when it runs. It has no
   methods that execute anything. It may be a typed model (pydantic) whose
   methods are views and serialization — never execution. A compiled document
   must pickle, print, and mean the same thing wherever it lands.
3. **Inside a session, the engine and its model are the only client-side
   objects.** Code that runs inside the session may reference the engine
   instance (`engine`, `engine.model`, `engine.forward`) and the plan; it
   may not close over anything else client-side: no loaded datasets, no
   separately-constructed tokenizer, no open files, no helper objects. The
   engine holds nothing but its model and its configuration. Everything else
   is module-level functions over plain data. (nnsight ships every name a
   block closes over, by value, to wherever the block runs.) The model's own
   tokenizer is reached through the model inside the session.
4. **The client never decides anything from a tensor.** If a decision depends
   on a value the model produces (which token position matches a piece of
   text, which rows qualify, a dynamic shape), the document's specification of
   that decision travels into the run and resolves where the model is.
5. **Nothing of your package ships by value.** Remote execution works because
   the server has the same package installed at the same version; code
   resolves by import. Never register your package (or nnterp) for by-value
   shipping.

## Structure the owner has fixed

These are the owner's decisions, not yours to relitigate (details and
anything unstated are yours):

- A document's root declares the model environment (checkpoint, revision,
  dtype, the kwargs that matter) and may declare named, reusable definitions
  — sites/locations, interventions — that steps reference by name; the same
  things may also be written inline in a step where they are used once.
- The format is engine-agnostic: any engine built to run interp workflows
  could ingest one of these documents. clab ships two — nnterp (primary) and
  plain-pytorch hooks (the extensibility proof) — and both conform to a small
  protocol: `engine.execute(plan, **kwargs)` as the single entry point, with
  `.forward`, `.generate` and the like taking intervention data.
- A shared, reusable `steps.py` walks a document's steps (a loop with a
  dispatch on step type: forward, generate, train, ...) assuming only that
  protocol. Both shipped engines use it; a future engine may ignore it
  entirely and handle the step types itself — keep it possible.
- Sibling steps share state keyed by the names of things, so a later step can
  reference an earlier step's values without saving them to disk. That state
  is pure data and tensors (laws 2 and 3 apply to it; it must survive a
  remote session).
- Generic logic for loading data, batching, and targeting token positions
  lives outside the engines; engines expose the primitives it needs to stay
  generic.
- A root `address.py` is the standard vocabulary of the locations clab
  allows.
- The result of a run is the filled plan: every step's result slots filled
  in place, swept steps as lists of points. `nnsight.save(plan)` at the end
  of the session is the only thing that comes home. `saves` names which
  values are kept in the plan; the rest are ephemeral cross-step state.
- The output of a run is a directory: each saved value at the path its
  document gives, and a manifest at the root that mirrors the plan.

## A hint, not a law

Different runtimes reach the same named place differently — different module
paths, different tensor layouts (one serves `[batch, seq, hidden]`, another
flat token rows), different instrumentation. You will save yourself pain if
what a name *means* and how a runtime *reaches it* do not live in the same
place. How you draw that line is yours.

## The runtimes

- **Primary: nnterp** (`NNTERP.md` is your cheat-sheet; its own docs live in
  the checkout it names). It standardizes transformer internals — one envoy
  tree and one set of value names across every architecture, with
  availability, forward order, layouts and sizes answered by its API. Build
  on it; do not hand-roll per-architecture model knowledge that nnterp
  already serves.
- **Second, required: plain HuggingFace.** An `AutoModelForCausalLM` driven by
  forward hooks, importing nothing from nnsight. It must run at least the
  simplest intervention documents with results numerically identical to the
  primary runtime's, and refuse clearly whatever it cannot serve. Its purpose
  is to prove your seams: if adding it requires touching your core, the core
  is wrong.
- **Remote is a design constraint from day one** (laws 1, 3, 5 exist for it),
  but actually running against a server is not a milestone yet.

## Acceptance

- All 31 documents run end to end on their pinned checkpoints, on CPU, from a
  CLI entry point.
- A test suite that makes the semantics you chose checkable: it should fail
  if a future change silently alters what a document means. Self-consistency
  is the bar — there are no reference numbers.
- Both runtimes agree numerically where both serve a document.
- `uvx pyright` clean; the suite green via
  `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`.
- Readable. A person should be able to learn how the system works by reading
  it. Prefer a plain function to a class, a string to an enum, a dict to a
  registry; no indirection you cannot justify in one sentence; every line
  earns itself.

## Process

- Work in this repo (`~/wd/clab`, its own git repo — never run git from the
  home directory). Build in milestones, one PR per milestone for the owner to
  review before the next begins. You propose the milestone cut in your first
  PR's description; something like "one intervention document end to end" is
  the right size for the first.
- Commit messages are one plain declarative sentence.
- Design and code are done by agents briefed from this charter and
  `PLAN.md`. An agent proposes format changes in its PR description with
  before/after JSON; none lands without the owner's ratification.
- `pyproject.toml` is scaffolded (editable nnsight/nnterp checkouts and one
  version override — leave those alone; add your own dependencies as needed).
  Tests run offline: the checkpoints the documents pin are already cached
  (`HF_HUB_OFFLINE=1` works).

## Clean-room rule

This is a from-scratch build and must stay one. Do NOT read, search, or
consult: `~/wd/causalab*` (any directory so named), any GitHub repository or
PR/issue thread matching `causalab`, or `GAPS.md` in the nnterp checkout.
The nnterp and nnsight checkouts themselves (code, docs, tests) are your
substrate and are fully allowed. If you land in a file that mentions a system
resembling this one, stop reading it.
