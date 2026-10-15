# clab — design (M0)

**Amended by PLAN.md §10 and §11 (through "Ratified after M4", whose last two entries are M5: one
`featurizer` op with per-kind modes, and LoRA as a featurizer kind), which win wherever they and this document disagree.** `FORMAT.md` and
`schema.json` (generated from the code) are the user-facing reference; the code now differs from this
document in these places:

- §1: the plan is one pydantic model, `Plan` (`clab/plan.py`), which is also the document: `Plan(path)` reads
  one, `compile()` returns it compiled (`compiled: true`). The records (sites, reads and writes, ops, `pos`,
  steps, datasets, featurizers) are the same models in both states; compile fills a few derived fields (a
  step's `at`, `dataset` and `dims`, a read's `dims`, a dataset's `sha256`). Sites and named interventions
  are inlined into the steps; featurizers stay at the root, by name. Steps carry no result slots: results
  are `plan.results`, one dict at the root keyed by the names in `saves`, a swept one a dict keyed by
  coordinate. `saves` is a root key and a list of names. A forward's `input` stays the string
  `"<dataset>.<column>[i]"`; `pos` stays as written and becomes a slice in the run.
- §2: `plan.py` holds the models and `find_sweeps`; `compiler.py` is the transform (sweeps, inlining, names,
  layouts, datasets, bundles); `output.py` is `dump`, `load` and `table`; `registry.py` checks a document's
  arguments against a function's signature; an op declares its group and, for `pca`, the featurizer its
  bundle is for in `@op`.
- §3: an engine loads its checkpoint in `__init__` from the plan's `model` block and registers by decorator
  (`@engine("nnterp")`); `execute` refuses an uncompiled plan and a plan for another checkpoint; an engine
  says what it cannot serve (`unserved()`, nnterp from its `support()`), and preflight refuses those
  components by name, inside the session, before any forward. `readout` is nnterp's `project_on_vocab` and the hooks engine's final norm and head.
- §5: positions resolve per row on the unpadded tokens in `steps.preflight`, are recorded in
  `plan.run.positions`, and are offset by each row's padding per batch. Scopes use the tokenizer's offset
  mapping; a chat is rendered through the template and its turns located by their content.
- §6: the cross-step state is as described; what `saves` keeps goes into `plan.results` when the run ends.
- §7: a swept result is keyed by the axis's value (a list value as a tuple; several axes as a tuple in axis
  order), not stored per point directory. A `layers: "all"` read runs its ops per layer before stacking, so it
  equals the layer sweep to the bit.
- §8: there are no coordinate directories and no table files: one file per save, `<name>.safetensors` or
  `<name>.json`, a swept one keyed by coordinate inside the file. `manifest.json` is the plan's own dump
  without its file-read rows, each result replaced by `{file, step, dims, axes, shape, dtype}`.
- §10: verified in M2 that a pydantic `Plan` survives `nnsight.save(plan)` under `remote="local"` and the
  stand-in server's `torch.save`/`torch.load` round trip, pickled by class reference with nothing of clab or
  nnterp by value; the stand-in test asserts it.
- §11 proposal 3: the inline `interventions: {writes, reads}` spelling is gone (writes and reads sit on the
  step).
- M3, §3: `generate`'s options are not three: the step forwards every key it does not own to the engine's
  `generate` as it is, which validates them when the step runs (clab refuses, before anything runs, only
  `num_return_sequences`, `num_beams`, `return_dict_in_generate` and `use_cache` at other than their
  defaults). A generated-frame read's value comes back as a list, one `[rows, 1, ...]` per step read. The
  hooks engine's bit-equality with the nnterp engine's generation is now **verified** (both generate
  documents, and rows that stop at different steps, at batch sizes 1, 2 and 4). EOS is the id the
  generation stops on (the step's `eos_token_id`, else the generation config's), not the tokenizer's.
- M3, §5: the multi-byte hazard of `char_spans` is now **tested**: a character split across byte tokens is
  the span of the token that completes it, and the tokens before it belong to the same character. A ragged
  read carries `tok` (its token index, or in the generated frame its step), not only `row`. A rectangular
  generated-frame read holds NaN on a row that did not reach its step, which is missing all the way down:
  the manifest records `alignment_missing`, metrics give NaN, `argmax`/`top_k` give id -1, and reductions
  skip the row. `select` replaces `heads`/`units`/`features`; on `key` it counts the row's own tokens.
- §5 (PLAN.md §11 "Ratified after M3"): `pos` is `frame` (forward calls: 0 the prefill, k a decode call,
  `generated`/`all`/`eos`) → `scope` (`column`/`text`/`turn`) → `index`; a position is where an input token
  sits (decode call k holds generated token k-1), the "generated index k = the call that chose token k"
  convention, `variable`, `segment` and the `span`/`all` keys are gone, and writes land in every frame; a
  write beyond the prefill may take a `column`/`text` scope, found online call by call, landing at the call
  fed the last token of each occurrence it picks (index -1). A `column`/`text` scope picks its occurrences
  with `occurrence` (default 0, the first; the frame/index selector) and may `strip` its text; ambiguity is
  gone: a doubled text takes the occurrence asked for, and `alignment_ambiguous` no longer exists. `column`
  and `text` may be lists, whose occurrences are taken together, ordered by where each ends.
- §5: `turn` is a role (`{"turn": "user", "occurrence": 1}` the second user message); its messages are its
  occurrences, and the `role[n]` spelling is gone.
- M4, §9: a declared featurizer is made before the run's first step (each point of it), a trained one at its
  fit's start; ops take the instance. A subspace's bundle holds its `[width, k]` basis, which a trained one
  computes from `origin` and `theta` once per optimizer step; every kind but `pca` and `sae` may be drawn or
  loaded (`gate` and `linear` take `file_path`; a probe's header records its class list), and a loaded one is
  fixed. Every bundle's header adds `component`, `layer`, `width` and `document` (the writer's name); compile
  checks the file holds exactly its kind's tensors (a swept fit's file, one bundle per point, is refused),
  the site at every use, and the plan keeps the header (`Featurizer.header`) so preflight can check `width`
  against the model. Each kind is one `@kind` function and one function per operation it supports. A fit's body runs on its own view of the plan: preflight prepares its
  inputs and positions over the training rows and, apart, over the eval substitution (each substituted
  dataset's rows under the body's name for it); a minibatch is that view narrowed to some rows. The body's
  forwards run in batches of the fit's `batch_size`. The run walks without gradients; a fit turns them on for
  its training passes only. A body uses no root value with a row per example (it would not line up with a
  minibatch), its datasets have as many rows as each other, and it holds no fit. The `linear` probe's scores
  are on a `classes` axis, which metrics score as they score `vocab_size`, a column resolved against the
  probe's class list, which preflight checks every scored column against over the training and the eval rows;
  a probe is never written through. `table` refuses to join values with rows of two datasets. A fit's `eval` names its runs after training (PLAN §11 "Ratified after M4"):
  each is a body view and its own prepared rows, and a member `<fit>.<eval>.<member>` tables over that eval's
  rows. A drawn `sae` (`latents`) trains; its scalars read what the body's `encode` reads of it saw in that run
  of the body, which the instance keeps (`seen`) only while its fit trains it; its decoder rows are projected
  back to unit norm after each optimizer step. The hooks engine serves `fit`: gradients flow
  through its hooks, and both engines agree to the bit (**verified**, every fit document).

For ratification before any code is written. Present tense throughout: this is
what the code will do. Every claim is either **verified** (with how, against the
environment in `pyproject.toml`: nnsight `dev` at d565f5f8, nnterp
`0.8-refactor`, transformers 5.19.0, Python 3.13, CPU) or marked **belief**.
The scratch scripts that verified things are not part of the repo; their
outcomes are quoted where they matter.

One sentence of the whole design: **`compile` turns a document into a plan of
plain dicts; inside one session, `steps.run` walks that plan, asks the engine
for nothing but forwards, generations, a readout and its tokenizer, and fills
the plan's result slots in place; `nnsight.save(plan)` brings it home, and
`output.write` lays it out as a directory.**

Two rules split the work and recur below:

- **compile knows the document and the files; the run knows the model.**
  Anything decided from the document, the data files and the bundle headers is
  checked by `compile`, on the client. Anything that needs the tokenizer or a
  size of the model is checked by `steps.preflight`, inside the session,
  before the first forward.
- **What a name means lives in `address.py`; how a runtime reaches it lives in
  that engine's module.**

---

## 1. The plan's data shape

`compile(document, root) -> plan` returns nested `dict` / `list` / `str` /
`int` / `float` / `bool` / `None`. Before it runs, a plan holds tensors in
exactly one place: the tensors of a bundle a document names by `file_path`
(§9). There are no classes
in a plan, no methods, no enums. `pickle.dumps(plan)` round-trips and
`pprint(plan)` is readable (it gets a test in M1).

The plan's root:

| key | what it holds |
|---|---|
| `format` | `"clab/1"`: the one format version clab writes; the manifest and every bundle header carry the same string, so a reader checks one thing |
| `document` | `{"name", "path", "sha256"}` of the source file |
| `model` | `{"key", "revision", "dtype", "attn_implementation"}`, with `dtype` spelled as torch spells it (`"float32"`) |
| `data` | each dataset by name: `{"path", "split", "columns", "rows"}`. The rows are in the plan because nothing else may reach the session (law 3); they keep only the columns the document uses: every step's field column, every column a metric argument names, every column a variable scope names, the `<column>_variables` column of an indexed field when a variable scope is used on that input (`entity_patch.json` needs `counterfactual_inputs_variables`, which no step names), plus `example_id` when present |
| `featurizers` | each declared featurizer's spec, with its `bundle` when it names a file (§9) |
| `sweeps` | each sweep axis by its label: `{"path", "values"}` (§7) |
| `steps` | each step by name, in document order. An entry is a step dict, or, when the step varies over a sweep, a **list of points** (§7) |

`saves` does not survive as its own key: each save becomes a **result slot** on
the step that produces the value.

**A result slot** is the dict `{"save": "<filename>", "value": None}` at
`step["result"]["<state name>"]`. It is the only part of a plan that changes
when it runs. `steps.run` fills it in place when its step (or point) finishes:
`slot["value"] = finished(state[name])`, where `finished` detaches, moves to
CPU and **clones** every tensor. The clone matters: a read such as
`lm_head.output[:, -1]` is a view of the whole `[rows, seq, vocab]` buffer, and
`torch.save` ships the whole storage. Verified: the same 4-row plan came home
as 5,635,429 bytes with view slots and 515,365 bytes with cloned ones.

### `documents/patching.json`, compiled

```python
{
  "format": "clab/1",
  "document": {"name": "patching", "path": "documents/patching.json",
               "sha256": "0fc9bf26d0d53b2e…"},
  "model": {"key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
            "dtype": "float32", "attn_implementation": None},
  "data": {
    "pairs": {
      "path": "weekdays/train", "split": None,
      "columns": ["input", "counterfactual_inputs", "cf_answer", "base_answer"],
      "rows": [
        {"input": "If today is Thursday, tomorrow is",
         "counterfactual_inputs": ["If today is Saturday, tomorrow is"],
         "cf_answer": " Sunday", "base_answer": " Friday"},
        {"input": "If today is Friday, tomorrow is",
         "counterfactual_inputs": ["If today is Sunday, tomorrow is"],
         "cf_answer": " Monday", "base_answer": " Saturday"},
        {"input": "If today is Saturday, tomorrow is",
         "counterfactual_inputs": ["If today is Thursday, tomorrow is"],
         "cf_answer": " Friday", "base_answer": " Sunday"},
        {"input": "If today is Sunday, tomorrow is",
         "counterfactual_inputs": ["If today is Friday, tomorrow is"],
         "cf_answer": " Saturday", "base_answer": " Monday"},
      ],
    },
  },
  "featurizers": {},
  "sweeps": {},
  "steps": {
    "counterfactual": {
      "kind": "forward", "at": {}, "data": "pairs",
      "input": {"column": "counterfactual_inputs", "index": 0, "variables": None},
      "writes": [],
      "reads": [
        {"name": "v_cf", "component": "block_output", "layer": 0, "select": {},
         "pos": {"frame": "prompt", "scope": None, "start": -1, "stop": None},
         "featurizer": None, "view": None,
         "dims": ["rows", "pos", "hidden_size"]},
      ],
      "result": {},
    },
    "patched": {
      "kind": "forward", "at": {}, "data": "pairs",
      "input": {"column": "input", "index": None, "variables": None},
      "writes": [
        {"name": "patch", "component": "block_output", "layer": 0, "select": {},
         "pos": {"frame": "prompt", "scope": None, "start": -1, "stop": None},
         "mechanism": "swap", "operand": "counterfactual.v_cf", "params": {},
         "featurizer": None, "features": None},
      ],
      "reads": [
        {"name": "logits", "component": "lm_head", "layer": None, "select": {},
         "pos": {"frame": "prompt", "scope": None, "start": -1, "stop": None},
         "featurizer": None, "view": None,
         "dims": ["rows", "pos", "vocab_size"]},
      ],
      "result": {},
    },
    "iia": {
      "kind": "metric", "at": {}, "metric": "match", "of": "patched.logits",
      "args": {"expected": "pairs.cf_answer"}, "token_form": "spaced",
      "dims": ["rows", "pos"],
      "result": {"iia": {"save": "iia.json", "value": None}},
    },
    "logit_diff": {
      "kind": "metric", "at": {}, "metric": "logit_diff", "of": "patched.logits",
      "args": {"a": "pairs.cf_answer", "b": "pairs.base_answer"}, "token_form": "spaced",
      "dims": ["rows", "pos"],
      "result": {"logit_diff": {"save": "logit_diff.json", "value": None}},
    },
  },
}
```

What compile did, in the order it does it:

- read `data/weekdays/train.json`, applied no split (no `#`), kept the four
  columns some step names;
- inlined the site `target` and the inline site `{"component": "lm_head"}`
  into records; a site is never referenced by name after compile;
- normalized `field: "counterfactual_inputs[0]"` to `{"column", "index",
  "variables"}`. `variables` says where a `variable` scope finds the row's texts
  for this input: `None` when no record of the step has a variable scope (as
  here), else `{"column": "<column>_variables", "index": i}` for an indexed
  field and `{"column": None, "index": None}` (the row itself) for a plain one;
- kept every narrowing in the document's own spelling, `"select": {"heads": [0]}`
  or `{"units": [0, 1, 2, 3]}` (empty here); binding turns it into axis indices
  per component (§3), so the plan says what the document said;
- normalized `pos: -1` to the one position shape (§5): `start: -1, stop: None`
  is "the last token of the scope";
- computed `dims` — axis **names**, never sizes — from `address.py`; a read is
  laid out `(rows, pos, *the component's other axes)` (§4);
- turned `saves` into two result slots, on the steps that make the values;
- gave every step `"at": {}`: the step is not a sweep point (a point's `at`
  names its coordinates, below).

### `documents/pos_sweep.json`, compiled (the parts that differ)

The sweep is on the write's `pos`. `counterfactual` does not depend on it, so
it stays one step and runs once; `patched` and both metrics depend on it, so
each becomes a list of points (§7 has the dependency rule).

```python
"sweeps": {
  "pos": {"path": "steps.patched.interventions.writes.patch.pos", "values": [-1, -2, -3]},
},
"steps": {
  "counterfactual": { …exactly as in patching, "at": {} … },
  "patched": [
    {"at": {"pos": 0},
     "kind": "forward", "data": "pairs", "input": {"column": "input", "index": None, "variables": None},
     "writes": [{"name": "patch", "component": "block_output", "layer": 0, "select": {},
                 "pos": {"frame": "prompt", "scope": None, "start": -1, "stop": None},
                 "mechanism": "swap", "operand": "counterfactual.v_cf", "params": {},
                 "featurizer": None, "features": None}],
     "reads": [ …logits at -1, as in patching… ],
     "result": {}},
    {"at": {"pos": 1},
     … "pos": {"frame": "prompt", "scope": None, "start": -2, "stop": -1} … },
    {"at": {"pos": 2},
     … "pos": {"frame": "prompt", "scope": None, "start": -3, "stop": -2} … },
  ],
  "iia": [
    {"at": {"pos": 0}, "kind": "metric", "metric": "match",
     "of": "patched.logits", "args": {"expected": "pairs.cf_answer"},
     "token_form": "spaced", "dims": ["rows", "pos"],
     "result": {"iia": {"save": "iia.json", "value": None}}},
    … points 1 and 2 …
  ],
  "logit_diff": [ …three points, same shape… ],
}
```

A point is a complete step dict whose `at` gives its coordinate on each axis
it varies over, as an index into `plan["sweeps"][label]["values"]`; the keys of
`at` are the step's axes, so nothing else lists them. Nothing else in a point is
special; the engine never sees `{"sweep": …}` and never sees `at`.

`head_patching.json` shows the other case: its sweep sits on the shared site
`target`, so `counterfactual` (which reads `target`) and `patched` (which writes
it) both become four points on **one** axis, `heads` (path `sites.target.heads`), and point
*i* of each uses head *i*. One sweep object referenced from two places is one
axis, which is what makes "read head *h* of the counterfactual, write it into
head *h*" mean what `attention_pattern_patching.json` needs.

### A reduce, a generate step and a fit, compiled

`mean_ablation.json`'s reduce:

```python
"mean": {"kind": "reduce", "at": {}, "reduce": "mean", "of": "harvest.acts", "k": None,
         "dims": ["hidden_size"],
         "result": {"mean": {"save": "mean.safetensors", "value": None}}},
```

`pca_harvest.json`'s is the same with `"reduce": "pca", "k": 4, "dims":
["hidden_size", "k"]` and its slot a bundle (§9).

`generated_answer.json`'s generate step:

```python
"answer": {
  "kind": "generate", "at": {}, "data": "prompts",
  "input": {"column": "input", "index": None, "variables": {"column": None, "index": None}},
  "options": {"max_new_tokens": 6, "min_new_tokens": 6, "do_sample": False},
  "writes": [],
  "reads": [
    {"name": "at_said", "component": "lm_head", "layer": None, "select": {},
     "pos": {"frame": "generated", "scope": {"variable": "said"}, "start": -1, "stop": None},
     "featurizer": None, "view": None, "dims": ["entries", "pos", "vocab_size"]},
    {"name": "at_stop", "component": "lm_head", "layer": None, "select": {},
     "pos": {"frame": "generated", "scope": {"segment": "eos"}, "start": -1, "stop": None},
     "featurizer": None, "view": None, "dims": ["entries", "pos", "vocab_size"]},
  ],
  "result": {},
},
```

A generated-frame read with a scope is ragged (`entries`, with a `row` vector):
which step it names depends on what each row said, and a row may have none.
One with a plain index (`generate_probe.json`'s swept `index`) is rectangular,
`["rows", "pos", "vocab_size"]`. The generate step's own value, `answer`, is
the new ids, `dims ["rows", "new"]`.

`das.json`'s fit. The body's steps live inside the fit entry, compiled exactly
like root steps; `eval.data` stays the substitution map it is in the document;
the slots for `fit.iia` and `fit.rot` sit on the fit entry, because the fit is
the thing that owns those members:

```python
"data": {"train": {"path": "weekdays/data", "split": "train", "rows": [ …2 rows… ], …},
         "test":  {"path": "weekdays/data", "split": "test",  "rows": [ …2 rows… ], …}},
"featurizers": {"rot": {"kind": "subspace", "k": 8, "parametrization": "cayley",
                        "seed": None, "file_path": None, "bundle": None}},
"steps": {
  "fit": {
    "kind": "fit", "at": {}, "train": ["rot"], "objective": [[1.0, "ce"]],
    "epochs": 10, "batch_size": 16, "seed": 0,
    "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
    "early_stop": {"metric": "iia", "mode": "max", "patience": 3},
    "anneal": {},
    "eval": {"data": {"train": "test"}},
    "steps": {
      "counterfactual": {"kind": "forward", "at": {}, "data": "train",
                         "input": {"column": "counterfactual_inputs", "index": 0, "variables": None},
                         "writes": [],
                         "reads": [{"name": "v_cf", "component": "block_output", "layer": 0, "select": {},
                                    "pos": {"frame": "prompt", "scope": None, "start": -1, "stop": None},
                                    "featurizer": "fit.rot", "view": None,
                                    "dims": ["rows", "pos", "features"]}],
                         "result": {}},
      "patched": { …the `das` intervention inlined: the swap with "featurizer": "fit.rot", the logits read… },
      "iia": {"kind": "metric", "at": {}, "metric": "logit_diff", "of": "patched.logits",
              "args": {"a": "train.cf_answer", "b": "train.base_answer"}, "token_form": "spaced",
              "dims": ["rows", "pos"], "result": {}},
      "ce": {"kind": "metric", "at": {}, "metric": "cross_entropy", "of": "patched.logits",
             "args": {"target": "train.cf_answer"}, "token_form": "spaced",
             "dims": ["rows", "pos"], "result": {}},
    },
    "result": {"fit.iia": {"save": "held_out_iia.json", "value": None},
               "fit.rot": {"save": "rot.safetensors", "value": None}},
  },
  "counterfactual": { …as the body's, at the root, "result": {}… },
  "patched": { … },
  "iia": { …, "result": {"iia": {"save": "iia.json", "value": None}} },
  "ce":  { …, "result": {"ce": {"save": "ce.json", "value": None}} },
},
```

### The filled plan

After `execute`, the plan is the same structure with slot values filled:

```python
# patching
plan["steps"]["iia"]["result"]["iia"]["value"]              # Tensor float32 [4, 1]   (rows, pos)
plan["steps"]["logit_diff"]["result"]["logit_diff"]["value"] # Tensor float32 [4, 1]
# pos_sweep
plan["steps"]["iia"][2]["result"]["iia"]["value"]           # Tensor float32 [4, 1], at pos = -3
```

Other values a slot holds, by what produced them:

| producer | slot value |
|---|---|
| metric (`match`, `logit_diff`, `token_prob`, `token_logit`, `cross_entropy`, `kl`, `js`) | float32 tensor with the metric step's `dims` |
| metric `top_k` | per row and position, a list of `{"token": str, "id": int, "prob": float}`, decoded inside the session |
| a read (`clean.pattern`) | the read tensor, `dims` as compiled (`["rows", "pos", "num_heads", "key"]`) |
| a ragged read (`harvest.acts` with `pos: {"all": true}`) | `{"values": Tensor[N, …], "row": int64 Tensor[N]}` (§5) |
| reduce `mean` | the mean over rows, positions and entries: `[hidden_size]` (the read's feature axes) for a rectangular and a ragged read alike (§3, "operands") |
| reduce `pca` | a bundle: `{"header": {...}, "tensors": {"basis": [d, k], "mean": [d]}}` |
| a generate step's own name (`patched` in generate_probe) | int64 `[rows, max_new_tokens]`: the new tokens only |
| a featurizer (`fit.rot`) | its bundle, `{"header", "tensors"}` (§9) |
| `fit.<metric>` | that body metric under the eval substitution, after training |

A row a read could not place is not a hole in a tensor: ragged values carry
`row`, and the output's tables report the absent rows by name (§5).

---

## 2. Module layout

```
clab/
  compile.py      document -> plan; validation; points()
  data.py         rows at compile time; fields, variables, chat rendering,
                  encoding with left padding, inside the session
  positions.py    pos records -> per-row token indices, inside the session
  address.py      the component vocabulary (meaning, dims, narrowing, writable)
  interventions.py what a read takes and a write does at one site, on any engine
  metrics.py      the eight metrics and the two reduces, over tensors
  featurizers.py  subspace / gate / pca / sae as dicts of tensors; bundles
  steps.py        run(engine, plan, batch_size): preflight, the walk, the fit loop
  output.py       filled plan -> directory + manifest
  cli.py          clab run <doc> [...]
  engines/
    __init__.py   the Engine protocol (typing.Protocol) and ENGINES
    nnterp.py     StandardizedTransformer inside one session
    hooks.py      AutoModelForCausalLM + torch hooks, no session
```

Changes from PLAN §5, one reason each:

- **`document.py` merged into `compile.py`**: validation and normalization walk
  the same tree in the same order, and splitting them would walk it twice.
- **`plan.py` dropped**: the plan has no behaviour; its shape is §1 of this file
  and `compile.py`'s docstring, and the one helper (`points`) lives in
  `compile.py`.
- **`state.py` merged into `steps.py`**: the namespace is a dict and a 15-line
  resolver whose only caller is `steps.run`.
- **`reduces.py` merged into `metrics.py`**: two pure functions dispatched by
  name from the same place do not earn a file.
- **`interventions.py` added**: both engines must do the identical tensor
  arithmetic at a site (selections, mechanisms, featurizer space, composition),
  and putting it in one engine-agnostic module is what makes the two engines
  bit-equal by construction rather than by care.
- **`engines/base.py` folded into `engines/__init__.py`**: a `Protocol` that
  pyright checks both engines against, and the `ENGINES` dict the CLI picks
  from; there is no base class to inherit.
- **`resolve(address)` dropped from the protocol**: only the engine itself
  reaches a component, so it is a private function of each engine module, not
  something `steps.py` calls.

### Public functions

`compile.py`

| signature | contract |
|---|---|
| `load(path: str) -> dict` | The document as JSON; nothing interpreted. |
| `compile(document: dict, path: str) -> dict` | The plan (§1). Reads `data/` and bundle files relative to the corpus root (the parent of the document's directory). Raises `CompileError` naming the document path of the offending value. |
| `points(entry: dict \| list) -> list[dict]` | A plan entry as its list of points; `[entry]` when it is not swept. |
| `class CompileError(ValueError)` | |

`data.py`

| signature | contract |
|---|---|
| `load_rows(root: str, path: str) -> list[dict]` | `data/<path>.json`, filtered by `split` when the path has `#<split>`. Compile time. |
| `field(row: dict, input: dict) -> str \| list[dict]` | The row's input text, or its chat messages. |
| `variables(row: dict, input: dict) -> dict[str, str]` | The texts a `variable` scope may name for this input: `row[<column>_variables][index]` when the field is indexed, else the row itself (§5). |
| `encode(tokenizer, inputs: list, width: int) -> dict` | Inside the session. Renders chats through the tokenizer's template, tokenizes each row with offsets, marks a token special when its id is in `tokenizer.all_special_ids`, left-pads by hand to `width` (the dataset input's longest row, §3). Returns `{"input_ids", "attention_mask", "position_ids", "offsets", "special", "turns", "lengths"}`; `position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`, which both engines pass to the model. |

`positions.py`

| signature | contract |
|---|---|
| `resolve(pos: dict, batch: dict, row_vars: list[dict]) -> dict` | Inside the session. The record's per-row token indices into the padded batch: `{"rows": Tensor[R, w]}` when rectangular, `{"row": Tensor[N], "tok": Tensor[N]}` when ragged; rows it cannot place listed in `"missing"`. |
| `continuation(pos: dict, new_ids: Tensor, tokenizer, row_vars: list[dict]) -> dict` | For the `generated` frame: per row, the decode step(s) the spec names, or missing (§5). |
| `char_spans(tokenizer, ids: list[int]) -> list[tuple[int, int]]` | Character span of each token by decoding growing prefixes; the alignment for generated text (and, later, for slow tokenizers). |

`address.py`

| signature | contract |
|---|---|
| `COMPONENTS: dict[str, dict]` | One entry per component name (§4): `meaning`, `dims`, `token_axis`, `narrow` (narrowing key -> axis), `per_layer`, `writable`, `stream` (may take `view: logits`). |
| `dims(component: str, select: dict, featurizer: str \| None, layers: str \| int \| None) -> list[str]` | The axis names a read of this component presents. Used by compile. |

`interventions.py`

| signature | contract |
|---|---|
| `gather(value: Tensor, component: str, read: dict) -> Tensor \| dict` | The read's selection of `value`, laid out `(rows, pos, …)`, through its featurizer when it has one. Pure torch. |
| `edit(value: Tensor, component: str, writes: list[dict]) -> Tensor` | `value` with `writes` applied in listed order, out of place; the original is what `renormalize` restores to. Pure torch. |
| `MECHANISMS: dict[str, Callable]` | `swap`, `add_scaled`, `renormalize`. |

`metrics.py`

| signature | contract |
|---|---|
| `METRICS: dict[str, Callable]` | Each `(logits, args) -> Tensor \| dict`. `logits` is `(…, pos, vocab_size)` with any leading axes (`rows`; `layers, rows`; or ragged `entries`); `args` holds tensors already aligned to those leading axes: token ids per row (indexed by `row` for a ragged value, broadcast over `layers`), or the other logits read (`kl`, `js`). Every metric returns a tensor over the leading axes and `pos`, except `top_k`, which returns `{"ids": Tensor[…, k], "probs": Tensor[…, k]}`. Metrics never see a tokenizer. |
| `token_ids(tokenizer, texts: list[str], form: str) -> Tensor` | Inside the session. One id per row; a text that is not one token is refused naming the row and the column. `steps.py` calls it before a metric, and decodes `top_k`'s ids with `tokenizer.convert_ids_to_tokens` after one, so the slot holds strings. |
| `mean(value) -> Tensor`, `pca(value, k: int) -> dict` | The two reduces. |

`featurizers.py`

| signature | contract |
|---|---|
| `make(spec: dict, width: int, seed: int) -> dict` | A featurizer instance, built where the run is: drawn from its seed, or from the bundle the plan carries. |
| `encode(f: dict, h: Tensor) -> Tensor` | `h` in the featurizer's space. |
| `replace(f: dict, h: Tensor, new: Tensor, features: list[int] \| None) -> Tensor` | `h` with its featurizer-space coordinates replaced by `new` (on `features` only, when given), written as a delta so a no-op is bit-exact. |
| `parameters(f: dict) -> list[Tensor]`, `scalars(f: dict) -> dict[str, Tensor]` | What a fit optimizes; a featurizer's own scalars (`mask`). |
| `to_bundle(f: dict, header: dict) -> dict`, `check(header: dict, expected: dict, name: str) -> None` | Bundle out; header vs document, refusing by key. |

`steps.py`

| signature | contract |
|---|---|
| `run(engine, plan: dict, batch_size: int) -> None` | Preflight, then every step and point in order; fills every result slot in place. Assumes only the protocol (§3). |
| `preflight(engine, plan: dict) -> None` | The model-dependent checks, before any forward (§5, §9). |

`output.py`

| signature | contract |
|---|---|
| `write(plan: dict, out: str, run: dict) -> dict` | Writes each filled slot to its file and `manifest.json`; returns the manifest. Any engine's filled plan. |

`cli.py`

| signature | contract |
|---|---|
| `main(argv: list[str] \| None = None) -> int` | `clab run DOC [--engine nnterp\|hooks] [--out DIR] [--remote no\|local\|yes] [--batch-size N]`; default `--out` is `out/<document name>`. |

`engines/`

| signature | contract |
|---|---|
| `ENGINES: dict[str, Callable[[dict], Engine]]` | `{"nnterp": nnterp.load, "hooks": hooks.load}`; each takes the plan's `model` block. |
| `nnterp.load(model: dict) -> NnterpEngine`, `hooks.load(model: dict) -> HooksEngine` | Load the checkpoint the plan pins. |

`pyproject.toml` needs one addition in M1: with `data/` and `documents/` at
the top level, setuptools' flat-layout discovery refuses to build, so
`uv sync` fails (verified). `[tool.setuptools.packages.find] include =
["clab*"]` fixes it (verified: `uv sync` then installs clab editable through a
meta-path finder).

---

## 3. The engine protocol

```python
class Engine(Protocol):
    def execute(self, plan: dict, *, batch_size: int = 8,
                remote: bool | str = False) -> dict: ...
    def forward(self, batch: dict, writes: list[dict], reads: list[dict]) -> dict[str, Tensor]: ...
    def generate(self, batch: dict, writes: list[dict], reads: list[dict],
                 options: dict) -> tuple[Tensor, dict[str, Tensor]]: ...
    def readout(self, hidden: Tensor) -> Tensor: ...
    def sizes(self) -> dict[str, int]: ...
    @property
    def tokenizer(self) -> Any: ...
```

- **`execute`** is the single entry point and the only method an engine that
  ignores `steps.py` must have. It returns the filled plan.
- **`forward(batch, writes, reads)`** runs one batch through the model once,
  applies `writes` where they land, and returns each read by name, laid out
  `(rows_in_batch, pos, …)` (or ragged, §5). It takes no `batch_size`: it is
  given one batch.
- **`generate(batch, writes, reads, options)`** runs `generate` with
  `options = {"max_new_tokens", "min_new_tokens", "do_sample"}`, applies
  prompt-frame writes on the prompt call only, and returns the new token ids
  `[rows, n]` and each read as `(steps, rows, 1, …)` over the decode steps it
  asked for. `steps.py` turns that into the layout every consumer expects: a
  read at a fixed index *k* becomes `value[k]`, `(rows, 1, …)`; a scoped read
  becomes `{"values": stack(value[s_r, r] for each placed row r), "row": [r…]}`,
  `(entries, 1, …)`, where `s_r` is the step §5 resolves for row *r*.
- **`readout(hidden)`** is the model's own unembedding of a residual-stream
  tensor (`view: logits`): nnterp's `project_on_vocab`; on hooks, the native
  final norm and `lm_head` called on the tensor.
- **`sizes()`** is `{num_layers, hidden_size, num_heads, num_kv_heads,
  head_dim, intermediate_size, vocab_size}`, read off nnterp (a meta shell on
  hooks).
- **`tokenizer`** is the checkpoint's own tokenizer. On nnterp it is the
  property `self.model.tokenizer`, never a stored copy: inside a session that
  resolves to the serving side's tokenizer as a persistent object (verified:
  `engine.model.tokenizer` tokenized inside a `remote="local"` session).

### The records the engine receives

The engine receives the plan's own write and read records (§1), each extended
by `steps.py` with the run-time binding, as a new dict (the plan is not
mutated):

```python
write = {**plan_write,                          # component, layer, select, pos, mechanism, params, features
         "layer": 0,                            # always an int here; "all" was expanded
         "select": {"hidden_size": [0, 1, 2, 3]},   # narrowing as axis -> indices (heads on premix become units)
         "at": {"rows": LongTensor[R, w]},      # positions.resolve output, for this batch
         "operand": Tensor[R, w, d] | Tensor[d] | float | None,  # the named value, bound by the rule below
         "featurizer": dict | None}             # the instance, not its name
read  = {**plan_read, "layer": 0, "select": {...}, "at": {...}, "featurizer": dict | None}
```

`select` is bound from the document's spelling: `{"heads": [0]}` on
`attention_premix` becomes `{"hidden_size": [0, 1, 2, 3]}` (head 0's block of
`head_dim` units), on `attention_probs` or `attention_z` `{"num_heads": [0]}`,
on keys and values `{"num_kv_heads": [0]}`; `{"units": […]}` becomes the
component's feature axis (`intermediate_size` on `mlp_activation`).

**Operands.** A named operand is either *row-aligned* (it has a `rows` axis:
a read, or a metric) or not (a reduce's value, a number). Binding slices a
row-aligned operand to the batch's rows, `[lo:lo + R]`, and passes any other
operand whole; `interventions.edit` broadcasts an operand without a `rows` axis
over the selection's rows and positions. So `mean` (`[hidden_size]`, §1) lands
at any window on any row, which is what `entity_mean_ablation.json`'s header
says ("one vector, so it lands at any window"), and `steer_renormalize.json`'s
`direction` and `two_observes.json`'s `mean` bind the same way. Compile refuses
a row-aligned operand whose producer ran over a different dataset from the
consumer (row *i* would not be the same example), and preflight refuses one
whose window width differs from the write's.

The engine looks only at `component` and `layer` (to reach the value) and
hands the rest to `interventions.edit` / `interventions.gather`. That is the
seam: **the engine knows where a value is; `interventions.py` knows what to do
with it.**

### `steps.run`, the walk

```python
def run(engine, plan, batch_size):
    widths = preflight(engine, plan)              # tokenizes every input once; refuses by name; per-input pad widths
    state = {}                                    # (name, coords) -> value (§6, §7); never leaves the session
    for name, entry in plan["steps"].items():
        for step in points(entry):                # every step has "at"; {} when it is not a point
            KINDS[step["kind"]](engine, plan, name, step, state, Run(batch_size, widths))
            for key, slot in step["result"].items():
                slot["value"] = finished(lookup(plan, state, key, step["at"]))

def forward(engine, plan, name, step, state, run, rows=None, grad=False):
    rows = rows if rows is not None else plan["data"][step["data"]]["rows"]
    width = run.widths[(step["data"], key_of(step["input"]))]
    parts = []
    for lo in range(0, len(rows), run.batch_size):
        chunk = rows[lo:lo + run.batch_size]
        batch = data.encode(engine.tokenizer, [data.field(r, step["input"]) for r in chunk], width)
        writes = [bind(plan, w, batch, chunk, lo, state, step) for w in step["writes"]]
        reads = [bind(plan, r, batch, chunk, lo, state, step) for r in per_layer(step["reads"], engine.sizes())]
        with torch.set_grad_enabled(grad):        # only the fit's training pass sets grad=True
            got = engine.forward(batch, writes, reads)
        parts.append(viewed(engine, got, step["reads"]))   # view: logits -> engine.readout
    for read in step["reads"]:
        state[(name + "." + read["name"], coords(step["at"]))] = joined(parts, read)
```

`coords(at)` is `tuple(sorted(at.items()))`; `lookup` restricts the
consumer's `at` to the producer's axes (§7). `Run` is a two-field tuple, not a
class with behaviour. `rows` lets the fit run the same function on a minibatch
or on a substituted dataset (§9).

**Batching and left padding.** `steps.py` cuts every dataset into consecutive
row ranges of `batch_size` (the fit's own `batch_size` inside a fit). Every
step over the same dataset sees the same row order, so row *i* of
`counterfactual.v_cf` is row *i* of `patched`'s batch.

`data.encode` pads on the left by hand (pad id: the tokenizer's, else its EOS,
else 0), and every batch of one dataset input is padded to **that input's
longest row** (computed once by preflight), not the batch's. Two reasons. A
read with a `key` axis (`clean.pattern` in `attention_knockout.json`) then has
the same key width in every batch and joins along `rows`; padded per batch,
`weekdays/train` (11/9/9/9 tokens) at `--batch-size 2` would give key widths
11 and 9 and could not be concatenated. And a row's numbers no longer depend
on which rows share its batch. Verified on the tiny Llama with the four
weekday prompts: padded per batch, the last-position logits at batch sizes 4
and 2 differ by up to 6.0e-8; padded to the dataset's width, batch sizes 4 and
2 are bit-equal, and batch size 1 still differs from 4 by up to 8.9e-8 (the
matrix shapes change). So results are reproducible to the bit only at a fixed
`batch_size`, and the snapshot tests pin it.

Padding by hand rather than through the tokenizer's settings means padding does
not depend on how a checkpoint ships its tokenizer, and nothing on the serving
side's tokenizer is mutated. `data.encode` also computes `position_ids =
(attention_mask.cumsum(-1) - 1).clamp(min=0)`, and **both engines pass them
to the model**, so neither relies on nnsight's own heuristic for supplying
position ids. Verified: on a left-padded batch, nnterp with these
`position_ids` passed explicitly gives logits bit-equal to nnterp without them
and to `AutoModelForCausalLM` with them; and with them every one of the
fifteen hookable components (§4) is bit-equal between the engines on both
checkpoints. Without them the plain HF forward differs (GPT-2's block output by
0.106, Llama's attention output too).

**How `generate` serves decode steps.** A read in the `generated` frame with a
fixed index *k* asks for step *k*; a scoped one (`scope: variable` or
`segment: eos`, or `index: -1`) asks for every step, because which step it
means depends on what the model said (law 4). The nnterp engine opens
`model.generate(...)` with a bounded `tracer.iter[:n]` and, at each asked-for
step, reads the value at the step's last position (`value[:, -1:]`), in forward
order; prompt-frame writes and reads are made outside the loop, which binds
them to the prompt call (step 0). `steps.py` then resolves each scoped read to
one step per row from the new ids (§5). Verified: inside a `remote="local"`
session, a function opening `generate` with a step-0 write and per-step
`lm_head` reads returned `[3, 4, 32000]`, and each step's argmax equalled the
token generated at that step, bit-equal to `remote=False`.

The hooks engine serves `generate` with `AutoModelForCausalLM.generate` and the
same hooks as `forward`, plus a forward pre-hook on the top module that counts
calls: call 0 is the prompt, call *k* is decode step *k*. Prompt-frame writes
apply only while the count is 0; a generated-frame read captures, at each step
it asked for, the value at the call's last position into a list indexed by the
step; the ids are `generate`'s return minus the prompt. It passes the batch's
`attention_mask` and leaves decode-step position ids to `generate`, which
derives them from the mask (**belief**: bit-equality with the nnterp engine's
generation is M3's to verify; if it does not hold, M3 makes the hooks engine
refuse `generate` steps by name rather than serve different numbers).

### The same code inside a session and in plain Python

```python
# engines/nnterp.py
def execute(self, plan, *, batch_size=8, remote=False):
    with self.model.session(remote=remote):
        steps.run(self, plan, batch_size)
        nnsight.save(plan)
    return plan

# engines/hooks.py
def execute(self, plan, *, batch_size=8, remote=False):
    if remote:
        raise ValueError("the hooks engine runs locally; remote needs the nnterp engine")
    refuse_unservable(plan)          # every component and step kind, by name, before anything runs
    steps.run(self, plan, batch_size)
    return plan
```

**What the hooks engine imports and what it executes.** It imports `torch`,
`transformers` and `nnterp`. nnterp (and through it nnsight) is used only to
build a meta shell, `StandardizedTransformer(key, revision=…, dispatch=False,
device_map=None)`, which loads no weights; from it the engine reads the sizes,
`support()`, each component's nnterp key and `inside_forward()`, and the native
module paths (`envoy.path`). Every forward and generation it runs is
`AutoModelForCausalLM` with `register_forward_hook` /
`register_forward_pre_hook`; no nnsight trace, session or envoy is ever
executed. That is how this design reads the charter's "importing nothing from
nnsight": nothing from nnsight *for execution*, as PLAN §1 allows.

`steps.run` calls only `engine.forward`, `engine.generate`, `engine.readout`,
`engine.sizes`, `engine.tokenizer` and torch. Inside the nnterp session, each
`engine.forward` opens a trace nested in the session: nnsight pushes every
local of a nested trace back into the enclosing frame and filters by `save()`
only at the outermost boundary (`push_result` in `nnsight/tracing/tracer.py`),
so the dict `forward` returns is an ordinary value and **no `.save()` appears
anywhere in clab except `nnsight.save(plan)`**. On hooks, the same calls are
plain PyTorch.

### What a future engine that ignores `steps.py` must honor

- `execute(plan, **kwargs)` fills exactly the result slots the plan has, with
  values of the types and `dims` §1 lists, as CPU tensors, and changes nothing
  else in the plan; `output.write` then serves it unchanged.
- The semantics in §§4–9: what each component is, how a position resolves,
  that a sweep point equals a whole run with that value, that a write lands
  before a read at the same site observes it, that `alignment_missing` is a
  result.
- Refuse by name, before running anything, what it cannot serve.

---

## 4. `address.py`: the vocabulary

The corpus's eight names come first; the rest are the siblings nnterp serves.
`dims` name the axes a component presents; `pos` indexes the axis marked † and
a read is laid out `(rows, pos, *the remaining axes in order)`. "nnterp value"
is the envoy attribute the nnterp engine reads and assigns (host:
`L` = `model.layers[i]`, `A` = `L.self_attn`, `M` = `L.mlp`, `R` = the root).
"Hooks" is where the hooks engine reaches it: the native module, `out` = a
forward hook on its output (first tensor of a tuple), `in` = a forward
pre-hook on its first argument. Native paths are the meta shell's own
`envoy.path`: the hooks engine finds `model.layers[1].mlp.act_fn` as
`transformer.h.1.mlp.act` on GPT-2 by asking nnterp, and holds no
per-architecture table.

Verification, on both pinned checkpoints:
- `model.support()` reports every nnterp value below available under
  `attn_implementation="eager"`; under `sdpa` the six interior values (the
  ones marked source-located) report "read inside the eager attention forward,
  but this model runs 'sdpa'", and `attention_premix` stays available.
- `model.order(layer=0)` gives the order the table is in.
- Each value's nnterp key (`descriptor.path(envoy)`) and `inside_forward()`
  were read off the descriptors: every hookable row has a plain
  `input`/`output` key; every source-located row has `inside_forward() == True`.
- The fifteen hookable components were read by hooks on `AutoModelForCausalLM`
  and by nnterp on the same left-padded batch at layer 1: **bit-equal on every
  one, on Llama and GPT-2**. The native paths below are the ones that test used.

| component | meaning | dims († = pos axis) | narrowed by | r/w | nnterp value | hooks, Llama | hooks, GPT-2 |
|---|---|---|---|---|---|---|---|
| `block_output` | residual stream leaving block *i* | batch, seq†, hidden_size | layers, units | rw | `L.layer_output` | `model.layers.i` out | `transformer.h.i` out |
| `attention_output` | what attention adds to the stream | batch, seq†, hidden_size | layers, units | rw | `A.attention_output` | `model.layers.i.self_attn` out | `transformer.h.i.attn` out |
| `attention_probs` | the pattern the values are mixed with | batch, num_heads, query†, key | layers, heads | rw | `A.attention_probabilities` | source-located: refused by name | refused |
| `attention_query` | queries after projection, head split and RoPE | batch, num_heads, seq†, qk_head_dim | layers, heads | rw | `A.attention_queries` | refused | refused |
| `attention_premix` | every head's result side by side, as the output projection takes it | batch, seq†, hidden_size (heads in blocks of head_dim) | layers, heads, units | rw | `A.attention_premix`, flattened | `…self_attn.o_proj` in | `…attn.c_proj` in |
| `mlp_activation` | `act(gate_proj x)` (gated), `act(up_proj x)` (plain) | batch, seq†, intermediate_size | layers, units | rw | `M.mlp_activation` | `…mlp.act_fn` out | `…mlp.act` out |
| `lm_head` | the head's own projection | batch, seq†, vocab_size | units | rw | `R.lm_head_output` | `lm_head` out | `lm_head` out |
| `logits` | what the model returns (after any softcap) | batch, seq†, vocab_size | units | rw | `R.logits` | root `output.logits` | root `output.logits` |
| `token_embeddings` | the embedding module's output | batch, seq†, hidden_size | units | rw | `R.token_embeddings` | `model.embed_tokens` out | `transformer.wte` out |
| `block_input` | residual stream entering block *i* | batch, seq†, hidden_size | layers, units | rw | `L.layer_input` | `model.layers.i` in | `transformer.h.i` in |
| `attention_norm_output` | the normed stream attention is called with | batch, seq†, hidden_size | layers, units | rw | `L.attention_norm_output` | `…input_layernorm` out | `…ln_1` out |
| `attention_key` | keys, before `repeat_kv` | batch, num_kv_heads, seq†, qk_head_dim | layers, heads (kv) | rw | `A.attention_keys` | refused | refused |
| `attention_value` | values, before `repeat_kv` | batch, num_kv_heads, seq†, head_dim | layers, heads (kv) | rw | `A.attention_values` | refused | refused |
| `attention_scores` | scaled, masked scores entering the softmax | batch, num_heads, query†, key | layers, heads | rw | `A.attention_scores` | refused | refused |
| `attention_z` | each head's output, held as (heads, head_dim) | batch, seq†, num_heads, head_dim | layers, heads | rw | `A.attention_head_outputs` | refused | refused |
| `block_mid` | the stream between the sublayers | batch, seq†, hidden_size | layers, units | rw | `L.layer_mid` | `…post_attention_layernorm` in | `…ln_2` in |
| `mlp_norm_output` | the normed stream entering the MLP | batch, seq†, hidden_size | layers, units | rw | `L.mlp_norm_output` | `…post_attention_layernorm` out | `…ln_2` out |
| `mlp_input` | what the MLP is called with | batch, seq†, hidden_size | layers, units | rw | `L.mlp_input` | `…mlp` in | `…mlp` in |
| `mlp_neurons` | what the down projection sums (`act(gate)·up` gated) | batch, seq†, intermediate_size | layers, units | rw | `M.mlp_neurons` | `…mlp.down_proj` in | `…mlp.c_proj` in |
| `mlp_output` | what the MLP adds to the stream | batch, seq†, hidden_size | layers, units | rw | `M.mlp_output` | `…mlp` out | `…mlp` out |
| `final_norm_output` | the final norm's output, what the head projects | batch, seq†, hidden_size | units | rw | `R.final_norm_output` | `model.norm` out | `transformer.ln_f` out |

Notes the table cannot carry:

- `attention_premix` is held flat, as the header of `head_patching.json`
  says ("every head's result side by side"); `heads: [h]` on it selects units
  `h·head_dim … (h+1)·head_dim − 1` of the hidden axis. `attention_z` is the
  same numbers held as `(heads, head_dim)`. Verified: nnterp's
  `attention_premix` and `attention_head_outputs` are bit-equal on both
  checkpoints, so the header's "gives the same numbers bit for bit" holds.
- `heads` on `attention_key`/`attention_value` indexes `num_kv_heads`; on the
  pattern, queries and z it indexes `num_heads`. On these checkpoints the two
  are equal (4 and 4).
- `view: "logits"` is allowed where `stream` is true: `block_input`,
  `block_mid`, `block_output`, `attention_output`, `mlp_output`,
  `token_embeddings`. Compile refuses it elsewhere (applying the final norm to
  `final_norm_output` would norm twice).
- `layers` is an int, `"all"` (every block, stacked layer axis first), or a
  sweep of ints. Root components take no `layers`.
- The token axis of the pattern and scores is `query`; `key` stays whole, and
  it is the padded batch's key axis, pads included. That is why
  `attention_pattern_patching.json` needs equal per-row lengths (§5).
- A read of `attention_probs` with `pos: -1` is laid out
  `(rows, 1, num_heads, key)`; of `attention_query`, `(rows, 1, num_heads,
  qk_head_dim)`. A write operand must have the read's layout.
- The nnterp engine sorts every site of a forward by `model.rank(name, layer)`
  before touching any (verified necessary: reading `attention_premix` after
  `attention_output` in one trace raises `OutOfOrderError`). At one site it
  applies the writes and then gathers the reads from the edited tensor it
  holds, so a read at a written site observes the write, on both engines.
- A future family whose nnterp value is not a plain module boundary (a
  sandwich-norm `attention_output` keyed `../post_attention_layernorm.output`)
  is reached by hooks at whatever module its nnterp key names; a key with a
  `source` segment is refused by name. That generalization is a **belief**
  until M2 runs it on a third family; M1 and M2 need only Llama and GPT-2.

---

## 5. Positions

### One shape

Every `pos` compiles to

```python
{"frame": "prompt" | "generated",
 "scope": None | {"variable": "<column>"} | {"segment": "user[1]" | "eos" | …},
 "start": int, "stop": int | None}          # a half-open slice of the scope's tokens
```

| corpus form | compiled |
|---|---|
| `-1` | `{"frame": "prompt", "scope": None, "start": -1, "stop": None}` |
| `-3` | `start -3, stop -2` |
| `{"span": [-4, -1]}` | `start -4, stop -1` |
| `{"all": true}` | `start 0, stop None` |
| `{"index": -1, "scope": {"variable": "entity"}}` | `scope {"variable": "entity"}, start -1, stop None` |
| `{"all": true, "scope": {"variable": "entity"}}` | `scope {"variable": "entity"}, start 0, stop None` |
| `{"index": -1, "scope": {"segment": "user[1]"}}` | `scope {"segment": "user[1]"}, start -1, stop None` |
| `{"frame": "generated", "index": k}` | `frame "generated", start k, stop k + 1` (`stop None` for `k = -1`) |
| `{"frame": "generated", "index": -1, "scope": {"variable": "said"}}` | `frame "generated", scope {"variable": "said"}, start -1, stop None` |

A slice whose width is the same on every row is **rectangular** (both ends
negative or `None`, or both non-negative): the read is `(rows, w, …)`. Any
other is **ragged** (`all`, or a variable scope whose text differs in length
between rows under `all`): the read is `{"values": [N, …], "row": [N]}`.
Compile decides which, from the record alone.

### What a scope's tokens are

All of this runs in `positions.resolve`, inside the session, on the output of
`data.encode` (law 4: which tokens a spec names is decided where the model is,
through its own tokenizer).

- **No scope (the prompt).** The row's tokens whose id is not in
  `tokenizer.all_special_ids`. So `-1` is the last token of the prompt and BOS
  is never "a token of the prompt", which is what `pca_harvest.json`'s header
  asks (verified: 34 non-BOS tokens over the four weekday prompts). The id test
  is used, not the tokenizer's `special_tokens_mask`, because a rendered chat is
  tokenized with `add_special_tokens=False` and its mask is then all zeros even
  though the text holds `<s>` and `</s>` (verified on the pinned Llama: mask sum
  0; the id test marks `<s>`, `</s>`, `<s>`; on a plain prompt both give
  `[1, 0, …, 0]`). Padding is never in any scope.
- **`{"variable": c}`.** The text `variables(row)[c]`: the row's own column for
  the plain input; for an indexed field (`counterfactual_inputs[0]`), the
  matching entry of `<column>_variables` (`counterfactual_inputs_variables[0]
  ["entity"]`), which is how `entity_patch.json` addresses each prompt "by what
  it says". The text must occur in the row's input; its tokens are those whose
  character offsets overlap its character span. Verified on the pinned Llama:
  `Thursday` is `▁Th urs day` (tokens 4–6), `Saturday`/`Friday`/`Sunday` one
  token each. Zero occurrences: the row is **missing**; two or more:
  **ambiguous**. In the prompt frame an ambiguous row is refused by name (the
  rule `chat_turn.json` cites; the data decides it, not the model). In the
  generated frame it is reported, like a missing row (below; question 9).
- **`{"segment": "role[n]"}`.** `data.encode` renders the chat through
  `tokenizer.apply_chat_template(..., tokenize=False)`, tokenizes the rendered
  text with `add_special_tokens=False` (the template already has `<s>`), and
  records each message's character span by finding its content in the rendered
  text in message order. `user[1]` is the second user message; a bare `user`
  with more than one is ambiguous and refused. Verified rendering on the pinned
  Llama: `<s>[INST] Name a weekday. [/INST] Thursday. </s><s>[INST] If today
  is Thursday, tomorrow is [/INST]`; `user[1]`'s last token is `▁is`, not
  `[/INST]`, as the header says.
- **`frame: "generated"`.** The scope is the new tokens. Generated index *k*
  means **the forward call that chose new token *k*, read at its last
  position** — `tracer.iter[k]`, whose logits are the distribution token *k*
  was drawn from. So `{"frame": "generated", "index": 0}` on `lm_head` is the
  prompt call's last-position logits, and every generated token, the last one
  included, has a step to read. A variable scope locates the text in the
  decoded continuation by `char_spans` (decoding growing prefixes of the new
  ids); `segment: eos` is the first EOS the row generated.

Two combinations the grammar admits are refused at compile, naming the record:
`segment: "role[n]"` in the generated frame (a continuation has no turns), and
`segment: "eos"` in the prompt frame (a prompt does not stop).

### Missing rows

| where | a missing row |
|---|---|
| a prompt-frame write | refused by name: document, step, write, row, the text that is not in the prompt |
| a prompt-frame `index`/`span` read | refused the same way (a rectangular read has no hole) |
| a prompt-frame ragged read | contributes no entries and stays a row (`row` simply lacks it) |
| any `generated`-frame read | a result: the row is absent from the ragged value, and every table over it reports `{"row": i, "alignment_missing": true}`; a row where the text occurs twice in what the model said reports `{"row": i, "alignment_ambiguous": true}` the same way (question 9) |

So `generated_answer.json` comes back with two rows scored and two
`alignment_missing` under `at_said`, and `at_stop` missing on every row (EOS is
held off by `min_new_tokens`), with no exception.

### The checks that need the tokenizer

`steps.preflight` tokenizes every forward step's input once, before any
forward, and refuses:

- a write or rectangular read whose slice does not fit inside some row's scope;
- a missing or ambiguous variable where the table above says refuse;
- a write to a component with a `key` axis (pattern, scores) whose operand was
  read on another input, unless the two inputs tokenize to the same length on
  every row — `attention_pattern_patching.json`'s check, run before any
  forward (verified: `weekdays_aligned` is 9/9 tokens on all six pairs; the
  same check on `weekdays/train` fails on rows 0 and 2, 11 vs 9);
- a metric token column that is not exactly one token on some row;
- a featurizer bundle whose width is not the site's (§9).

### The window and three single positions are the same tensor

`window_patch.json` writes one record `start -4, stop -1`;
`multi_position_patch.json` writes three, `-4/-3`, `-3/-2`, `-2/-1`. Both
resolve, per row, to the padded indices of the last-but-three, -two and -one
prompt tokens. `interventions.edit` applies all writes at a site to one clone
of the value, in listed order, and the engine hands the model back that one
tensor: three swaps at disjoint positions of the same counterfactual values
leave exactly the elements one window swap leaves. Same tensor into the rest
of the forward, same logits to the bit; the M2 suite asserts `torch.equal`.

### Slow tokenizers

Now: both pinned checkpoints load a fast (`TokenizersBackend`) tokenizer under
transformers 5.19 (verified; the Llama checkpoint ships `tokenizer.model` and
`tokenizer.json`). Prompt scopes use the tokenizer's offset mapping.
`data.encode` asks for offsets only when some record in the step has a
variable or segment scope, and a slow tokenizer there is refused by name: "this
tokenizer (`<class>`) has no offset mapping; a variable or segment scope needs
one". Unscoped positions need only the special-token mask, which slow
tokenizers return too.

Later: the generated frame already needs `char_spans` (growing-prefix
decoding) in M3, because generated text has no offsets. When a slow tokenizer
appears, prompt scopes use the same function on the prompt's ids, and the
refusal goes. One path is added in M3 for a reason of its own, and the dual
path PLAN §5 anticipates costs one `if` once it exists. Known hazard
(**belief**, untested): a prefix that ends mid-UTF-8 sequence decodes to `�`,
which can shift spans by a character; `char_spans` will decode each prefix
with the tokenizer's own byte handling and the M3 tests will include a
multi-byte word.

---

## 6. State and names

`state` is a dict that lives in `steps.run`'s frame, inside the session, and
is discarded when `run` returns. Every value a step produces goes into it under
its name, and every name a step references is looked up in it. **"Thing, then
member"**: a name splits at its first dot; the thing is a step, a dataset or a
featurizer (compile refuses a name used for two of them, and names with dots;
verified the 31 documents have neither).

| name | thing | member | value |
|---|---|---|---|
| `counterfactual.v_cf` | step | a read | the read tensor (or ragged dict) |
| `mean`, `iia` | step | none: the step's own value | reduce or metric value |
| `patched` (a generate step) | step | none | the new ids `[rows, n]` |
| `pairs.cf_answer` | dataset | a column | the column as a list, aligned with rows |
| `rot` | featurizer | none | the declared instance: drawn from its seed, or loaded |
| `mask.mask` | featurizer | its own scalar | the mean of the soft mask (§9) |
| `fit.rot` | step (the fit) | a trained featurizer | the fit's instance |
| `fit.iia` | step (the fit) | a body metric | that metric under the eval substitution, after training |

Scopes: a fit's body is a nested namespace. Inside it, a body step's name means
the body step (`counterfactual.v_cf` in the body is the body's read), and the
name of a featurizer the fit trains means the instance being trained, the same
object as `fit.<name>`. That is what makes `mask.mask` in `dbm.json`'s
objective the L1 of the mask being trained. Outside the fit, `rot` is the
declaration's own instance and `fit.rot` the trained one. Named interventions
are templates inlined at compile, so the names inside them resolve where they
are used: `das.json`'s `das` intervention reads the body's `counterfactual.v_cf`
inside the fit and the root's after it.

In a sweep, a name resolves to the producer's point that agrees with the
consumer's coordinates on the producer's axes (§7).

**What `saves` keeps.** A save makes a result slot (§1); `steps.run` copies the
value into it when the step finishes. Everything else in `state` is ephemeral.

**What travels home, and when.** Once, at the end of the session:
`nnsight.save(plan)` marks the plan, and the plan (its rows, its records and
its filled slots, nothing more) is the one value the session's outermost
boundary returns. Verified (§10): the only name that came home was `plan`.
`state`, the featurizer instances, the optimizer and the encodings never
leave the session.

---

## 7. Sweeps

**Expansion is compile-time and generic.** Compile walks the document and
treats any `{"sweep": [...]}` standing for a value as an axis. An axis is
named by its **label**, the last key of its document path that is not a list
index (`sites.target.heads` → `heads`, `featurizers.rot.seed` → `seed`,
`steps.fit.objective[1][0]` → `objective`,
`steps.patched.reads.logits.pos.index` → `index`); `plan["sweeps"]` keeps the
path beside the values for the manifest. Two different sweeps with one label
are refused at compile, naming both paths: the corpus has at most one sweep
per document, and a rename is cheaper than a naming scheme. A sweep may sit
anywhere a value does except under `header` and `model`, and not on a name,
a kind, a data `path` or a `file_path` (proposal 5; question 8). The engine
never sees a sweep.

**Which entries become points.** An entry's axes are the sweeps in its own
spec, in the sites, interventions and featurizers it uses, and the axes of
every entry whose values it references, transitively. A reference is any
name a record holds: an operand, a metric's `of` and arguments, a reduce's
`of`, an objective term, **and a `featurizer` field**. The last one is what
carries `dbm.json`'s objective sweep to its root `counterfactual` step: that
step's only link to the fit is `"featurizer": "fit.mask"` inside the inlined
`cf_read` intervention, so it becomes four points, one per trained mask. An entry with no
axes is one dict. An entry with axes is the list of points over the cartesian
product of its axes, in axis order of first appearance; each point is fully
concrete, with `at` giving its index on each of its axes. One sweep object
used in two places is one axis, so the points are zipped across those places,
not crossed. Featurizers follow the same rule: `random_subspace.json`'s
`featurizers.rot` becomes three points, and so do the steps that use it.

**Steps that no sweep touches run once.** `pos_sweep.json` runs its
counterfactual forward once and its patched forward three times. Each point's
numbers are what running the whole document with that value would produce,
since forwards are deterministic and share no state across points; what
differs from "the root is a list of plans" is only that untouched steps are
not repeated. §12 asks the owner to confirm this reading; §13 explains why it
is the charter's.

**How a reference finds its point.** `state` keys are `(name, coords)`, where
`coords` is the producer's `at` as sorted pairs (empty for an unswept
producer). A consumer point
at `at` looks a name up with `at` restricted to the producer's axes: unswept
producers have the empty restriction, so every point of `patched` reads the
one `counterfactual.v_cf`.

**A metric over a swept read reports per point**, because it references the
read and so inherits its axes: `pos_sweep.json`'s `iia` is three points with
three slots, each a `[4, 1]` tensor, written to three files (§8). Every table
records its point in an `at` field with the axis labels and values
(`{"pos": -2}`).

**`layers: "all"` is not a sweep.** It is one read whose layer axis comes
first, `(layers, rows, pos, …)`; its metric's table has a row per layer and
example, each saying its layer, as `logit_lens_all.json`'s header says. The
same numbers as `logit_lens.json`'s two points (the M2 suite checks it).

---

## 8. Output hierarchy and manifest

The output directory is `out/<document name>/` unless `--out DIR` is given,
in which case it is `DIR` itself. Both resolve against the working directory,
as a command-line path does; a document's data paths and `file_path`s resolve
against the corpus root (the document directory's parent). The two agree when
`clab` runs from the corpus root, which is how the corpus's headers say to run
it (`das.json --out documents/artifacts`). Within the output directory:

- A save's filename is a path **relative to the output directory**, used as
  written: `"compare/patching/iia.json"` lands at
  `<out>/compare/patching/iia.json`, and `das.json --out documents/artifacts`
  puts `fit.rot` at `documents/artifacts/rot.safetensors`, where
  `das_apply.json`'s `file_path` expects it.
- A value with sweep coordinates lands under one directory per axis it varies
  over, in axis order, named `<label>=<value>`: `pos=-2/iia.json`,
  `heads=0/iia.json`, `units=0,1,2,3/iia.json`, `objective=0.0001/mask.safetensors`.
  An unswept value of a swept document stays at the top.
- The bare list form (`"saves": ["kl", "js", "p_answer"]`) uses `<name>.json`
  for tables and `<name>.safetensors` for tensors.
- The extension picks the writer: `.json` for metric tables; `.safetensors` for
  tensors, ragged values (`values`, `row`), ids and bundles. A metric saved as
  `.safetensors`, or a tensor as `.json`, is refused at compile (none of the
  31 does either; verified by listing every save).
- `manifest.json` is always written at the root of the output directory.
- Compile refuses two saves that resolve to the same file (after the
  coordinate directories are added), and a save named `manifest.json`, naming
  both saves.

This lays files out by the paths the document gives, not as the plan's own
step hierarchy; the step hierarchy is in the manifest. That is a deviation
from the charter's wording, for the reasons in §13.1, and question 10 puts it
to the owner.

`two_observes.json`:

```
out/two_observes/
  manifest.json
  clean.json
  compare/patching/iia.json
  compare/patching/logit_diff.json
  compare/ablation/logit_diff.json
```

`pos_sweep.json`:

```
out/pos_sweep/
  manifest.json
  pos=-1/iia.json
  pos=-1/logit_diff.json
  pos=-2/iia.json
  pos=-2/logit_diff.json
  pos=-3/iia.json
  pos=-3/logit_diff.json
```

A metric table:

```json
{"name": "iia", "metric": "match", "of": "patched.logits", "at": {"pos": -2},
 "dims": ["rows", "pos"], "mean": 0.25,
 "rows": [{"row": 0, "pos": 0, "value": 0.0},
          {"row": 1, "pos": 0, "value": 1.0}, …]}
```

Rows carry `example_id` when the dataset has one, `layer` when the value has a
layer axis, and `{"alignment_missing": true}` (or `alignment_ambiguous`)
instead of `value` for a row a read could not place. `mean` is over the rows that have a value. A tensor file
holds its tensor under the save's name, with `dims` and `at` in the
safetensors metadata; a bundle holds its own tensors and header (§9).

The manifest:

| field | content |
|---|---|
| `format` | `"clab/1"`, the plan's format string |
| `document` | path, name, sha256 |
| `model` | the plan's model block |
| `engine` | `"nnterp"` or `"hooks"` |
| `remote` | `false`, `"local"` or `true` |
| `batch_size` | as run |
| `versions` | the installed clab, nnsight, nnterp, transformers, torch, python |
| `sweeps` | `[{"label", "path", "values"}]` |
| `data` | each dataset's `{"path", "split", "columns", "n_rows"}`: the plan's `data` without its rows |
| `steps` | the plan's `steps` as compiled, every slot's `value` replaced by `{"file", "kind", "dims", "shape"}`: the plan's hierarchy, pointing at the files |
| `saves` | `[{"name", "step", "at", "file"}]`, in document order |
| `started`, `seconds` | wall clock |

Two documents run into one `--out` directory (das.json, then pca_harvest.json,
both into `documents/artifacts`) share it: their saved files coexist, the
second run's `manifest.json` replaces the first's. §12 asks about that.

---

## 9. Featurizers and `fit`

M4 builds all of this; the shape is fixed now so M1's records already carry
the `featurizer` and `features` fields M4 fills.

### A featurizer is a dict of tensors, made where the model is

```python
{"kind": "subspace", "k": 8, "width": 16, "training": False,
 "origin": Tensor[16, 16],         # the seeded draw: a random orthogonal matrix
 "theta": Tensor[16, 16],          # the Cayley parameter, zero at the draw (requires_grad inside a fit)
 "basis": Tensor[16, 8]}           # (origin @ cayley(theta))[:, :k], recomputed when theta moves
{"kind": "gate", "width": 16, "training": False, "temperature": 1.0, "theta": Tensor[16]}
{"kind": "pca", "k": 4, "width": 16, "basis": Tensor[16, 4], "mean": Tensor[16]}
{"kind": "sae", "width": 16, "latents": 32,
 "W_enc": Tensor[16, 32], "b_enc": Tensor[32], "W_dec": Tensor[32, 16], "b_dec": Tensor[16]}
```

`featurizers.make` builds the instance inside the session, at the featurizer's
first use: from the bundle the plan carries (`file_path`), or drawn from a
`torch.Generator` seeded with the featurizer's `seed`, else the fit's `seed`,
else 0. Seeded generation on CPU is identical on both engines. No `nn.Module`:
`featurizers.py` is functions over these dicts, and `steps.py` holds them in
`state`.

What "drawn from its seed" means, per kind:

- **subspace**: `origin` is the Q of a QR decomposition of a `[width, width]`
  standard normal drawn from the generator, with the signs of R's diagonal
  folded into Q, which makes it uniformly (Haar) distributed over orthogonal
  matrices; `theta` starts at zero, so `cayley(theta)` is the identity and the
  untrained basis is `origin[:, :k]`, a uniformly random rank-k subspace.
  That is what makes `random_subspace.json`'s three seeds a null: a Cayley
  parameter of zero alone would be the unit basis, the same subspace for every
  seed. Training moves `theta` and leaves `origin` fixed.
- **gate**: `theta` starts at +1 on every unit (question 4); nothing is random.
- **pca, sae**: never drawn; they come only from a bundle (compile refuses a
  `pca` or `sae` featurizer without `file_path`).

### Reads and writes "in the featurizer's space"

A read with `featurizer` is `encode(f, h)` of the gathered selection; a write
with `featurizer` takes its operand in that space and is
`replace(f, h, operand, features)`. Every `replace` is a **delta**, so writing
back a value's own coordinates changes nothing to the bit:

| kind | `encode(f, h)` | `replace(f, h, new, features)` |
|---|---|---|
| subspace | `h @ B` | `h + (new' − h @ B) @ Bᵀ` |
| pca | `(h − μ) @ B` | `h + (new' − (h − μ) @ B) @ Bᵀ` |
| gate | `h` (the unit basis) | `h + m ⊙ (new − h)`, `m = σ(θ/T)` while training, `θ > 0` when scored |
| sae | `z = relu((h − b_dec) @ W_enc + b_enc)` | `h + (z' − z) @ W_dec` |

A `pca` read centres on the bundle's mean `μ`, so its coordinates are PCA
scores as PCA defines them; for a swap the mean cancels (`new'` was centred by
the same `μ`), so `pca_control.json`'s interchange is the same with or without
it.

`new'` and `z'` are the current coordinates with `new` written over the
`features` subset (all of them when `features` is absent). The SAE's error term
`h − decode(z)` is never formed: the delta form keeps it exactly, and an
ablated latent that was already zero gives `z' == z`, a zero delta and the
unchanged `h`, which is what `sae_feature_ablation.json`'s header promises.

### The training loop, inside the session

```python
def fit(engine, plan, name, step, state, run):
    at, sizes = step["at"], engine.sizes()
    trained = {}
    for f in step["train"]:                                    # declared featurizer names
        spec = featurizer_spec(plan, f, at)                    # plan["featurizers"][f], or its point at `at`
        width = site_width(plan, step, f, sizes)               # the feature width of the sites f is used at
        trained[f] = featurizers.make(spec, width, seed=spec["seed"] if spec["seed"] is not None else step["seed"])
    params = [p for inst in trained.values() for p in featurizers.parameters(inst)]
    opt = OPTIMIZERS[step["optimizer"]["name"]](params, **without_name(step["optimizer"]))  # adamw, adam, sgd
    datasets = {d: plan["data"][d]["rows"] for d in body_datasets(step)}
    stop, best, since = step["early_stop"], None, 0
    for epoch in range(step["epochs"]):
        anneal(trained, step["anneal"], epoch, step["epochs"])
        for rows in minibatches(datasets, step["batch_size"], step["seed"], epoch):   # {dataset: row slice}
            body = body_run(engine, plan, name, step, state, run, rows, trained, training=True)
            loss = sum(w * objective_term(body, trained, term) for w, term in step["objective"])
            grads = torch.autograd.grad(loss, params)
            for p, g in zip(params, grads):
                p.grad = g
            opt.step(); opt.zero_grad()
        scored = body_run(engine, plan, name, step, state, run, datasets, trained, training=False)
        best, since = keep_best(best, since, mean_of(scored, stop["metric"]), stop["mode"], trained)
        if since >= stop["patience"]:
            break
    restore(trained, best)                                     # the best epoch's featurizer tensors
    held_out = {step["eval"]["data"].get(d, d): plan["data"][step["eval"]["data"].get(d, d)]["rows"]
                for d in datasets}                             # train -> test rows, under the body's names
    scored = body_run(engine, plan, name, step, state, run, held_out, trained, training=False)
    for f, inst in trained.items():
        state[(name + "." + f, coords(at))] = inst
    for metric in body_metrics(step):
        state[(name + "." + metric, coords(at))] = scored[metric]
```

The helpers, each a few lines in `steps.py`:

- `body_run(engine, plan, name, step, state, run, rows, trained, training)`
  walks the fit's `steps` with the same `KINDS` functions as the root, in a
  namespace of its own that falls back to `state`, with every trained
  featurizer bound under both its declared name and `<fit>.<name>` (§6), each
  dataset of the body read from `rows` (a minibatch, the whole training set,
  or the eval substitution's rows under the body's dataset names), with
  gradients on only when `training`, and with gates soft only when `training`.
  It returns the body namespace.
- `objective_term(body, trained, term)`: a body metric's value averaged over
  every axis, or, for `<featurizer>.<scalar>`, `featurizers.scalars(...)[scalar]`.
- `keep_best` compares the epoch's score with the best so far by `mode` and
  copies the featurizer tensors when it improves; `since` counts epochs since.
- `minibatches` shuffles row indices per epoch with a generator seeded by
  `seed` and the epoch, and cuts them into `batch_size` slices.

- **Objective**: `[[weight, name], …]`; `name` is a body metric step (its value
  averaged over rows) or a featurizer scalar (`mask.mask`, the mean of
  `σ(θ/T)`). A swept weight makes the fit a list of points like any step.
- **Early stop**: after each epoch the body runs over the **training** data in
  scoring mode; `mode` max/min on the named metric's mean; `patience` epochs
  without improvement end the fit; the best epoch's featurizer is kept. The
  eval substitution runs **once**, after the fit, so `fit.iia` (saved as
  `held_out_iia.json`) is not the score the model was selected on. This is the
  default of question 7; the alternative selects on the eval split.
- **Anneal**: `{"mask": {"start", "end"}}` sets the gate's temperature,
  linearly over epochs.
- **`eval.data`** (`{"train": "test"}`) re-runs the body with dataset `train`
  read as `test`; `fit.<metric>` is the body metric from that run, for every
  metric step of the body.
- **`fit.rot`** is the trained instance; inside the body, `rot` and
  `fit.rot` are that same object (§6).
- **Gradients**: `torch.autograd.grad(loss, params)` computes gradients for the
  featurizer only; the model's weights are never frozen or touched, which
  matters on a shared server. Minibatch rows are shuffled per epoch by a
  generator seeded with `seed` and the epoch.

**Verified**, with a scratch fit inside one session (a Cayley-parametrized
rank-8 rotation at block 0, the counterfactual read through it in one trace,
swapped into the base in its space by out-of-place assignment in a second
trace, cross-entropy on the counterfactual answer, `autograd.grad`, AdamW
created inside the session): six steps lowered the loss from 10.43459 to
10.42788; no model parameter had a `.grad` afterwards; under
`remote="local"` the six losses were **bit-equal** to `remote=False`. The
gradient flows through both traces: the counterfactual read's graph stays
alive across the session.

### Bundles

A bundle is a safetensors file: the featurizer's tensors plus a string header
in the safetensors metadata:

```python
{"format": "clab/1", "kind": "subspace", "k": "8", "parametrization": "cayley",
 "width": "16", "model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
 "model.revision": "9fb19125…", "component": "block_output", "layer": "0",
 "document": "das"}
```

- **Saving** a featurizer (`fit.rot`) or a `pca` reduce writes this header from
  what produced it: the model block, and the site of the read the featurizer
  was used at or the reduce was taken over.
- **Loading** happens at compile: the file is read and its header and tensors go
  into the plan (`plan["featurizers"][name]["bundle"]`). Compile checks the
  header against the document key by key — `kind`, `k`, `parametrization`,
  `model.key`, `model.revision`, and `component`/`layer` against every site the
  featurizer is used at — and refuses naming the key: `rot.safetensors: header
  'layer' is 1; this document uses 'rot' at block_output layer 0`. `width` is
  checked against the model's size at preflight (it needs the model).
- `das_apply.json` and `pca_control.json` therefore refuse before anything runs
  when `das.json` or `pca_harvest.json` has not been run into
  `documents/artifacts`, naming the missing file.
- `sae_feature_ablation.json` names `documents/artifacts/sae.safetensors`,
  which nothing in the corpus writes and which does not exist (verified). §12
  asks how it should come to exist.

---

## 10. Remote

### How the shape satisfies the laws

1. **One request, one session.** `NnterpEngine.execute` opens exactly one
   `self.model.session(remote=remote)`; tokenization, every forward and
   generation, metrics, reduces, the whole fit loop and slot filling happen in
   `steps.run` inside it. `remote` is passed through and nothing else looks at
   it.
2. **The plan is pure data.** §1. The only tensors are bundle inputs, carried
   as data.
3. **Engine and model only.** The session body is two lines,
   `steps.run(self, plan, batch_size)` and `nnsight.save(plan)`; it references
   `self` (holding `model` and a config dict), `plan`, `batch_size`, and the
   modules `steps` and `nnsight`. Datasets travel as the plan's rows; the
   tokenizer is `self.model.tokenizer`; bundles are in the plan.
4. **No client decision from a tensor.** Compile reads only the document, the
   data files and bundle headers. Every tokenizer- or model-dependent decision
   (positions, alignment, variable presence, single-token answers, widths,
   ragged counts) is in `steps.preflight` and `steps.run`, in the session.
5. **Nothing ships by value.** clab is an installed (editable) package, so
   nnsight's `get_local_env` lists it with its version, not as `"local"`, and
   `pull_env` does not register it; nnterp likewise.

### What was verified, with a scratch engine on the tiny Llama

The scratch engine lived in a temporary `clab/probe.py` inside the worktree
(installed editable by `uv sync`, deleted before the commit), with the shape
above: `Engine(model, config)`, `forward` opening a trace, a module-level
`run(engine, plan, batch_size)` standing in for `steps.run`, and
`execute` = session + `run` + `nnsight.save(plan)`. The plan was patching.json's
experiment (counterfactual read at block 0, swap into the base, logits read,
`match` metric) over the four weekday rows. Scripts ran from a scratch
directory, `CUDA_VISIBLE_DEVICES=`, `HF_HUB_OFFLINE=1`.

**(a) A session body referencing the engine runs under `remote="local"`: yes.**

```
MODE False  OK; filled: True {'iia': ((4,), 'torch.float32'), 'patched.logits': ((4, 32000), 'torch.float32')}
MODE local  OK; filled: True {'iia': ((4,), 'torch.float32'), 'patched.logits': ((4, 32000), 'torch.float32')}
bit-equal iia True
bit-equal patched.logits True
payload bytes 9137
block source:  run(self, plan, batch_size)
               nnsight.save(plan)
shipped globals: {'run': 'function', 'nnsight': 'module'}
shipped locals:  {'self': 'Engine', 'plan': 'dict', 'batch_size': 'int'}
engine copy type: <class 'clab.probe.Engine'>  dict keys: ['model', 'config']
engine.model resolves to live model's module: True
'clab.probe' name in payload: True ; 'def forward' source in payload: False
by-value modules: ['probe_model', 'sitecustomize', 'test', 'verify_remote', 'xxlimited', 'xxlimited_35', 'xxsubtype']
local-env says: clab= 0.0.1  nnterp= 1.3.1.dev408+g27c8c266b
```

The engine instance is pickled as a reference to its class plus its two
attributes; its model resolves to the live model through nnsight's persistent
ids; `run` and the class travel by import path, not by source; clab and nnterp
are not in cloudpickle's by-value registry. (The by-value list holds the scratch
scripts themselves, which `pull_env` registered because they sat on a
non-site-packages path, as designed. That is also why clab must run from its
installed package and the scratch scripts from outside the repo.) Also
verified under `remote="local"`: the fit loop of §9 (bit-equal losses) and a
`generate` with a step-0 write and per-step reads (bit-equal logits).

**(b) `nnsight.save(plan)` brings a nested structure with tensors home:
shown only against a stand-in for the server, which is this design's own model
of one, not nnsight's server code.** `remote="local"` cannot show it (see (c)).
The stand-in is a scratch backend that does what I believe a server and
`RemoteBackend` do together: serialize the session; deserialize it with the
model's persistent objects; execute the **deserialized** tracer (nnsight's own
`_ServerRoundTrip` test helper does this much); capture what that tracer's
outermost boundary pushes; `torch.save` it (as `ResponseModel.pickle` does) and
`torch.load` it; then mark and push it into the caller's frame, copying the two
lines of `RemoteBackend._push`. Outcome:

```
names that came home: ['plan']   bytes: 515365   (5635429 before slots held clones)
returned is a new object (the copy that came home): True ; original plan left unfilled: True
nested structure intact: <class 'dict'> ['data', 'steps', 'saves', 'result_slots_filled'] <class 'torch.Tensor'>
bit-equal to local iia True
bit-equal to local patched.logits True
```

The scratch plan was a simplified stand-in for §1's shape (a root `saves`
dict of slots and a `result_slots_filled` flag instead of slots on the steps);
what it tests is the transport of nested dicts, lists and tensors, which does
not depend on which keys the dicts have.

What this shows: the deserialized tracer fills its own copy of the plan; that
copy survives `torch.save`/`torch.load` whole; and `torch.save` of a slot that
holds a view ships the view's whole storage, so the clone rule (§1) is needed.
What it does not show: that a real server collects the saved names the way the
stand-in does, and that the real `RemoteBackend._push` lands the copy in
`execute`'s local so `return plan` returns it. In the stand-in the push did
(Python 3.13), but that stays a **belief** until M5 runs against a server. M1
puts the stand-in in the suite beside `remote="local"`, as a regression test
for the serialization of the way home, labelled as what it is.

**(c) What `remote="local"` does and does not exercise** (read from
`nnsight/intervention/backends/local.py`, and confirmed by the `out is plan`
outcome above):

| exercised | not exercised |
|---|---|
| the session block reduced to source and the names it references | running the **deserialized** tracer: local runs the caller's original tracer with the deserialized code |
| pickling those names with nnsight's pickler (by reference vs by value, laws 3 and 5) | the result's trip home: no `torch.save`, no download, no push of a copy; the caller's own plan is filled in place |
| deserializing with non-installed modules hidden: sys.path entries outside site-packages removed, every non-server, non-stdlib module dropped from `sys.modules` and re-imported (clab and nnterp come back through their editable finders) | the server's installed versions of clab, nnterp, nnsight, transformers; its `attn_implementation`, dtype, device (GPU tensors in slots), or any import allowlist |
| persistent-id resolution of the model, its modules and its tokenizer | network, queueing, job errors arriving as `RuntimeError` |

So `remote="local"` proves the outbound path; the server-and-back backend
proves the serialization of the way home; a real server (M5) is still the only
proof of the rest. One consequence for M5 (**belief**): a public NDIF
deployment runs a clab document only if clab is installed server-side at the
client's version (law 5), so M5 targets a self-hosted server with clab and
nnterp installed (§12).

---

## 11. Format tightenings

Every proposal keeps the 31 documents valid exactly as written. Nothing here
lands without ratification.

### PLAN §6's four candidates

**1. Move `saves` to the root — rejected.** Before/after:

```json
{"steps": {"…": {}, "saves": {"iia": "iia.json"}}}
{"steps": {"…": {}}, "saves": {"iia": "iia.json"}}
```

The corpus form must keep working forever, so the move buys a second spelling
for the same thing and nothing expressible that was not before. Instead,
`saves` stays a reserved key of `steps` (proposal 6).

**2. One address string — rejected.**

```json
{"component": "attention_premix", "layers": 0, "heads": {"sweep": [[0], [1]]}}
"layers[0].attention_premix[heads=0]"
```

A string needs a grammar, a parser and an escaping story, and a sweep inside
one becomes a sweep over strings; the dict already validates key by key and
takes a sweep anywhere.

**3. Fewer spellings for `interventions` and `site` — rejected; keep all.**
`interventions` as a name (`"das"`), a list of names (`["steer", "restore"]`)
and an inline dict, and `site` as a name or an inline dict, are each used in
the corpus; compile normalizes them to inline records in about ten lines,
and the inline form is what keeps one-off definitions next to their use, as
the charter fixes.

**4. One explicit shape for `pos` — accepted, as a documented grammar with no
new spelling required.**

```json
-1
{"frame": "prompt", "index": -1}
```

```
pos  := int                                    # shorthand for {"index": int}
      | {"frame"?: "prompt" | "generated",     # default "prompt"
         "scope"?: {"variable": column} | {"segment": "role[n]" | "role" | "eos"},
         exactly one of "index": int | "span": [int, int] | "all": true}
```

`scope` is optional, not nullable: absent means the whole prompt (or the whole
continuation). Two combinations the grammar admits are refused at compile:
`segment: "role[n]"`/`"role"` in the generated frame, and `segment: "eos"` in
the prompt frame (§5).

Reason: the five forms are one shape already, and saying so makes every other
combination legal (a span inside a variable, `all` over the generated text) at
no cost, since compile reduces each to `(frame, scope, start, stop)` (§5).
Every corpus `pos` is an instance of the grammar.

### Mine

**5. A sweep may stand for any value that is not a name, a kind or a key,
outside `header` and `model`, and not on a data `path` or a `file_path` —
accepted, if ratified (question 8).**

```json
{"featurizers": {"rot": {"kind": "subspace", "k": {"sweep": [2, 4, 8]}}}}
```

Reason: compile finds sweeps by walking the document, so enumerating where
they may appear (PLAN §2 lists seven places) is more code than allowing them
everywhere a value is; names and kinds stay unsweepable because a sweep there
would change which code runs, not which value. `model` stays unsweepable
because a document has one engine and one session; data and bundle paths
because compile reads those files, and a sweep over them would be a sweep over
datasets, which nothing in the corpus asks for and whose row alignment is
undefined. The corpus's sweeps are all instances.

**6. Names: no dots, no reuse across steps, datasets and featurizers in one
scope, and `saves` is not a step name — accepted.** No before/after: these
documents are already conformant (verified over all 31). Reason: "thing, then
member" is unambiguous only under these three rules, and compile then
refuses an ambiguous document instead of guessing.

**7. A save's extension picks its writer — accepted.** `.json` for tables,
`.safetensors` for tensors and bundles, anything else refused. No before/after;
every corpus save conforms (verified). Reason: the filename is the only place a
document says how a value should look on disk, and one rule is easier to read
than a writer per kind.

**8. `layers` may be a list — optional.**

```json
{"component": "block_output", "layers": "all"}
{"component": "block_output", "layers": [0, 3, 7]}
```

Reason: `"all"` is already compiled to the list of every layer and stacked
layer-first, so a list costs one `isinstance` and gives a lens over chosen
layers in one forward. Take it or leave it; nothing in the corpus needs it.

---

## 12. Questions for the owner

1. **Shared unswept steps.** The design makes the swept *steps* lists of
   points and runs steps no sweep touches once (pos_sweep's counterfactual
   forward runs once, not three times). The numbers equal running each point
   as a whole experiment. The headers' wording ("a sweep point is a plan, and
   the root is three of them") describes the other arrangement: the root a list
   of whole plans, each re-running every step. *Default: shared steps, as
   designed; the header's sentence is read as what a point means, not how the
   plan is laid out.*
2. **Tensors in the plan.** A bundle a document names by `file_path` is read at
   compile and carried in the plan as tensors, since the server cannot read the
   client's file. Law 2 lists "strings, numbers, tuples, dicts". *Default:
   allowed, for document-named input files only, alongside their header; the
   client decides nothing from them.*
3. **The SAE fixture.** `sae_feature_ablation.json` loads
   `documents/artifacts/sae.safetensors`, "a random 16->32 dictionary, a
   fixture", which does not exist and which no document writes. *Default: M4
   commits a 20-line seeded generator under `tests/fixtures/` and the file it
   writes (≈ 4 KB), with a header for the tiny Llama's `block_output` layer 0,
   width 16, 32 latents.*
4. **Gate initialization, anneal schedule, early-stop restore.** `dbm.json`
   leaves them open, and its measured "4, 2, 1 and 0 of 16 units kept" came from
   another implementation, so it cannot be a target. *Default: θ₀ = +1 on every
   unit (soft mask 0.73 at T = 1, every unit kept when scored); temperature
   linear from `start` to `end` over epochs; the best epoch's featurizer is
   restored when the fit ends (judged as question 7 says).*
5. **One `--out` directory for two documents.** das.json and pca_harvest.json
   both write into `documents/artifacts`; their files coexist and the second
   `manifest.json` replaces the first. *Default: allow it. Alternative: name the
   manifest `manifest.<document>.json` when `--out` is given, which keeps both
   but changes the one-manifest-at-the-root rule.*
6. **M5's server.** Law 5 means the server has clab installed at the client's
   version, which a public NDIF deployment does not. *Default: M5 runs against a
   self-hosted NDIF (the existing bippu or hakone stacks) with clab and nnterp
   installed from the branch under test.*
7. **Which split early stopping selects on.** `das.json` and `dbm.json` name an
   early-stop metric and an eval substitution, not which data the stop is
   judged on. Selecting the best epoch on the eval split (`test`) and then
   saving that split's score as `fit.iia` (`held_out_iia.json`) reports the
   score the model was selected on, which is optimistically biased. *Default:
   judge early stopping on the training data in scoring mode, and run the eval
   substitution once, after the fit (§9). Alternative: judge on the eval split,
   as a validation set, and accept that `held_out_iia.json` is not held out.*
8. **How far proposal 5 reaches.** "A sweep may stand for any value" would let
   `model.revision`, `model.dtype`, `attn_implementation`, `data.path`,
   `file_path`, `epochs` and `batch_size` sweep; a swept `model` contradicts
   one engine and one session per document, and a swept data or bundle path
   is a sweep over files compile must read. *Default: no sweep under `header`
   or `model`, none on a data `path` or a `file_path`, none on a name or kind;
   everything else (including `epochs`, `batch_size`, `lr`) may sweep.*
9. **The model says the variable twice.** In the generated frame, `said`
   occurring twice in what a row generated is a fact about the model's output;
   refusing it would be an exception that depends on model output, which
   `generated_answer.json`'s header rules out ("a reported result rather than
   an exception that ends the run"). *Default: report it per row as
   `alignment_ambiguous`, like `alignment_missing`. Alternative: take a fixed
   occurrence (the first). In the prompt frame, where the data decides, an
   ambiguous variable stays refused.*
10. **The output layout and the charter's "the plan written out as its own
    hierarchy".** Files are laid out by the paths the document's `saves` give
    (plus `<label>=<value>/` per sweep coordinate), and the plan's step
    hierarchy lives in `manifest.json`, which points at each file. Laying files
    out as `<step>/<file>` would break `das_apply.json` and `pca_control.json`
    (their `file_path`s name the root of `--out`) and `two_observes.json`'s
    `compare/…` grouping (§13.1). *Default: accept the deviation explicitly, and
    reword the charter line as "the output is a directory: each saved value at
    the path its document gives, and a manifest at the root that mirrors the
    plan".*

---

## 13. Where PLAN.md is wrong or underspecified

1. **Save paths are relative to the output root, not to a step's directory**
   (§5 output proposal). `das_apply.json` and `pca_control.json` say to run their
   source documents with `--out documents/artifacts` and then load
   `documents/artifacts/rot.safetensors` and `…/pca.safetensors`; under "the
   path within the step's directory" those files would land at
   `documents/artifacts/fit/rot.safetensors` and `…/basis/pca.safetensors`, and
   both documents would fail. `two_observes.json`'s `compare/…` paths read the
   same way. Correction: §8 (root-relative; sweep coordinates as `<label>=<value>/`
   directories).
2. **"A sweep point is a whole experiment and the root is a list of them"**
   (§2's reading) contradicts the owner's fixed structure, "swept steps as lists
   of points" (charter, and PLAN §3). The design follows the charter and keeps
   the header's observable meaning (§7, question 1).
3. **"The compiler checks" tokenization-dependent facts** (§2: the
   attention-pattern length check). Under laws 3 and 4 the tokenizer is reached
   through the model inside the session, so those checks run in
   `steps.preflight`, in the session, before any forward. The PCA rank check
   ("four vectors have no rank-4 PCA; the compiler says so") does stay at
   compile when the read is rectangular, since rows × width is known from the
   data alone; for a ragged read it moves to preflight too.
4. **Part of the way home can be tested before M5.** A 25-line stand-in
   backend (§10b) runs the deserialized tracer and round-trips its result
   through `torch.save`/`torch.load`. It is the design's model of a server, not
   nnsight's server path, so it does not prove the return trip; it does test the
   serialization of a filled plan, and it found that a slot holding a view ships
   its whole storage (11× here), hence the clone rule. M1 should carry it; the
   rest of the way home stays a belief until M5, as PLAN says.
5. **`execute`, `forward`, `generate` "take a `batch_size`"** (§3). Only
   `execute` and `steps.run` do; `forward` and `generate` receive one batch.
   Inside a fit, the fit's own `batch_size` is the minibatch.
6. **The protocol's `resolve(address)`** (§5, `engines/base.py`) is not part of
   the protocol: `steps.py` never calls it; each engine reaches components
   privately. And the plan needs no `Slot` class: a slot is a dict (§1).
7. **The slow-tokenizer note is less "future" than it reads.** Both pinned
   checkpoints load fast tokenizers under transformers 5.19 (verified), and the
   growing-prefix alignment the note anticipates is needed in M3 anyway, for
   generated text; slow-tokenizer prompts then reuse it (§5).
8. **Data must travel inside the plan**, which PLAN does not say: law 3 forbids
   a loaded dataset in the session and the server has no `data/`, so compile
   carries each dataset's rows (pruned to the columns the document uses).
9. **`fit.rot` inside the body is underspecified** (§2). The objective term
   `mask.mask` only means "the L1 of the mask being trained" if, inside a fit,
   a trained featurizer's declared name and `fit.<name>` are one object; §6
   makes that the rule.
10. **`token_form: bare` is a no-op on the pinned Llama**, which §2 does not
    anticipate: its SentencePiece tokenizer spells a text's first word with the
    prefix space, so `"Friday"` and `" Friday"` are both `[28728]` (verified).
    The design implements the header's definition, "as the first word of a text
    is" (encode the stripped text as a text of its own), which differs on GPT-2
    (`Friday` vs `ĠFriday`) and coincides here; `readout.json`'s two
    probabilities are therefore equal on this checkpoint, and the M2 snapshot
    says so.
11. **M1's "bit-equal between engines" needs one fact PLAN does not state**: the
    hooks engine must pass `position_ids = cumsum(mask) − 1` (clamped) on
    left-padded batches, as nnsight does; without it the engines disagree on the
    padded rows (verified, §3).
12. **`pyproject.toml` does not build as scaffolded** once a `clab/` package
    exists: setuptools' flat-layout discovery sees `data/` and `documents/` and
    refuses. M1 adds `[tool.setuptools.packages.find] include = ["clab*"]`
    (verified).
13. **Module layout** (§5): `document.py`, `plan.py`, `state.py`, `reduces.py`
    and `engines/base.py` are merged or dropped and `interventions.py` added,
    each for the reason in §2.
