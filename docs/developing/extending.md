---
title: Extending clab
one_liner: "Adding an op, a featurizer kind, a component or a step kind: the decorator, the signature as the document's contract, where the tables come from, and the tests to add."
tags: [developing, extending, ops, featurizers, components]
related: [docs/developing/architecture.md, docs/developing/engines.md, docs/developing/testing.md, docs/developing/contributing.md, docs/usage/ops.md, docs/usage/featurizers.md]
sources: [clab/ops/__init__.py, clab/format/registry.py, clab/featurizers/__init__.py, clab/compile/uses.py, clab/format/address.py, clab/engines/standard.py, clab/plan.py, clab/steps/__init__.py, clab/compile/__init__.py, clab/docs.py]
---

# Extending clab

clab's vocabulary is read off the code: an op's signature is what a document may write for it, a featurizer kind's constructor is its fields, and [FORMAT.md](../../FORMAT.md) is generated from both. So extending clab means adding a function or a class in the right place, registering it, and regenerating FORMAT.md. A new op, featurizer kind, component or step kind is a format change: propose it with before-and-after JSON, and it lands when the owner ratifies it ([contributing.md](contributing.md#changing-the-format)). Adding an engine is [engines.md](engines.md).

## The signature is the document's contract

`clab/format/registry.py` `check` binds a document's arguments to the signature of the function they go to (an op, a featurizer kind, an engine's constructor) before anything runs:

- A parameter's name is the argument's spelling, with a trailing underscore dropped (`with_` is `"with"`). An unknown key is refused naming the valid ones, and a parameter without a default is required.
- A parameter's annotation is how a document fills it. `Value` (a named value or a number), `Tokens` (a column resolved through the value's vocabulary), `Text` (a column read as text) and `Featurizer` (a featurizer's name) are resolved by compile; any other annotation is a literal type, checked strictly.
- Parameters the run supplies are never the document's. For an op, those are `clab/ops/__init__.py` `SUPPLIED` (`engine`, `original`, `dims`, `layers`, `eos`, `rows`, `positions`). For a kind, they are `width`, `bundle`, `header`, `labels` and `kept`, and to its `apply` the `dims` and `layers` of the value it meets. Name one in the signature to receive it.
- The docstring's text is what `clab describe` and FORMAT.md print beside it.

## Adding an op

1. Write a plain function over tensors in its group's module (`clab/ops/metrics.py`, `reductions.py`, `edits.py` or `transforms.py`), its first parameter `of`, the value it acts on. The function's `__name__` is the op's name in a document.
   - **A builtin's name shadows the builtin in the whole module.** `clab/ops/reductions.py` `sum`, `min` and `max` already do, and so do `clab/ops/transforms.py` `abs` and `pow`. Call the builtin there as `builtins.<name>`, or pick another name.
2. Decorate it with `@op(group, ...)` (`clab/ops/__init__.py` `op`), declaring its axes and checks:
   - `needs`, the axes the value must carry; `removes` and `adds`, the axes it takes away and puts on.
   - `layout=`, a function of the value's axes, the arguments and the featurizer kind, where the axes depend on the arguments.
   - A reduction that takes `over` declares `layout=reduced` (`clab/ops/reductions.py` `reduced`). It removes the axes `over` names, by default `rows` and `pos`, and refuses a reduction that removes none. Name `dims` in the signature to receive the value's axes.
   - `bundle=<kind>`, marking a value a featurizer of that kind loads.
   - `check=`, raising on arguments the types accept but the op cannot mean.
   - `positionwise=True`, where its value at each position comes from that position's vector alone, with the same bits however many other positions the value holds. A generation then runs it at each call it reads ([usage/generate.md](../usage/generate.md#a-read-at-every-call)). That holds for a gather, a softmax over one axis, a rank, and arithmetic IEEE rounds exactly (`+`, `*`, `/`, `sqrt`, a comparison).
   - On a reduction, `positionwise=True` says each position's result over the feature axes it removes has the same bits whatever the value's shape. `max`, `min`, `median` and `count` declare it; `mean`, `sum` and `std` do not. A generation runs it at each call only while the result keeps `rows` and `pos`, so a reduction over `pos` runs after the generation.
   - Leave `positionwise` off an op that multiplies matrices or sums over an axis. Their reduction order follows the value's shape, so on a call's `[rows, 1, ...]` they can differ in the last bits from the same op on the whole read. Without it, the op runs after the generation.
3. **Add the function to the ordered list at the end of `clab/ops/__init__.py`.** `@op` declares the op; the list registers it. `import clab` fails with `AssertionError: every declared op is listed once` until the list has it. Within a group, that list's order is the tables' order. The prose FORMAT.md prints names the edits group from that list (`clab/ops/__init__.py` `listed`, written into `ABOUT` and the `Intervention` docstring in `clab/plan.py`), so a new edit joins it there with nothing written by hand.

A minimal edit, added to the edits module and listed after `clamp`. It is for illustration only: the format already spells negation as the transform `neg`, and a second spelling does not land ([contributing.md](contributing.md#changing-the-format)).

```python
@op("edits")
def negate(of: Tensor) -> Tensor:
    """`-of`."""
    return -of
```

`clab describe ops` then lists `negate()  [keeps the layout]` with its docstring, an entry's `{"op": "negate"}` writes the negated value back on both engines, and `{"op": "negate", "factor": 2}` is refused when the plan is built:

```text
steps.flipped.interventions.neg.ops[0]: 'negate' takes no 'factor'; its arguments: none
```

To see such a refusal without a run, compile a scratch document from Python: `Plan("scratch.json").compile()` raises it ([running.md](../usage/running.md#check-before-you-run)).

A metric over next-token logits declares the vocabulary axis it scores:

```python
@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def top_prob(of: Tensor) -> Tensor:
    """The probability of the most likely next token."""
    return of.float().softmax(-1).amax(-1)
```

A tuple inside `needs` is any one of its axes. `clab/ops/__init__.py` `VOCABULARY` is `vocab_size` or a probe's `classes`, so this metric scores logits and probe scores alike, and `removes=VOCABULARY` takes away whichever the value has.

What the group gives it, with no code of its own:

- An edit runs in float32 and is rounded once to the value's dtype (`clab/ops/__init__.py` `call`). Only an edit is upcast.
- A metric or a reduction receives the value in the model's dtype and casts it itself, as `of.float()` does above, since the format promises float32 from both ([usage/ops.md](../usage/ops.md#the-float32-rule)).
- An op whose result has no `rows` axis is refused inside an entry, which sees one batch.
- An op with an `original` parameter runs only inside an entry (`clab/compile/arguments.py` `layout`).
- A missing value arrives as NaN and should leave as NaN: compute plainly, with no `nansum` or `nan_to_num`.

What the groups mean for a document is [usage/ops.md](../usage/ops.md).

## Adding a featurizer kind

1. One module under `clab/featurizers/` holding one class that subclasses `Kind` (`clab/featurizers/__init__.py` `Kind`). The class's name in lower case is the kind, and subclassing registers it in `KINDS`.
2. Import the module in the line at the end of `clab/featurizers/__init__.py`. Until then the class never loads, and a document naming it (`"kind": "shift"` for the kind below) is refused: `unknown featurizer kind 'shift'; valid: ...`. That line's order is the tables' order.
3. Its constructor is the document's fields. A field annotated `Value` is a number or the name of a value a step makes, as an op's is. Compile holds the name to one vector of the featurizer's width, made before its first use (`clab/compile/uses.py` `check_named_values`). The run passes the constructor that value where it makes the instance (`instance`), as a constant's `init` is.
4. Its class attributes say what its bundle holds, what a fit trains, its coordinates' axis, its header fields, what a drawn one needs, its scalars and which fields count directions; their meanings are in [FORMAT.md](../../FORMAT.md#featurizer-kinds). Its methods `encode(h)`, `replace(h, z, new)` and `apply(h, ...)` serve `encode`, `decode` and `apply`; define only those it has, and it serves exactly those.
5. Hold tensors as plain attributes and move them to the value where they meet it (`self.b.to(h)`). The three methods run with autocast off, and a missing (NaN) vector goes in as zeros and comes out missing, without code of the kind's own (`clab/featurizers/__init__.py` `exact`, `Kind.call`). An `apply` whose every element of the result is made from that element of the value alone declares `elementwise = True`, as a gate's does. Then a missing element, not its whole vector, goes in as zero and comes out missing. That matters at a `key` axis, NaN past a short row's own keys, where a vector-wise rule would write the whole row back missing.

A minimal kind, which a fit trains and saves as a bundle with clab's header like any other:

```python
class Shift(Kind):
    """An offset `b` (starting at zero) added to every vector: its one op is `apply`."""
    saves = trains = ("b",)

    def __init__(self, width: int, file_path: str | None = None, bundle: dict[str, Tensor] | None = None) -> None:
        self.b = bundle["b"] if bundle is not None else torch.zeros(width)

    def apply(self, h: Tensor) -> Tensor:
        return h + self.b.to(h)
```

## Adding a component

A component is a row of `clab/format/address.py` `COMPONENTS`: its `meaning`, whether it is `per_layer`, its `dims`, and whether it is a residual-stream value `readout` may project (`stream`).

- **Add a component only under a name nnterp's `support()` lists.** The engines reach a component only through nnterp, and its name is nnterp's name for the value. Check that nnterp's `StandardizedTransformer.support()` lists the name before adding the row. A name it does not answer for is refused for every document, whether it uses the component or not: the hooks engine refuses it when it loads, the nnterp engine before the run's first step (`clab/engines/standard.py` `unserved`: `nnterp's support() does not name <name>, so nothing says whether it serves <component>; ...`).
- **Its `dims` are nnterp's layout of it**: `batch` first, then the token axis `seq`, or `query` on an attention pattern. A site's `select` may narrow every other axis.
- **On nnterp and ndif**, the engine reads and sets it on the envoy that hosts it (`clab/engines/standard.py` `envoy`). It asks nnterp's `support()` whether it is served and nnterp's `rank` where it falls.
- **A value nnterp serves on the attention or the MLP** is also listed in `clab/engines/standard.py` `BELOW`, under its host. A mixture of experts' values are the MLP's, nnterp's `Moe` being one.
- **On hooks**, the engine resolves at load the module boundary nnterp's key for it names. It refuses by name one nnterp reads inside a module's forward or among a module's call arguments.
  - The load's two-token forward must reach the boundary; `clab/engines/hooks.py` `order` refuses by name one it does not reach.
  - The boundary's tensor must have the component's rank, one less where nnterp holds the value flat over tokens, or three for a four-axis per-head layout. A boundary of any other rank is served as it holds the value, without a refusal.
- **An axis that is a size of the model** joins `clab/engines/standard.py` `SIZES`, which `sizes` reads off nnterp (`num_experts` and `top_k` off its `Moe`, absent on a model without one).

## Adding a step kind

A step kind is the widest change: the kind is dispatched in each stage.

- `clab/plan.py`: a `Record` subclass with `kind: Literal["<kind>"]`, its entry in `KINDS`, and a tagged member of the `Step` union. Where the step has a value of its own, the subclass is a `Labelled` one, whose record carries that value's layout and `axes`. `serving` says which of its points run.
- `clab/compile/__init__.py` `compile_step`: the function that compiles it, writing its records' layout and the sweep axes each value depends on.
- `clab/compile/preflight.py` `prepare`: what it checks with the model. A kind without a branch of its own there, as `op`, `fit` and `load` have, is prepared as a forward, its serving points' `schedule` written by `clab/compile/schedule.py` `write_schedule`.
- `clab/steps/__init__.py` `KINDS`: the function that runs it, called as `(engine, plan, name, step, state, batch_size, findings, cost)`, adding the model calls it makes and the tokens they feed to `cost`; a fit's body runs through the same table.
- `clab/plan.py` `calls_model`: whether a run loads the shell's weights. A new kind that calls the model and is not listed there runs on a meta shell.
- `clab/steps/__init__.py` `NO_MODEL`: the kinds `measure_cost` times without synchronizing the model's devices.
- The rest of the code that branches on a step's kind: `grep -rnE "\.kind (==|!=|in|not in)|NO_MODEL" clab/`.

`clab describe steps` and FORMAT.md list `clab/plan.py` `KINDS` with each record's docstring.

## The tests to add, and FORMAT.md

| change | tests |
|---|---|
| an op | `tests/test_ops.py`: the op on plain tensors. Its two listing tests, `test_every_op_is_listed_once_in_order_under_its_own_name` and `test_each_op_declares_its_group_and_bundle`, fail until the new op is in their lists. A metric or reduction joins `tests/test_findings.py` `test_every_metric_and_reduction_is_float32_on_a_bf16_model_and_an_edit_stays_bf16` in two places: a step in its `steps`, and the step's name in its `names` |
| a featurizer kind | `tests/test_featurizers.py` `every_kind`, so the device and autocast tests cover it; `tests/test_featurizer_ops.py`, whose test of each kind's ops lists every kind and fails until updated; a kind with both `encode` and `decode` joins the parameters of its identity-to-the-bit test |
| a component | a corpus document that uses it, run on both engines; a row in `tests/test_engines.py`, since a component means one thing on every engine |
| a step kind | a corpus document that uses it, run on both engines |
| a corpus document | pins for its numbers and its compiled form, and the README's corpus count, which `tests/test_plan.py` checks. One that trains or loads a featurizer is named in `tests/conftest.py` `FIT`, one that generates in `GENERATE`, and one the hooks engine refuses in `INSIDE`; any other runs as a forward in `tests/test_documents.py` |

A new op also changes the docs ([contributing.md](contributing.md#what-a-pull-request-carries)). It gets a when-to-use bullet in its group's section of `docs/usage/ops.md`, and joins every list of its group's ops under `docs/`, for a reduction in `docs/usage/ops.md` and `docs/reference/glossary.md`. Grep a sibling's name to find them.

An op, a kind, a component or a step kind changes FORMAT.md, which `clab/docs.py` `markdown` renders from the registries: regenerate it with `uv run --no-sync clab describe --markdown`, or `tests/test_format.py` fails ([contributing.md](contributing.md#changing-the-format)). Pinning a new document is [testing.md](testing.md); what lands with the change is [contributing.md](contributing.md).
