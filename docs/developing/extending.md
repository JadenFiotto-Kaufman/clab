---
title: Extending clab
one_liner: "Adding an op, a featurizer kind, a component or a step kind: the decorator, the signature as the document's contract, where the tables come from, and the tests to add."
tags: [developing, extending, ops, featurizers, components]
related: [docs/developing/architecture.md, docs/developing/engines.md, docs/developing/testing.md, docs/developing/contributing.md, docs/usage/ops.md, docs/usage/featurizers.md]
sources: [clab/ops/__init__.py, clab/format/registry.py, clab/featurizers/__init__.py, clab/format/address.py, clab/engines/standard.py, clab/plan.py, clab/steps/__init__.py, clab/compile/__init__.py, clab/docs.py]
---

# Extending clab

clab's vocabulary is read off the code: an op's signature is what a document may write for it, a featurizer kind's constructor is its fields, and [FORMAT.md](../../FORMAT.md) is generated from both. So extending clab means adding a function or a class in the right place, registering it, and regenerating FORMAT.md. Adding an engine is [engines.md](engines.md).

## The signature is the document's contract

`clab/format/registry.py` `check` binds a document's arguments to the signature of the function they go to (an op, a featurizer kind, an engine's constructor) before anything runs:

- A parameter's name is the argument's spelling, with a trailing underscore dropped (`with_` is `"with"`). An unknown key is refused naming the valid ones, and a parameter without a default is required.
- A parameter's annotation is how a document fills it. `Value` (a named value or a number), `Tokens` (a column resolved through the value's vocabulary), `Text` (a column read as text) and `Featurizer` (a featurizer's name) are resolved by compile; any other annotation is a literal type, checked strictly.
- Parameters the run supplies are never the document's: for an op, those in `clab/ops/__init__.py` `SUPPLIED` (`engine`, `original`, `dims`, `eos`, `rows`, `positions`); for a kind, `width`, `bundle`, `header` and `labels`. Name one in the signature to receive it.
- The docstring's text is what `clab describe` and FORMAT.md print beside it.

## Adding an op

1. Write a plain function over tensors in its group's module (`clab/ops/metrics.py`, `reductions.py`, `edits.py` or `transforms.py`), its first parameter `of`, the value it acts on.
2. Decorate it with `@op(group, ...)` (`clab/ops/__init__.py` `op`), declaring its axes: `needs` (axes the value must carry), `removes`, `adds`; or `layout=`, a function of the value's axes, the arguments and the featurizer kind, when they depend on the arguments. `bundle=<kind>` marks a value a featurizer of that kind loads; `check=` raises on arguments the types accept but the op cannot mean.
3. Add the function to the ordered list at the end of `clab/ops/__init__.py`. The decorator alone does not list it, and `import clab` fails with `AssertionError: every declared op is listed once` until the list has it. Within a group, that list's order is the tables' order. A new edit is also named, by hand, in two docstrings that FORMAT.md prints: the edits group in `clab/ops/__init__.py` `ABOUT` and in `clab/plan.py` `Intervention`. Add it to both; `tests/test_format.py` cannot see that they are short.

A minimal edit, added to the edits module and listed after `clamp`:

```python
@op("edits")
def negate(of: Tensor) -> Tensor:
    """`-of`."""
    return -of
```

`uv run clab describe ops` then lists `negate()  [keeps the layout]` with its docstring, an entry's `{"op": "negate"}` writes the negated value back on both engines, and `{"op": "negate", "factor": 2}` is refused when the plan is built:

```text
steps.flipped.interventions.neg.ops[0]: 'negate' takes no 'factor'; its arguments: none
```

A metric over next-token logits declares the vocabulary axis it scores:

```python
@op("metrics", needs=(VOCABULARY,), removes=VOCABULARY)
def entropy(of: Tensor) -> Tensor:
    """The entropy of the next-token distribution, in nats."""
    logp = of.float().log_softmax(-1)
    p = logp.exp()
    return -(p * logp.masked_fill(p == 0, 0.0)).sum(-1)
```

A tuple inside `needs` is any one of its axes. `clab/ops/__init__.py` `VOCABULARY` is `vocab_size` or a probe's `classes`, so this metric scores logits and probe scores alike, and `removes=VOCABULARY` takes away whichever the value has.

What the group gives it, with no code of its own: an edit runs in float32 and is rounded once to the value's dtype (`clab/ops/__init__.py` `call`); an op whose result has no `rows` axis is refused inside an entry, which sees one batch; an op with an `original` parameter runs only inside an entry (`clab/compile/arguments.py` `layout`). Only an edit is upcast. A metric or a reduction receives the value in the model's dtype and casts it itself, as `of.float()` does above, since the format promises float32 from both ([usage/ops.md](../usage/ops.md#the-float32-rule)). A missing value arrives as NaN and should leave as NaN: compute plainly, with no `nansum` or `nan_to_num`. What the four groups mean for a document is [usage/ops.md](../usage/ops.md).

## Adding a featurizer kind

1. One module under `clab/featurizers/` holding one class that subclasses `Kind` (`clab/featurizers/__init__.py` `Kind`). The class's name in lower case is the kind, and subclassing registers it in `KINDS`.
2. Import the module in the line at the end of `clab/featurizers/__init__.py`. Until then the class never loads, and a document naming it (`"kind": "shift"` for the kind below) is refused: `unknown featurizer kind 'shift'; valid: subspace, gate, pca, sae, linear, lora`. That line's order is the tables' order.
3. Its constructor is the document's fields. Its class attributes say what its bundle holds, what a fit trains, its coordinates' axis, its header fields, what a drawn one needs, its scalars and which fields count directions; their meanings are in [FORMAT.md](../../FORMAT.md#featurizer-kinds). Its methods `encode(h)`, `replace(h, z, new)` and `apply(h, ...)` serve `encode`, `decode` and `apply`; define only those it has, and it serves exactly those.
4. Hold tensors as plain attributes and move them to the value where they meet it (`self.b.to(h)`). The three methods run with autocast off, and a missing (NaN) vector goes in as zeros and comes out missing, without code of the kind's own (`clab/featurizers/__init__.py` `exact`, `Kind.call`).

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

A component is a row of `clab/format/address.py` `COMPONENTS`: its `meaning`, whether it is `per_layer`, its `dims` and whether it is a residual-stream value `readout` may project (`stream`). Its name is nnterp's name for the value, and its `dims` are nnterp's layout of it (`batch` first, the token axis `seq`, or `query` on an attention pattern); a site's `select` may narrow every other axis.

The engines reach a component only through nnterp, so nnterp must serve that name. The nnterp and ndif engines read and set it on the envoy that hosts it (`clab/engines/standard.py` `envoy`), ask nnterp's `support()` whether it is served and nnterp's `rank` where it falls; one nnterp serves on the attention or the MLP is also listed in `clab/engines/standard.py` `BELOW`, under its host. The hooks engine resolves, at load, the module boundary nnterp's key for it names, and refuses by name one nnterp reads inside a module's forward. A name nnterp does not have breaks both: the hooks engine fails to load for every document, and the nnterp engine counts it as served, then fails at `rank` when a document uses it. Check that nnterp's `StandardizedTransformer.support()` lists the name before adding the row.

## Adding a step kind

A step kind is the widest change: the kind is dispatched in each stage.

- `clab/plan.py`: a `Record` subclass with `kind: Literal["<kind>"]`, its entry in `KINDS`, and a tagged member of the `Step` union; `dims_in` for the layout of its value.
- `clab/compile/__init__.py` `step`: the function that compiles it and registers the names it produces.
- `clab/compile/preflight.py` `prepare`: what it checks with the model (any kind but `op` and `fit` is prepared as a forward).
- `clab/steps/__init__.py` `KINDS`: the function that runs it, called as `(engine, plan, name, step, state, batch_size, found)`; a fit's body runs through the same table.
- The rest of the code that branches on a step's kind: `grep -rn 'kind ==' clab/`.

`clab describe steps` and FORMAT.md list `clab/plan.py` `KINDS` with each record's docstring.

## The tests to add, and FORMAT.md

| change | tests |
|---|---|
| an op | `tests/test_ops.py`: the op on plain tensors; the two tests that list every op and its group fail until the new one is in them. A metric or reduction joins the list in `tests/test_findings.py` `test_every_metric_and_reduction_is_float32_on_a_bf16_model_and_an_edit_stays_bf16`, which names the ops it checks |
| a featurizer kind | `tests/test_featurizers.py` `every_kind`, so the device and autocast tests cover it; `tests/test_featurizer_ops.py`, whose test of each kind's ops lists every kind and fails until updated; a kind with both `encode` and `decode` joins the parameters of its identity-to-the-bit test |
| a component or a step kind | a corpus document that uses it, run on both engines |
| a corpus document | pins for its numbers and its compiled form, and the corpus count in `tests/test_plan.py` |

An op, a kind, a component or a step kind changes FORMAT.md, which `clab/docs.py` `markdown` renders from the registries: regenerate it with `uv run clab describe --markdown` from the repo root, since it writes `FORMAT.md` into the current directory, or `tests/test_format.py` fails. Pinning a new document is [testing.md](testing.md); what lands with the change is [contributing.md](contributing.md).
