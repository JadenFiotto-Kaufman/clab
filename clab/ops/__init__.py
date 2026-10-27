"""The one computation vocabulary: every op is a plain function over tensors.

`@op(group, ...)` declares a function as an op under its own name: its group, the axes the input must carry
(`needs`), the axes it removes (`removes`), the axes it appends (`adds`), when its value is a bundle a featurizer
loads, that featurizer's kind (`bundle`), and a rule its arguments keep beyond their types (`check`). `OPS` lists
them, in the order at the end of this file. An op whose axes depend on what it is given declares them as a function
instead (`layout`), as `encode` does for its featurizer's kind. Compile reads the axes to lay out every value. A
parameter's annotation says how a document fills it (`Value`, `Tokens`, `Featurizer`, or a literal); `engine`,
`original` and `dims` are filled by the run, never by a document; a trailing underscore is dropped from a
parameter's document spelling (`with_` is `"with"`).
"""

import inspect
from typing import Any, Callable

from torch import Tensor

from clab.format import registry
from clab.format.registry import Featurizer, Text, Tokens, Value  # noqa: F401  (each read as ops.<name>)

# `<the edits group>` stands for the edits group, written in from `DECLARED` once the ops are registered (the end of this file).
ABOUT = """An op runs in two places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a value earlier steps produced;
- inside an entry of `interventions`, `"ops": [...]`, on the entry's selection at its site, in order.

Inside an entry an op sees one batch, so it must act per example: one that aggregates over rows is
refused there (make it a step after the forward). An entry's edits act on the live value as the forward
reaches the site, and the value after its last edit is written back; the ops after that run on the value
once it is taken off the model (an entry nothing uses is not taken off, so they do not run).

A row a read beyond the prefill did not reach is missing all the way down, and so is a position past a row's
own where rows hold different numbers of them: NaN, every metric and edit of it NaN, `argmax` and `top_k`
giving it id -1 (`top_k` NaN `probs` or `values` and no token text). Reductions skip it: `mean` is the mean over the
rows and positions that have a value, and `pca` is fitted on them. A featurizer's math sees none of it: a
missing vector goes in as zeros and comes out missing.
An attention pattern's keys past a row's own are missing the same way, NaN, in a read and to a write's ops:
an edit that mixes a vector's entries (`normalize`, `renormalize`, `project_out`) takes the entries that are
there, so a row's own keys are edited as the row has them, and its tail stays NaN; `argmax` and `top_k` over
`key` rank a row's own keys, a vector missing only where it has none.

Ops come in four groups: **metrics** score each example (one number per row and position: `match` is 0 or
1, and the accuracy is a `mean` step after it; `contains` scores a detokenized generation's text);
**reductions** remove axes, by default the rows and positions, a `layers` axis kept (`over` names them;
over `pos`, one value per row); **edits** (<the edits group>) keep the value's layout and change its contents, and inside
an entry are what writes back; **transforms** turn each example's value into another of a different layout (logits from
a residual stream, coordinates from a value, tokens from logits, text from ids). An edit comes before any
transform, metric or reduction of its entry, or, after an `encode`, before the `decode` that closes it.

Metrics and the reductions `mean`, `sum` and `min` compute in float32 whatever the model's dtype, and return
float32 (`pca` works in double and returns the value's dtype), so a bfloat16 model's mean, a cosine near 1 or a probability near 1 is not
rounded to its few bits. An edit computes in float32 too, the value and every operand upcast, and rounds once,
to the value's dtype, since an edit writes back; a value read from the model keeps the model's dtype. A rank
(`match`'s most likely token, `argmax`, `top_k`) is the same either way: bfloat16 to float32 is exact. `top_k`
over a vocabulary (`vocab_size`, a probe's `classes`) gives `probs`, a float32 softmax, and over another axis
`values`, the ranked values in the value's own dtype.

A featurizer is reached by three ops. `encode` turns the value into its coordinates in the featurizer's space
(`select: {"features": <selector>}` keeps some of them), `decode` (naming the same featurizer) turns them
back: within an entry each `decode` closes the most recent `encode` not yet closed, which must be of the same
featurizer, and the ops between them act on the coordinates. An operand there is in the same coordinates (a
value read through the same encodes, or turned into them by `encode` steps, with the same `select`s, or a
number), and one in other coordinates is refused by name, whatever its width. `decode` writes the change of
coordinates back as a delta, so `[encode x, decode x]` is the identity to the bit, an op that changes no
coordinate changes nothing, and a sparse autoencoder's error term survives; a `select`ed coordinate is
scattered back and the rest gain a change of exactly zero (encoding the result again gives them back to float
rounding, about 1e-9, not to the bit). Brackets nest: `[encode a, encode b, <edits>, decode b, decode a]`
edits `b`'s coordinates of `a`'s, `b` as wide as `a`'s coordinates. An entry that writes back with an `encode`
still open, a `decode` with no `encode` open, one of another featurizer than the open one, and an `encode` or
`apply` of a featurizer inside its own `encode`, are refused by name. `apply` is the featurizer's own version
of the value (a subspace's projection, a dictionary's reconstruction without its error term, an adapter's
adapted value): an edit, which writes back. An entry that stops after `encode` reads coordinates. As a step,
`encode` and `apply` take a read (an entry's value, at its one site; an op's value is refused by name), and
`decode`, the close of an entry's `encode`, is no step.

An operand (a `Value` argument that names a value) meets the value the op acts on at one point: inside an
entry, the entry's selection at each layer the entry is made at in turn (in the coordinates of the encodes open
around the op); as a step, `of`, at the layers its `layers` axis is labelled with, or at the one layer a value
without that axis was read at. One rule says what it may be, and compile refuses the rest by name (preflight,
where a label waits on the model's sizes or the rows' positions): an operand may leave out any axis of the value
it meets, which it then broadcasts over (a singleton axis stands for each where the value has one: a direction
over the rows and positions, a per-row dose over the positions and features, a `[rows, pos]` factor over
`hidden_size`, a per-head scalar over `head_dim`); every axis it has is the value's, in the value's order, with
the same label: `layers` by model layer (a `layers` axis it has covers the layers the op acts at and is aligned
by id: at layer L the op takes its slice labelled L, whatever order either lists its layers in, so a `kl` step
between lenses at `[0, 1, 2]` and `[2, 1, 0]` pairs layer with layer, and an entry at layer 3 takes slice 3; an
op at no layer, on a model's own value or one reduced over its layers, takes no layered operand), `rows` by
dataset (an op takes one dataset's rows, its columns' too), `pos` by the read whose positions both hold, else by
width, as many positions on every row (a read whose rows hold different numbers of positions pairs with no
other read's, but an earlier entry of its own forward, row by row), an axis a site selects (`num_heads`, ...)
by the indices kept, in order (one head does not broadcast over several, and `[1, 4]` would trade places with
`[4, 1]`), `features` by the encodes it is in (the same featurizers and `select`s: an operand inside a bracket
is encoded as the bracket is). A number, or a named value of one number (a `mean` over every axis), broadcasts
in any coordinates. An adapter's input (`apply`'s `with`) is held to its rows and positions alone: its features
are the adapter's width.

An argument is filled by a document as its type says: `Value` is a named value or a number; `Tokens` a column (a
`Column` reference: the string, or the object with `strip` and `lower`) resolved through the value's own
vocabulary, the tokenizer's against logits or a probe's classes against its scores (a metric scores a
`vocab_size` or a `classes` axis); `Text` a column read as text; `Featurizer` a featurizer's name; anything else
is a literal. `renormalize`'s reference is the value at the site before the step's writes (inside an `encode`,
its coordinates). An operand may name an earlier entry of the same forward, at this entry's site or at one the
forward reaches before it (`patched.premix` in `patched`): it is that entry's value in this batch, at its
positions.

A metric lines an answer up with the read it scores, row by row. Each row's answer is tokenized exactly as
written, without special tokens (`" Friday"` and `"Friday"` are different tokens on most tokenizers), to k
tokens. A column whose every row is one token scores it at every position of its row (a logit lens over
positions). A column with an answer of several tokens lines every row up, one-token rows included: token i
scores the row's position i, so the read carries k positions on that row (a scope with `before` reads the
positions that predict the answer's tokens), and the metric keeps the `pos` axis, NaN past a row's own where
rows' answers differ in length; a `sum`, `min` or `mean` over `pos` makes one number per row of it. A row whose answer and
read do not line up is refused by name, before any forward where the read's positions are the prompt's; and
where the read is a `before` scope on the step's own input (teacher forcing), so is a row whose input's next
token at a position is not one the answer's token there may be. A token stands for its class under the
reference's flags (`Column`). A row's answer may be a set, a list of strings any of which is the answer, its
members one width: at each position the ids any member's token (or class) may be, each once. `match` hits
where the most likely token is any of a position's ids, `token_prob` is their summed probability,
`cross_entropy` minus its log; `token_logit` and `logit_diff` read one id's logit, which several ids have
not, so a class or a set of more than one is refused there. A missing row (a scope the row does not say)
stays missing: NaN.
"""

# `engine`: the engine running the document. `original`: inside an entry, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
# `eos`: the ids the generation `of` came from stops on. `rows`: the dataset row
# of each of `of`'s rows; `positions`: the token position of each of its
# positions (`[rows, pos]`, -1 past a row's own), where the run knows them.
SUPPLIED = ("engine", "original", "dims", "eos", "rows", "positions")
# A metric scores a vocabulary axis: the tokenizer's, or a probe's classes.
VOCABULARY = ("vocab_size", "classes")

OPS: dict[str, Callable] = {}
GROUPS = ("metrics", "reductions", "edits", "transforms")
DECLARED: dict[str, dict[str, Any]] = {}


def op(group: str, needs: tuple[str | tuple[str, ...], ...] = (), removes: tuple[str, ...] = (),
       adds: tuple[str, ...] = (), bundle: str | None = None, layout: Callable[..., list[str]] | None = None,
       check: Callable[[dict], None] | None = None) -> Callable[[Callable], Callable]:
    """Declare the decorated function as an op under its own name: its `group` (`metrics`, `reductions`,
    `edits` or `transforms`), its axes, and, for a value a featurizer loads, the featurizer kind its `bundle`
    is for. An op whose result's axes depend on its arguments declares `layout(dims, args, kind)` in place of
    the axes, `kind` that of the featurizer the op's `Featurizer` argument names. `check(args)` raises on
    arguments its signature accepts but the op cannot mean. It joins `OPS` by its place in the list at the end
    of this file."""
    assert group in GROUPS
    def declare(fn: Callable) -> Callable:
        DECLARED[fn.__name__] = {"group": group, "needs": needs, "removes": removes, "adds": adds, "bundle": bundle,
                                 "layout": layout, "check": check}
        return fn
    return declare


def listed(group: str) -> str:
    """The ops of `group`, each in backticks, in the tables' order: what the prose names a group by."""
    return ", ".join(f"`{name}`" for name in OPS if DECLARED[name]["group"] == group)


def parameters(name: str) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones, an optional thing (`Value |
    None`) as the thing."""
    return {k: p.replace(annotation=registry.thing(p.annotation))
            for k, p in registry.spelled(OPS[name], SUPPLIED + ("of",)).items()}


def check(name: str, args: dict) -> None:
    """Op `name` exists and `args` bind to its signature; a featurizer named where an op takes none is refused
    with the bracket that runs the op on its coordinates."""
    if name == "featurizer":
        raise ValueError("there is no 'featurizer' op: 'encode' gives a featurizer's coordinates of the value and "
                         "'apply' its version of it")
    if name not in OPS:
        raise ValueError(f"unknown op {name!r}; valid: {', '.join(OPS)}")
    takes = parameters(name)
    if "featurizer" in args and "featurizer" not in takes:
        raise ValueError(f"{name!r} takes no 'featurizer': an op runs on a featurizer's coordinates between an "
                         f"encode and a decode of it, [{{'op': 'encode', 'featurizer': ...}}, {{'op': {name!r}, ...}}, "
                         "{'op': 'decode', 'featurizer': ...}]")
    if "select" in args and "select" not in takes:
        raise ValueError(f"{name!r} takes no 'select': an encode keeps some of a featurizer's coordinates, "
                         "{'op': 'encode', 'featurizer': ..., 'select': {'features': <selector>}}")
    registry.check(OPS[name], args, SUPPLIED + ("of",), repr(name))
    if DECLARED[name]["check"] is not None:
        DECLARED[name]["check"](args)


def call(name: str, value: Any, args: dict[str, Any], **supplied: Any) -> Any:
    """Run op `name` on `value` with `args` (document spelling, checked and bound to tensors) and those of the
    run's `supplied` values its signature names. An edit runs on `value` and every floating tensor it is given
    in float32, and its result is cast to `value`'s dtype once (`ABOUT`)."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    given = {**{names[k]: v for k, v in args.items()}, **{k: v for k, v in supplied.items() if k in names}}
    if DECLARED[name]["group"] != "edits" or not isinstance(value, Tensor):
        return fn(value, **given)
    up = {k: v.float() if isinstance(v, Tensor) and v.is_floating_point() else v for k, v in given.items()}
    return fn(value.float(), **up).to(value.dtype)


def out_dims(name: str, dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`: its `layout`'s, for a featurizer of `kind`,
    when it declares one."""
    axes = DECLARED[name]
    if axes["layout"] is not None:
        return axes["layout"](dims, args, kind)
    for axis in axes["needs"]:
        either = axis if isinstance(axis, tuple) else (axis,)
        if not set(either) & set(dims):
            raise ValueError(f"{name!r} needs a value with a {' or '.join(map(repr, either))} axis; this one is "
                             f"{dims}")
    return [a for a in dims if a not in axes["removes"]] + list(axes["adds"])


# The ops, in the order the tables (`describe`, FORMAT.md) list them.
from clab.ops import edits, metrics, reductions, transforms  # noqa: E402

OPS.update((fn.__name__, fn) for fn in [
    edits.swap, edits.add, edits.sub, edits.scale, edits.renormalize, edits.normalize, edits.project_out,
    edits.decode, edits.apply, edits.noise, edits.clamp, transforms.readout, transforms.encode, reductions.mean,
    reductions.sum, reductions.min, reductions.pca, metrics.match, metrics.logit_diff, metrics.token_prob,
    metrics.token_logit, metrics.cross_entropy, metrics.contains, metrics.kl, metrics.js, transforms.top_k,
    transforms.argmax, transforms.detokenize, metrics.cosine])
assert OPS.keys() == DECLARED.keys(), "every declared op is listed once"
ABOUT = ABOUT.replace("<the edits group>", listed("edits"))
