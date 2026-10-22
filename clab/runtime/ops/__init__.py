"""The one computation vocabulary: every op is a plain function over tensors.

`@op(group, ...)` declares a function as an op under its own name: its group, the axes the input
must carry (`needs`), the axes it removes (`removes`: a tuple, or the name of the argument that lists them),
the axes it appends (`adds`), when its value is a bundle a featurizer loads, that featurizer's kind
(`bundle`), and a rule its arguments keep beyond their types (`check`). `OPS` lists them, in the order at the
end of this file. An op whose axes and further arguments depend on what it is given declares them as functions
instead (`layout`, `options`), as `encode` and `apply` do for their featurizer's kind. Compile reads the axes to lay out
every value. A parameter's annotation says how a document fills it (`Value`, `Tokens`,
`Featurizer`, or a literal); `engine`, `original` and `dims` are filled by the
run, never by a document; a trailing underscore is dropped from a parameter's document spelling (`with_` is
`"with"`).
"""

ABOUT = """An op runs in two places, always as the same function:

- as a step, `{"kind": "op", "op": "mean", "of": "harvest.acts"}`, over a value earlier steps produced;
- inside an entry of `interventions`, `"ops": [...]`, on the entry's selection at its site, in order.

Inside an entry an op sees one batch, so it must act per example: one that aggregates over rows is
refused there (make it a step after the forward). An entry's edits act on the live value as the forward
reaches the site, and the value after its last edit is written back; the ops after that run on the value
once it is taken off the model (an entry nothing uses is not taken off, so they do not run).

A row a read beyond the prefill did not reach is missing all the way down: in a rectangular value it is NaN,
every metric and edit of it is NaN, `argmax` and `top_k` give it id -1 (`top_k` NaN probabilities and no
token text). Reductions skip it: `mean` is the mean over the rows that have a value, and `pca` is fitted on them.

Ops come in four groups: **metrics** score each example (one number per row and position: `match` is 0 or
1, and the accuracy is a `mean` step after it; `contains` scores a detokenized generation's text);
**reductions** remove axes, by default the rows and positions, a `layers` axis kept (`over` names them;
over `pos`, one value per row); **edits** (`swap`, `add`, `sub`, `scale`, `renormalize`, `normalize`,
`project_out`, `decode`, `apply`, `noise`, `clamp`) keep the value's layout and change its contents, and inside
an entry are what writes back; **transforms** turn each example's value into another of a different layout (logits from
a residual stream, coordinates from a value, tokens from logits, text from ids). An edit comes before any
transform, metric or reduction of its entry, or, after an `encode`, before the `decode` that closes it.

A featurizer is reached by three ops. `encode` turns the value into its coordinates in the featurizer's
space (`select: {"features": <selector>}` keeps some of them), `decode` (naming the same featurizer) turns
them back: within an entry each `decode` closes the most recent `encode` not yet closed, which must be of
the same featurizer, and the ops between them act on the coordinates. An operand there is in the same
coordinates (a value read through the same encodes, with the same `select`s, or a number), and one in other
coordinates is refused by name, whatever its width. `decode` writes the change of coordinates back as a
delta, so `[encode x, decode x]` is the identity to the bit, an op that changes no coordinate changes
nothing, and a sparse autoencoder's error term survives; a `select`ed coordinate is scattered back and the
rest gain a change of exactly zero (encoding the result again gives them back to float rounding, about
1e-9, not to the bit). Brackets nest: `[encode a, encode b, <edits>, decode b, decode a]` edits `b`'s
coordinates of `a`'s, `b` as wide as `a`'s coordinates. An entry that writes back with an `encode` still
open, a `decode` with no `encode` open, one of another featurizer than the open one, and an `encode` or
`apply` of a featurizer inside its own `encode`, are refused by name. `apply` is the featurizer's own version
of the value (a subspace's projection, a dictionary's reconstruction without its error term, an adapter's
adapted value): an edit, which writes back. An entry that stops after `encode` reads coordinates.

An argument is filled by a document as its type says: `Value` is a named value or a number; `Tokens` a column
(a `Column` reference: the string, or the object with `strip` and `lower`) resolved through the value's own
vocabulary, the tokenizer's against logits or a probe's classes against its scores (a metric scores a
`vocab_size` or a `classes` axis); `Text` a column read as text; `Featurizer` a featurizer's name; anything
else is a literal. `renormalize`'s reference is the value at the site before the step's writes (inside an
`encode`, its coordinates). An operand may name an earlier entry of the same forward, at this entry's site or
at one the forward reaches before it (`patched.premix` in `patched`): it is that entry's value in this
batch, at its positions.

A metric lines an answer up with the read it scores, row by row. Each row's answer is tokenized exactly as
written, without special tokens (`" Friday"` and `"Friday"` are different tokens on most tokenizers), to k
tokens. A column whose every row is one token scores it at every position of its row (a logit lens over
positions). A column with an answer of several tokens lines every row up, one-token rows included: token i
scores the row's position i, so the read carries k positions on that row (a scope with `before` reads the
positions that predict the answer's tokens), and the metric keeps the `pos` axis, ragged where rows' answers
differ in length; a `sum`, `min` or `mean` over `pos` makes one number per row of it. A row whose answer and
read do not line up is refused by name, before any forward where the read's positions are the prompt's; and
where the read is a `before` scope on the step's own input (teacher forcing), so is a row whose input's next
token at a position is not one the answer's token there may be. A token stands for its class under the
reference's flags (`Column`). A row's answer may be a set, a list of strings any of which is the answer, its
members one width: at each position the ids any member's token (or class) may be, each once. `match` hits
where the most likely token is any of a position's ids, `token_prob` is their summed probability,
`cross_entropy` minus its log; `token_logit` and `logit_diff` read one id's logit, which several ids have
not, so a class or a set of more than one is refused there. A missing row (a scope the row does not say)
stays missing: NaN, or no entries.
"""

import inspect
from typing import Any, Callable

import torch
from torch import Tensor

from ... import registry
from ...address import EXAMPLE_AXES, pos_axis
from ...registry import Featurizer, Text, Tokens, Value  # noqa: F401  (ops.Text, ops.Tokens, ops.Value)
from .. import featurizers

# `engine`: the engine running the document. `original`: inside an entry, the
# value at the site before this forward's writes. `dims`: the axis names of `of`.
# `eos`: the ids the generation `of` came from stops on. `rows`: the dataset row
# of each of `of`'s rows or entries; `positions`: the token position of each of
# its positions (`[rows, pos]`) or entries, where the run knows them.
SUPPLIED = ("engine", "original", "dims", "eos", "rows", "positions")
# A metric scores a vocabulary axis: the tokenizer's, or a probe's classes.
VOCABULARY = ("vocab_size", "classes")

OPS: dict[str, Callable] = {}
GROUPS = ("metrics", "reductions", "edits", "transforms")
DECLARED: dict[str, dict] = {}


def op(group: str, needs: tuple = (), removes: tuple | str = (), adds: tuple = (), bundle: str | None = None,
       layout: Callable | None = None, options: Callable | None = None, check: Callable | None = None):
    """Declare the decorated function as an op under its own name: its `group` (`metrics`, `reductions`,
    `edits` or `transforms`), its axes, and, for a value a featurizer loads, the featurizer kind its `bundle`
    is for. An op whose result's axes depend on its arguments declares `layout(dims, args, kind)` in place of
    the axes, and one whose `**` arguments do declares `options(args, kind)`, the function they go to, whose
    first two parameters (the featurizer and the value) the op fills; `kind` is that of the featurizer the
    op's `Featurizer` argument names. `check(args)` raises on arguments its signature accepts but the op
    cannot mean. It joins `OPS` by its place in the list at the end of this file."""
    assert group in GROUPS
    def declare(fn: Callable) -> Callable:
        DECLARED[fn.__name__] = {"group": group, "needs": needs, "removes": removes, "adds": adds, "bundle": bundle,
                                 "layout": layout, "options": options, "check": check}
        return fn
    return declare


def parameters(name: str, args: dict | None = None, kind: str | None = None) -> dict[str, inspect.Parameter]:
    """An op's parameters by document spelling, without `of` and the supplied ones; with `args` and the `kind`
    of its featurizer, also those of the function its `options` choose (an optional thing, `Value | None`, as
    the thing)."""
    found = {k: p.replace(annotation=registry.thing(p.annotation))
             for k, p in registry.spelled(OPS[name], SUPPLIED + ("of",)).items()}
    fn = options(name, args, kind)
    if fn is not None:
        own = registry.spelled(fn, tuple(inspect.signature(fn).parameters)[:2])
        found.update({k: p.replace(annotation=registry.thing(p.annotation)) for k, p in own.items()})
    return found


def options(name: str, args: dict | None, kind: str | None) -> Callable | None:
    """The function op `name`'s `**` arguments go to, for these arguments and featurizer kind, if it has one."""
    chooses = DECLARED[name]["options"]
    return chooses(args, kind) if chooses is not None and args is not None and kind is not None else None


def featurizer_of(name: str, args: dict) -> Any:
    """Op `name`'s `Featurizer` argument in `args`, if it takes one: a name in a document, an instance in a run."""
    return next((args.get(k) for k, p in parameters(name).items() if p.annotation == Featurizer), None)


def kind(name: str, args: dict) -> str | None:
    """The kind of the featurizer instance bound as op `name`'s `Featurizer` argument, if any."""
    f = featurizer_of(name, args)
    return f.kind if isinstance(f, featurizers.Kind) else None


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
    """Run op `name` on `value` with `args` (document spelling, already bound to tensors); an argument the
    op's signature does not name goes to its `**` catch-all as spelled."""
    fn = OPS[name]
    names = {p.rstrip("_"): p for p in inspect.signature(fn).parameters}
    kwargs = {names.get(k, k): v for k, v in args.items()}
    kwargs.update((k, v) for k, v in supplied.items() if k in names and k not in kwargs)
    return fn(value, **kwargs)


def out_dims(name: str, dims: list[str], args: dict, kind: str | None = None) -> list[str]:
    """The axes of op `name`'s result, on a value laid out `dims`: its `layout`'s, for a featurizer of `kind`,
    when it declares one."""
    axes = DECLARED[name]
    if axes["layout"] is not None:
        return axes["layout"](dims, args, kind)
    for axis in axes["needs"]:
        if not set(axis if isinstance(axis, tuple) else (axis,)) & set(dims):
            ranks = "; it ranks a vocabulary's tokens, and over another axis (an sae's features, a probe's classes) " \
                "it is not an op yet: save the value and rank it after the run" if axis == "vocab_size" and \
                name in ("argmax", "top_k") else ""
            raise ValueError(f"{name!r} needs a value with a {' or '.join(map(repr, axis)) if isinstance(axis, tuple) else repr(axis)} "
                             f"axis; this one is {dims}{ranks}")
    removes = axes["removes"]
    if isinstance(removes, str):
        removes = args.get(removes)
        for axis in removes or []:
            if axis not in dims:
                raise ValueError(f"{name!r} cannot remove {axis!r}; the value is {dims}")
        removes = removes if removes is not None else EXAMPLE_AXES
    return [a for a in dims if a not in removes] + list(axes["adds"])


def taken(x: Tensor, record: dict) -> Tensor:
    """The record's positions and narrowing of `x`, a site's value with its token axis at 1.

    `record["positions"]` is `[rows, width]` token indices; `record["select"]`
    maps a dimension of that layout to the indices kept along it: one list for
    every row, or (`[rows, n]`, on the `key` axis) each row's own. A `key` axis it does not select is each row's
    own keys (`record["keys"]`), NaN past them. The indices, built where the plan runs, move to the value's device.
    """
    x = owned(x[torch.arange(x.shape[0], device=x.device)[:, None], record["positions"].to(x.device)], record)
    if "keys" in record:
        x = x.masked_fill(~per_row(x, record["keys"]["dim"], record["keys"]["own"].to(x.device)), float("nan"))
    for dim, index in record["select"].items():
        x = take(x, dim, index)
    return x


def owned(x: Tensor, record: dict) -> Tensor:
    """A selection's `key` axis in each row's own order, its keys and then its padding's, where the record does
    not select keys (`record["keys"]`); else as it is."""
    return take(x, record["keys"]["dim"], record["keys"]["index"]) if "keys" in record else x


def take(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """`x` at `index` along `dim`: the same indices on every row, or (`[rows, n]`) each row's own."""
    index = index.to(x.device)
    return x.index_select(dim, index) if index.dim() == 1 else x.gather(dim, per_row(x, dim, index))


def per_row(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Each row's `[rows, n]` indices expanded to `x`'s shape with `n` along `dim`, for gather and scatter."""
    shape = [1] * x.dim()
    shape[0], shape[dim] = index.shape
    size = list(x.shape)
    size[dim] = index.shape[1]
    return index.reshape(shape).expand(size)


def visit(value: Tensor, component: str, records: list[dict], got: dict) -> tuple[Tensor, bool]:
    """`value` with a site's records done in listed order, and whether any wrote: a write's ops run on its
    selection of the current value, out of place, and the result lands there; a read takes its selection of
    the current value into `got` under its name (at a decode call, appended: one per call it reads).

    `original` is a write's selection as it was before any of these writes. An argument bound to an entry of
    the same forward (`{"live": <its read's name>}`) is that read's value in `got`, where this forward put it.
    An op is given the dataset row of each of the selection's rows (the op's `rows`) and the token position of
    each of its positions (the write's `tok`, plus the decode `call` it lands at, which the engine stamps). A
    write narrows at most one dimension of its selection (compile refuses more). A ragged write's rows are
    padded to one width; only the positions its `keep` marks are written. A `key` axis a record does not select
    is each row's own keys in order (`owned`): a read's are NaN past them, and a write writes those alone. Every index, mask and result moves
    to the value's device, and a result to its dtype, where it meets the value.
    """
    axis = pos_axis(component)
    before = x = value.movedim(axis, 1)
    rows = torch.arange(x.shape[0], device=x.device)[:, None]
    wrote = False
    for w in records:
        if "ops" not in w:
            if "calls" in w:
                got[w["name"]].append(taken(x, w))
            else:
                got[w["name"]] = taken(x, w)
            continue
        if not wrote:
            x, wrote = before.clone(), True
        pos = w["positions"].to(x.device)
        current, original = owned(x[rows, pos], w), owned(before[rows, pos], w)
        new = current
        for dim, index in w["select"].items():
            new, original = take(new, dim, index), take(original, dim, index)
        at = w["tok"] + w.get("call", 0) if "tok" in w else None        # each position, at a decode call its own
        new = edited(new, original, w["ops"], got, {"positions": at})
        for dim, index in w["select"].items():
            new, index = new.to(device=current.device, dtype=current.dtype), index.to(current.device)
            new = current.index_copy(dim, index, new) if index.dim() == 1 else \
                current.scatter(dim, per_row(new, dim, index), new)
        if "keys" in w:                 # a row's own keys written, its padding's as they were, in the batch's order
            dim, index = w["keys"]["dim"], w["keys"]["index"].to(x.device)
            new = torch.where(per_row(current, dim, w["keys"]["own"].to(x.device)),
                              new.to(device=current.device, dtype=current.dtype), current)
            new = x[rows, pos].scatter(dim, per_row(new, dim, index), new)
        keep = w["keep"].to(x.device) if "keep" in w else torch.ones(pos.shape, dtype=torch.bool, device=x.device)
        x[rows.expand_as(pos)[keep], pos[keep]] = new.to(device=x.device, dtype=x.dtype)[keep]
    return x.movedim(1, axis), wrote


def edited(new: Tensor, original: Tensor, bound: list[dict], got: dict | None, where: dict) -> Tensor:
    """`new` after a write's bound ops, each on the current value in turn. An `encode` runs the ops its `decode`
    closes over (`inner`) on the value's coordinates, on the features its `select` keeps, and writes the change
    back (`Kind.within`); `original` goes in with them, encoded the same way."""
    for o in bound:
        args = {k: live(got, v) if isinstance(v, dict) and "live" in v else v for k, v in o["args"].items()}
        if o["op"] != "encode":
            new = call(o["op"], new, args, original=original, dims=o["dims"], rows=o.get("rows"), **where)
            continue
        f: Any = args["featurizer"]
        features = o.get("features")
        start = f.encode(original)
        start = start if features is None else start.index_select(-1, features.to(start.device))
        new = f.within(new, lambda z, start=start, inner=o["inner"]: edited(z, start, inner, got, where), features)
    return new


def live(got: dict | None, operand: dict) -> Tensor:
    """A write's operand that names a read of its own forward: that read's value, which the engine made before
    it reached the write."""
    if operand["live"] not in (got or {}):
        raise ValueError(f"operand {operand['live']!r} is a read of this forward the engine had not made when it "
                         "reached the write; it reads it at a later site")
    return (got or {})[operand["live"]]


# The ops, in the order the tables (`describe`, FORMAT.md, schema.json) list them.
from . import edits, metrics, reductions, transforms  # noqa: E402

OPS.update((fn.__name__, fn) for fn in [
    edits.swap, edits.add, edits.sub, edits.scale, edits.renormalize, edits.normalize, edits.project_out,
    edits.decode, edits.apply, edits.noise, edits.clamp, transforms.readout, transforms.encode, reductions.mean,
    reductions.sum, reductions.min, reductions.pca, metrics.match, metrics.logit_diff, metrics.token_prob,
    metrics.token_logit, metrics.cross_entropy, metrics.contains, metrics.kl, metrics.js, transforms.top_k,
    transforms.argmax, transforms.detokenize, metrics.cosine])
assert OPS.keys() == DECLARED.keys(), "every declared op is listed once"
