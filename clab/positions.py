"""Positions: the `pos` grammar, and pos records resolved to token indices inside the run."""

import torch


def parse(pos) -> dict:
    """`pos` names token positions within a scope.

    ```
    pos := int                                    # the one shorthand: {"index": int}
         | {"frame"?: "prompt" | "generated",     # default "prompt"
            "scope"?: {"variable": column} | {"segment": "role[n]" | "role" | "eos"},
            exactly one of "index": int | "span": [start, stop] | "all": true}
    ```

    Indices count the scope's tokens and may be negative (from its end); a
    span is half-open. The prompt's tokens are the row's tokens that are not
    special (BOS is never one) and not padding. `variable` scopes to the row's
    own text of that column, located in its input; `segment` to a chat turn
    (`user[1]` is the second user message) or, in the generated frame, to where
    the row stopped (`eos`). The generated frame counts decode steps: index
    *k* is the forward that chose new token *k*, read at its last position.
    A turn has no meaning in the generated frame, nor `eos` in the prompt.

    Every pos compiles to `{"frame", "scope", "start", "stop"}`, a slice of
    the scope's tokens.
    """
    if isinstance(pos, int) and not isinstance(pos, bool):
        pos = {"index": pos}
    if not isinstance(pos, dict):
        raise ValueError(f"{pos!r} is not a position: an int or a dict with one of 'index', 'span', 'all'")
    unknown = set(pos) - {"frame", "scope", "index", "span", "all"}
    if unknown:
        raise ValueError(f"unknown key {sorted(unknown)[0]!r}; valid: frame, scope, index, span, all")
    given = [k for k in ("index", "span", "all") if k in pos]
    if len(given) != 1:
        raise ValueError("a position takes exactly one of 'index', 'span', 'all'")
    frame = pos.get("frame", "prompt")
    if frame not in ("prompt", "generated"):
        raise ValueError(f"frame {frame!r}; valid: prompt, generated")
    scope = pos.get("scope")
    if scope is not None:
        if not (isinstance(scope, dict) and len(scope) == 1 and set(scope) <= {"variable", "segment"}
                and isinstance(next(iter(scope.values())), str)):
            raise ValueError("a scope is {'variable': column} or {'segment': 'role[n]' | 'role' | 'eos'}")
        segment = scope.get("segment")
        if segment == "eos" and frame == "prompt":
            raise ValueError("segment 'eos' is where a row stopped generating; a prompt does not stop")
        if segment not in (None, "eos") and frame == "generated":
            raise ValueError(f"segment {segment!r} is a chat turn; a continuation has no turns")
    if "index" in pos:
        k = pos["index"]
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(f"index {k!r} is not an int")
        start, stop = k, (None if k == -1 else k + 1)
    elif "span" in pos:
        span = pos["span"]
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(i, int) for i in span)):
            raise ValueError(f"span {span!r} is not [start, stop]")
        start, stop = span
    else:
        if pos["all"] is not True:
            raise ValueError("'all' takes true")
        start, stop = 0, None
    return {"frame": frame, "scope": None if scope is None else dict(scope), "start": start, "stop": stop}


def ragged(pos: dict) -> bool:
    """Whether the slice's width can differ between rows: one end counted from the start, one from the end."""
    start, stop = pos["start"], pos["stop"]
    return (start < 0) != (stop is None or stop < 0)


def resolve(pos: dict, batch: dict) -> torch.Tensor:
    """The record's token indices into the padded batch, `[rows, width]`."""
    if pos["frame"] == "generated":
        raise NotImplementedError("M3: positions in the generated frame")
    if pos["scope"] is not None or ragged(pos):
        raise NotImplementedError("M2: scoped and ragged positions")
    tokens = batch["attention_mask"].bool() & ~batch["special"]
    rows = []
    for row, keep in enumerate(tokens):
        index = keep.nonzero().flatten()
        n = len(index)
        start = pos["start"] + n if pos["start"] < 0 else pos["start"]
        stop = n if pos["stop"] is None else (pos["stop"] + n if pos["stop"] < 0 else pos["stop"])
        if not 0 <= start < stop <= n:
            raise ValueError(f"row {row}: tokens [{pos['start']}:{pos['stop']}] do not fit its {n} prompt tokens")
        rows.append(index[start:stop])
    return torch.stack(rows)
