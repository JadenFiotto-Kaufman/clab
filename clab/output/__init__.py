"""A run's output directory and back, and saved values as tables.

One file per save, named after it: `<name>.safetensors` when the value is a tensor or a dict of tensors (a
bundle; a ragged read's `{values, row, tok}`), `<name>.json` otherwise, tensors in it as nested lists. A swept
save holds one entry per point, keyed by its coordinate: `pos=-1` in safetensors, `"-1"` in JSON
(`layers=0,pos=-1` and `"0,-1"` over two axes), a dict's own keys after a `/`. A file is the value and
nothing else, except that a bundle a featurizer loads (the value of an op that declares one, `pca`, and a
fit's trained featurizer) carries the header that featurizer checks; a swept one holds each point's header keys
after its coordinate and a `/`, as its tensors are (`rank=2/rank`), beside one `format`.

`manifest.json` is the compiled plan without the rows it read from files and without bundle tensors, its
`results` each replaced by `{file, step, dims, axes, shape, dtype}`: that is how a result links back to the
step that made it. It records the document's path relative to the corpus root, and each dataset's `n_rows`. `shape` and `dtype` follow the value's structure (per key, per point).
The plan's `run` holds how each input tokenized, what each generate step's rows generated, and which
position (as `pos` counts them: an index into the row's prompt tokens, BOS first, then its generated ones)
every read and write resolved to per row, which
`table` uses to label positions. Values are never in the manifest.
"""

from .dump import describe, dump  # noqa: F401
from .load import load  # noqa: F401
from .table import table  # noqa: F401
