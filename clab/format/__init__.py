"""The vocabulary a document is written in, with no model and no data in hand: the locations (`address`), the
selector grammar (`selectors`), how a function's signature is a document's contract (`registry`), sweeps and
their points (`sweeps`), and datasets and inputs (`data`). Nothing here imports from the rest of clab; the
records themselves are `clab.plan`, which sits above the ops and featurizer kinds it validates."""
