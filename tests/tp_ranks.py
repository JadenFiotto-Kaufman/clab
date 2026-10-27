"""One rank of a tensor-parallel run, for `test_tp.py`: `torchrun --nproc-per-node=<tp> tests/tp_ranks.py <tp>
<device> <out> <name>...` runs each document named (with `CHANGES`) through `clab.run` on the nnterp engine at `tp` on
`device`, rank 0 writing `<out>/<name>`, and saves what this rank got, its results and run record or the refusal's
message, to `<out>/<name>.rank<RANK>.pt`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import BATCH_SIZE, HYBRID, edited  # noqa: E402

# A run's name, beside the documents': the document it edits, and how.
CHANGES = {"das": ("das", lambda d: d["steps"]["fit"].update(epochs=1)),
           "sampled": ("generated_contains", lambda d: d["steps"]["answer"].update(do_sample=True)),
           "sampled_again": ("generated_contains", lambda d: d["steps"]["answer"].update(do_sample=True)),
           "hybrid_patching": ("patching", lambda d: d["model"].update(HYBRID)),
           "hybrid_queries": ("attention_query",
                              lambda d: (d["model"].update(HYBRID), d["sites"]["q"].update(layers=3)))}


def planned(name: str):
    document, change = CHANGES.get(name, (name, lambda d: None))
    return edited(document, change)


if __name__ == "__main__":
    import torch

    import clab
    from clab.engines import load

    tp, device, out, names = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    loaded: dict = {}
    for name in names:
        plan = planned(name)
        key = plan.model.model_dump_json()
        try:
            if key not in loaded:
                loaded[key] = load("nnterp", plan.model, device, tp=tp)
            filled = clab.run(plan, engine=loaded[key], out=os.path.join(out, name), batch_size=BATCH_SIZE)
            got = {"results": filled.results, "run": filled.run}
        except ValueError as e:
            got = {"refused": str(e)}
        torch.save(got, os.path.join(out, f"{name}.rank{os.environ['RANK']}.pt"))
