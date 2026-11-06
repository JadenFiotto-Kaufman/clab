---
title: Names and Sweeps
one_liner: "How do names resolve, and how do sweeps multiply into points?"
tags: [concepts, names, sweeps, scope]
related: [docs/usage/sweeps.md, docs/usage/writing-a-document.md, docs/usage/interventions.md, docs/usage/fit.md, docs/concepts/plan-lifecycle.md]
sources: [clab/compile/__init__.py, clab/format/sweeps.py, clab/plan.py]
---

# Names and Sweeps

A document is a line of steps that name each other's values, and compile resolves every name before the model loads. A sweep turns that line into a grid of points. Each sweep label is an axis, and a step has a point for each combination of the axes it depends on. Each value is made once per distinct point on its own axes. The exact rules are [FORMAT.md#names-and-scoping](../../FORMAT.md#names-and-scoping) and [FORMAT.md#sweeps](../../FORMAT.md#sweeps); this page explains the model behind them. Writing sweeps, their labels, their cost and their results is [usage/sweeps.md](../usage/sweeps.md).

## A name is a thing, then a member

Every reference splits at its first dot. The part before it is a *thing*: a step, a dataset or a featurizer. The part after it is a *member* of that thing: a step's entry (`counterfactual.v_cf`), a dataset's column (`pairs.cf_answer`), a featurizer's scalar (`mask.mask`), a fit's member (`fit.rot`), a `top_k` step's member (`top3.ids`, and over a list of axes `top.ids.rows`). The thing alone decides what kind of value a reference is, so the three kinds share one namespace. A name has no dot, and no step may share a name with a dataset or a featurizer:

```text
clab: steps.pairs: 'pairs' is also a dataset or a featurizer
clab: steps.i.ia: a name has no dots; a dot separates a thing from its member
```

Values come into existence in document order, and a reference resolves only against what exists before its step. A plan therefore has no cycles, and compile knows every value's layout and dataset before any forward. The refusal lists what does exist at that point:

```text
clab: steps.iia.of: 'patched.logits' names no value produced before this step; known: counterfactual.v_cf
```

## Scopes: the root and a fit's body

The root is one scope. A `fit`'s body is a second scope. It sees every value produced before the fit, and it runs over minibatches of rows many times.

- A minibatch is a list of row ids in the body's datasets.
- A value made before the fit with one row per example of one of them (a harvest) comes into each run of the body at that run's ids, as it was made.
- So the model runs while the fit trains only where the body has a forward.

Such a value cannot follow a change of rows. An `eval` that reads the harvest's dataset as another is refused, since no step made the harvest over the other's rows. Evaluate on the other dataset in steps after the fit, which reach what it trained as `<fit>.<featurizer>`:

```text
clab: steps.fit.eval.held_out: the body uses 'counterfactual.h', made before the fit with a row per example of 'train', and this eval reads 'train' as 'test', over whose rows no step made it; …
```

Nothing leaves the body except through the fit's members ([usage/fit.md](../usage/fit.md#members)), so its step names do not collide with the root's. An eval's steps are the body's again, compiled as root steps after the fit under the members' names (`fit.held_out.adapted`). There a body name means the eval's own step, and a trained featurizer is `<fit>.<featurizer>`. A root step named `patched` can sit before a fit whose body also has a `patched`. A root step after the fit that names `patched.logits` means the root's. Naming a body value bare from outside is refused, and the refusal lists the members you can name instead.

## Bundles are templates

A root `interventions` bundle is not a value. It is a set of entries that are pasted into each step that names the bundle, and its names resolve there. An operand that names one of the bundle's own entries bare means that entry in whichever step pulls the bundle in. A bundle with entries `resid` and `shifted` (`"with": "resid"`) pulled into steps `clean` and `cf` compiles to `clean.resid` in one and `cf.resid` in the other. Writing the step into the name fixes it to that step, which the bundle refuses:

```text
clab: interventions.lens.shifted.ops[0].with: 'clean.resid' names step 'clean''s entry 'resid', which is also this bundle's own entry; …
```

An operand that names another step's entry (`counterfactual.v_cf` in `documents/two_observes.json`'s `patching` bundle) resolves as written. [usage/interventions.md](../usage/interventions.md) covers bundles in use.

## Points

`{"sweep": [...]}` stands for a value, and its label names an axis. The label is the `label` you give, or else the last key of the sweep's path: `layers`, `pos`, `frame`, `scale`, or `mask.mask` for a swept objective weight in `documents/dbm.json`. One label is one axis wherever it recurs, in one entry, across a step's entries or across steps. How two labels zip or cross is [usage/sweeps.md](../usage/sweeps.md#labels-zip-or-cross).

Compile builds the document once for each combination of the axes a step depends on, with every sweep replaced by its value there. Each combination is a *point*: a complete step whose `at` gives its index on each axis. A step depends on two sets of axes:

- **its own:** the sweeps inside its spec, and inside the sites, bundles and featurizers it names;
- **inherited:** every axis of every value it references.

A step that no sweep reaches stays a single step and runs once. Every other step is a list of points:

- An op step over a swept value is swept too.
- A swept featurizer `seed` (`documents/random_subspace.json`) makes every step that encodes with it a list of points.
- A swept objective weight makes the fit train once per weight, and every step that uses `fit.mask` inherits the axis.

Which points of a forward share one forward is [usage/sweeps.md](../usage/sweeps.md#what-a-sweep-costs).

## How a consumer finds the producer's point

During the run, each value is stored once per distinct point on the axes it depends on (its `axes`), and a saved one is keyed by them.

- A consumer at point *p* takes the producer's value at *p*'s coordinates on the producer's axes.
- A consumer has every axis of what it references, so exactly one producer value agrees.
- One producer value serves every consumer point that agrees on its axes. In the grid in [usage/sweeps.md](../usage/sweeps.md#labels-zip-or-cross), the patch at `src=1,dst=0` takes the read's value at `src=1`, which also serves `src=1,dst=1`.

The same lookup means a step cannot see across points. An op at *p* gets the producer's value at *p* and no other, so no op averages or compares the points of a sweep. To combine settings inside the document, put them on one value's axis with a wider read ([usage/sweeps.md](../usage/sweeps.md#widen-the-read-instead)).
