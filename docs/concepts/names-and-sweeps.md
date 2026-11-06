---
title: Names and Sweeps
one_liner: "How names resolve and sweeps multiply: one flat namespace per scope, a fit's own scope, bundles as templates, sweep labels as axes, points, coordinates, and how a consumer finds the producer's point."
tags: [concepts, names, sweeps, scope]
related: [docs/usage/sweeps.md, docs/usage/writing-a-document.md, docs/usage/interventions.md, docs/usage/fit.md, docs/concepts/plan-lifecycle.md]
sources: [clab/compile/__init__.py, clab/format/sweeps.py, clab/plan.py]
---

# Names and Sweeps

A document is a line of steps that name each other's values, and compile resolves every name before the model loads. A sweep turns that line into a grid: each sweep label is an axis, a step has a point for each combination of the axes it depends on, and each value is made once per distinct point on the axes it depends on itself. The exact rules are [FORMAT.md#names-and-scoping](../../FORMAT.md#names-and-scoping) and [FORMAT.md#sweeps](../../FORMAT.md#sweeps); this page explains the model behind them. The recipes are in [usage/sweeps.md](../usage/sweeps.md).

## A name is a thing, then a member

Every reference splits at its first dot. The part before it is a *thing*: a step, a dataset or a featurizer. The part after it is a *member* of that thing: a step's entry (`counterfactual.v_cf`), a dataset's column (`pairs.cf_answer`), a featurizer's scalar (`mask.mask`), a fit's member (`fit.rot`), a `top_k` step's member (`top3.ids`, and over a list of axes `top.ids.rows`). Because the thing alone decides what kind of value a reference is, the three kinds share one namespace. A name has no dot, and no step may share a name with a dataset or a featurizer:

```text
clab: steps.pairs: 'pairs' is also a dataset or a featurizer
clab: steps.i.ia: a name has no dots; a dot separates a thing from its member
```

Values come into existence in document order, and a reference resolves only against what exists before its step. A plan therefore has no cycles, and compile knows every value's layout and dataset before any forward. The refusal lists what does exist at that point:

```text
clab: steps.iia.of: 'patched.logits' names no value produced before this step; known: counterfactual.v_cf
```

## Scopes: the root and a fit's body

The root is one scope. A `fit`'s body is a second scope. It sees every value produced before the fit, but it runs over minibatches of rows many times. A minibatch is a list of row ids in the body's datasets, so a value made before the fit with one row per example of one of them (a harvest) comes into each run of the body at that run's ids, as it was made; the model then runs while the fit trains only where the body has a forward. What such a value cannot follow is a change of rows: an `eval` that reads the harvest's dataset as another is refused, since no step made the harvest over the other's rows:

```text
clab: steps.fit.eval.held_out: the body uses 'counterfactual.h', made before the fit with a row per example of 'train', and this eval reads 'train' as 'test', over whose rows no step made it; evaluate on 'test' in steps after the fit, which reach what it trained as 'fit.rot'
```

Nothing leaves the body except through the fit's members (`<fit>.<featurizer>`, `<fit>.<eval>.<step>.<entry>`, `<fit>.history.<op>`), so its step names do not collide with the root's. An eval's steps are the body's again, compiled as root steps after the fit under those names (`fit.held_out.adapted`), where a body name means the eval's own step and a trained featurizer is `<fit>.<featurizer>`. A root step named `patched` can sit before a fit whose body also has a `patched`, and a root step after the fit that names `patched.logits` means the root's. Naming a body value bare from outside is refused, and the refusal lists the members you can name instead. [usage/fit.md](../usage/fit.md) covers the members in use.

## Bundles are templates

A root `interventions` bundle is not a value. It is a set of entries that are pasted into each step that names the bundle, and its names resolve there. An operand that names one of the bundle's own entries bare means that entry in whichever step pulls the bundle in. A bundle with entries `resid` and `shifted` (`"with": "resid"`) pulled into steps `clean` and `cf` compiles to `clean.resid` in one and `cf.resid` in the other. Writing the step into the name fixes it to that step, which the bundle refuses:

```text
clab: interventions.lens.shifted.ops[0].with: 'clean.resid' names step 'clean''s entry 'resid', which is also this bundle's own entry; name the bundle's own entry bare ('resid'), which means it in whichever step pulls the bundle in
```

An operand that names another step's entry (`counterfactual.v_cf` in `documents/two_observes.json`'s `patching` bundle) resolves as written. [usage/interventions.md](../usage/interventions.md) covers bundles in use.

## A sweep label is an axis

`{"sweep": [...]}` stands for a value, and its label names an axis. The label is the `label` you give, or else the last key of the sweep's path: `layers`, `pos`, `frame`, `scale`, or `mask.mask` for a swept objective weight in `documents/dbm.json`. Each label is one axis:

- sweeps with the same label must have the same values, and they move together (they *zip*);
- sweeps with different labels are independent axes (they *cross*);
- one sweep reached from two places is one axis, for example a swept site that two steps use, or a swept bundle that two steps pull in.

Default labels are where this bites. Two `layers` sweeps pair their points even when you meant a grid, and `max_new_tokens` swept beside `min_new_tokens` crosses even when you meant the pairs. [usage/sweeps.md](../usage/sweeps.md#labels-zip-or-cross) shows both cases and their fixes.

The compiled plan lists its axes in `plan.sweeps`, each by label with its values (`{'num_heads': Axis(values=[0, 1, 2, 3])}` for `documents/head_patching.json`). The axes follow the order of the format's root keys (`data`, `sites`, `interventions`, `featurizers`, `steps`), then document order inside each key. The order of keys in your file does not matter: a featurizer's `seed` sweep is listed before a step's `pos` sweep even when `featurizers` is written after `steps`.

## Points

Compile builds the document once for each combination of the axes a step depends on, with every sweep replaced by its value there. Each combination is a *point*. A swept step becomes a list of points, and each point is a complete step whose `at` gives its index on each axis. A step depends on two sets of axes:

- **its own:** the sweeps inside its spec, and inside the sites, bundles and featurizers it names;
- **inherited:** every axis of every value it references.

A step that no sweep reaches stays a single step and runs once. Several consequences follow:

- A forward or generation depends on its options and its writes alone (each write's site, positions, ops up to its last edit and their operands), so points that agree on those share one forward, and a read that is all that is swept costs no forward of its own ([usage/sweeps.md](../usage/sweeps.md#what-a-sweep-costs)). Each point's `schedule` says which forward served it.
- An op step over a swept value is swept too.
- A swept featurizer `seed` (`documents/random_subspace.json`) makes every step that encodes with it a list of points.
- A swept objective weight makes the fit train once per weight, and every step that uses `fit.mask` inherits the axis.

## How a consumer finds the producer's point

During the run, each value is stored once per distinct point on the axes it depends on (its `axes`), and a saved one is keyed by them. A consumer at point *p* takes the value at *p*'s coordinates on those axes. A consumer has every axis of what it references, so exactly one agrees.

Take the grid in [usage/sweeps.md](../usage/sweeps.md#labels-zip-or-cross). The counterfactual read sweeps `src`, so it has 2 points, which share one forward. The patch sweeps `dst` and references the read, so it has 4 points, each its own forward. The patch at `src=1,dst=0` takes the read's value at `src=1`, and each of the read's values serves two patch points. The zipped version of the same document, which has one `layers` label, gives results equal to the grid's diagonal bit for bit.

The same lookup means a step cannot see across points. An op at point *p* gets the producer's value at *p* and no other, so there is no op that averages or compares the points of a sweep. To combine settings inside the document, put them on one value's axis with a wider read ([usage/sweeps.md](../usage/sweeps.md#widen-the-read-instead)).

## Coordinates

A point has two names:

- **`at`**, its *index* on each axis. This is what a compiled step carries, and it means something only inside its plan.
- **its coordinate**, its *values* on each axis (`pos=-1`, `src=0,dst=1`). This is how results, output files, the run record and tables are keyed, because a reader asks for layer 3, not for the fourth point.

[usage/sweeps.md](../usage/sweeps.md#reading-swept-results) lists the form a coordinate takes in each place.
