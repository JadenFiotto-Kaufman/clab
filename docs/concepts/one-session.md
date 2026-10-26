---
title: One Session
one_liner: "Why a document runs the same locally and on a server: one session per run, the plan as pure data, nothing decided from a tensor on the client, the engine as the only client object, and what ships by reference, so what a server needs."
tags: [concepts, session, ndif, remote, design]
related: [docs/usage/engines.md, docs/concepts/plan-lifecycle.md, docs/developing/architecture.md, docs/developing/engines.md]
sources: [clab/engines/nnterp.py, clab/engines/ndif.py, clab/engines/__init__.py, clab/steps/__init__.py, clab/plan.py]
---

# One Session

## The idea

A clab run is one nnsight session. The engine opens it, `clab.steps.run` does the whole run inside it (preflight, every step, the run record), and one value comes home at the end: a dict holding the results and the run record. The nnterp engine opens the session with `remote=False`. The ndif engine is the nnterp engine with a weightless model, and opens the same session with `remote=True`, or with `"local"` for nnsight's in-process simulation of the trip. The session body is identical in all three cases, so a document's meaning cannot depend on where it runs. Under `-o remote=local`, `documents/patching.json` and `documents/attention_query.json` give the nnterp engine's tensors bit for bit.

```text
client                                   inside the session (the server, under ndif)
  Plan(path), compile()                    rows not in the plan read again, digests checked
  engines.load()  (a meta shell)  ──▶      preflight against the model that runs
  opens the session                        every step and point, in order
  plan.results, plan.run  ◀── one dict ──  results and the run record (versions, dtype, positions)
  dump(out)
```

Five design constraints keep it so. Each is a rule a change to clab keeps, and each has a consequence for what you write and run; where the code holds each is [developing/architecture.md](../developing/architecture.md#where-the-code-holds-the-five-constraints).

## One session per run, and `remote` is the only difference

The run never leaves the session between steps, so it costs one request, one queue wait and one download, however many steps and sweep points the document has. Nothing the engine does depends on whether the session is local or remote. That is why `remote="local"` is a faithful rehearsal, and why a document needs no changes to move to a server.

The hooks engine runs the same `steps.run` without a session, as plain Python in your process. That makes it a second way of reaching the model to compare against, and it is also why hooks cannot run remotely.

## The compiled plan is pure data

A compiled plan is records of strings, numbers, lists and dicts: the resolved document, file datasets' rows, and the tensors of any bundle it loads. Its methods (`compile`, `dump`, `load`, `table`, `write`) are views and serialization, never execution. So it pickles, prints and means the same thing wherever it lands, and it can travel into a remote session and back.

Data that the plan only refers to is read where the run is. A hub dataset's rows never enter the plan, only their digest. The run loads them where it runs and refuses rows whose digest differs from the one compile recorded. A server therefore reads a hub dataset itself, and a file dataset's rows travel inside the request.

## Nothing is decided from a tensor on the client

Every decision that depends on what the model computes is made inside the session, next to the tensor. Such decisions include where a generated row stopped, which decode call a scoped write lands at, what an op returns, and whether a row is `alignment_missing`. Preflight sits inside too, so the tokens, sizes and served components it checks belong to the model that actually runs.

The client decides only from the document, the data files and the bundle headers, which is compile's job ([plan-lifecycle.md](plan-lifecycle.md)). Values come home once, after the last step, and never mid-run to steer what happens next. Under ndif, then, a compile refusal arrives before anything is sent, and a preflight or step refusal arrives from the server.

## The engine and its model are the only client-side objects in the session

Besides the modules it calls, the session body closes over three things: the engine, the plan (which is data) and the batch size. The engine holds its model and its configuration (the model block, device and options), nothing else. nnsight ships every name a block closes over, and on the server it replaces the client's model with the deployed one. The engine therefore arrives as configuration plus a handle to the server's model.

This is also why an engine keeps no caches, open files or other client state: any of them would ship with every request, or fail to serialize. The protocol an engine follows is [developing/engines.md](../developing/engines.md).

## Nothing of clab ships by value; it resolves by import on the server

The request names `clab.engines.ndif`, `clab.plan` and `clab.steps` by reference, and nnterp's model classes the same way. The only clab code that travels is the session block's own few lines, which nnsight always sends as source. Everything those lines call is imported on the server from its installed clab.

So the server must have clab installed at your version, with its dependencies. A change to your local clab has no effect on a server that does not have it, and the run record says which clab ran: `plan.run.versions` is read inside the session, so after a true remote run it lists the server's versions. [engines.md](../usage/engines.md#what-the-server-must-have) lists what else a server needs.
