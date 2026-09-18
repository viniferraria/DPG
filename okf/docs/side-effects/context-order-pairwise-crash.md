---
type: Side Effect
title: discover_dfg_context crashes for every context_order > 1
description: A "ruff fixes" commit replaced zip(nodes, nodes[1:]) with itertools.pairwise(nodes, nodes[1:]) — but itertools.pairwise takes exactly one iterable. Every execution_trace fit with context_order > 1 (including "auto" resolving above 1) raises TypeError before any graph is built.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [regression, crash, context_order, dpg-k, itertools, ruff]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

`DecisionPredicateGraph.discover_dfg_context` (the context-aware, DPG-k graph builder used whenever the
resolved `context_order > 1` — see [Graph construction modes](/conventions/graph-construction-modes.md))
pairs up consecutive contextual nodes per trace to build directly-follows edges. At the 0.3.0 release
(`f16a977`) this used the standard two-iterator idiom:

```python
for source, target in zip(nodes, nodes[1:]):
```

Commit `6df6e20` ("chore: ruff fixes") replaced it with `itertools.pairwise`, but called it with the
same two-argument shape instead of `pairwise`'s actual one-iterable signature:

```python
# dpg/core.py:636 at HEAD 276a503
for source, target in pairwise(nodes, nodes[1:]):
```

`itertools.pairwise` takes exactly one iterable and yields consecutive pairs from it — it is not a
drop-in replacement for `zip(seq, seq[1:])`'s two-argument call shape. Calling it with two positional
arguments raises `TypeError` immediately, before a single edge is added to `dfg`.

# Why

This reads as an automated or semi-automated "modernize this idiom" edit (ruff/pyupgrade-style
`zip(x, x[1:])` → `pairwise(x)` rewrites are common) that preserved the two-argument call instead of
also dropping the second argument — `pairwise(nodes)` is the equivalent rewrite; `pairwise(nodes,
nodes[1:])` is not.

# Who is affected

Anyone fitting a `DecisionPredicateGraph` / `DPGExplainer` with `graph_construction.mode =
"execution_trace"` and a `context_order` that resolves to a value greater than 1 — either an explicit
integer `context_order > 1`, or `context_order="auto"` when `resolve_context_order` picks an order
above 1 (which it does whenever `k=1` still has pooled-path recombination). `context_order=1`
(the default) is unaffected: `discover_dfg_context` early-returns to
`discover_dfg_execution_trace(log)` before reaching the `pairwise` call.

The project's own test suite already catches this: `uv run pytest tests/test_dpg_k.py` fails 2 tests
— `test_auto_context_has_one_sink_per_class_and_no_local_violations` and
`test_execution_trace_graph_preserves_long_case_order` — both of which fit at `context_order` values
that reach this code path.

# Reproduction

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg.explainer import DPGExplainer

iris = load_iris()
model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(iris.data, iris.target)

dpg_config = {
    "dpg": {
        "default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "execution_trace", "context_order": "auto"},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(iris.feature_names), target_names=[str(c) for c in iris.target_names], dpg_config=dpg_config)
explainer.fit(iris.data)
```

Run with `uv run python` from the repo root. Real output at HEAD `276a503`:

```text
DPG initialized with perc_var=1e-09, decimal_threshold=3, n_jobs=1, graph_construction_mode=execution_trace, context_order=auto
...
Total of paths: 750
Building DPG...
Traceback (most recent call last):
  File "<string>", line 17, in <module>
    explainer.fit(iris.data)
  File "/Users/vinicius/Projects/DPG/dpg/explainer.py", line 181, in fit
    self._dot = self._builder.fit(X)
  File "/Users/vinicius/Projects/DPG/dpg/core.py", line 283, in fit
    dfg = self.discover_dfg_context(log_df, self._resolved_context_order)
  File "/Users/vinicius/Projects/DPG/dpg/core.py", line 636, in discover_dfg_context
    for source, target in pairwise(nodes, nodes[1:]):
TypeError: pairwise expected 1 argument, got 2
```

(An explicit `context_order=2` in the config reproduces the identical crash; `"auto"` on this dataset
happens to resolve above 1, which is why it also triggers here.)

# Workaround

None from calling code — every path through `discover_dfg_context` with `context_order > 1` hits this
line. Any DPG-k feature (contextual graphs, `get_predicate_lrc`, the deprecation path documented in
[trace-consistent-lrc-deprecation](trace-consistent-lrc-deprecation.md)) that requires an actual
`context_order > 1` fit is currently unreachable at HEAD. This page's own crash reproduction and any
other page's example needing a genuine `k > 1` graph were instead demonstrated against a temporary
worktree checked out at `f16a977` (the 0.3.0 release commit, before this regression), removed after use.

The one-line fix, for reference (not applied — this page documents, it does not patch):
`pairwise(nodes, nodes[1:])` → `pairwise(nodes)`.

# File:line citations (HEAD `276a503`)

- `dpg/core.py:636` — the broken call, inside `discover_dfg_context`.
- `dpg/core.py:8` — `from itertools import pairwise` (added by the same commit).
- `dpg/core.py:629-630` — `discover_dfg_context`'s `context_order <= 1` early return, which is why
  `context_order=1` never reaches the broken line.
- Introduced by commit `6df6e20` ("chore: ruff fixes"), which replaced
  `for source, target in zip(nodes, nodes[1:]):` with the broken `pairwise` call.
- `uv run pytest tests/test_dpg_k.py` — 2 failing tests exercise this path:
  `test_auto_context_has_one_sink_per_class_and_no_local_violations`,
  `test_execution_trace_graph_preserves_long_case_order`.

# Related

- [trace-consistent-lrc-deprecation](trace-consistent-lrc-deprecation.md)
- [context-order-validation-errors](context-order-validation-errors.md)
- [Graph construction modes](/conventions/graph-construction-modes.md)
- [dpg.context_order](/modules/dpg-context-order.md)
