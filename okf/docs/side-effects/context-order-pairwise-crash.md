---
type: Side Effect
title: discover_dfg_context crashed for every context_order > 1 (fixed)
description: A "ruff fixes" commit replaced zip(nodes, nodes[1:]) with itertools.pairwise(nodes, nodes[1:]) — but itertools.pairwise takes exactly one iterable. Every execution_trace fit with context_order > 1 (including "auto" resolving above 1) raised TypeError before any graph was built. fixed on `feature/first_runs` by dropping the second argument.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [regression, crash, fixed, context_order, dpg-k, itertools, ruff]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: fixed
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
for source, target in pairwise(nodes, nodes[1:]):  # broken, at 6df6e20..276a503
```

`itertools.pairwise` takes exactly one iterable and yields consecutive pairs from it — it is not a
drop-in replacement for `zip(seq, seq[1:])`'s two-argument call shape. Calling it with two positional
arguments raised `TypeError` immediately, before a single edge was added to `dfg`.

**This is now fixed on `feature/first_runs`** on `feature/first_runs` after `276a503`: `dpg/core.py:636` calls `pairwise(nodes)`, and `tests/test_dpg_k.py:95`'s own use of the
same broken two-argument shape was fixed to `pairwise(labels)`.

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
(the default) was never affected: `discover_dfg_context` early-returns to
`discover_dfg_execution_trace(log)` before reaching the `pairwise` call. With the fix applied, all
`context_order` values now work.

`uv run pytest tests/test_dpg_k.py` confirms the fix: all 7 tests pass, including
`test_auto_context_has_one_sink_per_class_and_no_local_violations` and
`test_execution_trace_graph_preserves_long_case_order`, which previously failed on this exact path.

# The fix

```diff
-            for source, target in pairwise(nodes, nodes[1:]):
+            for source, target in pairwise(nodes):
```

(and, in the test file's own independent occurrence of the same bug)

```diff
-        (source, target): 1 for source, target in pairwise(labels, labels[1:])
+        (source, target): 1 for source, target in pairwise(labels)
```

# Reproduction (confirms the fix)

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg.explainer import DPGExplainer

iris = load_iris()
model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42).fit(iris.data, iris.target)

dpg_config = {
    "dpg": {
        "default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "execution_trace", "context_order": 2},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(iris.feature_names), target_names=[str(c) for c in iris.target_names], dpg_config=dpg_config)
explainer.fit(iris.data)
print("resolved context order:", explainer._builder.get_context_order())
print("FIT SUCCEEDED, pairwise crash is fixed")
```

Run with `uv run python -c "..."` from the repo root with the fix applied (after `276a503`). Actual output:

```text
...
Total of paths: 450
Building DPG...
Extracting graph...
resolved context order: 2
FIT SUCCEEDED, pairwise crash is fixed
```

No `TypeError`; the fit completes and returns a `context_order=2` graph. `uv run pytest
tests/test_dpg_k.py -q` also confirms: `7 passed`.

# File:line citations

- `dpg/core.py:636` — the fixed call, inside `discover_dfg_context`: `pairwise(nodes)`.
- `dpg/core.py:8` — `from itertools import pairwise` (added by `6df6e20`, unchanged by the fix).
- `dpg/core.py:629-630` — `discover_dfg_context`'s `context_order <= 1` early return, which is why
  `context_order=1` never reached the broken line even before the fix.
- Bug introduced by commit `6df6e20` ("chore: ruff fixes"), which replaced a working `zip(nodes,
  nodes[1:])`. Fixed on `feature/first_runs` after `276a503`.
- `tests/test_dpg_k.py:95` — the test file's own independent instance of the same two-argument
  `pairwise` misuse, fixed alongside `core.py`.
- `uv run pytest tests/test_dpg_k.py` — `7 passed` after the fix (was 2 failures:
  `test_auto_context_has_one_sink_per_class_and_no_local_violations`,
  `test_execution_trace_graph_preserves_long_case_order`).

# Related

- [trace-consistent-lrc-deprecation](trace-consistent-lrc-deprecation.md)
- [context-order-validation-errors](context-order-validation-errors.md)
- [Graph construction modes](/conventions/graph-construction-modes.md)
- [dpg.context_order](/modules/dpg-context-order.md)
