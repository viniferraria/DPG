---
type: Side Effect
title: explain_local crashes for every model
description: A ruff-driven cleanup commented out the node_lookup local variable in DPGExplainer.explain_local but left _trace_tree_path's node_lookup parameter required, so every call to explain_local (and everything built on it) now raises TypeError at HEAD.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [regression, crash, local-explanation, explain_local, ruff]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

`DPGExplainer.explain_local` is unconditionally broken at HEAD (`276a503`). Calling it — on any
model, in any graph construction mode — raises:

```text
TypeError: DPGExplainer._trace_tree_path() missing 1 required positional argument: 'node_lookup'
```

This is a regression introduced **after** the 0.3.0 release commit (`f16a977`), not a pre-existing
0.2.0 limitation. `explain_local` worked at `f16a977`.

# Why

Commit `6df6e20` ("chore: ruff fixes") commented out the `node_lookup` local variable in
`explain_local` (ruff had presumably flagged it as unused, since `_trace_tree_path` had been
rewritten to route via sklearn's own `decision_path`/`apply` and no longer needs a label→node-id
lookup — see [exact-routing-label-shift](exact-routing-label-shift.md)):

```python
# dpg/explainer.py:268
# node_lookup = {label: node_id for node_id, label in self._require_nodes()}
```

But the commit did not remove the now-dead `node_lookup: dict[str, str]` parameter from
`_trace_tree_path`'s signature (`dpg/explainer.py:651`), and the call site at
`dpg/explainer.py:274-281` still doesn't pass it — so every call to `_trace_tree_path` is missing a
required positional argument. The cleanup removed the *producer* of the value without removing the
*consumer*'s requirement for it.

# Who is affected

Everyone who calls `DPGExplainer.explain_local()` directly, or anything that calls it internally:

- `DPGExplainer.evaluate_faithfulness` (every sample's `explain_local` call fails; caught per-sample
  by its `except Exception`, so faithfulness evaluation silently reports `n_local_failures == n_samples`
  and raises `DPGExplanationError.all_local_explanations_failed()` — see
  [faithfulness evaluation](/concepts/faithfulness-evaluation.md)).
- `DPGExplainer.plot_local_on_dpg` and any other local-explanation plot.
- `experiments/local_explanation/` — the only experiment suite covered by a test
  (`tests/test_smoke.py`) — and any script under `examples/` that traces a single sample.

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
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(iris.feature_names), target_names=[str(c) for c in iris.target_names], dpg_config=dpg_config)
explainer.fit(iris.data)
explainer.explain_local(iris.data[0])
```

Run with `uv run python` from the repo root. Real output at HEAD:

```text
Traceback (most recent call last):
  File "<string>", line 18, in <module>
    explainer.explain_local(iris.data[0])
  File "/Users/vinicius/Projects/DPG/dpg/explainer.py", line 274, in explain_local
    path = self._trace_tree_path(
        tree=tree,
    ...<4 lines>...
        validate_graph=validate_graph,
    )
TypeError: DPGExplainer._trace_tree_path() missing 1 required positional argument: 'node_lookup'
```

# Workaround

None from calling code — the bug is in the signature/call-site mismatch itself. A fix would either
restore the `node_lookup = {label: node_id for node_id, label in self._require_nodes()}` line and
pass `node_lookup=node_lookup` at the `_trace_tree_path` call site, or (since `_trace_tree_path` no
longer uses `node_lookup` for routing after the `decision_path` rewrite — see
[exact-routing-label-shift](exact-routing-label-shift.md)) remove the now-unused parameter from
`_trace_tree_path`'s signature entirely. This page documents the break; no code change was made.

# File:line citations (HEAD `276a503`)

- `dpg/explainer.py:268` — commented-out `node_lookup` assignment.
- `dpg/explainer.py:274-281` — `_trace_tree_path(...)` call, missing `node_lookup=`.
- `dpg/explainer.py:645-653` — `_trace_tree_path` signature still requiring `node_lookup`.
- Introduced by commit `6df6e20` ("chore: ruff fixes"), after the 0.3.0 release merge `f16a977`.

# Related

- [Exact routing / label shift](exact-routing-label-shift.md) — why `_trace_tree_path` no longer
  needs a label lookup for routing.
- [Faithfulness evaluation](/concepts/faithfulness-evaluation.md) — the composite metric this breaks
  end-to-end.
- [dpg.explainer](/modules/dpg-explainer.md)
