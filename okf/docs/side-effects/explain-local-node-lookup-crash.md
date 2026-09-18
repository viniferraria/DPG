---
type: Side Effect
title: explain_local crashed for every model (fixed)
description: A ruff-driven cleanup commented out the node_lookup local variable in DPGExplainer.explain_local but left _trace_tree_path's node_lookup parameter required, so every call to explain_local raised TypeError. fixed on `feature/first_runs` by removing the now-vestigial parameter entirely.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [regression, crash, fixed, local-explanation, explain_local, ruff]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: fixed
---

# What changed

`DPGExplainer.explain_local` was unconditionally broken between commit `6df6e20` and the fix described
below. Calling it — on any model, in any graph construction mode — raised:

```text
TypeError: DPGExplainer._trace_tree_path() missing 1 required positional argument: 'node_lookup'
```

This was a regression introduced **after** the 0.3.0 release commit (`f16a977`), not a pre-existing
0.2.0 limitation. `explain_local` worked at `f16a977`.

**This is now fixed on `feature/first_runs`** on `feature/first_runs` after `276a503`: `explain_local` calls `explain_local` successfully again.

# Why

Commit `6df6e20` ("chore: ruff fixes") commented out the `node_lookup` local variable in
`explain_local` (ruff had presumably flagged it as unused, since `_trace_tree_path` had been
rewritten to route via sklearn's own `decision_path`/`apply` and no longer needs a label→node-id
lookup — see [exact-routing-label-shift](exact-routing-label-shift.md)):

```python
# node_lookup = {label: node_id for node_id, label in self._require_nodes()}  # removed by the fix
```

But the commit did not remove the now-dead `node_lookup: dict[str, str]` parameter from
`_trace_tree_path`'s signature, and the call site still didn't pass it — so every call to
`_trace_tree_path` was missing a required positional argument. The cleanup removed the *producer* of
the value without removing the *consumer*'s requirement for it.

The fix takes the second option `_trace_tree_path`'s own docstring implied was available: since the
method no longer uses `node_lookup` for routing at all (see
[exact-routing-label-shift](exact-routing-label-shift.md)), the parameter is deleted from the
signature rather than resurrected. The one-line comment above the old `node_lookup =` assignment is
also gone — there is nothing left for it to explain.

# Who is affected (while the bug was present)

Everyone who called `DPGExplainer.explain_local()` directly, or anything that called it internally:

- `DPGExplainer.evaluate_faithfulness` (every sample's `explain_local` call failed; caught per-sample
  by its `except Exception`, so faithfulness evaluation silently reported `n_local_failures == n_samples`
  and raised `DPGExplanationError.all_local_explanations_failed()` — see
  [faithfulness evaluation](/concepts/faithfulness-evaluation.md)).
- `DPGExplainer.plot_local_on_dpg` and any other local-explanation plot.
- `experiments/local_explanation/` — the only experiment suite covered by a test
  (`tests/test_smoke.py`) — and any script under `examples/` that traces a single sample.

With the fix applied, all of the above work again.

# The fix

```diff
-        # node_lookup = {label: node_id for node_id, label in self._require_nodes()}
         node_metrics_lookup = self._get_node_metrics_lookup()
```

```diff
     def _trace_tree_path(
         self,
         ...
-        node_lookup: dict[str, str],
         node_metrics_lookup: dict[str, dict[str, Any]],
         validate_graph: bool,
     ) -> DPGTreePathExplanation:
```

The call site at `explain_local` already passed no `node_lookup=` argument, so no change was needed
there — the mismatch is resolved entirely by deleting the unused parameter from the signature.

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
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(iris.feature_names), target_names=[str(c) for c in iris.target_names], dpg_config=dpg_config)
explainer.fit(iris.data)
local = explainer.explain_local(iris.data[0])
print("EXPLAIN_LOCAL SUCCEEDED, node_lookup crash is fixed")
print("majority_vote:", local.majority_vote)
print("num tree_paths:", len(local.tree_paths))
```

Run with `uv run python -c "..."` from the repo root with the fix applied (after `276a503`). Actual output:

```text
...
Extracting graph...
EXPLAIN_LOCAL SUCCEEDED, node_lookup crash is fixed
majority_vote: setosa
num tree_paths: 3
```

No `TypeError`; `explain_local` returns a populated `DPGLocalExplanation` with one `DPGTreePathExplanation`
per tree.

# File:line citations (on `feature/first_runs`, after `276a503`)

- `dpg/explainer.py:268` (previously) — the commented-out `node_lookup` assignment inside
  `explain_local` is gone entirely; every line from there on shifted by -1 relative to `276a503`.
- `dpg/explainer.py:645-652` — `_trace_tree_path` signature, now without `node_lookup`
  (was `dpg/explainer.py:645-653` at `276a503`, one line longer).
- Bug introduced by commit `6df6e20` ("chore: ruff fixes"), after the 0.3.0 release merge `f16a977`.
  Fixed on `feature/first_runs` after `276a503`.

# Related

- [Exact routing / label shift](exact-routing-label-shift.md) — why `_trace_tree_path` no longer
  needs a label lookup for routing.
- [Faithfulness evaluation](/concepts/faithfulness-evaluation.md) — the composite metric this used to break
  end-to-end.
- [dpg.explainer](/modules/dpg-explainer.md)
