---
type: Side Effect
title: extract_communities on a regressor now raises a clear ValueError
description: GraphMetrics.extract_communities requires at least one classifier "Class " sink to anchor its absorbing-Markov-chain clustering. 0.3.0 added an explicit guard that raises ValueError for regression DPGs, replacing an opaque numpy.linalg.LinAlgError from a singular (I - Q) matrix.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/graph.py
tags: [error-handling, communities, regression, classifier-only, markov-chain]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

`GraphMetrics.extract_communities` clusters nodes by absorption probability toward class sinks, using
an absorbing Markov chain over the pooled DPG. That computation requires at least one transient state
(a node that isn't a class sink) and inverts `(I - Q)`, where `Q` is the transition matrix restricted
to transient states. A regression DPG has `"Pred <value>"` leaves instead of `"Class <name>"` sinks —
every node is transient, `(I - Q)` is singular, and before 0.3.0 this raised numpy's opaque
`numpy.linalg.LinAlgError` deep inside the linear-algebra call. 0.3.0 added an explicit guard that
raises a clear, actionable `ValueError` instead:

```python
# metrics/graph.py:250-263
class_nodes = {i[0]: i[1] for i in nodes_list if 'Class' in i[1]}
if not class_nodes:
    # The absorbing-chain clustering below requires at least one classifier
    # sink ("Class ..." leaf) to anchor the Markov chain; a regression DPG
    # has "Pred ..." leaves instead and none is a class sink, which makes
    # every node transient and (I - Q) singular. Regression community/class-
    # boundary extraction is out of scope for 0.3.0 (see CHANGELOG); raise a
    # clear error instead of letting numpy fail with an opaque LinAlgError.
    raise ValueError(
        "extract_communities requires a classifier DPG with at least one "
        "'Class ' sink node; regression DPGs ('Pred ' leaves) are not "
        "supported by this community extraction in 0.3.0."
    )
```

`DPGExplainer.explain_global(communities=True)` calls straight through to this
(`dpg/explainer.py:221`), so the same clear error surfaces from the high-level API too.

# Why

Community/class-boundary extraction is explicitly out of scope for regression DPGs in 0.3.0 — see the
CHANGELOG's "Compatibility and limitations" section. Rather than leave that gap producing an internal
linear-algebra crash with no mention of DPG semantics, 0.3.0 turns it into a documented,
domain-specific error raised before any matrix is built.

# Who is affected

Anyone calling `DPGExplainer.explain_global(communities=True)`, or `GraphMetrics.extract_communities`
directly, on a DPG built from a regressor (`RandomForestRegressor`, `GradientBoostingRegressor`, etc.).
`extract_class_boundaries` remains classifier-only too, but returns an empty result for regressors
rather than raising (see CHANGELOG: "`class_boundaries` and `communities` remain classifier-only
features").

# Reproduction

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from dpg.explainer import DPGExplainer

data = load_diabetes()
model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42).fit(data.data, data.target)
dpg_config = {
    "dpg": {
        "default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(data.feature_names), dpg_config=dpg_config)
explainer.fit(data.data)
explainer.explain_global(communities=True)
```

Run with `uv run python` from the repo root. Real output:

```text
ValueError: extract_communities requires a classifier DPG with at least one 'Class ' sink node; regression DPGs ('Pred ' leaves) are not supported by this community extraction in 0.3.0.
```

# Workaround

Don't pass `communities=True` (or `clusters`/`threshold_clusters` in the `examples/` CLIs) when
explaining a regression model. `explain_global()` without `communities=True` works normally on
regressors and returns node/edge metrics as usual; only the community/absorbing-chain step is
unsupported.

# File:line citations (HEAD `276a503`)

- `metrics/graph.py:250-263` — the guard and `ValueError` message, inside `extract_communities`.
- `dpg/explainer.py:189-221` — `DPGExplainer.explain_global`, which calls `extract_communities` when
  `communities=True`.
- CHANGELOG.md 0.3.0 "Compatibility and limitations": "`class_boundaries` and `communities` remain
  classifier-only features; calling `DPGExplainer.explain_global(communities=True)` on a regressor now
  raises a clear `ValueError` instead of an internal `numpy.linalg.LinAlgError`."

# Related

- [regression-sink-collisions](regression-sink-collisions.md)
- [metrics.graph](/modules/metrics-graph.md)
