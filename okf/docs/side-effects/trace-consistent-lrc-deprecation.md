---
type: Side Effect
title: get_trace_consistent_lrc() is deprecated at context_order > 1
description: get_trace_consistent_lrc() still works at k=1 but emits a DeprecationWarning for contextual graphs; get_predicate_lrc(graph) is the k>1 replacement, and it sums per-predicate rather than tracking single-trace reach.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [deprecation, lrc, context_order, dpg-k, node-metrics]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

0.2.0 added `get_trace_consistent_lrc()`: a local-reaching-centrality-like score computed only from
same-trace reach (never credit gained purely from pooling edges across different traces). 0.3.0 adds
context-aware graph construction (`context_order` / DPG-k — see
[Graph construction modes](/conventions/graph-construction-modes.md)), under which a single predicate
label can occupy **multiple** contextual nodes (one per distinct recent-history window). At
`context_order > 1`, `get_trace_consistent_lrc()` is deprecated in favor of a new method,
`get_predicate_lrc(graph)`, which sums the pooled-graph NetworkX local reaching centrality of every
contextual node sharing a predicate:

```python
# dpg/core.py:772-800
def get_trace_consistent_lrc(self) -> dict[str, float]:
    """
    .. deprecated:: 0.3.0
       For contextual graphs (``context_order > 1``), use
       :meth:`get_predicate_lrc` on the fitted graph.  This getter remains
       supported for legacy k=1 execution-trace graphs.
    ...
    """
    if self.get_context_order() > 1:
        warnings.warn(
            "get_trace_consistent_lrc() is deprecated for context_order > 1; "
            "use get_predicate_lrc(graph) instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return dict(self._trace_consistent_lrc)
```

```python
# dpg/core.py:676-689
def get_predicate_lrc(self, graph: Any) -> dict[str, float]:
    """Aggregate unweighted node LRC scores back to predicate labels.

    At k>1 a predicate can occupy multiple contextual nodes.  The shipped
    aggregation is a sum, so predicates receive credit for every context
    in which they occur.
    """
    scores: dict[str, float] = defaultdict(float)
    for node, data in graph.nodes(data=True):
        label = data.get("predicate")
        if label is None or not self._is_predicate_label(label):
            continue
        scores[label] += float(nx.local_reaching_centrality(graph, node, weight=None))
    return dict(scores)
```

`DPGExplainer._get_node_metrics` (`dpg/explainer.py:767-780`) already routes automatically — it calls
`get_trace_consistent_lrc()` only at `context_order == 1` and `get_predicate_lrc(self._graph)` at
`context_order > 1` — so the warning fires only when code calls `get_trace_consistent_lrc()` directly
at `context_order > 1`, not through the normal `DPGExplainer` node-metrics path.

# Why

The trace-consistent score's definition ("fraction of labels found downstream of it within at least
one single observed sample-tree execution") is defined over *predicate labels*, which is exactly node
identity at `k=1`. At `k>1`, node identity is `(predicate, recent-history context)`, so a predicate can
have several trace-consistent scores (one per context) with no single well-defined scalar — hence a
different aggregation (`get_predicate_lrc`'s sum-of-contexts) rather than reusing the old getter.

# Who is affected

Anyone calling `DecisionPredicateGraph.get_trace_consistent_lrc()` directly (not through
`DPGExplainer`) on a graph fit with `context_order > 1` (explicit integer, or `"auto"` resolving to a
value greater than 1).

# Reproduction

At HEAD `276a503`, `context_order > 1` could not be fit due to an unrelated regression — see
[context-order-pairwise-crash](context-order-pairwise-crash.md) (`discover_dfg_context` called
`itertools.pairwise` with two arguments, which raised `TypeError` before any graph was built). To
demonstrate the deprecation itself at the time, this example was run in a temporary git worktree
checked out at `f16a977` (the 0.3.0 release merge commit, before the `pairwise` regression) and
removed afterward — the deprecation/getter behavior itself was otherwise unchanged through the
parallel worker's owned files at HEAD. **That regression is now fixed on `feature/first_runs`** (see
[context-order-pairwise-crash](context-order-pairwise-crash.md)), so this example can now also be run
directly against `feature/first_runs` without a worktree; the output shown below is unaffected either
way, since the deprecation logic itself never changed:

```python
import warnings
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from dpg.explainer import DPGExplainer

iris = load_iris()
model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(iris.data, iris.target)

dpg_config = {
    "dpg": {
        "default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "execution_trace", "context_order": 2},
        "visualization": {},
    },
}
explainer = DPGExplainer(model, list(iris.feature_names), target_names=[str(c) for c in iris.target_names], dpg_config=dpg_config)
explainer.fit(iris.data)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lrc = explainer._builder.get_trace_consistent_lrc()
    for w in caught:
        print("WARNING:", w.category.__name__, "-", str(w.message))

predicate_lrc = explainer._builder.get_predicate_lrc(explainer._graph)
print("get_trace_consistent_lrc() sample:", list(lrc.items())[:3])
print("get_predicate_lrc() sample:       ", list(predicate_lrc.items())[:3])
```

Real output (run with the worktree's `dpg/` on `sys.path`, via the main venv's interpreter, from that
worktree's directory):

```text
resolved context order: 2
WARNING: DeprecationWarning - get_trace_consistent_lrc() is deprecated for context_order > 1; use get_predicate_lrc(graph) instead
get_trace_consistent_lrc() sample: [('petal width (cm) <= 0.8', 0.038461538461538464), ('petal length (cm) <= 2.45', 0.038461538461538464), ('petal length (cm) <= 2.6', 0.038461538461538464)]
get_predicate_lrc() sample: [('petal width (cm) <= 1.75', 0.23529411764705882), ('petal length (cm) > 5.4', 0.029411764705882353), ('petal width (cm) > 1.75', 0.20588235294117646)]
```

Note the two methods also disagree on which predicates even appear near the top — they are genuinely
different scores, not just a renamed getter.

# Workaround

Call `get_predicate_lrc(graph)` (passing the fitted `nx.DiGraph`, e.g. `DPGExplainer._graph` or the
`graph` returned by `explain_global()`) instead of `get_trace_consistent_lrc()` whenever
`context_order > 1`. At `context_order == 1`, `get_trace_consistent_lrc()` remains fully supported —
no warning, no change in meaning.

# File:line citations (HEAD `276a503`, code shared with `f16a977`)

- `dpg/core.py:676-689` — `get_predicate_lrc`.
- `dpg/core.py:772-800` — `get_trace_consistent_lrc`, deprecation warning at `context_order > 1`.
- `dpg/explainer.py:767-780` — `DPGExplainer._get_node_metrics`, automatic routing between the two.
- CHANGELOG.md 0.3.0 "Compatibility and limitations": "`get_trace_consistent_lrc()` remains available
  for k=1 and is deprecated for contextual graphs; k>1 aggregates ordinary unweighted node LRC by
  predicate."

# Related

- [context-order-pairwise-crash](context-order-pairwise-crash.md)
- [Graph construction modes](/conventions/graph-construction-modes.md)
- [dpg.core](/modules/dpg-core.md)
