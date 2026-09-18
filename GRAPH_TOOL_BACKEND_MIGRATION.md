# Migrating DPG's graph-tool node-metrics backend into dpg_v

**Source:** `DPG` repo, branch `new_features`, commit `563eecd3df0370869a2bf0d7d6ea8f63ff2264d4` ("first runs", viniferraria, 2026-07-01).
**Parent / shared ancestor with dpg_v:** `f16a9779819e1e02609a64d04de6c8a123d18439` (DPG 0.3.0 core release merge).
**Target:** `dpg_v` repo, branch `feature/first_runs`, HEAD `e11aded`.
**Prepared:** 2026-09-14, by an orchestrator session with two Sonnet analysis agents (one per repo). All analysis was read-only; nothing in either repo was modified.

---

## 1. Verdict

**Yes, the change can be migrated, but it should not be cherry-picked.** The commit is an incomplete work-in-progress and is broken as committed (see §4). What is worth taking is the *algorithmic core*: four graph-tool functions in `metrics/nodes.py` that are numerically equivalent to NetworkX (verified to ~1e-16). Everything else in the commit is either noise (pickles, logs, lockfiles, experiment scripts) or actively harmful (duplicate class definition, unconditional `graph_tool` import that breaks the pip/uv install, dead code).

The migration is also a **different shape than "add a backend"**: dpg_v already left NetworkX behind and runs an **igraph** backend in `metrics/nodes.py` (`_nx_to_igraph`, `calc_betweenness_centrality`, `calc_local_reaching_centrality`), plus two extra metrics (closeness, harmonic) that DPG's commit does not have. So the real question is *igraph → graph-tool swap, or graph-tool as a second selectable backend*. The recommendation (§6) is to add graph-tool as an **optional backend behind a small selector**, keep igraph as the default (it is pip-installable and already in `uv.lock`), and keep the 9-column DataFrame contract untouched.

Effort estimate: small for degree and betweenness (line-for-line), moderate for local reaching centrality (different code shape, working template exists), net-new for closeness/harmonic on graph-tool (no template in DPG). The two hard parts are non-code: packaging graph-tool in a uv-only project/CI, and two pre-existing bugs in dpg_v that must be fixed before any test can run (§5).

---

## 2. What the DPG commit changes

`git show 563eecd --stat` touches 24 files. Only four are code:

| File | Change | Keep? |
|---|---|---|
| `metrics/nodes.py` | Appends a graph-tool implementation (`_nx_to_graph_tool`, `calc_node_metrics`, `calc_betweenness_centrality`, `calc_local_reaching_centrality`, a second `class NodeMetrics`) *below* the original NetworkX class, so the file defines `NodeMetrics` twice. The second definition wins at import time. | Algorithms yes, file layout no |
| `dpg/utils.py` | Adds `get_logger`, `log_timer` decorator, and a `descendants(g, src)` helper using `graph_tool.search.bfs_iterator`. Removes the `networkx`/`pandas`/`numpy` imports (safe, nothing else in the file used them). Adds an **unconditional** `from graph_tool import search` at module scope. | Mostly no (dpg_v already has logging helpers; `descendants` is dead code) |
| `pyproject.toml` | Converts from `[tool.poetry]` to PEP 621 `[project]`; adds `igraph>=1.0.0` to pip deps (never imported anywhere); adds a `[tool.pixi.*]` workspace with `graph-tool = ">=2.98,<3"` from conda-forge. graph-tool is **not** in `[project.dependencies]` and **not** in `uv.lock`. | Pattern yes, verbatim no |
| `main.py` | New experiment runner: fits a RandomForest on two CSV scenarios, pickles the explainer and explanation, and dumps `explanation.node_metrics` to `datasets/node_metrics.csv`. Reads the columns `Node, Degree, In degree nodes, Out degree nodes, Betweenness centrality, Local reaching centrality, Label`. | No (experiment script, not library code) |

Noise files (do not migrate): five `*.pkl_<timestamp>` binaries (~75 MB), `dpg_explainer.log`, `output.log`, `pixi.lock`, `uv.lock`, `.gitattributes`, `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, `dataset_scenario{1,2}{,_bin}.py`, `import numpy as np.py`.

### 2.1 Semantic comparison: NetworkX (ancestor) vs graph-tool (DPG commit) vs igraph (dpg_v today)

| Metric | Ancestor `f16a977` (NetworkX) | DPG `563eecd` (graph-tool) | dpg_v `e11aded` (igraph) |
|---|---|---|---|
| Degree / in / out | `dpg_model.in_degree`, `out_degree`, summed | unchanged, pure NetworkX (`calc_node_metrics`) | unchanged, pure NetworkX (`calc_node_metrics`, nodes.py:106-117) |
| Betweenness | `nx.betweenness_centrality(G, k=sample_size, normalized=True, weight='weight', endpoints=False)` | `graph_tool.centrality.betweenness(g, weight=ep["weight"], norm=False)` then `/ ((n-1)(n-2))`. Weight treated as distance, same as NetworkX. Keyed by int vertex index. | `g.betweenness(directed=True, normalized=False, weights="weight")` then `/ ((n-1)(n-2))`. Keyed by int index. (nodes.py:120-130) |
| Local reaching centrality | `nx.local_reaching_centrality(G, node, weight='weight')` per node | Hand-rolled replica: distance = `total_weight / w`, `shortest_distance(..., pred_map=True)` per source, walk predecessors to rebuild each path, average original edge weight per path, sum, `/ (total_weight/num_edges) / (n-1)`. Unreachable test: `dist_map[t] >= 2147483647`. | Same algorithm using `g.get_shortest_paths(i, mode="out", weights="distance", output="vpath")`; raises `DPGMetricError.non_positive_lrc_weight()` (nodes.py:133-180) |
| Closeness | absent | absent | `calc_closeness_centrality` via `g.distances(...)` on inverted weights (nodes.py:183-216) |
| Harmonic | absent | absent | `calc_harmonic_centrality` (nodes.py:219-241) |
| `trace_lrc_by_label` param | present, used inside LRC loop | **dropped** | **dropped** (docstring still mentions it) |
| Returned columns | 7 | 7 | **9** (adds Closeness, Harmonic) |
| Logging | none | `get_logger`/`log_timer` in `dpg/utils.py`, imported into metrics | `get_logger`/`log_timer` defined locally in `metrics/nodes.py:13-75` |

**Numerical verification (done by the DPG analysis agent inside DPG's pixi env, graph-tool 2.98):** the graph-tool betweenness and LRC functions were run head-to-head against `nx.betweenness_centrality` / `nx.local_reaching_centrality` on a random weighted 12-node digraph. Max absolute differences: betweenness 5.55e-17, LRC 2.22e-16. The algorithms are correct.

---

## 3. Verbatim diff of the code files in the DPG commit

Generated with `git show 563eecd -- metrics/nodes.py dpg/utils.py` in the DPG repo; the relevant `pyproject.toml` hunks follow separately. `main.py` is omitted (experiment script, not library code); see §2 for what it does.

```diff
commit 563eecd3df0370869a2bf0d7d6ea8f63ff2264d4
Author: viniferraria <39192461+viniferraria@users.noreply.github.com>
Date:   Wed Jul 1 16:30:34 2026 -0300

    first runs

diff --git a/dpg/utils.py b/dpg/utils.py
index d48103c..8e0c504 100644
--- a/dpg/utils.py
+++ b/dpg/utils.py
@@ -1,11 +1,77 @@
+from functools import wraps
 import os
 import re
 import shutil
+import logging
+import time
 import yaml
 from graphviz import Digraph
-import networkx as nx
-import pandas as pd
-import numpy as np
+from graph_tool import search
+
+
+def get_logger(name, log_file=None):
+    """
+    Factory function to create and configure a logger.
+    
+    Args:
+        name: Logger name (typically __name__)
+        log_file: Optional log file path. If provided, logs will also be written to this file.
+    
+    Returns:
+        Configured logger instance
+    """
+    logger = logging.getLogger(name)
+    logger.setLevel(logging.INFO)
+    
+    # Avoid adding duplicate handlers
+    if logger.hasHandlers():
+        return logger
+    
+    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
+    
+    # Console handler
+    console_handler = logging.StreamHandler()
+    console_handler.setFormatter(formatter)
+    logger.addHandler(console_handler)
+    
+    # File handler (if specified)
+    if log_file:
+        file_handler = logging.FileHandler(log_file)
+        file_handler.setFormatter(formatter)
+        logger.addHandler(file_handler)
+    
+    return logger
+
+logger = get_logger(__name__)
+
+
+def log_timer(func):
+    """
+    Decorator to log the execution time of a function.
+
+    Args:
+        func: The function to be decorated.
+
+    Returns:
+        Wrapped function that logs execution time.
+    """
+    logger.info(f"Decorating function {func.__name__} with log_timer")
+
+    @wraps(func)
+    def wrapper(self, *args, **kwargs):
+        start_time = time.time()
+        result = func(self, *args, **kwargs)
+        end_time = time.time()
+        logger.info(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
+        return result
+
+    return wrapper
+
+def descendants(g, src):
+    visited = set()
+    for e in search.bfs_iterator(g, g.vertex(src)):
+        visited.add(int(e.target()))
+    return visited
 
 
 def highlight_class_node(dot, dpg_config=None):
diff --git a/metrics/nodes.py b/metrics/nodes.py
index b09fa25..5a87430 100644
--- a/metrics/nodes.py
+++ b/metrics/nodes.py
@@ -54,4 +54,182 @@ class NodeMetrics:
         }
         df_data_node = pd.DataFrame(data_node).set_index('Node')
         df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
-        return pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
\ No newline at end of file
+        return pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
+import warnings
+from typing import Any, Dict, List, Tuple
+
+import graph_tool as gt
+from graph_tool.centrality import betweenness
+from graph_tool.topology import shortest_distance
+import networkx as nx
+import pandas as pd
+
+from dpg.utils import log_timer
+
+
+@log_timer
+def _nx_to_graph_tool(nx_graph: nx.DiGraph) -> Tuple[gt.Graph, List[str]]:
+    """Convert a NetworkX DiGraph to graph-tool, preserving node ID mapping.
+
+    Returns:
+        Tuple of (graph_tool.Graph, node_ids) where node_ids[i] is the original
+        NetworkX node ID for graph-tool vertex index *i*.
+    """
+    node_ids = list(nx_graph.nodes())
+    node_to_idx = {n: i for i, n in enumerate(node_ids)}
+
+    g = gt.Graph(directed=True)
+    g.add_vertex(len(node_ids))
+
+    weight_prop = g.new_edge_property("double")
+
+    for u, v, data in nx_graph.edges(data=True):
+        e = g.add_edge(node_to_idx[u], node_to_idx[v])
+        weight_prop[e] = data.get("weight", 1.0)
+
+    g.ep["weight"] = weight_prop
+    return g, node_ids
+
+
+@log_timer
+def calc_node_metrics(
+    dpg_model: nx.DiGraph,
+) -> Tuple[dict[str, int], Dict[str, int], Dict[str, int]]:
+    in_nodes = {}
+    out_nodes = {}
+    degree = {}
+    for node in dpg_model.nodes():
+        in_nodes[node] = dpg_model.in_degree(node)
+        out_nodes[node] = dpg_model.out_degree(node)
+        degree[node] = in_nodes[node] + out_nodes[node]
+    return in_nodes, out_nodes, degree
+
+
+@log_timer
+def calc_betweenness_centrality(gt_graph: gt.Graph) -> Dict[int, float]:
+    """Compute betweenness centrality for a graph-tool graph."""
+    n = gt_graph.num_vertices()
+    # betweenness() with norm=False returns unnormalised values.
+    vp_bc, _ep_bc = betweenness(gt_graph, weight=gt_graph.ep["weight"], norm=False)
+    # Normalise to match NetworkX convention: bc / ((n-1)*(n-2))
+    norm = (n - 1) * (n - 2) if n > 2 else 1.0
+    betweenness_centrality = {int(v): vp_bc[v] / norm for v in gt_graph.vertices()}
+    return betweenness_centrality
+
+
+@log_timer
+def calc_local_reaching_centrality(
+    gt_graph: gt.Graph, node_ids: List[str]
+) -> Dict[str, float]:
+    """Compute local reaching centrality for a graph-tool graph."""
+    # Local reaching centrality (weighted), matching NetworkX's algorithm:
+    # 1. Invert weights to get distances: distance = total_weight / w
+    # 2. Find shortest paths using these distances (graph-tool C-level)
+    # 3. For each reachable node, compute average original edge weight along path
+    # 4. Sum averages, normalise by (total_weight / num_edges), divide by (n-1)
+    n = gt_graph.num_vertices()
+    weight_prop = gt_graph.ep["weight"]
+    total_weight = sum(weight_prop[e] for e in gt_graph.edges())
+    num_edges = gt_graph.num_edges()
+    if total_weight <= 0:
+        raise ValueError("Total edge weight must be positive for LRC")
+
+    # Build distance property: distance = total_weight / original_weight
+    dist_prop = gt_graph.new_edge_property("double")
+    for e in gt_graph.edges():
+        w = weight_prop[e]
+        dist_prop[e] = total_weight / w if w > 0 else float("inf")
+
+    # Build edge weight lookup: (source_idx, target_idx) → original weight
+    edge_weight_lookup: Dict[Tuple[int, int], float] = {}
+    for e in gt_graph.edges():
+        edge_weight_lookup[(int(e.source()), int(e.target()))] = weight_prop[e]
+
+    lrc_norm = total_weight / num_edges if num_edges > 0 else 1.0
+
+    local_reaching_centrality = {}
+    for v in gt_graph.vertices():
+        i = int(v)
+        # Compute shortest distances and predecessor map from vertex v.
+        dist_map, pred_map = shortest_distance(
+            gt_graph, source=v, weights=dist_prop, pred_map=True
+        )
+
+        sum_avg_weight = 0.0
+        for t in gt_graph.vertices():
+            t_idx = int(t)
+            if t_idx == i:
+                continue
+            # Unreachable vertices have max int distance.
+            if dist_map[t] >= 2147483647:
+                continue
+
+            # Reconstruct path by walking predecessors from target back to source.
+            path = []
+            curr = t_idx
+            while curr != i:
+                path.append(curr)
+                pred = int(pred_map[curr])
+                if pred == curr:
+                    # Self-loop in pred_map means unreachable.
+                    path = []
+                    break
+                curr = pred
+            if not path:
+                continue
+            path.append(i)
+            path.reverse()
+
+            path_length = len(path) - 1
+            path_weight_sum = sum(
+                edge_weight_lookup.get((path[k], path[k + 1]), 1.0)
+                for k in range(path_length)
+            )
+            sum_avg_weight += path_weight_sum / path_length
+
+        lrc = (sum_avg_weight / lrc_norm) / (n - 1) if n > 1 else 0.0
+        local_reaching_centrality[node_ids[i]] = lrc
+
+    return local_reaching_centrality
+
+
+class NodeMetrics:
+    """Handles node-level metric calculations."""
+
+    @staticmethod
+    @log_timer
+    def extract_node_metrics(
+        dpg_model: nx.DiGraph, nodes_list: List[Tuple]
+    ) -> Any:
+        """Compute per-node graph metrics for a DPG model.
+
+        Args:
+            dpg_model: NetworkX DiGraph representing the DPG.
+            nodes_list: List of ``(node_id, label)`` tuples.
+
+        Returns:
+            DataFrame with columns ``['Node', 'Label', 'Degree', 'In degree nodes',
+            'Out degree nodes', 'Betweenness centrality', 'Local reaching centrality']``.
+        """
+
+        # Convert to graph-tool for fast C-level centrality computation.
+        gt_graph, node_ids = _nx_to_graph_tool(dpg_model)
+        in_nodes, out_nodes, degree = calc_node_metrics(dpg_model)
+        betweenness_centrality = calc_betweenness_centrality(gt_graph)
+        local_reaching_centrality = calc_local_reaching_centrality(gt_graph, node_ids)
+
+        data_node = {
+            "Node": list(dpg_model.nodes()),
+            "Degree": list(degree.values()),
+            "In degree nodes": list(in_nodes.values()),
+            "Out degree nodes": list(out_nodes.values()),
+            "Betweenness centrality": list(betweenness_centrality.values()),
+            "Local reaching centrality": list(local_reaching_centrality.values()),
+        }
+        df_data_node = pd.DataFrame(data_node).set_index("Node")
+        df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index(
+            "Node"
+        )
+        return pd.concat(
+            [df_data_node, df_nodes_list], axis=1, join="inner"
+        ).reset_index()
```

`pyproject.toml` diff, relevant hunks only (the poetry to PEP 621 conversion is omitted):

```diff
+dependencies = [
+    ...
+    "igraph>=1.0.0",
+]
...
+[tool.pixi.workspace]
+channels = ["conda-forge"]
+platforms = ["linux-64"]
+
+[tool.pixi.pypi-dependencies]
+dpg = { path = ".", editable = true }
+
+[tool.pixi.environments]
+default = { solve-group = "default" }
+dev = { features = ["dev"], solve-group = "default" }
+docs = { features = ["docs"], solve-group = "default" }
+
+[tool.pixi.dependencies]
+graph-tool = ">=2.98,<3"
```

---

## 4. Defects in the DPG commit (do not carry these over)

1. **Duplicate `NodeMetrics` class.** The new implementation was appended to the bottom of `metrics/nodes.py` (after line 57) with a second block of imports, instead of replacing the original. Python keeps the last definition, so the graph-tool version silently shadows the NetworkX one. Migrate the functions, not the file.
2. **`trace_lrc_by_label` dropped from `extract_node_metrics`.** `dpg/explainer.py:755` in DPG (untouched by the commit) still calls `NodeMetrics.extract_node_metrics(self._graph, self._nodes, trace_lrc_by_label=...)`. Reproduced: `TypeError: ... got an unexpected keyword argument 'trace_lrc_by_label'`. `explain_global()` cannot run as committed. The committed log and pickles predate the commit and are stale.
3. **graph-tool imported unconditionally but only installable via pixi.** `dpg/utils.py` and `metrics/nodes.py` do `from graph_tool import ...` at module scope; graph-tool is declared only under `[tool.pixi.dependencies]` (conda-forge). `.venv/bin/python -c "import dpg.utils"` fails with `ModuleNotFoundError: No module named 'graph_tool'`. The commit's own `output.log` captures this failure. graph-tool has no PyPI wheels; it must come from conda-forge or a system package.
4. **`igraph>=1.0.0` added as a dependency but never imported** in DPG.
5. **`descendants()` in `dpg/utils.py` is dead code** (never called). It was probably intended for the `nx.descendants` calls in `metrics/graph.py`.
6. **`log_timer` wrapper signature is `wrapper(self, *args, **kwargs)`** but it decorates module-level functions with no `self`. It works only because every call site passes at least one positional argument. It also logs "Decorating function X with log_timer" at import time for every decorated callable. dpg_v's own `log_timer` in `metrics/nodes.py:52-75` is the better version; keep that one.
7. **Misleading unreachable-node sentinel.** `if dist_map[t] >= 2147483647:` is commented as "max int distance", but with a `double` weight map graph-tool returns `float('inf')` for unreachable vertices. The check works only because `inf >= 2147483647` is `True`. Use `math.isinf` in the port.

---

## 5. State of dpg_v that affects the migration

### 5.1 Pre-existing blockers (must be fixed first, independent of backend)

- **Test suite fails to collect.** `dpg/core.py:5` imports only `Any, cast` from `typing`, but the module uses `List`, `Dict`, `Tuple`, `Set` as bare names (e.g. `core.py:211-220, 357, 613`). Verified: `uv run pytest tests/test_metrics.py -q` dies with `NameError: name 'List' is not defined` at `dpg/core.py:357`. One-line fix: add `Dict, List, Set, Tuple` to the import. Until this is done there is no green baseline to compare a new backend against.
- **`trace_lrc_by_label` mismatch already exists in dpg_v.** `dpg/explainer.py:778` passes `trace_lrc_by_label=` but `metrics/nodes.py:249` `extract_node_metrics(dpg_model, nodes_list)` no longer accepts it. This is the same regression as DPG defect 2 and was inherited when dpg_v rewrote nodes.py for igraph. `tests/test_dpg_core.py::TestTraceLRCNodeMetricsIntegration` and `TestIrisLRCRankingComparison` exercise this path and will fail. Neither repo contains a fix to copy; the ancestor `f16a977` version of the LRC loop shows how the mapping was applied (per-label override of the computed LRC) and is the reference for restoring it.

### 5.2 What dpg_v already has (reuse, do not duplicate)

- `metrics/nodes.py:13-75`: `get_logger` and `log_timer` (properly typed, `Callable`-based). No need to touch `dpg/utils.py`.
- `metrics/nodes.py:79-103`: `_nx_to_igraph` returning `(ig.Graph, node_ids)`. `_nx_to_graph_tool` from DPG returns the same shape `(gt.Graph, node_ids)`, so the orchestration in `extract_node_metrics` maps one-to-one.
- `dpg/exceptions.py`: `DPGMetricError.non_positive_lrc_weight()`, `.non_positive_closeness_weight()`, `.non_positive_harmonic_weight()`. The graph-tool functions must raise these instead of `ValueError` (the one sanctioned `metrics/` → `dpg/` import, per `CLAUDE.md`).
- `pyproject.toml:102-113`: mypy `ignore_missing_imports` override list already contains `igraph.*`; add `graph_tool.*` there.

### 5.3 Contracts that must not change

- `NodeMetrics.extract_node_metrics` returns a DataFrame with exactly these 9 columns, in `dpg_model.nodes()` order, `RangeIndex`, inner-joined with `nodes_list`: `Node, Degree, In degree nodes, Out degree nodes, Betweenness centrality, Local reaching centrality, Closeness centrality, Harmonic centrality, Label`.
- Readers by column-name string literal: `dpg/explainer.py:813-817`, `dpg/visualizer.py:2013` (raises `DPGValidationError.node_metrics_columns_required()` without `Node`/`Label`), `dpg/cli.py:81` (CSV header), `dpg/sklearn_dpg.py:221` (positional 2-arg call), all `examples/` and `experiments/` scripts.
- Pinned test values (`tests/test_metrics.py::TestNodeMetrics`, Iris RF seed 160898, 5 trees): shape `(31, 9)`; Degree min 1, max 7, mean 3.2903; `Betweenness centrality`.max ≈ 0.072414 (abs 1e-4); `Local reaching centrality`.max ≈ 0.927617 (abs 1e-4). `TestMetricsOnWine`: 87 rows.
- `tests/test_dpg_core.py::TestCentralityNetworkXParity` imports `_nx_to_igraph`, `calc_closeness_centrality`, `calc_harmonic_centrality` by name and checks against NetworkX to abs 1e-9, including `test_closeness_nonzero_for_partially_reaching_node` (a node reaching only some others must get strictly positive closeness).

### 5.4 Environment

- dpg_v is uv-only: `pyproject.toml` (poetry-core build backend, PEP 621 deps), `uv.lock`, no `pixi.toml`/`pixi.lock`/`.pixi`. `graph_tool` is not importable (`uv run python -c "import graph_tool"` → `ModuleNotFoundError`).
- The only live CI workflow is `.github/workflows/feature-pr.yaml`: `setup-uv` (Python 3.11) → `uv sync --locked --dev` → ruff → mypy → pytest. No conda/pixi step. `ci.yml` is fully commented out.
- On this machine `pixi` (global) and `micromamba` exist, so a local graph-tool env is feasible; DPG's own pixi env has graph-tool 2.98 working.

---

## 6. Migration plan

### Phase 0: restore a green baseline (prerequisite, backend-agnostic)

1. `dpg/core.py:5`: `from typing import Any, Dict, List, Set, Tuple, cast`. Run `uv run pytest -q` and record the pass/fail list.
2. Restore `trace_lrc_by_label: dict[str, float] | None = None` on `NodeMetrics.extract_node_metrics` in `metrics/nodes.py`. Apply it after `calc_local_reaching_centrality` returns: for each `(node_id, label)` in `nodes_list`, if `label in trace_lrc_by_label`, override `local_reaching_centrality[node_id]`. Cross-check against the ancestor loop in `git show f16a977:metrics/nodes.py` and against `TestTraceLRCNodeMetricsIntegration`. Update the `CLAUDE.md` signature table.
3. Commit these two fixes separately from the backend work.

### Phase 1: introduce a backend seam in `metrics/nodes.py` (no behaviour change)

4. Keep every public name and the 9-column contract. Rename nothing that tests import (`_nx_to_igraph`, `calc_closeness_centrality`, `calc_harmonic_centrality` stay).
5. Add a module-level selector:
   ```python
   _BACKEND = os.environ.get("DPG_METRICS_BACKEND", "igraph")  # "igraph" | "graph_tool"
   ```
   and a `_graph_tool_available()` probe using `importlib.util.find_spec("graph_tool")`. If `graph_tool` is requested but missing, raise `DPGConfigurationError` (add a factory in `dpg/exceptions.py`) rather than silently falling back. Keep the `import graph_tool` inside the backend functions or a lazily-imported `_gt_backend` module so `import dpg` never requires graph-tool.
6. Route `extract_node_metrics` through the selector: convert once (`_nx_to_igraph` or `_nx_to_graph_tool`), then call the matching `calc_*` set. Degree stays on NetworkX in both paths.

### Phase 2: port the graph-tool functions (from DPG `563eecd`)

7. **`_nx_to_graph_tool`**: copy as-is. Also attach an inverted `"distance"` edge property (`total_weight / w`, `inf` if `w <= 0`) at conversion time, mirroring `_nx_to_igraph`, so closeness/harmonic can reuse it.
8. **`calc_betweenness_centrality_gt`**: copy as-is (`betweenness(g, weight=ep["weight"], norm=False)`, then `/ ((n-1)(n-2))`). Return keyed by int vertex index like the igraph version.
9. **`calc_local_reaching_centrality_gt`**: copy, with these edits: replace `dist_map[t] >= 2147483647` with `math.isinf(dist_map[t])`; raise `DPGMetricError.non_positive_lrc_weight()` instead of `ValueError`; add full type hints (`disallow_untyped_defs` is on). Consider `shortest_distance(g, source=v, weights=dist_prop, pred_map=True)` once per vertex is O(V) Dijkstra runs, same as igraph; acceptable.
10. **`calc_closeness_centrality_gt` / `calc_harmonic_centrality_gt`**: no template exists. Implement with one `shortest_distance(g, source=v, weights=dist_prop)` per vertex (or a single call with `source=None` returning the full matrix if memory allows), then replicate exactly the formulas in dpg_v `nodes.py:183-241` (closeness with Wasserman-Faust correction over the reachable set; harmonic as the sum of `1/d` over reachable targets). Unreachable = `isinf`. Must satisfy the abs 1e-9 parity tests.
11. Decorate all new functions with the existing `log_timer` from `metrics/nodes.py`, not DPG's.

### Phase 3: packaging and CI

12. `pyproject.toml`: do **not** add `graph-tool` to `[project.dependencies]` (uv cannot resolve it). Add a documented optional path instead. Two options, pick one:
    - **Option A (recommended, mirrors DPG):** add a `[tool.pixi.*]` workspace with `channels = ["conda-forge"]`, `graph-tool = ">=2.98,<3"`, and `dpg = { path = ".", editable = true }` under `pypi-dependencies`. `uv sync` stays the default dev flow; `pixi run pytest` is the graph-tool flow.
    - **Option B:** document a `micromamba create -n dpg-gt -c conda-forge graph-tool` step and `pip install -e .` inside it. No pixi lockfile to maintain but less reproducible.
13. Add `graph_tool.*` to the mypy `ignore_missing_imports` overrides (`pyproject.toml:102-113`).
14. `.github/workflows/feature-pr.yaml`: keep the uv job as the primary gate. Add a second, allow-failure job that installs via `prefix-dev/setup-pixi` (Option A) or `mamba-org/setup-micromamba` (Option B) and runs `DPG_METRICS_BACKEND=graph_tool pytest tests/test_metrics.py tests/test_dpg_core.py`. Skip graph-tool tests in the uv job with `pytest.importorskip("graph_tool")`.
15. Add `graph-tool` note to `README.md` / `docs` install section. Do not touch `requirements.txt` (stale, only used by the disabled `ci.yml`).

### Phase 4: tests

16. Parametrize `TestNodeMetrics` and `TestCentralityNetworkXParity` over `backend in ("igraph", "graph_tool")`, skipping `graph_tool` when unavailable. The pinned values (0.072414, 0.927617, shape (31, 9), 87 rows) must hold for both.
17. Add a direct igraph-vs-graph-tool equality test on a random weighted digraph (abs 1e-12) for all five metrics.
18. Add a test that `import dpg` and `import metrics` succeed with `graph_tool` absent (e.g. `monkeypatch.setitem(sys.modules, "graph_tool", None)`).

### Phase 5: cleanup and docs

19. Update `CLAUDE.md`: the "converts to igraph, that conversion is the hot path" line becomes "converts to igraph (default) or graph-tool (`DPG_METRICS_BACKEND=graph_tool`)". Add the env var to the CLI docs.
20. Do **not** port `descendants()` from `dpg/utils.py` unless `metrics/graph.py:135,222` (`nx.descendants`) is measured as a bottleneck. Do **not** port `main.py`, the dataset scripts, or any pickle/log artifacts.

### Explicitly out of scope

- `metrics/graph.py` (`nx.descendants`, `nx.community.asyn_lpa_communities`): not touched by the DPG commit and no centrality calls there.
- `dpg/core.py:669` (`nx.local_reaching_centrality(graph, node, weight=None)` inside `get_predicate_lrc`): a separate unweighted LRC path for `context_order > 1`; leave as is.

---

## 7. Risk summary

| Risk | Severity | Mitigation |
|---|---|---|
| graph-tool not pip-installable; breaks `uv sync` if added as a hard dep | High | Optional backend + lazy import + pixi/micromamba env (Phase 3) |
| Silent numeric drift between backends | Medium | Cross-backend equality test at 1e-12; keep pinned Iris/Wine values (Phase 4) |
| Closeness/harmonic on graph-tool are net-new code | Medium | Reuse existing parity tests at 1e-9; port formulas verbatim from igraph functions |
| Pre-existing `NameError` / `TypeError` mask regressions | High | Phase 0 first, committed separately |
| Node-ordering assumptions (`list(dict.values())` zipped positionally onto `dpg_model.nodes()`) | Low | Both backends key by index in `node_ids` order built from `list(nx_graph.nodes())`; keep that invariant, add an assertion on length |
| Sentinel `>= 2147483647` for unreachable | Low | Use `math.isinf` |
