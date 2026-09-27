---
type: Python Module
title: metrics.backends — pluggable centrality backends
description: Registry and three interchangeable graph-library implementations (networkx, igraph, graph_tool) for the four DPG node centralities, selected by name via dpg.metrics.backend.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/backends/__init__.py
tags: [metrics, centrality, igraph, networkx, graph-tool, backends, performance]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# Responsibility

`metrics/backends/` makes the four expensive node-centrality computations pluggable by graph
library. `metrics/nodes.py` used to hard-code igraph; the centrality code that used to live there
now lives in this package behind a common interface, selected at metric time by name
(`"networkx"`, `"igraph"`, `"graph_tool"`) or by passing an already-instantiated backend object.
Degree, in-degree, and out-degree stay computed directly on the NetworkX graph in
`metrics/nodes.py` — they are not backend work. See
[/modules/metrics-nodes.md](/modules/metrics-nodes.md) for how the backend plugs into
`NodeMetrics.extract_node_metrics`.

Like the rest of `metrics/`, this package never imports `dpg/` at module level — only lazily,
inside function bodies, when a `DPGMetricError` needs raising.

# API

## Registry (`metrics/backends/__init__.py`)

| Name | Signature | Behavior |
|---|---|---|
| `BACKEND_NAMES` | `tuple[str, ...]` | `("networkx", "igraph", "graph_tool")` |
| `get_backend` | `(name: str) -> GraphBackend` | Looks `name` up in an internal `{name: (module_path, class_name)}` table, `import_module`s that module **on demand**, and instantiates the class. `graph_tool`'s library is therefore never imported unless `"graph_tool"` is actually requested. |

Errors:

- Unknown `name` → `ValueError` listing `BACKEND_NAMES`.
- Known `name` whose library is not installed → `ImportError`, message names the backend and how
  to install it. For `graph_tool` the hint points at conda-forge / `pixi add graph-tool`, not pip
  (graph_tool is not on PyPI); the other two hints are plain `pip install`.

## Interface (`metrics/backends/base.py`)

```python
@dataclass(frozen=True)
class NodeCentralities:
    """All four dicts are keyed by the NetworkX node id."""
    betweenness: dict[Any, float]
    local_reaching: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]

class GraphBackend(ABC):
    name: ClassVar[str]
    @abstractmethod
    def node_centralities(self, dpg_model: nx.DiGraph) -> NodeCentralities: ...
```

Every backend module (`networkx_backend.py`, `igraph_backend.py`, `graph_tool_backend.py`) defines
one class implementing `GraphBackend`, with `name` set to its registry key
(`NetworkXBackend.name = "networkx"`, `IGraphBackend.name = "igraph"`,
`GraphToolBackend.name = "graph_tool"`). `node_centralities` is the only public entry point; all
four centralities are computed together in one call.

# The metric contract

Notation: edge weight `w` = edge attribute `"weight"` (default `1.0`); `W` = sum of all edge
weights; `m` = number of edges; `n` = number of nodes. All graphs are directed. Every backend must
reproduce these four definitions to an absolute tolerance of `1e-3`; `networkx_backend.py`'s
docstring calls its own calls "the reference definitions."

1. **Betweenness** — shortest paths using the raw weight `w` as distance (not inverted), directed,
   endpoints excluded, normalized by `1 / ((n-1)(n-2))` (`1.0` when `n <= 2`). Reference call:
   `nx.betweenness_centrality(G, normalized=True, weight="weight")`.
2. **Local reaching centrality** — distance `d = W / w` per edge. For source `v`, take one
   shortest path (by `d`) to every reachable `t != v`; for each path, average the *original*
   weights `w` of its edges; sum those averages; divide by `(n - 1)`; then divide by `W / m`.
   Reference call: `nx.local_reaching_centrality(G, v, weight="weight", normalized=True)` per
   node. Raises `DPGMetricError.non_positive_lrc_weight()` (lazy `from dpg.exceptions import
   DPGMetricError`) when `W <= 0`.
3. **Closeness** (Wasserman–Faust, outgoing) — distance `d = W / w`; `R` = nodes reachable from
   `v` excluding `v`, `r = |R|`, `S` = sum of `d(v, t)` for `t in R`. Value = `(r / S) * (r /
   (n - 1))` if `r > 0 and S > 0`, else `0.0`. Reference: build a graph with `dist = W / w`, then
   `nx.closeness_centrality(H.reverse(copy=True), distance="dist", wf_improved=True)` — the
   reversal turns NetworkX's incoming-distance measure into the outgoing one DPG uses.
4. **Harmonic** (outgoing, unnormalized) — distance `d = W / w`; value = `sum(1 / d(v, t))` over
   reachable `t != v` with `d > 0`. Reference: `nx.harmonic_centrality(H.reverse(copy=True),
   distance="dist")` on the same reversed distance graph.

## Which library call each backend uses

| Metric | `networkx` | `igraph` | `graph_tool` |
|---|---|---|---|
| Betweenness | `nx.betweenness_centrality(dpg_model, normalized=True, weight="weight")` | `ig_graph.betweenness(directed=True, normalized=False, weights="weight")`, divided by `(n-1)(n-2)` | `graph_tool.centrality.betweenness(graph, weight=weight_prop, norm=n>2)` on the **raw** weight (not the distance) property |
| Local reaching | `nx.local_reaching_centrality(dpg_model, node, weight="weight")` per node | `ig_graph.get_shortest_paths(i, mode="out", weights="distance", output="vpath")`, averaged by hand | `graph_tool.topology.shortest_distance(graph, source=v, weights=dist_prop, pred_map=True)`, path reconstructed by walking `pred_map` back to the source and averaging original weights by hand |
| Closeness | `nx.closeness_centrality(reversed_dist_graph, distance="dist", wf_improved=True)` | `ig_graph.distances(i, mode="out", weights="distance")`, Wasserman–Faust formula applied by hand | `graph_tool.centrality.closeness(graph, weight=dist_prop, harmonic=False, norm=True)`, then a further `* r / (n-1)` correction (see below) |
| Harmonic | `nx.harmonic_centrality(reversed_dist_graph, distance="dist")` | `ig_graph.distances(i, mode="out", weights="distance")`, `sum(1/d)` applied by hand | `graph_tool.centrality.closeness(graph, weight=dist_prop, harmonic=True, norm=False)` |

# Per-backend notes

## `networkx_backend.py`

Calls NetworkX's own centrality functions directly — this is the module the other two backends
are checked against (`tests/test_metrics_backends.py` and the ground-truth fixtures both trace to
these calls, never to a backend). `_distance_graph` builds a fresh `nx.DiGraph` carrying
`dist = total_weight / w` per edge (`inf` when `w <= 0`), and reverses it once for the
closeness/harmonic calls. Betweenness runs on the original graph with the raw `weight` attribute,
unreversed. Raises `DPGMetricError.non_positive_lrc_weight()` when total weight is `<= 0`, checked
right before the local-reaching loop.

## `igraph_backend.py`

This is the code that used to live directly in `metrics/nodes.py`: `_nx_to_igraph` and the four
`calc_*_centrality` helpers (`calc_betweenness_centrality`, `calc_local_reaching_centrality`,
`calc_closeness_centrality`, `calc_harmonic_centrality`) moved here unchanged, still each wrapped
in `@log_timer` (imported from `..nodes`). `_nx_to_igraph` performs the weight-to-distance
inversion once, up front: it copies `weight` onto `es["weight"]` (default `1.0`) and derives
`es["distance"] = total_weight / w` (`inf` when `w <= 0`); every shortest-path call after that
passes `weights="distance"`, while weight *averages* inside local reaching centrality are taken
from the original `weight` attribute. `calc_betweenness_centrality` is the one calculator keyed by
igraph vertex index rather than node id; `IGraphBackend.node_centralities` maps it back through
`node_ids` before returning the `NodeCentralities` dataclass. `calc_local_reaching_centrality`,
`calc_closeness_centrality`, and `calc_harmonic_centrality` each raise their own
`DPGMetricError` variant (`non_positive_lrc_weight`, `non_positive_closeness_weight`,
`non_positive_harmonic_weight`) when total edge weight is `<= 0`; igraph's `"Couldn't reach some"`
`RuntimeWarning` is suppressed per call.

## `graph_tool_backend.py`

Builds a `graph_tool.Graph` with parallel `weight` and `dist = total_weight / w` edge properties in
one `add_edge_list` call. Betweenness and harmonic map onto `graph_tool.centrality.betweenness` and
`graph_tool.centrality.closeness(..., harmonic=True, norm=False)` directly. Closeness needs a
correction: graph_tool's own normalized closeness is `r / S` (`r` = out-component size excluding
the source), but the Wasserman–Faust value the contract requires is `(r / S) * (r / (n - 1))`, so
the backend multiplies graph_tool's result by `r / (n - 1)` by hand — this is the
"Wasserman-Faust `r/(n-1)` correction" mentioned in the module docstring's cross-reference. Local
reaching centrality has no direct graph_tool call: for each source `i`,
`graph_tool.topology.shortest_distance(graph, source=i, weights=dist_prop, pred_map=True)` returns
both distances and a predecessor map; the backend walks `pred_map` backward from each reachable
target to `i` to reconstruct one shortest path, averaging the original `weight` values along that
walk, then applies the same `(sum / (W/m)) / (n-1)` normalization as the other backends. Raises
`DPGMetricError.non_positive_lrc_weight()` up front when total weight is `<= 0` (this backend does
not raise the closeness/harmonic-specific variants — all three failure paths share one exception).

# Gotchas

## Shortest-path ties in local reaching centrality

When two shortest paths from a source tie on distance `d` but differ in the mean of their original
weights, backends may legitimately disagree on which tied path they picked, and therefore on the
resulting local reaching centrality value. This is a real ambiguity in the metric definition, not a
backend bug — if a ground-truth comparison fails only because of such a tie, the difference should
be reported rather than papered over with a looser tolerance.

## Running graph_tool tests

`graph_tool` is conda-forge only (not on PyPI), so it cannot live in the `uv`-managed environment.
`tests/test_metrics_backends.py` calls `pytest.importorskip("graph_tool")` before exercising that
backend, so `uv run pytest` silently skips the graph_tool cases. To actually run them, use the
separate pixi environment described in `pixi.toml` (`pixi run pytest` from the repo root) — see
[/conventions/config-resolution.md](/conventions/config-resolution.md) and the Development setup
section of `CLAUDE.md` for how the two environments relate.

# Ground-truth fixture mechanism

`tests/metrics_ground_truth/` pins every backend to networkx-derived values that are independent of
any backend implementation:

- `scenarios.py` defines the one shared recipe both the generator and the tests use: four CSVs from
  `experiments/causal_synthetic_scenarios/datasets/` (`scenario_1`, `scenario_2_updated`,
  `scenario_3`, `scenario_5`, last column `Y` as target), each fit with
  `RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)`, built into a DPG with an
  explicit `dpg_config` (`perc_var=0.0001`, `decimal_threshold=3`, `n_jobs=1`,
  `mode="aggregated_transitions"`, `context_order=1` — never relying on `config.yaml`/CWD), then
  `dpg.fit(X.values)` and `dpg.to_networkx()`. `build_scenario_graph(name)` is the single
  function both consumers call, so the generator and the tests build byte-identical graphs.
- `generate.py` computes betweenness/local-reaching/closeness/harmonic straight from the same
  `networkx` reference calls as `networkx_backend.py` (not via `NodeMetrics` or any backend
  object), and writes `tests/metrics_ground_truth/<scenario>.json`, keyed by node **label** (unique
  per graph) rather than node id. Regenerate with:

  ```bash
  uv run python -m tests.metrics_ground_truth.generate
  ```
- `tests/test_metrics_backends.py` parametrizes over `BACKEND_NAMES` and the four scenarios (a
  session-scoped fixture builds each scenario graph once), runs
  `NodeMetrics.extract_node_metrics(graph, nodes_list, backend=name)`, and asserts every float
  metric matches the fixture within absolute tolerance `1e-3` (degree columns must match exactly).
  It also covers `get_backend("bogus")` raising `ValueError`, an unknown
  `dpg_config["dpg"]["metrics"]["backend"]` raising `DPGError` at `DecisionPredicateGraph`
  construction, and the default backend being `"igraph"`.

# See also

* [/modules/metrics-nodes.md](/modules/metrics-nodes.md) — where `extract_node_metrics` plugs a
  backend in, and the degree/output-DataFrame logic that stays outside this package.
* [/conventions/config-resolution.md](/conventions/config-resolution.md) — `dpg.metrics.backend`
  config resolution and validation.
