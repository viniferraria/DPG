# Modules

Source-level concepts for the two packages. `metrics/` is imported by `dpg/`, never the reverse.
`metrics/` now has four modules: `nodes`, `edges`, `graph`, and `backends` (the pluggable
centrality implementations `nodes` delegates to).

## dpg — core library

* [dpg.core — DecisionPredicateGraph](dpg-core.md) - Core module that replays samples through a tree ensemble, mines a directly-follows graph from the resulting event log, and emits a Graphviz/NetworkX Decision Predicate Graph.
* [dpg.context_order](dpg-context-order.md) - Trie-based resolution of the smallest context order (DPG-k) with no pooled-graph path recombination, without enumerating simple paths. New in 0.3.0.
* [dpg.cli](dpg-cli.md) - `build_parser`/`main` for the packaged `dpg` command, which the installed console script runs (fixed 2026-09-30). New in 0.3.0.
* [dpg.sklearn_normalizer](dpg-sklearn-normalizer.md) - Normalizes heterogeneous scikit-learn ensembles to a uniform 1D `.estimators_` list of objects exposing `.tree_`, preserving the GradientBoosting class-slot index.
* [dpg.explainer](dpg-explainer.md) - High-level DPGExplainer API that builds a DPG, produces global and local explanations, evaluates faithfulness, and wraps the visualizer plot functions.
* [dpg.visualizer](dpg-visualizer.md) - Every rendering function in DPG — Graphviz graph plots and Matplotlib analytics plots — plus the shared theme, save/show and label-formatting conventions.
* [dpg.sklearn_dpg](dpg-sklearn-dpg.md) - Convenience layer that loads a standard sklearn dataset or a CSV, trains a tree ensemble, writes an evaluation report, and returns node/edge/graph metrics for the resulting DPG.
* [dpg.themes and dpg.utils](dpg-themes-utils.md) - Theme and palette resolution for every DPG plot, plus the small set of shared Graphviz and filesystem helpers.

## metrics — graph metrics

* [metrics.nodes — NodeMetrics](metrics-nodes.md) - Computes per-node degree and centrality metrics for a DPG by delegating to a pluggable centrality backend and returning a labelled pandas DataFrame.
* [metrics.backends — pluggable centrality backends](metrics-backends.md) - Registry and three interchangeable graph-library implementations (networkx, igraph, graph_tool) for the four DPG node centralities.
* [metrics.edges — EdgeMetrics](metrics-edges.md) - Flattens a DPG's directed edges into a pandas DataFrame carrying each edge's weight plus the resolved source and target predicate labels.
* [metrics.graph — GraphMetrics](metrics-graph.md) - Graph-level DPG analysis — LPA communities, absorbing-Markov-chain clustering toward class nodes, and per-class feature boundary extraction from predicate labels.

## See also

* [The DPG pipeline](/pipeline.md) - How these modules compose, stage by stage.
* [Conventions](/conventions/index.md) - The contracts every module above depends on.
