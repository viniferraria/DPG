# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development setup

This project uses `uv` for dependency management. To set up locally:

```bash
uv sync                          # install all dependencies
uv sync --group dev              # include dev tools (mypy, ruff, pytest)
uv sync --extra docs             # include docs dependencies
```

## Common commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_explainer.py

# Run a single test by name
uv run pytest tests/test_explainer.py::test_name

# Lint
uv run ruff check .

# Type checking
uv run mypy dpg/ metrics/

# CLI entrypoint (sklearn standard datasets)
uv run python examples/run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --plot

# CLI entrypoint (custom CSV dataset)
uv run python examples/run_dpg_custom.py --ds datasets/custom.csv --target_column <col>
```

## Architecture

The library has two top-level packages: `dpg/` (core library) and `metrics/` (graph metrics).

### `dpg/` — core library

- **`core.py`** — `DecisionPredicateGraph`: low-level graph construction. Extracts decision paths from ensemble trees via `SklearnEnsembleNormalizer`, builds the networkx/graphviz graph, and computes edges/nodes. All graph-construction logic lives here.
- **`sklearn_normalizer.py`** — `SklearnEnsembleNormalizer`: adapts heterogeneous sklearn ensembles (RandomForest, GradientBoosting, AdaBoost, ExtraTrees, Bagging, regressors) into a uniform tree-path interface consumed by `DecisionPredicateGraph`.
- **`explainer.py`** — `DPGExplainer`: high-level API. Wraps `DecisionPredicateGraph`, `NodeMetrics`, `EdgeMetrics`, and `GraphMetrics`. Entry point for users. Returns `DPGExplanation` (global) and `DPGLocalExplanation` (local).
- **`visualizer.py`** — all plot functions (`plot_dpg`, `plot_dpg_communities`, `plot_local_on_dpg`, etc.). Called by `DPGExplainer.plot*` methods.
- **`utils.py`** — shared helpers.
- **`themes.py`** — color palettes and visual themes for plots.
- **`sklearn_dpg.py`** — sklearn-compatible estimator wrapper.

### `metrics/` — graph metrics

- **`nodes.py`** — `NodeMetrics`: betweenness centrality, local reaching centrality (LRC), class boundaries (constraints).
- **`edges.py`** — `EdgeMetrics`: edge-level statistics.
- **`graph.py`** — `GraphMetrics`: community detection and graph-level summaries.

### Key data flow

```
sklearn model + X
    → SklearnEnsembleNormalizer  (normalize tree structures)
    → DecisionPredicateGraph     (build networkx graph)
    → NodeMetrics / EdgeMetrics / GraphMetrics  (compute metrics)
    → DPGExplanation             (structured output)
    → visualizer.py              (render to graphviz/matplotlib)
```

### Graph construction modes

Controlled via `dpg_config["dpg"]["graph_construction"]["mode"]`:
- `"aggregated_transitions"` (default): filters path variants first, then builds the graph.
- `"execution_trace"`: builds from raw traces, filters edges (not paths) when `perc_var > 0`.

### Config

`config.yaml` at the repo root holds default DPG parameters (`perc_var`, `decimal_threshold`, `n_jobs`). `DecisionPredicateGraph` also accepts `dpg_config` as a dict or OmegaConf `DictConfig` directly.

### Tests

Tests live in `tests/`. Integration tests (`test_integration_dpg.py`) exercise the full pipeline on small synthetic datasets in `test_datasets/`. Fixtures are minimal — `conftest.py` only adds the repo root to `sys.path`.
