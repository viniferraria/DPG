# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development setup

This project uses `uv` for dependency management (the `[build-system]` is still `poetry-core`).

```bash
uv sync                          # install runtime dependencies
uv sync --group dev              # include dev tools (mypy, ruff, pytest)
uv sync --extra docs             # include docs dependencies
```

Graph rendering needs the system Graphviz `dot` binary on `PATH` (`brew install graphviz`).

## Common commands

```bash
# Tests — run from the repo root (see note below)
uv run pytest
uv run pytest tests/test_explainer.py
uv run pytest tests/test_explainer.py::test_name

# Lint — CI lints tracked files explicitly, matching that avoids scanning .venv/experiments output
uv run ruff check $(git ls-files '*.py')

# Type checking — only these two packages are checked; mypy runs with disallow_untyped_defs = true
uv run mypy dpg/ metrics/

# Docs
uv run sphinx-build -b html docs/ docs/_build/html

# CLI entrypoints (sklearn standard datasets / custom CSV)
uv run python examples/run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --plot
uv run python examples/run_dpg_custom.py --ds datasets/custom.csv --target_column <col>
```

Tests must run from the repo root: `tests/conftest.py` only prepends the repo root to `sys.path` (no installed
package needed), and `tests/test_integration_dpg.py` resolves `datasets/custom.csv` through `os.getcwd()`.
`test_datasets/` exists but is currently empty.

CI: `feature-pr.yaml` runs ruff + mypy + pytest via `uv` on `feature/**` pushes and opens a PR; `ci.yml` runs
pytest on Python 3.10/3.11/3.12 but installs from `requirements.txt` with pip, so that file must stay in sync
with `pyproject.toml` dependencies. `docs.yml` treats unexpected Sphinx warnings as failures.

## Architecture

Two top-level packages: `dpg/` (core library) and `metrics/` (graph metrics). `metrics/` is imported by
`dpg/` — not the other way around.

### The pipeline is process-mining shaped

This is the key thing to understand before touching `core.py`. DPG does not walk trees into a graph directly;
it replays samples through the ensemble to produce an *event log*, then mines a directly-follows graph:

```
sklearn model + X
  → SklearnEnsembleNormalizer          normalize heterogeneous ensembles to a uniform .estimators_ / .tree_ shape
  → tracing_ensemble(_parallel)        replay each sample down every tree → [case_id, event] pairs
  → DataFrame["case:concept:name", "concept:name"]   the trace log; one case per (sample, tree)
  → filter_log / discover_dfg          count directly-follows pairs → {(src_label, dst_label): frequency}
  → generate_dot                       graphviz.Digraph; node id = str(int(sha1(label), 16))
  → to_networkx                        parses dot.body text back into nx.DiGraph + nodes_list
  → NodeMetrics / EdgeMetrics / GraphMetrics
  → DPGExplanation / DPGLocalExplanation
  → visualizer.py                      render to graphviz / matplotlib
```

Consequences worth knowing:

- **Node identity is the predicate text.** IDs are `sha1` of the label, so two predicates collapse into one
  node iff their label strings are byte-identical. `decimal_threshold` (rounding of tree thresholds) therefore
  directly controls how much the graph merges.
- **Event label formats are a contract** across `core.py`, `metrics/graph.py`, `explainer.py`, and
  `visualizer.py`: `"<feature> <= <threshold>"`, `"<feature> > <threshold>"`, `"Class <name>"` for
  classifiers, `"Pred <value>"` for regressors. Parsing helpers (`GraphMetrics._parse_predicate`,
  `_normalize_class_label`, `DPGExplainer._label_to_node_id`) depend on these exact shapes.
- `to_networkx` round-trips through the DOT *source text*, so any change to `generate_dot`'s label escaping or
  attribute ordering can silently break node/edge parsing.
- Local explanations use normalized class names (`"0"`) in `class_votes` / `majority_vote`, but keep raw DPG
  labels (`"Class 0"`) inside `tree_paths[*].labels`.

### `dpg/`

- **`core.py`** — `DecisionPredicateGraph`, `DPGError`, `DEFAULT_DPG_CONFIG`. All graph construction and
  config resolution.
- **`sklearn_normalizer.py`** — `SklearnEnsembleNormalizer`: flattens RandomForest, GradientBoosting (incl.
  the per-class-column layout and the binary sign-of-leaf-score case), AdaBoost, ExtraTrees, Bagging, and the
  regressor variants into one tree-path interface.
- **`explainer.py`** — `DPGExplainer` (high-level entry point) plus the `DPGExplanation`,
  `DPGLocalExplanation`, `DPGTreePathExplanation` dataclasses. Also owns local tracing, sample confidence,
  and `evaluate_faithfulness` (output fidelity, trace coverage, recombination, evidence margin, composite
  score). Largest module after `visualizer.py`.
- **`visualizer.py`** — every plot function; `DPGExplainer.plot*` methods are thin wrappers over it.
- **`sklearn_dpg.py`** — dataset selection + train/evaluate/report glue used by the `examples/run_dpg_*.py`
  CLIs.
- **`themes.py`**, **`utils.py`** — palettes/theme resolution and shared helpers.

### `metrics/`

`NodeMetrics`, `EdgeMetrics`, `GraphMetrics` are stateless — classmethod/staticmethod APIs taking
`(dpg_model: nx.DiGraph, nodes_list, target_names)`, where `nodes_list` is the `[node_id, label]` list from
`to_networkx`. `metrics/nodes.py` converts to `igraph` (`_nx_to_igraph`) for betweenness and reaching
centrality — that conversion is the hot path.

## Config

Resolution order in `DecisionPredicateGraph.__init__`: explicit `dpg_config` (dict or OmegaConf `DictConfig`)
→ `config.yaml` resolved **relative to the current working directory** → `DEFAULT_DPG_CONFIG` in `core.py`.
The two sources disagree, so the same code behaves differently depending on where it runs:

| | `config.yaml` | `DEFAULT_DPG_CONFIG` |
|---|---|---|
| `perc_var` | 0.0001 | 1e-9 |
| `decimal_threshold` | 3 | 6 |
| `n_jobs` | 1 | -1 |

Pass `dpg_config` explicitly in tests and experiments rather than relying on CWD.

### Graph construction modes

`dpg_config["dpg"]["graph_construction"]["mode"]`:

- `"aggregated_transitions"` (default) — `filter_log` drops whole path *variants* below
  `n_cases * perc_var`, then builds the DFG.
- `"execution_trace"` — builds the DFG from the raw log and drops individual *edges* below the same
  threshold.

Unsupported mode strings raise `DPGError` at construction time.

`explain_local` always re-traces the sample through the raw trees (reported as `path_mode="execution_trace"`)
and then checks each traced node/edge against the built graph. So an aggressive `perc_var` makes
`graph_path_valid` / `all_trees_valid` go False — that is filtering working as intended, not a tracing bug.

## Other directories

- `examples/` — runnable scripts and the two CLIs. Note `pyproject.toml` declares the console script as
  `scripts.run_dpg_standard:main`, but no `scripts/` package exists; the CLIs live in `examples/`.
- `experiments/` — research runners (`causal_synthetic_scenarios`, `local_explanation`) with committed
  logs/results; not covered by mypy, and only `test_run_dpg_causal_synthetic.py` touches them.
- `tutorials/` — notebooks, including the parameter-sensitivity benchmark for `perc_var` /
  `decimal_threshold`.
- `docs/` — Sphinx + autoapi; see `docs/README.md` for local build/serve.
