# Bundle Update Log

## 2026-10-01

* **Fix**: `GraphMetrics.clustering` (`metrics/graph.py`) solves `(I - Q) B = R` with
  `scipy.sparse.linalg.spsolve` on a sparse `P` instead of building a dense `n × n` `P` and the full
  inverse `N = np.linalg.solve(I - Q, I)`. On scenario_3 ExtraTrees split 0 (6,793 nodes): ~22–25 s
  and ~2.2 GB peak → ~0.2 s and ~315 MB (≈ the loaded graph), identical clusters / probabilities /
  confidence; also identical on the four ground-truth scenario graphs. The dense version is what got
  `run_dpg_causal_synthetic.py` killed by `earlyoom` during `explain_global` (the loky "leaked
  semlock" warnings were cleanup after that kill). A singular `I - Q` still raises
  `np.linalg.LinAlgError`. New tests `TestClusteringAbsorption` in `tests/test_metrics.py`.
  `explain_global` still calls `clustering` twice per split; left as is. Updated
  [metrics.graph](/modules/metrics-graph.md).

## 2026-09-30

* **Fix**: `main()` in `experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py` now creates
  `results/` (`OUTPUT_PATH.mkdir(exist_ok=True)`) alongside `states/`. `write_causal_accuracy_to_csv`
  never created it, so with `results/` absent every split raised `FileNotFoundError`, was logged as
  `Failed:` and lost its CSV rows (only the `states/` pickles were saved). New test
  `test_main_creates_results_and_states_dirs`. Corrected the "created on demand" claim in
  [causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md).

* **Fix**: `uv run dpg` works. `[project.scripts]` (`pyproject.toml:12`) now points at
  `dpg.cli:main` instead of the nonexistent `scripts.run_dpg_standard:main`; `uv sync` regenerates
  the launcher. `dpg/cli.py` gained `--n_jobs` (default `-1`, forwarded to `test_dpg`) in place of the
  hard-coded `n_jobs=-1`; the CLI still does not read `config.yaml`. New test
  `test_cli_forwards_n_jobs_to_test_dpg`. Marked
  [console-script-entrypoint](/side-effects/console-script-entrypoint.md) `status: fixed` and moved it
  under "Fixed regressions" in the side-effects index; updated [dpg.cli](/modules/dpg-cli.md),
  [Running the DPG CLI entrypoints](/workflows/cli-entrypoints.md), the modules/workflows indexes, and
  CLAUDE.md (module table, Known breakage).

* **Fix**: `GraphMetrics.calculate_boundaries` (`metrics/graph.py`) no longer uses
  `joblib.Parallel(n_jobs=-1)`; it runs sequentially. The hard-coded pool ignored `n_jobs` from
  `config.yaml` and started a loky process pool (with its semaphores) even at `n_jobs: 1`, reached via
  `dpg/sklearn_dpg.py::test_dpg` → `extract_graph_metrics_lpa`. Measured on scenario_3 (2 classes):
  ~517 ms with the pool vs ~1 ms sequential, identical output. `calculate_class_boundaries`'s unused
  `class_names` annotation widened to `Sequence[str]` (the `delayed` wrapper had hidden the mismatch
  from mypy). `metrics/graph.py` no longer imports `joblib`. Updated
  [metrics.graph](/modules/metrics-graph.md).

* **Update**: `extract_top_k_features` in
  `experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py` gained
  `drop_duplicates: bool = False`. The default keeps repeated features (`F2;F2;F2`) on purpose so new
  runs stay comparable with the accumulated `results/` history; the runner does not pass it. Noted in
  [causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md) (Gotchas, Test coverage).

## 2026-09-29

* **Update**: Both experiment runners now rank the new node metrics. `METRICS` in
  `experiments/monks/run_monk.py` and `experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py`
  gained `Betweenness centrality` (percolation's control), `Collective influence`, and
  `Percolation centrality`; `Local clustering coefficient` is saved, not ranked. The causal
  `NodeMetricRecord` gained `closeness_centrality`, `harmonic_centrality`, `collective_influence`,
  `local_clustering_coefficient`, `percolation_centrality`. New `tests/test_run_monk.py`; new column
  guard test in `tests/test_run_dpg_causal_synthetic.py`. Updated
  [causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md) and
  [MONK's problems](/experiments/monks.md) (metrics, outputs, test coverage, ℓ = 2 gotcha; MONK's
  stale "untracked / no tests" claims fixed). `CLAUDE.md` orchestrator pattern now names Opus 5.5 /
  Sonnet 5.5.

* **Update**: Added three node metrics to every metrics backend (`metrics/backends/*.py`) and to
  `NodeMetrics.extract_node_metrics` (`metrics/nodes.py`): **collective influence** (directed out,
  unweighted, radius `ci_radius`, default 2, new keyword on `extract_node_metrics` and
  `GraphBackend.node_centralities`; `ValueError` when `< 1`), **local clustering coefficient**
  (directed, unweighted Fagiolo = `nx.clustering`), and **percolation centrality** (raw weight as
  distance, states from the new `metrics.backends.base.percolation_states` node-flow helper).
  `NodeCentralities` gained `collective_influence`, `clustering`, `percolation`; the DataFrame gained
  `Collective influence`, `Local clustering coefficient`, `Percolation centrality`. Ground-truth
  fixtures regenerated (pre-existing values unchanged beyond 1e-16 float noise). New tests:
  `tests/test_metric_collective_influence.py`, `tests/test_metric_clustering.py`,
  `tests/test_metric_percolation.py`. Updated [metrics backends](/modules/metrics-backends.md)
  (interface, contract items 5–7, library-call table, two new gotchas, fixture notes) and
  [metrics.nodes](/modules/metrics-nodes.md) (signature, columns, normalization note).

## 2026-09-22

* **Update (breaking change)**: Documented a further uncommitted change on `feature/first_runs`
  (`dpg/core.py`, `dpg/exceptions.py`, `dpg/explainer.py`, `dpg/sklearn_dpg.py`):
  `DecisionPredicateGraph.fit(X)` now returns `self` instead of a `graphviz.Digraph` — it builds the
  graph and no longer renders DOT. New `to_dot()` renders the fitted graph via `generate_dot` afresh on
  every call (deliberately uncached, since the visualizer recolors a dot it is given in place); it
  raises `DPGNotFittedError` (new factory `DPGNotFittedError.for_builder()`) before `fit`.
  `to_networkx(graphviz_graph=None)` now takes an optional argument: returns the fitted graph when set;
  if not fitted and a dot is given, falls back to parsing it; if neither, raises
  `DPGNotFittedError`. **Breaking** for external callers that used `fit`'s return value as a `Digraph`
  (`.source`, `.render`, `.body`, or passing it to `plot_dpg`) — `to_networkx(dpg.fit(X))` still works
  because the argument is ignored once fitted, but the old `dot = dpg.fit(X_train)` idiom no longer
  yields a dot. `DPGExplainer.fit` and `dpg/sklearn_dpg.py::test_dpg` were updated to call `fit()`, then
  `to_dot()`, then `to_networkx()`; `DPGExplainer.fit` already returned `self`, so the README-level API
  is unaffected. Updated [the DPG pipeline](/pipeline.md) (diagram, stage table, "DOT is a rendering
  output" consequence), [dpg.core](/modules/dpg-core.md) (`fit`/`to_networkx` rows rewritten, new
  `to_dot` row, Behavior step 6, example), [dpg.explainer](/modules/dpg-explainer.md) and
  [dpg.sklearn_dpg](/modules/dpg-sklearn-dpg.md) (call-pattern descriptions),
  [metrics.edge](/modules/metrics-edges.md) and [metrics backends](/modules/metrics-backends.md)
  (stale `to_networkx(dot)` call-shape snippets), [explanation dataclasses](/concepts/explanation-dataclasses.md)
  (`dot` field now sourced from `builder.to_dot()`), and CLAUDE.md's pipeline diagram and consequences.

* **Update**: Documented an uncommitted code change on `feature/first_runs` (`dpg/core.py`,
  `dpg/explainer.py`) that moves graph construction off the DOT round trip. New public method
  `DecisionPredicateGraph.build_graph(dfg) -> tuple[nx.DiGraph, list[list[str]]]` builds the NetworkX
  DPG directly from the DFG; `fit()` now calls it and stores the result before rendering. `generate_dot`
  changed signature from `generate_dot(dfg)` to `generate_dot(graph)` — it renders the already-built
  graph, reading emission order from a new `graph.graph["edge_order"]` attribute the graph carries
  (verified byte-identical DOT output on the four causal synthetic scenarios plus Iris). `to_networkx`
  keeps its signature but now returns the graph `fit()` already built; its DOT text-parsing regex path
  survives only as a fallback for a dot this instance did not build. Node ids changed from
  `str(int(sha1(key).hexdigest(), 16))` (~48 decimal digits) to `"n" + sha1(key).hexdigest()[:12]` (13
  chars) — content-addressed the same way, verified as a pure relabelling of the old graphs on all five
  scenarios; the `n` prefix keeps the token a valid DOT id, where a bare leading digit is not.
  `dpg/explainer.py::_label_to_node_id` now delegates to `_node_id_for_key` instead of holding a second
  copy of the formula (currently has no call sites in the repo). Updated [the DPG pipeline](/pipeline.md)
  (diagram, stage table, and the "DOT round-trip is load-bearing" consequence, now "DOT is a rendering
  output, not a construction step"), [the label contract](/conventions/label-contract.md) (id formula,
  and the round-trip gotcha narrowed to what the contract still governs: parsing and display, not
  construction), [dpg.core](/modules/dpg-core.md) (API table gains `build_graph`, `generate_dot` and
  `to_networkx` rows rewritten, `edge_order` documented, node id derivation corrected), and CLAUDE.md's
  pipeline diagram and "Node identity is the predicate text" bullet. Also corrected two other stale
  mentions of the old integer id / hashing formula found via a bundle-wide grep:
  [experiments/causal-synthetic-scenarios.md](/experiments/causal-synthetic-scenarios.md) ("sha1-derived
  integers" → id strings) and [dpg.explainer](/modules/dpg-explainer.md) (`_label_to_node_id`'s cited
  formula).

## 2026-09-09

* **Creation**: [Regressor detection](/conventions/regressor-detection.md) — `REGRESSOR_MODELS` as
  the single source of truth for the classifier/regressor tracing branch, shared by `dpg/core.py` and
  `dpg/explainer.py`, and the side effect of closing the prior `GradientBoostingRegressor` mismatch
  between the two modules.
* **Creation**: [Causal forest support](/concepts/causal-forest-support.md) — how
  `econml.grf.CausalForest` is traversed as a regressor, why only `n_relevant_outputs_ == 1` is
  supported, and worked (observed) examples of the supported and rejected cases.
* **Creation**: [Faithfulness on regressors](/concepts/faithfulness-regressors.md) — a known
  limitation, verified by running `explain_local` / `evaluate_faithfulness` against a fitted
  `RandomForestRegressor`: the `"Class "`-gated voting/evidence logic in `dpg/explainer.py` never
  fires for `"Pred "` leaves, so `majority_vote` is always `None` and `output_fidelity` is always
  `0.0` for a pure regressor. Not fixed by the `REGRESSOR_MODELS` change.
* **Update**: [dpg.core](/modules/dpg-core.md) — documented `REGRESSOR_MODELS` (module-level table,
  now including `econml.grf.CausalForest` when `econml` is importable), the `n_relevant_outputs_`
  guard in `__init__`, and that a `CausalForest` leaf's `Pred` value is a CATE estimate.
* **Update**: [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md) — added a `CausalForest`
  row to the estimator support matrix, and rewrote the pitfalls bullet that previously described
  `GradientBoostingRegressor` as missing from the explainer's regressor tuple (fixed by the shared
  `REGRESSOR_MODELS` constant).
* **Update**: [Event label contract](/conventions/label-contract.md) — noted that a `CausalForest`
  `Pred <value>` leaf is the leaf's CATE estimate, rounded the same way as any other regressor leaf.
* **Update**: `CLAUDE.md` — removed the known-breakage bullet about `dpg/explainer.py` omitting
  `GradientBoostingRegressor` (resolved by `REGRESSOR_MODELS`), and added a note that
  `econml.grf.CausalForest` (single treatment, single outcome) is traversed as a regressor, requiring
  the `causal` extra.

## 2026-08-02

* **Creation**: Initial OKF v0.2 bundle generated by reading the source of `dpg/` (0.1.6), `metrics/`,
  `examples/`, `experiments/`, `tests/`, and `.github/workflows/`.
* **Creation**: Architecture concepts — [overview](/overview.md) and [the DPG pipeline](/pipeline.md).
* **Creation**: Cross-module contracts — [label contract](/conventions/label-contract.md),
  [config resolution](/conventions/config-resolution.md),
  [graph construction modes](/conventions/graph-construction-modes.md).
* **Creation**: Module concepts for `dpg/` and `metrics/` under [modules](/modules/index.md).
* **Creation**: [Explanation dataclasses](/concepts/explanation-dataclasses.md) and
  [faithfulness evaluation](/concepts/faithfulness-evaluation.md).
* **Creation**: [Experiment suites](/experiments/index.md) — causal synthetic scenarios, local
  explanation, MONK's.
* **Creation**: [Workflows](/workflows/index.md) — setup, testing/linting, CI, CLI entrypoints.
* **Update**: Corrected the `n_jobs` row of the config-disagreement table. `config.yaml` sets `-1`, the
  same as `DEFAULT_DPG_CONFIG`; only `perc_var`, `decimal_threshold`, and the `visualization` block
  actually differ between the two sources.

## 2026-09-18

* **Update**: Removed `python_version = "3.10"` from `[tool.mypy]` — numpy 2.5.1 stubs (locked for
  Python ≥ 3.12) use the `type` statement, so mypy failed on `.venv` under a 3.14 interpreter. mypy now
  targets the running interpreter. Updated [development setup](/workflows/development-setup.md) and
  [testing and linting](/workflows/testing-and-linting.md).
* **Creation**: [side-effects/](/side-effects/index.md) — eleven pages documenting side effects,
  regressions, and limitations introduced by, or still present after, DPG 0.3.0 (release merge
  `f16a977`), each verified against branch `feature/first_runs` HEAD `276a503` with a runnable example
  that was actually executed:
  [explain-local-node-lookup-crash](/side-effects/explain-local-node-lookup-crash.md) (`explain_local`
  raises `TypeError` for every model — a ruff cleanup, commit `6df6e20`, commented out `node_lookup`
  but left `_trace_tree_path` requiring it; also added to CLAUDE.md's Known breakage),
  [context-order-pairwise-crash](/side-effects/context-order-pairwise-crash.md) (`discover_dfg_context`
  raises `TypeError` for every `context_order > 1` fit — the same `6df6e20` commit replaced
  `zip(nodes, nodes[1:])` with a two-argument `itertools.pairwise` call; confirmed via
  `uv run pytest tests/test_dpg_k.py` → 2 failed),
  [console-script-entrypoint](/side-effects/console-script-entrypoint.md) (the installed `dpg` console
  script points at a nonexistent `scripts.run_dpg_standard:main`; `[project.scripts]` wins over
  `[tool.poetry.scripts] dpg = "dpg.cli:main"`; `uv run python -m dpg.cli` works),
  [exact-routing-label-shift](/side-effects/exact-routing-label-shift.md) (`execution_trace` and local
  explanations route on the exact sklearn `decision_path`; `aggregated_transitions` still rounds the
  threshold before comparing, so the two can disagree at threshold boundaries — demonstrated with a
  constructed threshold at `2.4999`),
  [trace-consistent-lrc-deprecation](/side-effects/trace-consistent-lrc-deprecation.md)
  (`get_trace_consistent_lrc()` deprecated for `context_order > 1` in favor of `get_predicate_lrc`;
  demonstrated in a temporary worktree at `f16a977` because of the pairwise crash above),
  [communities-regressor-valueerror](/side-effects/communities-regressor-valueerror.md)
  (`extract_communities` on a regression DPG now raises a clear `ValueError` instead of an opaque
  `numpy.linalg.LinAlgError`),
  [gb-predict-original-model](/side-effects/gb-predict-original-model.md) (`evaluate_faithfulness` on a
  `GradientBoostingClassifier` crashed with `TypeError: list indices must be integers or slices, not
  tuple` at the 0.2.0 baseline `11decd3`; fixed pre-release by commit `be5389a`'s `_original_model`),
  [decimal-threshold-auto-warning](/side-effects/decimal-threshold-auto-warning.md)
  (`decimal_threshold="auto"` derives precision from the data and warns when a threshold lands off that
  grid; `get_decimal_threshold()` raises `DPGError` before `fit()`),
  [context-order-validation-errors](/side-effects/context-order-validation-errors.md) (`DPGError` for
  invalid or incompatible `context_order`/`decimal_threshold`; `resolve_context_order` raises
  `ValueError` for a bad or insufficient `max_k`), and
  [regression-sink-collisions](/side-effects/regression-sink-collisions.md) (a regression sink is only
  as unique as its 2-decimal rounded leaf value — demonstrated with two distinct leaves, `3.001` and
  `3.004`, colliding into one `"Pred 3.0"` sink).
* **Update**: [dpg.core](/modules/dpg-core.md), [dpg.explainer](/modules/dpg-explainer.md),
  [metrics.graph](/modules/metrics-graph.md), [config resolution](/conventions/config-resolution.md),
  and [graph construction modes](/conventions/graph-construction-modes.md) updated for 0.3.0's
  context-aware (`context_order`/DPG-k) construction, `decimal_threshold="auto"`, exact
  `decision_path` routing, and the classifier-only `extract_communities` guard.
* **Creation**: [dpg.context_order](/modules/dpg-context-order.md) and [dpg.cli](/modules/dpg-cli.md) —
  module pages for two files new in 0.3.0 (`resolve_context_order`'s trie-based DPG-k resolution, and
  the packaged `build_parser`/`main` the installed console script fails to reach — see
  [console-script-entrypoint](/side-effects/console-script-entrypoint.md)).
* **Update**: [Running the DPG CLI entrypoints](/workflows/cli-entrypoints.md) updated for `dpg/cli.py`
  and the console-script entry-point mismatch.
* **Update**: CLAUDE.md's Known breakage list gained the `explain_local`/`node_lookup` crash (see
  [explain-local-node-lookup-crash](/side-effects/explain-local-node-lookup-crash.md)).
* **Update**: Three code fixes landed on `feature/first_runs` on `feature/first_runs` (uncommitted, on top
  of HEAD `276a503`), closing out two of the crashes documented above:
  `discover_dfg_context`'s `pairwise(nodes, nodes[1:])` is now `pairwise(nodes)` (`dpg/core.py:636`;
  `tests/test_dpg_k.py:95` had the identical bug and is now `pairwise(labels)` — `uv run pytest
  tests/test_dpg_k.py` passes all 7 tests), `_trace_tree_path`'s vestigial `node_lookup` parameter is
  removed from its signature (`dpg/explainer.py`, the dead commented-out producer line at ~268 is also
  gone), and `resolve_context_order`/`get_context_order`/`get_context_order_history` are now typed
  `int`/`dict[int, int]` instead of `int | float`/`dict[int | float, int]` (`path_violations` and
  `_node_windows` still take `k: float`, unchanged, since they accept `math.inf`). Confirmed both
  fixes with a 3-tree `RandomForestClassifier` on iris: `context_order=2` now fits successfully, and
  `explain_local` now returns a populated explanation instead of raising `TypeError`. Updated
  [explain-local-node-lookup-crash](/side-effects/explain-local-node-lookup-crash.md) and
  [context-order-pairwise-crash](/side-effects/context-order-pairwise-crash.md) in place to describe
  the fix rather than the crash (kept, per project convention, as the one page per side effect — now
  documenting a fixed regression instead of a live one), moved both out of
  [side-effects/index.md](/side-effects/index.md)'s "Crashes at HEAD" section into a new "Fixed
  regressions" section, removed the two corresponding bullets from CLAUDE.md's Known breakage, and
  updated the stale crash descriptions and signatures in
  [dpg.core](/modules/dpg-core.md), [dpg.explainer](/modules/dpg-explainer.md),
  [dpg.context_order](/modules/dpg-context-order.md),
  [graph construction modes](/conventions/graph-construction-modes.md),
  [trace-consistent-lrc-deprecation](/side-effects/trace-consistent-lrc-deprecation.md) (the
  worktree-based reproduction can now also be run directly, without a worktree), and
  [exact-routing-label-shift](/side-effects/exact-routing-label-shift.md) (`_trace_tree_path`'s cited
  line range shifted from `645-696` to `645-695` after the `node_lookup` parameter was removed).
* **Creation**: [metrics.backends](/modules/metrics-backends.md) — node-centrality computation
  (betweenness, local reaching, closeness, harmonic) made pluggable by graph library. The four
  `calc_*_centrality` helpers and `_nx_to_igraph` moved out of `metrics/nodes.py` into
  `metrics/backends/igraph_backend.py` unchanged; two new implementations,
  `metrics/backends/networkx_backend.py` (the reference definitions every backend is checked
  against) and `metrics/backends/graph_tool_backend.py` (with a Wasserman–Faust `r/(n-1)`
  correction on top of graph_tool's own closeness, and a `pred_map`-walk reconstruction of local
  reaching centrality's shortest paths), sit alongside it behind a `GraphBackend` ABC / lazy
  `get_backend(name)` registry in `metrics/backends/__init__.py` and `base.py`.
  `NodeMetrics.extract_node_metrics` gained a `backend: str | GraphBackend = "igraph"` parameter
  (`dpg/explainer.py`'s `_get_node_metrics` and `dpg/sklearn_dpg.py`'s `test_dpg` now pass
  `backend=<builder>.metrics_backend` instead of relying on the igraph default).
  `DEFAULT_DPG_CONFIG["dpg"]["metrics"] = {"backend": "igraph"}` and a matching `config.yaml`
  section were added; `DecisionPredicateGraph.__init__` resolves `self.metrics_backend`
  per-key (same style as `perc_var`), validates it against `metrics.backends.BACKEND_NAMES` at
  construction (`DPGError` on an unknown name) without importing the backend library, and exposes
  it via `get_metrics_backend()`. graph_tool is conda-forge only, so a standalone `pixi.toml`
  (`pixi run pytest` from the repo root) was added for the graph_tool test path;
  `tests/test_metrics_backends.py` skips it under `uv run pytest` via
  `pytest.importorskip("graph_tool")`. A new ground-truth fixture mechanism,
  `tests/metrics_ground_truth/` (`scenarios.py`'s shared recipe, `generate.py` writing
  `<scenario>.json` keyed by node label from four RandomForest-on-CSV scenarios, regenerated with
  `uv run python -m tests.metrics_ground_truth.generate`), pins all three backends to
  networkx-derived values within `abs=1e-3`, independent of any backend implementation.
  `tests/test_metrics.py`'s igraph-specific parity tests were repointed at
  `metrics.backends.igraph_backend` after the move. Updated
  [metrics.nodes](/modules/metrics-nodes.md) (stale helper table, "Why igraph" section, and an
  iteration-order gotcha that no longer applies now that the four centrality dicts are indexed by
  node id rather than relied on for dict order) and
  [config resolution](/conventions/config-resolution.md) (`metrics.backend` resolution and the
  no-disagreement row) to match. Updated `CLAUDE.md`'s `metrics/` section, Config table, and
  Development setup.
