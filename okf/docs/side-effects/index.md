# Side effects and limitations

Side effects, regressions, and limitations introduced by — or still present after — the DPG 0.3.0
release (merge `f16a977`), verified against branch `feature/first_runs` HEAD `276a503` unless noted
otherwise. Each page traces its claim to source at an exact revision and includes a runnable example
that was actually executed.

## Crashes at HEAD

* [console-script-entrypoint](console-script-entrypoint.md) - The installed `dpg` console script points at a nonexistent module; `[project.scripts]` silently wins over `[tool.poetry.scripts]`.

## Fixed regressions (on `feature/first_runs`, after `276a503`)

* [explain-local-node-lookup-crash](explain-local-node-lookup-crash.md) - `DPGExplainer.explain_local` raised `TypeError` for every model; a ruff cleanup commented out `node_lookup` but left `_trace_tree_path` requiring it. Fixed by deleting the now-unused parameter.
* [context-order-pairwise-crash](context-order-pairwise-crash.md) - `discover_dfg_context` raised `TypeError` for every `context_order > 1` fit; a "ruff fixes" commit called `itertools.pairwise` with two arguments instead of one. Fixed by dropping the second argument.

## Behavior changes vs 0.2.0

* [exact-routing-label-shift](exact-routing-label-shift.md) - `execution_trace` and local explanations now route on the exact sklearn decision path; `aggregated_transitions` still routes on the rounded threshold, so the two can disagree at threshold boundaries.
* [trace-consistent-lrc-deprecation](trace-consistent-lrc-deprecation.md) - `get_trace_consistent_lrc()` is deprecated for `context_order > 1`; `get_predicate_lrc(graph)` is the replacement and aggregates differently.
* [communities-regressor-valueerror](communities-regressor-valueerror.md) - `extract_communities` on a regression DPG now raises a clear `ValueError` instead of an opaque `numpy.linalg.LinAlgError`.
* [gb-predict-original-model](gb-predict-original-model.md) - `evaluate_faithfulness` on a `GradientBoostingClassifier` used to crash; fixed pre-release by keeping `_original_model` for sklearn's own `predict()`.

## New config surface and its failure modes

* [decimal-threshold-auto-warning](decimal-threshold-auto-warning.md) - `decimal_threshold="auto"` derives precision from the data and warns when a threshold lands off that grid; `get_decimal_threshold()` errors before `fit()`.
* [context-order-validation-errors](context-order-validation-errors.md) - `DPGError` for invalid or incompatible `context_order`/`decimal_threshold` config; `resolve_context_order` raises `ValueError` for a bad or insufficient `max_k`.

## Pre-existing / deferred limitations

* [regression-sink-collisions](regression-sink-collisions.md) - A regression sink is only as unique as its 2-decimal rounded leaf value; distinct leaves with distinct predictions can collide into one graph sink.

## See also

* [dpg.core](/modules/dpg-core.md) - Where most of these behaviors are implemented.
* [dpg.explainer](/modules/dpg-explainer.md) - `explain_local`, `evaluate_faithfulness`, and node-metrics wiring.
* [dpg.context_order](/modules/dpg-context-order.md) - `resolve_context_order`, the DPG-k resolver.
* [Graph construction modes](/conventions/graph-construction-modes.md) - `aggregated_transitions` vs `execution_trace`, and where `context_order` fits.
* [CHANGELOG.md](https://github.com/viniferraria/DPG/blob/main/CHANGELOG.md) - The 0.3.0 release notes these pages verify against source.
