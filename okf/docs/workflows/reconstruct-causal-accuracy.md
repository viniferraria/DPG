---
type: Playbook
title: Rebuilding causal_accuracy from node_metrics with DuckDB
description: How experiments/sql/node_metrics_to_causal_accuracy.sql recomputes a causal-synthetic causal_accuracy_<ts>.csv from node_metrics_<ts>.csv with the reference runner's scoring, what each CTE does, the traps measured on two real runs, and what still differs for the other suites now that they share the reference schema.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/sql/node_metrics_to_causal_accuracy.sql
tags: [experiments, playbook, duckdb, sql, causal-accuracy]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-17T00:00:00Z }
status: stable
---

# When to use

Recompute `causal_accuracy_<ts>.csv` offline from a saved `node_metrics_<ts>.csv`, with no model refit.
Typical uses: audit a results file, check that a run's scoring matches the reference, or re-score
after a change that only touches the reading side. The scoring is the reference suite's, unchanged
([causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md), implemented in the shared
[experiments.causal_accuracy](/modules/experiments-causal-accuracy.md)): drop `Class` and `Pred` labels, map each
label to its feature name, sort descending by the metric, take the top `TOP_K = 3`, keep duplicates, then
precision and recall on the sets. It adds no new scoring. The `duckdb-causal-accuracy` agent
(`.claude/agents/duckdb-causal-accuracy.md`) exists to build, run and verify this kind of query, including
porting it to another suite's schema.

# How to run

Needs the DuckDB CLI (`duckdb`, on `PATH`). The `duckdb` Python package is not in the uv environment. From
the repo root:

```bash
duckdb -csv \
  -c "SET VARIABLE node_metrics_csv = 'experiments/causal_synthetic_scenarios/results/node_metrics_2026-10-15T05-08-31.csv'" \
  -c ".read experiments/sql/node_metrics_to_causal_accuracy.sql" > causal_accuracy_rebuilt.csv
```

`node_metrics_csv` and `top_k` (default 3) are set at the top of the file and can be overridden with
`SET VARIABLE` before the `.read`. Paths resolve against the working directory. Output columns and order
match the runner: `experiment, split, k, ground_truth, explanation_top_features, metric, intersection,
precision, recall`, rows ordered by experiment, split, then the runner's `METRICS` order.

# What each CTE does

| CTE | Step |
|---|---|
| `metric_columns` | Maps each `METRICS` display name to its snake_case node_metrics column, and keeps the runner's metric order. |
| `ground_truth` | Inline copy of `SCENARIO_GROUND_TRUTH` (scenarios 1, 2_updated, 3, 5). Never read from the file under verification, which would be circular. |
| `nodes` | Reads the CSV. No scenario column is derived; the model name is not stripped. |
| `predicates` | Drops labels starting with `Class` or `Pred` (the shared `extract_top_k_features` excludes both in every suite; `Pred` leaves exist only in regression DPGs, so classifier files are unaffected); `feature` = first run of non-whitespace in `label`. |
| `long_scores` | Unpivots the six metric columns into one `(metric, score)` row per node. |
| `ranked` | `row_number()` per (experiment, split, metric), ordered `isnan(score), score DESC, node_idx`. |
| `top_k` | First `top_k` ranks as a rank-ordered feature list; duplicates kept (`F2;F2;F2`). |
| `scored` | Joins ground truth by prefix, `starts_with(experiment, scenario || '_')` (not `LIKE`, whose `_` is a wildcard); computes set intersection and the set sizes. |
| `output` | Precision and recall (0.0 on an empty denominator), lists joined with `;`, intersection sorted. |

Experiments whose scenario is not in `ground_truth` (for example the opt-in `--scm` runs, whose truth
lives in each `ground_truth.yaml`) are dropped by the inner join. To score them, add their truth to the
`ground_truth` CTE.

Caveat on the prefix join: a scenario name that is a prefix of another (for example `scenario_2` and
`scenario_2_updated`) would match both, duplicating rows for the longer one. Today `ground_truth` holds
only `scenario_2_updated`, so it does not occur. No code guards against it.

# Known traps

Measured on the two verification files below. Both node_metrics files hold 70,255 rows, 16
(experiment, split) groups, no `nan` values, no `Pred` labels and no duplicate (experiment, split, node_idx)
keys. Both predate the shared module but have its schema (the reference header did not change).

- **Ties.** pandas `sort_values` (unstable quicksort) and DuckDB `ORDER BY` order ties arbitrarily. The
  query breaks ties by `node_idx`. This is the only source of difference found (see Verification).
- **NaN.** pandas sorts NaN last, DuckDB's `DESC` puts it first. The query orders `isnan(score)` first, so
  it matches pandas. No NaN occurs in these two files, so that branch is untested on data.
- **Set order.** The two verification files predate the shared module, so their `intersection` and
  ground-truth join come from Python sets, in arbitrary order. The query emits the intersection sorted,
  and runs from 2026-10-17 on write it sorted too (`causal_accuracy_rows`); compare as sets across
  old and new files.
- **Floats.** Python writes `repr(float)`. Compare precision and recall numerically (tolerance 1e-12), not
  as text.
- **Append mode.** Both writers open with `a+`, so one file can hold several runs. No duplicate
  (experiment, split, metric) key occurred in either file (96 rows, 96 distinct keys in the second).
- **Line endings.** Old `causal_accuracy` files are CRLF, new ones LF; DuckDB reads both, but
  `cmp` against the rebuilt output needs `tr -d '\r'` first ([why](/side-effects/csv-line-endings-lf.md)).
- **NULL compare.** An empty `intersection` is NULL after `string_split`; use `IS NOT DISTINCT FROM` when
  comparing, or those rows silently drop out of the "exact" count.

# Verification

Rebuilt rows were joined to the runner's file on `(experiment, split, metric)`. A differing row counts as
tie-only when, at every top-k position, the runner's feature belongs to the group of nodes sharing the
score at that rank (a necessary check, so it does not prove the whole multiset is reachable).

| Pair (same timestamp) | (experiment, split) in node_metrics / in accuracy | Overlap rows | Exact | Tie-only diff | Other diff |
|---|---|---|---|---|---|
| `2026-10-01T03-42-13` | 16 / 1 | 6 | 5 | 1 | 0 |
| `2026-10-15T05-08-31` | 16 / 16 | 96 | 94 | 2 | 0 |

Re-run on 2026-10-17 after adding `AND label NOT LIKE 'Pred%'` to `predicates`: the second pair is still 96
rows, 94 exact, 2 tie-only, 0 other (intersection compared as sets; the two tie-only rows are the ones
listed below).

The first pair does not cover the same set: its accuracy file holds only `scenario_5_ExtraTrees` split 1
(6 rows), so verification is on that overlap. The query produced 96 rows for it, 90 of them with nothing
to compare against.

Tie-only rows (runner vs query):

- `scenario_5_ExtraTrees`, split 1, Collective influence (both pairs): `F8_Y;F1;F2` vs `F8_Y;F2;F5`. Ranks
  2 and 3 sit inside a group of three nodes at score 9.0 (`F2`, `F5`, `F1`).
- `scenario_1_ExtraTrees`, split 1, Collective influence (second pair): `F1;F1;F2` vs `F1;F1;F1`. Rank 3 is
  in a group of three nodes at score 48 (`F1 <= 1.145`, `F2 > -1.077`, `F2 <= 0.44`).

Every other column of every overlapping row matched (ground truth, intersection, precision, recall). The
result is exact on all rows except the three tie-order rows above, which differ only in which tied node
the runner's sort happened to pick.

# Other suites (MONK's, ACIC 2019, COMPAS)

Since 2026-10-17 all four suites write `node_metrics_<ts>.csv` in the same reference schema
(`experiment, split, node, degree, in_degree, out_degree, <metrics>, node_idx, label, processing_time`,
COMPAS adds `seed` last), so this query reads any **new** file of any suite without column changes, and
`Pred` is already excluded. No new variants were built. What still differs per suite:

- **Ground truth.** The `ground_truth` CTE is the causal-synthetic one. Each suite has its own source:
  MONK's `SCENARIO_GROUND_TRUTH` (`monks-1`: `a1;a2;a5`, `monks-3`: `a5;a4;a2`), ACIC's by DGP id (`dgp64_*`:
  `V3`, `dgp39_*`: `V8;V19`), COMPAS's `REFERENCE_FEATURES` (`priors_count;age`, a published reference
  ranking, not a ground truth). The prefix join `starts_with(experiment, scenario || '_')` works if the CTE
  rows are the suite's run-key prefixes (`monks-1`, `dgp64`, `compas`).
- **`feature_name` mapping (MONK's, COMPAS).** Their rankings map one-hot labels back to the source attribute
  (`a1_2` to `a1`, `race_Caucasian` to `race`) before taking the top 3. The query's `feature` expression
  would have to apply the same mapping: for MONK's `regexp_replace(feature, '_\d+$', '')`; for COMPAS the
  `CATEGORICAL` prefixes `c_charge_degree`, `race`, `sex`. Without it the rebuilt top-k lists are one-hot
  column names and will not match.
- **MONK's de-duplication.** MONK's passes `drop_duplicates=True`, so the `top_k` CTE would need to rank
  distinct features (keep each feature's best-ranked node, then take `top_k`) instead of the first `top_k`
  nodes.
- **Old files.** Pre-2026-10-17 MONK/ACIC/COMPAS `node_metrics` files use `node_index`, `split_idx`,
  `in_degree_nodes`, `out_degree_nodes`; rename on read ([mapping](/side-effects/node-metrics-schema-unification.md)).
- **Not checked.** Whether the ACIC/COMPAS/MONK's tie behaviour matches; no pair from these suites was run
  through the query.

# Updating it

When `TOP_K`, `METRICS`, `SCENARIO_GROUND_TRUTH`, the label contract, or the node_metrics columns change
in the reference runner, edit the SQL and the table above in the same change, re-run the verification, and
add a dated entry to [the log](/log.md).
