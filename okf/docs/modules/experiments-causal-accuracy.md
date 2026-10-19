---
type: Python Module
title: experiments.causal_accuracy — shared scoring and node-metrics export
description: The one module that holds the reference causal-accuracy scoring and the node_metrics/causal_accuracy CSV writers for every experiment suite; API table, schema rules, call-site deviation keywords, and how to add a metric.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_accuracy.py
tags: [experiments, causal-accuracy, scoring, csv, node-metrics, reference-protocol]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-17T00:00:00Z }
status: stable
---

# Responsibility

`experiments/causal_accuracy.py` is the single home of the reference scoring that used to live inside
`run_dpg_causal_synthetic.py` and was copied into MONK's, ACIC's and COMPAS's runners. The reference
runner, `run_monk.py`, `run_acic2019.py` and `run_compas.py` all import from it, so every suite scores
and writes the same way. The reference runner re-exports `TOP_K`, `METRICS`, `extract_top_k_features`,
`evaluate_causal_accuracy`, `append_csv`, `causal_accuracy_rows` and `node_metrics_rows` (marked
`# noqa: F401`), so `ref.extract_top_k_features` still resolves for tests and callers that used the old
location.

`experiments/` has no `__init__.py`; `from experiments.causal_accuracy import ...` resolves because the
installed `dpg.pth` puts the repo root on `sys.path` (checked in `.venv/lib/python3.14/site-packages`),
so `uv run python experiments/<suite>/run_*.py` works from any directory.

# API

| Name | Signature | Notes |
|---|---|---|
| `TOP_K` | `3` | The single top-k used by every suite. |
| `METRICS` | list of 6 display names | `Local reaching centrality`, `Closeness centrality`, `Harmonic centrality`, `Betweenness centrality`, `Collective influence`, `Percolation centrality`. The scored set, in CSV row order. |
| `FIXED_COLUMNS` | dict | Display name to canonical CSV name for the five fixed `NodeMetrics` columns: `Node` to `node`, `Degree` to `degree`, `In degree nodes` to `in_degree`, `Out degree nodes` to `out_degree`, `Label` to `label`. |
| `node_metrics_rows` | `(explanation, experiment, split, processing_time, **extra) -> DataFrame` | Typed rows from `explanation.node_metrics`. Column order below. `extra` keywords become trailing columns in keyword order (COMPAS passes `seed=`). |
| `extract_top_k_features` | `(node_metrics, top_k, metric="Local reaching centrality", *, feature_name=None, drop_duplicates=False) -> list[str]` | Drops `Class`/`Pred` labels, takes the first whitespace-free token of the label, optionally maps it with `feature_name`, sorts descending by `metric`, optionally `drop_duplicates()`, returns `head(top_k)`. |
| `evaluate_causal_accuracy` | `(ground_truth, explanation) -> (set, float, float)` | Set intersection, precision `|E and GT| / |E|`, recall `|E and GT| / |GT|`; 0.0 on an empty denominator. Unchanged from the reference. |
| `causal_accuracy_rows` | `(node_metrics, ground_truth, experiment, split, *, top_k=TOP_K, feature_name=None, drop_duplicates=False) -> DataFrame` | One row per `METRICS` entry. Columns `experiment, split, k, ground_truth, explanation_top_features, metric, intersection, precision, recall`; list fields `;`-joined, `intersection` sorted. |
| `append_csv` | `(rows, path) -> None` | Creates the parent directory, appends, writes the header only when the file is new, `na_rep="nan"`, `index=False`. |

## `node_metrics_rows` column order

`experiment, split, node, degree, in_degree, out_degree`, then **every other column of the frame in frame
order**, snake_cased by `name.lower().replace(" ", "_")` and cast to float, then `node_idx` (the frame
index), `label`, `processing_time` (rounded to 4 dp), then the `extra` keywords. With the current
`NodeMetrics` that is `betweenness_centrality, local_reaching_centrality, closeness_centrality,
harmonic_centrality, collective_influence, local_clustering_coefficient, percolation_centrality` between
`out_degree` and `node_idx`. `node` is cast to `str`, the degrees to `int`.

This is the same header the reference runner wrote before the refactor (the 16-column
`node_metrics_2026-10-15T05-08-31.csv`), so the reference's header did not change; the other suites'
did (see [node_metrics schema change](/side-effects/node-metrics-schema-unification.md)).

# Deviations are call-site keywords

The reference protocol is the default. A suite deviates only where its data forces it, by passing a
keyword to `causal_accuracy_rows` (which forwards it to `extract_top_k_features`):

| Suite | Keywords | Why |
|---|---|---|
| Causal synthetic (reference) | none | Reference. |
| MONK's | `feature_name=base_feature_name, drop_duplicates=True` | Maps one-hot `a1_2` back to `a1`; de-duplicates features before `head(top_k)` (existing MONK deviation). |
| COMPAS | `feature_name=base_feature_name` | Maps one-hot `race_Caucasian` back to `race`; no de-duplication. |
| ACIC 2019 | none | `Pred` exclusion is now universal, so the old ACIC-only deviation is gone. |

Nothing else differs between suites in scoring. A suite page must name every keyword it passes.

# Adding a metric

A new `NodeMetrics` column needs no edit in the shared module to be **exported**: `node_metrics_rows`
snake_cases every column that is not in `FIXED_COLUMNS` and writes it in frame order. To **score** it,
add its display name to `METRICS` (one edit; every suite and the SQL rebuild then see it). The
DuckDB rebuild's `metric_columns` CTE must get the same row, see
[rebuilding causal_accuracy](/workflows/reconstruct-causal-accuracy.md). `Local clustering coefficient`
is exported but deliberately not in `METRICS` (integer-like ties make its top-k arbitrary).

Per the experiment protocol, other ways of turning node scores into feature scores are not added here;
they are recorded under [opportunities](/opportunities/index.md).

# `Pred` labels are excluded in every suite

`extract_top_k_features` filters `startswith(("Class", "Pred"))`. Before the refactor the reference runner
and MONK's/COMPAS excluded only `Class`; ACIC alone excluded `Pred` too, as a named deviation. `Pred` leaves
exist only in regression DPGs (causal forests), so for the classifier suites (causal synthetic, MONK's,
COMPAS) the change has no effect on any result; for ACIC it is unchanged. It would only matter if a
regression model were added to a suite that used to keep `Pred` nodes in its ranking. The label formats
are in the [label contract](/conventions/label-contract.md).

# Output behaviour worth knowing

- **Write timing.** The reference runner collects every split's node-metrics rows and calls `append_csv`
  once at the end of the run; its causal-accuracy rows are appended per split. MONK's, ACIC and COMPAS
  append both files per split (COMPAS per seed and model). A crash mid-run therefore loses the reference's
  `node_metrics` rows but not the others'.
- **Empty runs write nothing.** The reference runner skips `append_csv` when no split produced rows (no
  header-only file).
- **Line endings are LF** (pandas `to_csv`); see [CSV line endings](/side-effects/csv-line-endings-lf.md).
- **NaN is written as the text `nan`** (`na_rep="nan"`), matching what `csv.DictWriter` wrote for a float
  NaN.

# Tests

`tests/test_causal_accuracy.py` covers: the header against the reference's 16 columns and rounding/dtypes
(`test_header_matches_reference`); a new metric column flowing through and `seed` landing last
(`test_extra_metric_and_seed_positions`); sink exclusion, `feature_name` and `drop_duplicates`
(`test_top_k_excludes_sinks_and_dedup_options`); the `causal_accuracy_rows` columns, `METRICS` row order and
sorted `intersection` (`test_causal_accuracy_rows_sorted_intersection`); parity with the reference
functions on a frame with no `Pred` rows (`test_parity_with_reference`); and `append_csv` header-once and
`nan` (`test_append_csv_header_once_and_nan`).

# Related

- [Causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md) - the reference runner.
- [Rebuilding causal_accuracy with DuckDB](/workflows/reconstruct-causal-accuracy.md)
- [Opportunities](/opportunities/index.md)
