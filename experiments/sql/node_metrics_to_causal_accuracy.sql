-- Rebuild causal_accuracy_<ts>.csv from node_metrics_<ts>.csv (causal_synthetic_scenarios suite).
-- Scoring is the reference runner's, unchanged: run_dpg_causal_synthetic.py
-- (extract_top_k_features + evaluate_causal_accuracy). See
-- okf/docs/workflows/reconstruct-causal-accuracy.md.
--
-- Run (from the repo root; paths are relative to the working directory):
--   duckdb -csv -c "SET VARIABLE node_metrics_csv = 'experiments/causal_synthetic_scenarios/results/node_metrics_<ts>.csv'" \
--          -c ".read experiments/sql/node_metrics_to_causal_accuracy.sql" > causal_accuracy_rebuilt.csv
-- Both variables below can be overridden with SET VARIABLE before the .read.

-- Inputs: node_metrics CSV to read, and the top-k size (TOP_K = 3 in the runner).
SET VARIABLE node_metrics_csv = coalesce(getvariable('node_metrics_csv'),
    'experiments/causal_synthetic_scenarios/results/node_metrics_2026-10-15T05-08-31.csv');
SET VARIABLE top_k = coalesce(getvariable('top_k'), 3);

WITH
-- METRICS order from the runner, mapped from display name to the node_metrics column.
metric_columns(metric_order, metric, column_name) AS (
    VALUES
        (1, 'Local reaching centrality', 'local_reaching_centrality'),
        (2, 'Closeness centrality',      'closeness_centrality'),
        (3, 'Harmonic centrality',       'harmonic_centrality'),
        (4, 'Betweenness centrality',    'betweenness_centrality'),
        (5, 'Collective influence',      'collective_influence'),
        (6, 'Percolation centrality',    'percolation_centrality')
),

-- Ground-truth features per scenario (SCENARIO_GROUND_TRUTH in the runner), not read from the
-- file under verification. Experiment = <scenario>_<model>, so `scored` joins by scenario prefix.
ground_truth(scenario, gt_features) AS (
    VALUES
        ('scenario_1',         ['F1']),
        ('scenario_2_updated', ['F1']),
        ('scenario_3',         ['F1', 'F2']),
        ('scenario_5',         ['F1', 'F2'])
),

-- Raw node rows from the node_metrics CSV.
nodes AS (
    SELECT
        experiment,
        split, node_idx, label,
        local_reaching_centrality, closeness_centrality, harmonic_centrality,
        betweenness_centrality, collective_influence, percolation_centrality
    FROM read_csv(getvariable('node_metrics_csv'), header = true)
),

-- Drop Class and Pred sinks (str.startswith(("Class", "Pred"))); feature = first non-whitespace run
-- of the label. Pred leaves exist only in regression DPGs (causal forests), so classifier files are
-- unaffected.
predicates AS (
    SELECT *, regexp_extract(label, '([^\s]+)', 1) AS feature
    FROM nodes
    WHERE label NOT LIKE 'Class%' AND label NOT LIKE 'Pred%'
),

-- One row per (node, metric): unpivot the six metric columns into (metric, score).
long_scores AS (
    SELECT p.experiment, p.split, p.node_idx, p.feature, m.metric_order, m.metric,
        CASE m.column_name
            WHEN 'local_reaching_centrality' THEN p.local_reaching_centrality
            WHEN 'closeness_centrality'      THEN p.closeness_centrality
            WHEN 'harmonic_centrality'       THEN p.harmonic_centrality
            WHEN 'betweenness_centrality'    THEN p.betweenness_centrality
            WHEN 'collective_influence'      THEN p.collective_influence
            WHEN 'percolation_centrality'    THEN p.percolation_centrality
        END AS score
    FROM predicates p CROSS JOIN metric_columns m
),

-- Descending rank per (experiment, split, metric). pandas sorts NaN last, DuckDB sorts it first
-- in DESC, so NaN is pushed last explicitly. Ties break by node_idx (pandas order is undefined).
ranked AS (
    SELECT *, row_number() OVER (
        PARTITION BY experiment, split, metric
        ORDER BY isnan(score), score DESC, node_idx
    ) AS rnk
    FROM long_scores
),

-- First TOP_K rows, as a rank-ordered feature list. Duplicates are kept (e.g. F2;F2;F2).
top_k AS (
    SELECT experiment, split, metric_order, metric,
           list(feature ORDER BY rnk) AS top_features
    FROM ranked
    WHERE rnk <= getvariable('top_k')
    GROUP BY ALL
),

-- Join ground truth by scenario prefix (experiment starts with '<scenario>_'; starts_with avoids
-- LIKE's '_' wildcard). Precision/recall on sets: |distinct(top) & GT| / |distinct(top)| and / |GT|, 0.0 when empty.
scored AS (
    SELECT t.*, g.gt_features,
        list_sort(list_intersect(list_distinct(t.top_features), g.gt_features)) AS inter,
        len(list_intersect(list_distinct(t.top_features), g.gt_features)) AS inter_n,
        len(list_distinct(t.top_features)) AS top_n,
        len(list_distinct(g.gt_features)) AS gt_n
    FROM top_k t JOIN ground_truth g ON starts_with(t.experiment, g.scenario || '_')
),

-- Runner's columns and order. Lists are joined with ';'; intersection is emitted sorted.
output AS (
    SELECT
        experiment, split,
        getvariable('top_k')::INTEGER AS k,
        array_to_string(gt_features, ';')    AS ground_truth,
        array_to_string(top_features, ';')   AS explanation_top_features,
        metric,
        array_to_string(inter, ';')          AS intersection,
        CASE WHEN top_n > 0 THEN inter_n::DOUBLE / top_n ELSE 0.0 END AS precision,
        CASE WHEN gt_n  > 0 THEN inter_n::DOUBLE / gt_n  ELSE 0.0 END AS recall,
        metric_order
    FROM scored
)

SELECT * EXCLUDE (metric_order) FROM output ORDER BY experiment, split, metric_order;
