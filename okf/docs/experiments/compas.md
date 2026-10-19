---
type: Experiment Suite
title: COMPAS recidivism score
description: Fits tree ensembles to ProPublica's COMPAS violent-recidivism score and checks whether DPG feature rankings agree with the reference ranking reported by Vuk et al. (priors and age); real data, so there is no ground truth.
resource: https://github.com/viniferraria/DPG/tree/main/experiments/compas
tags: [experiments, compas, real-data, fairness, one-hot, centrality, reference-ranking]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-07T00:00:00Z }
status: experimental
---

# Purpose

COMPAS is used in "Provable Recovery of Locally Important Signed Features and Interactions from Random
Forest" (Vuk, Ihlo, Behr; https://arxiv.org/abs/2512.11081v2, Sec. 5.3). That paper is about a random
forest local-importance method, not DPG; this suite only reuses its dataset and its qualitative finding.
The structure mirrors [MONK's problems](monks.md): same `METRICS`, top-k scoring, `results/` / `states/`
outputs, all through the shared [experiments.causal_accuracy](/modules/experiments-causal-accuracy.md) module.

# Reference, not ground truth

COMPAS has no structural causal model, so **there is no feature-level ground truth**. The paper reports no
numeric ranking either; it says prior offenses and age were "the most important features globally",
"signed interactions between age and prior offenses are the largest among all pairwise interactions", and
that ethnicity "exhibited lower importance". The runner encodes only that as
`REFERENCE_FEATURES = ["priors_count", "age"]`. `TOP_K = 3`, as in the reference runner, so precision is at
most 2/3. Precision/recall in
`causal_accuracy_*.csv` therefore mean *agreement with a published random-forest ranking of the COMPAS
score*, not recovery of a cause. The `ground_truth` column keeps the monks name for schema parity.

# Data

| Item | Value |
|---|---|
| File | `compas-scores-two-years.csv` from https://github.com/propublica/compas-analysis (7,214 rows, 53 columns, ~2.5 MB) |
| Filters | `days_b_screening_arrest` in [-30, 30], `is_recid != -1`, `c_charge_degree != 'O'`, `v_score_text` in Low/Medium/High (ProPublica's standard filters, applied to the violent score), then African-American and Caucasian only (the paper's restriction) |
| Features | `age`, `priors_count`, `c_charge_degree`, `race`, `sex` (the paper's five); categoricals one-hot encoded |
| Target | `v_score_text != "Low"` (the paper says only "high (= 1) and low (= 0)"; the Medium grouping is an assumption) |
| Models | RandomForest / ExtraTrees, 100 trees, `min_samples_leaf=5`; stratified 80/20 split per seed in `SEEDS = [42..46]` |

`dpg_config` is passed explicitly (`perc_var=0.0001`, `decimal_threshold=3`, igraph backend).

# Outputs

`results/node_metrics_<ts>.csv` in the reference schema (`experiment, split, node, degree, in_degree,
out_degree`, every `NodeMetrics` column in frame order, `node_idx, label, processing_time`) plus `seed` as
the **last** column (`node_metrics_rows(..., seed=seed)`); `split` is the seed's index in `SEEDS`, `seed`
the value. Old files have `node_index`, `in_degree_nodes`, `out_degree_nodes`, `split_idx` and `seed`
before `processing_time`, see [node_metrics schema change](/side-effects/node-metrics-schema-unification.md).
`results/causal_accuracy_<ts>.csv` has the reference columns. Both are appended per (seed, model) with LF
line endings ([why](/side-effects/csv-line-endings-lf.md)); `states/` pickles as before.

# Scoring

The reference scoring, imported from the shared module: `causal_accuracy_rows` ranks predicate nodes by each
metric, takes the top `TOP_K` node labels, maps each to its feature name and computes precision and recall
(`extract_top_k_features` and `evaluate_causal_accuracy` underneath). `TOP_K`, `METRICS` and the CSV columns
(`experiment, split, k, ground_truth, explanation_top_features, metric, intersection, precision, recall`,
`intersection` sorted) are the reference's.

Named deviation (forced by the data), a call-site keyword: `feature_name=base_feature_name` maps one-hot
dummies back to their parent feature before scoring (`race_Caucasian` -> `race`), as MONK's does. Unlike
MONK's, `drop_duplicates` is left at its default, so features are not de-duplicated. `Class` and `Pred`
labels are both excluded; COMPAS is a classifier, so `Pred` never occurs (before 2026-10-17 only `Class`
was excluded).

An earlier version summed node scores per feature and wrote a `race_rank` column. Both were removed; they
are recorded as untested ideas in [per-feature score aggregation](/opportunities/per-feature-score-aggregation.md)
and [true-feature rank column](/opportunities/true-feature-rank-column.md).

# How to run

```bash
cd experiments/compas
uv run python run_compas.py --models RandomForest --seeds 42 --output-dir results
```

Smoke result (RandomForest, seed 42). **Superseded (summed scoring):** produced when the runner summed node
scores per feature with `TOP_K = 2` and wrote `race_rank`; not re-run with the reference scoring, not a current
result. Held-out accuracy 0.8191 on 1,056 rows; all six metrics returned
top-2 `{priors_count, age}`; `race_rank` was 3 (LRC, closeness, harmonic) and 4-5 (betweenness, collective
influence, percolation). The explainer `fit` took about five minutes (4,222 training rows, 100 trees).

# Limitations, side effects and errors

## The CSV is not committed (download side effect)

ProPublica's repository has no license file (GitHub's license API returns 404 and the README contains no
licence text), and the CSV contains defendant names and dates of birth. It is therefore git-ignored
(`experiments/compas/.gitignore`). `download_dataset()` fetches it on first run into `datasets/`.
Example: offline, `urllib.error.URLError` propagates out of `run_experiments` before any model is fitted;
this call is outside the per-run `try/except`.

## Names and birth dates stay out of the pipeline

`load_dataset` selects only `FEATURES`; identifiers are never in `X`, pickled states, or results. The
downloaded CSV on disk still contains them.

## Predicate labels are one-hot column names

A node reads `race_Caucasian <= 0.5`. `base_feature_name` matches the `CATEGORICAL` prefixes, so
`c_charge_degree_F` -> `c_charge_degree` while `priors_count` is untouched. Example: a new categorical
column not listed in `CATEGORICAL` would be scored as its own feature.

## Many-split features have many nodes

Continuous `age` and `priors_count` create many threshold nodes while binary `race`/`sex` have two, so they have
more chances to fill the top-`TOP_K` node labels. The paper makes the same point (importance measures
favouring features with more distinct values). A low rank for `race` is thus weak evidence of fairness. How a
feature's nodes are combined is an [open idea](/opportunities/per-feature-score-aggregation.md).

## Betweenness and percolation values are unreliable

Both were inverted (raw edge frequency used as distance) and are being fixed concurrently; see
[betweenness/percolation inverted weights](/side-effects/betweenness-percolation-inverted-weights.md).
They are reported but should not be interpreted until the fix lands; re-run afterwards.

## Failures are swallowed per run

The explanation block is in `try/except Exception`; the accuracy line is still logged. Example: a failing
metric yields a logged error and a missing `causal_accuracy` row, not a crash.

## Explanations of the score, not of recidivism

The target is COMPAS's own violent-risk score, so the ranking explains a proprietary algorithm's
behaviour, not who reoffends. The paper's CV ROC-AUC of ~0.82 (ranger, 500 trees, 10-fold CV) is not
reproduced: this runner uses sklearn and a single split per seed.

# Data-quality and fairness caveats from the literature

- Angwin, Larson, Mattu, Kirchner, "Machine Bias", ProPublica 2016: black defendants were almost twice as
  likely as white defendants to be labelled higher risk without reoffending. Method: Larson et al., "How
  we analyzed the COMPAS recidivism algorithm", ProPublica 2016.
- Northpointe's rebuttal (Dieterich et al. 2016) argues predictive parity holds; Chouldechova, "Fair
  prediction with disparate impact" (2017) shows both cannot hold when base rates differ.
- Bao et al., "It's COMPASlicated" (NeurIPS 2021 Datasets track): the dataset is a poor fairness
  benchmark; label noise, sampling and preprocessing choices change conclusions.
- `two_year_recid` and arrests measure policing and arrests, not offending; the sample is Broward County,
  FL 2013-2014 only.
- Dressel and Farid (Science Advances 2018): simple models with few features match COMPAS accuracy.
- Restricting to two races discards Hispanic, Asian, Native American and Other defendants (about 15%).

None of these are verified by this suite; they are cited from the literature.

# Related

- [Opportunities](/opportunities/index.md)
- [Causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md)
