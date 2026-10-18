"""Shared node-metrics export and causal-accuracy scoring for the experiment suites.

Scoring follows ``causal_synthetic_scenarios/run_dpg_causal_synthetic.py``: rank
predicate nodes by each metric, take the top ``TOP_K`` features, and compare
them with the ground truth. A new ``NodeMetrics`` column is exported without
code changes here; add it to ``METRICS`` to score it.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

TOP_K = 3
METRICS = [
    "Local reaching centrality",
    "Closeness centrality",
    "Harmonic centrality",
    "Betweenness centrality",
    "Collective influence",
    "Percolation centrality",
]

# Display name of a fixed ``NodeMetrics`` column -> canonical CSV name.
FIXED_COLUMNS = {
    "Node": "node",
    "Degree": "degree",
    "In degree nodes": "in_degree",
    "Out degree nodes": "out_degree",
    "Label": "label",
}


def node_metrics_rows(
    explanation: Any,
    experiment: str,
    split: int,
    processing_time: float,
    **extra: Any,
) -> pd.DataFrame:
    """Return the explanation's node-metrics frame as typed CSV rows.

    Columns: experiment, split, node, degree, in_degree, out_degree, every other
    metric column (snake_cased, float), node_idx, label, processing_time, then
    ``extra`` columns in keyword order.
    """
    df = explanation.node_metrics
    n = len(df)
    cols: dict[str, Any] = {
        "experiment": [experiment] * n,
        "split": [split] * n,
        "node": df["Node"].astype(str).to_numpy(),
        "degree": df["Degree"].astype(int).to_numpy(),
        "in_degree": df["In degree nodes"].astype(int).to_numpy(),
        "out_degree": df["Out degree nodes"].astype(int).to_numpy(),
    }
    for name in df.columns:
        if name not in FIXED_COLUMNS:
            cols[name.lower().replace(" ", "_")] = df[name].astype(float).to_numpy()
    cols["node_idx"] = df.index.astype(int).to_numpy()
    cols["label"] = df["Label"].astype(str).to_numpy()
    cols["processing_time"] = [round(processing_time, 4)] * n
    for key, value in extra.items():
        cols[key] = [value] * n
    return pd.DataFrame(cols)


def extract_top_k_features(
    node_metrics: pd.DataFrame,
    top_k: int,
    metric: str = "Local reaching centrality",
    *,
    feature_name: Callable[[str], str] | None = None,
    drop_duplicates: bool = False,
) -> list[str]:
    """Return the top-k features by ``metric`` from a node-metrics frame.

    ``Class``/``Pred`` sinks are excluded. Several predicates on one feature can
    fill the top-k (e.g. ``F2;F2;F2``); that is kept by default so the CSV history
    stays comparable, ``drop_duplicates=True`` keeps each feature's best-ranked
    predicate. ``feature_name`` maps the first label token to its source attribute.
    """
    results = node_metrics[
        ~node_metrics["Label"].str.startswith(("Class", "Pred"))
    ].copy()
    results["Label"] = results["Label"].str.extract(r"([^\s]+)")
    if feature_name is not None:
        results["Label"] = results["Label"].map(feature_name)
    ranked = results.sort_values(by=metric, ascending=False)["Label"]
    if drop_duplicates:
        ranked = ranked.drop_duplicates()
    return ranked.head(top_k).tolist()


def evaluate_causal_accuracy(
    ground_truth: list[str],
    explanation: list[str],
) -> tuple[set[str], float, float]:
    """Return the causal accuracy of an explanation against a ground-truth feature list."""
    set_gt = set(ground_truth)
    set_expl = set(explanation)
    # intersection c_k = |E ∩ GT|, where E is the set of top-k features in the explanation and GT
    intersection_expl_gt = set_expl.intersection(set_gt)  # c_k ∩ c
    cardinality = len(intersection_expl_gt)  # |c_k ∩ c|
    # |c_k| len(set_expl) and |c| len(set_gt)
    precision = cardinality / len(set_expl) if set_expl else 0.0
    recall = cardinality / len(set_gt) if set_gt else 0.0
    return intersection_expl_gt, precision, recall


def causal_accuracy_rows(
    node_metrics: pd.DataFrame,
    ground_truth: list[str],
    experiment: str,
    split: int,
    *,
    top_k: int = TOP_K,
    feature_name: Callable[[str], str] | None = None,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Return one causal-accuracy row per metric in ``METRICS``."""
    rows = []
    for metric in METRICS:
        top = extract_top_k_features(
            node_metrics,
            top_k,
            metric,
            feature_name=feature_name,
            drop_duplicates=drop_duplicates,
        )
        intersection, precision, recall = evaluate_causal_accuracy(ground_truth, top)
        rows.append({
            "experiment": experiment,
            "split": split,
            "k": top_k,
            "ground_truth": ";".join(ground_truth),
            "explanation_top_features": ";".join(top),
            "metric": metric,
            "intersection": ";".join(sorted(intersection)),
            "precision": precision,
            "recall": recall,
        })
    return pd.DataFrame(rows)


def append_csv(rows: pd.DataFrame, path: Path) -> None:
    """Append ``rows`` to ``path``, writing the header only when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False, na_rep="nan")
