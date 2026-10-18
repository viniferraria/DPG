"""Unit tests for the shared ``experiments.causal_accuracy`` module."""

import types

import networkx as nx
import pandas as pd

import experiments.causal_accuracy as ca
import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as ref
from metrics.nodes import NodeMetrics


def _real_node_metrics() -> pd.DataFrame:
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [("a", "b", 3), ("a", "c", 2), ("b", "d", 3), ("c", "d", 1), ("c", "e", 1)]
    )
    nodes_list = [
        ["a", "F1 <= 0.5"],
        ["b", "F2 <= 1.5"],
        ["c", "F2 > 1.5"],
        ["d", "Class 0"],
        ["e", "Class 1"],
    ]
    return NodeMetrics.extract_node_metrics(G, nodes_list)


def _expl(df: pd.DataFrame) -> types.SimpleNamespace:
    return types.SimpleNamespace(node_metrics=df)


def test_header_matches_reference():
    rows = ca.node_metrics_rows(_expl(_real_node_metrics()), "exp", 0, 1.23456)
    assert list(rows.columns) == [
        "experiment", "split", "node", "degree", "in_degree", "out_degree",
        "betweenness_centrality", "local_reaching_centrality", "closeness_centrality",
        "harmonic_centrality", "collective_influence", "local_clustering_coefficient",
        "percolation_centrality", "node_idx", "label", "processing_time",
    ]  # fmt: skip
    assert rows["processing_time"].iloc[0] == 1.2346
    assert rows["node"].map(type).eq(str).all()
    assert rows["degree"].dtype.kind == "i"
    assert rows["betweenness_centrality"].dtype.kind == "f"


def test_extra_metric_and_seed_positions():
    df = _real_node_metrics()
    df["New metric"] = 1.0
    cols = list(ca.node_metrics_rows(_expl(df), "e", 0, 0.0, seed=7).columns)
    assert cols.index("out_degree") < cols.index("new_metric")
    assert cols[cols.index("node_idx") - 1] == "new_metric"
    assert cols[-1] == "seed"
    assert cols[-2] == "processing_time"


FRAME = pd.DataFrame(
    {
        "Label": ["F2 <= 0.5", "F2 > 0.5", "F1 <= 1.0", "F3 > 2.0", "Class 0", "Pred 1"],
        **{
            m: v
            for m, v in zip(
                ca.METRICS,
                [[0.6, 0.5, 0.4, 0.3, 0.99, 0.98], [0.1, 0.2, 0.3, 0.4, 0.9, 0.8]] * 3,
                strict=True,
            )
        },
    }
)


def test_top_k_excludes_sinks_and_dedup_options():
    assert ca.extract_top_k_features(FRAME, 3) == ["F2", "F2", "F1"]
    assert ca.extract_top_k_features(FRAME, 3, drop_duplicates=True) == ["F2", "F1", "F3"]
    mapped = ca.extract_top_k_features(
        FRAME, 3, feature_name=lambda s: s.replace("F2", "F1"), drop_duplicates=True
    )
    assert mapped == ["F1", "F3"]


def test_causal_accuracy_rows_sorted_intersection():
    gt = ["F2", "F1", "F9"]
    out = ca.causal_accuracy_rows(FRAME, gt, "exp", 2)
    assert list(out.columns) == [
        "experiment", "split", "k", "ground_truth", "explanation_top_features",
        "metric", "intersection", "precision", "recall",
    ]  # fmt: skip
    assert list(out["metric"]) == ca.METRICS
    assert out["intersection"].iloc[0] == "F1;F2"


def test_parity_with_reference():
    gt = ["F1", "F3"]
    frame = FRAME[~FRAME["Label"].str.startswith("Pred")]  # reference keeps Pred
    out = ca.causal_accuracy_rows(frame, gt, "exp", 0)
    for _, row in out.iterrows():
        top = ref.extract_top_k_features(frame, ref.TOP_K, row["metric"])
        _, p, r = ref.evaluate_causal_accuracy(gt, top)
        assert row["explanation_top_features"] == ";".join(top)
        assert (row["precision"], row["recall"]) == (p, r)


def test_append_csv_header_once_and_nan(tmp_path):
    path = tmp_path / "sub" / "out.csv"
    df = pd.DataFrame({"a": [1.0], "b": [float("nan")]})
    ca.append_csv(df, path)
    ca.append_csv(df, path)
    assert path.read_text().splitlines() == ["a,b", "1.0,nan", "1.0,nan"]
