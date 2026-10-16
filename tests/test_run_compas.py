"""Minimal tests for the COMPAS experiment runner (``experiments/compas``).

``experiments/compas`` is not a package, so ``run_compas`` is loaded by path.
Nothing here touches the network or the (uncommitted) COMPAS CSV.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as reference
from metrics.nodes import NodeMetrics

RUN_COMPAS = (
    Path(__file__).resolve().parent.parent / "experiments/compas/run_compas.py"
)


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("run_compas_under_test", RUN_COMPAS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_ranked_metric_is_a_node_metrics_column(m):
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [("n0", "n1", 4), ("n0", "n2", 2), ("n1", "n3", 3), ("n2", "n3", 1)]
    )
    labels = ["age <= 30.5", "priors_count <= 2.5", "priors_count > 2.5", "Class 0"]
    nodes_list = [[f"n{i}", label] for i, label in enumerate(labels)]

    df = NodeMetrics.extract_node_metrics(G, nodes_list)

    assert set(m.METRICS) <= set(df.columns)


def test_extract_top_k_features_matches_reference_apart_from_one_hot_mapping(m):
    node_metrics = pd.DataFrame(
        {
            "Label": [
                "age <= 30.5",
                "race_African-American <= 0.5",
                "priors_count <= 2.5",
                "race_Caucasian > 0.5",
                "Class 1",
            ],
            "Local reaching centrality": [0.9, 0.8, 0.7, 0.6, 9.0],
        }
    )
    metric = "Local reaching centrality"

    ours = m.extract_top_k_features(node_metrics, 4, metric)

    # Reference output, with the named deviation applied afterwards: dummies -> parent.
    expected = [
        m.base_feature_name(label)
        for label in reference.extract_top_k_features(node_metrics, 4, metric)
    ]
    assert ours == expected == ["age", "race", "priors_count", "race"]
    # Without dummies the two functions agree exactly.
    plain = node_metrics[~node_metrics["Label"].str.startswith("race")]
    assert m.extract_top_k_features(plain, 2, metric) == reference.extract_top_k_features(
        plain, 2, metric
    )


def test_causal_accuracy_csv_columns_match_reference(m, tmp_path):
    written = tmp_path / "acc.csv"
    m.write_causal_accuracy_to_csv(
        "run", 3, ["age"], ["age"], 0, "Closeness centrality", {"age"}, 1.0, 1.0, written
    )

    assert m.TOP_K == reference.TOP_K and m.METRICS == reference.METRICS
    assert written.read_text().splitlines()[0].split(",") == [
        "experiment", "split", "k", "ground_truth", "explanation_top_features",
        "metric", "intersection", "precision", "recall",
    ]


def test_calculate_causal_accuracy_against_reference(m):
    inter, precision, recall = m.calculate_causal_accuracy(
        m.REFERENCE_FEATURES, ["priors_count", "race"]
    )
    assert inter == {"priors_count"}
    assert (precision, recall) == (0.5, 0.5)


def test_base_feature_name_keeps_underscored_numeric_features(m):
    assert m.base_feature_name("priors_count") == "priors_count"
    assert m.base_feature_name("c_charge_degree_F") == "c_charge_degree"
    assert m.base_feature_name("sex_Male") == "sex"


def test_load_dataset_applies_propublica_filters(m, tmp_path):
    base = {
        "days_b_screening_arrest": 0,
        "is_recid": 0,
        "c_charge_degree": "F",
        "v_score_text": "Low",
        "race": "Caucasian",
        "age": 30,
        "priors_count": 1,
        "sex": "Male",
    }
    rows = [
        base,
        {**base, "v_score_text": "High", "race": "African-American"},
        {**base, "days_b_screening_arrest": 31},
        {**base, "is_recid": -1},
        {**base, "c_charge_degree": "O"},
        {**base, "v_score_text": "N/A"},
        {**base, "race": "Hispanic"},
    ]
    path = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    X, y = m.load_dataset(path)

    assert len(X) == 2
    assert y.tolist() == [0, 1]
    assert X.columns.tolist() == m.FEATURES
