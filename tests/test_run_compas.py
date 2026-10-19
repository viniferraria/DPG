"""Minimal tests for the COMPAS experiment runner (``experiments/compas``).

``experiments/compas`` is not a package, so ``run_compas`` is loaded by path.
Nothing here touches the network or the (uncommitted) COMPAS CSV.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from experiments import causal_accuracy as shared
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

    assert set(shared.METRICS) <= set(df.columns)


def test_one_hot_mapping_passed_as_feature_name(m):
    node_metrics = pd.DataFrame(
        {
            "Label": ["age <= 30.5", "race_Caucasian > 0.5", "Class 1"],
            "Local reaching centrality": [0.9, 0.8, 9.0],
        }
    )
    assert shared.extract_top_k_features(
        node_metrics, 2, feature_name=m.base_feature_name
    ) == ["age", "race"]


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
