"""Minimal tests for the MONK's experiment runner (``experiments/monks``).

``experiments/monks`` is not a package and importing ``run_monk`` opens a log
file in the CWD, so the module is loaded from its path inside a fixture that
first moves the CWD to a temp dir.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from metrics.nodes import NodeMetrics

RUN_MONK = Path(__file__).resolve().parent.parent / "experiments/monks/run_monk.py"


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("run_monk_cwd"))
    try:
        spec = importlib.util.spec_from_file_location("run_monk_under_test", RUN_MONK)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        mp.undo()


def test_every_ranked_metric_is_a_node_metrics_column(m):
    edges = [("n0", "n1", 4), ("n0", "n2", 2), ("n1", "n3", 3), ("n2", "n3", 1)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    labels = ["a1_0 <= 0.5", "a2_1 <= 0.5", "a2_1 > 0.5", "Class 0"]
    nodes_list = [[f"n{i}", label] for i, label in enumerate(labels)]

    df = NodeMetrics.extract_node_metrics(G, nodes_list)

    assert set(m.METRICS) <= set(df.columns)


def test_extract_top_k_features_maps_one_hot_labels_to_attributes(m):
    node_metrics = pd.DataFrame(
        {
            "Label": [
                "a3_1 <= 0.5",
                "a5_2 > 0.5",
                "a5_1 <= 0.5",
                "a1_3 <= 0.5",
                "Class 0",
            ],
            "Percolation centrality": [0.1, 0.9, 0.5, 0.3, 2.0],
        }
    )

    top = m.extract_top_k_features(
        node_metrics, top_k=3, metric="Percolation centrality"
    )

    # Class sink excluded; a5 deduplicated to its best-ranked occurrence.
    assert top == ["a5", "a1", "a3"]


def test_new_metric_columns_snake_case_for_node_csv(m):
    assert m.to_snake_case("Collective influence") == "collective_influence"
    assert (
        m.to_snake_case("Local clustering coefficient")
        == "local_clustering_coefficient"
    )
    assert m.to_snake_case("Percolation centrality") == "percolation_centrality"
