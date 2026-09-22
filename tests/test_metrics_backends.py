"""Parity tests: every metrics backend must match the networkx-derived ground
truth fixtures in ``tests/metrics_ground_truth/`` within abs tolerance 1e-3.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg.core import DecisionPredicateGraph, DPGError
from metrics.backends import BACKEND_NAMES
from metrics.nodes import NodeMetrics
from tests.metrics_ground_truth.scenarios import SCENARIOS, build_scenario_graph

_FIXTURES_DIR = Path(__file__).resolve().parent / "metrics_ground_truth"

_FLOAT_COLS = (
    "Betweenness centrality",
    "Local reaching centrality",
    "Closeness centrality",
    "Harmonic centrality",
)
_DEGREE_COLS = ("Degree", "In degree nodes", "Out degree nodes")


def _load_ground_truth(name: str) -> dict:
    with (_FIXTURES_DIR / f"{name}.json").open() as f:
        return json.load(f)["nodes"]


@pytest.fixture(scope="session", params=list(SCENARIOS))
def scenario(request):
    """(name, graph, nodes_list, ground_truth) for one scenario, built once."""
    name = request.param
    graph, nodes_list = build_scenario_graph(name)
    return name, graph, nodes_list, _load_ground_truth(name)


def _assert_matches_ground_truth(df, ground_truth):
    assert set(df["Label"]) == set(ground_truth)
    for _, row in df.iterrows():
        expected = ground_truth[row["Label"]]
        for col in _DEGREE_COLS:
            assert row[col] == expected[col]
        for col in _FLOAT_COLS:
            assert row[col] == pytest.approx(expected[col], abs=1e-3)


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_backend_matches_ground_truth(backend_name, scenario):
    if backend_name == "graph_tool":
        pytest.importorskip("graph_tool")
    _, graph, nodes_list, ground_truth = scenario
    df = NodeMetrics.extract_node_metrics(graph, nodes_list, backend=backend_name)
    _assert_matches_ground_truth(df, ground_truth)


def test_get_backend_unknown_name_raises_value_error():
    from metrics.backends import get_backend

    with pytest.raises(ValueError):
        get_backend("bogus")


def test_bad_metrics_backend_config_raises_dpg_error():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=1).fit(
        iris.data, iris.target
    )
    dpg_config = {
        "dpg": {
            "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
            "graph_construction": {"mode": "execution_trace", "context_order": 1},
            "metrics": {"backend": "bogus"},
        }
    }
    with pytest.raises(DPGError):
        DecisionPredicateGraph(
            model=model,
            feature_names=iris.feature_names,
            target_names=np.unique(iris.target).astype(str).tolist(),
            dpg_config=dpg_config,
        )


def test_default_metrics_backend_is_igraph():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=1).fit(
        iris.data, iris.target
    )
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=iris.feature_names,
        target_names=np.unique(iris.target).astype(str).tolist(),
        dpg_config={
            "dpg": {
                "default": {"perc_var": 1e-9, "decimal_threshold": 6, "n_jobs": 1},
                "graph_construction": {"mode": "execution_trace", "context_order": 1},
            }
        },
    )
    assert dpg.get_metrics_backend() == "igraph"
