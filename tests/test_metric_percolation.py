"""Percolation centrality: every backend must reproduce
``nx.percolation_centrality(G, states=percolation_states(G), weight="weight")``.
"""

import random
from typing import Any

import networkx as nx
import numpy as np
import pytest

from metrics.backends import BACKEND_NAMES, get_backend
from metrics.backends.base import percolation_states
from metrics.nodes import NodeMetrics

_TOL = 1e-9


def _require(name: str) -> None:
    if name == "graph_tool":
        pytest.importorskip("graph_tool")


def _percolation(name: str, graph: nx.DiGraph) -> dict[Any, float]:
    return get_backend(name).node_centralities(graph).percolation


def _helper(name: str, graph: nx.DiGraph, states: dict[Any, float]) -> dict[Any, float]:
    """Call one backend's percolation helper directly with explicit ``states``."""
    if name == "networkx":
        from metrics.backends.networkx_backend import _percolation as helper

        return helper(graph, states)
    if name == "igraph":
        from metrics.backends.igraph_backend import (
            _nx_to_igraph,
            calc_percolation_centrality,
        )

        ig_graph, node_ids = _nx_to_igraph(graph)
        return calc_percolation_centrality(ig_graph, node_ids, states)
    import graph_tool as gt

    from metrics.backends.graph_tool_backend import _percolation as helper

    node_ids = list(graph.nodes())
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    gt_graph = gt.Graph(directed=True)
    gt_graph.add_vertex(len(node_ids))
    weight_prop = gt_graph.new_edge_property("double")
    gt_graph.add_edge_list(
        [
            (index[u], index[v], float(data["weight"]))
            for u, v, data in graph.edges(data=True)
        ],
        eprops=[weight_prop],
    )
    return helper(gt_graph, weight_prop, node_ids, states)


def _random_digraph(seed: int, n: int = 25, p: float = 0.12) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(1, 5)
    return graph


def _hand_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [
            ("r", "a", 2),
            ("r", "b", 2),
            ("a", "c", 1),
            ("b", "c", 1),
            ("c", "d", 3),
            ("d", "e", 1),
        ]
    )
    return graph


def test_percolation_states_root_uses_out_flow_and_max_is_one():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("r", "a", 3), ("r", "b", 1), ("a", "b", 2)])
    # flows: r (root) = out 3 + 1 = 4; a = in 3; b = in 1 + 2 = 3; max = 4
    assert percolation_states(graph) == {"r": 1.0, "a": 0.75, "b": 0.75}


def test_percolation_states_all_zero_weights_give_zero():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("r", "a", 0), ("a", "b", 0)])
    assert percolation_states(graph) == {"r": 0.0, "a": 0.0, "b": 0.0}


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_percolation_hand_computed(backend_name):
    _require(backend_name)
    graph = _hand_graph()
    # Flows: r (root) = out 2 + 2 = 4, a = 2, b = 2, c = 1 + 1 = 2, d = 3, e = 1.
    # States (flow / 4): r=1, a=.5, b=.5, c=.5, d=.75, e=.25; X = 3.5; n - 2 = 4.
    states = percolation_states(graph)
    assert states == {"r": 1.0, "a": 0.5, "b": 0.5, "c": 0.5, "d": 0.75, "e": 0.25}
    # Shortest paths from r: a=2, b=2, c=3 (sigma=2, via a or via b), d=6, e=7.
    # delta_r: a = 1/2 * (t=c, d, e) = 1.5, b = 1.5, c = 2, d = 1, e = 0.
    # delta_a = delta_b: c = 2 (t=d, e), d = 1.  delta_c: d = 1.  delta_d = delta_e = 0.
    # sum_s x_s * delta_s(v):
    #   a = 1 * 1.5 = 1.5, b = 1.5
    #   c = 1*2 + .5*2 + .5*2 = 4
    #   d = 1*1 + .5*1 + .5*1 + .5*1 = 2.5
    #   r = e = 0
    # PC(v) = sum / (X - x_v) / (n - 2):
    #   a = 1.5 / 3 / 4 = 0.125, b = 0.125, c = 4 / 3 / 4 = 1/3,
    #   d = 2.5 / 2.75 / 4, e = 0, r = 0
    expected = {
        "r": 0.0,
        "a": 0.125,
        "b": 0.125,
        "c": 1 / 3,
        "d": 2.5 / 2.75 / 4,
        "e": 0.0,
    }
    reference = nx.percolation_centrality(graph, states=states, weight="weight")
    result = _percolation(backend_name, graph)
    for node, value in expected.items():
        assert reference[node] == pytest.approx(value, abs=_TOL)
        assert result[node] == pytest.approx(value, abs=_TOL)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_percolation_matches_networkx_on_random_digraphs(backend_name, seed):
    _require(backend_name)
    graph = _random_digraph(seed)
    reference = nx.percolation_centrality(
        graph, states=percolation_states(graph), weight="weight"
    )
    result = _percolation(backend_name, graph)
    assert set(result) == set(reference)
    for node, value in reference.items():
        assert result[node] == pytest.approx(value, abs=_TOL)


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_uniform_states_equal_normalized_betweenness(backend_name):
    _require(backend_name)
    graph = _random_digraph(7)
    uniform = dict.fromkeys(graph.nodes(), 1.0)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="weight")
    result = _helper(backend_name, graph, uniform)
    for node, value in betweenness.items():
        assert result[node] == pytest.approx(value, abs=_TOL)


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_two_node_graph_is_all_zero(backend_name):
    _require(backend_name)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 3)])
    assert _percolation(backend_name, graph) == {"a": 0.0, "b": 0.0}


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_zero_denominator_node_is_zero(backend_name):
    _require(backend_name)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 1), ("b", "c", 1)])
    # Only "a" is percolated, so X - x_a == 0 (nx would divide by zero) -> 0.0.
    # delta_a(b) = 1 (t=c), so PC(b) = 1 * 1 / (1 - 0) / (3 - 2) = 1; PC(c) = 0.
    result = _helper(backend_name, graph, {"a": 1.0, "b": 0.0, "c": 0.0})
    assert result["a"] == 0.0
    assert result["b"] == pytest.approx(1.0, abs=_TOL)
    assert result["c"] == pytest.approx(0.0, abs=_TOL)
    # No percolated node at all: X == 0, every denominator is zero.
    result = _helper(backend_name, graph, dict.fromkeys(graph.nodes(), 0.0))
    assert result == dict.fromkeys(graph.nodes(), 0.0)


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_percolation_column_in_node_metrics(backend_name):
    _require(backend_name)
    graph = _hand_graph()
    nodes_list = [[node, f"label {node}"] for node in graph.nodes()]
    df = NodeMetrics.extract_node_metrics(graph, nodes_list, backend=backend_name)
    assert "Percolation centrality" in df.columns
    by_label = dict(zip(df["Label"], df["Percolation centrality"], strict=True))
    reference = nx.percolation_centrality(
        graph, states=percolation_states(graph), weight="weight"
    )
    for node, value in reference.items():
        assert by_label[f"label {node}"] == pytest.approx(value, abs=_TOL)
    assert np.isfinite(df["Percolation centrality"]).all()
