"""Local clustering coefficient: every backend must reproduce ``nx.clustering``
(Fagiolo's directed, unweighted definition) on the NetworkX DiGraph.
"""

import networkx as nx
import pytest

from metrics.backends import BACKEND_NAMES, get_backend
from metrics.nodes import NodeMetrics

COLUMN = "Local clustering coefficient"


def _backend(name: str):
    if name == "graph_tool":
        pytest.importorskip("graph_tool")
    return get_backend(name)


def _with_weights(graph: nx.DiGraph) -> nx.DiGraph:
    # DPG edges always carry a frequency weight; the backends read it.
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def _hand_built_graph() -> tuple[nx.DiGraph, dict[str, float]]:
    """Graph plus hand-computed coefficients.

    With S = A + A^T, d = in + out degree, d_bi = reciprocated pairs:
    c_i = diag(S^3)_i / (2 * (d * (d - 1) - 2 * d_bi)).
    """
    graph = nx.DiGraph()
    # Directed triangle a -> b -> c -> a, with c also reciprocating with d.
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    graph.add_edges_from([("c", "d"), ("d", "c")])
    graph.add_edge("d", "e")  # e has degree 1
    graph.add_node("f")  # f has degree 0
    # Directed 4-cycle: neighbours of each node are not linked -> no triangle.
    graph.add_edges_from([("p", "q"), ("q", "r"), ("r", "s"), ("s", "p")])
    # Fully reciprocated triangle.
    for u, v in [("x", "y"), ("y", "z"), ("x", "z")]:
        graph.add_edges_from([(u, v), (v, u)])

    expected = {
        # a: neighbours b, c; d = 2, d_bi = 0. Ordered pairs (b, c), (c, b)
        # each close through b -> c: diag(S^3) = 2, c = 2 / (2 * 2) = 0.5.
        "a": 0.5,
        # b: neighbours a, c; same structure as a.
        "b": 0.5,
        # c: neighbours a, b (one edge each) and d (reciprocal, S = 2);
        # d = 1 + 1 + 2 = 4, d_bi = 1. Only (a, b), (b, a) close:
        # diag(S^3) = 2, c = 2 / (2 * (4 * 3 - 2)) = 2 / 20 = 0.1.
        "c": 0.1,
        # d: neighbours c (reciprocal) and e; c and e are not linked -> 0.
        "d": 0.0,
        "e": 0.0,  # degree 1: denominator is 0
        "f": 0.0,  # degree 0: denominator is 0
        # 4-cycle: d = 2, no closing edge between the two neighbours -> 0.
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "s": 0.0,
        # Fully reciprocated triangle: d = 4, d_bi = 2, every S entry is 2;
        # diag(S^3) = 2 ordered pairs * 2 * 2 * 2 = 16,
        # c = 16 / (2 * (4 * 3 - 4)) = 16 / 16 = 1.
        "x": 1.0,
        "y": 1.0,
        "z": 1.0,
    }
    return _with_weights(graph), expected


def _random_digraphs() -> list[nx.DiGraph]:
    graphs = [
        _with_weights(nx.gnp_random_graph(n, p, seed=seed, directed=True))
        for n, p, seed in [(12, 0.2, 0), (25, 0.3, 1), (30, 0.5, 2), (40, 0.1, 3)]
    ]
    # gnp with p >= 0.3 already yields reciprocal pairs; make sure of it.
    assert all(any(g.has_edge(v, u) for u, v in g.edges) for g in graphs[1:3])
    return graphs


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_clustering_hand_computed(backend_name):
    graph, expected = _hand_built_graph()
    clustering = _backend(backend_name).node_centralities(graph).clustering
    assert set(clustering) == set(expected)
    for node, value in expected.items():
        assert clustering[node] == pytest.approx(value, abs=1e-9), node


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("index", range(4))
def test_clustering_matches_networkx(backend_name, index):
    graph = _random_digraphs()[index]
    reference = nx.clustering(graph)
    clustering = _backend(backend_name).node_centralities(graph).clustering
    assert set(clustering) == set(reference)
    for node, value in reference.items():
        assert clustering[node] == pytest.approx(value, abs=1e-9), node


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_clustering_column_in_node_metrics(backend_name):
    _backend(backend_name)
    graph, expected = _hand_built_graph()
    nodes_list = [[node, f"label_{node}"] for node in graph.nodes]
    df = NodeMetrics.extract_node_metrics(graph, nodes_list, backend=backend_name)
    assert COLUMN in df.columns
    by_label = dict(zip(df["Label"], df[COLUMN]))
    for node, value in expected.items():
        assert by_label[f"label_{node}"] == pytest.approx(value, abs=1e-9), node
