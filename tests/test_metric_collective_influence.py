"""Collective influence (CI) across backends.

CI_l(i) = excess(i) * sum(excess(j)) over j at unweighted directed hop
distance exactly l from i, with excess(v) = max(out_degree(v) - 1, 0).
"""

import networkx as nx
import pytest

from metrics.backends import BACKEND_NAMES, get_backend
from metrics.nodes import NodeMetrics
from tests.metrics_ground_truth.generate import _collective_influence as reference_ci
from tests.metrics_ground_truth.scenarios import SCENARIOS, build_scenario_graph

# Edge weights are deliberately uneven: CI must ignore them.
_EDGES = [
    ("a", "b", 5), ("a", "c", 1), ("a", "d", 9),
    ("b", "e", 2), ("b", "f", 7),
    ("c", "e", 3),
    ("d", "g", 4),
    ("e", "g", 1), ("e", "h", 6),
    ("f", "g", 8),
    ("h", "i", 2), ("h", "j", 3),
]  # fmt: skip

# Hand-computed with excess = max(out_degree - 1, 0):
#   out-degree: a3 b2 c1 d1 e2 f1 g0 h2 i0 j0
#   excess:     a2 b1 c0 d0 e1 f0 g0 h1 i0 j0
#   hop distances: from a: b,c,d=1; e,f,g=2 (g via d; e->g would be 3); h=3; i,j=4
#                  from b: e,f=1; g,h=2 (g via f); i,j=3
#                  from e: g,h=1; i,j=2
#   l=1: a=2*(b1+c0+d0)=2  b=1*(e1+f0)=1  e=1*(g0+h1)=1  h=1*(i0+j0)=0
#   l=2: a=2*(e1+f0+g0)=2  b=1*(g0+h1)=1  e=1*(i0+j0)=0  h=0
#   l=3: a=2*(h1)=2        b=1*(i0+j0)=0  e=0             h=0
_EXPECTED = {
    1: {"a": 2.0, "b": 1.0, "e": 1.0},
    2: {"a": 2.0, "b": 1.0},
    3: {"a": 2.0},
}


@pytest.fixture(scope="module")
def hand_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(_EDGES)
    return graph


def _backend(name: str):
    if name == "graph_tool":
        pytest.importorskip("graph_tool")
    return get_backend(name)


@pytest.mark.parametrize("radius", sorted(_EXPECTED))
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_collective_influence_hand_computed(backend_name, radius, hand_graph):
    ci = _backend(backend_name).node_centralities(
        hand_graph, ci_radius=radius
    ).collective_influence
    expected = {node: _EXPECTED[radius].get(node, 0.0) for node in hand_graph}
    assert set(ci) == set(hand_graph.nodes())
    assert ci == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("radius", sorted(_EXPECTED))
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_extract_node_metrics_threads_ci_radius(backend_name, radius, hand_graph):
    _backend(backend_name)
    nodes_list = [[node, f"label_{node}"] for node in hand_graph.nodes()]
    df = NodeMetrics.extract_node_metrics(
        hand_graph, nodes_list, backend=backend_name, ci_radius=radius
    )
    by_label = df.set_index("Label")["Collective influence"]
    for node in hand_graph:
        expected = _EXPECTED[radius].get(node, 0.0)
        assert by_label[f"label_{node}"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_extract_node_metrics_rejects_zero_radius(backend_name, hand_graph):
    _backend(backend_name)
    nodes_list = [[node, f"label_{node}"] for node in hand_graph.nodes()]
    with pytest.raises(ValueError):
        NodeMetrics.extract_node_metrics(
            hand_graph, nodes_list, backend=backend_name, ci_radius=0
        )


@pytest.fixture(scope="module", params=list(SCENARIOS))
def scenario_graph(request) -> nx.DiGraph:
    graph, _ = build_scenario_graph(request.param)
    return graph


@pytest.mark.parametrize("radius", [1, 3])
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_collective_influence_matches_reference_on_dpg_graphs(
    backend_name, radius, scenario_graph
):
    ci = _backend(backend_name).node_centralities(
        scenario_graph, ci_radius=radius
    ).collective_influence
    assert ci == pytest.approx(reference_ci(scenario_graph, radius), abs=1e-3)
