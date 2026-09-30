"""Generate networkx-only ground-truth fixtures for the DPG node centralities
(betweenness, local reaching, closeness, harmonic, collective influence, local
clustering, percolation) plus degree.

These fixtures are the reference every ``metrics.backends`` implementation is
checked against, so they are computed directly from ``networkx`` (never via
``NodeMetrics`` or any backend). Run with:

    uv run python -m tests.metrics_ground_truth.generate
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx

from tests.metrics_ground_truth.scenarios import SCENARIOS, build_scenario_graph

_OUTPUT_DIR = Path(__file__).resolve().parent


def _distance_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Same graph with an added ``dist = W / w`` edge attribute."""
    total_weight = sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True))
    dist_graph = nx.DiGraph()
    dist_graph.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1.0)
        dist_graph.add_edge(u, v, dist=total_weight / w)
    return dist_graph


CI_RADIUS = 2


def _collective_influence(graph: nx.DiGraph, radius: int) -> dict[Any, float]:
    """(max(k_out(i)-1, 0)) * sum of max(k_out(j)-1, 0) over j exactly
    ``radius`` out-hops from i."""
    excess = {node: max(graph.out_degree(node) - 1, 0) for node in graph.nodes()}
    ci: dict[Any, float] = {}
    for node in graph.nodes():
        hops = nx.single_source_shortest_path_length(graph, node, cutoff=radius)
        frontier = sum(excess[j] for j, d in hops.items() if d == radius)
        ci[node] = float(excess[node] * frontier)
    return ci


def _percolation_states(graph: nx.DiGraph) -> dict[Any, float]:
    """In-weight flow per node (out-weight for roots), divided by the max."""
    flow = {}
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            flow[node] = graph.out_degree(node, weight="weight")
        else:
            flow[node] = graph.in_degree(node, weight="weight")
    max_flow = max(flow.values())
    return {node: value / max_flow for node, value in flow.items()}


def compute_ground_truth(
    graph: nx.DiGraph, nodes_list: list[list[str]]
) -> dict[str, Any]:
    labels = [label for _, label in nodes_list]
    assert len(labels) == len(set(labels)), "node labels must be unique per graph"
    for node_id, _ in nodes_list:
        assert node_id in graph.nodes, f"node {node_id!r} missing from graph"

    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="weight")
    local_reaching = {
        node: nx.local_reaching_centrality(
            graph, node, weight="weight", normalized=True
        )
        for node in graph.nodes()
    }

    dist_graph = _distance_graph(graph)
    reversed_dist_graph = dist_graph.reverse(copy=True)
    closeness = nx.closeness_centrality(
        reversed_dist_graph, distance="dist", wf_improved=True
    )
    harmonic = nx.harmonic_centrality(reversed_dist_graph, distance="dist")

    collective_influence = _collective_influence(graph, CI_RADIUS)
    clustering = nx.clustering(graph)
    percolation = nx.percolation_centrality(
        graph, states=_percolation_states(graph), weight="weight"
    )

    nodes: dict[str, Any] = {}
    for node_id, label in nodes_list:
        nodes[label] = {
            "Degree": graph.in_degree(node_id) + graph.out_degree(node_id),
            "In degree nodes": graph.in_degree(node_id),
            "Out degree nodes": graph.out_degree(node_id),
            "Betweenness centrality": betweenness[node_id],
            "Local reaching centrality": local_reaching[node_id],
            "Closeness centrality": closeness[node_id],
            "Harmonic centrality": harmonic[node_id],
            "Collective influence": collective_influence[node_id],
            "Local clustering coefficient": clustering[node_id],
            "Percolation centrality": percolation[node_id],
        }

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "ci_radius": CI_RADIUS,
        "nodes": nodes,
    }


def main() -> None:
    for name in SCENARIOS:
        graph, nodes_list = build_scenario_graph(name)
        fixture = compute_ground_truth(graph, nodes_list)
        fixture = {"scenario": name, **fixture}
        out_path = _OUTPUT_DIR / f"{name}.json"
        with out_path.open("w") as f:
            json.dump(fixture, f, indent=2, sort_keys=True)
        print(
            f"{name}: n_nodes={fixture['n_nodes']} n_edges={fixture['n_edges']} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
