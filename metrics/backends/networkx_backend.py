"""NetworkX implementation of the centrality backend.

These calls are the reference definitions of the DPG node centralities; every
other backend must reproduce them to an absolute tolerance of 1e-3.
"""

from typing import Any, ClassVar

import networkx as nx
import numpy as np

from .base import GraphBackend, NodeCentralities, percolation_states


def _distance_graph(dpg_model: nx.DiGraph) -> tuple[nx.DiGraph, float]:
    """Copy the graph with a ``dist = total_weight / w`` attribute per edge.

    Heavier edges mean *closer* nodes, so the raw frequency weights are
    inverted before any shortest-path based metric is computed.
    """
    total_weight = float(
        sum(data.get("weight", 1.0) for _, _, data in dpg_model.edges(data=True))
    )
    distance_graph = nx.DiGraph()
    distance_graph.add_nodes_from(dpg_model.nodes())
    for u, v, data in dpg_model.edges(data=True):
        weight = data.get("weight", 1.0)
        distance_graph.add_edge(
            u, v, dist=total_weight / weight if weight > 0 else float("inf")
        )
    return distance_graph, total_weight


def _collective_influence(dpg_model: nx.DiGraph, ci_radius: int) -> dict[Any, float]:
    """Directed-out collective influence at radius ``ci_radius``."""
    excess = {node: max(dpg_model.out_degree(node) - 1, 0) for node in dpg_model}
    influence: dict[Any, float] = {}
    for node in dpg_model:
        hops = nx.single_source_shortest_path_length(
            dpg_model, node, cutoff=ci_radius
        )
        frontier = sum(excess[j] for j, d in hops.items() if d == ci_radius)
        influence[node] = float(excess[node] * frontier)
    return influence


def _clustering(dpg_model: nx.DiGraph) -> dict[Any, float]:
    """Directed, unweighted local clustering coefficient (Fagiolo)."""
    return {node: float(c) for node, c in nx.clustering(dpg_model).items()}


def _percolation(dpg_model: nx.DiGraph, states: dict[Any, float]) -> dict[Any, float]:
    """Percolation centrality with the given per-node ``states``.

    ``n <= 2`` and nodes with ``sum(states) - state == 0`` score ``0.0``, where
    ``nx.percolation_centrality`` would divide by zero.
    """
    n = len(dpg_model)
    if n <= 2:
        return dict.fromkeys(dpg_model.nodes(), 0.0)
    # numpy scalars turn the zero denominators into nan instead of raising;
    # those entries are overwritten below.
    np_states = {node: np.float64(value) for node, value in states.items()}
    total = sum(np_states.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = nx.percolation_centrality(dpg_model, states=np_states, weight="weight")
    return {
        node: float(raw[node]) if total - np_states[node] != 0 else 0.0
        for node in dpg_model.nodes()
    }


class NetworkXBackend(GraphBackend):
    """Computes the node centralities with NetworkX itself."""

    name: ClassVar[str] = "networkx"

    def node_centralities(
        self, dpg_model: nx.DiGraph, ci_radius: int = 2
    ) -> NodeCentralities:
        distance_graph, total_weight = _distance_graph(dpg_model)
        # NetworkX measures closeness/harmonic on incoming paths; reverse the
        # graph to obtain the outgoing variant DPG uses.
        reversed_graph = distance_graph.reverse(copy=True)

        betweenness: dict[Any, float] = nx.betweenness_centrality(
            dpg_model, normalized=True, weight="weight"
        )
        closeness: dict[Any, float] = nx.closeness_centrality(
            reversed_graph, distance="dist", wf_improved=True
        )
        harmonic: dict[Any, float] = nx.harmonic_centrality(
            reversed_graph, distance="dist"
        )

        if total_weight <= 0:
            from dpg.exceptions import DPGMetricError

            raise DPGMetricError.non_positive_lrc_weight()
        local_reaching: dict[Any, float] = {
            node: nx.local_reaching_centrality(dpg_model, node, weight="weight")
            for node in dpg_model.nodes()
        }

        return NodeCentralities(
            betweenness=betweenness,
            local_reaching=local_reaching,
            closeness=closeness,
            harmonic=harmonic,
            collective_influence=_collective_influence(dpg_model, ci_radius),
            clustering=_clustering(dpg_model),
            percolation=_percolation(dpg_model, percolation_states(dpg_model)),
        )
