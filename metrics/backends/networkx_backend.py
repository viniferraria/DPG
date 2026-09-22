"""NetworkX implementation of the centrality backend.

These calls are the reference definitions of the four DPG centralities; every
other backend must reproduce them to an absolute tolerance of 1e-3.
"""

from typing import Any, ClassVar

import networkx as nx

from .base import GraphBackend, NodeCentralities


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


class NetworkXBackend(GraphBackend):
    """Computes the four node centralities with NetworkX itself."""

    name: ClassVar[str] = "networkx"

    def node_centralities(self, dpg_model: nx.DiGraph) -> NodeCentralities:
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
        )
