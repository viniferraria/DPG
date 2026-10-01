"""Backend interface for node centrality computation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import networkx as nx


@dataclass(frozen=True)
class NodeCentralities:
    """Centrality scores keyed by the NetworkX node id."""

    betweenness: dict[Any, float]
    local_reaching: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]
    collective_influence: dict[Any, float]
    clustering: dict[Any, float]
    percolation: dict[Any, float]


def percolation_states(dpg_model: nx.DiGraph) -> dict[Any, float]:
    """Per-node percolation state: node flow divided by the largest node flow.

    A node's flow is the sum of its incoming edge weights, or of its outgoing
    edge weights when it has no incoming edges (a tree root). All states are
    ``0.0`` when every flow is zero.
    """
    flow: dict[Any, float] = {}
    for node in dpg_model.nodes():
        edges = dpg_model.in_edges(node, data="weight", default=1.0)
        if dpg_model.in_degree(node) == 0:
            edges = dpg_model.out_edges(node, data="weight", default=1.0)
        flow[node] = float(sum(w for _, _, w in edges))
    max_flow = max(flow.values(), default=0.0)
    if max_flow <= 0:
        return dict.fromkeys(flow, 0.0)
    return {node: value / max_flow for node, value in flow.items()}


class GraphBackend(ABC):
    """A graph library that computes the seven DPG node centralities.

    See ``okf/docs/modules/metrics-backends.md`` for the metric contract every
    backend must satisfy.
    """

    name: ClassVar[str]

    @abstractmethod
    def node_centralities(
        self, dpg_model: nx.DiGraph, ci_radius: int = 2
    ) -> NodeCentralities:
        """Compute every node centrality; ``ci_radius`` is collective influence's ℓ."""
