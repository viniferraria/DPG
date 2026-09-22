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


class GraphBackend(ABC):
    """A graph library that computes the four DPG node centralities.

    See ``okf/docs/modules/metrics-backends.md`` for the metric contract every
    backend must satisfy.
    """

    name: ClassVar[str]

    @abstractmethod
    def node_centralities(self, dpg_model: nx.DiGraph) -> NodeCentralities:
        """Compute betweenness, local reaching, closeness and harmonic centrality."""
