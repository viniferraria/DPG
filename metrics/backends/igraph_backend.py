"""igraph implementation of the centrality backend.

The conversion to igraph exists because betweenness and local reaching
centrality are the two expensive metrics; igraph computes them at C level.
"""

import warnings
from typing import Any, ClassVar

import igraph as ig
import networkx as nx
import numpy as np
import scipy.sparse as sp

from ..nodes import log_timer
from .base import GraphBackend, NodeCentralities, percolation_states


@log_timer
def _nx_to_igraph(nx_graph: nx.DiGraph) -> tuple[ig.Graph, list[str]]:
    """Convert a NetworkX DiGraph to igraph, preserving node ID mapping.

    Returns:
        Tuple of (igraph.Graph, node_ids) where node_ids[i] is the original
        NetworkX node ID for igraph vertex index *i*.
    """
    node_ids = list(nx_graph.nodes())
    node_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}

    edges = []
    weights = []
    for u, v, data in nx_graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))

    graph = ig.Graph(n=len(node_ids), edges=edges, directed=True)
    graph.es["weight"] = weights

    # Invert weights to distances for igraph shortest path calculations.
    total_weight = sum(weights)
    graph.es["distance"] = [
        total_weight / w if w > 0 else float("inf") for w in graph.es["weight"]
    ]
    return graph, node_ids


@log_timer
def calc_betweenness_centrality(ig_graph: ig.Graph) -> dict[int, float]:
    """Compute betweenness centrality for an igraph graph."""
    n = ig_graph.vcount()
    # Betweenness centrality (igraph returns unnormalised values).
    ig_bc = ig_graph.betweenness(directed=True, normalized=False, weights="weight")
    # Normalise to match NetworkX convention: bc / ((n-1)*(n-2))
    norm = (n - 1) * (n - 2) if n > 2 else 1.0
    betweenness_centrality = {i: ig_bc[i] / norm for i in range(n)}
    # betweenness_centrality = ig_graph.betweenness(directed=True, normalized=True, weights="weight")
    return betweenness_centrality


@log_timer
def calc_local_reaching_centrality(
    ig_graph: ig.Graph, node_ids: list[str]
) -> dict[str, float]:
    """Compute local reaching centrality for an igraph graph."""
    # Local reaching centrality (weighted), matching NetworkX's algorithm:
    # 1. Invert weights to get distances: distance = total_weight / w
    # 2. Find shortest paths using these distances (igraph C-level)
    # 3. For each reachable node, compute average original edge weight along path
    # 4. Sum averages, normalise by (total_weight / num_edges), divide by (n-1)
    num_vertex = ig_graph.vcount()
    total_weight = sum(ig_graph.es["weight"])
    num_edges = ig_graph.ecount()
    if total_weight <= 0:
        from dpg.exceptions import DPGMetricError

        raise DPGMetricError.non_positive_lrc_weight()

    # Build edge weight lookup: (source, target) → original weight
    edge_weight_lookup = {}
    for e in ig_graph.es:
        edge_weight_lookup[(e.source, e.target)] = e["weight"]

    lrc_norm = total_weight / num_edges if num_edges > 0 else 1.0

    local_reaching_centrality = {}
    for i in range(num_vertex):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Couldn't reach some", category=RuntimeWarning
            )
            paths = ig_graph.get_shortest_paths(
                i, mode="out", weights="distance", output="vpath"
            )
        sum_avg_weight = 0.0
        for path in paths:
            path_length = len(path) - 1
            if path_length <= 0:
                continue
            path_weight_sum = sum(
                edge_weight_lookup.get((path[k], path[k + 1]), 1.0)
                for k in range(path_length)
            )
            sum_avg_weight += path_weight_sum / path_length
        lrc = (sum_avg_weight / lrc_norm) / (num_vertex - 1) if num_vertex > 1 else 0.0
        local_reaching_centrality[node_ids[i]] = lrc

    return local_reaching_centrality


@log_timer
def calc_closeness_centrality(
    ig_graph: ig.Graph, node_ids: list[str]
) -> dict[str, float]:
    """Compute closeness centrality for an igraph graph."""
    n = ig_graph.vcount()

    total_weight = sum(ig_graph.es["weight"])
    if total_weight <= 0:
        from dpg.exceptions import DPGMetricError

        raise DPGMetricError.non_positive_closeness_weight()

    closeness_centrality = {}
    for i in range(n):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Couldn't reach some", category=RuntimeWarning
            )
            path_lengths = ig_graph.distances(i, mode="out", weights="distance")[0]
        # Keep only reachable nodes; unreachable ones contribute inf and must be
        # excluded from the distance sum (otherwise totsp becomes inf -> 0).
        finite = [d for d in path_lengths if d < float("inf")]
        reachable_nodes = len(finite)  # includes the node itself (distance 0)
        totsp = sum(finite)
        # Wasserman-Faust closeness over OTHER reachable nodes (reachable_nodes - 1).
        if reachable_nodes > 1 and totsp > 0:
            closeness_centrality[node_ids[i]] = ((reachable_nodes - 1) / totsp) * (
                (reachable_nodes - 1) / (n - 1)
            )
        else:
            closeness_centrality[node_ids[i]] = 0.0

    return closeness_centrality


@log_timer
def calc_harmonic_centrality(
    ig_graph: ig.Graph, node_ids: list[str]
) -> dict[str, float]:
    """Compute harmonic centrality for an igraph graph."""
    n = ig_graph.vcount()
    total_weight = sum(ig_graph.es["weight"])
    if total_weight <= 0:
        from dpg.exceptions import DPGMetricError

        raise DPGMetricError.non_positive_harmonic_weight()

    harmonic_centrality: dict[str, float] = {}
    for i in range(n):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Couldn't reach some", category=RuntimeWarning
            )
            path_lengths = ig_graph.distances(i, mode="out", weights="distance")[0]
        harmonic_sum = sum(1 / d for d in path_lengths if d < float("inf") and d > 0)
        harmonic_centrality[node_ids[i]] = harmonic_sum

    return harmonic_centrality


@log_timer
def calc_collective_influence(
    ig_graph: ig.Graph, node_ids: list[str], ci_radius: int
) -> dict[str, float]:
    """Compute directed-out collective influence for an igraph graph."""
    excess = [max(degree - 1, 0) for degree in ig_graph.outdegree()]
    # Unweighted out-neighbourhoods restricted to distance exactly ci_radius.
    frontiers = ig_graph.neighborhood(order=ci_radius, mode="out", mindist=ci_radius)
    return {
        node_ids[i]: float(excess[i] * sum(excess[j] for j in frontier))
        for i, frontier in enumerate(frontiers)
    }


@log_timer
def calc_clustering_coefficient(
    ig_graph: ig.Graph, node_ids: list[str]
) -> dict[str, float]:
    """Compute the directed, unweighted local clustering coefficient.

    igraph only has an undirected version, so this evaluates Fagiolo's
    directed definition (the one ``nx.clustering`` uses) on the sparse
    adjacency: with S = A + A^T, c_i = diag(S^3)_i / (2 (d(d-1) - 2 d_bi)),
    d being in + out degree and d_bi the number of reciprocated pairs.
    """
    n = ig_graph.vcount()
    adj = sp.csr_matrix(ig_graph.get_adjacency_sparse(), dtype=float)
    adj.setdiag(0)  # self-loops are ignored, as in NetworkX
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    sym = adj + adj.T
    triangles = np.asarray((sym @ sym).multiply(sym.T).sum(axis=1)).ravel()
    degree = np.asarray(sym.sum(axis=1)).ravel()
    reciprocal = np.asarray((adj @ adj).multiply(sp.identity(n)).sum(axis=1)).ravel()
    denominator = 2 * (degree * (degree - 1) - 2 * reciprocal)
    coefficient = np.divide(
        triangles, denominator, out=np.zeros(n), where=denominator > 0
    )
    return {node_ids[i]: float(coefficient[i]) for i in range(n)}


@log_timer
def calc_percolation_centrality(
    ig_graph: ig.Graph, node_ids: list[str], states: dict[str, float]
) -> dict[str, float]:
    """Compute percolation centrality for an igraph graph."""
    n = ig_graph.vcount()
    if n <= 2:
        return dict.fromkeys(node_ids, 0.0)
    x = [states[node_id] for node_id in node_ids]
    total = sum(x)
    # sum_s x_s * delta_s(v): igraph's subset betweenness with only ``sources``
    # given is sum_{s in sources} delta_s(v) on the raw weights, so sources that
    # share a state value are handled in one call and scaled by that value.
    sources_by_state: dict[float, list[int]] = {}
    for i, value in enumerate(x):
        if value != 0:
            sources_by_state.setdefault(value, []).append(i)
    weighted_dependency = np.zeros(n)
    for value, sources in sources_by_state.items():
        weighted_dependency += value * np.asarray(
            ig_graph.betweenness(directed=True, weights="weight", sources=sources)
        )
    return {
        node_ids[i]: float(weighted_dependency[i] / (total - x[i]) / (n - 2))
        if total - x[i] != 0
        else 0.0
        for i in range(n)
    }


class IGraphBackend(GraphBackend):
    """Computes the node centralities through igraph."""

    name: ClassVar[str] = "igraph"

    def node_centralities(
        self, dpg_model: nx.DiGraph, ci_radius: int = 2
    ) -> NodeCentralities:
        ig_graph, node_ids = _nx_to_igraph(dpg_model)
        # calc_betweenness_centrality is keyed by igraph vertex index; map it
        # back to the NetworkX node ids the dataclass promises.
        betweenness_by_index = calc_betweenness_centrality(ig_graph)
        betweenness: dict[Any, float] = {
            node_ids[index]: value for index, value in betweenness_by_index.items()
        }
        return NodeCentralities(
            betweenness=betweenness,
            local_reaching=dict(calc_local_reaching_centrality(ig_graph, node_ids)),
            closeness=dict(calc_closeness_centrality(ig_graph, node_ids)),
            harmonic=dict(calc_harmonic_centrality(ig_graph, node_ids)),
            collective_influence=calc_collective_influence(
                ig_graph, node_ids, ci_radius
            ),
            clustering=calc_clustering_coefficient(ig_graph, node_ids),
            percolation=calc_percolation_centrality(
                ig_graph, node_ids, percolation_states(dpg_model)
            ),
        )
