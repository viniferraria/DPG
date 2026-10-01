"""graph-tool implementation of the node centrality backend."""

from typing import Any, ClassVar

import graph_tool as gt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from graph_tool import centrality as gt_centrality
from graph_tool import spectral as gt_spectral
from graph_tool import topology as gt_topology

from .base import GraphBackend, NodeCentralities, percolation_states


def _collective_influence(
    graph: gt.Graph, node_ids: list[Any], ci_radius: int
) -> dict[Any, float]:
    """Directed-out collective influence at radius ``ci_radius``."""
    # get_out_degrees is unsigned; cast before subtracting so sinks give 0, not wrap.
    out_degrees = graph.get_out_degrees(graph.get_vertices()).astype(np.int64)
    excess = np.maximum(out_degrees - 1, 0)
    influence: dict[Any, float] = {}
    for i, node_id in enumerate(node_ids):
        # Unweighted hop distances; nodes beyond max_dist get a huge sentinel.
        hops = gt_topology.shortest_distance(
            graph, source=graph.vertex(i), max_dist=ci_radius
        ).a
        frontier = excess[hops == ci_radius].sum()
        influence[node_id] = float(excess[i] * frontier)
    return influence


def _clustering(graph: gt.Graph, node_ids: list[Any]) -> dict[Any, float]:
    """Directed, unweighted local clustering coefficient.

    graph-tool's ``local_clustering`` uses a different directed definition, so
    this evaluates Fagiolo's (the one ``nx.clustering`` uses) on the sparse
    adjacency: with S = A + A^T, c_i = diag(S^3)_i / (2 (d(d-1) - 2 d_bi)),
    d being in + out degree and d_bi the number of reciprocated pairs.
    """
    n = len(node_ids)
    # graph-tool's adjacency is transposed (A[target, source]); every term
    # below is invariant under transposition.
    adj = sp.csr_matrix(gt_spectral.adjacency(graph), dtype=float)
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


def _percolation(
    graph: gt.Graph,
    weight_prop: Any,
    node_ids: list[Any],
    states: dict[Any, float],
) -> dict[Any, float]:
    """Percolation centrality with raw weights as distances."""
    n = len(node_ids)
    if n <= 2:
        return dict.fromkeys(node_ids, 0.0)
    x = [states[node_id] for node_id in node_ids]
    total = sum(x)
    # sum_s x_s * delta_s(v): with ``pivots`` and norm=False, betweenness is the
    # exact sum_{s in pivots} delta_s(v) (no rescaling), so pivots that share a
    # state value are handled in one call and scaled by that value.
    pivots_by_state: dict[float, list[int]] = {}
    for i, value in enumerate(x):
        if value != 0:
            pivots_by_state.setdefault(value, []).append(i)
    weighted_dependency = np.zeros(n)
    for value, pivots in pivots_by_state.items():
        vertex_bc, _ = gt_centrality.betweenness(
            graph, pivots=np.array(pivots), weight=weight_prop, norm=False
        )
        weighted_dependency += value * vertex_bc.a
    return {
        node_ids[i]: float(weighted_dependency[i] / (total - x[i]) / (n - 2))
        if total - x[i] != 0
        else 0.0
        for i in range(n)
    }


class GraphToolBackend(GraphBackend):
    """Compute the DPG node centralities with graph-tool."""

    name: ClassVar[str] = "graph_tool"

    def node_centralities(
        self, dpg_model: nx.DiGraph, ci_radius: int = 2
    ) -> NodeCentralities:
        node_ids: list[Any] = list(dpg_model.nodes())
        n = len(node_ids)
        index = {node_id: i for i, node_id in enumerate(node_ids)}

        edges = [
            (index[u], index[v], float(data.get("weight", 1.0)))
            for u, v, data in dpg_model.edges(data=True)
        ]
        total_weight = sum(w for *_, w in edges)
        if total_weight <= 0:
            from dpg.exceptions import DPGMetricError

            raise DPGMetricError.non_positive_lrc_weight()

        graph = gt.Graph(directed=True)
        graph.add_vertex(n)
        weight_prop = graph.new_edge_property("double")
        dist_prop = graph.new_edge_property("double")
        graph.add_edge_list(
            [
                (s, t, w, total_weight / w if w > 0 else float("inf"))
                for s, t, w in edges
            ],
            eprops=[weight_prop, dist_prop],
        )
        edge_weight = {(s, t): w for s, t, w in edges}

        # Betweenness uses the RAW weight as distance; graph-tool normalises
        # directed graphs by 1 / ((n-1)(n-2)), the NetworkX convention.
        vertex_bc, _ = gt_centrality.betweenness(
            graph, weight=weight_prop, norm=n > 2
        )
        betweenness = {node_ids[i]: float(vertex_bc.a[i]) for i in range(n)}

        # Harmonic: sum(1/d) over reachable targets, un-normalised.
        harmonic_prop = gt_centrality.closeness(
            graph, weight=dist_prop, harmonic=True, norm=False
        )
        harmonic = {node_ids[i]: float(harmonic_prop.a[i]) for i in range(n)}

        # graph-tool's normalised closeness is r / S (r = out-component size
        # minus the source). The Wasserman-Faust value needs a further r/(n-1).
        closeness_prop = gt_centrality.closeness(
            graph, weight=dist_prop, harmonic=False, norm=True
        )

        lrc_norm = total_weight / len(edges) if edges else 1.0
        closeness: dict[Any, float] = {}
        local_reaching: dict[Any, float] = {}
        for i in range(n):
            dist_map, pred_map = gt_topology.shortest_distance(
                graph, source=graph.vertex(i), weights=dist_prop, pred_map=True
            )
            distances = dist_map.a
            preds = pred_map.a

            reachable = 0
            sum_avg_weight = 0.0
            for t in range(n):
                if t == i or not np.isfinite(distances[t]):
                    continue
                reachable += 1
                # Walk the predecessor map back to i to recover one shortest
                # path, averaging the ORIGINAL weights along it.
                path_weight_sum = 0.0
                steps = 0
                current = t
                while current != i:
                    previous = int(preds[current])
                    if previous == current:
                        break
                    path_weight_sum += edge_weight.get((previous, current), 1.0)
                    steps += 1
                    current = previous
                if steps > 0:
                    sum_avg_weight += path_weight_sum / steps

            raw_closeness = float(closeness_prop.a[i])
            closeness[node_ids[i]] = (
                raw_closeness * reachable / (n - 1)
                if reachable > 0 and n > 1 and np.isfinite(raw_closeness)
                else 0.0
            )
            local_reaching[node_ids[i]] = (
                (sum_avg_weight / lrc_norm) / (n - 1) if n > 1 else 0.0
            )

        return NodeCentralities(
            betweenness=betweenness,
            local_reaching=local_reaching,
            closeness=closeness,
            harmonic=harmonic,
            collective_influence=_collective_influence(graph, node_ids, ci_radius),
            clustering=_clustering(graph, node_ids),
            percolation=_percolation(
                graph, weight_prop, node_ids, percolation_states(dpg_model)
            ),
        )
