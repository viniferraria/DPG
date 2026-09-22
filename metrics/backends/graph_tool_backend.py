"""graph-tool implementation of the node centrality backend."""

from typing import Any, ClassVar

import graph_tool as gt
import networkx as nx
import numpy as np
from graph_tool import centrality as gt_centrality
from graph_tool import topology as gt_topology

from .base import GraphBackend, NodeCentralities


class GraphToolBackend(GraphBackend):
    """Compute the four DPG node centralities with graph-tool."""

    name: ClassVar[str] = "graph_tool"

    def node_centralities(self, dpg_model: nx.DiGraph) -> NodeCentralities:
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
        )
