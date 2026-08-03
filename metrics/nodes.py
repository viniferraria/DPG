import logging
import time
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

import igraph as ig
import networkx as nx
import pandas as pd


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Factory function to create and configure a logger.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional log file path. If provided, logs will also be written to this file.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)


def log_timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        Wrapped function that logs execution time.
    """
    func_name = getattr(func, "__name__", type(func).__name__)
    logger.info(f"Decorating function {func_name} with log_timer")

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        logger.info(
            f"Execution time for {func_name}: {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


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
def calc_node_metrics(
    dpg_model: nx.DiGraph,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    in_nodes = {}
    out_nodes = {}
    degree = {}
    for node in dpg_model.nodes():
        in_nodes[node] = dpg_model.in_degree(node)
        out_nodes[node] = dpg_model.out_degree(node)
        degree[node] = in_nodes[node] + out_nodes[node]
    return in_nodes, out_nodes, degree


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


class NodeMetrics:
    """Handles node-level metric calculations."""

    @staticmethod
    @log_timer
    def extract_node_metrics(dpg_model: nx.DiGraph, nodes_list: list[list[str]]) -> Any:
        """Compute per-node graph metrics for a DPG model.

        Args:
            dpg_model: NetworkX DiGraph representing the DPG.
            nodes_list: List of ``[node_id, label]`` pairs, as returned by
                ``DecisionPredicateGraph.to_networkx``.

        Returns:
            DataFrame with columns ``['Node', 'Label', 'Degree', 'In degree nodes',
            'Out degree nodes', 'Betweenness centrality', 'Local reaching centrality',
            'Closeness centrality', 'Harmonic centrality']``.
        """

        # Convert to igraph for fast C-level centrality computation.
        ig_graph, node_ids = _nx_to_igraph(dpg_model)
        in_nodes, out_nodes, degree = calc_node_metrics(dpg_model)
        betweenness_centrality = calc_betweenness_centrality(ig_graph)
        local_reaching_centrality = calc_local_reaching_centrality(ig_graph, node_ids)
        closeness_centrality = calc_closeness_centrality(ig_graph, node_ids)
        harmonic_centrality = calc_harmonic_centrality(ig_graph, node_ids)
        data_node = {
            "Node": list(dpg_model.nodes()),
            "Degree": list(degree.values()),
            "In degree nodes": list(in_nodes.values()),
            "Out degree nodes": list(out_nodes.values()),
            "Betweenness centrality": list(betweenness_centrality.values()),
            "Local reaching centrality": list(local_reaching_centrality.values()),
            "Closeness centrality": list(closeness_centrality.values()),
            "Harmonic centrality": list(harmonic_centrality.values()),
        }
        df_data_node = pd.DataFrame(data_node).set_index("Node")
        df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index(
            "Node"
        )
        return pd.concat(
            [df_data_node, df_nodes_list], axis=1, join="inner"
        ).reset_index()
