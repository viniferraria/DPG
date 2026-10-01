import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import networkx as nx
import pandas as pd

from .backends import GraphBackend, get_backend


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
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"Execution time for {func_name}: {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


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


class NodeMetrics:
    """Handles node-level metric calculations."""

    @staticmethod
    @log_timer
    def extract_node_metrics(
        dpg_model: nx.DiGraph,
        nodes_list: list[list[str]],
        trace_lrc_by_label: dict[str, float] | None = None,
        backend: str | GraphBackend = "igraph",
        ci_radius: int = 2,
    ) -> Any:
        """Compute per-node graph metrics for a DPG model.

        Args:
            dpg_model: NetworkX DiGraph representing the DPG.
            nodes_list: List of ``[node_id, label]`` pairs, as returned by
                ``DecisionPredicateGraph.to_networkx``.
            trace_lrc_by_label: Optional mapping of node label to a
                trace-consistent local-reaching-centrality score (see
                ``DecisionPredicateGraph.get_trace_consistent_lrc``). When a
                label is present, its trace-consistent value is used in place
                of the backend-computed local reaching centrality. Labels
                absent from the mapping (including when the mapping is
                ``None``) keep the backend-computed value.
            backend: Centrality backend name (see
                ``metrics.backends.BACKEND_NAMES``) or an instantiated
                ``GraphBackend``.
            ci_radius: Ball radius ℓ for collective influence (``>= 1``).

        Returns:
            DataFrame with columns ``['Node', 'Label', 'Degree', 'In degree nodes',
            'Out degree nodes', 'Betweenness centrality', 'Local reaching centrality',
            'Closeness centrality', 'Harmonic centrality', 'Collective influence',
            'Local clustering coefficient', 'Percolation centrality']``.
        """

        if ci_radius < 1:
            raise ValueError(f"ci_radius must be >= 1, got {ci_radius}")
        graph_backend = get_backend(backend) if isinstance(backend, str) else backend
        centralities = graph_backend.node_centralities(dpg_model, ci_radius=ci_radius)
        in_nodes, out_nodes, degree = calc_node_metrics(dpg_model)
        betweenness_centrality = centralities.betweenness
        local_reaching_centrality = dict(centralities.local_reaching)
        closeness_centrality = centralities.closeness
        harmonic_centrality = centralities.harmonic
        if trace_lrc_by_label is not None:
            node_label_by_id = {node_id: label for node_id, label in nodes_list}
            for node in dpg_model.nodes():
                label = node_label_by_id.get(node)
                if label in trace_lrc_by_label:
                    local_reaching_centrality[node] = trace_lrc_by_label[label]
        nodes = list(dpg_model.nodes())
        data_node = {
            "Node": nodes,
            "Degree": list(degree.values()),
            "In degree nodes": list(in_nodes.values()),
            "Out degree nodes": list(out_nodes.values()),
            "Betweenness centrality": [betweenness_centrality[n] for n in nodes],
            "Local reaching centrality": [local_reaching_centrality[n] for n in nodes],
            "Closeness centrality": [closeness_centrality[n] for n in nodes],
            "Harmonic centrality": [harmonic_centrality[n] for n in nodes],
            "Collective influence": [
                centralities.collective_influence[n] for n in nodes
            ],
            "Local clustering coefficient": [centralities.clustering[n] for n in nodes],
            "Percolation centrality": [centralities.percolation[n] for n in nodes],
        }
        df_data_node = pd.DataFrame(data_node).set_index("Node")
        df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index(
            "Node"
        )
        return pd.concat(
            [df_data_node, df_nodes_list], axis=1, join="inner"
        ).reset_index()
