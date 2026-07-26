from typing import Any, List

import networkx as nx
import pandas as pd

class EdgeMetrics:
    """Handles edge-level metric calculations."""

    @staticmethod
    def extract_edge_metrics(dpg_model: nx.DiGraph, nodes_list: List[List[str]]) -> Any:
        """
        Extracts metrics from the edges of a DPG model, including:
        - Edge Load Centrality
        - Trophic Differences
        
        Args:
            dpg_model: A NetworkX graph representing the DPG.
            nodes_list: List of [node_id, label] pairs, as returned by
                DecisionPredicateGraph.to_networkx.

        Returns:
            df: A pandas DataFrame containing the metrics for each edge in the DPG.
        """
        # Map node IDs to labels for fast lookup.
        node_id_to_label = {node_id: label for node_id, label in nodes_list}

        # Build edge rows with labels and IDs.
        edge_data_with_labels = []
        for u, v, data in dpg_model.edges(data=True):
            u_label = node_id_to_label.get(u)
            v_label = node_id_to_label.get(v)
            edge_data_with_labels.append([
                f"{u}-{v}",
                data.get("weight", 0),
                u_label,
                v_label,
                u,
                v,
            ])

        # Build a DataFrame with edges, labels, and IDs.
        df_edges_with_labels = pd.DataFrame(
            edge_data_with_labels,
            columns=[
                "Edge",
                "Weight",
                "Node_u_label",
                "Node_v_label",
                "Source_id",
                "Target_id",
            ],
        )

        return df_edges_with_labels
