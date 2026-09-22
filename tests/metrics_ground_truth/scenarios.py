"""Deterministic scenario graphs used by the ground-truth generator and by
``tests/test_metrics_backends.py``.

Both consumers must build byte-identical graphs, so the recipe (dataset,
model, and DPG config) lives in exactly one place.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dpg.core import DecisionPredicateGraph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASETS_DIR = _REPO_ROOT / "experiments" / "causal_synthetic_scenarios" / "datasets"

SCENARIOS: dict[str, Path] = {
    "scenario_1": _DATASETS_DIR / "scenario_1.csv",
    "scenario_2_updated": _DATASETS_DIR / "scenario_2_updated.csv",
    "scenario_3": _DATASETS_DIR / "scenario_3.csv",
    "scenario_5": _DATASETS_DIR / "scenario_5.csv",
}

DPG_CONFIG: dict[str, object] = {
    "dpg": {
        "default": {"perc_var": 0.0001, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "aggregated_transitions", "context_order": 1},
    }
}


def build_scenario_graph(name: str) -> tuple[nx.DiGraph, list[list[str]]]:
    """Build the DPG graph for scenario ``name``.

    Returns ``(graph, nodes_list)`` exactly as ``DecisionPredicateGraph.to_networkx``
    produces them.
    """
    csv_path = SCENARIOS[name]
    df = pd.read_csv(csv_path)
    feature_names = list(df.columns[:-1])
    X = df[feature_names]
    y = df["Y"]

    model = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)
    model.fit(X, y)

    target_names = np.unique(y).astype(str).tolist()
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=feature_names,
        target_names=target_names,
        dpg_config=DPG_CONFIG,
    )
    dot = dpg.fit(X.values)
    graph, nodes_list = dpg.to_networkx(dot)
    return graph, nodes_list
