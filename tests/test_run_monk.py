"""Minimal tests for the MONK's experiment runner (``experiments/monks``).

``experiments/monks`` is not a package and importing ``run_monk`` opens a log
file in the CWD, so the module is loaded from its path inside a fixture that
first moves the CWD to a temp dir.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from experiments.causal_accuracy import METRICS
from metrics.nodes import NodeMetrics

RUN_MONK = Path(__file__).resolve().parent.parent / "experiments/monks/run_monk.py"


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("run_monk_cwd"))
    try:
        spec = importlib.util.spec_from_file_location("run_monk_under_test", RUN_MONK)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        mp.undo()


def test_every_ranked_metric_is_a_node_metrics_column():
    edges = [("n0", "n1", 4), ("n0", "n2", 2), ("n1", "n3", 3), ("n2", "n3", 1)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    labels = ["a1_0 <= 0.5", "a2_1 <= 0.5", "a2_1 > 0.5", "Class 0"]
    nodes_list = [[f"n{i}", label] for i, label in enumerate(labels)]

    df = NodeMetrics.extract_node_metrics(G, nodes_list)

    assert set(METRICS) <= set(df.columns)


def test_run_passes_monk_deviation_keywords_to_shared_scoring(m, monkeypatch, tmp_path):
    """MONK maps one-hot columns to attributes and de-duplicates them."""
    node_metrics = pd.DataFrame({"Label": ["a5_2 > 0.5", "Class 0"]})
    seen = {}

    def fake_rows(nm, gt, experiment, split, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"experiment": [experiment]})

    monkeypatch.setattr(m, "causal_accuracy_rows", fake_rows)
    monkeypatch.setattr(m, "node_metrics_rows", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(m, "load_dataset", lambda p: (pd.DataFrame({"a1": [1]}), pd.Series([0])))
    monkeypatch.setattr(m, "pre_process_dataset", lambda *f: list(f))
    monkeypatch.setattr(m, "evaluate_accuracy", lambda *a: 1.0)
    monkeypatch.setattr(
        m, "_build_explanation",
        lambda **k: (type("E", (), {"node_metrics": node_metrics})(), 0.1),
    )
    monkeypatch.setattr(m, "MODEL_FACTORIES", {"RF": lambda: type("F", (), {"fit": lambda s, X, y: None})()})

    m.run_experiments_with_ground_truth([("t.train", "t.test", ["a5"])], tmp_path, m.logger)

    assert seen["feature_name"] is m.base_feature_name
    assert seen["drop_duplicates"] is True
    assert m.base_feature_name("a5_2") == "a5"
