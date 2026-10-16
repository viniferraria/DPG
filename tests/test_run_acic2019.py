"""Minimal tests for the ACIC 2019 experiment runner (``experiments/acic2019``).

Loaded by file path (the suite is not a package) after ``chdir`` to a temp dir,
because importing ``run_acic2019`` opens a log file in the CWD.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as reference
from metrics.nodes import NodeMetrics

RUN_ACIC = (
    Path(__file__).resolve().parent.parent / "experiments/acic2019/run_acic2019.py"
)


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("run_acic2019_cwd"))
    try:
        spec = importlib.util.spec_from_file_location("run_acic2019_under_test", RUN_ACIC)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        mp.undo()


def test_every_ranked_metric_is_a_node_metrics_column(m):
    G = nx.DiGraph()
    G.add_weighted_edges_from([("n0", "n1", 4), ("n0", "n2", 2), ("n1", "n3", 3), ("n2", "n3", 1)])
    labels = ["V3 <= 17.5", "V9 <= 1.5", "V9 > 1.5", "Pred 2.5"]
    df = NodeMetrics.extract_node_metrics(G, [[f"n{i}", label] for i, label in enumerate(labels)])

    assert set(m.METRICS) <= set(df.columns)


def test_extract_top_k_features_matches_reference_apart_from_pred_sinks(m):
    node_metrics = pd.DataFrame(
        {
            "Label": ["V1 <= 1.5", "V2 <= 1.5", "V2 > 1.5", "V2 <= 3.5", "V3 <= 0.5", "Class 1"],
            "Percolation centrality": [0.9, 0.8, 0.7, 0.6, 0.5, 5.0],
        }
    )
    with_pred = pd.concat(
        [node_metrics, pd.DataFrame({"Label": ["Pred 9.0"], "Percolation centrality": [7.0]})],
        ignore_index=True,
    )

    ours = m.extract_top_k_features(node_metrics, top_k=3, metric="Percolation centrality")

    assert ours == reference.extract_top_k_features(
        node_metrics, top_k=3, metric="Percolation centrality"
    )
    # Top node labels, not summed features: V2 fills two of the three slots.
    assert ours == ["V1", "V2", "V2"]
    # Named deviation: Pred sinks are dropped here, kept (as "Pred") by the reference.
    assert m.extract_top_k_features(with_pred, 3, "Percolation centrality") == ours
    assert reference.extract_top_k_features(with_pred, 3, "Percolation centrality")[0] == "Pred"


def test_causal_accuracy_csv_columns_match_reference(m, tmp_path):
    path = tmp_path / "acc.csv"
    m.write_causal_accuracy_to_csv(
        "run", 3, ["V3"], ["V3"], 0, "Closeness centrality", {"V3"}, 1.0, 1.0, path
    )

    assert m.TOP_K == reference.TOP_K and m.METRICS == reference.METRICS
    header = path.read_text().splitlines()[0].split(",")
    assert header == [
        "experiment", "split", "k", "ground_truth", "explanation_top_features",
        "metric", "intersection", "precision", "recall",
    ]


def test_calculate_causal_accuracy(m):
    inter, precision, recall = m.calculate_causal_accuracy(["V3"], ["V29", "V3", "V27"])

    assert inter == {"V3"}
    assert precision == pytest.approx(1 / 3)
    assert recall == 1.0


def test_true_cate_dgp64_is_rank_of_age(m):
    X = pd.DataFrame({"V3": [15, 16, 16, 20]})

    cate = m.true_cate_dgp64(X)

    # average ranks 1, 2.5, 2.5, 4
    assert cate == pytest.approx(2 + 0.5 * np.array([1, 2.5, 2.5, 4]) / 100)


def test_load_and_list_datasets(m, tmp_path):
    folder = tmp_path / "dgp64"
    folder.mkdir()
    for n in (10, 9, 100, 11):
        pd.DataFrame({"Y": [1.0], "A": [1], "V1": [0], "V2": [3]}).to_csv(
            folder / f"low{n}.csv", index=False
        )

    groups = m.list_datasets(64, 2, tmp_path)
    merged = m.list_datasets(64, 5, tmp_path, n_merge=2)
    X, A, _ = m.load_dataset(merged[0])

    assert [[f.name for f in g] for g in groups] == [["low9.csv"], ["low10.csv"]]
    # 4 files -> 2 groups of 2; numeric order puts 100 last.
    assert [[f.name for f in g] for g in merged] == [
        ["low9.csv", "low10.csv"],
        ["low11.csv", "low100.csv"],
    ]
    assert len(X) == 2 and list(X.columns) == ["V1", "V2"] and A.iloc[0] == 1
    with pytest.raises(FileNotFoundError):
        m.list_datasets(1, 1, tmp_path)


def test_true_cate_dgp39_uses_v8_and_v19(m):
    X = pd.DataFrame({"V8": [0.0, 1.0], "V19": [0.0, 2.0], "V1": [9.0, 9.0]})

    assert m.true_cate_dgp39(X) == pytest.approx([3.0, 3 * (1 - 2 + 2.2)])


def test_ground_truth_names_are_in_the_dgp_columns(m):
    assert {dgp: gt for dgp, gt, _ in m.SCENARIO_GROUND_TRUTH} == {
        64: ["V3"],
        39: ["V8", "V19"],
    }
    assert set(m.TRUE_CATE) == {64, 39}


def test_build_explanation_runs_on_a_causal_forest(m, tmp_path):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randint(0, 5, size=(120, 3)), columns=["V1", "V2", "V3"])
    A = rng.binomial(1, 0.5, size=120)
    Y = A * X["V3"].to_numpy() + rng.normal(scale=0.1, size=120)
    model = m.CausalForest(n_estimators=4, criterion="het", random_state=0).fit(X.values, A, Y)

    explanation, _ = m._build_explanation(model, X, "t", 0, tmp_path)

    # Regression DPG: communities must stay off or extract_communities raises.
    assert set(m.METRICS) <= set(explanation.node_metrics.columns)
    assert m.extract_top_k_features(explanation.node_metrics, 3)
