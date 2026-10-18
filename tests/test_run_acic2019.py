"""Minimal tests for the ACIC 2019 experiment runner (``experiments/acic2019``).

Loaded by file path (the suite is not a package) after ``chdir`` to a temp dir,
because importing ``run_acic2019`` opens a log file in the CWD.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.causal_accuracy import METRICS, causal_accuracy_rows

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
    rows = causal_accuracy_rows(explanation.node_metrics, ["V3"], "t", 0)
    assert rows["metric"].tolist() == METRICS
    assert rows["explanation_top_features"].ne("").all()
