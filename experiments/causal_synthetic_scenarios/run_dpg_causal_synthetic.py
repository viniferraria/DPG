#!/usr/bin/env -S uv run python
# from __future__ import annotations
"""Run DPG explainer experiments across scenario datasets.

Refactor of ``main.py`` favouring small, pure, testable functions composed by a
thin imperative shell (``main``). Importing this module has no side effects, so
each function can be unit-tested in isolation.
"""

import csv
import datetime
import logging
import pickle
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from dpg import DPGExplainer
from dpg.explainer import DPGExplanation


def get_logger(name, log_file=None):
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


# --- Configuration -----------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 2
TEST_SIZE = 0.2
TOP_K = 3

STATES_DIR = Path(__file__).parent / "states"
OUTPUT_PATH = Path(__file__).parent / "results"

SCENARIO_GROUND_TRUTH: dict[str, list[str]] = {
    "datasets/scenario_1.csv": ["F1"],
    "datasets/scenario_2_updated.csv": ["F1"],
    "datasets/scenario_3.csv": ["F1", "F2"],
    "datasets/scenario_5.csv": ["F1", "F2"],
}
SCENARIOS_WITH_GT: list[str] = list(SCENARIO_GROUND_TRUTH)

METRICS = [
    "Local reaching centrality",
    "Closeness centrality",
    "Harmonic centrality",
]
CONFIG_PATH = ((Path(__file__).parent).parent) / "../config.yaml"

# Each entry maps a model name to a zero-arg factory producing a fresh estimator.
ModelFactory = Callable[[], Any]
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "RandomForest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
    "ExtraTrees": lambda: ExtraTreesClassifier(random_state=RANDOM_STATE),
}


@dataclass
class NodeMetricRecord:
    """Single row in the metrics CSV export."""

    experiment: str
    split: int
    node: str
    degree: int
    in_degree: int
    out_degree: int
    betweenness_centrality: float
    local_reaching_centrality: float
    node_idx: int
    label: str
    processing_time: float


# --- Pure helpers ------------------------------------------------------------


def timestamp(now: datetime.datetime | None = None) -> str:
    """Return a filesystem-safe timestamp string."""
    return (now or datetime.datetime.now(tz=datetime.timezone.utc)).strftime("%Y-%m-%dT%H-%M-%S")


def make_splitter() -> StratifiedShuffleSplit:
    """Return a fresh, reproducible train/test splitter using the module config."""
    return StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV scenario into a feature frame ``X`` and target series ``y``.

    The last column is treated as the target, all preceding columns as features.
    """
    df = pd.read_csv(path)
    return df.iloc[:, :-1], df.iloc[:, -1]


def evaluate_accuracy(model: Any, X_test: Any, y_test: Any) -> float:
    """Return accuracy of a fitted ``model`` on a held-out split."""
    return float(accuracy_score(y_test, model.predict(X_test)))


def extract_top_k_features(
    explanation: pd.DataFrame,
    top_k: int,
    metric: str = "Local reaching centrality",
) -> list[str]:
    """Return the top-k features by local reaching centrality from a DPG explanation."""
    results = explanation[~(explanation["Label"].str.startswith("Class"))].copy()
    results["Label"] = results["Label"].str.extract(r"([^\s]+)")
    top_features = (
        results.sort_values(by=metric, ascending=False).head(top_k)["Label"].tolist()
    )
    return top_features


def evaluate_causal_accuracy(
    ground_truth: list[str],
    explanation: list[str],
) -> tuple[set[str], float, float]:
    """Return the causal accuracy of an explanation against a ground-truth feature list."""
    set_gt = set(ground_truth)
    set_expl = set(explanation)

    # intersection c_k = |E ∩ GT|, where E is the set of top-k features in the explanation and GT
    intersection_expl_gt = set_expl.intersection(set_gt)  # c_k ∩ c
    cardinality_intersection_expl_gt = len(intersection_expl_gt)  # |c_k ∩ c|

    cardinality_expl = len(set_expl)  # |c_k|
    cardinality_gt = len(set_gt)  # |c|
    precision = (
        cardinality_intersection_expl_gt / cardinality_expl
        if cardinality_expl > 0
        else 0.0
    )
    recall = (
        cardinality_intersection_expl_gt / cardinality_gt if cardinality_gt > 0 else 0.0
    )
    return intersection_expl_gt, precision, recall


def records_from_explanation(
    explanation: Any,
    experiment: str,
    split_idx: int,
    processing_time: float,
) -> list[NodeMetricRecord]:
    """Convert a DPG explanation's node-metrics frame into typed records.

    Pure transformation: depends only on its arguments, so it can be tested with
    a lightweight stand-in object exposing a ``node_metrics`` DataFrame.
    """
    df = explanation.node_metrics
    return [
        NodeMetricRecord(
            experiment=experiment,
            split=split_idx,
            node=str(row["Node"]),
            degree=int(row["Degree"]),
            in_degree=int(row["In degree nodes"]),
            out_degree=int(row["Out degree nodes"]),
            betweenness_centrality=float(row["Betweenness centrality"]),
            local_reaching_centrality=float(row["Local reaching centrality"]),
            node_idx=int(node_idx),
            label=str(row["Label"]),
            processing_time=round(processing_time, 4),
        )
        for node_idx, row in df.iterrows()
    ]


def write_records_to_csv(records: list[NodeMetricRecord], output_path: Path) -> None:
    """Write all ``records`` to ``output_path`` (header + one row each)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    fieldnames = [f.name for f in fields(NodeMetricRecord)]
    with open(output_path, "a+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def mean_or_nan(values: list[float]) -> float:
    """Mean of ``values``, or NaN when empty."""
    return sum(values) / len(values) if values else float("nan")


def write_causal_accuracy_to_csv(
    run_key: str,
    k: int,
    ground_truth: list[str],
    explanation_top_features: list[str],
    split_idx: int,
    metric: str,
    intersection: set[str],
    precision: float,
    recall: float,
    output_path: Path,
) -> None:
    """Append a row with causal accuracy metrics to a CSV at ``output_path``."""
    fieldnames = [
        "experiment",
        "split",
        "k",
        "ground_truth",
        "explanation_top_features",
        "metric",
        "intersection",
        "precision",
        "recall",
    ]
    ts = timestamp()
    file_path = output_path / f"causal_accuracy_{ts}.csv"
    file_exists = file_path.exists()
    with open(file_path, "a+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "experiment": run_key,
            "split": split_idx,
            "k": k,
            "ground_truth": ";".join(ground_truth),
            "explanation_top_features": ";".join(explanation_top_features),
            "metric": metric,
            "intersection": ";".join(intersection),
            "precision": precision,
            "recall": recall,
        })


def extract_top_k_features_to_file(
    explanation: pd.DataFrame,
    gt_features: list[str],
    run_key: str,
    split_idx: int,
    metric_name: str,
    output_path: Path,
) -> None:
    explanation_top_features_lrc = extract_top_k_features(
        explanation=explanation,
        top_k=TOP_K,
        metric=metric_name,
    )
    intersection, precision, recall = evaluate_causal_accuracy(
        ground_truth=gt_features,
        explanation=explanation_top_features_lrc,
    )
    write_causal_accuracy_to_csv(
        run_key=run_key,
        k=TOP_K,
        ground_truth=gt_features,
        explanation_top_features=explanation_top_features_lrc,
        split_idx=split_idx,
        metric=metric_name,
        intersection=intersection,
        precision=precision,
        recall=recall,
        output_path=output_path,
    )


# --- Orchestration (impure shell) -------------------------------------------


def iter_splits(
    X: pd.DataFrame, y: pd.Series, splitter: StratifiedShuffleSplit
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Yield ``(split_idx, X_train, X_test, y_train, y_test)`` for each fold."""
    for split_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X.to_numpy(), y.to_numpy())
    ):
        yield (
            split_idx,
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )


def _build_explanation(
    model: Any,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
    processing_logger: Any,
) -> tuple[DPGExplanation, float]:
    """Fit a DPG explainer on one split, persist artifacts, return its explanation."""
    explainer = DPGExplainer(
        model=model,
        feature_names=X_train.columns.tolist(),
        target_names=np.unique(y).astype(str).tolist(),
        config_file=str(CONFIG_PATH.resolve(strict=True)),
    )
    # Fit on training data only to avoid test-data leak.
    explainer.fit(X_train.values)

    ts = timestamp()
    pkl_explainer = STATES_DIR / f"{run_key}_split{split_idx}_dpg_explainer_{ts}.pkl"
    processing_logger.info(f"Saving DPGExplainer → {pkl_explainer}")
    save_pickle(explainer, pkl_explainer)

    processing_logger.info(
        f"Generating global explanation for {run_key} split={split_idx}..."
    )
    start = time.time()
    explanation = explainer.explain_global(communities=True)
    processing_time = time.time() - start
    processing_logger.info(f"Global explanation generated in {processing_time:.2f}s")

    pkl_explanation = STATES_DIR / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
    processing_logger.info(f"Saving explanation → {pkl_explanation}")
    save_pickle(explanation, pkl_explanation)

    processing_logger.info(
        f"Collected {len(explanation.node_metrics)} node records for {run_key} split={split_idx}"
    )
    return explanation, processing_time


def explain_split(
    *,
    model: Any,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
    processing_logger: Any,
) -> list[NodeMetricRecord]:
    """Fit a DPG explainer on one split and return its node-metric records.

    ``X`` is the held-out split for this fold; the explanation itself is built
    from ``X_train`` only, to avoid test-data leak.
    """
    processing_logger.info(
        f"Explaining {run_key} split={split_idx} against {len(X)} held-out rows"
    )
    explanation, processing_time = _build_explanation(
        model=model,
        X_train=X_train,
        y=y,
        run_key=run_key,
        split_idx=split_idx,
        processing_logger=processing_logger,
    )
    return records_from_explanation(explanation, run_key, split_idx, processing_time)


def run_experiments(
    scenario_files: list[str],
    output_path: Path,
    logger: Any,
) -> list[NodeMetricRecord]:
    """Run every (scenario, model, split) combination and return their node-metric records."""
    all_records: list[NodeMetricRecord] = []
    splitter = make_splitter()

    for scenario_file in scenario_files:
        logger.info(f"Processing {scenario_file}...")
        scenario_name = Path(scenario_file).stem
        X, y = load_dataset(scenario_file)

        for model_name, model_factory in MODEL_FACTORIES.items():
            run_key = f"{scenario_name}_{model_name}"
            split_accuracies: list[float] = []

            for split_idx, X_train, X_test, y_train, y_test in iter_splits(
                X, y, splitter
            ):
                model = model_factory()
                model.fit(X_train, y_train)

                accuracy = evaluate_accuracy(model, X_test, y_test)
                split_accuracies.append(accuracy)

                try:
                    all_records.extend(
                        explain_split(
                            model=model,
                            X=X_test,
                            X_train=X_train,
                            y=y,
                            run_key=run_key,
                            split_idx=split_idx,
                            processing_logger=logger,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch resilient
                    logger.error(f"Failed: {run_key} split={split_idx}: {exc}")
                    traceback.print_exc()

            mean_accuracy = mean_or_nan(split_accuracies)
            logger.info(
                f"[{run_key}] mean accuracy over {len(split_accuracies)} splits: "
                f"{mean_accuracy:.4f}"
            )

    write_records_to_csv(all_records, output_path)
    return all_records


def run_experiments_with_ground_truth(
    scenarios_with_gt: dict[str, list[str]],
    output_dir: Path,
    logger: Any,
) -> list[NodeMetricRecord]:
    """Run experiments per scenario, additionally tracking causal accuracy against
    each scenario's ground-truth causal features.

    Node-metric records are persisted to a timestamped CSV under ``output_dir``;
    causal-accuracy rows are appended to ``output_dir / "causal_accuracy_2.csv"``.
    """
    all_records: list[NodeMetricRecord] = []
    splitter = make_splitter()

    for scenario_file, gt_features in scenarios_with_gt.items():
        logger.info(f"Processing {scenario_file}...")
        scenario_name = Path(scenario_file).stem
        X, y = load_dataset(scenario_file)

        for model_name, model_factory in MODEL_FACTORIES.items():
            run_key = f"{scenario_name}_{model_name}"
            split_accuracies: list[float] = []

            for split_idx, X_train, X_test, y_train, y_test in iter_splits(
                X, y, splitter
            ):
                model = model_factory()
                model.fit(X_train, y_train)

                accuracy = evaluate_accuracy(model, X_test, y_test)
                split_accuracies.append(accuracy)

                try:
                    explanation, processing_time = _build_explanation(
                        model=model,
                        X_train=X_train,
                        y=y,
                        run_key=run_key,
                        split_idx=split_idx,
                        processing_logger=logger,
                    )
                    all_records.extend(
                        records_from_explanation(
                            explanation, run_key, split_idx, processing_time
                        )
                    )

                    for metric_name in METRICS:
                        extract_top_k_features_to_file(
                            explanation=explanation.node_metrics,
                            gt_features=gt_features,
                            run_key=run_key,
                            split_idx=split_idx,
                            metric_name=metric_name,
                            output_path=output_dir,
                        )

                except Exception as exc:  # noqa: BLE001 - keep batch resilient
                    logger.error(f"Failed: {run_key} split={split_idx}: {exc}")
                    traceback.print_exc()

            mean_accuracy = mean_or_nan(split_accuracies)
            logger.info(
                f"[{run_key}] mean accuracy over {len(split_accuracies)} splits: "
                f"{mean_accuracy:.4f}"
            )

    write_records_to_csv(all_records, output_dir / f"node_metrics_{timestamp()}.csv")
    return all_records


def main() -> None:
    """Entry point: configure logging and run all experiments."""
    STATES_DIR.mkdir(exist_ok=True)
    logger = get_logger(__name__, log_file=f"dpg_explainer_{timestamp()}.log")

    logger.info("Starting DPG explainer experiments...")
    run_experiments_with_ground_truth(SCENARIO_GROUND_TRUTH, OUTPUT_PATH, logger)
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
