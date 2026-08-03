#!/usr/bin/env -S uv run python

# from __future__ import annotations
"""Run DPG explainer experiments across monk datasets.
"""

import csv
import datetime
import logging
import pickle
import re
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

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
TOP_K = 3

STATES_DIR = Path(__file__).parent / "states"
OUTPUT_PATH = Path(__file__).parent / "results"

# The nominal attributes shared by every MONK's problem.
ATTRIBUTES = ["a1", "a2", "a3", "a4", "a5", "a6"]

# (train file, test file, ground-truth causal attributes).
# MONK's ships a canonical split: each .test file is the complete 432-row
# attribute space, and the .train file is a labelled subset of it. Using that
# split as-is is the benchmark's intended protocol — accuracy over the full
# space measures how well the true concept was recovered. It also preserves the
# monks-3 design, where ~5% label noise sits in .train only and .test is clean.
SCENARIO_GROUND_TRUTH: list[tuple[str, str, list[str]]] = [
    # a1 = a2 OR a5 = 1
    ("datasets/monks-1.train", "datasets/monks-1.test", ["a1", "a2", "a5"]),
    # (a5 = 3 AND a4 = 1) OR (a5 != 4 AND a2 != 3)
    ("datasets/monks-3.train", "datasets/monks-3.test", ["a5", "a4", "a2"]),
]
SCENARIOS_WITH_GT: list[str] = [train for train, _, _ in SCENARIO_GROUND_TRUTH]

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

# --- Pure helpers ------------------------------------------------------------


def to_snake_case(name: str) -> str:
    """Normalise a DataFrame column label to snake_case.

    Handles the spaced titles produced by the metrics layer ("In degree nodes")
    as well as CamelCase, collapsing any run of non-alphanumerics to a single
    underscore.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return re.sub(r"[^0-9a-zA-Z]+", "_", spaced).strip("_").lower()


def timestamp(now: datetime.datetime | None = None) -> str:
    """Return a filesystem-safe timestamp string."""
    return (now or datetime.datetime.now(tz=datetime.timezone.utc)).strftime(
        "%Y-%m-%dT%H-%M-%S"
    )


# Defined after timestamp() so the log filename reuses that helper rather than
# duplicating its format (and its timezone handling).
logger = get_logger(__name__, log_file=f"dpg_explainer_{timestamp()}.log")


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV scenario into a feature frame ``X`` and target series ``y``.

    The last column is treated as the target, all preceding columns as features.
    """

    columns = ["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    # MONK's rows are whitespace-delimited with a leading space. A literal " "
    # separator turns that leading field into an all-NaN index; a whitespace
    # regex consumes it, leaving a clean RangeIndex.
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    df = df.drop(columns=["id"])
    y = df.loc[:, "class"]
    X = df.drop(columns=["class"])
    return X, y


def pre_process_dataset(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """One-hot encode the nominal MONK's attributes across several frames at once.

    The raw attributes are nominal, so leaving them as ordinal integers would
    let the trees split on meaningless orderings ("a5 <= 2.5"). Encoded columns
    are booleans, which sklearn accepts directly.

    All ``frames`` are encoded together so they end up with an identical column
    set — encoding train and test separately would silently produce different
    columns whenever one of them lacks an attribute level. Only the category
    vocabulary is shared, never the target, so this leaks nothing.

    Returns one encoded frame per input, in the order given.
    """
    combined = pd.concat(frames, keys=range(len(frames)))
    encoded = pd.get_dummies(
        combined,
        columns=ATTRIBUTES,
        drop_first=False,
        dtype=bool,
    )
    return [encoded.xs(position) for position in range(len(frames))]


def base_feature_name(label: str) -> str:
    """Map a one-hot column back to its source attribute ("a1_2" -> "a1").

    Only a trailing ``_<digits>`` is stripped, so attribute names that happen to
    contain underscores survive untouched.
    """
    return re.sub(r"_\d+$", "", str(label))



def evaluate_accuracy(model: Any, X_test: Any, y_test: Any) -> float:
    """Return accuracy of a fitted ``model`` on a held-out split."""
    return float(accuracy_score(y_test, model.predict(X_test)))


def extract_top_k_features(
    explanation: pd.DataFrame,
    top_k: int,
    metric: str = "Local reaching centrality",
) -> list[str]:
    """Return the top-k source attributes by ``metric`` from a DPG explanation.

    Predicate labels are one-hot column names ("a1_2 <= 0.5"), so each is mapped
    back to its source attribute and de-duplicated — keeping the highest-ranked
    occurrence — before taking the top ``top_k``. That keeps the result
    comparable to a ground truth expressed in raw attribute names, and keeps its
    cardinality at ``top_k`` rather than collapsing duplicates later.
    """
    results = explanation[~(explanation["Label"].str.startswith("Class"))].copy()
    results["Label"] = results["Label"].str.extract(r"([^\s]+)")
    results["Label"] = results["Label"].map(base_feature_name)
    top_features = (
        results.sort_values(by=metric, ascending=False)
        .drop_duplicates(subset="Label")
        .head(top_k)["Label"]
        .tolist()
    )
    return top_features


def calculate_causal_accuracy(
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


def records_from_explanation(explanation: Any) -> list[dict[str, Any]]:
    """Convert a DPG explanation's node-metrics frame into a list of dicts.

    Every column of ``explanation.node_metrics`` is carried through as-is, plus
    the frame's positional index as ``node_index``. Column labels are left in
    their original form here; :func:`write_records_to_csv` snake-cases them.

    Pure transformation: depends only on its argument, so it can be tested with
    a lightweight stand-in object exposing a ``node_metrics`` DataFrame.
    """
    df = explanation.node_metrics
    return [
        {"node_index": node_index, **row.to_dict()}
        for node_index, row in df.iterrows()
    ]


def write_records_to_csv(
    records: list[dict[str, Any]],
    output_path: Path,
    **fixed_values: Any,
) -> None:
    """Append ``records`` to ``output_path`` as CSV, snake-casing the columns.

    Each keyword in ``fixed_values`` becomes a column holding that single value
    for every row — used for per-split constants such as ``experiment``,
    ``split_idx`` and ``processing_time``. The header is written only when the
    file does not yet exist, so repeated calls append to one file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    for column, value in fixed_values.items():
        df[column] = value
    df.columns = [to_snake_case(column) for column in df.columns]
    df.to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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



def evaluate_causal_accuracy(
    explanation: pd.DataFrame,
    gt_features: list[str],
    metric_name: str,
) -> tuple[list[str], set[str], float, float]:
    explanation_top_features = extract_top_k_features(
        explanation=explanation,
        top_k=TOP_K,
        metric=metric_name,
    )
    intersection, precision, recall = calculate_causal_accuracy(
        ground_truth=gt_features,
        explanation=explanation_top_features,
    )
    return explanation_top_features, intersection, precision, recall


def extract_causal_accuracy_to_file(
    explanation: pd.DataFrame,
    gt_features: list[str],
    run_key: str,
    split_idx: int,
    metric_name: str,
    output_path: Path,
) -> None:
    """Extract causal accuracy metrics from an explanation and append them to a CSV."""
    explanation_top_features, intersection, precision, recall = evaluate_causal_accuracy(
        explanation=explanation,
        gt_features=gt_features,
        metric_name=metric_name,
    )

    write_causal_accuracy_to_csv(
        run_key=run_key,
        k=TOP_K,
        ground_truth=gt_features,
        explanation_top_features=explanation_top_features,
        split_idx=split_idx,
        metric=metric_name,
        intersection=intersection,
        precision=precision,
        recall=recall,
        output_path=output_path,
    )


# --- Orchestration (impure shell) -------------------------------------------


def _build_explanation(
    model: Any,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
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
    logger.info(f"Saving DPGExplainer → {pkl_explainer}")
    save_pickle(explainer, pkl_explainer)

    logger.info(
        f"Generating global explanation for {run_key} split={split_idx}..."
    )
    start = time.time()
    explanation = explainer.explain_global(communities=True)
    processing_time = time.time() - start
    logger.info(f"Global explanation generated in {processing_time:.2f}s")

    pkl_explanation = STATES_DIR / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
    logger.info(f"Saving explanation → {pkl_explanation}")
    save_pickle(explanation, pkl_explanation)

    logger.info(
        f"Collected {len(explanation.node_metrics)} node records for {run_key} split={split_idx}"
    )
    return explanation, processing_time


def run_experiments_with_ground_truth(
    scenarios_with_gt: list[tuple[str, str, list[str]]],
    output_dir: Path,
    logger: Any,
):
    """Run experiments per scenario, additionally tracking causal accuracy against
    each scenario's ground-truth causal features.

    Each scenario is a ``(train_file, test_file, gt_features)`` triple and uses
    MONK's canonical split: the model is fitted on ``train_file`` and scored on
    ``test_file`` (the complete attribute space). There is exactly one split per
    scenario, recorded as ``split_idx=0``.

    Node-metric rows are appended to a timestamped CSV under ``output_dir``;
    causal-accuracy rows go to their own timestamped CSV.
    """
    node_metrics_path = output_dir / f"node_metrics_{timestamp()}.csv"
    split_idx = 0

    for train_file, test_file, gt_features in scenarios_with_gt:
        logger.info(f"Processing {train_file} -> {test_file}...")
        scenario_name = Path(train_file).stem
        X_train_raw, y_train = load_dataset(train_file)
        X_test_raw, y_test = load_dataset(test_file)
        # Nominal attributes: encode before fitting so trees split on category
        # membership rather than an arbitrary ordinal threshold. Both frames go
        # through one call so their columns are guaranteed to match.
        X_train, X_test = pre_process_dataset(X_train_raw, X_test_raw)

        for model_name, model_factory in MODEL_FACTORIES.items():
            run_key = f"{scenario_name}_{model_name}"

            model = model_factory()
            model.fit(X_train, y_train)
            accuracy = evaluate_accuracy(model, X_test, y_test)

            try:
                explanation, processing_time = _build_explanation(
                    model=model,
                    X_train=X_train,
                    y=y_train,
                    run_key=run_key,
                    split_idx=split_idx,
                )
                records = records_from_explanation(explanation)
                write_records_to_csv(
                    records,
                    node_metrics_path,
                    experiment=run_key,
                    split_idx=split_idx,
                    processing_time=round(processing_time, 4),
                )

                for metric_name in METRICS:
                    extract_causal_accuracy_to_file(
                        explanation=explanation.node_metrics,
                        gt_features=gt_features,
                        run_key=run_key,
                        split_idx=split_idx,
                        metric_name=metric_name,
                        output_path=output_dir,
                    )

            except Exception as exc:  # noqa: BLE001 - keep batch resilient
                logger.error(f"Failed: {run_key}: {exc}")
                traceback.print_exc()

            logger.info(
                f"[{run_key}] accuracy on {Path(test_file).name} "
                f"({len(X_test)} rows): {accuracy:.4f}"
            )


def main() -> None:
    """Entry point: configure logging and run all experiments."""
    STATES_DIR.mkdir(exist_ok=True)
    logger = get_logger(__name__, log_file=f"dpg_explainer_{timestamp()}.log")

    logger.info("Starting DPG explainer experiments...")
    run_experiments_with_ground_truth(SCENARIO_GROUND_TRUTH, OUTPUT_PATH, logger)
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
