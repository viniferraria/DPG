#!/usr/bin/env -S uv run python

import logging
import csv
import datetime
import pickle
import time
import traceback
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from dpg import DPGExplainer

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



RANDOM_STATE = 42
N_SPLITS = 2
RNG = np.random.default_rng(RANDOM_STATE)

STATES_DIR = Path(__file__).parent / "states"
STATES_DIR.mkdir(exist_ok=True)

current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logger = get_logger(__name__, log_file=f"dpg_explainer_{current_time}.log")


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


logger.info("Starting DPG explainer experiments...")


all_records: list[NodeMetricRecord] = []

scenarios_files = [
    # "test_datasets/scenario_01_sanity_check.csv",
    # "test_datasets/scenario_02_moderate_noise_weak_corr.csv",
    # "test_datasets/scenario_01b_binary_y.csv",
    # "test_datasets/scenario_02b_binary_y.csv",
    # "test_datasets/scenario_1.csv",
    "test_datasets/scenario_2.csv",
    # "test_datasets/scenario_3.csv",
    # "test_datasets/scenario_5.csv",
]

for each in scenarios_files:
    logger.info(f"Processing {each}...")
    scenario_name = Path(each).stem
    df = pd.read_csv(each)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_STATE
    )

    for model_name in ["RandomForest", "ExtraTrees"]:
        run_key = f"{scenario_name}_{model_name}"
        split_accuracies: list[float] = []

        for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if model_name == "RandomForest":
                model = RandomForestClassifier(random_state=RANDOM_STATE)
            else:
                model = ExtraTreesClassifier(random_state=RANDOM_STATE)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            split_accuracies.append(accuracy)
            logger.info(f"[{run_key}] split={split_idx} accuracy={accuracy:.4f}")

            try:
                explainer = DPGExplainer(
                    model=model,
                    feature_names=X.columns,
                    target_names=np.unique(y).astype(str).tolist(),
                )
                # Fix: fit on training data only to avoid test-data leak
                explainer.fit(X_train.values)

                ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

                pkl_explainer = (
                    STATES_DIR / f"{run_key}_split{split_idx}_dpg_explainer_{ts}.pkl"
                )
                with open(pkl_explainer, "wb") as f:
                    logger.info(f"Saving DPGExplainer → {pkl_explainer}")
                    pickle.dump(explainer, f)

                logger.info(
                    f"Generating global explanation for {run_key} split={split_idx}..."
                )
                start = time.time()
                explanation = explainer.explain_global(communities=True)
                processing_time = time.time() - start
                logger.info(f"Global explanation generated in {processing_time:.2f}s")

                pkl_explanation = (
                    STATES_DIR / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
                )
                with open(pkl_explanation, "wb") as f:
                    logger.info(f"Saving explanation → {pkl_explanation}")
                    pickle.dump(explanation, f)

                node_metrics_df = explanation.node_metrics
                for node_idx, row in node_metrics_df.iterrows():
                    all_records.append(
                        NodeMetricRecord(
                            experiment=run_key,
                            split=split_idx,
                            node=str(row["Node"]),
                            degree=int(row["Degree"]),
                            in_degree=int(row["In degree nodes"]),
                            out_degree=int(row["Out degree nodes"]),
                            betweenness_centrality=float(row["Betweenness centrality"]),
                            local_reaching_centrality=float(
                                row["Local reaching centrality"]
                            ),
                            node_idx=int(node_idx),
                            label=str(row["Label"]),
                            processing_time=round(processing_time, 4),
                        )
                    )

                logger.info(
                    f"Collected {len(node_metrics_df)} node records for {run_key} split={split_idx}"
                )

                # Incremental save — all runs collected so far
                output_path = Path("datasets") / "node_metrics_3.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = [f.name for f in fields(NodeMetricRecord)]
                with open(output_path, "w+", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for record in all_records:
                        writer.writerow(asdict(record))
                logger.info(f"Saved {len(all_records)} total records to {output_path}")

            except Exception as exc:
                logger.error(f"Failed: {run_key} split={split_idx}: {exc}")
                traceback.print_exc()

        mean_acc = (
            float(np.mean(split_accuracies)) if split_accuracies else float("nan")
        )
        logger.info(
            f"[{run_key}] mean accuracy over {len(split_accuracies)} splits: {mean_acc:.4f}"
        )

logger.info("All experiments completed.")
