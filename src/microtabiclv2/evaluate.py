"""Repeatable evaluation against lightweight tabular baselines."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .estimator import MicroTabICLv2Classifier
from .train import load_checkpoint


def _datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    iris = load_iris()
    binary_mask = iris.target != 0
    return {
        "binary_iris": (iris.data[binary_mask], iris.target[binary_mask] - 1),
        "iris": (iris.data, iris.target),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "digits": load_digits(return_X_y=True),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": float(auc),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
    }


def _summary(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([item[metric] for item in values])),
            "std": float(np.std([item[metric] for item in values])),
        }
        for metric in ("accuracy", "roc_auc", "log_loss")
    }


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    seeds: int = 5,
    max_train_rows: int = 100,
) -> dict[str, Any]:
    model, metadata = load_checkpoint(checkpoint, device=device)
    if model.config.task != "classification":
        raise ValueError("This benchmark currently expects a classification checkpoint")

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": metadata.get("step"),
        "seeds": seeds,
        "max_train_rows": max_train_rows,
        "datasets": {},
    }

    for dataset_name, (x, y) in _datasets().items():
        train_rows = min(max_train_rows, len(x) // 2)
        runs: dict[str, list[dict[str, float]]] = defaultdict(list)
        for seed in range(seeds):
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                train_size=train_rows,
                random_state=seed,
                stratify=y,
            )
            estimators = {
                "microtabiclv2": MicroTabICLv2Classifier(model, device=device),
                "logistic_regression": make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)
                ),
                "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
            }
            for method, estimator in estimators.items():
                estimator.fit(x_train, y_train)
                probabilities = estimator.predict_proba(x_test)
                runs[method].append(_metrics(y_test, probabilities))

        report["datasets"][dataset_name] = {
            method: {"summary": _summary(values), "runs": values} for method, values in runs.items()
        }
    return report


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination


def print_report(report: dict[str, Any]) -> None:
    header = f"{'dataset':<16} {'method':<20} {'accuracy':>12} {'roc_auc':>12} {'log_loss':>12}"
    print(header)
    print("-" * len(header))
    for dataset, methods in report["datasets"].items():
        for method, results in methods.items():
            summary = results["summary"]
            print(
                f"{dataset:<16} {method:<20} "
                f"{summary['accuracy']['mean']:.3f}+/-{summary['accuracy']['std']:.3f} "
                f"{summary['roc_auc']['mean']:.3f}+/-{summary['roc_auc']['std']:.3f} "
                f"{summary['log_loss']['mean']:.3f}+/-{summary['log_loss']['std']:.3f}"
            )
