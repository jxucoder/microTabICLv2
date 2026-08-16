"""Small experiments that show what synthetic pretraining teaches."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .config import get_model_config, get_train_config
from .estimator import MicroTabICLv2Classifier
from .model import MicroTabICLv2
from .train import TrainingResult, load_checkpoint, train_model

_TASKS = (
    ("vertical", np.array([1.0, 0.0]), ((0.0, -2.0), (0.0, 2.0))),
    ("diagonal", np.array([1.0, 1.0]), ((-2.0, 2.0), (2.0, -2.0))),
    ("horizontal", np.array([0.0, 1.0]), ((-2.0, 0.0), (2.0, 0.0))),
    ("anti-diagonal", np.array([1.0, -1.0]), ((-2.0, -2.0), (2.0, 2.0))),
)


def _features(points: np.ndarray) -> np.ndarray:
    """Repeat two visible features so the proof stays inside the 4+ feature training range."""
    return np.column_stack((points, points)).astype(np.float32)


def _context(rows: int, rng: np.random.Generator, phase: float = 0.07) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, rows, endpoint=False) + phase
    radii = rng.uniform(0.55, 1.75, rows)
    return np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))


def _probability(
    model: MicroTabICLv2,
    context: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    device: str,
) -> np.ndarray:
    estimator = MicroTabICLv2Classifier(model, device=device)
    return estimator.fit(_features(context), labels).predict_proba(_features(test))[:, 1]


def _heat_color(value: float) -> str:
    blue, middle, orange = (34, 70, 132), (244, 241, 233), (224, 105, 52)
    start, end, amount = (blue, middle, value * 2) if value < 0.5 else (middle, orange, value * 2 - 1)
    rgb = tuple(round(a + (b - a) * amount) for a, b in zip(start, end, strict=True))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _rule_svg(
    output: Path,
    axis: np.ndarray,
    context: np.ndarray,
    surfaces: list[np.ndarray],
    aucs: dict[str, float],
    controls: dict[str, tuple[float, float]],
) -> None:
    width, height, size = 940, 1_035, 270
    panel_x, panel_y = (55, 475), (112, 430)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#20241f}"
        ".title{font-size:28px;font-weight:750}.sub{font-size:14px;fill:#667066}"
        ".panel{font-size:17px;font-weight:700}.metric{font-size:14px;font-weight:700}"
        ".tick{font-size:11px;fill:#667066}.bar{font-size:12px}</style>",
        '<defs><linearGradient id="heat"><stop stop-color="#224684"/><stop offset=".5" '
        'stop-color="#f4f1e9"/><stop offset="1" stop-color="#e06934"/></linearGradient></defs>',
        f'<rect width="{width}" height="{height}" fill="#f7f4ed"/>',
        '<text class="title" x="45" y="43">One frozen model, four different rules</text>',
        '<text class="sub" x="45" y="68">Same feature coordinates and test grid; '
        "only the context labels change.</text>",
    ]
    step = size / len(axis)

    for index, ((name, weight, boundary), probabilities) in enumerate(zip(_TASKS, surfaces, strict=True)):
        left = panel_x[index % 2]
        top = panel_y[index // 2]
        plot_x, plot_y = left + 42, top + 35
        pieces += [
            f'<text class="panel" x="{left}" y="{top + 17}">{name}</text>',
            f'<text class="metric" x="{left + 300}" y="{top + 17}" '
            f'text-anchor="end">AUC {aucs[name]:.3f}</text>',
        ]
        for cell_index, probability in enumerate(probabilities):
            column, row = cell_index % len(axis), cell_index // len(axis)
            x = plot_x + column * step
            y = plot_y + (len(axis) - 1 - row) * step
            pieces.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{step + 0.35:.2f}" '
                f'height="{step + 0.35:.2f}" fill="{_heat_color(float(probability))}"/>'
            )
        pieces.append(
            f'<rect x="{plot_x}" y="{plot_y}" width="{size}" height="{size}" '
            'fill="none" stroke="#c8c5bd"/>'
        )

        def location(
            point: tuple[float, float], plot_x: float = plot_x, plot_y: float = plot_y
        ) -> tuple[float, float]:
            return plot_x + (point[0] + 2) / 4 * size, plot_y + (2 - point[1]) / 4 * size

        (x1, y1), (x2, y2) = map(location, boundary)
        pieces.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#20241f" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        labels = (context @ weight > 0).astype(int)
        for point, label in zip(context, labels, strict=True):
            x, y = location((float(point[0]), float(point[1])))
            color = "#e06934" if label else "#224684"
            pieces.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" '
                'stroke="#fff" stroke-width="1.4"/>'
            )
        pieces += [
            f'<text class="tick" x="{plot_x}" y="{plot_y + size + 17}">−2</text>',
            f'<text class="tick" x="{plot_x + size / 2}" y="{plot_y + size + 17}" '
            'text-anchor="middle">0</text>',
            f'<text class="tick" x="{plot_x + size}" y="{plot_y + size + 17}" text-anchor="end">2</text>',
        ]

    pieces += [
        '<rect x="55" y="760" width="100" height="12" rx="6" fill="url(#heat)"/>',
        '<text class="tick" x="55" y="792">P(class 1): 0</text>',
        '<text class="tick" x="155" y="792" text-anchor="end">1</text>',
        '<line x1="205" y1="766" x2="250" y2="766" stroke="#20241f" '
        'stroke-width="2" stroke-dasharray="7 5"/>',
        '<text class="tick" x="260" y="770">true boundary</text>',
        '<text class="panel" x="55" y="835">Controls over random rotated rules</text>',
        '<text class="sub" x="55" y="858">Mean ROC AUC ± standard error; higher is better.</text>',
    ]
    bar_left, bar_top, bar_width, row_height = 245, 885, 520, 18
    pieces.append(
        f'<line x1="{bar_left + bar_width / 2}" y1="{bar_top - 7}" '
        f'x2="{bar_left + bar_width / 2}" y2="{bar_top + row_height * len(controls)}" '
        'stroke="#777" stroke-dasharray="4 4"/>'
    )
    colors = {
        "correct context": "#287d67",
        "wrong task": "#8b9089",
        "shuffled labels": "#7654a6",
        "random weights": "#b87938",
        "logistic regression": "#376da8",
        "5-NN": "#376da8",
    }
    for row, (name, (mean, error)) in enumerate(controls.items()):
        y = bar_top + row * row_height
        length = mean * bar_width
        pieces += [
            f'<text class="bar" x="55" y="{y + 10}">{name}</text>',
            f'<rect x="{bar_left}" y="{y}" width="{length:.1f}" height="11" '
            f'fill="{colors[name]}"/>',
            f'<text class="bar" x="{bar_left + length + 7:.1f}" y="{y + 10}">{mean:.3f} ± {error:.3f}</text>',
        ]
    pieces += [
        f'<text class="tick" x="{bar_left}" y="{bar_top + row_height * len(controls) + 13}">0</text>',
        f'<text class="tick" x="{bar_left + bar_width / 2}" '
        f'y="{bar_top + row_height * len(controls) + 13}" text-anchor="middle">chance 0.5</text>',
        f'<text class="tick" x="{bar_left + bar_width}" '
        f'y="{bar_top + row_height * len(controls) + 13}" text-anchor="end">1</text>',
    ]
    pieces.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(pieces) + "\n")


def rule_switching_proof(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "auto",
    seed: int = 11,
    repeats: int = 20,
    context_rows: int = 48,
    grid_size: int = 31,
) -> dict[str, float]:
    """Render four task-dependent boundaries plus randomized controls."""
    if repeats < 1 or context_rows < 8 or grid_size < 9:
        raise ValueError("repeats >= 1, context_rows >= 8, and grid_size >= 9 are required")
    model, _ = load_checkpoint(checkpoint, device=device)
    rng = np.random.default_rng(seed)
    context = _context(context_rows, rng)
    axis = np.linspace(-2, 2, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    surfaces, aucs = [], {}
    for name, weight, _ in _TASKS:
        labels = (context @ weight > 0).astype(int)
        truth = (grid @ weight > 0).astype(int)
        probability = _probability(model, context, labels, grid, device)
        surfaces.append(probability)
        aucs[name] = float(roc_auc_score(truth, probability))

    torch.manual_seed(seed)
    random_model = MicroTabICLv2(model.config)
    control_names = (
        "correct context",
        "wrong task",
        "shuffled labels",
        "random weights",
        "logistic regression",
        "5-NN",
    )
    scores = {name: [] for name in control_names}
    for _ in range(repeats):
        angle = rng.uniform(0, np.pi)
        weight = np.array([np.cos(angle), np.sin(angle)])
        wrong_weight = np.array([-weight[1], weight[0]])
        train = _context(context_rows, rng, phase=rng.uniform(0, 2 * np.pi))
        test = rng.uniform(-2, 2, size=(192, 2))
        labels = (train @ weight > 0).astype(int)
        truth = (test @ weight > 0).astype(int)
        variants = {
            "correct context": (model, labels),
            "wrong task": (model, (train @ wrong_weight > 0).astype(int)),
            "shuffled labels": (model, rng.permutation(labels)),
            "random weights": (random_model, labels),
        }
        for name, (candidate, candidate_labels) in variants.items():
            probability = _probability(candidate, train, candidate_labels, test, device)
            scores[name].append(roc_auc_score(truth, probability))
        for name, estimator in (
            ("logistic regression", LogisticRegression()),
            ("5-NN", KNeighborsClassifier(n_neighbors=5)),
        ):
            probability = estimator.fit(train, labels).predict_proba(test)[:, 1]
            scores[name].append(roc_auc_score(truth, probability))

    controls = {}
    for name, values in scores.items():
        array = np.asarray(values)
        error = array.std(ddof=1) / math.sqrt(len(array)) if len(array) > 1 else 0.0
        controls[name] = (float(array.mean()), float(error))
    _rule_svg(Path(output), axis, context, surfaces, aucs, controls)
    return {**aucs, **{f"control_{name}": value[0] for name, value in controls.items()}}


def _iris_splits(count: int) -> list[tuple[np.ndarray, ...]]:
    iris = load_iris()
    keep = iris.target != 0
    x, y = iris.data[keep], iris.target[keep] - 1
    return [
        train_test_split(x, y, train_size=50, random_state=seed, stratify=y)
        for seed in range(count)
    ]


def _auc_svg(records: list[dict[str, float]], output: Path, splits: int, shuffles: int) -> None:
    width, height = 760, 470
    left, top, plot_width, plot_height = 72, 82, 635, 300
    steps = np.array([row["step"] for row in records])
    low = min(row["auc"] - row["auc_se"] for row in records)
    low = min(low, min(row["shuffled_auc"] - row["shuffled_se"] for row in records), 0.5)
    high = max(row["auc"] + row["auc_se"] for row in records)
    high = max(high, max(row["shuffled_auc"] + row["shuffled_se"] for row in records), 0.5)
    y_min, y_max = max(0.0, low - 0.05), min(1.0, high + 0.05)
    def x_scale(value: float) -> float:
        return left + value / max(float(steps[-1]), 1.0) * plot_width

    def y_scale(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-6) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#20241f}"
        ".title{font-size:28px;font-weight:750}.sub{font-size:14px;fill:#667066}"
        ".tick{font-size:11px;fill:#667066}.legend{font-size:12px;font-weight:650}</style>",
        '<rect width="760" height="470" fill="#f7f4ed"/>',
        '<text class="title" x="38" y="42">Does more pretraining improve AUC?</text>',
        f'<text class="sub" x="38" y="64">Binary Iris; {splits} fixed splits, '
        f"{shuffles} shuffles/split; bands are ± standard error.</text>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#fffdf8" stroke="#cbc8c0"/>',
    ]
    for value in np.linspace(y_min, y_max, 5):
        y = y_scale(float(value))
        pieces += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd9d0"/>',
            f'<text class="tick" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>',
        ]
    chance_y = y_scale(0.5)
    pieces.append(
        f'<line x1="{left}" y1="{chance_y:.2f}" x2="{left + plot_width}" y2="{chance_y:.2f}" '
        'stroke="#777" stroke-dasharray="5 5"/>'
    )
    for key, error_key, color in (
        ("auc", "auc_se", "#287d67"),
        ("shuffled_auc", "shuffled_se", "#7654a6"),
    ):
        upper = [(x_scale(row["step"]), y_scale(row[key] + row[error_key])) for row in records]
        lower = [(x_scale(row["step"]), y_scale(row[key] - row[error_key])) for row in reversed(records)]
        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        line = " ".join(f"{x_scale(row['step']):.2f},{y_scale(row[key]):.2f}" for row in records)
        pieces += [
            f'<polygon points="{polygon}" fill="{color}" opacity=".14"/>',
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="3"/>',
        ]
        for row in records:
            x, y = x_scale(row["step"]), y_scale(row[key])
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" stroke="#fff"/>')
    tick_indexes = np.linspace(0, len(records) - 1, min(6, len(records)), dtype=int)
    for index in np.unique(tick_indexes):
        x = x_scale(records[index]["step"])
        label = int(records[index]["step"])
        pieces.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 20}" '
            f'text-anchor="middle">{label}</text>'
        )
    pieces += [
        f'<text class="legend" x="{left + 10}" y="{top + 20}" '
        'style="fill:#287d67">● correct labels</text>',
        f'<text class="legend" x="{left + 125}" y="{top + 20}" '
        'style="fill:#7654a6">● shuffled labels</text>',
        f'<text class="legend" x="{left + 255}" y="{top + 20}" '
        'style="fill:#777">- - chance</text>',
        f'<text class="sub" x="{left + plot_width / 2}" y="{height - 35}" '
        'text-anchor="middle">synthetic pretraining steps</text>',
        f'<text class="sub" x="20" y="{top + plot_height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + plot_height / 2})">ROC AUC</text>',
        "</svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(pieces) + "\n")


def auc_learning_curve(
    *,
    profile: str = "micro",
    steps: int = 500,
    every: int = 50,
    splits: int = 5,
    shuffles: int = 10,
    device: str = "auto",
    seed: int = 42,
    batch_size: int | None = None,
    output: str | Path = "evaluations/auc-vs-steps.svg",
    checkpoint: str | Path = "checkpoints/auc-curve.pt",
) -> tuple[list[dict[str, float]], TrainingResult]:
    """Train once and measure real-data AUC at fixed step intervals."""
    if every < 1 or splits < 1 or shuffles < 1:
        raise ValueError("every, splits, and shuffles must be positive")
    split_data = _iris_splits(splits)
    targets = set(range(0, steps + 1, every)) | {steps}
    records: list[dict[str, float]] = []

    def evaluate(step: int, model: MicroTabICLv2) -> None:
        if step not in targets:
            return
        correct, shuffled = [], []
        for split_seed, (x_train, x_test, y_train, y_test) in enumerate(split_data):
            estimator = MicroTabICLv2Classifier(model, device=device).fit(x_train, y_train)
            correct.append(roc_auc_score(y_test, estimator.predict_proba(x_test)[:, 1]))
            for permutation in range(shuffles):
                shuffle_seed = seed * 1_000 + split_seed * 100 + permutation
                shuffled_y = np.random.default_rng(shuffle_seed).permutation(y_train)
                shuffled_estimator = MicroTabICLv2Classifier(model, device=device).fit(
                    x_train, shuffled_y
                )
                shuffled.append(
                    roc_auc_score(y_test, shuffled_estimator.predict_proba(x_test)[:, 1])
                )
        correct_array, shuffled_array = np.asarray(correct), np.asarray(shuffled)
        records.append(
            {
                "step": float(step),
                "auc": float(correct_array.mean()),
                "auc_se": float(correct_array.std(ddof=1) / math.sqrt(splits)) if splits > 1 else 0.0,
                "shuffled_auc": float(shuffled_array.mean()),
                "shuffled_se": float(shuffled_array.std(ddof=1) / math.sqrt(len(shuffled_array)))
                if len(shuffled_array) > 1
                else 0.0,
            }
        )
        print(
            f"eval step {step:>5}: AUC {records[-1]['auc']:.3f} | "
            f"shuffled {records[-1]['shuffled_auc']:.3f}"
        )

    overrides = {
        "steps": steps,
        "device": device,
        "seed": seed,
        "log_every": every,
        "output": str(checkpoint),
    }
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    result = train_model(
        get_model_config(profile, "classification"),
        get_train_config(profile, **overrides),
        on_step=evaluate,
    )
    destination = Path(output)
    _auc_svg(records, destination, splits, shuffles)
    with destination.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return records, result
