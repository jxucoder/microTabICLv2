"""A small on-the-fly synthetic graph prior.

This is intentionally much smaller than the official TabICLv2 prior, but keeps
its central idea: sample diverse directed graphs and diverse parent-to-child
functions rather than training on one fixed synthetic mechanism.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PriorBatch:
    x: Tensor
    y_train: Tensor
    y_test: Tensor
    n_classes: int | None = None


class GraphPrior:
    """Generate batched classification or regression tasks on demand."""

    FUNCTION_NAMES = ("linear", "mlp", "quadratic", "product", "maximum", "step", "rbf", "sine")

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def _standardize(values: Tensor) -> Tensor:
        return (values - values.mean(dim=0, keepdim=True)) / values.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-5)

    def _random_function(self, parents: Tensor) -> Tensor:
        """Map ``(rows, parents)`` to one synthetic scalar graph node."""
        kind = self.rng.choice(self.FUNCTION_NAMES)
        width = parents.shape[1]
        weights = torch.randn(width, device=parents.device, dtype=parents.dtype) / width**0.5
        projection = parents @ weights

        if kind == "linear":
            output = projection
        elif kind == "mlp":
            hidden = self.rng.randint(2, 8)
            first = torch.randn(width, hidden, device=parents.device, dtype=parents.dtype) / width**0.5
            second = torch.randn(hidden, device=parents.device, dtype=parents.dtype) / hidden**0.5
            output = torch.tanh(parents @ first) @ second
        elif kind == "quadratic":
            output = projection.square() * self.rng.choice((-1.0, 1.0)) + 0.25 * projection
        elif kind == "product":
            scales = torch.randn(width, device=parents.device, dtype=parents.dtype)
            output = torch.tanh(parents * scales).prod(dim=1)
        elif kind == "maximum":
            output = (parents * weights).amax(dim=1)
        elif kind == "step":
            output = (projection > 0).to(parents.dtype) + 0.5 * (projection > 1).to(parents.dtype)
        elif kind == "rbf":
            center = torch.randn(width, device=parents.device, dtype=parents.dtype)
            bandwidth = 0.5 + 2.0 * self.rng.random()
            output = torch.exp(-((parents - center).square().mean(dim=1)) / bandwidth)
        else:
            frequency = 0.5 + 2.5 * self.rng.random()
            output = torch.sin(frequency * projection)

        output = torch.nan_to_num(output)
        output = (output - output.mean()) / output.std(unbiased=False).clamp_min(1e-5)
        return output * (0.3 + 2.7 * self.rng.random())

    def _make_categorical(self, values: Tensor) -> Tensor:
        bins = self.rng.randint(2, min(8, max(2, values.numel() // 4)))
        sorted_values = values.sort().values
        ranks = torch.linspace(0, values.numel() - 1, bins + 1, device=values.device)[1:-1].long()
        thresholds = sorted_values[ranks]
        return (values[:, None] > thresholds[None, :]).sum(dim=1).to(values.dtype)

    def _sample_one(self, rows: int, features: int, device: torch.device) -> tuple[Tensor, Tensor]:
        root_count = self.rng.randint(2, min(6, max(2, features)))
        nodes = [torch.randn(rows, device=device) for _ in range(root_count)]
        required_nodes = features + self.rng.randint(2, 5)

        while len(nodes) < root_count + required_nodes:
            max_parents = min(len(nodes), 6)
            # A heavy-tailed parent count is a compact analogue of the paper's
            # random-Cauchy graph connectivity.
            cauchy_like = abs(math_tan_pi(self.rng.random() - 0.5))
            parent_count = min(max_parents, 1 + int(cauchy_like))
            parent_indices = self.rng.sample(range(len(nodes)), parent_count)
            parents = torch.stack([nodes[index] for index in parent_indices], dim=1)
            nodes.append(self._random_function(parents))

        candidate_nodes = nodes[root_count:]
        feature_indices = self.rng.sample(range(len(candidate_nodes)), features)
        x = torch.stack([candidate_nodes[index] for index in feature_indices], dim=1)
        for column in range(features):
            if self.rng.random() < 0.18:
                x[:, column] = self._make_categorical(x[:, column])

        target_parent_count = self.rng.randint(1, min(features, 6))
        target_indices = self.rng.sample(range(features), target_parent_count)
        target = self._random_function(x[:, target_indices])
        if self.rng.random() < 0.35:
            target = target + torch.randn_like(target) * (0.02 + 0.18 * self.rng.random())

        permutation = torch.randperm(features, device=device)
        return torch.nan_to_num(x[:, permutation]), torch.nan_to_num(target)

    @staticmethod
    def _balanced_classes(scores: Tensor, n_classes: int) -> Tensor:
        order = scores.argsort()
        labels_by_rank = torch.arange(scores.numel(), device=scores.device) * n_classes // scores.numel()
        labels = torch.empty_like(labels_by_rank)
        labels[order] = labels_by_rank
        class_permutation = torch.randperm(n_classes, device=scores.device)
        return class_permutation[labels]

    def sample(
        self,
        *,
        batch_size: int,
        train_rows: int,
        test_rows: int,
        features: int,
        task: str,
        max_classes: int,
        device: torch.device,
    ) -> PriorBatch:
        rows = train_rows + test_rows
        n_classes = self.rng.randint(2, min(max_classes, 6)) if task == "classification" else None
        x_items: list[Tensor] = []
        y_items: list[Tensor] = []

        for _ in range(batch_size):
            x, target = self._sample_one(rows, features, device)
            if task == "classification":
                assert n_classes is not None
                target = self._balanced_classes(target, n_classes)
            else:
                target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1e-5)
            x_items.append(x)
            y_items.append(target)

        x_batch = torch.stack(x_items)
        y_batch = torch.stack(y_items)
        return PriorBatch(
            x=x_batch,
            y_train=y_batch[:, :train_rows],
            y_test=y_batch[:, train_rows:],
            n_classes=n_classes,
        )


def math_tan_pi(value: float) -> float:
    # Isolated to keep the hot prior loop readable and deterministic via
    # ``random.Random`` rather than torch's global RNG.
    import math

    return math.tan(math.pi * value)
