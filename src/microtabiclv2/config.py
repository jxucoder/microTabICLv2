"""Configuration profiles for laptop and cloud training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration.

    ``paper`` matches the dimensions in TabICLv2 Appendix A.4. The smaller
    profiles keep the same stages and attention patterns while reducing width,
    depth, inducing points, and output quantiles.
    """

    task: Task = "classification"
    max_classes: int = 10
    num_quantiles: int = 31
    embed_dim: int = 48
    col_blocks: int = 1
    row_blocks: int = 1
    icl_blocks: int = 2
    col_heads: int = 4
    row_heads: int = 4
    icl_heads: int = 4
    inducing_tokens: int = 16
    row_cls_tokens: int = 2
    feature_group_size: int = 3
    qass_hidden_dim: int = 64
    ff_factor: int = 2
    dropout: float = 0.0

    @property
    def icl_dim(self) -> int:
        return self.embed_dim * self.row_cls_tokens

    @property
    def out_dim(self) -> int:
        return self.max_classes if self.task == "classification" else self.num_quantiles

    def validate(self) -> ModelConfig:
        if self.task not in ("classification", "regression"):
            raise ValueError(f"Unknown task: {self.task!r}")
        positive = {
            "max_classes": self.max_classes,
            "num_quantiles": self.num_quantiles,
            "embed_dim": self.embed_dim,
            "col_blocks": self.col_blocks,
            "row_blocks": self.row_blocks,
            "icl_blocks": self.icl_blocks,
            "inducing_tokens": self.inducing_tokens,
            "row_cls_tokens": self.row_cls_tokens,
            "feature_group_size": self.feature_group_size,
        }
        if invalid := [name for name, value in positive.items() if value < 1]:
            raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
        if self.embed_dim % self.col_heads or self.embed_dim % self.row_heads:
            raise ValueError("embed_dim must be divisible by col_heads and row_heads")
        if self.icl_dim % self.icl_heads:
            raise ValueError("embed_dim * row_cls_tokens must be divisible by icl_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Synthetic pretraining configuration."""

    steps: int = 500
    batch_size: int = 4
    min_train_rows: int = 32
    max_train_rows: int = 96
    test_rows: int = 24
    min_features: int = 4
    max_features: int = 12
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    optimizer: Literal["adamw", "muon"] = "adamw"
    device: str = "auto"
    amp: bool = False
    seed: int = 42
    log_every: int = 25
    checkpoint_every: int = 0
    output: str = "checkpoints/microtabiclv2.pt"

    def validate(self) -> TrainConfig:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if not 1 <= self.min_train_rows <= self.max_train_rows:
            raise ValueError("Require 1 <= min_train_rows <= max_train_rows")
        if not 1 <= self.min_features <= self.max_features:
            raise ValueError("Require 1 <= min_features <= max_features")
        if self.test_rows < 1:
            raise ValueError("test_rows must be positive")
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MODEL_PROFILES: dict[str, ModelConfig] = {
    "micro": ModelConfig(),
    "small": ModelConfig(
        embed_dim=64,
        col_blocks=2,
        row_blocks=2,
        icl_blocks=4,
        inducing_tokens=32,
        row_cls_tokens=4,
        num_quantiles=63,
    ),
    "paper": ModelConfig(
        embed_dim=128,
        col_blocks=3,
        row_blocks=3,
        icl_blocks=12,
        col_heads=8,
        row_heads=8,
        icl_heads=8,
        inducing_tokens=128,
        row_cls_tokens=4,
        num_quantiles=999,
    ),
    "large": ModelConfig(
        embed_dim=192,
        col_blocks=4,
        row_blocks=4,
        icl_blocks=18,
        col_heads=8,
        row_heads=8,
        icl_heads=8,
        inducing_tokens=192,
        row_cls_tokens=4,
        num_quantiles=999,
    ),
}

_TRAIN_PROFILES: dict[str, TrainConfig] = {
    "micro": TrainConfig(),
    "small": TrainConfig(
        steps=5_000,
        batch_size=8,
        min_train_rows=64,
        max_train_rows=256,
        test_rows=64,
        min_features=4,
        max_features=32,
        amp=False,
        output="checkpoints/microtabiclv2-small.pt",
    ),
    "paper": TrainConfig(
        steps=100_000,
        batch_size=4,
        min_train_rows=400,
        max_train_rows=1_024,
        test_rows=128,
        min_features=4,
        max_features=100,
        learning_rate=2e-4,
        min_learning_rate=1e-6,
        optimizer="muon",
        device="cuda",
        amp=True,
        log_every=50,
        checkpoint_every=5_000,
        output="/checkpoints/microtabiclv2-paper.pt",
    ),
    "large": TrainConfig(
        steps=150_000,
        batch_size=1,
        min_train_rows=400,
        max_train_rows=2_048,
        test_rows=256,
        min_features=4,
        max_features=128,
        learning_rate=1.5e-4,
        min_learning_rate=1e-6,
        optimizer="muon",
        device="cuda",
        amp=True,
        log_every=50,
        checkpoint_every=5_000,
        output="/checkpoints/microtabiclv2-large.pt",
    ),
}


def profile_names() -> tuple[str, ...]:
    return tuple(_MODEL_PROFILES)


def get_model_config(profile: str = "micro", task: Task = "classification", **overrides: Any) -> ModelConfig:
    try:
        config = _MODEL_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown profile {profile!r}; choose from {profile_names()}") from error
    return replace(config, task=task, **overrides).validate()


def get_train_config(profile: str = "micro", **overrides: Any) -> TrainConfig:
    try:
        config = _TRAIN_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"Unknown profile {profile!r}; choose from {profile_names()}") from error
    return replace(config, **overrides).validate()
