"""Synthetic pretraining, checkpointing, and losses."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ModelConfig, TrainConfig
from .device import autocast_context, resolve_device
from .model import MicroTabICLv2
from .optim import Muon
from .prior import GraphPrior, PriorBatch


@dataclass
class TrainingResult:
    model: MicroTabICLv2
    final_loss: float
    step: int
    elapsed_seconds: float
    device: str
    checkpoint: str | None


def quantile_levels(count: int, *, device: torch.device | None = None) -> Tensor:
    return torch.arange(1, count + 1, device=device, dtype=torch.float32) / (count + 1)


def pinball_loss(predicted_quantiles: Tensor, targets: Tensor) -> Tensor:
    levels = quantile_levels(predicted_quantiles.shape[-1], device=predicted_quantiles.device)
    errors = targets.unsqueeze(-1) - predicted_quantiles
    return torch.maximum(levels * errors, (levels - 1) * errors).mean()


def task_loss(logits: Tensor, batch: PriorBatch, config: ModelConfig) -> Tensor:
    if config.task == "classification":
        assert batch.n_classes is not None
        return torch.nn.functional.cross_entropy(
            logits[..., : batch.n_classes].reshape(-1, batch.n_classes),
            batch.y_test.reshape(-1).long(),
        )
    return pinball_loss(logits, batch.y_test)


def _make_optimizer(model: MicroTabICLv2, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "muon":
        return Muon(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )


def _make_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    warmup_steps = max(1, int(config.steps * config.warmup_fraction))
    minimum_ratio = config.min_learning_rate / config.learning_rate

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(config.steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def save_checkpoint(
    path: str | Path,
    model: MicroTabICLv2,
    *,
    train_config: TrainConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "step": step,
        "loss": loss,
        "train_config": None if train_config is None else asdict(train_config),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MicroTabICLv2, dict[str, Any]]:
    map_location = resolve_device(device) if isinstance(device, str) else device
    payload = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=True)
    model = MicroTabICLv2(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    metadata = {key: value for key, value in payload.items() if key not in ("model_state", "optimizer_state")}
    if "optimizer_state" in payload:
        metadata["optimizer_state"] = payload["optimizer_state"]
    return model, metadata


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    resume: str | Path | None = None,
    quiet: bool = False,
    on_step: Callable[[int, MicroTabICLv2], None] | None = None,
) -> TrainingResult:
    model_config.validate()
    train_config.validate()
    _seed_everything(train_config.seed)
    device = resolve_device(train_config.device)
    prior = GraphPrior(train_config.seed)

    start_step = 0
    if resume is None:
        model = MicroTabICLv2(model_config).to(device)
        resume_metadata: dict[str, Any] = {}
    else:
        model, resume_metadata = load_checkpoint(resume, device=device)
        if model.config != model_config:
            raise ValueError("Checkpoint model configuration does not match the requested configuration")
        start_step = int(resume_metadata.get("step", 0))

    optimizer = _make_optimizer(model, train_config)
    if state := resume_metadata.get("optimizer_state"):
        optimizer.load_state_dict(state)
    scheduler = _make_scheduler(optimizer, train_config)
    for _ in range(start_step):
        scheduler.step()

    scale_gradients = train_config.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scale_gradients)
    except (AttributeError, TypeError):  # PyTorch 2.2 on Intel macOS
        scaler = torch.cuda.amp.GradScaler(enabled=scale_gradients)
    started = time.perf_counter()
    final_loss = float("nan")
    saved_path: Path | None = None

    if not quiet:
        print(
            f"Training {model_config.task} model with {model.parameter_count():,} parameters "
            f"on {device} for {train_config.steps:,} steps"
        )

    if on_step is not None:
        model.eval()
        on_step(start_step, model)
        model.train()

    for step in range(start_step + 1, train_config.steps + 1):
        train_rows = prior.rng.randint(train_config.min_train_rows, train_config.max_train_rows)
        features = prior.rng.randint(train_config.min_features, train_config.max_features)
        batch = prior.sample(
            batch_size=train_config.batch_size,
            train_rows=train_rows,
            test_rows=train_config.test_rows,
            features=features,
            task=model_config.task,
            max_classes=model_config.max_classes,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_config.amp):
            predictions = model(batch.x, batch.y_train)
            loss = task_loss(predictions, batch, model_config)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final_loss = float(loss.detach().cpu())

        should_log = step == 1 or step == train_config.steps or step % train_config.log_every == 0
        if should_log and not quiet:
            elapsed = time.perf_counter() - started
            print(
                f"step {step:>7,}/{train_config.steps:,} | loss {final_loss:.4f} | "
                f"rows {train_rows:>4} | features {features:>3} | {elapsed / step:.3f}s/step"
            )

        if on_step is not None:
            model.eval()
            on_step(step, model)
            model.train()

        if train_config.checkpoint_every and step % train_config.checkpoint_every == 0:
            base = Path(train_config.output)
            periodic = base.with_name(f"{base.stem}-step{step:07d}{base.suffix or '.pt'}")
            saved_path = save_checkpoint(
                periodic,
                model,
                train_config=train_config,
                optimizer=optimizer,
                step=step,
                loss=final_loss,
            )

    if train_config.output:
        saved_path = save_checkpoint(
            train_config.output,
            model,
            train_config=train_config,
            optimizer=optimizer,
            step=train_config.steps,
            loss=final_loss,
        )

    return TrainingResult(
        model=model,
        final_loss=final_loss,
        step=train_config.steps,
        elapsed_seconds=time.perf_counter() - started,
        device=str(device),
        checkpoint=None if saved_path is None else str(saved_path),
    )
