"""Scale microTabICLv2 pretraining from a Mac to Modal GPUs.

Examples:
    modal run modal_app.py --profile small --steps 5000
    modal run --detach modal_app.py --profile paper --task regression
"""

from __future__ import annotations

import re

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "numpy>=1.26,<3",
        "scikit-learn>=1.5,<2",
    )
    .add_local_python_source("microtabiclv2")
)
app = modal.App("microtabiclv2", image=image)
checkpoint_volume = modal.Volume.from_name("microtabiclv2-checkpoints", create_if_missing=True)
CHECKPOINT_ROOT = "/checkpoints"


def _train(profile: str, task: str, steps: int, run_name: str) -> dict[str, object]:
    from microtabiclv2 import get_model_config, get_train_config, train_model

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, dot, underscore, and dash")
    overrides = {"output": f"{CHECKPOINT_ROOT}/{run_name}.pt"}
    if steps > 0:
        overrides["steps"] = steps
    model_config = get_model_config(profile, task)
    train_config = get_train_config(profile, **overrides)
    result = train_model(model_config, train_config)
    checkpoint_volume.commit()
    return {
        "checkpoint": result.checkpoint,
        "loss": result.final_loss,
        "elapsed_seconds": result.elapsed_seconds,
        "parameters": result.model.parameter_count(),
        "device": result.device,
    }


@app.function(
    gpu=["L40S", "A10", "T4"],
    timeout=24 * 60 * 60,
    volumes={CHECKPOINT_ROOT: checkpoint_volume},
)
def train_value_gpu(profile: str, task: str, steps: int, run_name: str) -> dict[str, object]:
    return _train(profile, task, steps, run_name)


@app.function(
    gpu=["H100", "A100-80GB"],
    timeout=24 * 60 * 60,
    volumes={CHECKPOINT_ROOT: checkpoint_volume},
)
def train_large_gpu(profile: str, task: str, steps: int, run_name: str) -> dict[str, object]:
    return _train(profile, task, steps, run_name)


@app.local_entrypoint()
def main(
    profile: str = "paper",
    task: str = "classification",
    steps: int = 0,
    run_name: str = "",
) -> None:
    if profile not in ("micro", "small", "paper", "large"):
        raise ValueError("profile must be micro, small, paper, or large")
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    run_name = run_name or f"{profile}-{task}"
    remote_function = train_large_gpu if profile in ("paper", "large") else train_value_gpu
    result = remote_function.remote(profile, task, steps, run_name)
    print(result)
    print(
        f"Download with:\nmodal volume get microtabiclv2-checkpoints {run_name}.pt checkpoints/{run_name}.pt"
    )
