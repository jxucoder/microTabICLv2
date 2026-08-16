"""A compact single-device Muon optimizer with an AdamW fallback group."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _zeroth_power_newton_schulz(gradient: Tensor, steps: int = 5) -> Tensor:
    if gradient.ndim != 2:
        raise ValueError("Muon expects a matrix")
    transposed = gradient.shape[0] > gradient.shape[1]
    x = gradient.T if transposed else gradient
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon for matrix weights and AdamW for all remaining parameters."""

    def __init__(
        self,
        parameters,
        *,
        lr: float = 2e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        parameters = list(parameters)
        matrix_parameters = [parameter for parameter in parameters if parameter.ndim == 2]
        other_parameters = [parameter for parameter in parameters if parameter.ndim != 2]
        groups = [
            {"params": matrix_parameters, "use_muon": True},
            {"params": other_parameters, "use_muon": False},
        ]
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "ns_steps": ns_steps,
            "betas": adamw_betas,
            "eps": eps,
        }
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter.grad))
            momentum_buffer.mul_(group["momentum"]).add_(parameter.grad)
            update = parameter.grad.add(momentum_buffer, alpha=group["momentum"])
            update = _zeroth_power_newton_schulz(update, group["ns_steps"])
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(parameter.shape))
            parameter.add_(update, alpha=-adjusted_lr)

    def _step_adamw(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            step = state.setdefault("step", 0) + 1
            state["step"] = step
            average = state.setdefault("average", torch.zeros_like(parameter.grad))
            square_average = state.setdefault("square_average", torch.zeros_like(parameter.grad))
            average.lerp_(parameter.grad, 1 - beta1)
            square_average.lerp_(parameter.grad.square(), 1 - beta2)
            bias1 = 1 - beta1**step
            bias2 = 1 - beta2**step
            update = average / bias1 / (square_average.div(bias2).sqrt() + group["eps"])
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-group["lr"])
