"""Small scikit-learn compatible inference wrappers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .device import autocast_context, resolve_device
from .model import MicroTabICLv2


def _numeric_2d_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _chunks(array: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(array), size):
        yield array[start : start + size]


class _BaseMicroEstimator(BaseEstimator):
    def __init__(
        self,
        model: MicroTabICLv2,
        *,
        device: str = "auto",
        amp: bool = False,
        predict_batch_size: int = 256,
    ) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.predict_batch_size = predict_batch_size

    def _fit_x(self, x) -> None:
        if self.predict_batch_size < 1:
            raise ValueError("predict_batch_size must be positive")
        self.device_ = resolve_device(self.device)
        self.model.to(self.device_).eval()
        x_array = _numeric_2d_array(x, "X")
        self.n_features_in_ = x_array.shape[1]
        self.x_train_ = torch.from_numpy(x_array).to(self.device_)

    def _predict_raw(self, x) -> np.ndarray:
        x_array = _numeric_2d_array(x, "X")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {x_array.shape[1]}")
        outputs = []
        # no_grad keeps prediction cheap while still allowing this model to be
        # trained later; inference_mode would turn the lazy RoPE cache into
        # inference tensors that autograd cannot reuse.
        with torch.no_grad():
            for chunk in _chunks(x_array, self.predict_batch_size):
                x_test = torch.from_numpy(chunk).to(self.device_)
                context = torch.cat((self.x_train_, x_test)).unsqueeze(0)
                with autocast_context(self.device_, self.amp):
                    output = self.model(context, self.y_train_.unsqueeze(0))
                outputs.append(output.squeeze(0).float().cpu().numpy())
        return np.concatenate(outputs)


class MicroTabICLv2Classifier(ClassifierMixin, _BaseMicroEstimator):
    def fit(self, x, y):
        if self.model.config.task != "classification":
            raise ValueError("Classifier requires a classification checkpoint")
        self._fit_x(x)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or len(y_array) != len(self.x_train_):
            raise ValueError("y must be a 1D array aligned with X")
        self.classes_, encoded = np.unique(y_array, return_inverse=True)
        if len(self.classes_) > self.model.config.max_classes:
            limit = self.model.config.max_classes
            raise ValueError(f"This checkpoint supports at most {limit} classes; got {len(self.classes_)}")
        self.y_train_ = torch.from_numpy(encoded).long().to(self.device_)
        return self

    def predict_proba(self, x) -> np.ndarray:
        logits = self._predict_raw(x)[:, : len(self.classes_)]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, x) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class MicroTabICLv2Regressor(RegressorMixin, _BaseMicroEstimator):
    def fit(self, x, y):
        if self.model.config.task != "regression":
            raise ValueError("Regressor requires a regression checkpoint")
        self._fit_x(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim != 1 or len(y_array) != len(self.x_train_):
            raise ValueError("y must be a 1D array aligned with X")
        if not np.isfinite(y_array).all():
            raise ValueError("y contains NaN or infinite values")
        self.y_mean_ = float(y_array.mean())
        self.y_scale_ = float(y_array.std()) or 1.0
        standardized = (y_array - self.y_mean_) / self.y_scale_
        self.y_train_ = torch.from_numpy(standardized).to(self.device_)
        return self

    def predict_quantiles(self, x) -> np.ndarray:
        quantiles = np.sort(self._predict_raw(x), axis=1)
        return quantiles * self.y_scale_ + self.y_mean_

    def predict(self, x) -> np.ndarray:
        return self.predict_quantiles(x).mean(axis=1)
