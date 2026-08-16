"""A compact, trainable implementation of the TabICLv2 architecture."""

from .config import ModelConfig, TrainConfig, get_model_config, get_train_config
from .estimator import MicroTabICLv2Classifier, MicroTabICLv2Regressor
from .model import MicroTabICLv2
from .train import load_checkpoint, save_checkpoint, train_model

__all__ = [
    "MicroTabICLv2",
    "MicroTabICLv2Classifier",
    "MicroTabICLv2Regressor",
    "ModelConfig",
    "TrainConfig",
    "get_model_config",
    "get_train_config",
    "load_checkpoint",
    "save_checkpoint",
    "train_model",
]

__version__ = "0.1.0"
