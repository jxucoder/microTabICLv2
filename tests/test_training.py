from pathlib import Path

import numpy as np

from microtabiclv2 import (
    MicroTabICLv2Classifier,
    get_model_config,
    load_checkpoint,
    train_model,
)
from microtabiclv2.config import ModelConfig, TrainConfig


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        embed_dim=16,
        col_blocks=1,
        row_blocks=1,
        icl_blocks=1,
        col_heads=4,
        row_heads=4,
        icl_heads=4,
        inducing_tokens=4,
        row_cls_tokens=2,
        qass_hidden_dim=8,
    )


def test_one_training_step_and_checkpoint_round_trip(tmp_path: Path):
    checkpoint = tmp_path / "tiny.pt"
    train_config = TrainConfig(
        steps=1,
        batch_size=2,
        min_train_rows=8,
        max_train_rows=8,
        test_rows=3,
        min_features=4,
        max_features=4,
        device="cpu",
        log_every=1,
        output=str(checkpoint),
    )
    result = train_model(tiny_model_config(), train_config, quiet=True)
    assert result.step == 1
    assert result.final_loss > 0
    assert checkpoint.exists()

    loaded, metadata = load_checkpoint(checkpoint)
    assert loaded.config == result.model.config
    assert metadata["step"] == 1


def test_classifier_wrapper_shapes():
    from microtabiclv2.model import MicroTabICLv2

    rng = np.random.default_rng(42)
    x = rng.normal(size=(18, 4)).astype(np.float32)
    y = np.array(["a", "b", "c"] * 6)
    classifier = MicroTabICLv2Classifier(MicroTabICLv2(tiny_model_config()), device="cpu")
    classifier.fit(x[:12], y[:12])
    probabilities = classifier.predict_proba(x[12:])
    assert probabilities.shape == (6, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert classifier.predict(x[12:]).shape == (6,)


def test_training_step_callback():
    seen = []
    config = TrainConfig(
        steps=1,
        batch_size=1,
        min_train_rows=6,
        max_train_rows=6,
        test_rows=2,
        min_features=4,
        max_features=4,
        device="cpu",
        output="",
    )
    def predict_during_training(step, model):
        seen.append(step)
        x = np.arange(32, dtype=np.float32).reshape(8, 4)
        MicroTabICLv2Classifier(model, device="cpu").fit(x[:6], [0, 1, 0, 1, 0, 1]).predict(x[6:])

    train_model(tiny_model_config(), config, quiet=True, on_step=predict_during_training)
    assert seen == [0, 1]


def test_paper_profile_matches_appendix_a4():
    config = get_model_config("paper", task="regression")
    assert config.embed_dim == 128
    assert (config.col_blocks, config.row_blocks, config.icl_blocks) == (3, 3, 12)
    assert config.inducing_tokens == 128
    assert config.row_cls_tokens == 4
    assert config.num_quantiles == 999
