from pathlib import Path

import numpy as np

from microtabiclv2 import (
    MicroTabICLv2,
    MicroTabICLv2Classifier,
    load,
    rule_switching,
    sample_prior,
    save,
    train,
)


def tiny_model():
    return MicroTabICLv2(d=16, col=1, row=1, icl=1, inducing=4, cls=2, heads=4)


def test_prior_and_model_shapes():
    x, y_train, y_test = sample_prior(2, 8, 3, 4, 3)
    model = tiny_model()
    output = model(x, y_train)
    assert output.shape == (2, 3, 6)
    assert y_test.shape == (2, 3)
    output.square().mean().backward()


def test_one_training_step_and_checkpoint(tmp_path: Path):
    model, _ = train(tiny_model(), steps=1, batch=1, device="cpu")
    checkpoint = tmp_path / "tiny.pt"
    save(model, checkpoint)
    assert load(checkpoint).config == model.config


def test_classifier_and_rule_switching():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 4))
    y = np.array([0, 1] * 8)
    classifier = MicroTabICLv2Classifier(tiny_model(), "cpu").fit(x[:12], y[:12])
    probability = classifier.predict_proba(x[12:])
    assert probability.shape == (4, 2)
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
    scores, _ = rule_switching(tiny_model(), "cpu", repeats=1, grid_size=9)
    assert {"vertical", "correct", "shuffled"} <= scores.keys()
