import numpy as np
import torch

from microtabiclv2 import MODES, MicroTabICLv2, predict, prior


def test_one_file_model():
    model = MicroTabICLv2(**MODES["micro"])
    x, y_train, y_test = prior(2, 12, 5, 4, 2)
    logits = model(x, y_train)
    assert logits.shape == (2, 5, 6)
    torch.nn.functional.cross_entropy(logits[..., :2].flatten(0, 1), y_test.flatten()).backward()

    probability = predict(model, x[0, :12], y_train[0], x[0, 12:])
    assert probability.shape == (5, 2)
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
