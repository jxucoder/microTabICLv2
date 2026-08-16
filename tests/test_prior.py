import torch

from microtabiclv2.prior import GraphPrior


def test_classification_prior_shapes_and_balanced_labels():
    prior = GraphPrior(seed=7)
    batch = prior.sample(
        batch_size=3,
        train_rows=12,
        test_rows=5,
        features=6,
        task="classification",
        max_classes=10,
        device=torch.device("cpu"),
    )
    assert batch.x.shape == (3, 17, 6)
    assert batch.y_train.shape == (3, 12)
    assert batch.y_test.shape == (3, 5)
    assert batch.n_classes is not None
    assert batch.y_train.dtype == torch.int64
    assert batch.y_test.max() < batch.n_classes
    assert torch.isfinite(batch.x).all()


def test_regression_prior_is_finite():
    batch = GraphPrior(seed=11).sample(
        batch_size=2,
        train_rows=10,
        test_rows=4,
        features=5,
        task="regression",
        max_classes=10,
        device=torch.device("cpu"),
    )
    assert batch.n_classes is None
    assert batch.y_train.dtype == torch.float32
    assert torch.isfinite(batch.x).all()
    assert torch.isfinite(batch.y_train).all()
    assert torch.isfinite(batch.y_test).all()
