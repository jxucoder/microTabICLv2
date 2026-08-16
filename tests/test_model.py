import torch

from microtabiclv2.config import ModelConfig
from microtabiclv2.model import MicroTabICLv2, MPSafeLinear
from microtabiclv2.train import pinball_loss


def tiny_config(task: str = "classification") -> ModelConfig:
    return ModelConfig(
        task=task,
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
        num_quantiles=7,
    ).validate()


def test_classification_forward_and_backward():
    model = MicroTabICLv2(tiny_config())
    x = torch.randn(2, 12, 5)
    y = torch.randint(0, 3, (2, 8))
    output = model(x, y)
    assert output.shape == (2, 4, 10)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_regression_forward_and_pinball_loss():
    model = MicroTabICLv2(tiny_config("regression"))
    x = torch.randn(2, 11, 6)
    y_train = torch.randn(2, 7)
    y_test = torch.randn(2, 4)
    output = model(x, y_train)
    assert output.shape == (2, 4, 7)
    loss = pinball_loss(output, y_test)
    assert loss.ndim == 0
    assert loss > 0


def test_mps_safe_linear_matches_torch_linear():
    regular = torch.nn.Linear(5, 3)
    safe = MPSafeLinear(5, 3)
    safe.load_state_dict(regular.state_dict())
    inputs = torch.randn(2, 4, 5)
    torch.testing.assert_close(safe(inputs), regular(inputs))


def test_feature_grouping_uses_zero_one_three_shifts():
    model = MicroTabICLv2(tiny_config())
    x = torch.arange(5.0).reshape(1, 1, 5)
    grouped = model._group_features(x)
    assert grouped.shape == (1, 1, 5, 3)
    torch.testing.assert_close(grouped[0, 0, 0], torch.tensor([0.0, 1.0, 3.0]))
