import torch

from utils.neural_vehicle_appearance import (
    neural_vehicle_color,
    reliability_conditioned_parameters,
)


def _inputs(n=4, hidden=6, channels=5):
    torch.manual_seed(7)
    appearance_input = torch.randn(n, channels)
    w1 = torch.randn(n, hidden, channels, requires_grad=True)
    b1 = torch.randn(n, hidden, requires_grad=True)
    w2 = torch.randn(n, 3, hidden, requires_grad=True)
    b2 = torch.randn(n, 3, requires_grad=True)
    return appearance_input, w1, b1, w2, b2


def test_color_shape_range_and_gradients():
    appearance_input, w1, b1, w2, b2 = _inputs()
    color = neural_vehicle_color(appearance_input, w1, b1, w2, b2)
    assert color.shape == (4, 3)
    assert torch.all((color >= 0.0) & (color <= 1.0))

    color.sum().backward()
    for tensor in (w1, b1, w2, b2):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_zero_modulation_preserves_base_parameters():
    _, w1, b1, w2, _ = _inputs()
    adapter = torch.randn(4, 6)
    output_adapter = torch.randn(4, 3, 6)
    effective = reliability_conditioned_parameters(
        w1,
        b1,
        w2,
        context_gain=adapter,
        view_adapter=adapter,
        bias_adapter=adapter,
        hidden_gain=adapter,
        output_adapter=output_adapter,
        modulation=torch.zeros(4, 1),
    )
    for actual, expected in zip(effective, (w1, b1, w2)):
        assert torch.allclose(actual, expected)


def test_reliable_adapter_changes_the_response():
    appearance_input, w1, b1, w2, b2 = _inputs()
    adapter = torch.full((4, 6), 0.5, requires_grad=True)
    output_adapter = torch.full((4, 3, 6), 0.25, requires_grad=True)
    effective_w1, effective_b1, effective_w2 = reliability_conditioned_parameters(
        w1,
        b1,
        w2,
        context_gain=adapter,
        view_adapter=adapter,
        bias_adapter=adapter,
        hidden_gain=adapter,
        output_adapter=output_adapter,
        modulation=torch.ones(4, 1),
    )
    base = neural_vehicle_color(appearance_input, w1, b1, w2, b2)
    adapted = neural_vehicle_color(
        appearance_input, effective_w1, effective_b1, effective_w2, b2
    )
    assert not torch.allclose(base, adapted)

    adapted.sum().backward()
    assert adapter.grad is not None
    assert output_adapter.grad is not None

