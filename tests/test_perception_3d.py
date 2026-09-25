import pytest
import torch
from torch.nn import functional as F

from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.perception_3d import _perception_weight, perceive_3d


def test_perception_shape_and_locality():
    state = torch.zeros(1, 2, 7, 7, 7)
    assert perceive_3d(state).shape == (1, 10, 7, 7, 7)
    model = NeuralCA3D(2, 8, fire_rate=1)
    state[:, 0, 3, 3, 3] = 1
    changed = (model(state) - model(torch.zeros_like(state))).abs().sum(1)[0]
    assert changed[:2].sum() == changed[5:].sum() == changed[:, :2].sum() == changed[:, 5:].sum() == 0


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float64),
    pytest.param("mps", torch.float32, marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
    )),
])
def test_cached_filters_match_stencil_and_allow_training_after_inference(device, dtype):
    _perception_weight.cache_clear()
    state = torch.randn(2, 3, 5, 5, 5, device=device, dtype=dtype)
    with torch.inference_mode():
        perceive_3d(state)
        _perception_weight(state.shape[1], state.device, state.dtype)
    state.requires_grad_()
    padded = F.pad(state, (1, 1, 1, 1, 1, 1))
    left, right = padded[:, :, 1:-1, 1:-1, :-2], padded[:, :, 1:-1, 1:-1, 2:]
    front, back = padded[:, :, 1:-1, :-2, 1:-1], padded[:, :, 1:-1, 2:, 1:-1]
    bottom, top = padded[:, :, :-2, 1:-1, 1:-1], padded[:, :, 2:, 1:-1, 1:-1]
    expected = torch.stack((
        state, (right - left) / 2, (back - front) / 2, (top - bottom) / 2,
        left + right + front + back + bottom + top - 6 * state,
    ), dim=2).flatten(1, 2)
    actual = perceive_3d(state)
    torch.testing.assert_close(actual, expected)
    convolution = F.conv3d(state, _perception_weight(state.shape[1], state.device, state.dtype), padding=1, groups=state.shape[1])
    torch.testing.assert_close(actual, convolution)
    actual_grad, = torch.autograd.grad(actual.square().sum(), state)
    expected_grad, = torch.autograd.grad(expected.square().sum(), state)
    torch.testing.assert_close(actual_grad, expected_grad)
    convolution_grad, = torch.autograd.grad(convolution.square().sum(), state)
    torch.testing.assert_close(actual_grad, convolution_grad)
