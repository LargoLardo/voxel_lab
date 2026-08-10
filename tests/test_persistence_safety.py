import pytest
import torch

from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.training.losses import morphology_loss, stability_loss


@pytest.mark.parametrize(
    ("model", "shape", "center", "corner"),
    [
        (NeuralCA2D(3, 4, fire_rate=1), (1, 3, 9, 9), (4, 4), (0, 0)),
        (NeuralCA3D(3, 4, fire_rate=1), (1, 3, 7, 7, 7), (3, 3, 3), (0, 0, 0)),
    ],
)
def test_update_clears_every_channel_in_dead_cells(model, shape, center, corner):
    state = torch.zeros(shape)
    state[(0, 0, *center)] = 1
    state[(0, 2, *corner)] = 7
    with torch.no_grad():
        for parameter in model.update.parameters():
            parameter.zero_()

    updated = model(state)

    assert updated[(0, 0, *center)] == 1
    assert torch.count_nonzero(updated[(0, slice(None), *corner)]) == 0


@pytest.mark.parametrize(
    ("model", "shape", "center"),
    [
        (NeuralCA2D(3, 4, fire_rate=1), (1, 3, 7, 7), (3, 3)),
        (NeuralCA3D(3, 4, fire_rate=1), (1, 3, 7, 7, 7), (3, 3, 3)),
    ],
)
def test_update_clears_cells_that_die_during_the_step(model, shape, center):
    state = torch.zeros(shape)
    state[(0, 0, *center)] = 0.2
    with torch.no_grad():
        for parameter in model.update.parameters():
            parameter.zero_()
        model.update[-1].bias[0] = -0.2

    assert torch.count_nonzero(model(state)) == 0


def test_raw_occupancy_and_range_losses_have_gradients_outside_unit_interval():
    layout = StateLayout(materials=2, hidden=2)
    state = torch.zeros(1, layout.channels, 1, 2)
    state[:, 0] = torch.tensor([[[-2.0, 3.0]]])
    state.requires_grad_()
    target = torch.tensor([[[0.0, 1.0]]])
    material = torch.zeros_like(target, dtype=torch.long)
    weights = {"occupancy": 1, "occupancy_range": 1, "leakage": 0, "material": 0, "magnitude": 0}

    loss, components = morphology_loss(state, target, material, layout, weights)
    loss.backward()

    assert components["occupancy"].item() == pytest.approx(4)
    assert components["occupancy_range"].item() == pytest.approx(4)
    assert components["leakage"].item() >= 0
    assert state.grad[0, 0, 0, 0] < 0
    assert state.grad[0, 0, 0, 1] > 0


def test_magnitude_penalty_covers_occupancy_material_and_hidden_channels():
    layout = StateLayout(materials=2, hidden=2)
    state = torch.zeros(1, layout.channels, 1, 1)
    state[0, layout.occupancy] = 5
    state[0, layout.material_slice.start] = -5
    state[0, layout.hidden_slice.start] = 5
    target = torch.zeros(1, 1, 1)
    material = torch.zeros_like(target, dtype=torch.long)

    _, components = morphology_loss(state, target, material, layout, state_limit=4)

    assert components["magnitude"].item() == pytest.approx(3 / layout.channels)


def test_stability_loss_does_not_hide_out_of_range_drift():
    assert stability_loss(torch.tensor([[[[2.0]]]]), torch.tensor([[[[3.0]]]])).item() == 1
