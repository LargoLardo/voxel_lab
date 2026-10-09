import pytest
import torch

from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.training.losses import _distance_field, morphology_loss, stability_loss


@pytest.mark.parametrize("shape", [(3, 5), (3, 4, 5)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
))])
def test_distance_field_matches_chebyshev_distance_including_empty_targets(shape, device):
    target = torch.zeros(3, *shape, device=device)
    target[(0, *(0 for _ in shape))] = 1
    target[1] = 1
    coordinates = torch.meshgrid(*(torch.arange(size, device=device) for size in shape), indexing="ij")
    expected = torch.zeros_like(target)
    expected[0] = torch.stack(coordinates).amax(0) / max(shape)
    # Fully occupied and empty targets both retain zero distance, as before.
    torch.testing.assert_close(_distance_field(target), expected)


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

    assert components["magnitude"].item() == pytest.approx(1 + 3 / layout.channels)


@pytest.mark.parametrize("size", [4, 16])
def test_sparse_state_spikes_keep_a_strong_gradient(size):
    layout = StateLayout(materials=2, hidden=2)
    state = torch.zeros(2, layout.channels, size, size)
    state[0, -1, 0, 0] = 6
    state.requires_grad_()
    target = torch.zeros(2, size, size)
    _, components = morphology_loss(state, target, target.long(), layout)
    components["magnitude"].backward()
    assert components["magnitude"] >= 2
    assert state.grad[0, -1, 0, 0] >= 2
    assert not state.grad[1].any()
    _, bounded = morphology_loss(state.detach().clamp(-4, 4), target, target.long(), layout)
    assert bounded["magnitude"] == 0


def test_sparse_foreground_is_not_diluted_by_empty_background():
    layout = StateLayout(materials=2, hidden=1)
    state = torch.zeros(1, layout.channels, 1, 100)
    target = torch.zeros(1, 1, 100)
    target[0, 0, 0] = 1
    material = torch.zeros_like(target, dtype=torch.long)

    _, components = morphology_loss(state, target, material, layout)

    assert components["occupancy"].item() == pytest.approx(0.5)


@pytest.mark.parametrize("shape", [(4, 4), (16, 16), (4, 4, 4), (16, 16, 16)])
def test_optional_occupancy_peak_penalty_keeps_sparse_gradients_without_changing_default(shape):
    layout = StateLayout(materials=2, hidden=1)
    state = torch.zeros(2, layout.channels, *shape)
    origin = (0,) * len(shape)
    state[(0, 0, *origin)] = -.5
    state[(1, 0, *origin)] = 1.25
    state.requires_grad_()
    target = torch.zeros(2, *shape)
    default, components = morphology_loss(state, target, target.long(), layout)
    disabled, _ = morphology_loss(state, target, target.long(), layout, {"occupancy_range_peak": 0.})
    torch.testing.assert_close(default, disabled)
    assert "occupancy_range_peak" not in components
    enabled, components = morphology_loss(state, target, target.long(), layout, {"occupancy_range_peak": .1})
    peak = components["occupancy_range_peak"]
    assert peak.item() == pytest.approx((.5 ** 2 + .25 ** 2) / 2)
    torch.testing.assert_close(enabled - default, .1 * peak)
    peak.backward()
    assert state.grad[(0, 0, *origin)] == -.5
    assert state.grad[(1, 0, *origin)] == .25
    assert torch.count_nonzero(state.grad) == 2


def test_stability_loss_does_not_hide_out_of_range_drift():
    assert stability_loss(torch.tensor([[[[2.0]]]]), torch.tensor([[[[3.0]]]])).item() == 1
