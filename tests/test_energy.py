import torch

from morphovoxel.ecology.energy import transport_energy, update_energy


def test_energy_costs_reduce_energy():
    shape = (1, 3, 3, 3)
    energy = torch.ones(shape)
    updated, _ = update_energy(energy, torch.ones(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape), maintenance_cost=0.1)
    assert torch.all(updated < energy)
    localized = torch.zeros(1, 3, 3, 3)
    localized[0, 1, 1, 1] = 1
    transported = transport_energy(localized, torch.ones_like(localized, dtype=torch.bool), 0.1)
    assert torch.isclose(transported.sum(), localized.sum()) and transported[0, 1, 1, 2] > 0
