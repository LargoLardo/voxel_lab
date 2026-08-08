import torch

from morphovoxel.ecology.environment import EcologyWorld
from morphovoxel.ecology.simulator import ProceduralEcologyBaseline, ecology_step


class IdentityModel:
    def __call__(self, state, genome):
        return state.clone()


def test_ecology_step_keeps_identities_and_valid_resources():
    states = torch.zeros(2, 3, 6, 6, 6)
    states[0, 0, 3, 3, 2] = states[1, 0, 3, 3, 4] = 1
    substrate = torch.ones(6, 6, 6, dtype=torch.bool)
    world = EcologyWorld(states, torch.eye(2), substrate, torch.ones(6, 6, 6), torch.ones(6, 6, 6), torch.ones(2, 6, 6, 6))
    updated, flows = ecology_step(world, IdentityModel())
    assert updated.water.min() >= 0 and 0 <= updated.light.min() <= updated.light.max() <= 1
    assert updated.ownership[3, 3, 2] == 1 and updated.ownership[3, 3, 4] == 2
    assert len(flows["water_absorbed"]) == 2
    assert ProceduralEcologyBaseline()(states, torch.eye(2))[:, 0].sum() > states[:, 0].sum()
