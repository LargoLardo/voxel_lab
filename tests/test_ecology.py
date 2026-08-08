import torch

from morphovoxel.ecology.environment import EcologyWorld
from morphovoxel.ecology.metrics import ecology_metrics


def test_identities_and_per_organism_metrics_are_separate():
    states = torch.zeros(2, 3, 5, 5, 5)
    states[0, 0, 2, 2, 1] = 1
    states[1, 0, 2, 2, 3] = 1
    field = torch.zeros(5, 5, 5)
    world = EcologyWorld(states, torch.eye(2), field.bool(), field, field, torch.ones(2, 5, 5, 5))
    assert world.ownership[2, 2, 1] == 1 and world.ownership[2, 2, 3] == 2
    assert [row["organism"] for row in ecology_metrics(world)] == [0, 1]

