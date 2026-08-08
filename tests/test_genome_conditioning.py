import torch

from morphovoxel.genomes import GenomeEncoder, one_hot_genomes
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.seeding import seed_state
from morphovoxel.state import StateLayout


def test_genomes_change_shared_model_updates():
    layout = StateLayout(materials=4, hidden=2)
    model = NeuralCA3D(layout.channels, 8, 4, fire_rate=1)
    state = seed_state(2, 10, layout, dimensions=3)
    genomes = one_hot_genomes(torch.tensor([0, 1]))
    output = model(state, genomes)
    assert output.shape == state.shape
    assert not torch.equal(output[0], output[1])
    assert GenomeEncoder(4, 3)(torch.tensor([0, 1])).shape == (2, 3)
