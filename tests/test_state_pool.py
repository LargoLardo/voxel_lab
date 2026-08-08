import torch

from morphovoxel.training.state_pool import StatePool


def test_pool_preserves_genome_pairings():
    states = torch.arange(8).float().view(4, 2)
    genomes = torch.arange(4).view(4, 1)
    pool = StatePool(states, genomes)
    batch = pool.sample(3, torch.Generator().manual_seed(1))
    assert torch.equal(batch.states[:, 0] / 2, batch.genomes[:, 0])
    pool.commit(batch, batch.states + 10, 3)
    assert torch.equal(pool.genomes[batch.indices], batch.genomes)

