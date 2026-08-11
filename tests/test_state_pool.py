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


def test_pool_keeps_tree_target_environment_and_style_identity_paired():
    count, size = 4, 5
    ids = torch.arange(count)
    states = ids[:, None, None, None, None].expand(count, 2, size, size, size).float().clone()
    genomes = ids[:, None].float()
    targets = ids[:, None, None, None].expand(count, size, size, size).float().clone()
    materials = ids[:, None, None, None].expand(count, size, size, size).long().clone()
    environments = ids[:, None, None, None, None].expand(count, 3, size, size, size).float().clone()
    specs = torch.stack((ids.double(), ids.double() + 10), 1)
    seeds = ids.long() + 100
    pool = StatePool(
        states, genomes, target_occupancy=targets, target_materials=materials,
        environments=environments, environment_specs=specs, style_seeds=seeds,
    )
    batch = pool.sample(3, torch.Generator().manual_seed(4))
    sampled_ids = batch.genomes[:, 0].long()
    assert torch.equal(batch.target_occupancy[:, 0, 0, 0].long(), sampled_ids)
    assert torch.equal(batch.target_materials[:, 0, 0, 0], sampled_ids)
    assert torch.equal(batch.environments[:, 0, 0, 0, 0].long(), sampled_ids)
    assert torch.equal(batch.style_seeds, sampled_ids + 100)

    index = torch.tensor([1])
    pool.replace_entries(
        index, states=torch.full_like(states[:1], 9), genomes=torch.tensor([[9.0]]),
        target_occupancy=torch.full_like(targets[:1], 9), target_materials=torch.full_like(materials[:1], 9),
        environments=torch.full_like(environments[:1], 9),
        environment_specs=torch.tensor([[9.0, 19.0]], dtype=torch.float64), style_seeds=torch.tensor([109]),
    )
    assert pool.genomes[1, 0] == 9 and pool.style_seeds[1] == 109 and pool.ages[1] == 0
