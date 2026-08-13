import torch
import pytest

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


def test_pool_appends_a_cpu_backed_paired_suffix():
    def make_pool(ids, device="cpu"):
        ids = torch.tensor(ids, device=device)
        return StatePool(
            ids[:, None].float(), ids[:, None].float(), ids.long(),
            target_occupancy=ids[:, None].float(),
            target_materials=ids[:, None].long(),
            environments=ids[:, None].float(),
            environment_specs=ids[:, None].double(),
            style_seeds=ids.long(),
        )

    saved = make_pool([10, 11])
    initialized = make_pool([20, 21, 22, 23])
    saved.append_from(initialized, len(saved.states))

    for name, value in saved.state_dict().items():
        assert len(value) == 4, name
        assert value.device.type == "cpu", name
    assert saved.genomes[:, 0].tolist() == [10, 11, 22, 23]
    assert saved.style_seeds.tolist() == [10, 11, 22, 23]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pool_indices_can_select_reseed_entries_and_commit_to_cpu():
    pool = StatePool(torch.zeros(4, 2), torch.arange(4).view(4, 1).float())
    batch = pool.sample(2, torch.Generator().manual_seed(3), "cuda")
    reseed = torch.tensor([0], device="cuda")

    selected = batch.indices[reseed]
    pool.replace_entries(
        selected,
        states=torch.ones(1, 2, device="cuda"),
        genomes=torch.full((1, 1), 9.0, device="cuda"),
    )
    pool.commit(batch, batch.states + 2, 3)

    assert batch.indices.device.type == "cuda"
    assert pool.states.device.type == "cpu"
