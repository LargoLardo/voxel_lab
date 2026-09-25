import copy

import pytest
import torch
from torch.nn import functional as F

from morphovoxel.genomes import TREE_FAMILIES, TreeGenome, tree_genome_tensor
from morphovoxel.model_3d import TreeFamilyNCA3D
from morphovoxel.random_utils import seed_everything
from morphovoxel.rollout import rollout
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.training.family import family_style_seeds
from morphovoxel.training.losses import morphology_loss, prepare_morphology_targets


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
))]


def test_default_basic_styles_produce_distinct_trees():
    seeds = family_style_seeds({})
    for family in TREE_FAMILIES:
        targets = [make_tree_target(TreeGenome(family=family, style_seed=seed), 16)[0] for seed in seeds]
        assert len({target.tobytes() for target in targets}) == len(seeds) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_prepared_rollout_preserves_gradients_and_refreshes_after_updates(device, monkeypatch):
    seed_everything(123, deterministic=device != "mps")
    model = TreeFamilyNCA3D(5, 8, context_channels=2).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.03)
    reference = copy.deepcopy(model)
    initial = torch.rand(4, 5, 5, 5, 5, device=device)
    context = torch.rand(4, 2, 5, 5, 5, device=device)
    calls = []
    prepare = model.prepare_genome

    def counted(genomes):
        calls.append(1)
        return prepare(genomes)

    monkeypatch.setattr(model, "prepare_genome", counted)
    for update in range(2):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        genomes = tree_genome_tensor([TreeGenome.random(update * 10 + i, family=f) for i, f in enumerate(TREE_FAMILIES)], device=device).requires_grad_()
        original_genomes = genomes.detach().clone().requires_grad_()
        state = initial.clone().requires_grad_()
        original_state = initial.clone().requires_grad_()
        seed_everything(7, deterministic=device != "mps")
        actual, frames = rollout(model, state, 3, genomes, capture_every=2, context=lambda step, _: context + step * .01)
        seed_everything(7, deterministic=device != "mps")
        expected = original_state
        for step in range(3):
            expected = reference(expected, original_genomes, context + step * .01)
        torch.testing.assert_close(actual, expected)
        assert len(frames) == 3
        assert len(calls) == update + 1  # Once per rollout, not once per step.
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(state.grad, original_state.grad)
        torch.testing.assert_close(genomes.grad, original_genomes.grad, atol=1e-7, rtol=1e-4)
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, original.grad, atol=1e-7, rtol=1e-4)
        for net in (model, reference):
            torch.optim.SGD(net.parameters(), lr=.01).step()


@pytest.mark.parametrize("device", DEVICES)
def test_single_basic_trajectories_match_duplicate_pairs_through_persistence(device):
    seed_everything(321, deterministic=device != "mps")
    layout = StateLayout(4, 1)
    model = TreeFamilyNCA3D(layout.channels, 8).to(device)
    reference = copy.deepcopy(model)
    genomes = tree_genome_tensor([TreeGenome(family=f) for f in TREE_FAMILIES], device=device)
    state = torch.rand(4, layout.channels, 5, 5, 5, device=device)
    target = (torch.rand(4, 5, 5, 5, device=device) > .7).float()
    material = torch.randint(0, 4, target.shape, device=device)
    results = []
    for net, duplicate in ((model, False), (reference, True)):
        seed_everything(9, deterministic=device != "mps")
        x, g, t, m = (value.repeat_interleave(2, 0) if duplicate else value for value in (state, genomes, target, material))
        prepared = prepare_morphology_targets(t, m)
        grown, _ = rollout(net, x, 3, g, shared_fire_pairs=duplicate)
        loss, _ = morphology_loss(grown, t, m, layout, prepared_targets=prepared)
        final, _ = rollout(net, grown, 2, g, shared_fire_pairs=duplicate)
        persistence, _ = morphology_loss(final, t, m, layout, prepared_targets=prepared)
        (loss + persistence).backward()
        results.append((final[::2] if duplicate else final, loss + persistence))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
    for parameter, original in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, original.grad, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spatial", [(4, 5), (3, 4, 5)])
def test_prepared_losses_and_dense_material_mask_match_uncached_and_sparse_reference(device, spatial):
    seed_everything(11, deterministic=device != "mps")
    layout = StateLayout(4, 2)
    target = (torch.rand(3, *spatial, device=device) > .8).float()
    target[0] = 0
    target[1] = 1
    material = torch.randint(0, 4, target.shape, device=device)
    material[target == 0] = -1  # Background labels must never enter cross entropy.
    prepared = prepare_morphology_targets(target, material)
    for _ in range(2):  # Reuse constants with different predictions/backward graphs.
        state = torch.randn(3, layout.channels, *spatial, device=device, requires_grad=True)
        original = state.detach().clone().requires_grad_()
        actual, components = morphology_loss(state, target, material, layout, prepared_targets=prepared)
        expected, _ = morphology_loss(original, target, material, layout)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(state.grad, original.grad)
        logits = state.detach()[:, layout.material_slice].clone().requires_grad_()
        sparse_logits = logits.detach().clone().requires_grad_()
        occupied = target > .5
        dense = F.cross_entropy(logits, prepared.material_labels, reduction="sum") / occupied.sum()
        sparse = F.cross_entropy(sparse_logits.movedim(1, -1)[occupied], material[occupied])
        torch.testing.assert_close(components["material"], sparse)
        dense.backward()
        sparse.backward()
        torch.testing.assert_close(logits.grad, sparse_logits.grad)
    empty_target = torch.zeros_like(target)
    empty, terms = morphology_loss(state.detach().requires_grad_(), empty_target, torch.full_like(material, -1), layout)
    assert torch.isfinite(empty) and terms["material"] == 0
    empty.backward()
