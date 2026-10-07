import copy
from types import MethodType

import pytest
import torch
from torch.nn import functional as F

from morphovoxel.genomes import TREE_FAMILIES, TreeGenome, tree_genome_tensor
from morphovoxel.model_3d import NeuralCA3D, TreeFamilyNCA3D
from morphovoxel.random_utils import seed_everything
from morphovoxel.rollout import rollout
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.training.family import family_style_seeds
from morphovoxel.training.losses import _soft_overlap, morphology_loss, prepare_morphology_targets


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model_type,genome_size,context_channels,layers", [
    (NeuralCA3D, 0, 0, [8]),
    (NeuralCA3D, 15, 2, [8, 5]),
    (TreeFamilyNCA3D, 15, 2, [8, 5]),
])
def test_pointwise_layers_match_conv3d_through_rollout_and_backward(
    device, model_type, genome_size, context_channels, layers,
):
    seed_everything(123, deterministic=device != "mps")
    model = model_type(5, genome_size=genome_size, context_channels=context_channels, hidden_layers=layers).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .03)
    reference = copy.deepcopy(model)
    for layer in reference.modules():
        if isinstance(layer, torch.nn.Conv3d):
            layer.forward = MethodType(torch.nn.Conv3d.forward, layer)
    # Existing checkpoints retain identical parameter keys and shapes.
    reference.load_state_dict(model.state_dict(), strict=True)
    initial = torch.rand(4, 5, 3, 4, 5, device=device) + .2
    genomes = tree_genome_tensor([TreeGenome(family=f) for f in TREE_FAMILIES], device=device) if genome_size else None
    context = torch.rand(4, context_channels, 3, 4, 5, device=device) if context_channels else None
    with torch.inference_mode():
        rollout(model, initial, 2, genomes, context=context)
    for update in range(2):
        results, inputs = [], []
        for net in (model, reference):
            net.zero_grad(set_to_none=True)
            state = initial.clone().requires_grad_()
            g = genomes.clone().requires_grad_() if genomes is not None else None
            c = context.clone().requires_grad_() if context is not None else None
            seed_everything(7 + update, deterministic=device != "mps")
            grown, _ = rollout(net, state, 3, g, context=c, shared_fire_pairs=True)
            final, _ = rollout(net, grown, 2, g, context=c, shared_fire_pairs=True)
            loss = grown.square().mean() + final.square().mean()
            loss.backward()
            results.append(final.detach())
            inputs.append([value.grad for value in (state, g, c) if value is not None])
        torch.testing.assert_close(results[0], results[1], atol=2e-6, rtol=1e-5)
        for actual, expected in zip(inputs[0], inputs[1]):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, original.grad, atol=1e-7, rtol=1e-4)
        for net in (model, reference):
            torch.optim.SGD(net.parameters(), lr=.01).step()


@pytest.mark.parametrize("device", DEVICES)
def test_overlap_handles_mixed_empty_and_present_targets_with_useful_gradients(device):
    prediction = torch.tensor([[.2, .1], [.25, .75], [0., 0.]], device=device, requires_grad=True)
    target = torch.tensor([[0., 0.], [0., 1.], [0., 0.]], device=device)
    dice, iou = _soft_overlap(prediction, target)
    intersection = prediction[1, 1]
    expected_dice = (1 - (2 * intersection + 1e-6) / (prediction[1].sum() + 1 + 1e-6) + .15) / 3
    expected_iou = (1 - (intersection + 1e-6) / (prediction[1].sum() + 1 - intersection + 1e-6) + .15) / 3
    torch.testing.assert_close(dice, expected_dice)
    torch.testing.assert_close(iou, expected_iou)
    (dice + iou).backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(prediction.grad[0], torch.full((2,), 1 / 3, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [4, 16])
def test_absent_tree_material_penalizes_false_material_fraction_not_grid_size(device, size):
    layout = StateLayout(4, 1)
    target = torch.zeros(1, size, size, size, device=device)
    target[:, 1:3, 1, 1] = 1
    material = target.long()  # Trunk only; no branches or leaves.
    state = torch.zeros(1, layout.channels, size, size, size, device=device)
    state[:, 0] = target
    state[:, layout.material_slice.start + 1] = 8
    state.requires_grad_()
    _, terms = morphology_loss(state, target, material, layout)
    for name in ("trunk_dice", "branch_dice", "leaf_dice"):
        assert terms[name] < .001
    (terms["branch_dice"] + terms["leaf_dice"]).backward()
    for index in (2, 3):
        assert state.grad[:, layout.material_slice.start + index][target.bool()].min() > 0

    wrong = state.detach().clone()
    wrong[:, layout.material_slice.start + 2] = 16
    _, wrong_terms = morphology_loss(wrong, target, material, layout)
    assert wrong_terms["branch_dice"] > .99

    # Background material labels must not create semantic targets.
    material[target == 0] = 2
    prepared = prepare_morphology_targets(target, material)
    assert not prepared.material_masks[1].any()
    _, background_terms = morphology_loss(state, target, material, layout, prepared_targets=prepared)
    torch.testing.assert_close(background_terms["branch_dice"], terms["branch_dice"])
    empty = torch.zeros_like(target)
    _, empty_terms = morphology_loss(torch.zeros_like(state), empty, material, layout)
    for name in ("soft_dice", "soft_iou", "trunk_dice", "branch_dice", "leaf_dice"):
        assert empty_terms[name] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_transition_warmup_skips_mature_sources_and_preserves_pairs(device, monkeypatch):
    from morphovoxel.training import trainer

    calls = []
    def grow(model, state, steps, genomes, *, context, shared_fire_pairs):
        assert not torch.is_grad_enabled() and shared_fire_pairs
        calls.append((len(state), steps))
        torch.testing.assert_close(genomes[:, 0], state[:, 0, 0, 0, 0])
        torch.testing.assert_close(context[:, 0, 0, 0, 0], genomes[:, 0])
        return state + steps, []
    monkeypatch.setattr(trainer, "rollout", grow)
    ages = torch.tensor([0, 0, 3, 3, 8, 8, 12, 12], device=device)
    state = ages.float().reshape(8, 1, 1, 1, 1).requires_grad_()
    mature, updated_ages = trainer._mature_transition_sources(
        None, state, ages[:, None].float(), state, ages, 8, shared_fire_pairs=True,
    )
    assert calls == [(2, 5), (2, 8)]  # 26 organism-steps; old whole-batch warmup did 64.
    torch.testing.assert_close(updated_ages, ages.clamp_min(8))
    torch.testing.assert_close(mature.flatten(), updated_ages.float())
    torch.testing.assert_close(state.flatten(), ages.float())
    assert not mature.requires_grad
    trainer._mature_transition_sources(None, mature, ages[:, None], state, updated_ages, 8, shared_fire_pairs=True)
    assert len(calls) == 2


def test_default_basic_styles_produce_distinct_trees():
    seeds = family_style_seeds({})
    for family in TREE_FAMILIES:
        targets = [make_tree_target(TreeGenome(family=family, style_seed=seed), 16)[0] for seed in seeds]
        assert len({target.tobytes() for target in targets}) == len(seeds) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_cached_family_targets_are_reused_without_aliasing_or_changing_gradients(device):
    from morphovoxel.environment import EnvironmentSpec
    from morphovoxel.training import family
    from morphovoxel.training.losses import _distance_field

    family._cached_tree_target.cache_clear()
    family._cached_environment.cache_clear()
    options = dict(genome_span=0, style_seeds=[0], condition_ids=[0, 8, 16, 24], device=device)
    with torch.inference_mode():
        first = family.sample_counterfactual_family_data(4, 12, 17, **options)
    second = family.sample_counterfactual_family_data(4, 12, 18, **options)
    assert family._cached_tree_target.cache_info().misses == 4
    assert family._cached_tree_target.cache_info().hits == 4
    assert family._cached_environment.cache_info().misses == 1
    with torch.inference_mode():
        first.target_occupancy.zero_()
        first.target_materials.zero_()
        first.target_distances.zero_()
        first.environments.zero_()
    third = family.sample_counterfactual_family_data(4, 12, 19, **options)
    for name in ("target_occupancy", "target_materials", "target_distances", "environments"):
        torch.testing.assert_close(getattr(second, name), getattr(third, name))

    layout = StateLayout(4, 1)
    state = torch.randn(8, layout.channels, 12, 12, 12, device=device, requires_grad=True)
    original = state.detach().clone().requires_grad_()
    prepared = prepare_morphology_targets(second.target_occupancy, second.target_materials, distance=second.target_distances)
    loss, _ = morphology_loss(state, second.target_occupancy, second.target_materials, layout, prepared_targets=prepared)
    reference, _ = morphology_loss(original, second.target_occupancy, second.target_materials, layout)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(state.grad, original.grad)

    # Every target-defining input participates in the key, including style and size.
    base = TreeGenome()
    for genome, size, environment in (
        (base.with_values({"height": .5}), 12, EnvironmentSpec()),
        (TreeGenome(style_seed=970806), 12, EnvironmentSpec()),
        (base, 16, EnvironmentSpec()),
        (base, 12, EnvironmentSpec(wind_strength=1, wind_direction_x=1)),
    ):
        cached, material, distance = family._cached_tree_target(genome, size, environment)
        wanted, wanted_material = make_tree_target(genome, size, environment)
        torch.testing.assert_close(torch.from_numpy(cached), torch.from_numpy(wanted))
        torch.testing.assert_close(torch.from_numpy(material), torch.from_numpy(wanted_material))
        if distance is not None:
            torch.testing.assert_close(distance, _distance_field(torch.from_numpy(wanted)[None])[0])
    assert family._cached_tree_target.cache_info().misses == 8
    varied = family.sample_counterfactual_family_data(2, 12, 20, genome_span=.6, environment_span=.5, device=device)
    torch.testing.assert_close(varied.target_distances, _distance_field(varied.target_occupancy))


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_damage_preserves_reseeding_without_ranking_pair_errors(device, monkeypatch):
    from morphovoxel.training.trainer import _paired_pool_actions, _pool_actions

    state = torch.ones(8, 1, 3, 3, 3, device=device)
    target = torch.ones(8, 3, 3, 3, device=device)
    state[2] = 0  # A dead entry must still force both members of its pair to reset.
    ages = torch.tensor([30, 30, 10, 10, 80, 80, 20, 20], device=device)
    expected, damage = _paired_pool_actions(state, target, ages, 1, 1, 15)
    assert len(damage) > 0
    regular, _ = _pool_actions(state, target, ages, 1, 1, 15)
    actual_regular, empty = _pool_actions(state, target, ages, 1, 0, 15)
    torch.testing.assert_close(actual_regular, regular)
    assert len(empty) == 0

    def unexpected_error_ranking(*args, **kwargs):
        pytest.fail("disabled paired damage must not compute error rankings")
    monkeypatch.setattr(torch, "nan_to_num", unexpected_error_ranking)
    actual, empty = _paired_pool_actions(state, target, ages, 1, 0, 15)
    torch.testing.assert_close(actual, expected)
    assert actual.tolist() == [2, 3, 4, 5] and len(empty) == 0


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
