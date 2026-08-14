import numpy as np
import torch

from morphovoxel.checkpointing import convert_specialist_to_family
from morphovoxel.environment import EnvironmentSpec
from morphovoxel.genomes import (
    ENVIRONMENT_GENE_NAMES,
    FAMILY_GENE_NAMES,
    TREE_FAMILIES,
    TREE_GENE_SPECS,
    TreeGenome,
)
from morphovoxel.model_3d import NeuralCA3D, TreeFamilyNCA3D
from morphovoxel.rollout import rollout
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.training.family import sample_counterfactual_family_data
from morphovoxel.training.losses import counterfactual_loss, morphology_loss
from morphovoxel.training.state_pool import StatePool


def _gene_descriptor(name, family, occupancy, materials):
    points = np.argwhere(occupancy)
    branches = np.argwhere(materials == 2)
    if name == "height":
        return float(np.ptp(points[:, 0]) + 1)
    if name == "trunk_thickness":
        return float(len(points))
    if name == "branch_density":
        return float((materials == 2).sum())
    if name == "branch_inclination":
        mean = float(np.argwhere(materials == 3)[:, 0].mean())
        return mean if family == "weeping" else -mean
    if name == "branch_length":
        return float(np.sqrt(((points[:, 1:] - 8) ** 2).sum(1)).max())
    if name == "canopy_spread":
        return float((materials == 3).sum())
    if name in {"asymmetry", "light_tropism"}:
        return float(points[:, 2].mean())
    return float((materials == 3).sum() / max(1, (materials == 1).sum()))


def test_gene_schema_has_no_taper_and_locks_tropism_to_environment_stage():
    assert "taper" not in {spec.name for spec in TREE_GENE_SPECS}
    assert ENVIRONMENT_GENE_NAMES == ("light_tropism",)
    assert "light_tropism" not in FAMILY_GENE_NAMES


def test_each_gene_has_a_visible_monotonic_target_descriptor():
    values = np.linspace(-1, 1, 5)
    for spec in TREE_GENE_SPECS:
        for family in TREE_FAMILIES:
            environment = EnvironmentSpec(light_direction_x=1.0) if spec.name == "light_tropism" else EnvironmentSpec()
            descriptors = []
            targets = []
            for value in values:
                target, materials = make_tree_target(
                    TreeGenome(family=family).with_values({spec.name: float(value)}), 16, environment,
                )
                targets.append(target)
                descriptors.append(_gene_descriptor(spec.name, family, target, materials))
            assert all(right >= left - 1e-6 for left, right in zip(descriptors, descriptors[1:])), (spec.name, family, descriptors)
            assert descriptors[-1] > descriptors[0]
            assert np.count_nonzero(targets[0] != targets[-1]) >= 16


def test_counterfactual_sampler_changes_exactly_one_gene_and_preserves_nuisance_inputs():
    data = sample_counterfactual_family_data(32, 16, 9, genome_span=0.6)
    assert set(data.condition_ids.tolist()) == set(range(len(TREE_FAMILIES) * len(FAMILY_GENE_NAMES)))
    for index in range(0, len(data.genomes), 2):
        low, high = data.genomes[index : index + 2]
        changed = [name for name in low.values if low.value(name) != high.value(name)]
        assert len(changed) == 1
        assert changed[0] in FAMILY_GENE_NAMES
        assert low.family == high.family and low.style_seed == high.style_seed
        assert data.environment_specs[index] == data.environment_specs[index + 1]
        assert torch.equal(data.environments[index], data.environments[index + 1])
        assert low.value("light_tropism") == high.value("light_tropism") == 0


def test_state_pool_cycles_complete_counterfactual_pairs():
    data = sample_counterfactual_family_data(32, 16, 4)
    states = torch.zeros(64, 2)
    pool = StatePool(states, data.model_genomes, condition_ids=data.condition_ids, pair_ids=data.pair_ids)
    first = pool.sample_stratified_pairs(8, 0)
    second = pool.sample_stratified_pairs(8, 4)
    assert torch.equal(first.pair_ids.view(-1, 2)[:, 0], first.pair_ids.view(-1, 2)[:, 1])
    assert set(first.condition_ids.tolist()).isdisjoint(set(second.condition_ids.tolist()))


def test_family_model_conversion_preserves_specialist_then_genes_receive_gradients():
    source = NeuralCA3D(5, hidden=8, fire_rate=1.0)
    family = TreeFamilyNCA3D(5, hidden=8, genome_size=TreeGenome.model_size(), fire_rate=1.0)
    convert_specialist_to_family(source, family)
    state = torch.zeros(4, 5, 7, 7, 7)
    state[:, 0, 3, 3, 3] = 1
    genomes = torch.stack([TreeGenome(family=name).model_vector() for name in TREE_FAMILIES])
    expected = source(state)
    actual = family(state, genomes)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.square().mean().backward()
    assert any(layer.weight.grad is not None and bool((layer.weight.grad != 0).any()) for layer in family.film)


def test_structural_and_counterfactual_losses_reject_a_blurry_average():
    layout = StateLayout(4, 1)
    target = torch.zeros(2, 7, 7, 7)
    target[0, 2:5, 3, 3] = 1
    target[1, 2:5, 2:5, 3] = 1
    materials = torch.zeros_like(target, dtype=torch.long)
    materials[target > 0] = 2
    exact = torch.zeros(2, layout.channels, 7, 7, 7)
    exact[:, 0] = target
    exact[:, layout.material_slice.start + 2] = 8
    blurry = exact.clone()
    blurry[:, 0] = 0.5
    weights = {name: 1.0 for name in (
        "soft_dice", "soft_iou", "distance", "height", "width", "volume", "centroid", "branch_distribution",
    )}
    exact_loss, components = morphology_loss(exact, target, materials, layout, weights)
    blurry_loss, _ = morphology_loss(blurry, target, materials, layout, weights)
    assert blurry_loss > exact_loss
    assert set(weights) <= set(components)
    assert counterfactual_loss(exact, target, layout) == 0
    assert counterfactual_loss(blurry, target, layout) > 0


def test_rollout_reuses_each_fire_mask_within_a_pair():
    class Recorder:
        fire_rate = 0.5

        def __init__(self):
            self.masks = []

        def __call__(self, state, genome=None, context=None, fire_mask=None):
            self.masks.append(fire_mask.clone())
            return state

    model = Recorder()
    rollout(model, torch.zeros(4, 1, 5, 5, 5), 3, shared_fire_pairs=True)
    assert all(torch.equal(mask[0], mask[1]) and torch.equal(mask[2], mask[3]) for mask in model.masks)
