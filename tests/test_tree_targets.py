import numpy as np
import pytest

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, make_environment_context
from morphovoxel.genomes import TreeGenome
from morphovoxel.targets import make_tree_target


def test_tree_genomes_make_distinct_reproducible_semantic_targets():
    short = TreeGenome(family="branching").with_values({"height": -1, "canopy_spread": -1})
    tall = TreeGenome(family="branching").with_values({"height": 1, "canopy_spread": 1})
    short_a = make_tree_target(short, 16)
    short_b = make_tree_target(short, 16)
    tall_target = make_tree_target(tall, 16)
    assert np.array_equal(short_a[0], short_b[0])
    assert np.array_equal(short_a[1], short_b[1])
    assert not np.array_equal(short_a[0], tall_target[0])
    assert short_a[0].sum() > 0 and tall_target[0].sum() > 0
    assert set(np.unique(tall_target[1])).issubset({0, 1, 2, 3})


def test_tree_targets_respond_to_environment_without_changing_genome():
    genome = TreeGenome.random(5, family="broad_canopy")
    calm = make_tree_target(genome, 16, EnvironmentSpec())
    windy = make_tree_target(
        genome, 16,
        EnvironmentSpec(light_direction_x=1, wind_direction_y=1, wind_strength=1, obstacle_density=0.15, seed=9),
    )
    assert not np.array_equal(calm[0], windy[0])


def test_tree_targets_supervise_resource_crowding_and_water_responses():
    genome = TreeGenome.random(8, family="branching")
    calm = make_tree_target(genome, 16, EnvironmentSpec())
    constrained = make_tree_target(
        genome,
        16,
        EnvironmentSpec(water_level=0.1, energy=0.1, neighbor_pressure=1.0),
    )
    assert constrained[0].sum() < calm[0].sum()

    west = make_tree_target(genome, 16, EnvironmentSpec(water_direction_x=-1))[1]
    east = make_tree_target(genome, 16, EnvironmentSpec(water_direction_x=1))[1]
    z, _, x = np.indices(west.shape)
    west_roots = (west == 1) & (z >= 13)
    east_roots = (east == 1) & (z >= 13)
    assert x[east_roots].mean() > x[west_roots].mean()

    open_target = make_tree_target(genome, 16, EnvironmentSpec(seed=33))[0]
    crowded_spec = EnvironmentSpec(neighbor_pressure=1, seed=33)
    crowded_target = make_tree_target(genome, 16, crowded_spec)[0]
    neighbor_field = make_environment_context(crowded_spec, 16)[
        ENVIRONMENT_CHANNELS.index("neighbor_occupancy")
    ].numpy() > 0.5
    assert neighbor_field.any()
    assert not crowded_target[neighbor_field].any()
    assert crowded_target.sum() < open_target.sum()


def test_style_seed_variation_uses_the_same_smooth_phase_as_the_model_input():
    left = TreeGenome(family="broad_canopy", style_seed=10_000)
    right = TreeGenome(family="broad_canopy", style_seed=1_000_000)
    left_style = left.model_vector()[-2:].numpy()
    assert left_style == pytest.approx([np.sin(left.style_phase), np.cos(left.style_phase)])
    assert not np.array_equal(make_tree_target(left, 16)[0], make_tree_target(right, 16)[0])
