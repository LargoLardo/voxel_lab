import numpy as np
import pytest

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, make_environment_context
from morphovoxel.genomes import TREE_FAMILIES, TreeGenome
from morphovoxel.targets import make_tree_target


@pytest.mark.parametrize("start,end", [
    ((4.5, 8.5, 8.5), (10.5, 8.5, 8.5)),
    ((2.5, 2.5, 2.5), (10.5, 10.5, 10.5)),
    ((10.5, 2.5, 8.5), (2.5, 10.5, 4.5)),
    ((-3., 8.5, 8.5), (19., 8.5, 8.5)),
    ((4.5, 8.5, 8.5), (4.5, 8.5, 8.5)),
])
def test_thin_segments_are_face_connected_at_fractional_coordinates(start, end):
    from morphovoxel.targets.targets_3d import _segment, _seed_component

    mask = np.zeros((16, 16, 16), dtype=bool)
    _segment(mask, start, end, .5, connected=True)
    assert mask.any()
    np.testing.assert_array_equal(_seed_component(mask, tuple(np.argwhere(mask)[0])), mask)
    reverse = np.zeros_like(mask)
    _segment(reverse, end, start, .5, connected=True)
    np.testing.assert_array_equal(reverse, mask)


def test_unobstructed_thin_trees_do_not_lose_disconnected_foliage(monkeypatch):
    from morphovoxel.targets import targets_3d

    cleanup = targets_3d._seed_component
    def check_connected(mask, seed):
        kept = cleanup(mask, seed)
        np.testing.assert_array_equal(kept, mask)
        return kept
    monkeypatch.setattr(targets_3d, "_seed_component", check_connected)
    for family in TREE_FAMILIES:
        for seed in range(500, 532):
            genome = TreeGenome.random(seed, family=family, locked=["light_tropism"])
            occupancy, materials = make_tree_target(genome, 16)
            assert occupancy.any() and (materials == 3).any()


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


@pytest.mark.parametrize("size", (12, 16))
@pytest.mark.parametrize("family", TREE_FAMILIES)
def test_thinnest_tree_target_contains_its_planted_base(family, size):
    genome = TreeGenome(family=family).with_values({"trunk_thickness": -1})
    occupancy, materials = make_tree_target(
        genome,
        size,
        EnvironmentSpec(obstacle_density=0.3, neighbor_pressure=1, wind_strength=1, seed=91),
    )
    base_seed = (size - 3, size // 2, size // 2)
    assert occupancy[base_seed] == 1
    assert materials[base_seed] == 1


def test_random_tree_targets_never_lose_their_planted_base():
    size = 16
    base_seed = (size - 3, size // 2, size // 2)
    for seed in range(256):
        family = TREE_FAMILIES[seed % len(TREE_FAMILIES)]
        genome = TreeGenome.random(seed, family=family)
        occupancy, _ = make_tree_target(genome, size, EnvironmentSpec.random(seed + 10_000))
        assert occupancy[base_seed] == 1, (seed, genome)


def test_tree_targets_respond_to_environment_without_changing_genome():
    genome = TreeGenome.random(5, family="broad_canopy")
    calm = make_tree_target(genome, 16, EnvironmentSpec())
    windy = make_tree_target(
        genome, 16,
        EnvironmentSpec(light_direction_x=1, wind_direction_y=1, wind_strength=1, obstacle_density=0.15, seed=9),
    )
    assert not np.array_equal(calm[0], windy[0])


@pytest.mark.parametrize("left,right", [
    (EnvironmentSpec(wind_direction_x=1, wind_strength=.5), EnvironmentSpec(wind_direction_x=.5, wind_strength=1)),
    (EnvironmentSpec(wind_direction_x=-.5, wind_direction_y=.5, wind_strength=.5),
     EnvironmentSpec(wind_direction_x=-.25, wind_direction_y=.25, wind_strength=1)),
    (EnvironmentSpec(), EnvironmentSpec(wind_strength=1)),
])
def test_identical_wind_inputs_require_identical_targets(left, right):
    assert np.array_equal(make_environment_context(left, 16), make_environment_context(right, 16))
    for family in TREE_FAMILIES:
        genome = TreeGenome(family=family)
        for a, b in zip(make_tree_target(genome, 16, left), make_tree_target(genome, 16, right)):
            assert np.array_equal(a, b)


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


def test_cached_ball_coordinates_preserve_rasterization_and_cannot_be_mutated(monkeypatch):
    from morphovoxel.targets import targets_3d

    original_ball = targets_3d._ball
    def uncached_ball(mask, z, y, x, radius):
        zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        mask[(zz - z)**2 + (yy - y)**2 + (xx - x)**2 <= radius**2] = True
    for size in (12, 16, 32):
        for index, family in enumerate(TREE_FAMILIES):
            genome = TreeGenome.random(index + 20, family=family)
            environment = EnvironmentSpec.random(index + 30)
            monkeypatch.setattr(targets_3d, '_ball', original_ball)
            cached = make_tree_target(genome, size, environment)
            monkeypatch.setattr(targets_3d, '_ball', uncached_ball)
            expected = make_tree_target(genome, size, environment)
            for actual, reference in zip(cached, expected):
                np.testing.assert_array_equal(actual, reference)
    coordinates = targets_3d._ball_coordinates((12, 16, 32))
    assert coordinates is targets_3d._ball_coordinates((12, 16, 32))
    with pytest.raises(ValueError):
        coordinates[0][0] = 10
