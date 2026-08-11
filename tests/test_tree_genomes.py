import json

import numpy as np
import pytest
import torch

from morphovoxel.genomes import TREE_GENE_SPECS, TreeGenome, tree_genome_from_vector, tree_genome_tensor
from morphovoxel.targets import make_tree_target


def test_tree_genome_roundtrip_bounds_and_model_vector():
    genome = TreeGenome.random(17, family="conifer")
    restored = TreeGenome.from_dict(json.loads(json.dumps(genome.to_dict())))
    assert restored == genome
    assert tree_genome_tensor([genome, restored]).shape == (2, TreeGenome.model_size())
    assert torch.equal(genome.model_vector(), restored.model_vector())
    decoded = tree_genome_from_vector(genome.model_vector(), genome.style_seed)
    assert decoded.family == genome.family and decoded.style_seed == genome.style_seed
    assert decoded.genes == pytest.approx(genome.genes)
    with pytest.raises(ValueError, match="within"):
        genome.with_values({TREE_GENE_SPECS[0].name: 1.01})
    with pytest.raises(ValueError, match="unknown tree genome fields"):
        TreeGenome.from_dict({"famly": "weeping"})


def test_tree_genome_random_mutation_interpolation_and_locks_are_reproducible():
    left = TreeGenome.random(2, family="branching")
    right = TreeGenome.random(3, family="branching")
    assert TreeGenome.random(2, family="branching") == left
    assert left.mutate(0.2, 9) == left.mutate(0.2, 9)
    locked = TREE_GENE_SPECS[0].name
    assert left.mutate(1.0, 9, locked=[locked]).value(locked) == left.value(locked)
    neutral = left.mutate(0, 999)
    assert neutral == left
    assert np.array_equal(make_tree_target(neutral, 16)[0], make_tree_target(left, 16)[0])
    midpoint = left.interpolate(right, 0.5)
    assert midpoint.style_seed == left.style_seed
    assert midpoint.genes == pytest.approx(tuple((a + b) / 2 for a, b in zip(left.genes, right.genes)))
    with pytest.raises(ValueError, match="same discrete"):
        left.interpolate(TreeGenome.random(4, family="weeping"), 0.5)
