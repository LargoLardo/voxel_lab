import json

import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, environment_context_batch, make_environment_context


def test_environment_spec_roundtrip_and_deterministic_local_fields():
    spec = EnvironmentSpec.random(11)
    restored = EnvironmentSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert EnvironmentSpec.from_vector(spec.vector()) == spec
    context = make_environment_context(spec, 16)
    assert context.shape == (len(ENVIRONMENT_CHANNELS), 16, 16, 16)
    assert torch.equal(context, make_environment_context(restored, 16))
    assert torch.all(context[:6] >= 0)
    assert torch.all(context[:6] <= 1)
    assert torch.all(context[ENVIRONMENT_CHANNELS.index("gravity_z")] == -1)


def test_environment_batch_is_paired_and_bounds_are_checked():
    left, right = EnvironmentSpec.random(1), EnvironmentSpec.random(2)
    batch = environment_context_batch([left, right], 8)
    assert batch.shape == (2, len(ENVIRONMENT_CHANNELS), 8, 8, 8)
    assert not torch.equal(batch[0], batch[1])
    with pytest.raises(ValueError, match="wind_strength"):
        EnvironmentSpec(wind_strength=1.1)
