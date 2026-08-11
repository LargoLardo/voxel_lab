import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, environment_context_batch
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.rollout import rollout


def test_local_environment_context_reaches_update_rule_before_growth():
    channels = 5
    model = NeuralCA3D(channels, hidden=8, fire_rate=1, context_channels=len(ENVIRONMENT_CHANNELS))
    state = torch.zeros(1, channels, 8, 8, 8)
    state[:, 0, 4, 4, 4] = 1
    context = environment_context_batch([EnvironmentSpec(light_direction_x=1)], 8)
    seen = []
    hook = model.update[0].register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach().clone()))
    try:
        model(state, context=context)
    finally:
        hook.remove()
    assert seen[0].shape[1] == channels * 5 + len(ENVIRONMENT_CHANNELS)
    assert torch.equal(seen[0][:, -len(ENVIRONMENT_CHANNELS) :], context)
    with pytest.raises(ValueError, match="context must have shape"):
        model(state, context=context[:, :, :-1])


def test_rollout_keeps_environment_context_attached_to_every_update():
    model = NeuralCA3D(3, hidden=4, fire_rate=1, context_channels=len(ENVIRONMENT_CHANNELS))
    state = torch.zeros(2, 3, 8, 8, 8)
    state[:, 0, 4, 4, 4] = 1
    context = environment_context_batch([EnvironmentSpec.random(1), EnvironmentSpec.random(2)], 8)
    calls = []
    hook = model.update[0].register_forward_pre_hook(
        lambda _module, inputs: calls.append(inputs[0][:, -len(ENVIRONMENT_CHANNELS) :].clone())
    )
    try:
        rollout(model, state, 3, context=context)
    finally:
        hook.remove()
    assert len(calls) == 3
    assert all(torch.equal(call, context) for call in calls)
