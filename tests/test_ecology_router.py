import torch

from morphovoxel.ecology.environment import EcologyWorld
from morphovoxel.ecology.light import compute_light
from morphovoxel.ecology.router import ModelRouter
from morphovoxel.ecology.simulator import ecology_step
from morphovoxel.environment import ENVIRONMENT_CHANNELS


class TaggedSpecialist:
    genome_size = 0
    context_channels = 0

    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def __call__(self, states):
        self.calls.append(states[:, 1, 0, 0, 0].clone())
        proposed = states.clone()
        proposed[:, 0] = self.tag
        return proposed


def test_specialist_router_groups_models_and_restores_organism_order():
    states = torch.zeros(4, 2, 4, 4, 4)
    states[:, 1, 0, 0, 0] = torch.arange(4)
    genomes = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    context = torch.zeros(4, len(ENVIRONMENT_CHANNELS), 4, 4, 4)
    oak, pine = TaggedSpecialist(1), TaggedSpecialist(2)

    proposed = ModelRouter({"oak": oak, "pine": pine})(
        states, genomes, context, ("pine", "oak", "pine", "oak")
    )

    assert torch.equal(proposed[:, 0, 0, 0, 0], torch.tensor([2, 1, 2, 1]))
    assert torch.equal(oak.calls[0], torch.tensor([1, 3]))
    assert torch.equal(pine.calls[0], torch.tensor([0, 2]))


def test_shared_router_keeps_continuous_genomes_and_context_paired():
    class SharedModel:
        genome_size = 2
        context_channels = len(ENVIRONMENT_CHANNELS)

        def __init__(self):
            self.calls = 0

        def __call__(self, states, genomes, context):
            self.calls += 1
            proposed = states.clone()
            proposed[:, :1] = genomes[:, :, None, None, None][:, :1] + context[:, :1]
            return proposed

    model = SharedModel()
    states = torch.zeros(3, 2, 4, 4, 4)
    genomes = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    context = torch.zeros(3, len(ENVIRONMENT_CHANNELS), 4, 4, 4)
    context[:, 0] = torch.tensor([1.0, 2.0, 3.0])[:, None, None, None]

    proposed = ModelRouter(model)(states, genomes, context)

    assert model.calls == 1
    assert torch.allclose(proposed[:, 0, 0, 0, 0], torch.tensor([1.1, 2.2, 3.3]))


def test_ecology_passes_current_local_fields_before_model_proposal():
    class ContextSpy:
        genome_size = 2
        context_channels = len(ENVIRONMENT_CHANNELS)

        def __init__(self):
            self.context = None

        def __call__(self, states, genomes, context):
            self.context = context.clone()
            return states.clone()

    states = torch.zeros(2, 3, 6, 6, 6)
    states[0, 0, 2, 2, 1] = 1
    states[1, 0, 3, 3, 4] = 1
    substrate = torch.ones(6, 6, 6, dtype=torch.bool)
    water = torch.full((6, 6, 6), 0.4)
    energy = torch.ones(2, 6, 6, 6)
    obstacles = torch.zeros_like(substrate)
    obstacles[2, 2, 2] = True
    world = EcologyWorld(
        states,
        torch.eye(2),
        substrate,
        water,
        torch.zeros_like(water),
        energy,
        obstacles=obstacles,
    )
    model = ContextSpy()

    ecology_step(world, model, {"wind": (0.25, -0.5, 0.75)})

    context = model.context
    assert context is not None
    channel = ENVIRONMENT_CHANNELS.index
    assert torch.equal(context[0, channel("light")], compute_light(world.occupancy))
    assert torch.equal(context[0, channel("water")], water)
    assert torch.equal(context[:, channel("energy")], energy)
    assert torch.equal(context[0, channel("substrate")], substrate.float())
    assert torch.equal(context[0, channel("obstacles")], obstacles.float())
    assert torch.equal(context[0, channel("neighbor_occupancy")], world.occupancy[1])
    assert torch.equal(context[1, channel("neighbor_occupancy")], world.occupancy[0])
    assert torch.all(context[:, channel("gravity_z")] == -1)
    assert torch.all(context[:, channel("wind_z")] == 0.25)
    assert torch.all(context[:, channel("wind_y")] == -0.5)
    assert torch.all(context[:, channel("wind_x")] == 0.75)


def test_legacy_models_explicitly_use_the_no_context_path():
    class LegacyModel:
        def __init__(self):
            self.genomes = None

        def __call__(self, states, genomes):
            self.genomes = genomes.clone()
            return states.clone()

    model = LegacyModel()
    states = torch.zeros(2, 2, 4, 4, 4)
    genomes = torch.eye(2)
    context = torch.ones(2, len(ENVIRONMENT_CHANNELS), 4, 4, 4)

    ModelRouter(model)(states, genomes, context)

    assert torch.equal(model.genomes, genomes)
