"""One-step resource-aware multi-organism simulation."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .energy import gate_growth, transport_energy, update_energy
from .environment import EcologyWorld, local_environment_context
from .light import compute_light
from .router import ModelRouter
from .water import allocate_water, diffuse_water


class ProceduralEcologyBaseline:
    """Fixed non-neural local growth schedule for ecological context."""

    def __call__(self, state: torch.Tensor, genome: torch.Tensor | None = None) -> torch.Tensor:
        proposed = state.clone()
        proposed[:, :1] = F.max_pool3d(state[:, :1], 3, stride=1, padding=1)
        return proposed


def ecology_step(world: EcologyWorld, model, config: dict | None = None) -> tuple[EcologyWorld, dict[str, torch.Tensor]]:
    config = config or {}
    incident_light = float(config.get("incident_light", 1))
    light_attenuation = float(config.get("light_attenuation", 0.7))
    lateral_light = float(config.get("lateral_light_diffusion", 0))
    pre_update_light = compute_light(world.occupancy, incident_light, light_attenuation, lateral_light)
    growth_cost = float(config.get("growth_cost", 0.1))
    living = F.max_pool3d(world.occupancy[:, None], 3, stride=1, padding=1)[:, 0] > 0.1
    available_energy = transport_energy(world.energy, living, float(config.get("energy_diffusion", 0.05)))
    context = local_environment_context(
        world,
        light=pre_update_light,
        energy=available_energy,
        gravity=config.get("gravity", (-1, 0, 0)),
        wind=config.get("wind", (0, 0, 0)),
    )
    router = model if isinstance(model, ModelRouter) else ModelRouter(model)
    proposed = router(world.states, world.genomes, context, world.model_ids)
    proposed[:, :, world.obstacles] = 0
    proposed[:, :1] = gate_growth(world.states, proposed, available_energy, growth_cost)
    # Keep only the strongest claimant where organisms try to occupy the same voxel.
    claims = proposed[:, 0].clamp(0, 1)
    winners = claims.argmax(0)
    collision = claims > float(config.get("occupancy_threshold", 0.5))
    for organism in range(len(proposed)):
        losing = collision[organism] & collision.any(0) & (winners != organism)
        proposed[organism, :, losing] = 0
    occupancy = proposed[:, 0].clamp(0, 1)
    light = compute_light(occupancy, incident_light, light_attenuation, lateral_light)
    material = proposed[:, 1:5].softmax(1)
    collector = material[:, min(3, material.shape[1] - 1)]
    root_tissue = material[:, min(1, material.shape[1] - 1)]
    light_gain = occupancy * collector * light * float(config.get("light_absorption", 0.05))
    root_strength = occupancy * root_tissue
    root_reach = F.max_pool3d(root_strength[:, None], 3, stride=1, padding=1)[:, 0]
    demand = world.substrate[None] * root_reach * float(config.get("water_absorption", 0.05))
    water, water_absorbed = allocate_water(world.water, demand)
    water_gain = occupancy * F.max_pool3d(water_absorbed[:, None], 3, stride=1, padding=1)[:, 0]
    water = diffuse_water(water, world.substrate, float(config.get("water_diffusion", 0.1)))
    positive = (occupancy - world.occupancy).clamp_min(0)
    energy, costs = update_energy(
        available_energy, occupancy, light_gain, water_gain, positive, occupancy - world.occupancy,
        maintenance_cost=float(config.get("maintenance_cost", 0.001)), growth_cost=growth_cost,
        remodeling_cost=float(config.get("remodeling_cost", 0.01)),
    )
    next_world = EcologyWorld(
        proposed,
        world.genomes,
        world.substrate,
        water,
        light,
        energy,
        obstacles=world.obstacles,
        model_ids=world.model_ids,
    )
    return next_world, {"light_absorbed": light_gain.sum((1, 2, 3)), "water_absorbed": water_absorbed.sum((1, 2, 3)), **{name: value.sum((1, 2, 3)) for name, value in costs.items()}}
