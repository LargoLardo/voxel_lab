"""Per-organism ecological measurements."""
from __future__ import annotations

import torch

from .environment import EcologyWorld


def ecology_metrics(world: EcologyWorld, flows: dict[str, torch.Tensor] | None = None) -> list[dict[str, float]]:
    flows = flows or {}
    output = []
    volumes = (world.occupancy > 0.5).sum((1, 2, 3)).float()
    light_total = flows.get("light_absorbed", torch.zeros(len(world.states), device=world.states.device))
    water_total = flows.get("water_absorbed", torch.zeros_like(light_total))
    for index, occupancy in enumerate(world.occupancy):
        points = torch.nonzero(occupancy > 0.5)
        height = float(world.states.shape[-3] - points[:, 0].min()) if len(points) else 0.0
        spread = float(torch.linalg.vector_norm((points[:, 1:].max(0).values - points[:, 1:].min(0).values).float())) if len(points) else 0.0
        output.append({
            "organism": index, "occupied_volume": float((occupancy > 0.5).sum().detach()), "final_energy": float(world.energy[index].sum().detach()),
            "maximum_height": height, "horizontal_spread": spread,
            "resource_efficiency": float((volumes[index] / (light_total[index] + water_total[index]).clamp_min(1e-12)).detach()),
            "resource_share": float(((light_total[index] + water_total[index]) / (light_total + water_total).sum().clamp_min(1e-12)).detach()),
            "occupied_volume_share": float((volumes[index] / volumes.sum().clamp_min(1)).detach()),
            "shading_received": float((occupancy * (1 - world.light)).sum().detach()),
            "shading_imposed": float((occupancy * (1 - world.light.roll(1, 0))).sum().detach()),
            **{name: float(values[index].detach()) for name, values in flows.items()},
        })
    return output
