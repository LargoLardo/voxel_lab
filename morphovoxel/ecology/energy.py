"""Local resource gains and biomass costs."""
from __future__ import annotations

import torch


def update_energy(
    energy: torch.Tensor,
    occupancy: torch.Tensor,
    light_gain: torch.Tensor,
    water_gain: torch.Tensor,
    positive_growth: torch.Tensor,
    occupancy_change: torch.Tensor,
    *,
    maintenance_cost: float = 0.001,
    growth_cost: float = 0.1,
    remodeling_cost: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    usable = torch.minimum(light_gain, water_gain)
    costs = {
        "maintenance": maintenance_cost * occupancy,
        "growth": growth_cost * positive_growth,
        "remodeling": remodeling_cost * occupancy_change.abs(),
    }
    return (energy + usable - sum(costs.values())).clamp_min(0), costs


def gate_growth(previous: torch.Tensor, proposed: torch.Tensor, energy: torch.Tensor, growth_cost: float) -> torch.Tensor:
    positive = (proposed[:, :1] - previous[:, :1]).clamp_min(0)
    affordable = (energy[:, None] / max(growth_cost, 1e-12)).clamp(0, 1)
    return torch.where(proposed[:, :1] > previous[:, :1], previous[:, :1] + positive * affordable, proposed[:, :1])


def transport_energy(energy: torch.Tensor, valid: torch.Tensor, rate: float = 0.05) -> torch.Tensor:
    """Conservatively diffuse energy through each organism's local living neighborhood."""
    if energy.ndim != 4 or energy.shape != valid.shape or not 0 <= rate <= 1 / 6:
        raise ValueError("energy and valid masks must be [N,D,H,W], with rate in [0,1/6]")
    result = energy.clone()
    for axis in range(1, 4):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        pair = valid[left] & valid[right]
        flux = rate * (energy[left] - energy[right]) * pair
        result[left] -= flux
        result[right] += flux
    return result.clamp_min(0)
