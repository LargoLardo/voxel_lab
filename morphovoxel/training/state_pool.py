"""Genome-safe pool of intermediate states."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PoolBatch:
    indices: torch.Tensor
    states: torch.Tensor
    genomes: torch.Tensor
    ages: torch.Tensor
    target_occupancy: torch.Tensor | None = None
    target_materials: torch.Tensor | None = None
    environments: torch.Tensor | None = None
    environment_specs: torch.Tensor | None = None
    style_seeds: torch.Tensor | None = None


class StatePool:
    def __init__(
        self,
        states: torch.Tensor,
        genomes: torch.Tensor,
        ages: torch.Tensor | None = None,
        *,
        target_occupancy: torch.Tensor | None = None,
        target_materials: torch.Tensor | None = None,
        environments: torch.Tensor | None = None,
        environment_specs: torch.Tensor | None = None,
        style_seeds: torch.Tensor | None = None,
    ):
        paired = {
            "genome": genomes,
            "age": ages,
            "target occupancy": target_occupancy,
            "target materials": target_materials,
            "environment": environments,
            "environment specification": environment_specs,
            "style seed": style_seeds,
        }
        for name, value in paired.items():
            if value is not None and len(value) != len(states):
                raise ValueError(f"each state must have exactly one {name}")
        self.states = states.detach().cpu().clone()
        self.genomes = genomes.detach().cpu().clone()
        self.ages = torch.zeros(len(states), dtype=torch.long) if ages is None else ages.detach().cpu().clone()
        self.target_occupancy = self._copy(target_occupancy)
        self.target_materials = self._copy(target_materials)
        self.environments = self._copy(environments)
        self.environment_specs = self._copy(environment_specs)
        self.style_seeds = self._copy(style_seeds)

    @staticmethod
    def _copy(value: torch.Tensor | None) -> torch.Tensor | None:
        return value.detach().cpu().clone() if value is not None else None

    @staticmethod
    def _sample(value: torch.Tensor | None, indices: torch.Tensor, device: str | torch.device) -> torch.Tensor | None:
        return value[indices].to(device) if value is not None else None

    def sample(self, count: int, generator: torch.Generator | None = None, device: str | torch.device = "cpu") -> PoolBatch:
        if not 0 < count <= len(self.states):
            raise ValueError("sample count must fit the pool")
        indices = torch.randperm(len(self.states), generator=generator)[:count]
        return PoolBatch(
            indices.to(device),
            self.states[indices].to(device),
            self.genomes[indices].to(device),
            self.ages[indices].to(device),
            self._sample(self.target_occupancy, indices, device),
            self._sample(self.target_materials, indices, device),
            self._sample(self.environments, indices, device),
            self._sample(self.environment_specs, indices, device),
            self._sample(self.style_seeds, indices, device),
        )

    def commit(self, batch: PoolBatch, states: torch.Tensor, elapsed: int) -> None:
        if len(states) != len(batch.indices):
            raise ValueError("replacement batch size mismatch")
        indices = batch.indices.detach().cpu()
        self.states[indices] = states.detach().cpu()
        self.ages[indices] = batch.ages.cpu() + elapsed

    def replace_entries(
        self,
        indices: torch.Tensor,
        *,
        states: torch.Tensor,
        genomes: torch.Tensor,
        target_occupancy: torch.Tensor | None = None,
        target_materials: torch.Tensor | None = None,
        environments: torch.Tensor | None = None,
        environment_specs: torch.Tensor | None = None,
        style_seeds: torch.Tensor | None = None,
    ) -> None:
        """Replace complete organism identities without breaking pool pairing."""
        indices = indices.detach().cpu()
        count = len(indices)
        values = {
            "states": states,
            "genomes": genomes,
            "target_occupancy": target_occupancy,
            "target_materials": target_materials,
            "environments": environments,
            "environment_specs": environment_specs,
            "style_seeds": style_seeds,
        }
        for name, value in values.items():
            current = getattr(self, name)
            if value is None and current is not None:
                raise ValueError(f"replacement must include {name}")
            if value is not None and len(value) != count:
                raise ValueError(f"replacement {name} size mismatch")
            if value is not None and current is None and name not in {"states", "genomes"}:
                raise ValueError(f"pool was created without {name}")
        for name, value in values.items():
            if value is not None:
                getattr(self, name)[indices] = value.detach().cpu()
        self.ages[indices] = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        values = {
            "states": self.states,
            "genomes": self.genomes,
            "ages": self.ages,
            "target_occupancy": self.target_occupancy,
            "target_materials": self.target_materials,
            "environments": self.environments,
            "environment_specs": self.environment_specs,
            "style_seeds": self.style_seeds,
        }
        return {name: value for name, value in values.items() if value is not None}
