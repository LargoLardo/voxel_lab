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
    condition_ids: torch.Tensor | None = None
    pair_ids: torch.Tensor | None = None


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
        condition_ids: torch.Tensor | None = None,
        pair_ids: torch.Tensor | None = None,
    ):
        paired = {
            "genome": genomes,
            "age": ages,
            "target occupancy": target_occupancy,
            "target materials": target_materials,
            "environment": environments,
            "environment specification": environment_specs,
            "style seed": style_seeds,
            "condition id": condition_ids,
            "pair id": pair_ids,
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
        self.condition_ids = self._copy(condition_ids)
        self.pair_ids = self._copy(pair_ids)

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
            self._sample(self.condition_ids, indices, device),
            self._sample(self.pair_ids, indices, device),
        )

    def sample_stratified_pairs(self, count: int, cursor: int, device: str | torch.device = "cpu") -> PoolBatch:
        """Cycle through adjacent counterfactual pairs instead of global random eviction."""
        if count < 2 or count % 2 or count > len(self.states) or self.pair_ids is None or self.condition_ids is None:
            raise ValueError("stratified pair sampling needs an even count and paired pool metadata")
        groups: list[tuple[int, int, torch.Tensor]] = []
        for pair_id in self.pair_ids.unique(sorted=True).tolist():
            indices = torch.nonzero(self.pair_ids == pair_id, as_tuple=False).flatten()
            if len(indices) != 2:
                raise ValueError("each counterfactual pair must contain exactly two states")
            conditions = self.condition_ids[indices]
            if not bool((conditions == conditions[0]).all()):
                raise ValueError("counterfactual pair condition ids disagree")
            groups.append((int(conditions[0]), int(pair_id), indices))
        groups.sort(key=lambda value: (value[0], value[1]))
        pair_count = count // 2
        selected = [groups[(cursor + offset) % len(groups)][2] for offset in range(pair_count)]
        indices = torch.cat(selected)
        return PoolBatch(
            indices.to(device), self.states[indices].to(device), self.genomes[indices].to(device), self.ages[indices].to(device),
            self._sample(self.target_occupancy, indices, device), self._sample(self.target_materials, indices, device),
            self._sample(self.environments, indices, device), self._sample(self.environment_specs, indices, device),
            self._sample(self.style_seeds, indices, device), self._sample(self.condition_ids, indices, device),
            self._sample(self.pair_ids, indices, device),
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
        condition_ids: torch.Tensor | None = None,
        pair_ids: torch.Tensor | None = None,
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
            "condition_ids": condition_ids,
            "pair_ids": pair_ids,
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

    def append_from(self, other: "StatePool", start: int) -> None:
        """Append a paired suffix from another pool."""
        names = (
            "states", "genomes", "ages", "target_occupancy", "target_materials",
            "environments", "environment_specs", "style_seeds",
            "condition_ids", "pair_ids",
        )
        if not 0 <= start <= len(other.states):
            raise ValueError("pool append start is out of range")
        for name in names:
            if (getattr(self, name) is None) != (getattr(other, name) is None):
                raise ValueError(f"cannot append pool with different {name} pairing")
        for name in names:
            current, incoming = getattr(self, name), getattr(other, name)
            if current is not None:
                setattr(self, name, torch.cat((current, incoming[start:].detach().cpu())))

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
            "condition_ids": self.condition_ids,
            "pair_ids": self.pair_ids,
        }
        return {name: value for name, value in values.items() if value is not None}
