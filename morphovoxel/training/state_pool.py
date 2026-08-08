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


class StatePool:
    def __init__(self, states: torch.Tensor, genomes: torch.Tensor, ages: torch.Tensor | None = None):
        if len(states) != len(genomes):
            raise ValueError("each state must have exactly one genome")
        self.states = states.detach().cpu().clone()
        self.genomes = genomes.detach().cpu().clone()
        self.ages = torch.zeros(len(states), dtype=torch.long) if ages is None else ages.detach().cpu().clone()

    def sample(self, count: int, generator: torch.Generator | None = None, device: str | torch.device = "cpu") -> PoolBatch:
        if not 0 < count <= len(self.states):
            raise ValueError("sample count must fit the pool")
        indices = torch.randperm(len(self.states), generator=generator)[:count]
        return PoolBatch(indices, self.states[indices].to(device), self.genomes[indices].to(device), self.ages[indices].to(device))

    def commit(self, batch: PoolBatch, states: torch.Tensor, elapsed: int) -> None:
        if len(states) != len(batch.indices):
            raise ValueError("replacement batch size mismatch")
        self.states[batch.indices] = states.detach().cpu()
        self.ages[batch.indices] = batch.ages.cpu() + elapsed

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"states": self.states, "genomes": self.genomes, "ages": self.ages}

