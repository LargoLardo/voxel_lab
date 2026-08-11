"""Inference routing for shared and specialist ecology models."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch


class ModelRouter:
    """Batch organisms by model id and restore their original order."""

    def __init__(self, models: Callable | Mapping[str, Callable]):
        if isinstance(models, Mapping):
            if not models:
                raise ValueError("at least one specialist model is required")
            self.models = dict(models)
            self.shared = False
        else:
            self.models = {"shared": models}
            self.shared = True

    def parameters(self):
        """Expose routed module parameters for existing run metadata."""
        seen: set[int] = set()
        for model in self.models.values():
            if id(model) not in seen and hasattr(model, "parameters"):
                seen.add(id(model))
                yield from model.parameters()

    @staticmethod
    def _propose(model: Callable, states: torch.Tensor, genomes: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        genome_size = getattr(model, "genome_size", None)
        context_channels = int(getattr(model, "context_channels", 0))
        if genome_size not in (None, 0) and genomes.shape != (len(states), int(genome_size)):
            raise ValueError("routed genomes do not match the model genome size")
        if context_channels:
            if context is None:
                context = states.new_zeros((len(states), context_channels, *states.shape[-3:]))
            elif context.shape != (len(states), context_channels, *states.shape[-3:]):
                raise ValueError("routed context does not match the model context channels")
            proposed = model(states, genomes, context) if genome_size else model(states, context=context)
        elif genome_size == 0:
            proposed = model(states)
        else:
            # Explicit compatibility path for legacy callables without context metadata.
            proposed = model(states, genomes)
        if proposed.shape != states.shape:
            raise ValueError("ecology models must return the same shape they receive")
        return proposed

    def __call__(
        self,
        states: torch.Tensor,
        genomes: torch.Tensor,
        context: torch.Tensor | None = None,
        model_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if len(states) != len(genomes):
            raise ValueError("one genome vector is required per organism")
        if self.shared:
            return self._propose(next(iter(self.models.values())), states, genomes, context)
        if model_ids is None or len(model_ids) != len(states):
            raise ValueError("specialist routing requires one model id per organism")

        groups: dict[str, list[int]] = {}
        for index, model_id in enumerate(model_ids):
            if model_id not in self.models:
                raise ValueError(f"unknown ecology model id: {model_id}")
            groups.setdefault(model_id, []).append(index)

        ordered: list[torch.Tensor | None] = [None] * len(states)
        for model_id, positions in groups.items():
            index = torch.as_tensor(positions, device=states.device)
            batch_context = context.index_select(0, index) if context is not None else None
            batch = self._propose(
                self.models[model_id], states.index_select(0, index), genomes.index_select(0, index), batch_context
            )
            for local_index, organism_index in enumerate(positions):
                ordered[organism_index] = batch[local_index : local_index + 1]
        return torch.cat([value for value in ordered if value is not None])
