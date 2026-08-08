"""Genome encodings."""
from __future__ import annotations

import torch
from torch import nn

MORPHOLOGIES = ("branching", "conical", "radial", "mushroom")


def one_hot_genomes(labels: torch.Tensor, classes: int = len(MORPHOLOGIES)) -> torch.Tensor:
    if labels.ndim != 1 or bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError("genome labels must be a one-dimensional in-range tensor")
    return torch.nn.functional.one_hot(labels, classes).to(torch.float32)


class GenomeEncoder(nn.Module):
    """Select one-hot genomes or a learned embedding with the same caller API."""

    def __init__(self, classes: int = len(MORPHOLOGIES), embedding_size: int | None = None):
        super().__init__()
        self.classes = classes
        self.embedding = nn.Embedding(classes, embedding_size) if embedding_size else None

    @property
    def output_size(self) -> int:
        return self.embedding.embedding_dim if self.embedding else self.classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels) if self.embedding else one_hot_genomes(labels, self.classes)
