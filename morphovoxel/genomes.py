"""Inherited organism genomes and legacy label encodings."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import ClassVar, Iterable, Mapping

import numpy as np
import torch
from torch import nn

MORPHOLOGIES = ("branching", "conical", "radial", "mushroom")
TREE_FAMILIES = ("branching", "conifer", "broad_canopy", "weeping")
TREE_GENOME_VERSION = 2
TREE_STYLE_PHASE_SCALE = 0.00000161803398875


@dataclass(frozen=True)
class GeneSpec:
    """One normalized, interpretable tree-family control."""

    name: str
    label: str
    minimum: float = -1.0
    maximum: float = 1.0
    default: float = 0.0
    stage: str = "family"
    descriptor: str = ""


TREE_GENE_SPECS = (
    GeneSpec("height", "Height", descriptor="height"),
    GeneSpec("trunk_thickness", "Trunk thickness", descriptor="trunk volume"),
    GeneSpec("branch_density", "Branch density", descriptor="branch count"),
    GeneSpec("branch_inclination", "Branch inclination", descriptor="branch elevation"),
    GeneSpec("branch_length", "Branch length", descriptor="branch reach"),
    GeneSpec("canopy_spread", "Canopy spread", descriptor="canopy width"),
    GeneSpec("asymmetry", "Asymmetry", descriptor="signed canopy offset"),
    GeneSpec("light_tropism", "Light tropism", stage="environment", descriptor="light-facing offset"),
    GeneSpec("root_canopy_allocation", "Root / canopy allocation", descriptor="canopy/root ratio"),
)

FAMILY_GENE_NAMES = tuple(spec.name for spec in TREE_GENE_SPECS if spec.stage == "family")
ENVIRONMENT_GENE_NAMES = tuple(spec.name for spec in TREE_GENE_SPECS if spec.stage == "environment")


@dataclass(frozen=True)
class TreeGenome:
    """Serializable continuous genome for one reproducible tree organism.

    Family is deliberately discrete because radically different topologies do
    not have a trustworthy smooth midpoint. Continuous genes remain bounded in
    ``[-1, 1]``. ``style_seed`` is inherited identity, not per-step fire noise.
    """

    family: str = TREE_FAMILIES[0]
    genes: tuple[float, ...] = tuple(spec.default for spec in TREE_GENE_SPECS)
    style_seed: int = 0
    schema_version: int = TREE_GENOME_VERSION

    MODEL_STYLE_FEATURES: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if self.schema_version != TREE_GENOME_VERSION:
            raise ValueError(f"unsupported tree genome schema version: {self.schema_version}")
        if self.family not in TREE_FAMILIES:
            raise ValueError(f"family must be one of {TREE_FAMILIES}")
        if len(self.genes) != len(TREE_GENE_SPECS):
            raise ValueError(f"tree genome requires {len(TREE_GENE_SPECS)} genes")
        normalized = tuple(float(value) for value in self.genes)
        for spec, value in zip(TREE_GENE_SPECS, normalized):
            if not math.isfinite(value) or not spec.minimum <= value <= spec.maximum:
                raise ValueError(f"{spec.name} must be finite and within [{spec.minimum}, {spec.maximum}]")
        if isinstance(self.style_seed, bool) or not isinstance(self.style_seed, int) or not 0 <= self.style_seed < 2**31:
            raise ValueError("style_seed must be an integer from 0 through 2^31-1")
        object.__setattr__(self, "genes", normalized)

    @classmethod
    def model_size(cls) -> int:
        return len(TREE_FAMILIES) + len(TREE_GENE_SPECS) + cls.MODEL_STYLE_FEATURES

    @property
    def values(self) -> dict[str, float]:
        return {spec.name: value for spec, value in zip(TREE_GENE_SPECS, self.genes)}

    def value(self, name: str) -> float:
        try:
            return self.genes[next(index for index, spec in enumerate(TREE_GENE_SPECS) if spec.name == name)]
        except StopIteration as error:
            raise KeyError(name) from error

    def with_values(self, values: Mapping[str, float]) -> "TreeGenome":
        unknown = set(values) - {spec.name for spec in TREE_GENE_SPECS}
        if unknown:
            raise ValueError(f"unknown tree genes: {', '.join(sorted(unknown))}")
        genes = tuple(float(values.get(spec.name, current)) for spec, current in zip(TREE_GENE_SPECS, self.genes))
        return replace(self, genes=genes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "style_seed": self.style_seed,
            "genes": self.values,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TreeGenome":
        if not isinstance(value, Mapping):
            raise ValueError("tree genome JSON must be an object")
        allowed = {"schema_version", "family", "style_seed", "genes"}
        unknown_fields = set(value) - allowed
        if unknown_fields:
            raise ValueError(f"unknown tree genome fields: {', '.join(sorted(map(str, unknown_fields)))}")
        raw_genes = value.get("genes", {})
        if not isinstance(raw_genes, Mapping):
            raise ValueError("tree genome genes must be an object")
        expected = {spec.name for spec in TREE_GENE_SPECS}
        unknown = set(raw_genes) - expected
        if unknown:
            raise ValueError(f"unknown tree genes: {', '.join(sorted(map(str, unknown)))}")
        genes = tuple(float(raw_genes.get(spec.name, spec.default)) for spec in TREE_GENE_SPECS)
        return cls(
            family=str(value.get("family", TREE_FAMILIES[0])),
            genes=genes,
            style_seed=value.get("style_seed", 0),  # type: ignore[arg-type]
            schema_version=int(value.get("schema_version", TREE_GENOME_VERSION)),
        )

    @classmethod
    def random(
        cls,
        seed: int,
        *,
        family: str | None = None,
        span: float = 1.0,
        locked: Iterable[str] = (),
        base: "TreeGenome | None" = None,
    ) -> "TreeGenome":
        if not 0 <= span <= 1:
            raise ValueError("span must be within [0, 1]")
        rng = np.random.default_rng(seed)
        base = base or cls()
        locked_names = set(locked)
        unknown = locked_names - {spec.name for spec in TREE_GENE_SPECS}
        if unknown:
            raise ValueError(f"unknown locked tree genes: {', '.join(sorted(unknown))}")
        genes = tuple(
            current if spec.name in locked_names else float(rng.uniform(-span, span))
            for spec, current in zip(TREE_GENE_SPECS, base.genes)
        )
        chosen_family = family or TREE_FAMILIES[int(rng.integers(len(TREE_FAMILIES)))]
        return cls(chosen_family, genes, int(rng.integers(0, 2**31)))

    def mutate(self, strength: float, seed: int, *, locked: Iterable[str] = ()) -> "TreeGenome":
        if not 0 <= strength <= 1:
            raise ValueError("mutation strength must be within [0, 1]")
        locked_names = set(locked)
        unknown = locked_names - {spec.name for spec in TREE_GENE_SPECS}
        if unknown:
            raise ValueError(f"unknown locked tree genes: {', '.join(sorted(unknown))}")
        rng = np.random.default_rng(seed)
        genes = tuple(
            current if spec.name in locked_names else float(np.clip(current + rng.normal(0, strength), spec.minimum, spec.maximum))
            for spec, current in zip(TREE_GENE_SPECS, self.genes)
        )
        # Keep style identity fixed so bounded mutation stays local and a
        # zero-strength mutation is exactly neutral. random() explicitly
        # creates a new style identity when that is wanted.
        return replace(self, genes=genes)

    def interpolate(self, other: "TreeGenome", amount: float) -> "TreeGenome":
        if self.family != other.family:
            raise ValueError("interpolation requires the same discrete tree family")
        if not 0 <= amount <= 1:
            raise ValueError("interpolation amount must be within [0, 1]")
        genes = tuple(left + (right - left) * amount for left, right in zip(self.genes, other.genes))
        # Style identity stays fixed across the path so the slider changes only
        # the semantic genes instead of injecting a discontinuous random jump.
        return replace(self, genes=genes)

    @property
    def style_phase(self) -> float:
        """Smooth phase shared by the model input and procedural target."""
        return self.style_seed * TREE_STYLE_PHASE_SCALE

    def model_vector(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        family = torch.zeros(len(TREE_FAMILIES), dtype=torch.float32, device=device)
        family[TREE_FAMILIES.index(self.family)] = 1
        phase = self.style_phase
        style = torch.tensor((math.sin(phase), math.cos(phase)), dtype=torch.float32, device=device)
        return torch.cat((family, torch.tensor(self.genes, dtype=torch.float32, device=device), style))


def tree_genome_tensor(genomes: Iterable[TreeGenome], *, device: torch.device | str | None = None) -> torch.Tensor:
    values = [genome.model_vector(device=device) for genome in genomes]
    if not values:
        raise ValueError("at least one tree genome is required")
    return torch.stack(values)


def tree_genome_from_vector(vector: torch.Tensor, style_seed: int) -> TreeGenome:
    """Recover the semantic portion of a model vector using its paired seed."""
    flat = vector.detach().cpu().flatten()
    if len(flat) != TreeGenome.model_size() or not bool(torch.isfinite(flat).all()):
        raise ValueError("tree genome model vector is invalid")
    family_values = flat[: len(TREE_FAMILIES)]
    if float(family_values.max()) <= 0:
        raise ValueError("tree genome vector has no family identity")
    family = TREE_FAMILIES[int(family_values.argmax())]
    start = len(TREE_FAMILIES)
    genes = tuple(float(value) for value in flat[start : start + len(TREE_GENE_SPECS)])
    return TreeGenome(family, genes, int(style_seed))


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
