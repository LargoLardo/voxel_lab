"""Deterministic long-horizon validation for 3D tree organisms."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch

from .damage import damage_3d
from .environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, environment_context_batch
from .genomes import FAMILY_GENE_NAMES, TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome, tree_genome_tensor
from .metrics import connected_components, material_accuracy, morphology_metrics, recovery_metrics, threshold_iou
from .random_utils import fork_rng
from .rollout import rollout
from .seeding import seed_state
from .state import StateLayout
from .targets import make_tree_target


VALIDATION_SCHEMA_VERSION = 2
DEFAULT_BOUNDARY_GENES = ("height", "canopy_spread")


@dataclass(frozen=True)
class ValidationCriteria:
    """Explicit archive/checkpoint acceptance thresholds."""

    min_steps: int = 256
    min_recovery_steps: int = 64
    state_limit: float = 4.0
    occupancy_epsilon: float = 1e-4
    max_occupancy_violation_fraction: float = 0.0
    min_largest_component_fraction: float = 0.85
    min_target_iou: float = 0.5
    min_material_accuracy: float = 0.5
    min_descriptor_agreement: float = 0.5
    max_late_drift: float = 0.25
    min_regeneration_score: float = 0.5
    min_transition_edit_accuracy: float = 0.5
    min_gene_response_accuracy: float = 0.5

    def __post_init__(self) -> None:
        if self.min_steps < 1 or self.min_recovery_steps < 1 or self.state_limit <= 0 or self.occupancy_epsilon < 0:
            raise ValueError("validation step counts and state limit must be positive")
        unit_values = (
            self.max_occupancy_violation_fraction,
            self.min_largest_component_fraction,
            self.min_target_iou,
            self.min_material_accuracy,
            self.min_descriptor_agreement,
            self.max_late_drift,
            self.min_regeneration_score,
            self.min_transition_edit_accuracy,
            self.min_gene_response_accuracy,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in unit_values):
            raise ValueError("validation fractions must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationCase:
    """One exact inherited/environment/stochastic validation combination."""

    case_id: str
    category: str
    genome: TreeGenome
    environment: EnvironmentSpec
    fire_seed: int
    source_genome: TreeGenome | None = None
    source_steps: int = 0
    comparison_genome: TreeGenome | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.category:
            raise ValueError("validation case id and category cannot be empty")
        if isinstance(self.fire_seed, bool) or not isinstance(self.fire_seed, int) or not 0 <= self.fire_seed < 2**63:
            raise ValueError("fire seed must be an integer from 0 through 2^63-1")
        if isinstance(self.source_steps, bool) or not isinstance(self.source_steps, int) or self.source_steps < 0:
            raise ValueError("source_steps must be a non-negative integer")
        if (self.source_genome is None) != (self.source_steps == 0):
            raise ValueError("transition validation needs a source genome and positive source_steps")
        if self.comparison_genome is not None:
            other = self.comparison_genome
            changed = [spec.name for spec in TREE_GENE_SPECS if other.value(spec.name) != self.genome.value(spec.name)]
            if (self.source_genome is not None or other.family != self.genome.family
                    or other.style_seed != self.genome.style_seed or len(changed) != 1):
                raise ValueError("gene response validation needs independent growth with exactly one changed gene")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "genome": self.genome.to_dict(),
            "environment": self.environment.to_dict(),
            "fire_seed": self.fire_seed,
            **({"source_genome": self.source_genome.to_dict(), "source_steps": self.source_steps}
               if self.source_genome is not None else {}),
            **({"comparison_genome": self.comparison_genome.to_dict()} if self.comparison_genome is not None else {}),
        }


@dataclass(frozen=True)
class ValidationTrial:
    """Serializable result for one validation case."""

    case: ValidationCase
    steps: int
    recovery_steps: int
    validated: bool
    accepted: bool
    score: float
    failure_reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    descriptors: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.steps < 1 or self.recovery_steps < 1:
            raise ValueError("validation trial step counts must be positive")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("validation score must be finite and within [0, 1]")
        if self.accepted and (not self.validated or self.failure_reasons):
            raise ValueError("an accepted trial must be validated and have no failures")
        if any(not math.isfinite(float(value)) for values in (self.metrics, self.descriptors) for value in values.values()):
            raise ValueError("validation metrics and descriptors must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case.to_dict(),
            "steps": self.steps,
            "recovery_steps": self.recovery_steps,
            "validated": self.validated,
            "accepted": self.accepted,
            "score": self.score,
            "failure_reasons": list(self.failure_reasons),
            "metrics": dict(self.metrics),
            "descriptors": dict(self.descriptors),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate worst/low-percentile persistence result for a panel."""

    trials: tuple[ValidationTrial, ...]
    criteria: ValidationCriteria = ValidationCriteria()
    aggregation: Literal["worst", "low_percentile"] = "worst"
    low_percentile: float = 0.1

    def __post_init__(self) -> None:
        if not self.trials:
            raise ValueError("validation report requires at least one trial")
        if self.aggregation not in ("worst", "low_percentile") or not 0 <= self.low_percentile <= 0.5:
            raise ValueError("aggregation must be worst/low_percentile and percentile within [0, 0.5]")

    @property
    def validated(self) -> bool:
        return all(trial.validated for trial in self.trials)

    @property
    def accepted(self) -> bool:
        return self.validated and all(trial.accepted for trial in self.trials)

    @property
    def worst_score(self) -> float:
        return min(trial.score for trial in self.trials)

    @property
    def low_percentile_score(self) -> float:
        return float(np.quantile([trial.score for trial in self.trials], self.low_percentile))

    @property
    def score(self) -> float:
        return self.worst_score if self.aggregation == "worst" else self.low_percentile_score

    @property
    def failures(self) -> dict[str, list[str]]:
        return {trial.case.case_id: list(trial.failure_reasons) for trial in self.trials if trial.failure_reasons}

    def mean_metrics(self, field: Literal["metrics", "descriptors"]) -> dict[str, float]:
        names = sorted({name for trial in self.trials for name in getattr(trial, field)})
        return {
            name: float(np.mean([getattr(trial, field)[name] for trial in self.trials if name in getattr(trial, field)]))
            for name in names
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validated": self.validated,
            "accepted": self.accepted,
            "aggregation": self.aggregation,
            "score": self.score,
            "worst_score": self.worst_score,
            "low_percentile": self.low_percentile,
            "low_percentile_score": self.low_percentile_score,
            "criteria": self.criteria.to_dict(),
            "failures": self.failures,
            "mean_metrics": self.mean_metrics("metrics"),
            "mean_descriptors": self.mean_metrics("descriptors"),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def _representative_environments(seed: int) -> tuple[EnvironmentSpec, ...]:
    return (
        EnvironmentSpec(),
        EnvironmentSpec(
            light_direction_x=1.0,
            water_direction_x=-0.5,
            water_level=0.7,
            energy=0.8,
            obstacle_density=0.08,
            wind_direction_y=1.0,
            wind_strength=0.6,
            seed=seed + 1,
        ),
    )


def _panel_cases(
    genomes: Sequence[tuple[str, TreeGenome]],
    environments: Sequence[EnvironmentSpec],
    fire_seeds: Sequence[int],
) -> tuple[ValidationCase, ...]:
    if not genomes or not environments or not fire_seeds:
        raise ValueError("validation panels require genomes, environments, and fire seeds")
    if any(not isinstance(environment, EnvironmentSpec) for environment in environments):
        raise ValueError("validation environments must be EnvironmentSpec values")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63 for value in fire_seeds):
        raise ValueError("fire seeds must be integers from 0 through 2^63-1")
    return tuple(
        ValidationCase(
            f"{category}-{genome_index:03d}-e{environment_index:02d}-f{fire_index:02d}",
            category,
            genome,
            environment,
            fire_seed,
        )
        for genome_index, (category, genome) in enumerate(genomes)
        for environment_index, environment in enumerate(environments)
        for fire_index, fire_seed in enumerate(fire_seeds)
    )


def build_validation_panel(
    *,
    seed: int = 0,
    boundary_genes: Iterable[str] | None = None,
    random_count: int = 2,
    interpolation_steps: int = 2,
    mutation_count: int = 2,
    mutation_strength: float = 0.15,
    archived: Iterable[TreeGenome] = (),
    fire_seeds: Sequence[int] = (0, 1),
    environments: Sequence[EnvironmentSpec] | None = None,
) -> tuple[ValidationCase, ...]:
    """Build a reproducible panel; sample counts apply to each tree family."""
    if min(random_count, interpolation_steps, mutation_count) < 0:
        raise ValueError("validation panel counts must be non-negative")
    if not 0 <= mutation_strength <= 1:
        raise ValueError("mutation strength must be within [0, 1]")
    chosen_names = tuple(DEFAULT_BOUNDARY_GENES if boundary_genes is None else boundary_genes)
    specs = {spec.name: spec for spec in TREE_GENE_SPECS}
    unknown = set(chosen_names) - set(specs)
    if unknown:
        raise ValueError(f"unknown boundary genes: {', '.join(sorted(unknown))}")

    entries: list[tuple[str, TreeGenome]] = []
    for family_index, family in enumerate(TREE_FAMILIES):
        default = TreeGenome(family=family)
        family_seed = seed + family_index * 100_000
        entries.append(("default", default))
        for name in chosen_names:
            spec = specs[name]
            entries.extend(("boundary", default.with_values({name: value})) for value in (spec.minimum, spec.maximum))
        if chosen_names:
            entries.append(("corner", default.with_values({name: specs[name].minimum for name in chosen_names})))
            entries.append(("corner", default.with_values({name: specs[name].maximum for name in chosen_names})))
        entries.extend(
            ("random", TreeGenome.random(family_seed + 100 + index, family=family))
            for index in range(random_count)
        )
        if interpolation_steps:
            left = TreeGenome.random(family_seed + 10_000, family=family)
            right = TreeGenome.random(family_seed + 10_001, family=family)
            entries.extend(
                ("interpolation", left.interpolate(right, amount))
                for amount in np.linspace(0, 1, interpolation_steps + 2)[1:-1]
            )
        entries.extend(("mutation", default.mutate(mutation_strength, family_seed + 20_000 + index)) for index in range(mutation_count))
    for genome in archived:
        if not isinstance(genome, TreeGenome):
            raise ValueError("archived validation genomes must be TreeGenome values")
        entries.append(("archive", genome))
    return _panel_cases(entries, tuple(environments or _representative_environments(seed)), tuple(fire_seeds))


def build_gene_response_panel(
    *, style_seeds: Sequence[int], fire_seeds: Sequence[int],
) -> tuple[ValidationCase, ...]:
    """Check every shape gene with independent low/high growth under matching noise.

    Distribute styles and fire seeds over genes to avoid a Cartesian product.
    Interior values exercise interpolation beyond the training endpoints.
    """
    if not style_seeds or not fire_seeds:
        raise ValueError("gene response validation requires style and fire seeds")
    cases = []
    for family_index, family in enumerate(TREE_FAMILIES):
        for gene_index, name in enumerate(FAMILY_GENE_NAMES):
            index = family_index * len(FAMILY_GENE_NAMES) + gene_index
            base = TreeGenome(family=family, style_seed=style_seeds[index % len(style_seeds)])
            cases.append(ValidationCase(
                f"gene-response-{family}-{name}", "gene_response",
                base.with_values({name: .75}), EnvironmentSpec(), fire_seeds[index % len(fire_seeds)],
                comparison_genome=base.with_values({name: -.75}),
            ))
    return tuple(cases)


def build_candidate_panel(
    genome: TreeGenome,
    *,
    seed: int = 0,
    fire_seeds: Sequence[int] = (0, 1, 2),
    environments: Sequence[EnvironmentSpec] | None = None,
) -> tuple[ValidationCase, ...]:
    """Cross one proposed genome with representative environments and fire masks."""
    if not isinstance(genome, TreeGenome):
        raise ValueError("candidate genome must be a TreeGenome")
    return _panel_cases((("candidate", genome),), tuple(environments or _representative_environments(seed)), tuple(fire_seeds))


def build_transition_panel(
    *, style_seeds: Sequence[int], fire_seeds: Sequence[int], source_steps: int = 128,
) -> tuple[ValidationCase, ...]:
    """Test every directed family switch at fixed genes, style and environment."""
    return tuple(
        ValidationCase(
            f"transition-{source}-to-{destination}-s{style}-f{fire}", "transition",
            TreeGenome(family=destination, style_seed=style), EnvironmentSpec(), fire,
            TreeGenome(family=source, style_seed=style), source_steps,
        )
        for source in TREE_FAMILIES for destination in TREE_FAMILIES if source != destination
        for style in style_seeds for fire in fire_seeds
    )


def build_gene_transition_panel(
    *, style_seeds: Sequence[int], fire_seeds: Sequence[int], source_steps: int = 128,
) -> tuple[ValidationCase, ...]:
    """Test both directions of small/large edits to every Phase 2 gene (128 cases).

    Distribute styles and fire seeds across cases instead of multiplying the
    panel by every combination; each reverse pair uses the same seeds.
    """
    if not style_seeds or not fire_seeds:
        raise ValueError("gene transition validation requires style and fire seeds")
    cases = []
    for family_index, family in enumerate(TREE_FAMILIES):
        for gene_index, name in enumerate(FAMILY_GENE_NAMES):
            for magnitude_index, magnitude in enumerate((0.25, 0.75)):
                index = (family_index * len(FAMILY_GENE_NAMES) + gene_index) * 2 + magnitude_index
                base = TreeGenome(family=family, style_seed=style_seeds[index % len(style_seeds)])
                low, high = (base.with_values({name: value}) for value in (-magnitude, magnitude))
                for direction, source, destination in (("increase", low, high), ("decrease", high, low)):
                    cases.append(ValidationCase(
                        f"gene-transition-{family}-{name}-{magnitude}-{direction}", "gene_transition",
                        destination, EnvironmentSpec(), fire_seeds[index % len(fire_seeds)], source, source_steps,
                    ))
    return tuple(cases)


def _model_inputs(model, case: ValidationCase, size: int, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    genome_size = int(getattr(model, "genome_size", 0))
    if genome_size not in (0, TreeGenome.model_size()):
        raise ValueError(f"model genome size {genome_size} is incompatible with tree genome size {TreeGenome.model_size()}")
    context_channels = int(getattr(model, "context_channels", 0))
    if context_channels not in (0, len(ENVIRONMENT_CHANNELS)):
        raise ValueError(f"model context size {context_channels} is incompatible with environment size {len(ENVIRONMENT_CHANNELS)}")
    genome = tree_genome_tensor((case.genome,), device=device) if genome_size else None
    context = environment_context_batch((case.environment,), size, device=device) if context_channels else None
    return genome, context


def _shape_descriptors(occupancy, materials=None, threshold: float = 0.5) -> dict[str, float]:
    values = occupancy.detach().cpu().numpy() if isinstance(occupancy, torch.Tensor) else np.asarray(occupancy)
    mask = np.asarray(values) > threshold
    components, largest = connected_components(mask)
    points = np.argwhere(mask)
    result = {
        "connected_components": float(components),
        "largest_component_fraction": float(largest),
        "occupied_volume": float(mask.sum()),
        "height": float(np.ptp(points[:, 0]) + 1) if len(points) else 0.0,
        "spread_y": float(np.ptp(points[:, 1]) + 1) if len(points) else 0.0,
        "spread_x": float(np.ptp(points[:, 2]) + 1) if len(points) else 0.0,
        "asymmetry": float(np.linalg.norm(points[:, 1:].mean(0) - (np.asarray(mask.shape[1:]) - 1) / 2) / max(mask.shape)) if len(points) else 0.0,
    }
    canopy = points[points[:, 0] <= np.quantile(points[:, 0], 0.45)] if len(points) else points
    result["canopy_spread"] = float(max(np.ptp(canopy[:, 1]) + 1, np.ptp(canopy[:, 2]) + 1)) if len(canopy) else 0.0
    if materials is not None and mask.any():
        labels = materials.detach().cpu().numpy() if isinstance(materials, torch.Tensor) else np.asarray(materials)
        for index in range(int(labels.max()) + 1):
            result[f"material_{index}_fraction"] = float((labels[mask] == index).mean())
        result["branch_material_fraction"] = result.get("material_2_fraction", 0.0)
        result["canopy_material_fraction"] = result.get("material_3_fraction", 0.0)
    return result


def _agreement(left: float, right: float) -> float:
    return 1.0 if left == right == 0 else 0.0 if min(left, right) <= 0 else min(left, right) / max(left, right)


def _safe_metric_values(occupancy: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = morphology_metrics(occupancy, target)
    return {f"target_{name}": float(value) for name, value in values.items() if math.isfinite(float(value))}


def _state_safety(states: Sequence[torch.Tensor], layout: StateLayout, criteria: ValidationCriteria) -> dict[str, float]:
    finite = all(bool(torch.isfinite(state).all()) for state in states)
    sentinel = criteria.state_limit * 2
    maximum = max(float(torch.nan_to_num(state, nan=sentinel, posinf=sentinel, neginf=-sentinel).abs().max()) for state in states)
    occupancies = [state[:, layout.occupancy] for state in states]
    violations = sum(int(((value < -criteria.occupancy_epsilon) | (value > 1 + criteria.occupancy_epsilon) | ~torch.isfinite(value)).sum()) for value in occupancies)
    count = sum(value.numel() for value in occupancies)
    excess = max(
        float(torch.nan_to_num(torch.maximum((-value).clamp_min(0), (value - 1).clamp_min(0)), nan=sentinel, posinf=sentinel).max())
        for value in occupancies
    )
    return {
        "finite_state": float(finite),
        "bounded_state": float(finite and maximum <= criteria.state_limit),
        "max_channel_magnitude": maximum,
        "occupancy_range_violation_fraction": violations / count,
        "max_occupancy_range_excess": excess,
    }


def _infer_device(model, requested: torch.device | str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def validate_candidate(
    model,
    case: ValidationCase,
    *,
    layout: StateLayout,
    world_size: int = 16,
    steps: int = 512,
    recovery_steps: int = 128,
    seed_size: int = 1,
    device: torch.device | str | None = None,
    criteria: ValidationCriteria = ValidationCriteria(),
) -> ValidationTrial:
    """Run one no-gradient growth, persistence, damage, and recovery trial."""
    if world_size < 12 or steps < 1 or recovery_steps < 1:
        raise ValueError("tree validation requires world_size >= 12 and positive step counts")
    run_device = _infer_device(model, device)
    genome, context = _model_inputs(model, case, world_size, run_device)
    target_np, target_material_np = make_tree_target(case.genome, world_size, case.environment)
    target = torch.as_tensor(target_np, device=run_device)
    target_material = torch.as_tensor(target_material_np, device=run_device, dtype=torch.long)
    state = seed_state(1, world_size, layout, dimensions=3, seed_size=seed_size, random_seed=case.fire_seed, device=run_device)
    late_steps = min(128, max(1, steps // 2))
    mature_steps = steps - late_steps
    was_training = bool(model.training)
    model.eval()
    source_state = comparison_state = None
    source_iou = None
    try:
        with torch.inference_mode(), fork_rng(run_device):
            torch.manual_seed(case.fire_seed)
            if run_device.type == "cuda":
                torch.cuda.manual_seed_all(case.fire_seed)
            if case.comparison_genome is not None:
                if genome is None:
                    raise ValueError("gene response validation requires a gene-conditioned model")
                # Independent seedlings see identical fire masks. Restore RNG
                # before the main rollout so growth noise cannot fake a response.
                with fork_rng(run_device):
                    comparison_state, _ = rollout(
                        model, state.clone(), steps,
                        tree_genome_tensor((case.comparison_genome,), device=run_device), context=context,
                    )
            if case.source_genome is not None:
                if genome is None:
                    raise ValueError("transition validation requires a tree-family model")
                source_genome = tree_genome_tensor((case.source_genome,), device=run_device)
                state, _ = rollout(model, state, case.source_steps, source_genome, context=context)
                source_state = state
                source_target, source_material = make_tree_target(case.source_genome, world_size, case.environment)
                source_iou = threshold_iou(state[0, layout.occupancy], torch.as_tensor(source_target, device=run_device))
            mature, _ = rollout(model, state, mature_steps, genome, context=context)
            final, _ = rollout(model, mature, late_steps, genome, context=context) if bool(torch.isfinite(mature).all()) else (mature, [])
            if bool(torch.isfinite(final).all()):
                damage_kinds = ("sphere", "cuboid", "top", "dropout")
                damaged, removed = damage_3d(final, 0.25, damage_kinds[case.fire_seed % len(damage_kinds)], case.fire_seed)
                recovered, _ = rollout(model, damaged, recovery_steps, genome, context=context)
            else:
                damaged = recovered = final
                removed = torch.zeros(final.shape[-3:], dtype=torch.bool, device=run_device)
    finally:
        model.train(was_training)

    occupancy = final[0, layout.occupancy]
    mature_occupancy = mature[0, layout.occupancy]
    recovered_occupancy = recovered[0, layout.occupancy]
    predicted_materials = final[0, layout.material_slice].argmax(0)
    descriptors = _shape_descriptors(occupancy, predicted_materials)
    target_descriptors = _shape_descriptors(target, target_material)
    descriptor_agreements = {
        name: _agreement(descriptors[name], target_descriptors[name])
        for name in ("occupied_volume", "height", "canopy_spread")
    }
    descriptor_agreement = min(descriptor_agreements.values())
    target_iou = threshold_iou(occupancy, target)
    late_drift = 1 - threshold_iou(mature_occupancy, occupancy)
    if bool(torch.isfinite(final).all()) and bool(torch.isfinite(recovered).all()):
        recovery = recovery_metrics(occupancy, damaged[0, layout.occupancy], recovered_occupancy, target, removed)
        damaged_iou, recovered_iou = float(recovery["post_damage_iou"]), float(recovery["final_iou"])
        denominator = target_iou - damaged_iou
        regeneration_score = float(np.clip((recovered_iou - damaged_iou) / denominator, 0, 1)) if denominator > 1e-8 else float(recovered_iou >= target_iou)
        recovered_fraction = float(recovery["recovered_target_fraction"])
    else:
        damaged_iou = recovered_iou = regeneration_score = recovered_fraction = 0.0
    # Score materials over the required target body. Using predicted occupancy
    # as the mask would reward an empty organism with perfect material accuracy.
    material_score = material_accuracy(final[0, layout.material_slice], target_material, target) if bool(torch.isfinite(final).all()) else 0.0
    safety = _state_safety((mature, final, recovered)
                           + tuple(value for value in (source_state, comparison_state) if value is not None), layout, criteria)
    metrics = {
        **safety,
        **_safe_metric_values(occupancy, target),
        "target_iou": target_iou,
        "material_accuracy": material_score,
        "volume_agreement": descriptor_agreements["occupied_volume"],
        "height_agreement": descriptor_agreements["height"],
        "canopy_spread_agreement": descriptor_agreements["canopy_spread"],
        "descriptor_agreement": descriptor_agreement,
        "late_drift": late_drift,
        "post_damage_iou": damaged_iou,
        "recovered_iou": recovered_iou,
        "recovered_target_fraction": recovered_fraction,
        "regeneration_score": regeneration_score,
    }

    failures: list[str] = []
    if comparison_state is not None:
        comparison_target, comparison_material = make_tree_target(case.comparison_genome, world_size, case.environment)
        low_mask, high_mask = comparison_target > .5, target_np > .5
        changed = torch.as_tensor(
            (low_mask != high_mask) | (low_mask & high_mask & (comparison_material != target_material_np)),
            device=run_device,
        )
        metrics["gene_response_edited_voxels"] = float(changed.sum())
        # Identical rasterized targets have no measurable response to score;
        # leave accuracy absent rather than claiming perfect gene control.
        if metrics["gene_response_edited_voxels"]:
            low_target = torch.as_tensor(comparison_target, device=run_device)
            low_material = torch.as_tensor(comparison_material, device=run_device)
            low_correct = ((comparison_state[0, layout.occupancy] > .5) == (low_target > .5)) & (
                (low_target <= .5) | (comparison_state[0, layout.material_slice].argmax(0) == low_material))
            high_correct = ((occupancy > .5) == (target > .5)) & (
                (target <= .5) | (predicted_materials == target_material))
            response = float((low_correct & high_correct)[changed].float().mean()) if safety["finite_state"] else 0.
            metrics["gene_response_accuracy"] = response
            if response < criteria.min_gene_response_accuracy:
                failures.append("gene_response_below_minimum")
        metrics["gene_comparison_target_iou"] = threshold_iou(
            comparison_state[0, layout.occupancy], torch.as_tensor(comparison_target, device=run_device))
        if metrics["gene_comparison_target_iou"] < criteria.min_target_iou:
            failures.append("gene_comparison_target_iou_below_minimum")
    if source_iou is not None:
        metrics["source_target_iou"] = source_iou
        metrics["transition_initial_target_iou"] = threshold_iou(source_state[0, layout.occupancy], target)
        # Whole-tree IoU can hide an ignored slider edit when most voxels stay
        # unchanged. Score only the occupancy/material changes it calls for.
        source_mask, target_mask = source_target > 0.5, target_np > 0.5
        edited = torch.as_tensor(
            (source_mask != target_mask) | (source_mask & target_mask & (source_material != target_material_np)),
            device=run_device,
        )
        edited_count = int(edited.sum())
        metrics["transition_edited_voxels"] = float(edited_count)
        if edited_count:
            correct = ((occupancy > 0.5) == (target > 0.5)) & ((target <= 0.5) | (predicted_materials == target_material))
            edit_accuracy = float(correct[edited].float().mean())
            metrics["transition_edit_accuracy"] = edit_accuracy
            if edit_accuracy < criteria.min_transition_edit_accuracy:
                failures.append("transition_edits_below_minimum")
        if source_iou < criteria.min_target_iou:
            failures.append("source_target_iou_below_minimum")
    if steps < criteria.min_steps:
        failures.append("insufficient_validation_steps")
    if recovery_steps < criteria.min_recovery_steps:
        failures.append("insufficient_recovery_steps")
    checks = (
        (not bool(safety["finite_state"]), "non_finite_state"),
        (not bool(safety["bounded_state"]), "state_magnitude_exceeded"),
        (safety["occupancy_range_violation_fraction"] > criteria.max_occupancy_violation_fraction, "occupancy_out_of_range"),
        (descriptors["largest_component_fraction"] < criteria.min_largest_component_fraction, "disconnected"),
        (target_iou < criteria.min_target_iou, "target_iou_below_minimum"),
        (material_score < criteria.min_material_accuracy, "material_accuracy_below_minimum"),
        (descriptor_agreement < criteria.min_descriptor_agreement, "descriptor_agreement_below_minimum"),
        (late_drift > criteria.max_late_drift, "late_drift_above_maximum"),
        (regeneration_score < criteria.min_regeneration_score, "regeneration_below_minimum"),
    )
    failures.extend(reason for failed, reason in checks if failed)
    validated = steps >= criteria.min_steps and recovery_steps >= criteria.min_recovery_steps
    score_parts = (
        safety["finite_state"], safety["bounded_state"], 1 - safety["occupancy_range_violation_fraction"],
        descriptors["largest_component_fraction"], target_iou, material_score, descriptor_agreement,
        max(0.0, 1 - late_drift), regeneration_score,
    )
    if source_iou is not None:
        score_parts += (source_iou,)
        if "transition_edit_accuracy" in metrics:
            score_parts += (metrics["transition_edit_accuracy"],)
    if comparison_state is not None:
        score_parts += (metrics["gene_comparison_target_iou"],)
        if "gene_response_accuracy" in metrics:
            score_parts += (metrics["gene_response_accuracy"],)
    score = min(score_parts) if validated else 0.0
    return ValidationTrial(case, steps, recovery_steps, validated, not failures, float(np.clip(score, 0, 1)), tuple(failures), metrics, descriptors)


def validate_panel(
    model,
    panel: Iterable[ValidationCase],
    *,
    layout: StateLayout,
    world_size: int = 16,
    steps: int = 512,
    recovery_steps: int = 128,
    seed_size: int = 1,
    device: torch.device | str | None = None,
    criteria: ValidationCriteria = ValidationCriteria(),
    aggregation: Literal["worst", "low_percentile"] = "worst",
    low_percentile: float = 0.1,
    on_trial: Callable[[int, int, ValidationTrial], None] | None = None,
) -> ValidationReport:
    """Validate a deterministic panel sequentially within a safe memory envelope."""
    cases = tuple(panel)
    if not cases:
        raise ValueError("validation panel cannot be empty")
    trials = []
    for index, case in enumerate(cases, 1):
        trial = validate_candidate(
            model, case, layout=layout, world_size=world_size, steps=steps,
            recovery_steps=recovery_steps, seed_size=seed_size, device=device, criteria=criteria,
        )
        trials.append(trial)
        if on_trial is not None:
            on_trial(index, len(cases), trial)
    return ValidationReport(tuple(trials), criteria, aggregation, low_percentile)
