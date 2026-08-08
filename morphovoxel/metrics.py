"""Growth and regeneration metrics."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


def _binary(value, threshold: float = 0.5) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value) > threshold


def threshold_iou(prediction, target, threshold: float = 0.5) -> float:
    pred, true = _binary(prediction, threshold), _binary(target, threshold)
    union = np.logical_or(pred, true).sum()
    return float(np.logical_and(pred, true).sum() / union) if union else 1.0


def soft_iou(prediction, target, epsilon: float = 1e-8) -> float:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction, dtype=float)
    true = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target, dtype=float)
    intersection = np.minimum(pred, true).sum()
    union = np.maximum(pred, true).sum()
    return float((intersection + epsilon) / (union + epsilon))


def dice_score(prediction, target, threshold: float = 0.5) -> float:
    pred, true = _binary(prediction, threshold), _binary(target, threshold)
    denominator = pred.sum() + true.sum()
    return float(2 * np.logical_and(pred, true).sum() / denominator) if denominator else 1.0


def connected_components(value, threshold: float = 0.5) -> tuple[int, float]:
    mask = _binary(value, threshold)
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for origin in map(tuple, np.argwhere(mask & ~seen)):
        if seen[origin]:
            continue
        queue, size = deque([origin]), 0
        seen[origin] = True
        while queue:
            point = queue.popleft()
            size += 1
            for axis in range(mask.ndim):
                for change in (-1, 1):
                    neighbor = list(point)
                    neighbor[axis] += change
                    neighbor = tuple(neighbor)
                    if all(0 <= neighbor[i] < mask.shape[i] for i in range(mask.ndim)) and mask[neighbor] and not seen[neighbor]:
                        seen[neighbor] = True
                        queue.append(neighbor)
        sizes.append(size)
    occupied = int(mask.sum())
    return len(sizes), (max(sizes) / occupied if occupied else 0.0)


def morphology_metrics(prediction, target, threshold: float = 0.5) -> dict[str, float]:
    pred, true = _binary(prediction, threshold), _binary(target, threshold)
    intersection = np.logical_and(pred, true).sum()
    components, largest = connected_components(pred)
    pred_points, true_points = np.argwhere(pred), np.argwhere(true)
    centroid = float(np.linalg.norm(pred_points.mean(0) - true_points.mean(0))) if len(pred_points) and len(true_points) else float("nan")
    bounds = float(np.linalg.norm(np.r_[pred_points.min(0) - true_points.min(0), pred_points.max(0) - true_points.max(0)])) if len(pred_points) and len(true_points) else float("nan")
    outside = np.logical_and(pred, ~true).sum()
    padded = np.pad(pred.astype(np.int8), 1)
    surface = sum(np.abs(np.diff(padded, axis=axis)).sum() for axis in range(pred.ndim))
    center = (np.asarray(pred.shape) - 1) / 2
    maximum_radius = float(np.linalg.norm(pred_points - center, axis=1).max()) if len(pred_points) else 0.0
    compactness = float(pred.sum() / max(surface, 1))
    return {
        "soft_iou": soft_iou(prediction, target), "iou": threshold_iou(pred, true), "dice": dice_score(pred, true),
        "target_recall": float(intersection / true.sum()) if true.sum() else 1.0,
        "empty_space_precision": float((~pred & ~true).sum() / (~pred).sum()) if (~pred).sum() else 1.0,
        "incorrect_growth_fraction": float(outside / pred.sum()) if pred.sum() else 0.0,
        "connected_components": float(components), "largest_component_fraction": float(largest),
        "centroid_error": centroid, "bounding_box_error": bounds, "occupied_volume": float(pred.sum()),
        "surface_area": float(surface), "compactness": compactness, "maximum_radius": maximum_radius,
        "maximum_vertical_height": float(np.ptp(pred_points[:, 0]) + 1) if len(pred_points) else 0.0,
    }


def material_accuracy(prediction_logits, target, occupancy) -> float:
    logits = prediction_logits.detach().cpu().numpy() if isinstance(prediction_logits, torch.Tensor) else np.asarray(prediction_logits)
    true = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
    mask = _binary(occupancy)
    if not mask.any():
        return 1.0
    if logits.shape[1:] == true.shape:
        prediction = logits.argmax(0)
    elif logits.shape[0] == true.shape[0] and logits.shape[2:] == true.shape[1:]:
        prediction = logits.argmax(1)
    else:
        raise ValueError("material logits do not align with target shape")
    return float((prediction[mask] == true[mask]).mean())


def paired_bootstrap_interval(left, right, seed: int = 0, samples: int = 2000, confidence: float = 0.95) -> tuple[float, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired samples must be nonempty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (samples, len(differences)))].mean(1)
    tail = (1 - confidence) / 2
    return float(differences.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_permutation_test(left, right, seed: int = 0, samples: int = 10000) -> float:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired samples must be nonempty one-dimensional arrays")
    rng, observed = np.random.default_rng(seed), abs(differences.mean())
    permuted = (differences * rng.choice((-1, 1), (samples, len(differences)))).mean(1)
    return float((np.count_nonzero(np.abs(permuted) >= observed) + 1) / (samples + 1))


def recovery_metrics(pre_damage, damaged, recovered, target, removed_mask, threshold: float = 0.5) -> dict[str, float | bool]:
    removed = _binary(removed_mask)
    restored = _binary(recovered) & _binary(target) & removed
    return {
        "pre_damage_iou": threshold_iou(pre_damage, target),
        "post_damage_iou": threshold_iou(damaged, target),
        "final_iou": threshold_iou(recovered, target),
        "recovered_target_fraction": float(restored.sum() / removed.sum()) if removed.sum() else 1.0,
        "recovery_success": threshold_iou(recovered, target) >= 0.85,
        **{f"final_{k}": v for k, v in morphology_metrics(recovered, target).items() if k != "iou"},
    }
