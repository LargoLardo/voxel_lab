---
type: "query"
date: "2026-08-11T03:30:48.217785+00:00"
question: "Implement the Goal Mode persistence validator and Variant Archive as focused modules and tests"
contributor: "graphify"
outcome: "useful"
source_nodes: ["rollout()", "morphology_metrics()", "damage_3d()", "rendering_3d.py", "checkpointing.py", "StateLayout"]
---

# Q: Implement the Goal Mode persistence validator and Variant Archive as focused modules and tests

## Answer

Expanded from original query via graph vocab: checkpoint, damage, environment, finite, genome, metrics, morphology, persistence, rendering, rollout, state, target. Reused ValidationCase with TreeGenome and EnvironmentSpec; validate_candidate uses make_tree_target, rollout, damage_3d, morphology_metrics, and deterministic fire RNG; ValidationReport exposes explicit failures plus worst and low-percentile scores. VariantArchive gates save on an accepted validated report and persists checkpoint SHA256, exact genome/environment/model identity, voxel NPZ data, and Pillow previews in an atomic staged directory.

## Outcome

- Signal: useful

## Source Nodes

- rollout()
- morphology_metrics()
- damage_3d()
- rendering_3d.py
- checkpointing.py
- StateLayout