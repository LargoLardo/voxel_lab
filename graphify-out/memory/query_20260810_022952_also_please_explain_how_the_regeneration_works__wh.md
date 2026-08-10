---
type: "query"
date: "2026-08-10T02:29:52.687690+00:00"
question: "also please explain how the regeneration works, what are each of the three options. Also explain the genome lab, ecology and their options, and how they differ from the plain 2d and 3d voxel options. also make it so the smoke tests are the last options and the full runs are default"
contributor: "graphify"
outcome: "useful"
source_nodes: ["evaluate_regeneration.py", "train()", "run_ecology.py", "_launch()", "Full MorphoVoxel Experiment Pipeline"]
---

# Q: also please explain how the regeneration works, what are each of the three options. Also explain the genome lab, ecology and their options, and how they differ from the plain 2d and 3d voxel options. also make it so the smoke tests are the last options and the full runs are default

## Answer

Expanded from original query via graph vocabulary: [regeneration, genome, genomes, ecology, ecological, checkpoint, checkpoints, launch, smoke, full, training, voxel]. The FileNotFoundError occurred because phase4 regeneration evaluation requires runs/phase3_conditional/checkpoints/latest.pt and runs/phase4_regeneration_training/checkpoints/latest.pt. The UI now defaults to Full experiment, orders every full preset before the five smoke presets, preserves canonical run names for full training so dependent checkpoint paths resolve, blocks checkpoint-dependent launches with an actionable message, and adds preset explanations. Regeneration options are damage training, full evaluation, and an untrained smoke pipeline check. Genome lab trains one shared 3D NCA conditioned by four one-hot morphologies. Ecology adds multiple organisms, resources, energy costs, collisions, and competition to the trained conditional NCA. Plain 2D and 3D each train an unconditioned rule for one target. All 29 tests and the embedded JavaScript syntax check pass.

## Outcome

- Signal: useful

## Source Nodes

- evaluate_regeneration.py
- train()
- run_ecology.py
- _launch()
- Full MorphoVoxel Experiment Pipeline