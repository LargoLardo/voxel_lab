---
type: "implementation"
date: "2026-08-10T03:43:51.594432+00:00"
question: "what does model became non finite mean, also how long would you suggest i run regeneration to obtain a stable model, also add an option to view the targets for each run, including a 3d viewer for the 3d runs"
contributor: "graphify"
outcome: "useful"
source_nodes: ["LabSession", "train()", "Damage and regeneration evaluation.", ".voxel_data()", "stability_loss()"]
---

# Q: what does model became non finite mean, also how long would you suggest i run regeneration to obtain a stable model, also add an option to view the targets for each run, including a 3d viewer for the 3d runs

## Answer

Expanded tokens: [finite, training, loss, regeneration, checkpoint, target, lab, voxel, view, stability, pool, damage]. The lab rejects a candidate state when any channel contains NaN or infinity; the completed regen run remained finite during 4000 training iterations but has low final IoU and volatile late loss, so stability should be gated by long-horizon finite rollout and regeneration metrics rather than iteration count alone. Added an organism/training-target selector backed by procedural targets matching each config/genome, reusing the 2D frame and 3D voxel/slice/projection renderers.

## Outcome

- Signal: useful

## Source Nodes

- LabSession
- train()
- Damage and regeneration evaluation.
- .voxel_data()
- stability_loss()