---
type: "query"
date: "2026-08-11T03:51:33.263426+00:00"
question: "Implement the Goal Mode browser UI for the Tree Genome Lab and Variant Archive while preserving View Checkpoints and request-token concurrency protections."
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "LabSession", "ui.py", "test_ui.py"]
---

# Q: Implement the Goal Mode browser UI for the Tree Genome Lab and Variant Archive while preserving View Checkpoints and request-token concurrency protections.

## Answer

Expanded from original query via graph vocab: [dashboard, lab, genome, environment, checkpoint, target, voxel, token]. DashboardHandler and LabSession were the shared integration points. The UI now selects checkpoints, edits continuous genomes with pending versus active semantics, controls environment overlays, validates candidates without holding the lab lock, and saves or reloads strictly bound accepted variants through VariantArchive. Existing 3D voxel, target, damage, playback, and stale-token retry flows remain in place.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- LabSession
- ui.py
- test_ui.py