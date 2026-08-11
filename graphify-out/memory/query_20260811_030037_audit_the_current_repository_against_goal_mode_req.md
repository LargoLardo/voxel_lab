---
type: "query"
date: "2026-08-11T03:00:37.876977+00:00"
question: "Audit the current repository against Goal Mode requirements for Tree Genome Lab, Variant Archive, View Checkpoints compatibility, live HTTP dashboard, and checkpoint interaction/3D viewer."
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "LabSession", ".voxel_data()", "test_design_lab_http_api()", "checkpointing.py"]
---

# Q: Audit the current repository against Goal Mode requirements for Tree Genome Lab, Variant Archive, View Checkpoints compatibility, live HTTP dashboard, and checkpoint interaction/3D viewer.

## Answer

Expanded from original query via graph vocab: [genome, lab, checkpoint, dashboard, http, api, interactive, voxel, renderer, target, seed, erase]. DashboardHandler exposes legacy state, run, lab load/action, frame, voxel, and stop routes. LabSession reconstructs config-driven specialist or four-way one-hot models, prefers best.pt, supports seed, erase, advance, target frames, and z/y/x voxel data. No archive or variant vocabulary, routes, or UI exist, and the continuous TreeGenome schema is not yet consumed by ui.py or lab.py. Focused UI and lab tests pass, but the persistent port 8765 server was not running.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- LabSession
- .voxel_data()
- test_design_lab_http_api()
- checkpointing.py