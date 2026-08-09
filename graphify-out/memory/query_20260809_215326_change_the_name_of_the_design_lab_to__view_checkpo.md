---
type: "implementation"
date: "2026-08-09T21:53:26.489535+00:00"
question: "change the name of the design lab to \"View Checkpoints\" also, make the 3d voxel viewer the first option instead of z slice"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "setLab", "showRun"]
---

# Q: change the name of the design lab to "View Checkpoints" also, make the 3d voxel viewer the first option instead of z slice

## Answer

Expanded from graph vocab: [design, lab, view, views, voxel, three, checkpoint, checkpoints, dashboard]. Renamed the dashboard navigation, section heading, open button, result action, tests, and README to View Checkpoints. Moved 3D voxels ahead of Z slice in the selector and changed setLab() so newly opened 3D checkpoints default to voxels. Focused UI tests and JavaScript syntax check passed.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- setLab
- showRun