---
type: "implementation"
date: "2026-08-09T21:37:52.722829+00:00"
question: "make an option to display the 3d voxel runs in the design lab as an actual 3d depiction with 3d cubes/voxel representations instead of 2d perspectives."
contributor: "graphify"
outcome: "useful"
source_nodes: ["LabSession", "DashboardHandler", "voxel_data", "StateLayout"]
---

# Q: make an option to display the 3d voxel runs in the design lab as an actual 3d depiction with 3d cubes/voxel representations instead of 2d perspectives.

## Answer

Added LabSession.voxel_data and /api/lab/voxels plus a dependency-free canvas cube renderer with depth sorting, material colors, drag rotation, wheel zoom, and live playback. Existing slice and projection modes remain available for editing. Browser validation and all 27 tests passed.

## Outcome

- Signal: useful

## Source Nodes

- LabSession
- DashboardHandler
- voxel_data
- StateLayout