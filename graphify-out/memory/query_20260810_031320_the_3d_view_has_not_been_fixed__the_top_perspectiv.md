---
type: "query"
date: "2026-08-10T03:13:20.933121+00:00"
question: "the 3d view has not been fixed, the top perspective in 3d is still the side projection in 2d. The top of the object in 3d is the side."
contributor: "graphify"
outcome: "corrected"
correction: "The axis code was corrected, but the running dashboard had to be restarted before the browser could receive it; verify the live HTTP response, not only the source file."
source_nodes: ["DashboardServer", "ui.py", "projection()", "LabSession", "._view()"]
---

# Q: the 3d view has not been fixed, the top perspective in 3d is still the side projection in 2d. The top of the object in 3d is the side.

## Answer

Expanded from original query via graph vocab: [top, projection, voxel, view, renderer, lab, dashboard]. The source contained the corrected z-up mapping, but the live DashboardServer on port 8765 was an older process still serving the original y-up mapping. Its job list had no running jobs, so PID 27744 was stopped and the current UI was started as PID 37976. The live HTML now maps tensor [z,y,x] to world [x,-z,y], making its top footprint [height,width], exactly the same axes as LabSession top projection volume.max(0). Focused tests, JavaScript syntax, and the live HTTP response pass. The already-open page must be hard-refreshed because its old JavaScript remains in memory.

## Outcome

- Signal: corrected
- Correction: The axis code was corrected, but the running dashboard had to be restarted before the browser could receive it; verify the live HTTP response, not only the source file.

## Source Nodes

- DashboardServer
- ui.py
- projection()
- LabSession
- ._view()