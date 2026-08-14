---
type: "query"
date: "2026-08-14T02:04:24.279476+00:00"
question: "Add per-run stats and run/job deletion controls to the dashboard"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "_job_view()", "build_state()"]
---

# Q: Add per-run stats and run/job deletion controls to the dashboard

## Answer

Expanded from graph vocabulary: [dashboard, run, job, state, ui, checkpoint]. DashboardHandler exposes /api/run/delete and /api/job/delete; the run-list UI renders Stats and deletion actions, while active dashboard jobs are protected from deletion.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- _job_view()
- build_state()