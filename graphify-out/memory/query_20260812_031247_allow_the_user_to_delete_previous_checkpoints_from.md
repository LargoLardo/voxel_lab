---
type: "query"
date: "2026-08-12T03:12:47.018399+00:00"
question: "allow the user to delete previous checkpoints from the ui, remove Initialize from specialist checkpoint from specialist tab"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "ui.py", "test_ui.py"]
---

# Q: allow the user to delete previous checkpoints from the ui, remove Initialize from specialist checkpoint from specialist tab

## Answer

Expanded from original query via vocab: [dashboard, handler, checkpoint, checkpoints, config, launch, run, runs, view, files]. DashboardHandler now exposes a token-protected, path-scoped checkpoint deletion route. View Checkpoints has a confirmed delete action that refreshes selectors and closes the loaded lab when necessary. The specialist stage explicitly suppresses the dependency checkpoint field while tree-family and later stages retain their prerequisites. UI and route tests cover deletion, traversal rejection, active-run protection, and specialist field hiding.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- ui.py
- test_ui.py