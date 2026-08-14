---
type: "query"
date: "2026-08-14T02:11:44.958880+00:00"
question: "Show each run's preset and saved settings from a compact run-card disclosure"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "build_state()", "config.py"]
---

# Q: Show each run's preset and saved settings from a compact run-card disclosure

## Answer

Expanded from graph vocabulary: [dashboard, run, state, ui, config]. build_state now exposes the saved preset and curated configuration values; run cards replace the Stats button with a bottom arrow disclosure while card clicks retain the detailed results view.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- build_state()
- config.py