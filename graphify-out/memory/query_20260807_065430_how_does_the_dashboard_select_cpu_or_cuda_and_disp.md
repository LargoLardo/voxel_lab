---
type: "query"
date: "2026-08-07T06:54:30.469654+00:00"
question: "How does the dashboard select CPU or CUDA and display live cellular automaton frames?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["resolve_device()", "rollout()", "write_live_preview()", "_launch()", "_job_view()", "train()"]
---

# Q: How does the dashboard select CPU or CUDA and display live cellular automaton frames?

## Answer

The UI validates device and frame cadence, resolve_device chooses CUDA for auto when available, rollout observers publish atomic live previews, and the job view serves cache-busted frames while polling.

## Outcome

- Signal: useful

## Source Nodes

- resolve_device()
- rollout()
- write_live_preview()
- _launch()
- _job_view()
- train()