---
type: "query"
date: "2026-08-07T07:14:14.478646+00:00"
question: "PermissionError replacing live.json during training preview"
contributor: "graphify"
outcome: "useful"
source_nodes: ["write_live_preview()", "_job_view()", "train()"]
---

# Q: PermissionError replacing live.json during training preview

## Answer

Expanded from original report via graph vocab: atomically, dashboard, json, live, path, preview, progress, replace, training, write. The dashboard reader can transiently lock live preview targets on Windows; write_live_preview now retries atomic replacement and drops only the locked preview after bounded retries instead of aborting training.

## Outcome

- Signal: useful

## Source Nodes

- write_live_preview()
- _job_view()
- train()