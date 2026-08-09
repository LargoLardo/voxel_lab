---
type: "query"
date: "2026-08-07T08:41:59.866045+00:00"
question: "its erroring and saying: invalid request token; it does not let me place a seed nor play anything"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "DashboardServer", "InvalidRequestToken", "test_design_lab_http_api"]
---

# Q: its erroring and saying: invalid request token; it does not let me place a seed nor play anything

## Answer

Expanded from the original report via graph vocab: [api, dashboard, design, error, handler, load, payload, post, request, seed, server, session]. Root cause: an open dashboard tab retained the old per-process token after the local server restarted. Fixed DashboardHandler to return a replacement only for InvalidRequestToken and the shared browser api helper to update the token and retry the rejected POST exactly once. Restarted the live server and verified stale-token recovery, run loading, seed placement, and stepping.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- DashboardServer
- InvalidRequestToken
- test_design_lab_http_api