---
type: "query"
date: "2026-08-07T08:38:49.636450+00:00"
question: "Inspect existing tests and add the smallest regression test for stale dashboard tokens after server restart."
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "DashboardServer", "test_design_lab_http_api()"]
---

# Q: Inspect existing tests and add the smallest regression test for stale dashboard tokens after server restart.

## Answer

Expanded from original query via graph vocab: [api, dashboard, handler, lab, load, payload, post, request, server]. DashboardHandler rejects a token issued by a prior DashboardServer with HTTP 403 and returns the current token; the dashboard can retry the same POST with that replacement. Extended test_design_lab_http_api to simulate this restart handshake and verified the retried lab load succeeds.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- DashboardServer
- test_design_lab_http_api()