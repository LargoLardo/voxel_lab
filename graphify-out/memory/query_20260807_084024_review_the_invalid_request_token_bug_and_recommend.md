---
type: "query"
date: "2026-08-07T08:40:24.616428+00:00"
question: "Review the invalid request token bug and recommend the smallest secure restart-safe POST retry fix."
contributor: "graphify"
outcome: "useful"
source_nodes: ["DashboardHandler", "DashboardServer", ".do_POST()", "._payload()"]
---

# Q: Review the invalid request token bug and recommend the smallest secure restart-safe POST retry fix.

## Answer

Expanded from original query via graph vocab: [dashboard, handler, post, request, server, session, state, valid]. The source already implements the correct one-retry token refresh: DashboardHandler returns the new per-process token only for InvalidRequestToken, and the central api wrapper replaces the stale token and retries once. The running PID serves older code, proven because its 403 omitted token and its HTML lacked the retry wrapper. Restart that verified server process. Test stale POST -> 403 with current token -> identical POST retried once reaches route; ensure ordinary 403, missing token, malformed response, and second 403 do not loop.

## Outcome

- Signal: useful

## Source Nodes

- DashboardHandler
- DashboardServer
- .do_POST()
- ._payload()