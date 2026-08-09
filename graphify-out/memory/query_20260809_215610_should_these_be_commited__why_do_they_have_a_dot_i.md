---
type: "codebase"
date: "2026-08-09T21:56:10.643961+00:00"
question: "should these be commited? why do they have a dot in front of them? What are they used for"
contributor: "graphify"
outcome: "useful"
source_nodes: ["test_ui.py", "test_lab.py"]
---

# Q: should these be commited? why do they have a dot in front of them? What are they used for

## Answer

Expanded tokens: test, tests, run, runs. The listed .tmp-pytest, .tmp-ui-race, and .tmp-voxel directories are generated pytest basetemp scratch artifacts; .verification-runs contains generated manual verification output. They are tracked in commits d7896ae and 4614a84 but are not source and generally should be ignored and removed from the index. Leading dots are a Unix hidden-name convention and have no special ignore meaning to Git.

## Outcome

- Signal: useful

## Source Nodes

- test_ui.py
- test_lab.py