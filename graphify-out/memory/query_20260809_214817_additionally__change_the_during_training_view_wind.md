---
type: "implementation"
date: "2026-08-09T21:48:17.349133+00:00"
question: "additionally, change the during-training view window so that it only displays the end result of every 10 iterations, so that it is no longer save/view every N steps but instead automatically only views the end result of the steps for every 10 iterations."
contributor: "graphify"
outcome: "useful"
source_nodes: ["train", "write_live_preview", "rollout", "DashboardHandler"]
---

# Q: additionally, change the during-training view window so that it only displays the end result of every 10 iterations, so that it is no longer save/view every N steps but instead automatically only views the end result of the steps for every 10 iterations.

## Answer

Expanded from graph vocab: [training, trainer, train, rollout, frame, preview, live, step, steps, view, write, config]. Updated train() so live.png is written once after the completed rollout at absolute iterations 10, 20, 30, and so on; removed per-automaton-step callbacks and the post-training live replay. Removed the frame interval control and payload override from DashboardHandler/UI while keeping final GIF sampling internal. All 27 tests passed.

## Outcome

- Signal: useful

## Source Nodes

- train
- write_live_preview
- rollout
- DashboardHandler