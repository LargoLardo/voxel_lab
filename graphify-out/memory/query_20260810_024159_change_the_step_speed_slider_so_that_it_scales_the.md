---
type: "query"
date: "2026-08-10T02:41:59.720597+00:00"
question: "change the step speed slider so that it scales the number of steps a second it takes in the simulation/viewer. with the slowest being 1/60x the default, 1x being the default, and 5x being the max speed the device allows for steps/sec, with adequate scaling settings in between"
contributor: "graphify"
outcome: "useful"
source_nodes: [".advance()", "LabSession", "ui.py", "steps_per_second()"]
---

# Q: change the step speed slider so that it scales the number of steps a second it takes in the simulation/viewer. with the slowest being 1/60x the default, 1x being the default, and 5x being the max speed the device allows for steps/sec, with adequate scaling settings in between

## Answer

Expanded from original query via graph vocabulary: [advance, device, display, frame, frames, lab, play, second, steps]. Replaced the View Checkpoints Steps per frame control with a device-relative Playback speed slider. Stops are 1/60x, 1/30x, 1/15x, 1/10x, 1/5x, 1/2x, 1x, 2x, 3x, 4x, and 5x. LabSession.advance reports raw device compute throughput; the browser scheduler defines 1x as one-fifth of that measured rate, 1/60x as exactly one-sixtieth of the default, and 5x as unthrottled with five-step batches. The output shows the estimated target steps/s, speed changes cancel stale timers, and the manual Step button advances exactly one update. All 29 tests, focused UI tests, JavaScript syntax validation, and diff checks pass.

## Outcome

- Signal: useful

## Source Nodes

- .advance()
- LabSession
- ui.py
- steps_per_second()