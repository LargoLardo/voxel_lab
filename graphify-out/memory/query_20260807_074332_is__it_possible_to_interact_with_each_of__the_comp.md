---
type: "query"
date: "2026-08-07T07:43:32.718576+00:00"
question: "is  it possible to interact with each of  the completed runs in a design lab, such as placing a seed down with double click and then watching how the run's result would grow it accordingly and erasing parts using the mouse to see growth/regeneration. something similar to [https://distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/)"
contributor: "graphify"
outcome: "useful"
source_nodes: ["load_checkpoint()", "seed_state()", "rollout()", "damage_2d()", "damage_3d()", "ui.py"]
---

# Q: is  it possible to interact with each of  the completed runs in a design lab, such as placing a seed down with double click and then watching how the run's result would grow it accordingly and erasing parts using the mouse to see growth/regeneration. something similar to [https://distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/)

## Answer

Expanded from original query via vocab: [checkpoint, config, damage, live, model, preview, regeneration, rollout, seed, state, training, visualization]. Yes: runs with a saved model checkpoint can power an interactive lab. Reuse load_checkpoint(), seed_state(), rollout()/model forward calls, and damage_2d()/damage_3d(); add a small stateful session API and canvas controls to ui.py. Artifact-only runs without model weights cannot continue evolving.

## Outcome

- Signal: useful

## Source Nodes

- load_checkpoint()
- seed_state()
- rollout()
- damage_2d()
- damage_3d()
- ui.py